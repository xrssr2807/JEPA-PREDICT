"""
ECG+PPG 双通道融合下游微调
concat(ECG_embed, PPG_embed) → MLP → 2类
部署需要 ECG+PPG
"""
import os, time, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score, classification_report

from config import Config
from dataset.data import DownstreamDataset
from models.encoder import SignalEncoder
from models.classifier import DualChannelClassifier
from models.losses import FocalLoss, compute_pos_weight

config = Config()
device = torch.device("cuda")
B = config.train.downstream_batch_size

# ── Data ──
print("Loading ECG+PPG paired data...")
split = config.data.chd_ppg_dir + "/train_test_split.json"
train_ppg = DownstreamDataset(config.data.chd_ppg_dir + "/ppg_chd", split, "train", normalize="zscore")
test_ppg  = DownstreamDataset(config.data.chd_ppg_dir + "/ppg_chd", split, "test",  normalize="zscore")
train_ecg = DownstreamDataset(config.data.chd_ecg_dir + "/ecg_chd", split, "train", normalize="zscore")
test_ecg  = DownstreamDataset(config.data.chd_ecg_dir + "/ecg_chd", split, "test",  normalize="zscore")

print(f"PPG: {len(train_ppg)} train / {len(test_ppg)} test")
print(f"ECG: {len(train_ecg)} train / {len(test_ecg)} test")

# PPG和ECG按index一一对应 → zip加载
def paired_loader(ppg_ds, ecg_ds, batch_size, shuffle=False):
    # 确保两个数据集长度一致
    n = min(len(ppg_ds), len(ecg_ds))
    return DataLoader(
        list(range(n)), batch_size=batch_size, shuffle=shuffle,
        num_workers=4, drop_last=shuffle, pin_memory=True,
        collate_fn=lambda idxs: (
            torch.stack([ppg_ds[i][0] for i in idxs]),  # PPG
            torch.stack([ecg_ds[i][0] for i in idxs]),  # ECG
            torch.tensor([ppg_ds[i][1] for i in idxs])   # label
        )
    )

train_loader = paired_loader(train_ppg, train_ecg, B, shuffle=True)
test_loader  = paired_loader(test_ppg, test_ecg, B, shuffle=False)

# ── Encoders ──
enc_kwargs = dict(
    in_channels=1,
    cnn_channels=tuple(config.model.cnn_channels),
    cnn_kernel_sizes=tuple(config.model.cnn_kernel_sizes),
    cnn_strides=tuple(config.model.cnn_strides),
    transformer_layers=config.model.transformer_layers,
    transformer_dim=config.model.transformer_dim,
    transformer_heads=config.model.transformer_heads,
    transformer_ff_dim=config.model.transformer_ff_dim,
    transformer_dropout=config.model.transformer_dropout,
    max_seq_len=config.model.max_seq_len, pool_type=config.model.pool_type,
    use_se=config.model.cnn_use_se,
    use_inception=config.model.cnn_use_inception,
)

ckpt = torch.load("outputs/jepa_best.pt", map_location="cpu")
msd = ckpt["model_state_dict"]
def load_enc(enc_type):
    enc = SignalEncoder(**enc_kwargs).to(device)
    prefix = f"{enc_type}_encoder."
    state = {k[len(prefix):]: v for k, v in msd.items() if k.startswith(prefix)}
    enc.load_state_dict(state, strict=True)
    return enc

ecg_enc = load_enc("context")
ppg_enc = load_enc("target")

# ── Gated Fusion (ACMGF-style) ──
class GatedFusion(nn.Module):
    """
    论文级实现: Gated Multimodal Networks
    门控: 1024→128→BN→ReLU→1→Sigmoid, 标量输出(B,1)
    融合: gate*ECG + (1-gate)*PPG, 分类: 512→256→BN→ReLU→Dropout→64→BN→ReLU→Dropout→2
    """
    def __init__(self, ecg_enc, ppg_enc, dim=512, num_classes=2):
        super().__init__()
        self.ecg_encoder = ecg_enc
        self.ppg_encoder = ppg_enc
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, ecg, ppg):
        h_ecg, _ = self.ecg_encoder(ecg)
        h_ppg, _ = self.ppg_encoder(ppg)
        alpha = self.gate(torch.cat([h_ecg, h_ppg], dim=-1))  # (B, 1)
        fused = alpha * h_ecg + (1 - alpha) * h_ppg
        return self.classifier(fused)

    def freeze_encoders(self):
        for p in self.ecg_encoder.parameters(): p.requires_grad = False
        for p in self.ppg_encoder.parameters(): p.requires_grad = False

    def unfreeze_encoders(self):
        for p in self.ecg_encoder.parameters(): p.requires_grad = True
        for p in self.ppg_encoder.parameters(): p.requires_grad = True

model = GatedFusion(ecg_enc, ppg_enc, config.model.transformer_dim, 2).to(device)
print(f"Model: GatedFusion ({sum(p.numel() for p in model.parameters()):,} params)")

# ── Train ──
pos_weight = compute_pos_weight(train_ppg, 2, device)
criterion = FocalLoss(gamma=2.0)

# ── Layer-wise LR (train_downstream.py 风格) ──
base_lr = 5e-5
layer_decay = 0.75
param_groups = []

# 分类头: 全LR
head_params = []
for n, p in model.named_parameters():
    if 'classifier' in n or 'fusion' in n:
        head_params.append(p)
if head_params:
    param_groups.append({"params": head_params, "lr": base_lr, "name": "head"})

# 每个编码器的 transformer layers: 逐层衰减
for enc_prefix in ['ecg_encoder', 'ppg_encoder']:
    enc = getattr(model, enc_prefix)
    nlayers = len(enc.transformer.blocks)
    for li in range(nlayers):
        lr = base_lr * (layer_decay ** (nlayers - 1 - li))
        lp = []
        for n, p in enc.named_parameters():
            if f"transformer.blocks.{li}." in n:
                lp.append(p)
        if lp:
            param_groups.append({"params": lp, "lr": lr, "name": f"{enc_prefix}_L{li}"})

    # CNN stem + pos_enc + proj: 最低LR
    stem_lr = base_lr * (layer_decay ** nlayers)
    stem_p = []
    for n, p in enc.named_parameters():
        if p not in set(pp for g in param_groups for pp in g["params"]):
            stem_p.append(p)
    if stem_p:
        param_groups.append({"params": stem_p, "lr": stem_lr, "name": f"{enc_prefix}_stem"})

for g in param_groups:
    print(f"  {g['name']}: lr={g['lr']:.2e} ({len(g['params'])} params)")

optimizer = AdamW(param_groups, weight_decay=1e-3)
steps = len(train_loader)
scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=3*steps)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=67*steps, eta_min=1e-7)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[3*steps])

os.makedirs("outputs", exist_ok=True)
log_fh = open("outputs/dual_log.txt", "w")
log_fh.write(f"DualChannel ECG+PPG Fusion | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_fh.write(f"{'='*60}\n")

# ── Phase 1: Probe (2 epochs, frozen encoders) ──
print("[Probe] Freezing encoders, training classifier head only...")
model.freeze_encoders()
probe_opt = AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=2e-4, weight_decay=1e-4)
for ep in range(20):
    model.train()
    for ppg, ecg, labels in train_loader:
        ppg, ecg, labels = ppg.to(device), ecg.to(device), labels.to(device)
        loss = criterion(model(ecg, ppg), labels)
        probe_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        probe_opt.step()
    # quick eval
    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for ppg, ecg, labels in test_loader:
            ppg, ecg = ppg.to(device), ecg.to(device)
            preds.extend(model(ecg, ppg).softmax(-1)[:,1].cpu().numpy())
            labs.extend(labels.numpy())
    print(f"  Probe E{ep+1}: AUC={roc_auc_score(labs, preds):.4f}")

print("[FT] Unfreezing encoders, full fine-tune with Layer-wise LR...")
model.unfreeze_encoders()

best_auc, best_state = 0, None
for ep in range(70):
    # Train
    model.train()
    train_loss_total, train_correct, train_total = 0.0, 0, 0
    for ppg, ecg, labels in train_loader:
        ppg, ecg, labels = ppg.to(device), ecg.to(device), labels.to(device)
        logits = model(ecg, ppg)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        train_loss_total += loss.item()
        train_correct += logits.argmax(-1).eq(labels).sum().item()
        train_total += labels.size(0)
    train_loss = train_loss_total / len(train_loader)
    train_acc = train_correct / train_total

    # Eval
    model.eval()
    test_loss_total = 0.0
    all_preds, all_labs, all_probs = [], [], []
    with torch.no_grad():
        for ppg, ecg, labels in test_loader:
            ppg, ecg, labels = ppg.to(device), ecg.to(device), labels.to(device)
            logits = model(ecg, ppg)
            test_loss_total += criterion(logits, labels).item()
            all_probs.append(logits.softmax(-1).cpu().numpy())
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labs.extend(labels.cpu().numpy())
    test_loss = test_loss_total / len(test_loader)
    all_probs = np.concatenate(all_probs)

    from sklearn.metrics import precision_score, recall_score, fbeta_score
    auc   = roc_auc_score(all_labs, all_probs[:,1])
    acc   = np.mean(np.array(all_preds) == np.array(all_labs))
    prec  = precision_score(all_labs, all_preds, average='macro', zero_division=0)
    rec   = recall_score(all_labs, all_preds, average='macro', zero_division=0)
    f1    = fbeta_score(all_labs, all_preds, beta=1, average='macro', zero_division=0)
    f05   = fbeta_score(all_labs, all_preds, beta=0.5, average='macro', zero_division=0)

    if auc > best_auc:
        best_auc = auc; best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    log_line = (f"FT Epoch {ep+1:2d} | "
                f"Train L={loss.item():.4f} Acc={train_acc*100:.2f}% | "
                f"Test L={test_loss:.4f} Acc={acc*100:5.2f}% AUC={auc:.4f} "
                f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f}")
    print(log_line)
    log_fh.write(log_line + "\n"); log_fh.flush()

# ── Final ──
model.load_state_dict(best_state)
preds, labs = [], []
with torch.no_grad():
    for ppg, ecg, labels in test_loader:
        ppg, ecg = ppg.to(device), ecg.to(device)
        preds.extend(model(ecg, ppg).softmax(-1)[:,1].cpu().numpy())
        labs.extend(labels.numpy())

final_auc = roc_auc_score(labs, preds)
report = classification_report(labs, (np.array(preds)>=0.5).astype(int), digits=4)
print(f"\n{'='*60}")
print(f"DualChannel Fusion AUC: {final_auc:.4f}")
print(report)

log_fh.write(f"\n{'='*60}\nFINAL AUC: {final_auc:.4f}\n{report}\n")
log_fh.close()

torch.save({"model": best_state, "auc": final_auc}, "outputs/dual_best.pt")
print("Saved → outputs/dual_best.pt | Log → outputs/dual_log.txt")
