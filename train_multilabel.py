"""
7疾病多标签下游微调 — PPG only (data_updated ch0)
L = BCEWithLogitsLoss, 7个独立sigmoid, per-disease AUC
"""
import os, time, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score, classification_report
import pickle

from config import Config
from models.encoder import SignalEncoder
from models.classifier import SignalClassifier
from models.losses import compute_pos_weight

config = Config()
device = torch.device("cuda")
B = config.train.downstream_batch_size

# ── 5 diseases (移除心律失常, 合并糖尿病+高血糖→代谢综合征) ──
DISEASES = ['高血压', '高血脂', '冠心病', '颈动脉斑块', '代谢综合征']
N_CLASSES = len(DISEASES)

def make_multihot(raw_label):
    """从原始9疾病dict → 5类 multi-hot"""
    diabetic = raw_label.get('糖尿病', 0) or raw_label.get('高血糖', 0)
    return torch.tensor([
        raw_label['高血压'],
        raw_label['高血脂'],
        raw_label['冠心病'],
        raw_label['颈动脉斑块'],
        1 if diabetic else 0,
    ], dtype=torch.float32)

# ── Dataset ──
class MultiLabelDataset(Dataset):
    def __init__(self, data_dir, split_file, split="train"):
        import json
        with open(split_file) as f:
            self.files = json.load(f)[split]
        self.data_dir = data_dir

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.data_dir, self.files[idx]), 'rb') as f:
            s = pickle.load(f)
        # PPG only: ch0
        x = s['data'].astype(np.float32)[0]  # (1000,)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = x.std()
        if std < 1e-6:
            x = np.zeros_like(x)
        else:
            x = (x - x.mean()) / std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(x).float().unsqueeze(0)  # (1, 1000)
        # Multi-hot label
        y = make_multihot(s['label'])
        return x, y

# ── Data ──
data_dir = "/root/ppgchd/ppgchd/data_updated"
split_file = "/root/ppgchd/ppgchd/train_test_split.json"

train_ds = MultiLabelDataset(data_dir, split_file, "train")
test_ds  = MultiLabelDataset(data_dir, split_file, "test")
train_loader = DataLoader(train_ds, B, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
test_loader  = DataLoader(test_ds,  B, shuffle=False, num_workers=4, pin_memory=True)
print(f"Data: {len(train_ds)} train / {len(test_ds)} test")

# ── Model ──
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
base_enc = SignalEncoder(**enc_kwargs).to(device)
ckpt = torch.load("outputs/jepa_best.pt", map_location="cpu")
tgt_state = {k[len("target_encoder."):]: v
             for k, v in ckpt["model_state_dict"].items()
             if k.startswith("target_encoder.")}
base_enc.load_state_dict(tgt_state, strict=True)

class MultiLabelClassifier(nn.Module):
    def __init__(self, encoder, dim=512, num_classes=7):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),  # 7 sigmoid outputs
        )
    def forward(self, x):
        h, _ = self.encoder(x)
        return self.head(h)  # raw logits, BCEWithLogits handles sigmoid
    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True

model = MultiLabelClassifier(base_enc, config.model.transformer_dim, N_CLASSES).to(device)

# ★ AsymmetricLoss + CHD FocalLoss 双损失
from models.losses import AsymmetricLoss
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
chd_criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.0)  # CHD专用
chd_weight = 0.5  # CHD辅助损失权重
CHD_IDX = DISEASES.index('冠心病')
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
print(f"Loss: ASL(multi) + {chd_weight}×ASL(CHD-only)")

# ── Probe + FT ──
os.makedirs("outputs", exist_ok=True)
log_fh = open("outputs/multilabel_log.txt", "w")
log_fh.write(f"Multilabel 7-disease | {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")

# Probe (frozen encoder, 10 epochs)
print("=== Probe ===")
model.freeze_encoder()
probe_opt = AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        probe_opt.zero_grad(); loss.backward(); probe_opt.step()

    model.eval(); preds = [[] for _ in range(N_CLASSES)]; labs = [[] for _ in range(N_CLASSES)]
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            for c in range(N_CLASSES):
                preds[c].extend(torch.sigmoid(logits[:, c]).cpu().numpy())
                labs[c].extend(y[:, c].cpu().numpy())
    aucs = [roc_auc_score(labs[c], np.nan_to_num(preds[c], nan=0.5)) if len(set(labs[c])) > 1 else 0.5 for c in range(N_CLASSES)]
    print(f"  Probe E{ep+1}: avg AUC={np.mean(aucs):.4f}")
    log_fh.write(f"Probe E{ep+1}: avg AUC={np.mean(aucs):.4f} | {' '.join(f'{d[:2]}={a:.3f}' for d,a in zip(DISEASES,aucs))}\n")

# FT (layer-wise LR, 50 epochs, early stop)
print("=== FT ===")
model.unfreeze_encoder()
base_lr = 5e-5; nlayers = len(base_enc.transformer.blocks)
param_groups = [{"params": model.head.parameters(), "lr": base_lr, "name": "head"}]
for li in range(nlayers):
    lp = []
    for n, p in base_enc.named_parameters():
        if f"transformer.blocks.{li}." in n: lp.append(p)
    param_groups.append({"params": lp, "lr": base_lr * (0.75 ** (nlayers-1-li)), "name": f"L{li}"})
stem_p = [p for n, p in base_enc.named_parameters() if p not in set(pp for g in param_groups for pp in g["params"])]
param_groups.append({"params": stem_p, "lr": base_lr * (0.75 ** nlayers), "name": "stem"})

ft_opt = AdamW(param_groups, weight_decay=1e-3)
steps = len(train_loader)
sch = SequentialLR(ft_opt,
    schedulers=[LinearLR(ft_opt, start_factor=0.1, total_iters=3*steps),
                CosineAnnealingLR(ft_opt, T_max=47*steps, eta_min=1e-7)],
    milestones=[3*steps])

best_auc, no_improve, best_state = 0, 0, None
for ep in range(50):
    model.train()
    train_loss_total, train_chd_loss_total, n_batches = 0.0, 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        # ★ CHD辅助损失: 只对CHD列计算
        chd_loss = chd_criterion(logits[:, CHD_IDX:CHD_IDX+1],
                                  y[:, CHD_IDX:CHD_IDX+1])
        total_loss = loss + chd_weight * chd_loss
        ft_opt.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ft_opt.step(); sch.step()
        train_loss_total += loss.item()
        train_chd_loss_total += chd_loss.item()
        n_batches += 1

    train_loss = train_loss_total / n_batches
    train_chd = train_chd_loss_total / n_batches

    model.eval(); preds = [[] for _ in range(N_CLASSES)]; labs = [[] for _ in range(N_CLASSES)]
    test_loss_total, test_n = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_loss_total += criterion(logits, y).item(); test_n += 1
            for c in range(N_CLASSES):
                preds[c].extend(torch.sigmoid(logits[:, c]).cpu().numpy())
                labs[c].extend(y[:, c].cpu().numpy())
    test_loss = test_loss_total / test_n
    aucs = [roc_auc_score(labs[c], np.nan_to_num(preds[c], nan=0.5)) if len(set(labs[c])) > 1 else 0.5 for c in range(N_CLASSES)]
    avg_auc = np.mean(aucs)

    if avg_auc > best_auc:
        best_auc = avg_auc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; no_improve = 0
    else:
        no_improve += 1

    line = (f"FT E{ep+1:2d}: TrainL={train_loss:.4f}(CHD={train_chd:.4f}) "
            f"TestL={test_loss:.4f} AUC={avg_auc:.4f} | "
            + ' '.join(f'{d[:2]}={a:.3f}' for d,a in zip(DISEASES,aucs)))
    print(f"  {line}"); log_fh.write(line + "\n"); log_fh.flush()
    if no_improve >= 5:
        print(f"  [EarlyStop]"); break

# ── Final ──
model.load_state_dict(best_state)
model.eval(); preds = [[] for _ in range(N_CLASSES)]; labs = [[] for _ in range(N_CLASSES)]
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        for c in range(N_CLASSES):
            preds[c].extend(torch.sigmoid(model(x)[:, c]).cpu().numpy())
            labs[c].extend(y[:, c].cpu().numpy())

print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
log_fh.write(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}\n")
for c, d in enumerate(DISEASES):
    auc = roc_auc_score(labs[c], np.nan_to_num(preds[c], nan=0.5)) if len(set(labs[c])) > 1 else 0.5
    acc = np.mean((np.array(preds[c]) >= 0.5).astype(int) == np.array(labs[c]))
    pos_rate = np.mean(labs[c])
    print(f"  {d}: AUC={auc:.4f}  Acc={acc:.4f}  PosRate={pos_rate:.3f}")
    log_fh.write(f"  {d}: AUC={auc:.4f}  Acc={acc:.4f}  PosRate={pos_rate:.3f}\n")

avg = np.mean([roc_auc_score(labs[c], np.nan_to_num(preds[c], nan=0.5)) if len(set(labs[c])) > 1 else 0.5 for c in range(N_CLASSES)])
print(f"  MACRO AUC: {avg:.4f}")
log_fh.write(f"  MACRO AUC: {avg:.4f}\n")
log_fh.close()

torch.save(best_state, "outputs/multilabel_best.pt")
print(f"\nSaved → outputs/multilabel_best.pt | Log → outputs/multilabel_log.txt")
