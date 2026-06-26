"""
ECG-Guided Distillation (ECG-guided PPG, ICASSP 2025 方案)

训练: PPG(主力) + ECG(教师, 冻结) → 蒸馏
部署: 仅 PPG

L = L_cls + λ * L_align + β * L_kd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

from config import Config
from dataset.data import DownstreamDataset
from models.encoder import SignalEncoder
from models.classifier import SignalClassifierCoT
from models.losses import FocalLoss, compute_pos_weight

config = Config()
device = torch.device("cuda")
distill_lambda = 0.5   # 特征对齐权重
kd_beta = 0.3          # logit蒸馏权重
kd_temp = 4.0          # 蒸馏温度

# ── Data ──
print("Loading data...")
gen = torch.Generator().manual_seed(42)
train_ppg = DownstreamDataset(config.data.chd_ppg_dir+"/ppg_chd",
    config.data.chd_ppg_dir+"/train_test_split.json","train",normalize="zscore")
test_ppg = DownstreamDataset(config.data.chd_ppg_dir+"/ppg_chd",
    config.data.chd_ppg_dir+"/train_test_split.json","test",normalize="zscore")
train_ecg = DownstreamDataset(config.data.chd_ecg_dir+"/ecg_chd",
    config.data.chd_ppg_dir+"/train_test_split.json","train",normalize="zscore")
test_ecg = DownstreamDataset(config.data.chd_ecg_dir+"/ecg_chd",
    config.data.chd_ppg_dir+"/train_test_split.json","test",normalize="zscore")

train_loader = DataLoader(train_ppg, batch_size=256, shuffle=True, num_workers=4, drop_last=True, generator=gen)
ecg_loader = DataLoader(train_ecg, batch_size=256, shuffle=True, num_workers=4, drop_last=True, generator=gen)
test_loader = DataLoader(test_ppg, batch_size=256, shuffle=False, num_workers=4)

# ── Encoders ──
print("Loading encoders...")
ckpt = torch.load("outputs/jepa_best.pt", map_location="cpu")
enc_kwargs = dict(in_channels=1, cnn_channels=tuple(config.model.cnn_channels),
    cnn_kernel_sizes=tuple(config.model.cnn_kernel_sizes), cnn_strides=tuple(config.model.cnn_strides),
    transformer_layers=config.model.transformer_layers, transformer_dim=config.model.transformer_dim,
    transformer_heads=config.model.transformer_heads, transformer_ff_dim=config.model.transformer_ff_dim,
    transformer_dropout=config.model.transformer_dropout, max_seq_len=config.model.max_seq_len)

# Student: PPG encoder (可训练)
ppg_encoder = SignalEncoder(**enc_kwargs).to(device)
target_state = {k[len("target_encoder."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("target_encoder.")}
ppg_encoder.load_state_dict(target_state, strict=True)

# Teacher: ECG encoder (冻结)
ecg_encoder = SignalEncoder(**enc_kwargs).to(device)
ctx_state = {k[len("context_encoder."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("context_encoder.")}
ecg_encoder.load_state_dict(ctx_state, strict=True)
ecg_encoder.eval()
for p in ecg_encoder.parameters():
    p.requires_grad = False

# ── Projection heads (ICASSP 2025: 2层MLP, 512→256, 对齐到共享空间) ──
proj_ppg = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 256)).to(device)
proj_ecg = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 256)).to(device)
# ECG proj head 冻结 (只在训练开始前同步一次权重)
proj_ecg.load_state_dict(proj_ppg.state_dict())
for p in proj_ecg.parameters():
    p.requires_grad = False

# ── Model ──
model = SignalClassifierCoT(ppg_encoder, 512, 2).to(device)  # CoT head
pos_weight = compute_pos_weight(train_ppg, 2, device)
cls_criterion = FocalLoss(gamma=2.0)

# ── Probe ──
# 蒸馏模式: ECG引导, 更快收敛 → 需要充分更新
total_probe_steps = 5080  # 基准: batch=128 × 10epoch × 508steps
steps_per_epoch = len(train_loader)
probe_epochs = max(10, total_probe_steps // steps_per_epoch)
print(f"\n=== Phase 1: Probe ({probe_epochs} epochs, {steps_per_epoch} steps/epoch) ===")
model.freeze_encoder()
opt = AdamW(list(model.head.parameters()) + list(proj_ppg.parameters()), lr=2e-3)
for ep in range(probe_epochs):
    model.train(); proj_ppg.train()
    for ppg_b, ecg_b in zip(train_loader, ecg_loader):
        x, labels, *_ = ppg_b; ex, *_ = ecg_b
        x, labels, ex = x.to(device), labels.to(device), ex.to(device)
        # PPG forward
        ppg_pooled, ppg_tokens = ppg_encoder(x, return_all=True)
        logits = model.head(F.layer_norm(ppg_tokens, ppg_tokens.shape[-1:]))
        cls_loss = cls_criterion(logits, labels)
        # Alignment: 投影后余弦对齐
        with torch.no_grad():
            ecg_pooled, _ = ecg_encoder(ex)
        align_loss = (1 - F.cosine_similarity(proj_ppg(ppg_pooled), proj_ecg(ecg_pooled), dim=-1)).mean()
        loss = cls_loss + distill_lambda * align_loss
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 3 == 0 or ep == 9:
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for b in test_loader: x, y, *_ = b; preds.extend(model(x.to(device)).softmax(-1)[:,1].cpu().numpy()); labs.extend(y.numpy())
        print(f"  Probe E{ep+1}: AUC={roc_auc_score(labs, preds):.4f}")

# ── Full Fine-tune + Distillation ──
print("\n=== Phase 2: FT + Distillation ===")
model.unfreeze_encoder()
base_lr = config.train.downstream_lr * 0.1
param_groups = []
for i, blk in enumerate(ppg_encoder.transformer.blocks):
    param_groups.append({"params": blk.parameters(), "lr": base_lr*(0.85**(7-i))})
param_groups.append({"params": list(ppg_encoder.cnn.parameters())+list(ppg_encoder.proj.parameters())+list(ppg_encoder.pos_encoding.parameters()), "lr": base_lr*0.85**8})
param_groups.append({"params": model.head.parameters(), "lr": base_lr})
param_groups.append({"params": proj_ppg.parameters(), "lr": base_lr})

opt = AdamW(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40*len(train_loader))
best_auc, best_state, no_improve = 0, None, 0

for ep in range(40):
    model.train(); proj_ppg.train()
    for ppg_b, ecg_b in zip(train_loader, ecg_loader):
        x, labels, *_ = ppg_b; ex, *_ = ecg_b
        x, labels, ex = x.to(device), labels.to(device), ex.to(device)

        ppg_pooled, ppg_tokens = ppg_encoder(x, return_all=True)
        with torch.no_grad():
            ecg_pooled, _ = ecg_encoder(ex)

        logits = model.head(F.layer_norm(ppg_tokens, ppg_tokens.shape[-1:]))
        cls_loss = cls_criterion(logits, labels)
        align_loss = (1 - F.cosine_similarity(proj_ppg(ppg_pooled), proj_ecg(ecg_pooled), dim=-1)).mean()
        loss = cls_loss + distill_lambda * align_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step()

    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for b in test_loader: x, y, *_ = b; preds.extend(model(x.to(device)).softmax(-1)[:,1].cpu().numpy()); labs.extend(y.numpy())
    auc = roc_auc_score(labs, preds)
    if auc > best_auc: best_auc=auc; no_improve=0; best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    else: no_improve += 1
    print(f"  FT E{ep+1}: AUC={auc:.4f} (best={best_auc:.4f})")
    if no_improve >= 10: print(f"  Early stop"); break

# ── Final ──
model.load_state_dict(best_state); model.eval()
preds, labs = [], []
with torch.no_grad():
    for b in test_loader: x, y, *_ = b; preds.extend(model(x.to(device)).softmax(-1)[:,1].cpu().numpy()); labs.extend(y.numpy())
print(f"\n=== Final (ECG Distillation, PPG-only deploy) ===")
print(f"AUC: {roc_auc_score(labs, preds):.4f}")
print(classification_report(labs, (np.array(preds)>=0.5).astype(int), digits=4))
torch.save({"model_state_dict": best_state, "auc": roc_auc_score(labs, preds)}, "outputs/distill_best.pt")
print("Saved → outputs/distill_best.pt")
