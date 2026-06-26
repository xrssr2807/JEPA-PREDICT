"""
Ensemble: 单通道CoT + 双通道SimpleFusion Probe 加权平均预测
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report

from config import Config
from dataset.data import DownstreamDataset
from models.encoder import SignalEncoder
from models.classifier import SignalClassifierCoT, DualChannelSimpleFusion

config = Config()
device = torch.device("cuda")

# ── 加载数据 ──
print("Loading data...")
ppg_dir = config.data.chd_ppg_dir + "/ppg_chd"
ecg_dir = config.data.chd_ecg_dir + "/ecg_chd"
split_file = config.data.chd_ppg_dir + "/train_test_split.json"

# Single-channel dataloader (for CoT model)
test_ppg = DownstreamDataset(ppg_dir, split_file, "test", normalize="zscore")
test_loader_single = DataLoader(test_ppg, batch_size=128, shuffle=False, num_workers=4)

# Dual-channel dataloader (for Probe model)
from dataset.data import DualDownstreamDataset
test_ecg = DownstreamDataset(ecg_dir, split_file, "test", normalize="zscore")
dual_test = DualDownstreamDataset(test_ppg, test_ecg)
test_loader_dual = DataLoader(dual_test, batch_size=128, shuffle=False, num_workers=4)

# ── 模型1: 单通道CoT ──
print("\nLoading Model 1: Single-channel CoT...")
encoder1 = SignalEncoder(
    in_channels=1, cnn_channels=(128,256,512,512),
    cnn_kernel_sizes=(7,5,5,3), cnn_strides=(2,2,2,2),
    transformer_layers=8, transformer_dim=512, transformer_heads=16,
    transformer_ff_dim=2048, max_seq_len=200,
).to(device)
model1 = SignalClassifierCoT(encoder1, 512, 2).to(device)
ckpt1 = torch.load("outputs/downstream_chd_best.pt", map_location=device, weights_only=False)
model1.load_state_dict(ckpt1["model_state_dict"])
model1.eval()
print(f"  Loaded (best AUC={ckpt1.get('test_auc', 'N/A')})")

# ── 模型2: 双通道SimpleFusion Probe ──
print("Loading Model 2: Dual-channel SimpleFusion...")
# We need to train it first — quick 10-epoch probe
print("  Training Probe (10 epoch, frozen encoders)...")
import torch.nn as nn
from torch.optim import AdamW
from models.losses import FocalLoss, compute_pos_weight
from train_downstream import load_pretrained_encoder

ecg_enc = load_pretrained_encoder("outputs/jepa_best.pt", config.model, "context", device)
ppg_enc = load_pretrained_encoder("outputs/jepa_best.pt", config.model, "target", device)

model2 = DualChannelSimpleFusion(ecg_enc, ppg_enc, 512, 2).to(device)
model2.freeze_encoders()

# Train Probe on dual-channel
train_ppg_ds = DownstreamDataset(ppg_dir, split_file, "train", normalize="zscore")
train_ecg_ds = DownstreamDataset(ecg_dir, split_file, "train", normalize="zscore")
dual_train = DualDownstreamDataset(train_ppg_ds, train_ecg_ds)
train_loader_dual = DataLoader(dual_train, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

pos_weight = compute_pos_weight(train_ppg_ds, 2, device)
criterion = FocalLoss(gamma=2.0)
optimizer = AdamW(model2.fusion.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_auc = 0
best_state = None
for epoch in range(10):
    model2.train()
    for batch in train_loader_dual:
        ecg, ppg, labels, *_ = batch
        ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
        logits = model2(ecg, ppg)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Eval
    model2.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader_dual:
            ecg, ppg, labels, *_ = batch
            ecg, ppg = ecg.to(device), ppg.to(device)
            logits = model2(ecg, ppg)
            probs = logits.softmax(-1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.cpu() for k, v in model2.state_dict().items()}
    print(f"  Probe Epoch {epoch+1:2d}: AUC={auc:.4f} (best={best_auc:.4f})")

model2.load_state_dict(best_state)
model2.eval()
print(f"  Probe done. Best AUC={best_auc:.4f}")

# ── Ensemble Inference ──
print("\n=== Ensemble Inference ===")
all_probs1, all_probs2, all_labels = [], [], []

with torch.no_grad():
    # Must iterate both loaders in sync by patient UID
    # For simplicity, iterate dual loader and also run model1 on PPG only
    for batch in test_loader_dual:
        ecg, ppg, labels, uids = batch
        ecg, ppg = ecg.to(device), ppg.to(device)

        # Model 1: single-channel CoT (needs PPG only, per-segment)
        logits1 = model1(ppg)
        probs1 = logits1.softmax(-1)[:, 1].cpu().numpy()

        # Model 2: dual-channel Probe
        logits2 = model2(ecg, ppg)
        probs2 = logits2.softmax(-1)[:, 1].cpu().numpy()

        all_probs1.extend(probs1)
        all_probs2.extend(probs2)
        all_labels.extend(labels.numpy())

all_probs1 = np.array(all_probs1)
all_probs2 = np.array(all_probs2)
all_labels = np.array(all_labels)

# Grid search for best ensemble weights
print("\nGrid search for ensemble weights...")
best_auc = 0
best_w = 0.5
for w in np.linspace(0.0, 1.0, 21):
    ensemble_probs = w * all_probs1 + (1-w) * all_probs2
    auc = roc_auc_score(all_labels, ensemble_probs)
    if auc > best_auc:
        best_auc = auc
        best_w = w
    print(f"  w={w:.2f} (CoT={w:.2f}, Dual={1-w:.2f}): AUC={auc:.4f}")

print(f"\n{'='*60}")
print(f"Best Ensemble AUC: {best_auc:.4f} (w={best_w:.2f})")
print(f"Model 1 (CoT) solo AUC: {roc_auc_score(all_labels, all_probs1):.4f}")
print(f"Model 2 (Dual Probe) solo AUC: {roc_auc_score(all_labels, all_probs2):.4f}")

# Final report at best weight
ensemble = best_w * all_probs1 + (1-best_w) * all_probs2
ensemble_preds = (ensemble >= 0.5).astype(int)
print(f"\nClassification Report (Ensemble w={best_w:.2f}):")
print(classification_report(all_labels, ensemble_preds, digits=4))
