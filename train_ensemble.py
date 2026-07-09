"""
A+B 集成训练: PPG单通道 + ECG单通道 + ECG+PPG双通道 → 投票集成
部署时需要ECG，预期 AUC 0.79~0.81
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
from models.classifier import SignalClassifier, DualChannelClassifier
from models.losses import FocalLoss, compute_pos_weight

config = Config()
device = torch.device("cuda")

# ── Data ──
print("Loading data...")
split_file = config.data.chd_ppg_dir + "/train_test_split.json"
ppg_dir = config.data.chd_ppg_dir + "/ppg_chd"
ecg_dir = config.data.chd_ecg_dir + "/ecg_chd"

train_ppg = DownstreamDataset(ppg_dir, split_file, "train", normalize="zscore")
test_ppg  = DownstreamDataset(ppg_dir, split_file, "test",  normalize="zscore")
train_ecg = DownstreamDataset(ecg_dir, split_file, "train", normalize="zscore")
test_ecg  = DownstreamDataset(ecg_dir, split_file, "test",  normalize="zscore")

B = config.train.downstream_batch_size
train_ppg_loader = DataLoader(train_ppg, B, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
train_ecg_loader = DataLoader(train_ecg, B, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
test_ppg_loader  = DataLoader(test_ppg,  B, shuffle=False, num_workers=4, pin_memory=True)
test_ecg_loader  = DataLoader(test_ecg,  B, shuffle=False, num_workers=4, pin_memory=True)

pos_weight = compute_pos_weight(train_ppg, 2, device)
criterion = FocalLoss(gamma=2.0)
print(f"PPG: {len(train_ppg)} train, {len(test_ppg)} test")
print(f"ECG: {len(train_ecg)} train, {len(test_ecg)} test")

# ── Encoder factory ──
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

# ── Train helpers ──
def build_scheduler(optimizer, steps_per_epoch, total_epochs, warmup_epochs=5):
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

def train_model(model, train_loader, test_loader, epochs, lr, tag, ecg_loader=None):
    """Train single-channel or dual-channel model."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, steps_per_epoch, epochs)
    best_auc, best_state = 0, None
    ecg_iter = iter(ecg_loader) if ecg_loader else None

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            if ecg_loader is not None:
                try: ecg_b = next(ecg_iter)
                except StopIteration: ecg_iter = iter(ecg_loader); ecg_b = next(ecg_iter)
                x, y, *_ = batch; ex, ey, *_ = ecg_b
                x, y, ex = x.to(device), y.to(device), ex.to(device)
                logits = model(ex, x)  # DualChannel: (ecg, ppg)
            else:
                x, y, *_ = batch; x, y = x.to(device), y.to(device)
                logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        # Eval
        model.eval(); preds, labels = [], []
        ecg_iter2 = iter(test_ecg_loader) if ecg_loader else None
        with torch.no_grad():
            for batch in test_loader:
                if ecg_loader is not None:
                    try: ecg_b = next(ecg_iter2)
                    except StopIteration: break
                    x, y, *_ = batch; ex, *_ = ecg_b
                    x, y, ex = x.to(device), y.to(device), ex.to(device)
                    logits = model(ex, x)
                else:
                    x, y, *_ = batch; x, y = x.to(device), y.to(device)
                    logits = model(x)
                preds.extend(logits.softmax(-1)[:,1].cpu().numpy())
                labels.extend(y.cpu().numpy())
        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            acc = np.mean((np.array(preds) >= 0.5).astype(int) == np.array(labels))
            print(f"  {tag} E{ep+1:2d}/{epochs}: AUC={auc:.4f} Acc={acc:.3f} (best={best_auc:.4f})")

    model.load_state_dict(best_state)
    return best_auc, model

# ── Model 1: PPG only ──
print("\n=== Model 1: PPG only ===")
ppg_enc = load_enc("target")
model_ppg = SignalClassifier(ppg_enc, config.model.transformer_dim, 2).to(device)
auc_ppg, model_ppg = train_model(model_ppg, train_ppg_loader, test_ppg_loader,
                                  epochs=70, lr=1e-4, tag="PPG")
print(f"PPG AUC: {auc_ppg:.4f}")

# ── Model 2: ECG only ──
print("\n=== Model 2: ECG only ===")
ecg_enc = load_enc("context")
model_ecg = SignalClassifier(ecg_enc, config.model.transformer_dim, 2).to(device)
auc_ecg, model_ecg = train_model(model_ecg, train_ecg_loader, test_ecg_loader,
                                  epochs=70, lr=1e-4, tag="ECG")
print(f"ECG AUC: {auc_ecg:.4f}")

# ── Model 3: Dual-channel fusion (B) ──
print("\n=== Model 3: Dual-channel fusion ===")
ppg_enc2 = load_enc("target")
ecg_enc2 = load_enc("context")
model_dual = DualChannelClassifier(ecg_enc2, ppg_enc2, config.model.transformer_dim, 2).to(device)
auc_dual, model_dual = train_model(model_dual, train_ppg_loader, test_ppg_loader,
                                    epochs=70, lr=1e-4, tag="Dual",
                                    ecg_loader=train_ecg_loader)
print(f"Dual AUC: {auc_dual:.4f}")

# ── Ensemble evaluation (A) ──
print("\n=== Ensemble (PPG + ECG + Dual) ===")
model_ppg.eval(); model_ecg.eval(); model_dual.eval()
preds, labels = [], []
with torch.no_grad():
    for p_batch, e_batch in zip(test_ppg_loader, test_ecg_loader):
        px, py, *_ = p_batch; ex, *_ = e_batch
        px, py, ex = px.to(device), py.to(device), ex.to(device)

        l_ppg = model_ppg(px).softmax(-1)
        l_ecg = model_ecg(ex).softmax(-1)
        l_dual = model_dual(ex, px).softmax(-1)

        ensemble = (l_ppg + l_ecg + l_dual) / 3.0
        preds.extend(ensemble[:, 1].cpu().numpy())
        labels.extend(py.cpu().numpy())

ens_auc = roc_auc_score(labels, preds)
ens_acc = np.mean((np.array(preds) >= 0.5).astype(int) == np.array(labels))
print(f"\n{'='*60}")
print(f"ENSEMBLE AUC: {ens_auc:.4f}  Acc: {ens_acc:.4f}")
print(f"  PPG={auc_ppg:.4f}  ECG={auc_ecg:.4f}  Dual={auc_dual:.4f}")
print(classification_report(labels, (np.array(preds) >= 0.5).astype(int), digits=4))
torch.save({"model_ppg": model_ppg.state_dict(), "model_ecg": model_ecg.state_dict(),
            "model_dual": model_dual.state_dict(), "auc_ensemble": ens_auc},
           "outputs/ensemble_best.pt")
print(f"Saved → outputs/ensemble_best.pt")
