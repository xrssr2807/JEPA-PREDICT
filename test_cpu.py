"""
CPU quick-test: small model + synthetic data → verify pipeline.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models.jepa import JEPA, cosine_schedule
from dataset.augment import PhysioAugment

# ── Synthetic dataset ──────────────────────────────────────────
class SyntheticDataset(Dataset):
    """Generate synthetic ECG/PPG signals for pipeline testing."""
    def __init__(self, n_samples=500, length=1000, return_stats=True):
        self.n = n_samples
        self.L = length
        self.return_stats = return_stats
        self.rng = np.random.RandomState(42)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        t = np.linspace(0, 10, self.L)
        # ECG-like: sharp peaks
        ecg = np.sin(2*np.pi*1.2*t)
        ecg += 0.3 * np.sin(2*np.pi*3.6*t + 0.5)
        ecg += 0.1 * self.rng.randn(self.L)
        # PPG-like: smoother waves
        ppg = np.sin(2*np.pi*1.2*t - 0.3)
        ppg += 0.2 * np.sin(2*np.pi*2.4*t + 1.0)
        ppg += 0.08 * self.rng.randn(self.L)
        # Normalize
        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-6)
        ecg_t = torch.from_numpy(ecg).float().unsqueeze(0)
        ppg_t = torch.from_numpy(ppg).float().unsqueeze(0)
        if self.return_stats:
            from dataset.data import compute_signal_stats
            stats = compute_signal_stats(ecg)
            return ecg_t, ppg_t, torch.from_numpy(stats)
        return ecg_t, ppg_t

# ── Config ─────────────────────────────────────────────────────
class C:
    epochs = 5
    batch_size = 32
    lr = 1e-3
    warmup_epochs = 1
    weight_decay = 0.01
    # Tiny model
    in_channels = 1
    cnn_channels = (32, 64, 128, 128)
    cnn_kernels = (7, 5, 5, 3)
    cnn_strides = (2, 2, 2, 2)
    transformer_layers = 2
    transformer_dim = 128
    transformer_heads = 4
    transformer_ff_dim = 512
    transformer_dropout = 0.0
    max_seq_len = 100
    pool_type = "adaptive_avg"
    embedding_dim = 64
    predictor_hidden = 64
    latent_dim = 16
    num_latent_samples = 4
    ema_momentum = 0.996
    ema_end = 1.0
    use_stats_loss = True
    stats_loss_weight = 0.1
    use_augment = True
    signal_len = 1000

cfg = C()

# ── Data ───────────────────────────────────────────────────────
train_ds = SyntheticDataset(n_samples=400, length=cfg.signal_len, return_stats=True)
test_ds = SyntheticDataset(n_samples=100, length=cfg.signal_len, return_stats=True)
train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

# Augment
aug = PhysioAugment(seed=42) if cfg.use_augment else None

print(f"Train: {len(train_ds)} samples, {len(train_dl)} batches")
print(f"Test:  {len(test_ds)} samples, {len(test_dl)} batches")
print(f"Model: {cfg.transformer_layers} layers, dim={cfg.transformer_dim}, "
      f"stats_loss={cfg.use_stats_loss}, augment={cfg.use_augment}")

# ── Model ──────────────────────────────────────────────────────
model = JEPA(
    in_channels=cfg.in_channels,
    cnn_channels=cfg.cnn_channels,
    cnn_kernel_sizes=cfg.cnn_kernels,
    cnn_strides=cfg.cnn_strides,
    transformer_layers=cfg.transformer_layers,
    transformer_dim=cfg.transformer_dim,
    transformer_heads=cfg.transformer_heads,
    transformer_ff_dim=cfg.transformer_ff_dim,
    transformer_dropout=cfg.transformer_dropout,
    max_seq_len=cfg.max_seq_len,
    pool_type=cfg.pool_type,
    embedding_dim=cfg.embedding_dim,
    predictor_hidden=cfg.predictor_hidden,
    latent_dim=cfg.latent_dim,
    num_latent_samples=cfg.num_latent_samples,
    ema_momentum=cfg.ema_momentum,
    use_stats_loss=cfg.use_stats_loss,
    stats_loss_weight=cfg.stats_loss_weight,
)
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {n_params:,} total, {n_trainable:,} trainable")

# ── Optimizer ──────────────────────────────────────────────────
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
steps_per_epoch = len(train_dl)
total_steps = steps_per_epoch * cfg.epochs
warmup_steps = steps_per_epoch * cfg.warmup_epochs
cosine_steps = total_steps - warmup_steps
warmup_sch = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine_sch = CosineAnnealingLR(optimizer, T_max=cosine_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_steps])

# ── Training ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting CPU training...")
print("=" * 60)

for epoch in range(cfg.epochs):
    model.train()
    epoch_start = time.time()
    train_loss = 0.0
    train_jepa = 0.0
    train_stats = 0.0

    for bi, batch in enumerate(train_dl):
        if len(batch) == 3:
            ecg, ppg, stats = batch
        else:
            ecg, ppg = batch
            stats = None

        # Augment ECG
        if aug is not None:
            ecg_np = ecg.numpy()
            for b in range(ecg_np.shape[0]):
                ecg_np[b] = aug(ecg_np[b])
            ecg = torch.from_numpy(ecg_np.copy()).float()

        global_step = epoch * steps_per_epoch + bi
        ema_m = cosine_schedule(cfg.ema_momentum, cfg.ema_end, global_step / total_steps)

        loss, info = model.compute_loss(ecg, ppg, stats)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        model.update_target_encoder(ema_m)

        train_loss += info["total_loss"]
        train_jepa += info.get("jepa", 0)
        train_stats += info.get("stats", 0)

        if bi % 10 == 0:
            print(f"  E{epoch} B{bi:3d} | loss={info['total_loss']:.4f} "
                  f"jepa={info.get('jepa',0):.4f} stats={info.get('stats',0):.4f} "
                  f"| EMA={ema_m:.4f}")

    n = steps_per_epoch
    epoch_time = time.time() - epoch_start

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_dl:
            if len(batch) == 3:
                ecg, ppg, stats = batch
            else:
                ecg, ppg = batch; stats = None
            _, info = model.compute_loss(ecg, ppg, stats)
            val_loss += info["total_loss"]
    val_loss /= len(test_dl)

    print(f"Epoch {epoch:2d} | Train: {train_loss/n:.4f} "
          f"(jepa={train_jepa/n:.4f} stats={train_stats/n:.4f}) "
          f"| Val: {val_loss:.4f} | Time: {epoch_time:.1f}s")
    print("-" * 60)

# ── Final ──────────────────────────────────────────────────────
print("\nTraining complete!")
print(f"Final train loss: {train_loss/len(train_dl):.4f}")
print(f"Final val loss:  {val_loss:.4f}")

# Quick inference test
model.eval()
with torch.no_grad():
    sample_ecg = torch.randn(2, 1, cfg.signal_len)
    sample_ppg = torch.randn(2, 1, cfg.signal_len)
    pred, target, ctx = model(sample_ecg, sample_ppg)
    print(f"Inference: pred={pred.shape}, target={target.shape}, ctx={ctx.shape}")
    print(f"MSE (random): {nn.functional.mse_loss(pred, target):.4f}")
print("Pipeline verified OK!")
