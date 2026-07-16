"""
CPU training with real data: JEPA ECG→PPG pre-training.
"""
import os, sys, time, math
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.jepa import JEPA, cosine_schedule
from dataset.data import PretrainDataset
from dataset.augment import PhysioAugment

# ── CONFIG ─────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\86189\Downloads\split\split"
OUTPUT_DIR = r"F:\skills\jepa-ecg-ppg\outputs"

CFG = dict(
    # Data
    normalize="iqr",        # IQR robust normalization
    normalize_clip=10.0,
    use_augment=True,
    max_files=500,          # ~2.5 min/epoch on CPU

    # Architecture (small but real)
    cnn_channels=(64, 128, 256, 256),
    cnn_kernels=(7, 5, 5, 3),
    cnn_strides=(2, 2, 2, 2),
    transformer_layers=4,
    transformer_dim=256,
    transformer_heads=8,
    transformer_ff_dim=1024,
    transformer_dropout=0.1,
    embedding_dim=128,
    predictor_hidden=128,
    latent_dim=32,
    num_latent_samples=4,

    # EMA
    ema_momentum=0.996,
    ema_end=1.0,

    # Auxiliary losses
    use_stats_loss=True,
    stats_loss_weight=0.1,

    # Training
    epochs=30,
    batch_size=16,
    lr=1e-3,
    warmup_epochs=2,
    weight_decay=0.05,
    grad_clip=1.0,
    save_every=5,
)

# ── DATA ───────────────────────────────────────────────────────
ds = PretrainDataset(
    data_dir=DATA_DIR,
    channels=[0, 4],
    normalize=CFG["normalize"],
    normalize_clip=CFG["normalize_clip"],
    max_files=CFG["max_files"],
    augment=CFG["use_augment"],
    augment_config=dict(jitter_std=0.02, scale_range=(0.85, 1.15),
                        max_shift=50, wander_amp=0.05, apply_prob=0.8),
    return_stats=CFG["use_stats_loss"],
)

dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True, drop_last=True, num_workers=0)
print(f"Data: {len(ds)} files, {len(dl)} batches/epoch @ batch={CFG['batch_size']}")

# ── MODEL ──────────────────────────────────────────────────────
model = JEPA(
    in_channels=1,
    cnn_channels=tuple(CFG["cnn_channels"]),
    cnn_kernel_sizes=tuple(CFG["cnn_kernels"]),
    cnn_strides=tuple(CFG["cnn_strides"]),
    transformer_layers=CFG["transformer_layers"],
    transformer_dim=CFG["transformer_dim"],
    transformer_heads=CFG["transformer_heads"],
    transformer_ff_dim=CFG["transformer_ff_dim"],
    transformer_dropout=CFG["transformer_dropout"],
    embedding_dim=CFG["embedding_dim"],
    predictor_hidden=CFG["predictor_hidden"],
    latent_dim=CFG["latent_dim"],
    num_latent_samples=CFG["num_latent_samples"],
    ema_momentum=CFG["ema_momentum"],
    use_stats_loss=CFG["use_stats_loss"],
    stats_loss_weight=CFG["stats_loss_weight"],
)

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {n_total:,} params ({n_train:,} trainable)")

# ── OPTIMIZER ──────────────────────────────────────────────────
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=CFG["weight_decay"],
                         betas=(0.9, 0.95))

steps_per_epoch = len(dl)
total_steps = steps_per_epoch * CFG["epochs"]
warmup_steps = steps_per_epoch * CFG["warmup_epochs"]

# ── TRAINING ───────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
log = open(os.path.join(OUTPUT_DIR, "cpu_train_log.txt"), "w")
best_loss = float("inf")

print(f"\n{'='*60}")
print(f"Training: {CFG['epochs']} epochs × {steps_per_epoch} batches = {total_steps} steps")
print(f"Warmup: {CFG['warmup_epochs']} epochs ({warmup_steps} steps)")
print(f"Norm: {CFG['normalize']} | Augment: {CFG['use_augment']} | Stats: {CFG['use_stats_loss']}")
print(f"{'='*60}\n")

for epoch in range(CFG["epochs"]):
    model.train()
    t0 = time.time()
    ep_loss = ep_jepa = ep_stats = 0.0
    n = 0

    for bi, batch in enumerate(dl):
        if len(batch) == 3:
            ecg, ppg, stats = batch
        else:
            ecg, ppg = batch; stats = None

        gs = epoch * steps_per_epoch + bi

        # LR schedule: warmup + cosine
        if gs < warmup_steps:
            lr = CFG["lr"] * (gs + 1) / max(warmup_steps, 1)
        else:
            progress = (gs - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = CFG["lr"] * 0.5 * (1 + math.cos(math.pi * progress))

        for pg in opt.param_groups:
            pg["lr"] = lr

        # EMA schedule
        ema_m = cosine_schedule(CFG["ema_momentum"], CFG["ema_end"], gs / total_steps)

        loss, info = model.compute_loss(ecg, ppg, stats)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, CFG["grad_clip"])
        opt.step()
        model.update_target_encoder(ema_m)

        ep_loss += info["total_loss"]
        ep_jepa += info.get("jepa", 0)
        ep_stats += info.get("stats", 0)
        n += 1

        if bi % 10 == 0:
            line = (f"  E{epoch:2d} B{bi:3d}/{steps_per_epoch} | "
                    f"loss={info['total_loss']:.4f} "
                    f"j={info.get('jepa',0):.4f} s={info.get('stats',0):.4f} "
                    f"| lr={lr:.2e} ema={ema_m:.4f}")
            print(line)

    ep_loss /= n; ep_jepa /= n; ep_stats /= n
    t = time.time() - t0

    line = (f"Epoch {epoch:2d} | L={ep_loss:.4f} "
            f"j={ep_jepa:.4f} s={ep_stats:.4f} "
            f"| {t:.1f}s")
    print(line); print("-" * 60)
    log.write(line + "\n"); log.flush()

    # Save
    if ep_loss < best_loss:
        best_loss = ep_loss
        torch.save({
            "epoch": epoch, "loss": ep_loss,
            "model_state_dict": model.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
        }, os.path.join(OUTPUT_DIR, "jepa_best.pt"))
        print(f"  → Best model saved (loss={best_loss:.4f})")

    if (epoch + 1) % CFG["save_every"] == 0:
        torch.save({
            "epoch": epoch,
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
        }, os.path.join(OUTPUT_DIR, f"jepa_epoch_{epoch+1}.pt"))

log.close()
print(f"\nDone! Best loss: {best_loss:.4f}")
