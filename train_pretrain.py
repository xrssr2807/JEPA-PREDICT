"""
JEPA Pre-training: ECG → PPG cross-channel predictive learning.
"""
import os
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import os

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import PretrainDataset, PretrainDatasetPT
from models.jepa import JEPA, cosine_schedule


def build_dataloader(
    data_config: DataConfig,
    train_config: TrainConfig,
    use_processed: bool = True,
) -> DataLoader:
    """构建预训练 DataLoader。

    Args:
        data_config: 数据配置
        train_config: 训练配置
        use_processed: True=加载预处理好的 .pt 文件（更快），
                       False=从原始 .pkl 实时处理
    """
    processed_dir = os.path.join(data_config.pretrain_processed_dir)
    if use_processed and os.path.isdir(processed_dir):
        print(f"[DataLoader] 使用预处理数据: {processed_dir}")
        dataset = PretrainDatasetPT(data_dir=processed_dir)
        num_workers = 4  # 预处理数据加载快，可以开多进程
    else:
        print(f"[DataLoader] 从原始数据加载: {data_config.pretrain_dir}")
        augment_kwargs = {}
        if data_config.use_augment:
            augment_kwargs = dict(
                augment=True,
                augment_config=dict(
                    jitter_std=data_config.augment_jitter_std,
                    scale_range=(data_config.augment_scale_min, data_config.augment_scale_max),
                    max_shift=data_config.augment_max_shift,
                    wander_amp=data_config.augment_wander_amp,
                    apply_prob=data_config.augment_apply_prob,
                ),
            )
        return_stats = data_config.normalize != "none"  # use stats for auxiliary loss
        dataset = PretrainDataset(
            data_dir=data_config.pretrain_dir,
            channels=data_config.pretrain_channels,
            normalize=data_config.normalize,
            normalize_clip=data_config.normalize_clip,
            return_stats=return_stats,
            **augment_kwargs,
        )
        num_workers = 0  # pickle 加载受限

    return DataLoader(
        dataset,
        batch_size=train_config.pretrain_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_model(model_config: ModelConfig) -> JEPA:
    return JEPA(
        in_channels=model_config.in_channels,
        cnn_channels=tuple(model_config.cnn_channels),
        cnn_kernel_sizes=tuple(model_config.cnn_kernel_sizes),
        cnn_strides=tuple(model_config.cnn_strides),
        transformer_layers=model_config.transformer_layers,
        transformer_dim=model_config.transformer_dim,
        transformer_heads=model_config.transformer_heads,
        transformer_ff_dim=model_config.transformer_ff_dim,
        transformer_dropout=model_config.transformer_dropout,
        max_seq_len=model_config.max_seq_len,
        pool_type=model_config.pool_type,
        embedding_dim=model_config.embedding_dim,
        predictor_hidden=model_config.predictor_hidden,
        latent_dim=model_config.latent_dim,
        num_latent_samples=model_config.num_latent_samples,
        ema_momentum=model_config.ema_momentum,
        # ★ JETS 掩码
        mask_ratio=model_config.jets_mask_ratio,
        mask_patch_size=model_config.jets_mask_patch_size,
        use_stats_loss=model_config.use_stats_loss,
        stats_loss_weight=model_config.stats_loss_weight,
        use_contrast_loss=model_config.use_contrast_loss,
        contrast_loss_weight=model_config.contrast_loss_weight,
    )


def train(config: Config, resume_from: str = None, start_epoch: int = 0):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataloader = build_dataloader(config.data, config.train)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.train.pretrain_epochs

    # Model
    model = build_model(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    trainable_params = list(model.context_encoder.parameters()) + \
                       list(model.context_proj.parameters()) + \
                       list(model.predictor.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=config.train.pretrain_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.pretrain_weight_decay,
    )

    # ── Resume logic ──
    if resume_from is not None:
        print(f"[Resume] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        # Load encoder weights
        if "context_encoder" in ckpt:
            model.context_encoder.load_state_dict(ckpt["context_encoder"])
            print("[Resume] Loaded context_encoder weights")
        if "target_encoder" in ckpt:
            model.target_encoder.load_state_dict(ckpt["target_encoder"])
            print("[Resume] Loaded target_encoder weights")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print("[Resume] Loaded full model_state_dict")
        # Restore optimizer if available
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("[Resume] Loaded optimizer state")
            except Exception as e:
                print(f"[Resume] Optimizer state restore failed (reinit): {e}")
        print(f"[Resume] Continuing from epoch {start_epoch}")

    # LR schedule: warmup + cosine (adjusted for resume)
    remaining_epochs = config.train.pretrain_epochs - start_epoch
    remaining_steps = remaining_epochs * steps_per_epoch
    # Skip warmup when resuming (already past warmup phase)
    if start_epoch >= config.train.pretrain_warmup_epochs:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=1.0, total_iters=1
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[1],
        )
    else:
        warmup_steps = (config.train.pretrain_warmup_epochs - start_epoch) * steps_per_epoch
        cosine_steps = remaining_steps - warmup_steps
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(1, warmup_steps)
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cosine_steps))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[max(1, warmup_steps)],
        )

    # Logging
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "pretrain_log.txt")

    best_loss = float("inf")

    for epoch in range(start_epoch, config.train.pretrain_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_start = time.time()

        for batch_idx, batch_data in enumerate(dataloader):
            # Handle both old (ecg, ppg) and new (ecg, ppg, stats) formats
            if len(batch_data) == 3:
                ecg, ppg, ecg_stats = batch_data
                ecg_stats = ecg_stats.to(device)
            else:
                ecg, ppg = batch_data
                ecg_stats = None

            ecg = ecg.to(device)
            ppg = ppg.to(device)

            global_step = epoch * steps_per_epoch + batch_idx

            # EMA momentum schedule (cosine towards 1.0)
            ema_progress = global_step / total_steps
            ema_momentum = cosine_schedule(
                config.model.ema_momentum,
                config.model.ema_end_momentum,
                ema_progress,
            )

            # Forward + loss (with optional stats)
            loss, info = model.compute_loss(ecg, ppg, ecg_stats)
            epoch_losses["loss"] += info.get("total_loss", loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # EMA update target encoder
            model.update_target_encoder(ema_momentum)

            if batch_idx % 50 == 0:
                log_msg = (
                    f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.6f} | EMA: {ema_momentum:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                print(log_msg)

        # Epoch summary
        epoch_loss = epoch_losses["loss"] / steps_per_epoch
        epoch_time = time.time() - epoch_start

        summary = f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | Time: {epoch_time:.1f}s"
        print(summary)
        print("-" * 60)

        with open(log_file, "a") as f:
            f.write(summary + "\n")

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(config.output_dir, "jepa_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

        # Save periodic
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(config.output_dir, f"jepa_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                },
                ckpt_path,
            )

    print("Pre-training complete.")
    return model


def train_token_align(config: Config, checkpoint_path: str):
    """
    Token 对齐续训练 (方案A).
    加载已预训练的 checkpoint → 冻结 target_encoder → 只训练 context_encoder
    逐 token 对齐: ECG_token_i ↔ PPG_token_i

    论文: Cross-Modal Representational KD (NeurIPS 2025)
    替代: M2AE InfoNCE 对比损失
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("★ Token 级跨模态对齐续训练 (方案A)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Data
    dataloader = build_dataloader(config.data, config.train, config.model)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.train.token_align_epochs
    print(f"Data: {len(dataloader.dataset)} samples, {steps_per_epoch} steps/epoch")

    # Model
    model = build_model(config.model).to(device)

    # 加载完整 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Load] Full model state from {checkpoint_path}")
    else:
        model.context_encoder.load_state_dict(ckpt["context_encoder"])
        model.target_encoder.load_state_dict(ckpt["target_encoder"])
        model.context_proj.load_state_dict(ckpt["context_proj"])
        model.target_proj.load_state_dict(ckpt["target_proj"])
        print(f"[Load] Encoder weights from {checkpoint_path}")

    # ★ 冻结 target_encoder (做 teacher, 不更新)
    model.freeze_target_encoder()

    # ★ 开启 Token 对齐模式 (替代 InfoNCE)
    model.use_token_align = True
    model.token_align_weight = config.model.token_align_weight
    print(f"[TokenAlign] weight={model.token_align_weight}")
    # ★ 开启频域掩码 (WavesFM)
    model.use_freq_loss = config.model.use_freq_loss
    model.freq_loss_weight = config.model.freq_loss_weight
    if model.use_freq_loss:
        print(f"[FreqMask] weight={model.freq_loss_weight}")

    # 只优化 context_encoder + predictor (target 冻结)
    trainable_params = list(model.context_encoder.parameters()) + \
                       list(model.context_proj.parameters()) + \
                       list(model.predictor.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=config.train.token_align_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.pretrain_weight_decay,
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # LR schedule: warmup + cosine
    warmup_steps = min(200, total_steps // 20)
    cosine_steps = total_steps - warmup_steps
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, cosine_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[max(1, warmup_steps)])

    # Logging
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "token_align_log.txt")
    best_loss = float("inf")

    for epoch in range(config.train.token_align_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_start = time.time()

        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 3:
                ecg, ppg, ecg_stats = batch_data
            else:
                ecg, ppg = batch_data
                ecg_stats = None
            ecg, ppg = ecg.to(device), ppg.to(device)

            # 前向: JETS 掩码 → 获取可见 token 位置
            ecg_masked, token_mask = model._apply_jets_mask(ecg)
            token_loss, token_info = model._compute_token_align_loss(
                ecg_masked, ppg, token_mask=token_mask
            )

            # JEPA 预测损失 (保持)
            context_embed = model.forward_context(ecg_masked)
            target_embed = model.forward_target(ppg)
            jepa_loss, jepa_info = model._compute_jepa_loss(ecg, ppg, context_embed, target_embed)

            # 总损失 (DWT 频域已移除: 编码器不参与计算图, 无训练效果)
            loss = jepa_loss + model.token_align_weight * token_loss
            epoch_losses["jepa"] += jepa_loss.item()
            epoch_losses["token_align"] += token_loss.item()
            epoch_losses["total"] += loss.item()

            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if batch_idx % 50 == 0:
                print(f"E{epoch:3d} B{batch_idx:4d}/{steps_per_epoch} | "
                      f"JEPA={jepa_loss.item():.4f} Token={token_loss.item():.4f} "
                      f"Total={loss.item():.4f} | LR={scheduler.get_last_lr()[0]:.2e}")

        # Epoch summary
        avg_total = epoch_losses["total"] / steps_per_epoch
        avg_jepa = epoch_losses["jepa"] / steps_per_epoch
        avg_token = epoch_losses["token_align"] / steps_per_epoch
        avg_freq = epoch_losses.get("freq", 0) / steps_per_epoch
        epoch_time = time.time() - epoch_start

        summary = (f"E{epoch:2d} | JEPA={avg_jepa:.4f} "
                   f"Token={avg_token:.4f} Freq={avg_freq:.4f} "
                   f"Total={avg_total:.4f} | Time={epoch_time:.1f}s")
        print(summary)
        print("-" * 60)
        with open(log_file, "a") as f:
            f.write(summary + "\n")

        # Save best
        if avg_total < best_loss:
            best_loss = avg_total
            ckpt_path = os.path.join(config.output_dir, "jepa_token_align_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "loss": avg_total,
            }, ckpt_path)
            print(f"Saved best → {ckpt_path}")

    print(f"\nToken 对齐续训练完成! 最佳 loss = {best_loss:.4f}")
    print(f"输出: {os.path.join(config.output_dir, 'jepa_token_align_best.pt')}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JEPA Pre-training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to start/resume from")
    parser.add_argument("--token_align", type=str, default=None,
                        help="Token 对齐续训练: --token_align outputs/jepa_best.pt")
    args = parser.parse_args()

    config = Config()

    if args.token_align is not None:
        # ★ Token 对齐续训练模式
        train_token_align(config, args.token_align)
    else:
        # 正常预训练
        train(config, resume_from=args.resume, start_epoch=args.start_epoch)
