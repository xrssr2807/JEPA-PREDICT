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
        dataset = PretrainDataset(
            data_dir=data_config.pretrain_dir,
            channels=data_config.pretrain_channels,
            normalize=data_config.normalize,
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
    )


def train(config: Config):
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

    # LR schedule: warmup + cosine
    warmup_steps = config.train.pretrain_warmup_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Logging
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "pretrain_log.txt")

    best_loss = float("inf")

    for epoch in range(config.train.pretrain_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_start = time.time()

        for batch_idx, (ecg, ppg) in enumerate(dataloader):
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

            # Forward + loss
            loss, info = model.compute_loss(ecg, ppg)
            epoch_losses["loss"] += loss.item()

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


if __name__ == "__main__":
    config = Config()
    train(config)
