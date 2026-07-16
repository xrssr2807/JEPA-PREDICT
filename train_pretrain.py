"""
JEPA Pre-training: ECG → PPG cross-channel predictive learning.
"""
import os
import json
import math
import random
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import (
    PretrainDataset,
    PretrainDatasetPT,
    infer_pretrain_uid,
    split_pretrain_files,
)
from models.jepa import JEPA, cosine_schedule


def seed_everything(seed: int, deterministic: bool = True):
    """Seed Python, NumPy and PyTorch for a reproducible baseline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_pretrain_dataset(
    data_config: DataConfig,
    use_processed: bool,
    return_stats: bool,
    files=None,
    augment: bool = False,
):
    if use_processed:
        return PretrainDatasetPT(
            data_dir=data_config.pretrain_processed_dir,
            return_stats=return_stats,
            files=files,
        )

    augment_kwargs = {}
    if augment and data_config.use_augment:
        augment_kwargs = dict(
            augment=True,
            augment_config=dict(
                jitter_std=data_config.augment_jitter_std,
                scale_range=(
                    data_config.augment_scale_min,
                    data_config.augment_scale_max,
                ),
                max_shift=data_config.augment_max_shift,
                wander_amp=data_config.augment_wander_amp,
                apply_prob=data_config.augment_apply_prob,
            ),
        )
    return PretrainDataset(
        data_dir=data_config.pretrain_dir,
        channels=data_config.pretrain_channels,
        normalize=data_config.normalize,
        normalize_clip=data_config.normalize_clip,
        return_stats=return_stats,
        files=files,
        **augment_kwargs,
    )


def build_pretrain_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    return_stats: bool,
    use_processed: bool = True,
    seed: int = 42,
):
    """Build deterministic patient-grouped train/validation loaders."""
    processed_dir = data_config.pretrain_processed_dir
    use_processed = use_processed and os.path.isdir(processed_dir)
    source = processed_dir if use_processed else data_config.pretrain_dir
    print(f"[DataLoader] Source: {source} ({'PT' if use_processed else 'PKL'})")

    catalog = _make_pretrain_dataset(
        data_config, use_processed, return_stats, files=None, augment=False
    )
    train_files, val_files = split_pretrain_files(
        catalog.files, data_config.pretrain_val_split, seed
    )
    train_dataset = _make_pretrain_dataset(
        data_config, use_processed, return_stats,
        files=train_files, augment=True,
    )
    val_dataset = _make_pretrain_dataset(
        data_config, use_processed, return_stats,
        files=val_files, augment=False,
    )

    train_uids = {infer_pretrain_uid(name) for name in train_files}
    val_uids = {infer_pretrain_uid(name) for name in val_files}
    if train_uids & val_uids:
        raise RuntimeError("Patient leakage detected in pre-training split")
    print(
        f"[DataSplit] train={len(train_files)} segments/{len(train_uids)} UIDs | "
        f"val={len(val_files)} segments/{len(val_uids)} UIDs | seed={seed}"
    )

    num_workers = 4 if use_processed else 0
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    common = dict(
        batch_size=train_config.pretrain_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        generator=val_generator,
        **common,
    )
    return train_loader, val_loader


def build_dataloader(
    data_config: DataConfig,
    train_config: TrainConfig,
    use_processed: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Compatibility wrapper returning the Phase 0 training loader."""
    train_loader, _ = build_pretrain_dataloaders(
        data_config,
        train_config,
        return_stats=False,
        use_processed=use_processed,
        seed=seed,
    )
    return train_loader


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
        use_se=model_config.cnn_use_se,
        use_inception=model_config.cnn_use_inception,
        use_token_align=model_config.use_token_align,
        token_align_weight=model_config.token_align_weight,
        token_align_window=model_config.token_align_window,
    )


def _move_pretrain_batch(batch_data, device):
    if len(batch_data) == 3:
        ecg, ppg, ecg_stats = batch_data
        ecg_stats = ecg_stats.to(device, non_blocking=True)
    else:
        ecg, ppg = batch_data
        ecg_stats = None
    return (
        ecg.to(device, non_blocking=True),
        ppg.to(device, non_blocking=True),
        ecg_stats,
    )


def _accumulate_metrics(totals, info):
    for key, value in info.items():
        if isinstance(value, (int, float)):
            totals[key] += float(value)


@torch.no_grad()
def evaluate_pretrain(model, dataloader, device, seed: int):
    """Evaluate a stable, unmasked held-out objective and collapse metrics."""
    model.eval()
    totals = defaultdict(float)
    num_batches = 0
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index or torch.cuda.current_device()]

    # Predictor z samples are identical across epochs for comparable val loss.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batch_data in dataloader:
            ecg, ppg, ecg_stats = _move_pretrain_batch(batch_data, device)
            loss, info = model.compute_loss(
                ecg,
                ppg,
                ecg_stats,
                collect_diagnostics=True,
            )
            info.setdefault("total_loss", loss.item())
            _accumulate_metrics(totals, info)
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Pre-training validation loader produced zero batches")
    metrics = {key: value / num_batches for key, value in totals.items()}
    metrics["teacher_student_cosine"] = model.teacher_student_parameter_cosine()
    return metrics


def _metric(metrics, key: str) -> float:
    return float(metrics.get(key, 0.0))


def _load_jepa_state_dict(model, state_dict):
    """Load old checkpoints while allowing only the new Phase 0 stats counter."""
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"stats_pred_head.num_updates"}
    unexpected_missing = set(result.missing_keys) - allowed_missing
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint structure mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if result.missing_keys:
        print(f"[Resume] Initialized new buffers: {result.missing_keys}")


def train(config: Config, resume_from: str = None, start_epoch: int = 0):
    seed_everything(config.seed, config.deterministic)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | seed={config.seed} | "
        f"deterministic={config.deterministic}"
    )

    # Data
    train_loader, val_loader = build_pretrain_dataloaders(
        config.data,
        config.train,
        return_stats=config.model.use_stats_loss,
        use_processed=True,
        seed=config.seed,
    )
    steps_per_epoch = len(train_loader)
    accum_steps = max(1, config.train.pretrain_accum_steps)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_steps = optimizer_steps_per_epoch * config.train.pretrain_epochs

    # Model
    model = build_model(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    trainable_params = list(model.context_encoder.parameters()) + \
                       list(model.context_proj.parameters()) + \
                       list(model.predictor.parameters())
    if model.use_stats_loss:
        trainable_params += list(model.stats_pred_head.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=config.train.pretrain_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.pretrain_weight_decay,
    )

    # ── Resume logic ──
    resume_best_loss = float("inf")
    if resume_from is not None:
        print(f"[Resume] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        # Load encoder weights
        if "context_encoder" in ckpt:
            model.context_encoder.load_state_dict(ckpt["context_encoder"])
            print("[Resume] Loaded context_encoder weights")
        if "target_encoder" in ckpt:
            model.target_encoder.load_state_dict(ckpt["target_encoder"])
            print("[Resume] Loaded target_encoder weights")
        if "model_state_dict" in ckpt:
            _load_jepa_state_dict(model, ckpt["model_state_dict"])
            print("[Resume] Loaded full model_state_dict")
        # Restore optimizer if available
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("[Resume] Loaded optimizer state")
            except Exception as e:
                print(f"[Resume] Optimizer state restore failed (reinit): {e}")
        resume_best_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        model._enforce_teacher_eval()
        print(f"[Resume] Continuing from epoch {start_epoch}")

    # LR schedule: warmup + cosine (adjusted for resume)
    remaining_epochs = config.train.pretrain_epochs - start_epoch
    remaining_steps = remaining_epochs * optimizer_steps_per_epoch
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
        warmup_steps = (
            config.train.pretrain_warmup_epochs - start_epoch
        ) * optimizer_steps_per_epoch
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
    split_manifest = {
        "seed": config.seed,
        "val_ratio": config.data.pretrain_val_split,
        "train_files": list(train_loader.dataset.files),
        "val_files": list(val_loader.dataset.files),
        "train_uids": sorted({
            infer_pretrain_uid(name) for name in train_loader.dataset.files
        }),
        "val_uids": sorted({
            infer_pretrain_uid(name) for name in val_loader.dataset.files
        }),
    }
    with open(os.path.join(config.output_dir, "pretrain_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    best_loss = resume_best_loss
    optimizer_step = start_epoch * optimizer_steps_per_epoch

    for epoch in range(start_epoch, config.train.pretrain_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(train_loader):
            ecg, ppg, ecg_stats = _move_pretrain_batch(batch_data, device)

            # ★ MixUp: 随机混合batch内样本 → 正则化
            if config.train.use_mixup and config.train.mixup_alpha > 0:
                lam = np.random.beta(config.train.mixup_alpha, config.train.mixup_alpha)
                idx = torch.randperm(ecg.size(0), device=device)
                ecg = lam * ecg + (1 - lam) * ecg[idx]
                ppg = lam * ppg + (1 - lam) * ppg[idx]
                if ecg_stats is not None:
                    ecg_stats = lam * ecg_stats + (1 - lam) * ecg_stats[idx]

            # EMA momentum schedule (cosine towards 1.0)
            ema_progress = optimizer_step / max(total_steps - 1, 1)
            ema_momentum = cosine_schedule(
                config.model.ema_momentum,
                config.model.ema_end_momentum,
                ema_progress,
            )

            # Forward + loss (with optional stats)
            loss, info = model.compute_loss(ecg, ppg, ecg_stats)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_idx}. "
                    "Check the preprocessed ECG/PPG tensors and ecg_stats."
                )
            _accumulate_metrics(epoch_losses, info)

            group_start = (batch_idx // accum_steps) * accum_steps
            group_size = min(accum_steps, steps_per_epoch - group_start)
            (loss / group_size).backward()

            should_step = (
                (batch_idx + 1) % accum_steps == 0
                or (batch_idx + 1) == steps_per_epoch
            )
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0, error_if_nonfinite=False
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, "
                        f"batch={batch_idx}; optimizer step was skipped."
                    )
                optimizer.step()
                scheduler.step()
                model.update_target_encoder(ema_momentum)
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if batch_idx % 50 == 0:
                log_msg = (
                    f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.6f} | JEPA: {_metric(info, 'jepa'):.5f} | "
                    f"Stats: {_metric(info, 'stats'):.5f} | "
                    f"CtxStd: {_metric(info, 'context_std'):.4f} | "
                    f"TgtStd: {_metric(info, 'target_std'):.4f} | "
                    f"EMA: {ema_momentum:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                print(log_msg)

        # Epoch summary
        train_metrics = {
            key: value / steps_per_epoch for key, value in epoch_losses.items()
        }
        epoch_loss = _metric(train_metrics, "total_loss")
        epoch_time = time.time() - epoch_start

        should_validate = (
            (epoch + 1) % max(1, config.train.pretrain_val_every) == 0
            or epoch == config.train.pretrain_epochs - 1
        )
        val_metrics = None
        if should_validate:
            val_metrics = evaluate_pretrain(
                model, val_loader, device, seed=config.seed + 10_000
            )

        summary = (
            f"Epoch {epoch:3d} | "
            f"Train total={epoch_loss:.6f} jepa={_metric(train_metrics, 'jepa'):.6f} "
            f"stats={_metric(train_metrics, 'stats'):.6f} "
            f"token={_metric(train_metrics, 'token_align'):.6f} "
            f"ctx_std={_metric(train_metrics, 'context_std'):.4f} "
            f"tgt_std={_metric(train_metrics, 'target_std'):.4f}"
        )
        if val_metrics is not None:
            summary += (
                f" | Val total={_metric(val_metrics, 'total_loss'):.6f} "
                f"jepa={_metric(val_metrics, 'jepa'):.6f} "
                f"stats={_metric(val_metrics, 'stats'):.6f} "
                f"ctx_std={_metric(val_metrics, 'context_std'):.4f} "
                f"tgt_std={_metric(val_metrics, 'target_std'):.4f} "
                f"ctx_collapse={_metric(val_metrics, 'context_collapsed_fraction'):.3f} "
                f"tgt_collapse={_metric(val_metrics, 'target_collapsed_fraction'):.3f} "
                f"ctx_cov={_metric(val_metrics, 'context_cov_offdiag_rms'):.4f} "
                f"tgt_cov={_metric(val_metrics, 'target_cov_offdiag_rms'):.4f} "
                f"teacher_cos={_metric(val_metrics, 'teacher_student_cosine'):.6f}"
            )
            if (
                _metric(val_metrics, "context_collapsed_fraction") > 0.90
                or _metric(val_metrics, "target_collapsed_fraction") > 0.90
            ):
                print("[CollapseWarning] More than 90% of embedding dimensions have near-zero variance")
        summary += f" | Time: {epoch_time:.1f}s"
        print(summary)
        print("-" * 60)

        with open(log_file, "a") as f:
            f.write(summary + "\n")

        # Save best by patient-held-out validation loss, not training loss.
        current_val_loss = (
            _metric(val_metrics, "total_loss") if val_metrics is not None else None
        )
        if current_val_loss is not None and current_val_loss < best_loss:
            best_loss = current_val_loss
            checkpoint_path = os.path.join(config.output_dir, "jepa_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "train_metrics": train_metrics,
                    "val_loss": current_val_loss,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_loss,
                    "seed": config.seed,
                    "train_segments": len(train_loader.dataset),
                    "val_segments": len(val_loader.dataset),
                },
                checkpoint_path,
            )
            print(f"Saved best validation model to {checkpoint_path}")

        # Save periodic
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(config.output_dir, f"jepa_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_loss,
                    "seed": config.seed,
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
    seed_everything(config.seed, config.deterministic)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("★ Token 级跨模态对齐续训练 (方案A)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Data
    dataloader = build_dataloader(
        config.data, config.train, use_processed=True, seed=config.seed
    )
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.train.token_align_epochs
    print(f"Data: {len(dataloader.dataset)} samples, {steps_per_epoch} steps/epoch")

    # Model
    model = build_model(config.model).to(device)

    # 加载完整 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        _load_jepa_state_dict(model, ckpt["model_state_dict"])
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
        model.target_encoder.eval()
        model.target_proj.eval()
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
            epoch_losses["token_std"] += token_info.get("token_std", 0.0)
            epoch_losses["visible"] += token_info.get("visible", 0.0)
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
        avg_std = epoch_losses.get("token_std", 0) / steps_per_epoch
        avg_vis = epoch_losses.get("visible", 0) / steps_per_epoch
        epoch_time = time.time() - epoch_start

        summary = (f"E{epoch:2d} | JEPA={avg_jepa:.4f} Token={avg_token:.6f} "
                   f"std={avg_std:.4f} vis={avg_vis:.2f} "
                   f"Total={avg_total:.4f} | {epoch_time:.0f}s")
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
