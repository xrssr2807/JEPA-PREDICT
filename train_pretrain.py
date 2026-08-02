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
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import (
    PretrainDataset,
    PretrainDatasetPT,
    infer_pretrain_uid,
    split_pretrain_files,
)
from models.jepa import JEPA, cosine_schedule


def seed_everything(
    seed: int,
    deterministic: bool = True,
    enable_tf32: bool = True,
):
    """Seed Python, NumPy and PyTorch for a reproducible baseline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        if enable_tf32:
            torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _global_pretrain_lr_factor(
    optimizer_step: int,
    warmup_steps: int,
    total_steps: int,
    start_factor: float = 1e-6,
) -> float:
    """Return the original warmup-cosine factor at an absolute optimizer step."""
    step = max(0, min(int(optimizer_step), int(total_steps)))
    warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
    if warmup_steps > 0 and step <= warmup_steps:
        progress = step / warmup_steps
        return start_factor + (1.0 - start_factor) * progress
    cosine_steps = max(1, int(total_steps) - warmup_steps)
    cosine_step = min(max(step - warmup_steps, 0), cosine_steps)
    return 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_steps))


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
    batch_size: int = None,
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

    num_workers = (
        max(0, int(train_config.pretrain_dataloader_workers))
        if use_processed else 0
    )
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    common = dict(
        batch_size=batch_size or train_config.pretrain_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        common["prefetch_factor"] = max(
            1, int(train_config.pretrain_prefetch_factor)
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
        use_stats_loss=(
            model_config.use_stats_loss
            if model_config.pretrain_phase == 0
            else (
                model_config.phase2_use_stats_loss
                if model_config.pretrain_phase == 2
                else model_config.phase1_use_stats_loss
            )
        ),
        stats_loss_weight=model_config.stats_loss_weight,
        use_se=model_config.cnn_use_se,
        use_inception=model_config.cnn_use_inception,
        use_token_align=model_config.use_token_align,
        token_align_weight=model_config.token_align_weight,
        token_align_window=model_config.token_align_window,
        pretrain_phase=model_config.pretrain_phase,
        phase1_mask_ratio=model_config.phase1_mask_ratio,
        phase1_mask_block_tokens=model_config.phase1_mask_block_tokens,
        phase1_bidirectional=model_config.phase1_bidirectional,
        phase1_token_loss_weight=model_config.phase1_token_loss_weight,
        phase2_transport_enabled=model_config.phase2_transport_enabled,
        phase2_transport_mode=model_config.phase2_transport_mode,
        phase2_sample_rate_hz=model_config.phase2_sample_rate_hz,
        phase2_min_delay_ms=model_config.phase2_min_delay_ms,
        phase2_max_delay_ms=model_config.phase2_max_delay_ms,
        phase2_delay_prior_ms=model_config.phase2_delay_prior_ms,
        phase2_delay_head_hidden=model_config.phase2_delay_head_hidden,
        phase2_transport_temperature=model_config.phase2_transport_temperature,
        phase2_unmatched_bias=model_config.phase2_unmatched_bias,
        phase2_transport_loss_weight=model_config.phase2_transport_loss_weight,
        phase2_delay_prior_weight=model_config.phase2_delay_prior_weight,
        phase2_monotonic_weight=model_config.phase2_monotonic_weight,
        phase2_delay_smoothness_weight=(
            model_config.phase2_delay_smoothness_weight
        ),
        phase2_match_mass_weight=model_config.phase2_match_mass_weight,
        phase2_target_match_mass=model_config.phase2_target_match_mass,
        phase2_variance_weight=model_config.phase2_variance_weight,
        phase2_covariance_weight=model_config.phase2_covariance_weight,
        phase2_target_std=model_config.phase2_target_std,
        phase2_shared_private_enabled=(
            model_config.phase2_shared_private_enabled
        ),
        phase2_private_dim=model_config.phase2_private_dim,
        phase2_shared_private_hidden=(
            model_config.phase2_shared_private_hidden
        ),
        phase2_private_loss_weight=model_config.phase2_private_loss_weight,
        phase2_orthogonality_weight=(
            model_config.phase2_orthogonality_weight
        ),
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
def evaluate_pretrain(model, dataloader, device, seed: int, use_amp: bool = False):
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
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(use_amp and device.type == "cuda"),
            ):
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


def phase2_transport_progress(
    epoch: int, start_epoch: int, ramp_epochs: int
) -> float:
    """Keep transport off initially, then linearly ramp it to full weight."""
    if epoch < start_epoch:
        return 0.0
    return min((epoch - start_epoch + 1) / max(1, ramp_epochs), 1.0)


def _representation_is_healthy(metrics) -> bool:
    """Reject deceptively low validation losses caused by latent collapse."""
    if not metrics:
        return False
    values = (
        _metric(metrics, "total_loss"),
        _metric(metrics, "context_std"),
        _metric(metrics, "target_std"),
        _metric(metrics, "context_collapsed_fraction"),
        _metric(metrics, "target_collapsed_fraction"),
    )
    if "ecg_private_std" in metrics or "ppg_private_std" in metrics:
        values = values + (
            _metric(metrics, "ecg_private_std"),
            _metric(metrics, "ppg_private_std"),
            _metric(metrics, "ecg_private_collapsed_fraction"),
            _metric(metrics, "ppg_private_collapsed_fraction"),
        )
    if not all(math.isfinite(value) for value in values):
        return False
    healthy = (
        _metric(metrics, "context_std") >= 0.01
        and _metric(metrics, "target_std") >= 0.01
        and _metric(metrics, "context_collapsed_fraction") <= 0.10
        and _metric(metrics, "target_collapsed_fraction") <= 0.10
    )
    if "ecg_private_std" in metrics or "ppg_private_std" in metrics:
        healthy = healthy and (
            _metric(metrics, "ecg_private_std") >= 0.01
            and _metric(metrics, "ppg_private_std") >= 0.01
            and _metric(metrics, "ecg_private_collapsed_fraction") <= 0.10
            and _metric(metrics, "ppg_private_collapsed_fraction") <= 0.10
        )
    return healthy


def _checkpoint_is_eligible(
    metrics,
    phase: int,
    transport_progress: float = 1.0,
    shared_private_progress: float = 1.0,
    transport_required: bool = True,
) -> bool:
    """Compare checkpoints only after every scheduled objective is active."""
    objective_ready = phase != 2 or (
        (not transport_required or transport_progress >= 1.0 - 1e-8)
        and shared_private_progress >= 1.0 - 1e-8
    )
    return objective_ready and _representation_is_healthy(metrics)


def _early_stopping_step(
    best_loss: float,
    bad_epochs: int,
    current_loss: float,
    min_delta: float,
):
    """Update validation plateau state using an absolute improvement margin."""
    if current_loss < best_loss - min_delta:
        return current_loss, 0, True
    return best_loss, bad_epochs + 1, False


def _save_checkpoint(payload, path):
    """Atomically replace a checkpoint so interruptions cannot corrupt it."""
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _encoder_checkpoint_payload(model) -> dict:
    """Expose stable downstream keys plus dual-online/teacher weights."""
    payload = {
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
    }
    if model.pretrain_phase >= 1:
        payload.update({
            "ppg_encoder": model.ppg_encoder.state_dict(),
            "context_teacher": model.context_teacher.state_dict(),
        })
    return payload


def _phase_checkpoint_metadata(model, config: Config) -> dict:
    """Persist the Phase 2 schedule and physical delay interpretation."""
    if model.pretrain_phase != 2:
        return {}
    return {
        "phase2_transport_progress": float(model.phase2_progress),
        "phase2_config": {
            "transport_enabled": bool(
                config.model.phase2_transport_enabled
            ),
            "transport_mode": str(config.model.phase2_transport_mode),
            "effective_constraint_weights": (
                model._phase2_effective_regularizer_weights()
            ),
            "sample_rate_hz": float(config.model.phase2_sample_rate_hz),
            "token_ms": float(model.phase2_token_ms),
            "delay_offsets_tokens": model.phase2_delay_offsets.detach().cpu().tolist(),
            "min_delay_ms": float(config.model.phase2_min_delay_ms),
            "max_delay_ms": float(config.model.phase2_max_delay_ms),
            "delay_prior_ms": float(config.model.phase2_delay_prior_ms),
            "variance_weight": float(config.model.phase2_variance_weight),
            "covariance_weight": float(config.model.phase2_covariance_weight),
            "target_std": float(config.model.phase2_target_std),
            "shared_private_enabled": bool(
                config.model.phase2_shared_private_enabled
            ),
            "private_dim": int(config.model.phase2_private_dim),
            "shared_private_hidden": int(
                config.model.phase2_shared_private_hidden
            ),
            "private_loss_weight": float(
                config.model.phase2_private_loss_weight
            ),
            "orthogonality_weight": float(
                config.model.phase2_orthogonality_weight
            ),
            "shared_private_progress": float(
                getattr(model, "phase2_shared_private_progress", 0.0)
            ),
            "transport_start_epoch": int(
                config.train.phase2_transport_start_epoch
            ),
            "transport_ramp_epochs": int(
                config.train.phase2_transport_ramp_epochs
            ),
            "shared_private_start_epoch": int(
                config.train.phase2_shared_private_start_epoch
            ),
            "shared_private_ramp_epochs": int(
                config.train.phase2_shared_private_ramp_epochs
            ),
        },
    }


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


_SHARED_PRIVATE_PREFIXES = (
    "ecg_shared_private.",
    "ppg_shared_private.",
    "ecg_teacher_shared_private.",
    "ppg_teacher_shared_private.",
    "ecg_private_predictor.",
    "ppg_private_predictor.",
)


def _checkpoint_uses_shared_private(checkpoint: dict) -> bool:
    phase2_config = checkpoint.get("phase2_config", {})
    if "shared_private_enabled" in phase2_config:
        return bool(phase2_config["shared_private_enabled"])
    state_dict = checkpoint.get("model_state_dict", {})
    return any(
        key.startswith(_SHARED_PRIVATE_PREFIXES) for key in state_dict
    )


def _initialize_shared_private_from_phase2(model, checkpoint: dict) -> list:
    """Load a legacy Phase 2 model and leave only new P2 modules initialized."""
    if model.pretrain_phase != 2 or not model.phase2_shared_private_enabled:
        raise ValueError(
            "--init_checkpoint requires Phase 2 with --shared_private"
        )
    if int(checkpoint.get("pretrain_phase", 0)) != 2:
        raise ValueError("Shared-private initialization requires a Phase 2 checkpoint")
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Initialization checkpoint must contain model_state_dict"
        )
    checkpoint_phase2 = checkpoint.get("phase2_config") or {}
    checkpoint_transport = bool(
        checkpoint_phase2.get("transport_enabled", True)
    )
    if checkpoint_transport != bool(model.phase2_transport_enabled):
        raise ValueError(
            "Initialization checkpoint transport enabled/disabled state "
            "does not match the requested Shared-Private run"
        )
    checkpoint_mode = str(
        checkpoint_phase2.get("transport_mode", "full")
    )
    if (
        model.phase2_transport_enabled
        and checkpoint_mode != model.phase2_transport_mode
    ):
        raise ValueError(
            "Initialization checkpoint Transport constraint mode "
            f"{checkpoint_mode!r} does not match "
            f"{model.phase2_transport_mode!r}"
        )

    result = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    unexpected_missing = [
        key for key in result.missing_keys
        if not key.startswith(_SHARED_PRIVATE_PREFIXES)
    ]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            "Phase 2 initialization structure mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if not result.missing_keys:
        print(
            "[Init] Source already contains shared-private modules; "
            "optimizer and validation state will still restart"
        )
    model._enforce_teacher_eval()
    return list(result.missing_keys)


def train(
    config: Config,
    resume_from: str = None,
    start_epoch: int = 0,
    init_from: str = None,
):
    if resume_from is not None and init_from is not None:
        raise ValueError("Use only one of resume_from and init_from")
    seed_everything(
        config.seed,
        config.deterministic,
        enable_tf32=config.train.enable_tf32,
    )
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | seed={config.seed} | "
        f"data_split_seed={config.pretrain_split_seed} | "
        f"deterministic={config.deterministic}"
    )

    phase = int(config.model.pretrain_phase)
    if phase == 2:
        batch_size = int(config.train.phase2_batch_size)
        accum_steps = max(1, int(config.train.phase2_accum_steps))
        pretrain_lr = float(config.train.phase2_lr)
        warmup_epochs = int(config.train.phase2_warmup_epochs)
        use_amp = bool(config.train.phase2_use_amp and device.type == "cuda")
    elif phase == 1:
        batch_size = int(config.train.phase1_batch_size)
        accum_steps = max(1, int(config.train.phase1_accum_steps))
        pretrain_lr = float(config.train.phase1_lr)
        warmup_epochs = int(config.train.phase1_warmup_epochs)
        use_amp = bool(config.train.phase1_use_amp and device.type == "cuda")
    else:
        batch_size = int(config.train.pretrain_batch_size)
        accum_steps = max(1, int(config.train.pretrain_accum_steps))
        pretrain_lr = float(config.train.pretrain_lr)
        warmup_epochs = int(config.train.pretrain_warmup_epochs)
        use_amp = False
    print(
        f"[Pretrain] phase={phase} batch={batch_size} accum={accum_steps} "
        f"effective_batch={batch_size * accum_steps} lr={pretrain_lr:.2e} "
        f"amp={use_amp}"
    )
    use_stats_targets = (
        config.model.use_stats_loss
        if phase == 0
        else (
            config.model.phase2_use_stats_loss
            if phase == 2
            else config.model.phase1_use_stats_loss
        )
    )

    # Data
    train_loader, val_loader = build_pretrain_dataloaders(
        config.data,
        config.train,
        return_stats=use_stats_targets,
        use_processed=True,
        seed=config.pretrain_split_seed,
        batch_size=batch_size,
    )
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_steps = optimizer_steps_per_epoch * config.train.pretrain_epochs

    model = build_model(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if phase == 2:
        delay_offsets = model.phase2_delay_offsets.detach().cpu().tolist()
        delay_ms = [round(offset * model.phase2_token_ms) for offset in delay_offsets]
        print(
            "[Phase2] transport="
            f"{'on' if config.model.phase2_transport_enabled else 'off'} | "
            f"mode={config.model.phase2_transport_mode} | "
            f"causal delay bins={delay_offsets} tokens ({delay_ms} ms) | "
            f"transport_start={config.train.phase2_transport_start_epoch} | "
            f"ramp={config.train.phase2_transport_ramp_epochs} epochs"
        )
        print(
            "[Phase2] effective constraint weights="
            f"{model._phase2_effective_regularizer_weights()}"
        )
        if config.model.phase2_shared_private_enabled:
            print(
                "[Priority2] Shared-Private JEPA enabled | "
                f"private_dim={config.model.phase2_private_dim} | "
                f"private_weight={config.model.phase2_private_loss_weight:.3f} | "
                f"orthogonality_weight="
                f"{config.model.phase2_orthogonality_weight:.3f} | "
                f"start={config.train.phase2_shared_private_start_epoch} | "
                f"ramp={config.train.phase2_shared_private_ramp_epochs}"
            )

    if init_from is not None:
        print(f"[Init] Loading Phase 2 initialization checkpoint: {init_from}")
        init_checkpoint = torch.load(
            init_from, map_location="cpu", weights_only=False
        )
        initialized_keys = _initialize_shared_private_from_phase2(
            model, init_checkpoint
        )
        print(
            "[Init] Loaded existing Phase 2 weights; initialized "
            f"{len(initialized_keys)} new shared-private tensors"
        )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
    )
    optimizer = AdamW(
        trainable_params,
        lr=pretrain_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.pretrain_weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Resume logic ──
    resume_best_loss = float("inf")
    early_stop_best_loss = float("inf")
    early_stop_bad_epochs = 0
    resume_optimizer_step = None
    if resume_from is not None:
        print(f"[Resume] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        checkpoint_phase = int(ckpt.get("pretrain_phase", 0))
        if checkpoint_phase != phase:
            raise ValueError(
                f"Cannot resume Phase {phase} from a Phase {checkpoint_phase} "
                "checkpoint. Start from scratch or use a matching-phase checkpoint."
            )
        checkpoint_shared_private = _checkpoint_uses_shared_private(ckpt)
        if checkpoint_shared_private != bool(
            config.model.phase2_shared_private_enabled
        ):
            raise ValueError(
                "Resume checkpoint shared-private mode does not match the model. "
                "Use --init_checkpoint to initialize Shared-Private JEPA from "
                "a standard Phase 2 checkpoint."
            )
        checkpoint_transport = bool(
            (ckpt.get("phase2_config") or {}).get(
                "transport_enabled", True
            )
        )
        if (
            phase == 2
            and checkpoint_transport
            != bool(config.model.phase2_transport_enabled)
        ):
            raise ValueError(
                "Resume checkpoint transport mode does not match the model. "
                "Resume transport-on and transport-off runs separately."
            )
        checkpoint_transport_mode = str(
            (ckpt.get("phase2_config") or {}).get(
                "transport_mode", "full"
            )
        )
        if (
            phase == 2
            and checkpoint_transport
            and checkpoint_transport_mode
            != config.model.phase2_transport_mode
        ):
            raise ValueError(
                "Resume checkpoint Transport constraint mode does not match "
                f"the model ({checkpoint_transport_mode!r} != "
                f"{config.model.phase2_transport_mode!r})."
            )
        # Load encoder weights
        if "context_encoder" in ckpt:
            model.context_encoder.load_state_dict(ckpt["context_encoder"])
            print("[Resume] Loaded context_encoder weights")
        if phase >= 1 and "ppg_encoder" in ckpt:
            model.ppg_encoder.load_state_dict(ckpt["ppg_encoder"])
            print("[Resume] Loaded ppg_encoder weights")
        if "target_encoder" in ckpt:
            model.target_encoder.load_state_dict(ckpt["target_encoder"])
            print("[Resume] Loaded target_encoder weights")
        if phase >= 1 and "context_teacher" in ckpt:
            model.context_teacher.load_state_dict(ckpt["context_teacher"])
            print("[Resume] Loaded context_teacher weights")
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
        if ckpt.get("best_checkpoint_eligible", False):
            resume_best_loss = float(
                ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            )
        else:
            print(
                "[Resume] Reset best validation loss because this checkpoint "
                "predates collapse-aware selection"
            )
        early_stop_best_loss = float(
            ckpt.get("early_stop_best_loss", resume_best_loss)
        )
        early_stop_bad_epochs = int(ckpt.get("early_stop_bad_epochs", 0))
        if ckpt.get("optimizer_step") is not None:
            resume_optimizer_step = int(ckpt["optimizer_step"])
        model._enforce_teacher_eval()
        print(f"[Resume] Continuing from epoch {start_epoch}")

    # LR schedule: warmup + cosine (adjusted for resume)
    remaining_epochs = config.train.pretrain_epochs - start_epoch
    remaining_steps = remaining_epochs * optimizer_steps_per_epoch
    optimizer_step = (
        resume_optimizer_step
        if resume_optimizer_step is not None
        else start_epoch * optimizer_steps_per_epoch
    )
    if resume_from is not None:
        full_warmup_steps = warmup_epochs * optimizer_steps_per_epoch
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = pretrain_lr
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _global_pretrain_lr_factor(
                step,
                full_warmup_steps,
                total_steps,
            ),
            last_epoch=optimizer_step - 1,
        )
        print(
            f"[Resume] Restored global LR schedule at optimizer_step="
            f"{optimizer_step}/{total_steps} | lr="
            f"{scheduler.get_last_lr()[0]:.2e}"
        )
    elif start_epoch >= warmup_epochs:
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
            warmup_epochs - start_epoch
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
        "pretrain_phase": phase,
        "shared_private_enabled": bool(
            config.model.phase2_shared_private_enabled
        ),
        "seed": config.seed,
        "optimization_seed": config.seed,
        "data_split_seed": config.pretrain_split_seed,
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

    for epoch in range(start_epoch, config.train.pretrain_epochs):
        if phase == 2:
            transport_progress = (
                phase2_transport_progress(
                    epoch,
                    config.train.phase2_transport_start_epoch,
                    config.train.phase2_transport_ramp_epochs,
                )
                if config.model.phase2_transport_enabled
                else 0.0
            )
            model.set_phase2_progress(transport_progress)
            if config.model.phase2_shared_private_enabled:
                shared_private_progress = phase2_transport_progress(
                    epoch,
                    config.train.phase2_shared_private_start_epoch,
                    config.train.phase2_shared_private_ramp_epochs,
                )
                model.set_shared_private_progress(shared_private_progress)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_losses = defaultdict(float)
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(train_loader):
            ecg, ppg, ecg_stats = _move_pretrain_batch(batch_data, device)

            # ★ MixUp: 随机混合batch内样本 → 正则化
            if (
                phase == 0
                and config.train.use_mixup
                and config.train.mixup_alpha > 0
            ):
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

            # Forward + loss. Phase 1 uses AMP because it runs four encoders.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss, info = model.compute_loss(ecg, ppg, ecg_stats)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_idx}. "
                    "Check the preprocessed ECG/PPG tensors and ecg_stats."
                )
            _accumulate_metrics(epoch_losses, info)

            group_start = (batch_idx // accum_steps) * accum_steps
            group_size = min(accum_steps, steps_per_epoch - group_start)
            scaler.scale(loss / group_size).backward()

            should_step = (
                (batch_idx + 1) % accum_steps == 0
                or (batch_idx + 1) == steps_per_epoch
            )
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0, error_if_nonfinite=False
                )
                if not torch.isfinite(grad_norm):
                    if use_amp:
                        scale_before = scaler.get_scale()
                        # GradScaler recorded the overflow in unscale_ and
                        # therefore skips this optimizer step. Updating it is
                        # essential: the next batch retries with a lower scale.
                        scaler.step(optimizer)
                        scaler.update()
                        scale_after = scaler.get_scale()
                        optimizer.zero_grad(set_to_none=True)
                        print(
                            f"[AMPOverflow] epoch={epoch} batch={batch_idx}; "
                            f"optimizer step skipped, scale "
                            f"{scale_before:.0f}->{scale_after:.0f}"
                        )
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, "
                        f"batch={batch_idx}; optimizer step was skipped."
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.update_target_encoder(ema_momentum)
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if batch_idx % 50 == 0:
                phase_detail = ""
                if phase >= 1:
                    phase_detail = (
                        f" E2P: {_metric(info, 'ecg_to_ppg_token'):.5f} |"
                        f" P2E: {_metric(info, 'ppg_to_ecg_token'):.5f} |"
                        f" Mask: {_metric(info, 'masked_fraction'):.3f} |"
                    )
                if phase == 2:
                    phase_detail += (
                        f" Tr: {_metric(info, 'phase2_progress'):.2f} |"
                        f" Delay: {_metric(info, 'delay_mean_ms'):.0f}ms |"
                        f" Mass: {_metric(info, 'matched_mass'):.3f} |"
                        f" MinMass: {_metric(info, 'minimum_matched_mass'):.2e} |"
                    )
                    if config.model.phase2_shared_private_enabled:
                        phase_detail += (
                            f" SP: {_metric(info, 'shared_private_progress'):.2f} |"
                            f" Priv: {_metric(info, 'private_reconstruction'):.5f} |"
                            f" Orth: {_metric(info, 'shared_private_orthogonality'):.5f} |"
                        )
                log_msg = (
                    f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.6f} | JEPA: {_metric(info, 'jepa'):.5f} | "
                    f"{phase_detail} "
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
                model,
                val_loader,
                device,
                seed=config.seed + 10_000,
                use_amp=use_amp,
            )

        summary = (
            f"Epoch {epoch:3d} | Phase {phase} | "
            f"Train total={epoch_loss:.6f} jepa={_metric(train_metrics, 'jepa'):.6f} "
            f"stats={_metric(train_metrics, 'stats'):.6f} "
            f"token={_metric(train_metrics, 'token_align'):.6f} "
            f"ctx_std={_metric(train_metrics, 'context_std'):.4f} "
            f"tgt_std={_metric(train_metrics, 'target_std'):.4f}"
        )
        if phase >= 1:
            summary += (
                f" e2p={_metric(train_metrics, 'ecg_to_ppg_token'):.6f} "
                f"p2e={_metric(train_metrics, 'ppg_to_ecg_token'):.6f} "
                f"mask={_metric(train_metrics, 'masked_fraction'):.3f}"
            )
        if phase == 2:
            summary += (
                f" direct={_metric(train_metrics, 'direct_token_jepa'):.6f} "
                f"transport={_metric(train_metrics, 'transport_token_jepa'):.6f} "
                f"tr_progress={_metric(train_metrics, 'phase2_progress'):.3f} "
                f"delay={_metric(train_metrics, 'delay_mean_ms'):.1f}ms "
                f"delay_std={_metric(train_metrics, 'delay_std_ms'):.1f}ms "
                f"mono={_metric(train_metrics, 'monotonic'):.6f} "
                f"mass={_metric(train_metrics, 'matched_mass'):.3f} "
                f"min_mass={_metric(train_metrics, 'minimum_matched_mass'):.2e}"
            )
            if config.model.phase2_shared_private_enabled:
                summary += (
                    f" sp_progress="
                    f"{_metric(train_metrics, 'shared_private_progress'):.3f} "
                    f"private="
                    f"{_metric(train_metrics, 'private_reconstruction'):.6f} "
                    f"orth="
                    f"{_metric(train_metrics, 'shared_private_orthogonality'):.6f} "
                    f"ppg_private_std="
                    f"{_metric(train_metrics, 'ppg_private_std'):.4f}"
                )
        if val_metrics is not None:
            summary += (
                f" | Val total={_metric(val_metrics, 'total_loss'):.6f} "
                f"jepa={_metric(val_metrics, 'jepa'):.6f} "
                f"stats={_metric(val_metrics, 'stats'):.6f} "
                f"token={_metric(val_metrics, 'token_align'):.6f} "
                f"ctx_std={_metric(val_metrics, 'context_std'):.4f} "
                f"tgt_std={_metric(val_metrics, 'target_std'):.4f} "
                f"ctx_collapse={_metric(val_metrics, 'context_collapsed_fraction'):.3f} "
                f"tgt_collapse={_metric(val_metrics, 'target_collapsed_fraction'):.3f} "
                f"ctx_cov={_metric(val_metrics, 'context_cov_offdiag_rms'):.4f} "
                f"tgt_cov={_metric(val_metrics, 'target_cov_offdiag_rms'):.4f} "
                f"teacher_cos={_metric(val_metrics, 'teacher_student_cosine'):.6f}"
            )
            if phase >= 1:
                summary += (
                    f" val_e2p={_metric(val_metrics, 'ecg_to_ppg_token'):.6f} "
                    f"val_p2e={_metric(val_metrics, 'ppg_to_ecg_token'):.6f} "
                    f"val_mask={_metric(val_metrics, 'masked_fraction'):.3f}"
                )
            if phase == 2:
                summary += (
                    f" val_transport={_metric(val_metrics, 'transport_token_jepa'):.6f} "
                    f"val_delay={_metric(val_metrics, 'delay_mean_ms'):.1f}ms "
                    f"val_mono={_metric(val_metrics, 'monotonic'):.6f} "
                    f"val_mass={_metric(val_metrics, 'matched_mass'):.3f} "
                    f"val_min_mass={_metric(val_metrics, 'minimum_matched_mass'):.2e}"
                )
                if config.model.phase2_shared_private_enabled:
                    summary += (
                        f" val_private="
                        f"{_metric(val_metrics, 'private_reconstruction'):.6f} "
                        f"val_orth="
                        f"{_metric(val_metrics, 'shared_private_orthogonality'):.6f} "
                        f"val_ppg_private_std="
                        f"{_metric(val_metrics, 'ppg_private_std'):.4f} "
                        f"val_ppg_private_collapse="
                        f"{_metric(val_metrics, 'ppg_private_collapsed_fraction'):.3f}"
                    )
            if (
                _metric(val_metrics, "context_collapsed_fraction") > 0.90
                or _metric(val_metrics, "target_collapsed_fraction") > 0.90
                or _metric(
                    val_metrics, "ecg_private_collapsed_fraction"
                ) > 0.90
                or _metric(
                    val_metrics, "ppg_private_collapsed_fraction"
                ) > 0.90
            ):
                print("[CollapseWarning] More than 90% of embedding dimensions have near-zero variance")
        summary += f" | Time: {epoch_time:.1f}s"
        samples_per_second = steps_per_epoch * batch_size / max(epoch_time, 1e-6)
        summary += f" | Throughput={samples_per_second:.1f} samples/s"
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(device).total_memory
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            summary += (
                f" peak_alloc={peak_allocated / 2**30:.2f}GB"
                f"({100.0 * peak_allocated / total_vram:.1f}%)"
                f" peak_reserved={peak_reserved / 2**30:.2f}GB"
                f"({100.0 * peak_reserved / total_vram:.1f}%)"
            )
        print(summary)
        print("-" * 60)

        with open(log_file, "a") as f:
            f.write(summary + "\n")

        # Save by held-out loss only after ruling out a collapsed representation.
        current_val_loss = (
            _metric(val_metrics, "total_loss") if val_metrics is not None else None
        )
        transport_progress = (
            float(model.phase2_progress) if phase == 2 else 1.0
        )
        shared_private_progress = (
            float(model.phase2_shared_private_progress)
            if phase == 2 and config.model.phase2_shared_private_enabled
            else 1.0
        )
        checkpoint_eligible = _checkpoint_is_eligible(
            val_metrics,
            phase,
            transport_progress,
            shared_private_progress,
            transport_required=config.model.phase2_transport_enabled,
        )
        should_early_stop = False
        if val_metrics is not None and not checkpoint_eligible:
            if phase == 2 and (
                (
                    config.model.phase2_transport_enabled
                    and transport_progress < 1.0 - 1e-8
                )
                or shared_private_progress < 1.0 - 1e-8
            ):
                print(
                    "[CheckpointSkip] Phase 2 objectives are still ramping; "
                    "best-model comparison starts when all objectives are active"
                )
            else:
                print(
                    "[CheckpointSkip] Validation representation is collapsed; "
                    "not eligible for jepa_best.pt"
                )
        if (
            phase == 2
            and current_val_loss is not None
            and checkpoint_eligible
            and config.train.phase2_early_stop_patience > 0
        ):
            (
                early_stop_best_loss,
                early_stop_bad_epochs,
                meaningful_improvement,
            ) = _early_stopping_step(
                early_stop_best_loss,
                early_stop_bad_epochs,
                current_val_loss,
                config.train.phase2_early_stop_min_delta,
            )
            if meaningful_improvement:
                print(
                    "[EarlyStop] Meaningful validation improvement: "
                    f"best={early_stop_best_loss:.6f} "
                    f"min_delta={config.train.phase2_early_stop_min_delta:.1e}"
                )
            else:
                print(
                    "[EarlyStop] No meaningful validation improvement: "
                    f"{early_stop_bad_epochs}/"
                    f"{config.train.phase2_early_stop_patience} "
                    f"(best={early_stop_best_loss:.6f}, "
                    f"current={current_val_loss:.6f}, "
                    f"min_delta={config.train.phase2_early_stop_min_delta:.1e})"
                )
            should_early_stop = (
                early_stop_bad_epochs
                >= config.train.phase2_early_stop_patience
            )
        if (
            current_val_loss is not None
            and checkpoint_eligible
            and current_val_loss < best_loss
        ):
            best_loss = current_val_loss
            checkpoint_path = os.path.join(config.output_dir, "jepa_best.pt")
            _save_checkpoint(
                {
                    "epoch": epoch,
                    "pretrain_phase": phase,
                    "model_state_dict": model.state_dict(),
                    **_encoder_checkpoint_payload(model),
                    **_phase_checkpoint_metadata(model, config),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_step": optimizer_step,
                    "loss": epoch_loss,
                    "train_metrics": train_metrics,
                    "val_loss": current_val_loss,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_loss,
                    "best_checkpoint_eligible": True,
                    "early_stop_best_loss": early_stop_best_loss,
                    "early_stop_bad_epochs": early_stop_bad_epochs,
                    "seed": config.seed,
                    "data_split_seed": config.pretrain_split_seed,
                    "train_segments": len(train_loader.dataset),
                    "val_segments": len(val_loader.dataset),
                },
                checkpoint_path,
            )
            print(f"Saved best validation model to {checkpoint_path}")

        # Always keep a resumable non-corrupt checkpoint, independent of best selection.
        last_path = os.path.join(config.output_dir, "jepa_last.pt")
        _save_checkpoint(
            {
                "epoch": epoch,
                "pretrain_phase": phase,
                "model_state_dict": model.state_dict(),
                **_encoder_checkpoint_payload(model),
                **_phase_checkpoint_metadata(model, config),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_step": optimizer_step,
                "loss": epoch_loss,
                "train_metrics": train_metrics,
                "val_loss": current_val_loss,
                "val_metrics": val_metrics,
                "best_val_loss": best_loss,
                "best_checkpoint_eligible": checkpoint_eligible,
                "early_stop_best_loss": early_stop_best_loss,
                "early_stop_bad_epochs": early_stop_bad_epochs,
                "seed": config.seed,
                "data_split_seed": config.pretrain_split_seed,
            },
            last_path,
        )

        # Save periodic checkpoints only when explicitly enabled.
        checkpoint_interval = int(
            config.train.pretrain_checkpoint_interval
        )
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f"jepa_epoch_{epoch+1}.pt")
            _save_checkpoint(
                {
                    "epoch": epoch,
                    "pretrain_phase": phase,
                    "model_state_dict": model.state_dict(),
                    **_encoder_checkpoint_payload(model),
                    **_phase_checkpoint_metadata(model, config),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_step": optimizer_step,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_loss,
                    "best_checkpoint_eligible": checkpoint_eligible,
                    "early_stop_best_loss": early_stop_best_loss,
                    "early_stop_bad_epochs": early_stop_bad_epochs,
                    "seed": config.seed,
                    "data_split_seed": config.pretrain_split_seed,
                },
                ckpt_path,
            )

        if should_early_stop:
            stop_message = (
                "[EarlyStop] Phase 2 stopped after "
                f"{early_stop_bad_epochs} consecutive eligible validation epochs "
                "without a meaningful loss decrease. "
                f"Best={early_stop_best_loss:.6f}, "
                f"min_delta={config.train.phase2_early_stop_min_delta:.1e}. "
                f"Last checkpoint: {last_path}"
            )
            print(stop_message)
            with open(log_file, "a") as f:
                f.write(stop_message + "\n")
            break

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

    # This legacy continuation belongs to the Phase 0 baseline only.
    config.model.pretrain_phase = 0
    model = build_model(config.model).to(device)

    # 加载完整 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(ckpt.get("pretrain_phase", 0)) != 0:
        raise ValueError("--token_align only accepts a Phase 0 checkpoint")
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
    parser.add_argument(
        "--init_checkpoint", type=str, default=None,
        help=(
            "Initialize new Shared-Private modules from an existing Phase 2 "
            "checkpoint while restarting optimizer and validation state"
        ),
    )
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to start/resume from")
    parser.add_argument(
        "--phase", type=int, choices=[0, 1, 2], default=None,
        help="Pre-training stage; default comes from config.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Checkpoint/log directory (Phase 1/2 get phase-specific defaults)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the total number of pre-training epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the experiment seed",
    )
    parser.add_argument(
        "--data_split_seed", type=int, default=None,
        help=(
            "Seed for the patient-grouped pre-training train/validation split; "
            "kept separate from model/optimizer randomness"
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override per-GPU batch size for the selected phase",
    )
    parser.add_argument(
        "--accum_steps", type=int, default=None,
        help="Override gradient accumulation steps for the selected phase",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override peak learning rate for the selected phase",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override pre-training DataLoader worker count",
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=None,
        help="Batches prefetched by each DataLoader worker",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=None,
        help="Periodic checkpoint interval in epochs; 0 disables periodic saves",
    )
    parser.add_argument(
        "--performance_mode", action="store_true",
        help="Use benchmarked CUDA kernels instead of deterministic kernels",
    )
    parser.add_argument(
        "--transport_start_epoch", type=int, default=None,
        help="Phase 2 epoch where transport blending starts",
    )
    parser.add_argument(
        "--disable_transport", action="store_true",
        help=(
            "Phase 2 ablation: train with direct bidirectional masked-token "
            "prediction while disabling causal transport and delay losses"
        ),
    )
    parser.add_argument(
        "--transport_mode",
        choices=(
            "full",
            "static_delay",
            "fixed_prior",
            "zero_delay",
            "no_monotonic",
            "token_shuffled",
        ),
        default=None,
        help=(
            "Phase 2 Transport constraint-composition ablation. "
            "'full' is the paper model."
        ),
    )
    parser.add_argument(
        "--transport_ramp_epochs", type=int, default=None,
        help="Phase 2 epochs used to ramp transport from 0 to 1",
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=None,
        help="Phase 2 eligible validation epochs without meaningful improvement",
    )
    parser.add_argument(
        "--early_stop_min_delta", type=float, default=None,
        help="Minimum absolute Phase 2 validation-loss decrease",
    )
    parser.add_argument(
        "--shared_private", action="store_true",
        help=(
            "Enable Priority-2 Shared-Private JEPA in Phase 2; causal "
            "transport is applied only to shared tokens"
        ),
    )
    parser.add_argument(
        "--private_dim", type=int, default=None,
        help="Modality-private token dimension",
    )
    parser.add_argument(
        "--private_loss_weight", type=float, default=None,
        help="Weight of same-modality private masked prediction",
    )
    parser.add_argument(
        "--orthogonality_weight", type=float, default=None,
        help="Weight of shared-private cross-correlation penalty",
    )
    parser.add_argument(
        "--shared_private_start_epoch", type=int, default=None,
        help="Epoch where Shared-Private auxiliary losses begin",
    )
    parser.add_argument(
        "--shared_private_ramp_epochs", type=int, default=None,
        help="Epochs used to ramp Shared-Private auxiliary losses to full weight",
    )
    parser.add_argument("--token_align", type=str, default=None,
                        help="Token 对齐续训练: --token_align outputs/jepa_best.pt")
    args = parser.parse_args()

    config = Config()
    if args.phase is not None:
        config.model.pretrain_phase = args.phase
    if args.shared_private:
        config.model.phase2_shared_private_enabled = True
    if args.disable_transport:
        config.model.phase2_transport_enabled = False
    if args.transport_mode is not None:
        config.model.phase2_transport_mode = args.transport_mode
    if args.private_dim is not None:
        config.model.phase2_private_dim = args.private_dim
    if args.private_loss_weight is not None:
        config.model.phase2_private_loss_weight = args.private_loss_weight
    if args.orthogonality_weight is not None:
        config.model.phase2_orthogonality_weight = args.orthogonality_weight
    if args.seed is not None:
        config.seed = args.seed
    if args.data_split_seed is not None:
        config.pretrain_split_seed = args.data_split_seed
    if args.epochs is not None:
        config.train.pretrain_epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    elif config.model.pretrain_phase == 1:
        config.output_dir = config.output_dir.rstrip("/\\") + "_phase1"
    elif config.model.pretrain_phase == 2:
        suffix = (
            "_phase2_shared_private"
            if config.model.phase2_shared_private_enabled
            else "_phase2"
        )
        if not config.model.phase2_transport_enabled:
            suffix += "_no_transport"
        elif config.model.phase2_transport_mode != "full":
            suffix += f"_{config.model.phase2_transport_mode}"
        config.output_dir = config.output_dir.rstrip("/\\") + suffix

    if args.performance_mode:
        config.deterministic = False
    if args.workers is not None:
        config.train.pretrain_dataloader_workers = args.workers
    if args.prefetch_factor is not None:
        config.train.pretrain_prefetch_factor = args.prefetch_factor
    if args.checkpoint_interval is not None:
        config.train.pretrain_checkpoint_interval = args.checkpoint_interval
    if config.train.pretrain_checkpoint_interval < 0:
        parser.error("--checkpoint_interval must be >= 0")

    if config.model.pretrain_phase == 2:
        if args.batch_size is not None:
            config.train.phase2_batch_size = args.batch_size
        if args.accum_steps is not None:
            config.train.phase2_accum_steps = args.accum_steps
        if args.lr is not None:
            config.train.phase2_lr = args.lr
        if args.transport_start_epoch is not None:
            config.train.phase2_transport_start_epoch = args.transport_start_epoch
        if args.transport_ramp_epochs is not None:
            config.train.phase2_transport_ramp_epochs = args.transport_ramp_epochs
        if args.early_stop_patience is not None:
            config.train.phase2_early_stop_patience = args.early_stop_patience
        if args.early_stop_min_delta is not None:
            config.train.phase2_early_stop_min_delta = args.early_stop_min_delta
        if args.shared_private_start_epoch is not None:
            config.train.phase2_shared_private_start_epoch = (
                args.shared_private_start_epoch
            )
        if args.shared_private_ramp_epochs is not None:
            config.train.phase2_shared_private_ramp_epochs = (
                args.shared_private_ramp_epochs
            )
        if config.train.phase2_transport_start_epoch < 0:
            parser.error("--transport_start_epoch must be >= 0")
        if config.train.phase2_transport_ramp_epochs < 1:
            parser.error("--transport_ramp_epochs must be >= 1")
        if config.train.phase2_early_stop_patience < 0:
            parser.error("--early_stop_patience must be >= 0")
        if config.train.phase2_early_stop_min_delta < 0:
            parser.error("--early_stop_min_delta must be >= 0")
        if config.train.phase2_shared_private_start_epoch < 0:
            parser.error("--shared_private_start_epoch must be >= 0")
        if config.train.phase2_shared_private_ramp_epochs < 1:
            parser.error("--shared_private_ramp_epochs must be >= 1")
        if config.model.phase2_private_dim < 1:
            parser.error("--private_dim must be >= 1")
        if config.model.phase2_private_loss_weight < 0:
            parser.error("--private_loss_weight must be >= 0")
        if config.model.phase2_orthogonality_weight < 0:
            parser.error("--orthogonality_weight must be >= 0")
    elif config.model.pretrain_phase == 1:
        if args.batch_size is not None:
            config.train.phase1_batch_size = args.batch_size
        if args.accum_steps is not None:
            config.train.phase1_accum_steps = args.accum_steps
        if args.lr is not None:
            config.train.phase1_lr = args.lr
    else:
        if args.batch_size is not None:
            config.train.pretrain_batch_size = args.batch_size
        if args.accum_steps is not None:
            config.train.pretrain_accum_steps = args.accum_steps
        if args.lr is not None:
            config.train.pretrain_lr = args.lr

    if args.shared_private and config.model.pretrain_phase != 2:
        parser.error("--shared_private requires --phase 2")
    if args.disable_transport and config.model.pretrain_phase != 2:
        parser.error("--disable_transport requires --phase 2")
    if args.transport_mode is not None and config.model.pretrain_phase != 2:
        parser.error("--transport_mode requires --phase 2")
    if args.disable_transport and args.transport_mode not in (None, "full"):
        parser.error(
            "--disable_transport cannot be combined with a non-full "
            "--transport_mode"
        )
    if args.init_checkpoint is not None and not args.shared_private:
        parser.error("--init_checkpoint requires --shared_private")
    if args.init_checkpoint is not None and args.resume is not None:
        parser.error("--init_checkpoint cannot be combined with --resume")
    if args.init_checkpoint is not None and args.start_epoch != 0:
        parser.error("--init_checkpoint starts a new run; keep --start_epoch 0")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.token_align is not None:
        # ★ Token 对齐续训练模式
        train_token_align(config, args.token_align)
    else:
        # 正常预训练
        train(
            config,
            resume_from=args.resume,
            start_epoch=args.start_epoch,
            init_from=args.init_checkpoint,
        )
