"""Phase 3A: continue Phase 2 JEPA with guarded downstream feedback."""
import argparse
import math
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler

from config import Config
from dataset.data import MultiDiseasePatientMILDataset
from models.classifier import DualStreamPatientMILClassifier
from models.jepa import cosine_schedule
from models.losses import build_criterion
from train_downstream import (
    compute_best_metric,
    compute_multidisease_objective,
    compute_multilabel_pos_weight,
    evaluate_multilabel,
    get_focus_auc,
    load_taskaware_multidisease_split_manifest,
    seed_dataloader_worker,
)
from train_pretrain import (
    _encoder_checkpoint_payload,
    _load_jepa_state_dict,
    _move_pretrain_batch,
    _phase_checkpoint_metadata,
    _representation_is_healthy,
    _save_checkpoint,
    build_model,
    build_pretrain_dataloaders,
    evaluate_pretrain,
    phase2_transport_progress,
    seed_everything,
)


class FocusBalancedBatchSampler(Sampler[List[int]]):
    """Create patient batches containing both focus-positive and negative cases."""

    def __init__(
        self,
        targets: Sequence[int],
        batch_size: int,
        positive_fraction: float = 0.25,
        seed: int = 42,
    ):
        targets = np.asarray(targets, dtype=np.int64)
        self.positive = np.flatnonzero(targets == 1)
        self.negative = np.flatnonzero(targets == 0)
        if not self.positive.size or not self.negative.size:
            raise ValueError("Focus-balanced sampling requires positive and negative patients")
        self.batch_size = max(2, int(batch_size))
        self.num_positive = min(
            self.batch_size - 1,
            max(1, int(round(self.batch_size * positive_fraction))),
        )
        self.num_negative = self.batch_size - self.num_positive
        self.seed = int(seed)
        self.epoch = 0
        self.num_batches = math.ceil(targets.size / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            positives = rng.choice(
                self.positive,
                self.num_positive,
                replace=self.positive.size < self.num_positive,
            )
            negatives = rng.choice(
                self.negative,
                self.num_negative,
                replace=self.negative.size < self.num_negative,
            )
            batch = np.concatenate([positives, negatives])
            rng.shuffle(batch)
            yield batch.tolist()


def _available_multidisease_files(data_dir: str) -> List[str]:
    return sorted(
        name
        for name in os.listdir(data_dir)
        if name.endswith(".pkl") and name.startswith(("train_", "test_"))
    )


def _make_patient_dataset(config: Config, files: List[str], train: bool):
    return MultiDiseasePatientMILDataset(
        data_dir=config.data.multidisease_dir,
        split="train" if train else "test",
        disease_labels=config.data.multidisease_labels,
        normalize=config.data.normalize,
        normalize_clip=config.data.normalize_clip,
        channel="both",
        target_length=config.data.signal_align_to or None,
        max_segments=config.train.taskaware_feedback_segments,
        files=files,
        train=train,
    )


def _patient_targets(dataset, focus_index: int) -> np.ndarray:
    targets = []
    label_name = dataset.disease_labels[focus_index]
    for filename in dataset.files:
        with open(os.path.join(dataset.data_dir, filename), "rb") as handle:
            sample = pickle.load(handle)
        targets.append(int(bool(sample.get("label", {}).get(label_name, 0))))
    return np.asarray(targets, dtype=np.int64)


def build_feedback_dataloaders(config: Config, split_file: str):
    available = _available_multidisease_files(config.data.multidisease_dir)
    train_files, meta_files, val_files, test_files, resolved = (
        load_taskaware_multidisease_split_manifest(
            split_file, config.data.multidisease_dir, available
        )
    )
    feedback_train = _make_patient_dataset(config, train_files, train=True)
    feedback_meta = _make_patient_dataset(config, meta_files, train=False)
    validation = _make_patient_dataset(config, val_files, train=False)

    focus_targets = _patient_targets(feedback_train, config.train.chd_label_index)
    batch_sampler = FocusBalancedBatchSampler(
        focus_targets,
        config.train.taskaware_feedback_batch_size,
        seed=config.seed,
    )
    workers = max(0, int(config.train.dataloader_workers))
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": seed_dataloader_worker,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = max(
            1, int(config.train.dataloader_prefetch_factor)
        )
    train_loader = DataLoader(
        feedback_train,
        batch_sampler=batch_sampler,
        **common,
    )
    eval_batch_size = max(1, config.train.taskaware_feedback_batch_size)
    meta_loader = DataLoader(
        feedback_meta,
        batch_size=eval_batch_size,
        shuffle=False,
        **common,
    )
    val_loader = DataLoader(
        validation,
        batch_size=eval_batch_size,
        shuffle=False,
        **common,
    )
    print(
        f"[TaskAwareSplit] feedback_train={len(feedback_train)} patients | "
        f"feedback_meta={len(feedback_meta)} | val={len(validation)} | "
        f"test={len({name.split('_')[1] for name in test_files})} sealed patients"
    )
    return train_loader, meta_loader, val_loader, feedback_train, resolved


def _head_state_dict(model) -> Dict[str, torch.Tensor]:
    prefixes = ("ecg_encoder.", "ppg_encoder.")
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(prefixes)
    }


def _load_head_state_dict(model, state_dict):
    result = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key
        for key in result.missing_keys
        if not key.startswith(("ecg_encoder.", "ppg_encoder."))
    ]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            "Feedback-head checkpoint mismatch: "
            f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
        )


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def _scale_gradients(parameters: Iterable[torch.nn.Parameter], scale: float):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(float(scale))


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _feedback_step(
    model,
    feedback_model,
    batch,
    criterion,
    focus_pos_weight,
    pretrain_optimizer,
    head_optimizer,
    scaler,
    device,
    config,
    feedback_step: int,
    pretrain_grad_ema: float,
    ema_momentum: float,
    use_amp: bool,
):
    feedback_model.train()
    warmup = (
        feedback_step < config.train.taskaware_head_warmup_steps
        or config.train.taskaware_feedback_encoder_grad_ratio <= 0.0
    )
    if warmup:
        # Keep normalization buffers fixed while only the feedback head learns.
        feedback_model.ecg_encoder.eval()
        feedback_model.ppg_encoder.eval()
    pretrain_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    signals, labels, *rest = batch
    segment_mask = None
    if len(rest) >= 2 and torch.is_tensor(rest[1]):
        segment_mask = rest[1].to(device, non_blocking=True, dtype=torch.bool)
    signals = torch.nan_to_num(
        signals.to(device, non_blocking=True), nan=0.0, posinf=10.0, neginf=-10.0
    )
    labels = labels.to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(use_amp and device.type == "cuda"),
    ):
        logits = feedback_model(signals, segment_mask=segment_mask)
    loss, components = compute_multidisease_objective(
        logits.float(),
        labels.float(),
        criterion,
        focus_label_index=config.train.chd_label_index,
        focus_loss_weight=config.train.chd_focus_loss_weight,
        focus_pos_weight=focus_pos_weight,
        focus_auc_loss_weight=config.train.chd_auc_loss_weight,
        focus_auc_margin=config.train.chd_auc_margin,
        return_components=True,
    )
    if not torch.isfinite(loss):
        old_scale = float(scaler.get_scale())
        scaler.update(new_scale=max(1.0, old_scale / 2.0))
        pretrain_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        print(
            f"[FeedbackSkip] non-finite loss at step={feedback_step}; "
            f"scale={old_scale:.0f}->{scaler.get_scale():.0f}"
        )
        return {"loss": 0.0, "skipped": 1.0}
    scaler.scale(loss).backward()
    scaler.unscale_(head_optimizer)

    shared_parameters = list(feedback_model.shared_encoder_parameters())
    head_parameters = list(feedback_model.head_parameters())
    encoder_grad_norm = 0.0
    applied_scale = 0.0
    if warmup:
        for parameter in shared_parameters:
            parameter.grad = None
    else:
        scaler.unscale_(pretrain_optimizer)
        encoder_grad_norm = _gradient_norm(shared_parameters)
        reference = max(float(pretrain_grad_ema), 1e-8)
        budget = config.train.taskaware_feedback_encoder_grad_ratio * reference
        applied_scale = min(1.0, budget / max(encoder_grad_norm, 1e-8))
        _scale_gradients(shared_parameters, applied_scale)
    head_grad_norm = torch.nn.utils.clip_grad_norm_(
        head_parameters,
        config.train.taskaware_feedback_grad_clip,
        error_if_nonfinite=False,
    )
    if not warmup:
        encoder_grad_after_scale = torch.nn.utils.clip_grad_norm_(
            shared_parameters,
            config.train.taskaware_feedback_grad_clip,
            error_if_nonfinite=False,
        )
    else:
        encoder_grad_after_scale = torch.zeros((), device=device)
    gradients_finite = bool(
        torch.isfinite(head_grad_norm) and torch.isfinite(encoder_grad_after_scale)
    )
    if not gradients_finite:
        old_scale = float(scaler.get_scale())
        scaler.update(new_scale=max(1.0, old_scale / 2.0))
        pretrain_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        print(
            f"[FeedbackSkip] non-finite gradient at step={feedback_step}; "
            f"scale={old_scale:.0f}->{scaler.get_scale():.0f}"
        )
        return {
            "loss": float(loss.item()),
            "skipped": 1.0,
            "head_warmup": float(warmup),
        }
    if not warmup:
        scaler.step(pretrain_optimizer)
    scaler.step(head_optimizer)
    scaler.update()
    if not warmup:
        model.update_target_encoder(ema_momentum)
    pretrain_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.item()),
        "base": float(components["base"].item()),
        "focus_bce": float(components["focus_bce"].item()),
        "focus_auc": float(components["focus_auc"].item()),
        "encoder_grad_norm": encoder_grad_norm,
        "encoder_grad_scale": applied_scale,
        "head_warmup": float(warmup),
        "skipped": 0.0,
    }


def _evaluate_feedback(model, loader, criterion, device, config, use_amp):
    values = evaluate_multilabel(
        model,
        loader,
        criterion,
        device,
        config.data.multidisease_labels,
        aggregate_by_uid=True,
        use_amp=use_amp,
    )
    return {
        "loss": float(values[0]),
        "accuracy": float(values[1]),
        "macro_auc": float(values[2]),
        "auc_list": [float(value) for value in values[3]],
        "focus_auc": get_focus_auc(values[3], config.train.chd_label_index),
        "precision": float(values[4]),
        "recall": float(values[5]),
        "f1": float(values[6]),
    }


def _checkpoint_payload(
    model,
    feedback_model,
    pretrain_optimizer,
    head_optimizer,
    scheduler,
    scaler,
    config,
    epoch,
    pretrain_step,
    feedback_step,
    best_score,
    pretrain_grad_ema,
    split_file,
    pretrain_metrics,
    feedback_metrics,
    from_scratch,
):
    payload = {
        "taskaware_version": 1,
        "pretrain_phase": 2,
        "epoch": int(epoch),
        "pretrain_step": int(pretrain_step),
        "feedback_step": int(feedback_step),
        "best_taskaware_score": float(best_score),
        "pretrain_grad_ema": float(pretrain_grad_ema),
        "model_state_dict": model.state_dict(),
        "feedback_head_state_dict": _head_state_dict(feedback_model),
        "optimizer_state_dict": pretrain_optimizer.state_dict(),
        "feedback_optimizer_state_dict": head_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "taskaware_split_file": split_file,
        "taskaware_from_scratch": bool(from_scratch),
        "pretrain_val_metrics": pretrain_metrics,
        "feedback_val_metrics": feedback_metrics,
    }
    payload.update(_encoder_checkpoint_payload(model))
    payload.update(_phase_checkpoint_metadata(model, config))
    return payload


def train_taskaware(
    config: Config,
    checkpoint: str,
    split_file: str,
    resume=None,
    from_scratch: bool = False,
):
    config.model.pretrain_phase = 2
    seed_everything(
        config.seed,
        config.deterministic,
        enable_tf32=config.train.enable_tf32,
    )
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.train.phase2_use_amp and device.type == "cuda")
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Device: {device} | Phase 3A task-aware pre-training | AMP={use_amp}")
    source_path = resume or checkpoint
    if source_path and not os.path.isfile(source_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {source_path}")

    pretrain_loader, pretrain_val_loader = build_pretrain_dataloaders(
        config.data,
        config.train,
        return_stats=config.model.phase2_use_stats_loss,
        use_processed=True,
        seed=config.seed,
        batch_size=config.train.phase2_batch_size,
    )
    feedback_loader, meta_loader, val_loader, feedback_dataset, resolved_split = (
        build_feedback_dataloaders(config, split_file)
    )

    model = build_model(config.model).to(device)
    source = None
    if source_path:
        source = torch.load(source_path, map_location=device, weights_only=False)
        if int(source.get("pretrain_phase", -1)) != 2:
            raise ValueError("Task-aware training requires a Phase 2 checkpoint")
        _load_jepa_state_dict(model, source["model_state_dict"])
        print(f"[Init] Loaded Phase 2 checkpoint: {source_path}")
    else:
        if not from_scratch:
            raise ValueError("Provide --checkpoint/--resume or select --from_scratch")
        print("[Init] Random Phase 2 initialization; transport and feedback will warm up")
    continuing_phase2 = source is not None and not bool(
        source.get("taskaware_from_scratch", False)
    )
    if continuing_phase2:
        checkpoint_transport = float(source.get("phase2_transport_progress", 1.0))
        if checkpoint_transport < 1.0 - 1e-8:
            raise ValueError(
                "The Phase 2 checkpoint is still ramping transport; use a full-transport "
                "checkpoint or start with --from_scratch"
            )
    model.set_phase2_progress(1.0 if continuing_phase2 else 0.0)
    model._enforce_teacher_eval()

    feedback_model = DualStreamPatientMILClassifier(
        ecg_encoder=model.context_encoder,
        ppg_encoder=model.ppg_encoder,
        encoder_dim=config.model.transformer_dim,
        num_classes=len(config.data.multidisease_labels),
        use_multiscale=config.data.multidisease_use_multiscale,
        dropout=0.3,
        encoder_chunk_size=config.train.taskaware_feedback_encoder_chunk_size,
        ppg_channel=config.data.multidisease_ppg_channel,
        ecg_channel=config.data.multidisease_ecg_channel,
    ).to(device)
    if resume:
        _load_head_state_dict(feedback_model, source["feedback_head_state_dict"])

    jepa_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    shared_parameters = list(feedback_model.shared_encoder_parameters())
    head_parameters = list(feedback_model.head_parameters())
    pretrain_optimizer = AdamW(
        jepa_parameters,
        lr=config.train.phase2_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.pretrain_weight_decay,
    )
    head_optimizer = AdamW(
        head_parameters,
        lr=config.train.taskaware_head_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=1e-4,
    )
    accum_steps = max(1, config.train.phase2_accum_steps)
    optimizer_steps_per_epoch = math.ceil(len(pretrain_loader) / accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * config.train.taskaware_epochs
    if continuing_phase2:
        scheduler = CosineAnnealingLR(
            pretrain_optimizer,
            T_max=max(1, total_optimizer_steps),
            eta_min=1e-6,
        )
    else:
        warmup_steps = min(
            total_optimizer_steps - 1,
            config.train.phase2_warmup_epochs * optimizer_steps_per_epoch,
        )
        warmup = LinearLR(
            pretrain_optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        cosine = CosineAnnealingLR(
            pretrain_optimizer,
            T_max=max(1, total_optimizer_steps - warmup_steps),
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            pretrain_optimizer,
            schedulers=[warmup, cosine],
            milestones=[max(1, warmup_steps)],
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    pretrain_step = 0
    feedback_step = 0
    best_score = float("-inf")
    pretrain_grad_ema = 0.0
    if resume:
        pretrain_optimizer.load_state_dict(source["optimizer_state_dict"])
        head_optimizer.load_state_dict(source["feedback_optimizer_state_dict"])
        scheduler.load_state_dict(source["scheduler_state_dict"])
        scaler.load_state_dict(source.get("scaler_state_dict", {}))
        start_epoch = int(source["epoch"]) + 1
        pretrain_step = int(source.get("pretrain_step", 0))
        feedback_step = int(source.get("feedback_step", 0))
        best_score = float(source.get("best_taskaware_score", best_score))
        pretrain_grad_ema = float(source.get("pretrain_grad_ema", 0.0))
        print(f"[Resume] epoch={start_epoch} pretrain_step={pretrain_step}")

    pos_weight = compute_multilabel_pos_weight(feedback_dataset, device)
    criterion_pos_weight = None if config.train.multilabel_loss_type == "asl" else pos_weight
    criterion = build_criterion(
        loss_type=config.train.multilabel_loss_type,
        num_classes=len(config.data.multidisease_labels),
        pos_weight=criterion_pos_weight,
        gamma=config.train.focal_gamma,
        gamma_neg=config.train.asl_gamma_neg,
        gamma_pos=config.train.asl_gamma_pos,
        clip=config.train.asl_clip,
        label_smoothing=config.train.label_smoothing,
    )
    focus_pos_weight = pos_weight[config.train.chd_label_index]
    feedback_iterator = iter(feedback_loader)
    log_path = os.path.join(config.output_dir, "taskaware_pretrain_log.txt")
    feedback_start_epoch = (
        0 if continuing_phase2 else config.train.taskaware_feedback_start_epoch
    )
    print(
        f"[Schedule] feedback_start_epoch={feedback_start_epoch} | "
        f"head_warmup_steps={config.train.taskaware_head_warmup_steps} | "
        f"feedback_interval={config.train.taskaware_feedback_interval}"
    )

    for epoch in range(start_epoch, config.train.taskaware_epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if continuing_phase2:
            transport_progress = 1.0
        else:
            transport_progress = phase2_transport_progress(
                epoch,
                config.train.phase2_transport_start_epoch,
                config.train.phase2_transport_ramp_epochs,
            )
        model.set_phase2_progress(transport_progress)
        model.train()
        feedback_model.train()
        model._enforce_teacher_eval()
        pretrain_optimizer.zero_grad(set_to_none=True)
        train_totals = defaultdict(float)
        feedback_totals = defaultdict(float)
        feedback_updates = 0

        for batch_index, batch in enumerate(pretrain_loader):
            ecg, ppg, ecg_stats = _move_pretrain_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss, info, _ = model.compute_loss(
                    ecg,
                    ppg,
                    ecg_stats,
                    return_components=True,
                )
                scaled_loss = loss / accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite pre-training loss at epoch={epoch}, batch={batch_index}"
                )
            scaler.scale(scaled_loss).backward()
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    train_totals[key] += float(value)

            should_step = (
                (batch_index + 1) % accum_steps == 0
                or (batch_index + 1) == len(pretrain_loader)
            )
            if not should_step:
                continue

            scaler.unscale_(pretrain_optimizer)
            shared_grad_norm = _gradient_norm(shared_parameters)
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                jepa_parameters, 1.0, error_if_nonfinite=False
            )
            if not torch.isfinite(raw_grad_norm):
                scaler.step(pretrain_optimizer)
                scaler.update()
                pretrain_optimizer.zero_grad(set_to_none=True)
                print(
                    f"[AMPOverflow] epoch={epoch} batch={batch_index}; "
                    "pre-training optimizer step skipped"
                )
                continue
            global_clip_scale = min(1.0, 1.0 / max(float(raw_grad_norm.item()), 1e-8))
            applied_norm = shared_grad_norm * global_clip_scale
            pretrain_grad_ema = (
                applied_norm
                if pretrain_grad_ema == 0.0
                else 0.95 * pretrain_grad_ema + 0.05 * applied_norm
            )
            scaler.step(pretrain_optimizer)
            scaler.update()
            scheduler.step()
            pretrain_step += 1
            progress = min(
                pretrain_step
                / max(1, optimizer_steps_per_epoch * config.train.taskaware_epochs),
                1.0,
            )
            ema_momentum = cosine_schedule(
                config.model.ema_momentum,
                config.model.ema_end_momentum,
                progress,
            )
            model.update_target_encoder(ema_momentum)
            pretrain_optimizer.zero_grad(set_to_none=True)

            if (
                epoch >= feedback_start_epoch
                and pretrain_step % config.train.taskaware_feedback_interval == 0
            ):
                feedback_batch, feedback_iterator = _next_batch(
                    feedback_iterator, feedback_loader
                )
                values = _feedback_step(
                    model,
                    feedback_model,
                    feedback_batch,
                    criterion,
                    focus_pos_weight,
                    pretrain_optimizer,
                    head_optimizer,
                    scaler,
                    device,
                    config,
                    feedback_step,
                    pretrain_grad_ema,
                    ema_momentum,
                    use_amp,
                )
                if values.get("skipped", 0.0) < 0.5:
                    feedback_step += 1
                feedback_updates += 1
                for key, value in values.items():
                    feedback_totals[key] += float(value)

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch:3d} | Batch {batch_index:4d}/{len(pretrain_loader)} | "
                    f"Total={loss.item():.6f} token={info.get('jepa', 0.0):.6f} "
                    f"var={info.get('variance', 0.0):.5f} | "
                    f"feedback_steps={feedback_step} | "
                    f"pre_grad_ema={pretrain_grad_ema:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_batches = max(1, len(pretrain_loader))
        train_metrics = {
            key: value / train_batches for key, value in train_totals.items()
        }
        feedback_train_metrics = {
            key: value / max(1, feedback_updates)
            for key, value in feedback_totals.items()
        }

        pretrain_metrics = evaluate_pretrain(
            model,
            pretrain_val_loader,
            device,
            seed=config.seed + 1000,
            use_amp=use_amp,
        )
        if epoch >= feedback_start_epoch:
            meta_metrics = _evaluate_feedback(
                feedback_model, meta_loader, criterion, device, config, use_amp
            )
            val_metrics = _evaluate_feedback(
                feedback_model, val_loader, criterion, device, config, use_amp
            )
        else:
            empty_auc = [0.5] * len(config.data.multidisease_labels)
            meta_metrics = {
                "loss": 0.0,
                "accuracy": 0.0,
                "macro_auc": 0.5,
                "auc_list": empty_auc,
                "focus_auc": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            val_metrics = dict(meta_metrics)
        score = compute_best_metric(
            val_metrics["macro_auc"], val_metrics["focus_auc"], config.train
        )
        healthy = _representation_is_healthy(pretrain_metrics)
        summary = (
            f"Epoch {epoch:3d} | Train total={train_metrics.get('total_loss', 0.0):.6f} "
            f"token={train_metrics.get('jepa', 0.0):.6f} "
            f"var={train_metrics.get('variance', 0.0):.5f} "
            f"feedback={feedback_train_metrics.get('loss', 0.0):.6f} | "
            f"transport={transport_progress:.3f} "
            f"PreVal={pretrain_metrics.get('total_loss', 0.0):.6f} "
            f"healthy={healthy} | Meta AUC={meta_metrics['macro_auc']:.4f} "
            f"CHD={meta_metrics['focus_auc']:.4f} | Val AUC={val_metrics['macro_auc']:.4f} "
            f"CHD={val_metrics['focus_auc']:.4f} score={score:.4f} | "
            f"Time={time.time() - epoch_start:.1f}s"
        )
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(device).total_memory
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            summary += (
                f" | peak_alloc={peak_allocated / 2**30:.2f}GB"
                f"({100.0 * peak_allocated / total_vram:.1f}%)"
                f" peak_reserved={peak_reserved / 2**30:.2f}GB"
                f"({100.0 * peak_reserved / total_vram:.1f}%)"
            )
        print(summary)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")

        checkpoint_ready = (
            healthy
            and transport_progress >= 1.0 - 1e-8
            and feedback_step > config.train.taskaware_head_warmup_steps
            and config.train.taskaware_feedback_encoder_grad_ratio > 0.0
        )
        payload = _checkpoint_payload(
            model,
            feedback_model,
            pretrain_optimizer,
            head_optimizer,
            scheduler,
            scaler,
            config,
            epoch,
            pretrain_step,
            feedback_step,
            max(best_score, score if healthy else best_score),
            pretrain_grad_ema,
            resolved_split,
            pretrain_metrics,
            val_metrics,
            not continuing_phase2,
        )
        _save_checkpoint(payload, os.path.join(config.output_dir, "jepa_taskaware_last.pt"))
        if checkpoint_ready and score > best_score:
            best_score = score
            payload["best_taskaware_score"] = best_score
            _save_checkpoint(
                payload, os.path.join(config.output_dir, "jepa_taskaware_best.pt")
            )
            print(f"[Checkpoint] New best task-aware score={best_score:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Start Phase 2 + task-aware training from random initialization",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--output_dir", default="outputs_taskaware")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrain_batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--feedback_batch_size", type=int, default=None)
    parser.add_argument("--feedback_interval", type=int, default=None)
    parser.add_argument("--feedback_start_epoch", type=int, default=None)
    parser.add_argument("--feedback_segments", type=int, default=None)
    parser.add_argument("--head_warmup_steps", type=int, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--feedback_grad_ratio", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.seed = args.seed
    config.output_dir = args.output_dir
    config.model.pretrain_phase = 2
    if args.epochs is not None:
        config.train.taskaware_epochs = max(1, args.epochs)
    if args.pretrain_batch_size is not None:
        config.train.phase2_batch_size = max(1, args.pretrain_batch_size)
    if args.accum_steps is not None:
        config.train.phase2_accum_steps = max(1, args.accum_steps)
    if args.feedback_batch_size is not None:
        config.train.taskaware_feedback_batch_size = max(2, args.feedback_batch_size)
    if args.feedback_interval is not None:
        config.train.taskaware_feedback_interval = max(1, args.feedback_interval)
    if args.feedback_start_epoch is not None:
        config.train.taskaware_feedback_start_epoch = max(
            0, args.feedback_start_epoch
        )
    if args.feedback_segments is not None:
        config.train.taskaware_feedback_segments = max(1, args.feedback_segments)
    if args.head_warmup_steps is not None:
        config.train.taskaware_head_warmup_steps = max(0, args.head_warmup_steps)
    if args.head_lr is not None:
        config.train.taskaware_head_lr = max(0.0, args.head_lr)
    if args.feedback_grad_ratio is not None:
        config.train.taskaware_feedback_encoder_grad_ratio = max(
            0.0, args.feedback_grad_ratio
        )
    if args.workers is not None:
        config.train.pretrain_dataloader_workers = max(0, args.workers)
        config.train.dataloader_workers = max(0, args.workers)
    if args.no_amp:
        config.train.phase2_use_amp = False
    split_file = args.split or config.data.multidisease_taskaware_split_file
    if args.from_scratch and (args.checkpoint or args.resume):
        parser.error("--from_scratch cannot be combined with --checkpoint or --resume")
    if not args.from_scratch and not (args.checkpoint or args.resume):
        parser.error("provide --checkpoint/--resume or use --from_scratch")
    train_taskaware(
        config,
        args.checkpoint,
        split_file,
        resume=args.resume,
        from_scratch=args.from_scratch,
    )


if __name__ == "__main__":
    main()
