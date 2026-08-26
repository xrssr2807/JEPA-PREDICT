"""Teacher-regularized PPG-only continued self-supervised adaptation.

Only patients in the frozen downstream training split are exposed here.  The
student receives a corrupted PPG segment while an EMA teacher sees the clean
segment.  Local waveform/derivative and spectral targets provide morphology
supervision, but their prediction heads are discarded after adaptation.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from dataset.data import MultiDiseaseDataset
from train_downstream import (
    load_multidisease_named_split_manifest,
    load_pretrained_encoder,
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corrupt_ppg(
    signal: torch.Tensor,
    mask_ratio: float = 0.40,
    patch_size: int = 25,
    noise_std: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply patch masking, gain jitter, and small additive noise."""
    if signal.ndim != 3:
        raise ValueError("signal must have shape (B,C,L)")
    batch, _, length = signal.shape
    num_patches = max(1, math.ceil(length / patch_size))
    patch_mask = torch.rand(batch, 1, num_patches, device=signal.device)
    patch_mask = patch_mask < float(mask_ratio)
    # Never hide an entire segment.
    all_hidden = patch_mask.all(dim=-1, keepdim=True)
    if bool(all_hidden.any()):
        patch_mask[:, :, 0:1] &= ~all_hidden
    sample_mask = patch_mask.repeat_interleave(patch_size, dim=-1)[..., :length]
    gain = torch.empty(batch, 1, 1, device=signal.device).uniform_(0.90, 1.10)
    corrupted = signal * gain
    corrupted = corrupted + torch.randn_like(corrupted) * float(noise_std)
    corrupted = corrupted.masked_fill(sample_mask, 0.0)
    return corrupted, patch_mask


def local_morphology_targets(
    signal: torch.Tensor, num_tokens: int,
) -> torch.Tensor:
    """Return ordered local waveform and derivative-envelope targets."""
    x = torch.nan_to_num(signal.float()).mean(dim=1, keepdim=True)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
    d1 = F.pad((x[..., 1:] - x[..., :-1]).abs(), (1, 0))
    d2 = F.pad((d1[..., 1:] - d1[..., :-1]).abs(), (1, 0))
    targets = torch.cat([
        F.adaptive_avg_pool1d(x, num_tokens),
        torch.log1p(10.0 * F.adaptive_avg_pool1d(d1, num_tokens)),
        torch.log1p(10.0 * F.adaptive_avg_pool1d(d2, num_tokens)),
    ], dim=1)
    return targets.transpose(1, 2)


def spectral_targets(signal: torch.Tensor, sample_rate_hz: float) -> torch.Tensor:
    """Normalized PPG power in four morphology-relevant frequency bands."""
    x = torch.nan_to_num(signal.float()).mean(dim=1)
    x = x - x.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(x, dim=-1)
    power = spectrum.real.square() + spectrum.imag.square()
    power[:, 0] = 0.0
    power = power / power.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    frequencies = torch.fft.rfftfreq(
        x.size(-1), d=1.0 / float(sample_rate_hz), device=x.device
    )
    bands = []
    for low, high in ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 15.0)):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(
            power[:, mask].sum(dim=-1, keepdim=True)
            if bool(mask.any())
            else torch.zeros(x.size(0), 1, device=x.device)
        )
    return torch.cat(bands, dim=-1)


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    student_params = dict(student.named_parameters())
    for name, target in teacher.named_parameters():
        target.mul_(momentum).add_(student_params[name], alpha=1.0 - momentum)
    student_buffers = dict(student.named_buffers())
    for name, target in teacher.named_buffers():
        source = student_buffers[name]
        if target.dtype.is_floating_point:
            target.mul_(momentum).add_(source, alpha=1.0 - momentum)
        else:
            target.copy_(source)


def build_train_dataset(config: Config, split_path: str) -> MultiDiseaseDataset:
    data_dir = config.data.multidisease_dir
    available = sorted(
        name for name in os.listdir(data_dir) if name.endswith(".pkl")
    )
    split_files, resolved = load_multidisease_named_split_manifest(
        split_path,
        data_dir,
        available,
        split_names=("train", "val", "test"),
        expected_disease_labels=config.data.multidisease_labels,
    )
    train_uids = {
        name.split("_")[1] for name in split_files["train"]
    }
    print(
        f"[Protocol] train_only=true files={len(split_files['train'])} "
        f"patients={len(train_uids)} val_test_unread=true split={resolved}"
    )
    return MultiDiseaseDataset(
        data_dir=data_dir,
        split="train",
        disease_labels=config.data.multidisease_labels,
        normalize=config.data.normalize,
        normalize_clip=config.data.normalize_clip,
        channel=config.data.multidisease_ppg_channel,
        target_length=(
            config.data.signal_align_to if config.data.signal_align_to > 0 else None
        ),
        files=split_files["train"],
        default_source_sample_rate_hz=(
            config.data.multidisease_source_sample_rate_hz
        ),
        canonical_sample_rate_hz=(
            config.data.multidisease_canonical_sample_rate_hz
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.40)
    parser.add_argument("--patch_size", type=int, default=25)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument(
        "--max_steps_per_epoch", type=int, default=0,
        help="Debug-only cap; zero runs every batch",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for PPG continued SSL")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(args.seed)
    config = Config()
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    split_path = Path(args.split).resolve()

    dataset = build_train_dataset(config, str(split_path))
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "generator": generator,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    student = load_pretrained_encoder(
        str(checkpoint_path), config.model, "target", device, in_channels=1
    )
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    dim = int(config.model.transformer_dim)
    local_head = nn.Linear(dim, 3).to(device)
    spectral_head = nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 4)
    ).to(device)
    optimizer = AdamW([
        {"params": student.parameters(), "lr": args.encoder_lr},
        {"params": local_head.parameters(), "lr": args.head_lr},
        {"params": spectral_head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.encoder_lr * 0.05
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(
        f"[Runtime] device={torch.cuda.get_device_name(0)} epochs={args.epochs} "
        f"batch={args.batch_size} steps={total_steps} amp={amp_dtype}"
    )

    global_step = 0
    history = []
    for epoch in range(args.epochs):
        student.train()
        local_head.train()
        spectral_head.train()
        sums = {"total": 0.0, "token": 0.0, "local": 0.0, "spectral": 0.0}
        started = time.time()
        completed_steps = 0
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            clean = batch[0].to(device, non_blocking=True)
            corrupted, _ = corrupt_ppg(
                clean, args.mask_ratio, args.patch_size, args.noise_std
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                _, teacher_tokens = teacher(clean, return_all=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                student_pooled, student_tokens = student(
                    corrupted, return_all=True
                )
                count = min(student_tokens.size(1), teacher_tokens.size(1))
                student_tokens = student_tokens[:, :count]
                teacher_tokens = teacher_tokens[:, :count]
                student_norm = F.layer_norm(
                    student_tokens.float(), (student_tokens.size(-1),)
                )
                teacher_norm = F.layer_norm(
                    teacher_tokens.float(), (teacher_tokens.size(-1),)
                )
                token_loss = 0.5 * F.smooth_l1_loss(
                    student_norm, teacher_norm
                ) + 0.5 * (
                    1.0 - F.cosine_similarity(
                        student_norm, teacher_norm, dim=-1
                    ).mean()
                )
                local_target = local_morphology_targets(clean, count)
                local_loss = F.smooth_l1_loss(
                    local_head(student_tokens).float(), local_target
                )
                spectral_target = spectral_targets(
                    clean, config.data.multidisease_canonical_sample_rate_hz
                )
                spectral_loss = F.smooth_l1_loss(
                    spectral_head(student_pooled).float(), spectral_target
                )
                loss = token_loss + 0.25 * local_loss + 0.10 * spectral_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite SSL loss at epoch={epoch} step={step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress = global_step / max(total_steps - 1, 1)
            momentum = 0.996 + (0.9999 - 0.996) * (
                0.5 - 0.5 * math.cos(math.pi * progress)
            )
            ema_update(student, teacher, momentum)
            global_step += 1
            for name, value in (
                ("total", loss), ("token", token_loss),
                ("local", local_loss), ("spectral", spectral_loss),
            ):
                sums[name] += float(value.detach())
            completed_steps += 1
            if step % 50 == 0:
                print(
                    f"Epoch {epoch + 1:2d}/{args.epochs} step={step:3d}/{len(loader)} "
                    f"loss={loss.item():.5f} token={token_loss.item():.5f} "
                    f"local={local_loss.item():.5f} spectral={spectral_loss.item():.5f} "
                    f"ema={momentum:.5f}",
                    flush=True,
                )
        epoch_metrics = {
            "epoch": epoch + 1,
            **{
                name: value / max(completed_steps, 1)
                for name, value in sums.items()
            },
            "seconds": time.time() - started,
        }
        history.append(epoch_metrics)
        print("[Epoch] " + json.dumps(epoch_metrics, ensure_ascii=False))

    encoder_state = {
        key: value.detach().cpu() for key, value in student.state_dict().items()
    }
    metadata = {
        "method": "ppg_teacher_regularized_continued_ssl",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "mask_ratio": args.mask_ratio,
        "patch_size": args.patch_size,
        "noise_std": args.noise_std,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "split": str(split_path),
        "split_sha256": sha256(split_path),
        "train_files": len(dataset),
        "test_set_sealed": True,
        "history": history,
    }
    output_path = output_dir / "ppg_continued_ssl_last.pt"
    torch.save({
        "ppg_encoder": encoder_state,
        "target_encoder": encoder_state,
        "adaptation_metadata": metadata,
    }, output_path)
    (output_dir / "adaptation_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Complete] checkpoint={output_path} test_set_sealed=true")


if __name__ == "__main__":
    main()
