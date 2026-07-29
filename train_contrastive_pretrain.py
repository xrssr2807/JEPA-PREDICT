"""Cross-modal ECG/PPG InfoNCE pre-training baseline for ICASSP experiments."""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from train_downstream import build_encoder, save_torch_checkpoint_atomic
from train_pretrain import build_pretrain_dataloaders


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


class CrossModalContrastiveModel(nn.Module):
    """Two modality encoders trained with symmetric paired InfoNCE."""

    def __init__(self, config, projection_dim: int = 128):
        super().__init__()
        self.ecg_encoder = build_encoder(config, in_channels=1)
        self.ppg_encoder = build_encoder(config, in_channels=1)
        hidden_dim = config.transformer_dim
        self.ecg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.ppg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        ecg_embedding, _ = self.ecg_encoder(ecg)
        ppg_embedding, _ = self.ppg_encoder(ppg)
        ecg_projection = F.normalize(
            self.ecg_projection(ecg_embedding), dim=-1
        )
        ppg_projection = F.normalize(
            self.ppg_projection(ppg_embedding), dim=-1
        )
        return ecg_projection, ppg_projection


def augment_waveform(x: torch.Tensor) -> torch.Tensor:
    """Mild morphology-preserving augmentation for contrastive training."""
    batch = x.size(0)
    scale = torch.empty(
        (batch, 1, 1), device=x.device, dtype=x.dtype
    ).uniform_(0.9, 1.1)
    noise = torch.randn_like(x) * 0.01
    return torch.clamp(x * scale + noise, -10.0, 10.0)


def contrastive_loss(
    ecg_projection: torch.Tensor,
    ppg_projection: torch.Tensor,
    temperature: float,
):
    logits = ecg_projection @ ppg_projection.transpose(0, 1)
    logits = logits / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.transpose(0, 1), targets)
    )
    accuracy = 0.5 * (
        (logits.argmax(dim=1) == targets).float().mean()
        + (logits.argmax(dim=0) == targets).float().mean()
    )
    return loss, accuracy


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def schedule(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        progress = (step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


@torch.no_grad()
def validate(model, loader, device, temperature, amp_dtype):
    model.eval()
    loss_sum = 0.0
    accuracy_sum = 0.0
    steps = 0
    for batch in loader:
        ecg, ppg = batch[:2]
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            ecg_projection, ppg_projection = model(ecg, ppg)
            loss, accuracy = contrastive_loss(
                ecg_projection, ppg_projection, temperature
            )
        loss_sum += float(loss)
        accuracy_sum += float(accuracy)
        steps += 1
    return loss_sum / max(steps, 1), accuracy_sum / max(steps, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs_baseline_contrastive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.accum_steps) < 1:
        raise ValueError("epochs, batch_size, and accum_steps must be positive")
    if not 0.0 < args.temperature:
        raise ValueError("temperature must be positive")

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    config = Config()
    config.seed = args.seed
    config.train.pretrain_dataloader_workers = args.workers
    config.model.downstream_encoder_arch = "jepa_transformer"
    train_loader, val_loader = build_pretrain_dataloaders(
        config.data,
        config.train,
        return_stats=False,
        use_processed=True,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = CrossModalContrastiveModel(
        config.model, projection_dim=args.projection_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.accum_steps
    )
    total_steps = max(args.epochs * optimizer_steps_per_epoch, 1)
    scheduler = make_scheduler(
        optimizer,
        args.warmup_epochs * optimizer_steps_per_epoch,
        total_steps,
    )
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best_loss = float("inf")
    no_improve = 0
    global_step = 0
    log_path = os.path.join(args.output_dir, "contrastive_pretrain.log")

    manifest = {
        "experiment": "P3_cross_modal_infonce",
        "seed": args.seed,
        "train_segments": len(train_loader.dataset),
        "val_segments": len(val_loader.dataset),
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "test_set_used": False,
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_handle:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            train_accuracy = 0.0
            train_steps = 0
            epoch_start = time.time()
            for batch_index, batch in enumerate(train_loader):
                ecg, ppg = batch[:2]
                ecg = augment_waveform(ecg.to(device, non_blocking=True))
                ppg = augment_waveform(ppg.to(device, non_blocking=True))
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    ecg_projection, ppg_projection = model(ecg, ppg)
                    loss, accuracy = contrastive_loss(
                        ecg_projection,
                        ppg_projection,
                        args.temperature,
                    )
                    scaled_loss = loss / args.accum_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite contrastive loss at epoch={epoch}, "
                        f"batch={batch_index}"
                    )
                scaler.scale(scaled_loss).backward()
                should_step = (
                    (batch_index + 1) % args.accum_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                train_loss += float(loss)
                train_accuracy += float(accuracy)
                train_steps += 1

            val_loss, val_accuracy = validate(
                model, val_loader, device, args.temperature, amp_dtype
            )
            message = (
                f"epoch={epoch} train_loss={train_loss/max(train_steps,1):.6f} "
                f"train_retrieval_acc={train_accuracy/max(train_steps,1):.4f} "
                f"val_loss={val_loss:.6f} "
                f"val_retrieval_acc={val_accuracy:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.8e} "
                f"time={time.time()-epoch_start:.1f}s"
            )
            print(message)
            log_handle.write(message + "\n")

            state = {
                "epoch": epoch,
                "context_encoder": model.ecg_encoder.state_dict(),
                "target_encoder": model.ppg_encoder.state_dict(),
                "ppg_encoder": model.ppg_encoder.state_dict(),
                "ecg_projection": model.ecg_projection.state_dict(),
                "ppg_projection": model.ppg_projection.state_dict(),
                "val_loss": float(val_loss),
                "val_retrieval_accuracy": float(val_accuracy),
                "pretraining_objective": "symmetric_cross_modal_infonce",
                "seed": args.seed,
                "test_evaluated": False,
            }
            save_torch_checkpoint_atomic(
                state, os.path.join(args.output_dir, "contrastive_last.pt")
            )
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                no_improve = 0
                save_torch_checkpoint_atomic(
                    state, os.path.join(args.output_dir, "contrastive_best.pt")
                )
            else:
                no_improve += 1
            if no_improve >= args.patience:
                stop_message = (
                    f"[EarlyStop] no validation improvement for "
                    f"{args.patience} epochs"
                )
                print(stop_message)
                log_handle.write(stop_message + "\n")
                break

        completion = (
            "[Complete] Cross-modal contrastive baseline pre-training "
            "| test_set_used=False"
        )
        print(completion)
        log_handle.write(completion + "\n")


if __name__ == "__main__":
    main()
