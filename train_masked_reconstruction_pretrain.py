"""Train capacity-controlled multimodal MAE and xMAE-objective baselines."""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from config import Config
from models.crossmodal_mae import (
    CrossModalMaskedAutoencoder,
    make_patch_mask,
)
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


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def schedule(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def objective_mask_ratios(args, epoch: int):
    if args.objective == "multimodal_mae":
        return args.mae_mask_ratio, args.mae_mask_ratio
    progress = min(max(epoch - 1, 0) / max(args.curriculum_epochs, 1), 1.0)
    ecg_ratio = args.xmae_start_mask_ratio + progress * (
        args.xmae_mask_ratio - args.xmae_start_mask_ratio
    )
    return ecg_ratio, 0.0


def build_masks(
    model,
    ecg,
    ppg,
    ecg_ratio,
    ppg_ratio,
    objective,
    generator=None,
):
    num_tokens = model.token_count(ecg.size(-1))
    ecg_mode = "anchor" if objective == "xmae_objective" else "scatter"
    ecg_mask = make_patch_mask(
        ecg.size(0),
        num_tokens,
        ecg_ratio,
        ecg_mode,
        ecg.device,
        generator,
    )
    ppg_mask = make_patch_mask(
        ppg.size(0),
        num_tokens,
        ppg_ratio,
        "scatter",
        ppg.device,
        generator,
    )
    return ecg_mask, ppg_mask


@torch.no_grad()
def validate(model, loader, device, amp_dtype, args, epoch):
    model.eval()
    ecg_ratio, ppg_ratio = objective_mask_ratios(args, epoch)
    generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
    sums = {"total": 0.0, "ecg": 0.0, "ppg": 0.0}
    steps = 0
    for batch in loader:
        ecg, ppg = batch[:2]
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        ecg_mask, ppg_mask = build_masks(
            model,
            ecg,
            ppg,
            ecg_ratio,
            ppg_ratio,
            args.objective,
            generator,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            losses = model.compute_loss(
                model(ecg, ppg, ecg_mask, ppg_mask)
            )
        for name in sums:
            sums[name] += float(losses[name])
        steps += 1
    return {name: value / max(steps, 1) for name, value in sums.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        choices=("multimodal_mae", "xmae_objective"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accum_steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--mae_mask_ratio", type=float, default=0.60)
    parser.add_argument("--xmae_start_mask_ratio", type=float, default=0.60)
    parser.add_argument("--xmae_mask_ratio", type=float, default=0.80)
    parser.add_argument("--curriculum_epochs", type=int, default=10)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_split_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.accum_steps) < 1:
        raise ValueError("epochs, batch_size, and accum_steps must be positive")
    for value in (
        args.mae_mask_ratio,
        args.xmae_start_mask_ratio,
        args.xmae_mask_ratio,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("mask ratios must be in [0, 1]")
    if args.xmae_start_mask_ratio > args.xmae_mask_ratio:
        raise ValueError("xMAE start mask ratio cannot exceed final ratio")

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    config = Config()
    config.seed = args.seed
    config.train.pretrain_dataloader_workers = args.workers
    train_loader, val_loader = build_pretrain_dataloaders(
        config.data,
        config.train,
        return_stats=False,
        use_processed=True,
        seed=args.data_split_seed,
        batch_size=args.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    ecg_encoder = build_encoder(config.model, in_channels=1)
    ppg_encoder = build_encoder(config.model, in_channels=1)
    max_patch_size = max(32, math.prod(config.model.cnn_strides) + 2)
    model = CrossModalMaskedAutoencoder(
        ecg_encoder=ecg_encoder,
        ppg_encoder=ppg_encoder,
        objective=args.objective,
        model_dim=config.model.transformer_dim,
        heads=config.model.transformer_heads,
        ff_dim=config.model.transformer_ff_dim,
        dropout=config.model.transformer_dropout,
        decoder_depth=args.decoder_depth,
        max_patch_size=max_patch_size,
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
    encoder_parameters = sum(
        parameter.numel()
        for encoder in (model.ecg_encoder, model.ppg_encoder)
        for parameter in encoder.parameters()
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "experiment": f"P0_{args.objective}",
        "implementation": (
            "capacity_controlled_official_objective_reproduction"
            if args.objective == "xmae_objective"
            else "capacity_controlled_multimodal_mae"
        ),
        "seed": args.seed,
        "data_split_seed": args.data_split_seed,
        "train_segments": len(train_loader.dataset),
        "val_segments": len(val_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "optimizer_steps_budget": total_steps,
        "encoder_parameters": encoder_parameters,
        "total_pretrain_parameters": total_parameters,
        "mae_mask_ratio": args.mae_mask_ratio,
        "xmae_start_mask_ratio": args.xmae_start_mask_ratio,
        "xmae_final_mask_ratio": args.xmae_mask_ratio,
        "test_set_used": False,
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    best_loss = float("inf")
    no_improve = 0
    log_path = os.path.join(args.output_dir, "pretrain.log")
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_handle:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ecg_ratio, ppg_ratio = objective_mask_ratios(args, epoch)
            sums = {"total": 0.0, "ecg": 0.0, "ppg": 0.0}
            steps = 0
            start = time.time()
            for batch_index, batch in enumerate(train_loader):
                ecg, ppg = batch[:2]
                ecg = ecg.to(device, non_blocking=True)
                ppg = ppg.to(device, non_blocking=True)
                ecg_mask, ppg_mask = build_masks(
                    model,
                    ecg,
                    ppg,
                    ecg_ratio,
                    ppg_ratio,
                    args.objective,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    losses = model.compute_loss(
                        model(ecg, ppg, ecg_mask, ppg_mask)
                    )
                    scaled_loss = losses["total"] / args.accum_steps
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, batch={batch_index}"
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
                for name in sums:
                    sums[name] += float(losses[name])
                steps += 1

            train_metrics = {
                name: value / max(steps, 1) for name, value in sums.items()
            }
            val_metrics = validate(
                model, val_loader, device, amp_dtype, args, epoch
            )
            message = (
                f"epoch={epoch} objective={args.objective} "
                f"mask_ecg={ecg_ratio:.3f} mask_ppg={ppg_ratio:.3f} "
                f"train_loss={train_metrics['total']:.6f} "
                f"train_ecg={train_metrics['ecg']:.6f} "
                f"train_ppg={train_metrics['ppg']:.6f} "
                f"val_loss={val_metrics['total']:.6f} "
                f"val_ecg={val_metrics['ecg']:.6f} "
                f"val_ppg={val_metrics['ppg']:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8e} "
                f"time={time.time()-start:.1f}s"
            )
            print(message)
            log_handle.write(message + "\n")
            state = {
                "epoch": epoch,
                "context_encoder": model.ecg_encoder.state_dict(),
                "target_encoder": model.ppg_encoder.state_dict(),
                "decoder_state_dict": {
                    key: value
                    for key, value in model.state_dict().items()
                    if not key.startswith(("ecg_encoder.", "ppg_encoder."))
                },
                "val_loss": float(val_metrics["total"]),
                "pretraining_objective": args.objective,
                "pretraining_manifest": manifest,
                "seed": args.seed,
                "test_evaluated": False,
            }
            last_path = os.path.join(args.output_dir, f"{args.objective}_last.pt")
            best_path = os.path.join(args.output_dir, f"{args.objective}_best.pt")
            save_torch_checkpoint_atomic(state, last_path)
            if val_metrics["total"] < best_loss - args.min_delta:
                best_loss = val_metrics["total"]
                no_improve = 0
                save_torch_checkpoint_atomic(state, best_path)
            else:
                no_improve += 1
            if no_improve >= args.patience:
                stop = f"[EarlyStop] plateau for {args.patience} epochs"
                print(stop)
                log_handle.write(stop + "\n")
                break

        completion = (
            f"[Complete] {args.objective} baseline pre-training "
            "| test_set_used=False"
        )
        print(completion)
        log_handle.write(completion + "\n")


if __name__ == "__main__":
    main()
