"""
Downstream fine-tuning: linear probe → full fine-tune on CHD & Arrhythmia classification.

Supports:
  - CHD: PPG binary classification (2 classes)
  - Arrhythmia: PPG multi-class classification (6 classes)
  - Arrhythmia binary: normal vs abnormal (2 classes)

v2 Improvements (from CWT-MAE v3):
  - FocalLoss / AsymmetricLoss for class imbalance
  - Step-based LR scheduler (warmup + cosine, per-step updates)
  - Per-class AUC + Classification Report (sklearn)
  - Auto pos_weight from training data distribution
"""
import os
import sys
import time
import math
import json
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, fbeta_score
)

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import DownstreamDataset
from models.encoder import SignalEncoder
from models.classifier import (
    SignalClassifier, DualChannelClassifier,
    SignalClassifierCoT, DualChannelClassifierCoT,
)
from models.losses import build_criterion, compute_pos_weight


# ── Data ────────────────────────────────────────────────────────

def build_downstream_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: str = "chd",
) -> tuple:
    """Build train and test dataloaders for downstream fine-tuning."""
    binary_abnormal = (dataset == "arrhythmia_binary")

    if dataset == "arrhythmia" or dataset == "arrhythmia_binary":
        data_dir = data_config.arrhythmia_dir + "/data"
        split_file = data_config.arrhythmia_dir + "/split.json"
    elif dataset == "chd":
        data_dir = data_config.chd_ppg_dir + "/ppg_chd"
        split_file = data_config.chd_ppg_dir + "/train_test_split.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_dataset = DownstreamDataset(
        data_dir=data_dir, split_file=split_file, split="train",
        normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
        binary_abnormal=binary_abnormal,
    )
    test_dataset = DownstreamDataset(
        data_dir=data_dir, split_file=split_file, split="test",
        normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
        binary_abnormal=binary_abnormal,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.downstream_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.downstream_batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# ── Encoder ─────────────────────────────────────────────────────

def build_encoder(model_config: ModelConfig) -> SignalEncoder:
    return SignalEncoder(
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
    )


def load_pretrained_encoder(
    checkpoint_path: str, model_config: ModelConfig,
    encoder_type: str, device: torch.device,
) -> SignalEncoder:
    """Load a pre-trained encoder from JEPA checkpoint."""
    encoder = build_encoder(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    key = f"{encoder_type}_encoder"
    if key in ckpt:
        state_dict = ckpt[key]
    else:
        msd = ckpt["model_state_dict"]
        prefix = f"{encoder_type}_encoder."
        state_dict = {
            k[len(prefix):]: v for k, v in msd.items()
            if k.startswith(prefix)
        }

    encoder.load_state_dict(state_dict, strict=True)
    print(f"Loaded {encoder_type}_encoder from {checkpoint_path}")
    return encoder


# ── Layer-wise LR ───────────────────────────────────────────────

def get_layerwise_param_groups(model, base_lr: float, layer_decay: float,
                               encoder_attr: str = "encoder"):
    """Create parameter groups with layer-wise learning rate decay."""
    encoder = getattr(model, encoder_attr, None)
    if encoder is None:
        return [{"params": model.parameters(), "lr": base_lr}]

    num_layers = len(encoder.transformer.blocks) if hasattr(encoder, 'transformer') else 0
    if num_layers == 0:
        return [{"params": model.parameters(), "lr": base_lr}]

    param_groups = []
    handled = set()

    # Head: full LR
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith(encoder_attr + "."):
            head_params.append(param)
            handled.add(param)
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr, "name": "head"})

    # Encoder layers: decayed LR (deeper = smaller LR)
    for layer_idx in range(num_layers):
        lr = base_lr * (layer_decay ** (num_layers - 1 - layer_idx))
        layer_params = []
        for name, param in encoder.named_parameters():
            if (param.requires_grad and
                name.startswith(f"transformer.blocks.{layer_idx}.")):
                layer_params.append(param)
                handled.add(param)
        if layer_params:
            param_groups.append({
                "params": layer_params, "lr": lr, "name": f"layer_{layer_idx}",
            })

    # CNN stem + pos_encoding + proj: bottom-most LR
    bottom_lr = base_lr * (layer_decay ** num_layers)
    bottom_params = []
    for name, param in encoder.named_parameters():
        if param.requires_grad and param not in handled:
            bottom_params.append(param)
            handled.add(param)
    if bottom_params:
        param_groups.append({
            "params": bottom_params, "lr": bottom_lr, "name": "cnn_stem",
        })

    # Remaining
    remaining = [p for _, p in model.named_parameters()
                 if p.requires_grad and p not in handled]
    if remaining:
        param_groups.append({"params": remaining, "lr": base_lr, "name": "other"})

    print(f"[Layer-wise LR] {num_layers} layers, decay={layer_decay}")
    for g in param_groups:
        print(f"  {g['name']}: lr={g['lr']:.2e} ({len(g['params'])} params)")
    return param_groups


# ── Scheduler ───────────────────────────────────────────────────

def build_scheduler(optimizer, train_config, steps_per_epoch: int):
    """
    Build LR scheduler: warmup + cosine annealing.

    If scheduler_type == "step": per-batch updates, T_max = total_steps
    If "epoch": per-epoch updates, T_max = total_epochs
    """
    if train_config.downstream_scheduler == "step":
        total_steps = train_config.downstream_epochs * steps_per_epoch
        warmup_steps = train_config.downstream_warmup_epochs * steps_per_epoch
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                    eta_min=train_config.downstream_min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                  milestones=[warmup_steps])
        step_mode = "batch"
    else:
        warmup = LinearLR(optimizer, start_factor=0.01,
                           total_iters=train_config.downstream_warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                    T_max=train_config.downstream_epochs - train_config.downstream_warmup_epochs,
                                    eta_min=train_config.downstream_min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                  milestones=[train_config.downstream_warmup_epochs])
        step_mode = "epoch"

    print(f"[Scheduler] {step_mode}-based warmup+cosine | "
          f"total_epochs={train_config.downstream_epochs} | per_epoch={steps_per_epoch} steps")
    return scheduler, step_mode


# ── Training ────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device,
                scheduler=None, sched_mode="epoch", is_dual=False):
    """Single training epoch with optional per-step scheduler."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if is_dual:
            ecg, ppg, labels = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
        else:
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and sched_mode == "batch":
            scheduler.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes: int,
             is_dual: bool = False):
    """
    Comprehensive evaluation.

    Returns:
        loss, accuracy, per-class AUCs, classification_report_str,
        all_preds, all_labels, all_probs
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if is_dual:
            ecg, ppg, labels = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
        else:
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)

        running_loss += loss.item()
        probs = logits.softmax(dim=-1)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    acc = 100.0 * correct / total
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)

    # Per-class AUC
    auc_list = []
    for c in range(num_classes):
        if num_classes == 2 and c == 0:
            continue  # binary: only compute AUC for class 1
        try:
            y_true_c = (all_labels_arr == c).astype(int)
            y_prob_c = all_probs[:, c]
            if len(np.unique(y_true_c)) > 1:
                auc_c = roc_auc_score(y_true_c, y_prob_c)
            else:
                auc_c = 0.5
        except Exception:
            auc_c = 0.5
        auc_list.append(auc_c)

    macro_auc = float(np.mean(auc_list)) if auc_list else 0.5

    # Classification report (sklearn)
    try:
        report = classification_report(all_labels_arr, all_preds_arr, digits=4,
                                        zero_division=0)
    except Exception:
        report = "N/A"

    # Precision / Recall / F1 / F0.5 (macro)
    try:
        precision = precision_score(all_labels_arr, all_preds_arr,
                                     average='macro', zero_division=0)
        recall = recall_score(all_labels_arr, all_preds_arr,
                              average='macro', zero_division=0)
        f1 = fbeta_score(all_labels_arr, all_preds_arr, beta=1, average='macro',
                         zero_division=0)
        f05 = fbeta_score(all_labels_arr, all_preds_arr, beta=0.5, average='macro',
                          zero_division=0)
    except Exception:
        precision = recall = f1 = f05 = 0.0

    return (avg_loss, acc, macro_auc, auc_list,
            precision, recall, f1, f05, report,
            all_preds_arr, all_labels_arr, all_probs)


# ── Main Pipeline ───────────────────────────────────────────────

def train_downstream(
    config: Config,
    checkpoint_path: str,
    dataset: str = "chd",
):
    """
    Downstream fine-tuning pipeline.

    Args:
        config: master configuration
        checkpoint_path: path to pre-trained JEPA checkpoint
        dataset: "chd" or "arrhythmia"
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {dataset}")

    # Num classes
    if dataset == "arrhythmia":
        num_classes = config.data.arrhythmia_num_classes
    elif dataset == "arrhythmia_binary":
        num_classes = 2
    else:
        num_classes = config.data.num_classes
    print(f"Num classes: {num_classes}")

    # Data
    train_loader, test_loader, train_ds, test_ds = build_downstream_dataloaders(
        config.data, config.train, dataset
    )

    # Load encoder
    encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)

    # Build classifier
    if config.model.use_cot_head:
        print("[Model] CoT classification head")
        model = SignalClassifierCoT(
            encoder=encoder, encoder_dim=config.model.transformer_dim,
            num_classes=num_classes, num_heads=config.model.transformer_heads,
            num_reasoning_tokens=config.model.cot_tokens,
        ).to(device)
    else:
        model = SignalClassifier(
            encoder=encoder, encoder_dim=config.model.transformer_dim,
            num_classes=num_classes,
        ).to(device)

    # ── Auto pos_weight ──
    pos_weight = None
    if config.train.auto_pos_weight:
        pos_weight = compute_pos_weight(train_ds, num_classes, device)

    # ── Criterion ──
    criterion = build_criterion(
        loss_type=config.train.loss_type,
        num_classes=num_classes,
        pos_weight=pos_weight,
        gamma=config.train.focal_gamma,
        gamma_neg=config.train.asl_gamma_neg,
        gamma_pos=config.train.asl_gamma_pos,
        clip=config.train.asl_clip,
        label_smoothing=config.train.label_smoothing,
    )
    print(f"[Loss] {type(criterion).__name__}"
          f"{' pos_weight=' + str([round(w,2) for w in pos_weight.tolist()]) if pos_weight is not None else ''}")

    # ── Phase 1: Linear Probe ──
    print("\n" + "=" * 60)
    print("Phase 1: Linear Probe (frozen encoder)")
    print("=" * 60)
    model.freeze_encoder()

    trainable = [p for p in model.parameters() if p.requires_grad]
    probe_steps = len(train_loader)
    optimizer = AdamW(trainable, lr=config.train.downstream_lr)
    scheduler, sched_mode = build_scheduler(
        optimizer, config.train, probe_steps
    )

    for epoch in range(config.train.downstream_probe_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, sched_mode=sched_mode,
        )
        test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate(
            model, test_loader, criterion, device, num_classes,
        )

        if sched_mode == "epoch":
            scheduler.step()

        print(f"Probe Epoch {epoch+1:2d} | "
              f"Train L={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Test L={test_loss:.4f} Acc={test_acc:5.2f}% AUC={auc:.4f} "
              f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f}")

    # ── Phase 2: Full Fine-tune ──
    print("\n" + "=" * 60)
    print("Phase 2: Full Fine-tune")
    print("=" * 60)
    model.unfreeze_encoder()

    ft_epochs = config.train.downstream_epochs - config.train.downstream_probe_epochs
    ft_lr = config.train.downstream_lr * 0.1
    ft_steps = len(train_loader)

    if config.model.use_layerwise_lr:
        print(f"[Optimizer] Layer-wise LR (base={ft_lr}, decay={config.model.layer_decay})")
        param_groups = get_layerwise_param_groups(model, ft_lr, config.model.layer_decay)
        optimizer = AdamW(param_groups)
    else:
        optimizer = AdamW(model.parameters(), lr=ft_lr)

    scheduler, sched_mode = build_scheduler(optimizer, config.train, ft_steps)

    best_auc = 0.0
    best_state = None

    for epoch in range(ft_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, sched_mode=sched_mode,
        )
        test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate(
            model, test_loader, criterion, device, num_classes,
        )

        if sched_mode == "epoch":
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"FT Epoch {epoch+1:2d} | "
              f"Train L={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Test L={test_loss:.4f} Acc={test_acc:5.2f}% AUC={auc:.4f} "
              f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f} | lr={lr:.2e}")

        if auc > best_auc:
            best_auc = auc
            best_state = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "test_acc": test_acc, "test_auc": auc, "test_f1": f1,
            }

    # ── Final Report ──
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (Best Model)")
    print("=" * 60)

    # Load best model and re-evaluate
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    (_, test_acc, auc, auc_list,
     prec, rec, f1, f05, report, _, _, _) = evaluate(
        model, test_loader, criterion, device, num_classes,
    )

    print(f"Best Test Acc:       {test_acc:.2f}%")
    print(f"Best Test AUC (macro): {auc:.4f}")
    if auc_list:
        print(f"Per-class AUC:        {[round(a, 4) for a in auc_list]}")
    print(f"Precision (macro):   {prec:.4f}")
    print(f"Recall (macro):      {rec:.4f}")
    print(f"F1 (macro):          {f1:.4f}")
    print(f"F0.5 (macro):        {f05:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Save
    save_path = os.path.join(config.output_dir, f"downstream_{dataset}_best.pt")
    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"Model saved → {save_path}")

    return test_acc


# ── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained JEPA checkpoint")
    parser.add_argument("--dataset", type=str, default="chd",
                        choices=["chd", "arrhythmia", "arrhythmia_binary"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    config = Config()
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    train_downstream(config, args.checkpoint, args.dataset)
