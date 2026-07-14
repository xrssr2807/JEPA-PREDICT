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
import copy
import pickle
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, fbeta_score
)

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import DownstreamDataset, DualDownstreamDataset, MultiDiseaseDataset
from models.encoder import SignalEncoder
from models.classifier import (
    SignalClassifier, DualChannelClassifier,
    SignalClassifierCoT, DualChannelClassifierCoT,
    MultiScaleClassifier,
)
from models.losses import build_criterion, compute_pos_weight


def uid_from_filename(fname: str) -> str:
    """Extract patient id from train/test_<uid>_<segment>.pkl style names."""
    parts = fname.split("_")
    if parts[0] in {"train", "test", "val"} and len(parts) >= 3:
        return parts[1]
    return parts[0]


def split_files_by_uid(files: List[str], labels: List[int], val_split: float):
    """Group split by patient id to avoid segment leakage across train/val."""
    if val_split <= 0:
        return files, []

    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    groups = [uid_from_filename(f) for f in files]
    if len(set(groups)) >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_split, random_state=42
        )
        train_idx, val_idx = next(splitter.split(files, labels, groups))
    else:
        stratify = labels if len(set(labels)) > 1 else None
        train_idx, val_idx = train_test_split(
            range(len(files)), test_size=val_split,
            stratify=stratify, random_state=42,
        )

    return [files[i] for i in train_idx], [files[i] for i in val_idx]


# ── Data ────────────────────────────────────────────────────────

def build_downstream_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: str = "chd",
    use_dual: bool = False,
) -> tuple:
    """Build train and test dataloaders for downstream fine-tuning."""
    binary_abnormal = (dataset == "arrhythmia_binary")

    if dataset == "arrhythmia" or dataset == "arrhythmia_binary":
        ppg_dir = data_config.arrhythmia_dir + "/data"
        split_file = data_config.arrhythmia_dir + "/split.json"
        ecg_dir = None
    elif dataset == "chd":
        split_file = data_config.chd_ppg_dir + "/train_test_split.json"
        ppg_dir = data_config.chd_ppg_dir + "/ppg_chd"
        ecg_dir = os.path.join(data_config.chd_ecg_dir, data_config.chd_ecg_subdir)
    elif dataset in ("multidisease", "multilabel"):
        target_len = data_config.signal_align_to if data_config.signal_align_to > 0 else None
        train_dataset = MultiDiseaseDataset(
            data_dir=data_config.multidisease_dir,
            split="train",
            disease_labels=data_config.multidisease_labels,
            normalize=data_config.normalize,
            normalize_clip=data_config.normalize_clip,
            channel=0,
            target_length=target_len,
        )
        test_dataset = MultiDiseaseDataset(
            data_dir=data_config.multidisease_dir,
            split="test",
            disease_labels=data_config.multidisease_labels,
            normalize=data_config.normalize,
            normalize_clip=data_config.normalize_clip,
            channel=0,
            target_length=target_len,
        )
        val_dataset = None
        if data_config.val_split > 0:
            labels_for_split = []
            for fname in train_dataset.files:
                with open(os.path.join(data_config.multidisease_dir, fname), "rb") as f:
                    item = pickle.load(f)
                labels_for_split.append(int(item["label"].get("冠心病", 0)))
            train_files, val_files = split_files_by_uid(
                train_dataset.files, labels_for_split, data_config.val_split
            )
            val_dataset = copy.deepcopy(train_dataset)
            train_dataset.files = train_files
            val_dataset.files = val_files
            print(f"[Data] UID-group train/val split: {len(train_files)} train + {len(val_files)} val")

        train_loader = DataLoader(
            train_dataset, batch_size=train_config.downstream_batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=train_config.downstream_batch_size,
                shuffle=False, num_workers=4, pin_memory=True,
            )
        test_loader = DataLoader(
            test_dataset, batch_size=train_config.downstream_batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        vlen = len(val_dataset) if val_dataset is not None else 0
        print(f"[Data] train={len(train_dataset)} val={vlen} test={len(test_dataset)}")
        return train_loader, val_loader, test_loader, train_dataset, test_dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # PPG (always loaded)
    target_len = data_config.signal_align_to if data_config.signal_align_to > 0 else None
    ppg_train = DownstreamDataset(
        data_dir=ppg_dir, split_file=split_file, split="train",
        normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
        binary_abnormal=binary_abnormal,
        signal_quality_gate=data_config.signal_quality_gate,
        target_length=target_len,
    )
    ppg_test = DownstreamDataset(
        data_dir=ppg_dir, split_file=split_file, split="test",
        normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
        binary_abnormal=binary_abnormal,
        signal_quality_gate=data_config.signal_quality_gate,
        target_length=target_len,
    )

    if use_dual and ecg_dir is not None:
        ecg_train = DownstreamDataset(
            data_dir=ecg_dir, split_file=split_file, split="train",
            normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
            binary_abnormal=binary_abnormal,
            signal_quality_gate=data_config.signal_quality_gate,
            target_length=target_len,
        )
        ecg_test = DownstreamDataset(
            data_dir=ecg_dir, split_file=split_file, split="test",
            normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
            binary_abnormal=binary_abnormal,
            signal_quality_gate=data_config.signal_quality_gate,
            target_length=target_len,
        )
        train_dataset = DualDownstreamDataset(ppg_train, ecg_train)
        test_dataset = DualDownstreamDataset(ppg_test, ecg_test)
    else:
        train_dataset, test_dataset = ppg_train, ppg_test

    # ★ 从训练集划分验证集 (按 UID, 防数据泄露)
    val_loader = None
    val_dataset = None
    if data_config.val_split > 0:
        # 收集训练集UID和标签
        train_uids, train_labels = [], []
        for fname in ppg_train.files:
            uid = uid_from_filename(fname)
            train_uids.append(uid)
        # 简化: 直接用文件索引做分层抽样
        train_files = ppg_train.files
        train_file_labels = []
        for fname in train_files:
            # 读取标签 (临时)
            with open(os.path.join(ppg_dir, fname), 'rb') as f:
                d = pickle.load(f)
            train_file_labels.append(d['label'][0]['class'])
        # 分层拆分: 85% train, 15% val
        train_files, val_files = split_files_by_uid(
            train_files, train_file_labels, data_config.val_split
        )
        print(f"[Data] Train→Val split: {len(train_files)} train + {len(val_files)} val")

        # 重建数据集
        ppg_val = copy.deepcopy(ppg_train)
        ppg_val.files = val_files
        ppg_train.files = train_files

        val_dataset = ppg_val
        if use_dual and ecg_dir is not None:
            ecg_val = copy.deepcopy(ecg_train)
            ecg_val.files = val_files
            ecg_train.files = train_files
            val_dataset = DualDownstreamDataset(ppg_val, ecg_val)
            train_dataset = DualDownstreamDataset(ppg_train, ecg_train)
        else:
            train_dataset = ppg_train
            val_dataset = ppg_val

        val_loader = DataLoader(
            val_dataset, batch_size=train_config.downstream_batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.downstream_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.downstream_batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    vlen = len(val_dataset) if val_dataset is not None else 0
    print(f"[Data] train={len(train_dataset)} val={vlen} test={len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_dataset, test_dataset


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
        use_se=model_config.cnn_use_se,
        use_inception=model_config.cnn_use_inception,
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
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
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
                scheduler=None, sched_mode="epoch", is_dual=False,
                distill_mode=False, ecg_encoder=None,
                proj_ppg=None, proj_ecg=None,
                ecg_loader=None, distill_lambda=0.5,
                cotrain_mode=False, ecg_model=None, classifier=None,
                multilabel=False, focus_label_index: int = 4,
                focus_loss_weight: float = 0.0,
                focus_pos_weight: Optional[torch.Tensor] = None):
    """Single training epoch with optional ECG distillation."""
    model.train()
    if distill_mode:
        proj_ppg.train()
    running_loss = 0.0
    correct = 0
    total = 0
    valid_steps = 0

    ecg_iter = iter(ecg_loader) if (distill_mode or cotrain_mode) else None

    for batch in dataloader:
        if is_dual:
            ecg, ppg, labels, *_ = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
            loss = criterion(logits, labels)
        else:
            if len(batch) >= 3:
                x, labels, *_ = batch
            else:
                x, labels = batch
            x, labels = x.to(device), labels.to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

            if distill_mode:
                # PPG forward with embedding for alignment
                logits, ppg_pooled = model(x, return_embedding=True)
                cls_loss = criterion(logits, labels)

                # ECG alignment
                try:
                    ecg_batch = next(ecg_iter)
                except StopIteration:
                    ecg_iter = iter(ecg_loader)
                    ecg_batch = next(ecg_iter)
                ex, *_ = ecg_batch
                ex = ex.to(device)
                with torch.no_grad():
                    ecg_pooled, _ = ecg_encoder(ex)
                align_loss = (1 - F.cosine_similarity(
                    proj_ppg(ppg_pooled), proj_ecg(ecg_pooled), dim=-1
                )).mean()
                loss = cls_loss + distill_lambda * align_loss
            elif cotrain_mode:
                # ★ Co-training: PPG batch + ECG batch, shared classifier
                logits, ppg_pooled = model(x, return_embedding=True)
                cls_loss_ppg = criterion(logits, labels)
                # ECG batch
                try:
                    ecg_batch = next(ecg_iter)
                except (StopIteration, NameError):
                    ecg_iter = iter(ecg_loader)
                    ecg_batch = next(ecg_iter)
                ex, elabels, *_ = ecg_batch
                ex, elabels = ex.to(device), elabels.to(device)
                ecg_pooled, _ = ecg_encoder(ex)
                ecg_logits = classifier(ecg_pooled)
                cls_loss_ecg = criterion(ecg_logits, elabels)
                loss = cls_loss_ppg + cls_loss_ecg
            else:
                logits = model(x)
                loss = criterion(logits, labels)

        if multilabel and focus_loss_weight > 0 and 0 <= focus_label_index < logits.size(1):
            focus_logits = logits[:, focus_label_index]
            focus_labels = labels[:, focus_label_index].float()
            focus_loss = F.binary_cross_entropy_with_logits(
                focus_logits, focus_labels, pos_weight=focus_pos_weight
            )
            loss = loss + focus_loss_weight * focus_loss

        optimizer.zero_grad()
        if not torch.isfinite(loss):
            print("[Warn] non-finite loss detected; skipping this batch")
            continue
        loss.backward()
        all_params = list(model.parameters()) + (list(proj_ppg.parameters()) if distill_mode else [])
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        if scheduler is not None and sched_mode == "batch":
            scheduler.step()

        running_loss += loss.item()
        valid_steps += 1
        total += labels.size(0)
        if multilabel:
            predicted = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
            correct += predicted.eq(labels).float().mean(dim=1).sum().item()
        else:
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

    if valid_steps == 0 or total == 0:
        return float("nan"), 0.0
    return running_loss / valid_steps, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes: int,
             is_dual: bool = False, aggregate_by_uid: bool = True):
    """
    Comprehensive evaluation with optional per-patient segment aggregation.

    ECG-FM 论文证明：同一患者的多段logits聚合（平均/最大）
    可提升 AUPRC 达 16.65%。

    Args:
        aggregate_by_uid: True → 按患者聚合多段logits再算指标
    Returns:
        loss, accuracy, per-class AUCs, classification_report_str,
        all_preds, all_labels, all_probs
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # ★ 用于 segment 聚合的 uid 级缓存
    uid_logits = {}   # uid → list of logits (per segment)
    uid_labels = {}   # uid → label (同一患者所有段标签相同)

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if is_dual:
            ecg, ppg, labels, *_ = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
            uids = batch[3] if len(batch) >= 4 else None
        else:
            if len(batch) >= 3:
                x, labels, *rest = batch
                uids = rest[0] if rest else None
            else:
                x, labels = batch
                uids = None
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)
        running_loss += loss.item()
        probs = logits.softmax(dim=-1)
        _, predicted = logits.max(1)

        # ★ 按uid收集（segment聚合用）
        if aggregate_by_uid and uids is not None:
            for i, uid in enumerate(uids):
                uid_str = str(uid)
                if uid_str not in uid_logits:
                    uid_logits[uid_str] = []
                    uid_labels[uid_str] = labels[i].item()
                uid_logits[uid_str].append(logits[i:i+1])  # keep as (1, C)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / len(dataloader)

    # ── 按uid聚合计算最终指标 ──
    if aggregate_by_uid and uid_logits:
        # 每个患者：平均所有段logits → 再做分类
        uid_agg_preds = []
        uid_agg_labels = []
        uid_agg_probs = []
        for uid in uid_logits:
            # 平均logits (ECG-FM: 平均提升最大)
            stacked = torch.cat(uid_logits[uid], dim=0)  # (N_segments, C)
            avg_logit = stacked.mean(dim=0, keepdim=True)  # (1, C)
            avg_prob = avg_logit.softmax(dim=-1)  # (1, C)
            _, avg_pred = avg_logit.max(dim=-1)

            uid_agg_probs.append(avg_prob.cpu().numpy())
            uid_agg_preds.append(avg_pred.item())
            uid_agg_labels.append(uid_labels[uid])

        all_labels_arr = np.array(uid_agg_labels)
        all_preds_arr = np.array(uid_agg_preds)
        all_probs = np.concatenate(uid_agg_probs, axis=0)
        agg_n = len(uid_logits)
        acc = 100.0 * (all_preds_arr == all_labels_arr).sum() / agg_n

        print(f"[Evaluate] ★ Segment聚合: {agg_n} 患者 "
              f"(来自 {total} 段PPG, 平均每患者 {total/agg_n:.1f} 段)")
    else:
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


@torch.no_grad()
def evaluate_multilabel(model, dataloader, criterion, device,
                        label_names: List[str], aggregate_by_uid: bool = True,
                        thresholds: Optional[np.ndarray] = None):
    """Evaluate multi-label disease prediction with sigmoid probabilities."""
    model.eval()
    running_loss = 0.0
    uid_logits = {}
    uid_labels = {}
    all_logits = []
    all_labels = []

    for batch in dataloader:
        x, labels, *rest = batch
        uids = rest[0] if rest else None
        x, labels = x.to(device), labels.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = model(x)
        loss = criterion(logits, labels)
        running_loss += loss.item()

        if aggregate_by_uid and uids is not None:
            for i, uid in enumerate(uids):
                uid_str = str(uid)
                if uid_str not in uid_logits:
                    uid_logits[uid_str] = []
                    uid_labels[uid_str] = labels[i:i + 1]
                uid_logits[uid_str].append(logits[i:i + 1])
        else:
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = running_loss / max(len(dataloader), 1)

    if aggregate_by_uid and uid_logits:
        logits_arr = []
        labels_arr = []
        for uid in uid_logits:
            logits_arr.append(torch.cat(uid_logits[uid], dim=0).mean(dim=0, keepdim=True).cpu())
            labels_arr.append(uid_labels[uid].cpu())
        logits_arr = torch.cat(logits_arr, dim=0).numpy()
        labels_arr = torch.cat(labels_arr, dim=0).numpy()
        print(f"[Evaluate] UID aggregation: {len(uid_logits)} patients")
    else:
        logits_arr = torch.cat(all_logits, dim=0).numpy()
        labels_arr = torch.cat(all_labels, dim=0).numpy()

    logits_arr = np.nan_to_num(logits_arr, nan=0.0, posinf=60.0, neginf=-60.0)
    logits_arr = np.clip(logits_arr, -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    if thresholds is None:
        thresholds = np.full(labels_arr.shape[1], 0.5, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    preds = (probs >= thresholds).astype(np.float32)

    auc_list = []
    for c in range(labels_arr.shape[1]):
        try:
            if len(np.unique(labels_arr[:, c])) > 1:
                auc_list.append(float(roc_auc_score(labels_arr[:, c], probs[:, c])))
            else:
                auc_list.append(0.5)
        except Exception:
            auc_list.append(0.5)
    macro_auc = float(np.mean(auc_list)) if auc_list else 0.5

    precision = precision_score(labels_arr, preds, average="macro", zero_division=0)
    recall = recall_score(labels_arr, preds, average="macro", zero_division=0)
    f1 = fbeta_score(labels_arr, preds, beta=1, average="macro", zero_division=0)
    f05 = fbeta_score(labels_arr, preds, beta=0.5, average="macro", zero_division=0)
    acc = 100.0 * (preds == labels_arr).mean()
    report = classification_report(
        labels_arr, preds, target_names=label_names, digits=4, zero_division=0
    )

    return (avg_loss, acc, macro_auc, auc_list,
            precision, recall, f1, f05, report,
            preds, labels_arr, probs)


# ── Main Pipeline ───────────────────────────────────────────────

def tune_multilabel_thresholds(labels: np.ndarray, probs: np.ndarray,
                               beta: float = 1.0,
                               min_threshold: float = 0.05,
                               max_threshold: float = 0.95) -> np.ndarray:
    """Tune one decision threshold per label on validation predictions."""
    grid = np.linspace(min_threshold, max_threshold, 91)
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)

    for c in range(labels.shape[1]):
        y_true = labels[:, c]
        if len(np.unique(y_true)) < 2:
            continue

        best_score = -1.0
        best_thr = 0.5
        for thr in grid:
            y_pred = (probs[:, c] >= thr).astype(np.float32)
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds[c] = best_thr

    return thresholds


def multilabel_per_class_metrics(label_names: List[str], labels: np.ndarray,
                                 preds: np.ndarray, probs: np.ndarray,
                                 auc_list: List[float]) -> List[dict]:
    """Build per-disease metrics for final multi-label reporting."""
    rows = []
    for i, name in enumerate(label_names):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        rows.append({
            "name": name,
            "auc": float(auc_list[i]) if i < len(auc_list) else 0.5,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f05": float(f05),
            "support": int(y_true.sum()),
            "pred_pos": int(y_pred.sum()),
        })
    return rows


def format_multilabel_metrics_table(rows: List[dict]) -> str:
    """Format per-disease metrics as a compact text table."""
    lines = [
        "Per-disease metrics:",
        "Disease\tAUC\tPrecision\tRecall\tF1\tF0.5\tSupport\tPred+",
    ]
    for row in rows:
        lines.append(
            f"{row['name']}\t{row['auc']:.4f}\t{row['precision']:.4f}\t"
            f"{row['recall']:.4f}\t{row['f1']:.4f}\t{row['f05']:.4f}\t"
            f"{row['support']}\t{row['pred_pos']}"
        )
    return "\n".join(lines)


def get_focus_auc(auc_list: List[float], focus_index: int) -> float:
    """Return AUC for the focused label, falling back to macro AUC if needed."""
    if 0 <= focus_index < len(auc_list):
        return float(auc_list[focus_index])
    return float(np.mean(auc_list)) if auc_list else 0.5


def compute_best_metric(macro_auc: float, focus_auc: float, train_config: TrainConfig) -> float:
    """Metric used for checkpoint selection and early stopping."""
    if train_config.best_metric == "chd_auc":
        return focus_auc
    if train_config.best_metric == "hybrid":
        alpha = train_config.best_metric_chd_alpha
        return alpha * focus_auc + (1.0 - alpha) * macro_auc
    return macro_auc


def compute_multilabel_pos_weight(dataset, device, max_weight: float = 20.0):
    """Compute BCE pos_weight from multi-label train files."""
    if not all(hasattr(dataset, attr) for attr in ("files", "data_dir", "disease_labels")):
        return None

    pos = np.zeros(len(dataset.disease_labels), dtype=np.float64)
    for fname in dataset.files:
        with open(os.path.join(dataset.data_dir, fname), "rb") as f:
            item = pickle.load(f)
        label_dict = item.get("label", {})
        pos += np.array(
            [float(label_dict.get(name, 0)) for name in dataset.disease_labels],
            dtype=np.float64,
        )

    total = max(len(dataset.files), 1)
    neg = total - pos
    weights = neg / np.maximum(pos, 1.0)
    weights = np.clip(weights, 0.2, max_weight)
    pos_rate = pos / total
    print(f"[Loss] multilabel pos_rate={[round(float(x), 4) for x in pos_rate]}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


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

    # ── Log file ──
    os.makedirs(config.output_dir, exist_ok=True)
    log_path = os.path.join(config.output_dir, "downstream_log.txt")
    log_fh = open(log_path, "a")
    log_fh.write(f"\n{'='*60}\n")
    log_fh.write(f"Downstream training | Dataset: {dataset} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fh.write(f"{'='*60}\n")

    # Num classes
    multilabel = dataset in ("multidisease", "multilabel")
    if multilabel:
        num_classes = len(config.data.multidisease_labels)
    elif dataset == "arrhythmia":
        num_classes = config.data.arrhythmia_num_classes
    elif dataset == "arrhythmia_binary":
        num_classes = 2
    else:
        num_classes = config.data.num_classes
    print(f"Num classes: {num_classes}")

    # ── Check for ECG modes ──
    ecg_data_dir = os.path.join(config.data.chd_ecg_dir, config.data.chd_ecg_subdir)
    has_ecg = os.path.isdir(ecg_data_dir) and not multilabel
    use_dual = has_ecg and config.model.use_dual_channel
    use_distill = has_ecg and config.model.use_ecg_distill and not use_dual
    use_cotrain = has_ecg and config.model.use_cotrain and not use_dual and not use_distill
    distill_lambda = 0.1
    if use_dual:
        print(f"[Dual] ★ ECG data at {ecg_data_dir} → ECG+PPG concat融合 (AUC target 0.79)")
    elif use_distill:
        print(f"[Distill] ★ ECG data at {ecg_data_dir} → ECG蒸馏模式 (部署仅需PPG)")
    elif use_cotrain:
        print(f"[CoTrain] ★ ECG data at {ecg_data_dir} → ECG+PPG协同训练")
    else:
        print(f"[SingleChannel] No ECG data at {ecg_data_dir} → PPG only")

    # Data
    train_loader, val_loader, test_loader, train_ds, test_ds = build_downstream_dataloaders(
        config.data, config.train, dataset, use_dual=use_dual,
    )

    # ── ECG mode setup ──
    # Dual-channel: load both encoders, concat fusion
    if use_dual:
        ecg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "context", device)
        ppg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)
        encoder = None
        print("[Model] ★ DualChannel ECG+PPG concat融合")
        model = DualChannelClassifier(
            ecg_encoder=ecg_encoder, ppg_encoder=ppg_encoder,
            encoder_dim=config.model.transformer_dim, num_classes=num_classes,
        ).to(device)
        # For layer-wise LR compatibility
        model.encoder = None  # DualChannel has two encoders

    # ── ECG dataloader ──
    ecg_train_loader = None
    ecg_encoder = None
    proj_ppg = None
    proj_ecg = None
    ecg_model = None  # For co-training: separate ECG model with shared classifier
    target_len = config.data.signal_align_to if config.data.signal_align_to > 0 else None
    if use_distill:
        ecg_train_ds = DownstreamDataset(
            data_dir=ecg_data_dir,
            split_file=config.data.chd_ppg_dir + "/train_test_split.json",
            split="train", normalize=config.data.normalize,
            normalize_clip=config.data.normalize_clip,
            target_length=target_len,
        )
        ecg_train_loader = DataLoader(
            ecg_train_ds, batch_size=config.train.downstream_batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )
        # Teacher: ECG encoder (frozen)
        ecg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "context", device)
        ecg_encoder.eval()
        for p in ecg_encoder.parameters():
            p.requires_grad = False
        # Projection heads: 512→256→256
        proj_ppg = nn.Sequential(
            nn.Linear(config.model.transformer_dim, 256), nn.GELU(),
            nn.Linear(256, 256),
        ).to(device)
        proj_ecg = nn.Sequential(
            nn.Linear(config.model.transformer_dim, 256), nn.GELU(),
            nn.Linear(256, 256),
        ).to(device)
        proj_ecg.load_state_dict(proj_ppg.state_dict())
        for p in proj_ecg.parameters():
            p.requires_grad = False
        print(f"[Distill] Projection heads ready, λ={distill_lambda}")

    # ── Co-training: ECG encoder + shared classifier ──
    if use_cotrain:
        ecg_train_ds = DownstreamDataset(
            data_dir=ecg_data_dir,
            split_file=config.data.chd_ppg_dir + "/train_test_split.json",
            split="train", normalize=config.data.normalize,
            normalize_clip=config.data.normalize_clip,
            target_length=target_len,
        )
        ecg_train_loader = DataLoader(
            ecg_train_ds, batch_size=config.train.downstream_batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )
        ecg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "context", device)
        print(f"[CoTrain] ECG encoder loaded (trainable), shared classifier, {len(ecg_train_ds)} ECG samples")

    # Load encoder (PPG student / primary) — skip if dual-channel already set
    if not use_dual:
        encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)

    # Build classifier (skip if dual-channel already created above)
    if not use_dual and not use_distill and not use_cotrain:
        # Pure PPG single-channel
        if config.model.use_multiscale:
            print("[Model] MultiScale classification head")
            model = MultiScaleClassifier(
                encoder=encoder, encoder_dim=config.model.transformer_dim,
                num_classes=num_classes,
            ).to(device)
        elif config.model.use_cot_head:
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
    elif use_distill or use_cotrain:
        # Distill/Cotrain: single-channel + extra components
        if config.model.use_multiscale:
            print("[Model] MultiScale classification head")
            model = MultiScaleClassifier(
                encoder=encoder, encoder_dim=config.model.transformer_dim,
                num_classes=num_classes,
            ).to(device)
        elif config.model.use_cot_head:
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
    # else: use_dual=True — model already created as DualChannelClassifier above

    # ── Auto pos_weight ──
    pos_weight = None
    if config.train.auto_pos_weight:
        if multilabel:
            pos_weight = compute_multilabel_pos_weight(train_ds, device)
        else:
            pos_weight = compute_pos_weight(train_ds, num_classes, device)

    # ── Criterion ──
    criterion_loss_type = config.train.multilabel_loss_type if multilabel else config.train.loss_type
    criterion_pos_weight = pos_weight
    if multilabel and criterion_loss_type == "asl":
        # ASL already down-weights easy negatives; use pos_weight only in the
        # focused CHD auxiliary BCE term below.
        criterion_pos_weight = None
    criterion = build_criterion(
        loss_type=criterion_loss_type,
        num_classes=num_classes,
        pos_weight=criterion_pos_weight,
        gamma=config.train.focal_gamma,
        gamma_neg=config.train.asl_gamma_neg,
        gamma_pos=config.train.asl_gamma_pos,
        clip=config.train.asl_clip,
        label_smoothing=config.train.label_smoothing,
    )
    print(f"[Loss] {type(criterion).__name__}"
          f"{' pos_weight=' + str([round(w,2) for w in pos_weight.tolist()]) if pos_weight is not None else ''}")
    focus_idx = config.train.chd_label_index
    focus_pos_weight = pos_weight[focus_idx] if (multilabel and pos_weight is not None) else None
    if multilabel:
        print(
            f"[Focus] CHD label index={focus_idx} "
            f"extra_loss_weight={config.train.chd_focus_loss_weight} "
            f"best_metric={config.train.best_metric}"
        )

    # ── Phase 1: Linear Probe ──
    n_probe = config.train.downstream_probe_epochs
    if use_distill or use_cotrain:
        n_probe = 1  # 快速初始化, 避免冻结下坍塌
    # dual-channel uses full probe (30 epochs) — 即使不稳定, FT能恢复
    if n_probe > 0:
        print("\n" + "=" * 60)
        distill_tag = " + ECG Distill" if use_distill else ""
        print(f"Phase 1: Linear Probe (frozen encoder{distill_tag}, {n_probe} epochs)")
        print("=" * 60)
        if use_dual:
            model.freeze_encoders()
        else:
            model.freeze_encoder()

        trainable = list(model.parameters())
        if use_distill:
            trainable += list(proj_ppg.parameters())
        if use_cotrain:
            trainable += list(ecg_encoder.parameters())
        trainable = [p for p in trainable if p.requires_grad]
        probe_steps = len(train_loader)
        probe_lr = config.train.downstream_lr * 4 if (use_distill or use_cotrain) else config.train.downstream_lr
        optimizer = AdamW(trainable, lr=probe_lr, weight_decay=1e-4)
        scheduler, sched_mode = build_scheduler(optimizer, config.train, probe_steps)

        for epoch in range(n_probe):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                scheduler=scheduler, sched_mode=sched_mode, is_dual=use_dual,
                distill_mode=use_distill, ecg_encoder=ecg_encoder,
                proj_ppg=proj_ppg, proj_ecg=proj_ecg,
                ecg_loader=ecg_train_loader, distill_lambda=distill_lambda,
                cotrain_mode=use_cotrain, ecg_model=ecg_encoder,
                classifier=model.classifier if use_cotrain else None,
                multilabel=multilabel,
                focus_label_index=focus_idx,
                focus_loss_weight=config.train.chd_focus_loss_weight if multilabel else 0.0,
                focus_pos_weight=focus_pos_weight,
            )
            eval_loader = val_loader if val_loader is not None else test_loader
            if multilabel:
                test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate_multilabel(
                    model, eval_loader, criterion, device, config.data.multidisease_labels,
                )
            else:
                test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate(
                    model, eval_loader, criterion, device, num_classes, is_dual=use_dual,
                )
            focus_auc = get_focus_auc(auc_list, focus_idx) if multilabel else None

            if sched_mode == "epoch":
                scheduler.step()

            log_line = (f"Probe Epoch {epoch+1:2d} | "
                        f"Train L={train_loss:.4f} Acc={train_acc:.2f}% | "
                        f"Test L={test_loss:.4f} Acc={test_acc:5.2f}% AUC={auc:.4f} "
                        f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f}")
            if focus_auc is not None:
                log_line += f" CHD_AUC={focus_auc:.4f}"
            print(log_line)
            log_fh.write(log_line + "\n"); log_fh.flush()
    else:
        print("\n[Probe] Skipped → direct Full Fine-tune (signal aligned)")

    # ── Phase 2: Full Fine-tune ──
    print("\n" + "=" * 60)
    print("Phase 2: Full Fine-tune")
    print("=" * 60)
    if use_dual:
        model.unfreeze_encoders()
    else:
        model.unfreeze_encoder()

    ft_epochs = config.train.downstream_epochs - n_probe
    ft_lr = config.train.downstream_lr * 0.1
    ft_steps = len(train_loader)

    if use_dual:
        print(f"[Optimizer] Dual: uniform LR={ft_lr:.2e}")
        optimizer = AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
    elif use_distill:
        print(f"[Optimizer] Distill: uniform LR={ft_lr:.2e}")
        all_params = list(model.parameters()) + list(proj_ppg.parameters())
        optimizer = AdamW(all_params, lr=ft_lr, weight_decay=1e-4)
    elif use_cotrain:
        print(f"[Optimizer] CoTrain: uniform LR={ft_lr:.2e} (PPG+ECG+classifier)")
        all_params = list(model.parameters()) + list(ecg_encoder.parameters())
        optimizer = AdamW(all_params, lr=ft_lr, weight_decay=1e-4)
    elif config.model.use_layerwise_lr:
        print(f"[Optimizer] Layer-wise LR (base={ft_lr}, decay={config.model.layer_decay})")
        param_groups = get_layerwise_param_groups(model, ft_lr, config.model.layer_decay)
        optimizer = AdamW(param_groups, weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)

    scheduler, sched_mode = build_scheduler(optimizer, config.train, ft_steps)

    best_score = float("-inf")
    best_state = None
    no_improve = 0

    for epoch in range(ft_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, sched_mode=sched_mode, is_dual=use_dual, distill_mode=use_distill, ecg_encoder=ecg_encoder, proj_ppg=proj_ppg, proj_ecg=proj_ecg, ecg_loader=ecg_train_loader, distill_lambda=distill_lambda,
            multilabel=multilabel,
            focus_label_index=focus_idx,
            focus_loss_weight=config.train.chd_focus_loss_weight if multilabel else 0.0,
            focus_pos_weight=focus_pos_weight,
        )
        eval_loader = val_loader if val_loader is not None else test_loader
        eval_name = "Val" if val_loader is not None else "Test"
        if multilabel:
            test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate_multilabel(
                model, eval_loader, criterion, device, config.data.multidisease_labels,
            )
        else:
            test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate(
                model, eval_loader, criterion, device, num_classes, is_dual=use_dual,
            )
        focus_auc = get_focus_auc(auc_list, focus_idx) if multilabel else None
        selected_metric = (
            compute_best_metric(auc, focus_auc, config.train) if multilabel else auc
        )

        if sched_mode == "epoch":
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        log_line = (f"FT Epoch {epoch+1:2d} | "
                    f"Train L={train_loss:.4f} Acc={train_acc:.2f}% | "
                    f"{eval_name} L={test_loss:.4f} Acc={test_acc:5.2f}% AUC={auc:.4f} "
                    f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f} | lr={lr:.2e}")
        if focus_auc is not None:
            log_line += f" CHD_AUC={focus_auc:.4f} Select={selected_metric:.4f}"
        print(log_line)
        log_fh.write(log_line + "\n"); log_fh.flush()

        if selected_metric > best_score:
            best_score = selected_metric
            best_state = {
                "epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                "val_acc": test_acc, "val_auc": auc, "val_f1": f1,
                "val_chd_auc": focus_auc, "val_best_metric": selected_metric,
                "best_metric": config.train.best_metric,
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 5:
            print(f"\n[EarlyStop] {config.train.best_metric} no improvement for {no_improve} epochs -> stop")
            break

    # ── Final Report ──
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (Best Model)")
    print("=" * 60)

    # Load best model and re-evaluate
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    if multilabel:
        thresholds = None
        if val_loader is not None:
            (_, _, _, _, _, _, _, _, _, _, val_labels, val_probs) = evaluate_multilabel(
                model, val_loader, criterion, device, config.data.multidisease_labels,
            )
            thresholds = tune_multilabel_thresholds(val_labels, val_probs, beta=1.0)
            print(f"Tuned thresholds:     {[round(float(t), 3) for t in thresholds]}")
            log_fh.write(f"Tuned thresholds: {[round(float(t), 3) for t in thresholds]}\n")
            if best_state is not None:
                best_state["thresholds"] = thresholds.tolist()

        (_, test_acc, auc, auc_list,
         prec, rec, f1, f05, report, test_preds, test_labels, test_probs) = evaluate_multilabel(
            model, test_loader, criterion, device, config.data.multidisease_labels,
            thresholds=thresholds,
        )
        per_class_rows = multilabel_per_class_metrics(
            config.data.multidisease_labels, test_labels, test_preds, test_probs, auc_list
        )
        per_class_table = format_multilabel_metrics_table(per_class_rows)
        # In config.data.multidisease_labels, CHD/冠心病 is the 5th label.
        chd_row = per_class_rows[4] if len(per_class_rows) > 4 else None
    else:
        (_, test_acc, auc, auc_list,
         prec, rec, f1, f05, report, _, _, _) = evaluate(
            model, test_loader, criterion, device, num_classes, is_dual=use_dual,
        )
        per_class_table = None
        chd_row = None

    print(f"Best Test Acc:       {test_acc:.2f}%")
    print(f"Best Test AUC (macro): {auc:.4f}")
    if auc_list:
        print(f"Per-class AUC:        {[round(a, 4) for a in auc_list]}")
    if chd_row is not None:
        print(
            f"CHD/冠心病 AUC:        {chd_row['auc']:.4f} "
            f"(P={chd_row['precision']:.4f}, R={chd_row['recall']:.4f}, "
            f"F1={chd_row['f1']:.4f}, support={chd_row['support']})"
        )
    print(f"Precision (macro):   {prec:.4f}")
    print(f"Recall (macro):      {rec:.4f}")
    print(f"F1 (macro):          {f1:.4f}")
    print(f"F0.5 (macro):        {f05:.4f}")
    if per_class_table is not None:
        print(f"\n{per_class_table}")
    print(f"\nClassification Report:\n{report}")

    # Save
    save_path = os.path.join(config.output_dir, f"downstream_{dataset}_best.pt")
    if best_state is not None:
        if multilabel and per_class_table is not None:
            best_state["test_auc"] = float(auc)
            best_state["test_per_class_metrics"] = per_class_rows
            if chd_row is not None:
                best_state["test_chd_auc"] = float(chd_row["auc"])
        torch.save(best_state, save_path)
        print(f"Model saved → {save_path}")
        log_fh.write(f"Model saved → {save_path}\n")

    # ── Final log ──
    log_fh.write(f"\n{'='*60}\n")
    log_fh.write(f"FINAL | Acc={test_acc:.2f}% AUC={auc:.4f} F1={f1:.4f}\n")
    if auc_list:
        log_fh.write(f"Per-class AUC: {[round(float(a), 4) for a in auc_list]}\n")
    if chd_row is not None:
        log_fh.write(
            f"CHD/冠心病 AUC: {chd_row['auc']:.4f} "
            f"P={chd_row['precision']:.4f} R={chd_row['recall']:.4f} "
            f"F1={chd_row['f1']:.4f} support={chd_row['support']}\n"
        )
    if per_class_table is not None:
        log_fh.write(per_class_table + "\n")
    log_fh.write(f"Classification Report:\n{report}\n")
    log_fh.close()

    return test_acc


# ── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained JEPA checkpoint")
    parser.add_argument("--dataset", type=str, default="chd",
                        choices=["chd", "arrhythmia", "arrhythmia_binary",
                                 "multidisease", "multilabel"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    config = Config()
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    train_downstream(config, args.checkpoint, args.dataset)
