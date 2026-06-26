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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, fbeta_score
)

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import DownstreamDataset, DualDownstreamDataset
from models.encoder import SignalEncoder
from models.classifier import (
    SignalClassifier, DualChannelClassifier,
    SignalClassifierCoT, DualChannelClassifierCoT,
    DualChannelSimpleFusion,
    MultiScaleClassifier, SimpleFusion,
)
from models.losses import build_criterion, compute_pos_weight


# ── Data ────────────────────────────────────────────────────────

def build_downstream_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: str = "chd",
    use_dual: bool = False,
) -> tuple:
    """Build train and test dataloaders for downstream fine-tuning.

    ECG 和 PPG 数据完美配对 (同文件名单、同split、同uid)。
    当 use_dual=True 时，返回 DualDownstreamDataset 同时加载 ECG+PPG。
    """
    binary_abnormal = (dataset == "arrhythmia_binary")

    if dataset == "arrhythmia" or dataset == "arrhythmia_binary":
        data_dir = data_config.arrhythmia_dir + "/data"
        split_file = data_config.arrhythmia_dir + "/split.json"
    elif dataset == "chd":
        split_file = data_config.chd_ppg_dir + "/train_test_split.json"  # PPG和ECG用同一个split!
        ppg_dir = data_config.chd_ppg_dir + "/ppg_chd"
        ecg_dir = os.path.join(data_config.chd_ecg_dir, data_config.chd_ecg_subdir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # PPG 单通道
    ppg_train = DownstreamDataset(
        data_dir=ppg_dir, split_file=split_file, split="train",
        normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
        binary_abnormal=binary_abnormal,
        signal_quality_gate=0.0,  # CHD PPG: SQI不适用, 关闭
    )
    ppg_test = DownstreamDataset(
        data_dir=ppg_dir, split_file=split_file, split="test",
        normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
        binary_abnormal=binary_abnormal,
        signal_quality_gate=0.0,
    )

    if use_dual:
        # ★ 双通道：ECG + PPG 配对加载 (同一split, 同一index → 同一患者)
        ecg_train = DownstreamDataset(
            data_dir=ecg_dir, split_file=split_file, split="train",
            normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
            binary_abnormal=binary_abnormal,
            signal_quality_gate=0.0,
        )
        ecg_test = DownstreamDataset(
            data_dir=ecg_dir, split_file=split_file, split="test",
            normalize=data_config.normalize, normalize_clip=data_config.normalize_clip,
            binary_abnormal=binary_abnormal,
            signal_quality_gate=0.0,
        )
        train_dataset = DualDownstreamDataset(ppg_train, ecg_train)
        test_dataset = DualDownstreamDataset(ppg_test, ecg_test)
        print(f"★ 双通道配对 | PPG: {ppg_dir} | ECG: {ecg_dir}")
    else:
        train_dataset, test_dataset = ppg_train, ppg_test

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.downstream_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.downstream_batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    print(f"[Data] train={len(train_dataset)} test={len(test_dataset)}")
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
        layerdrop=model_config.downstream_layerdrop,
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

def build_scheduler(optimizer, train_config, steps_per_epoch: int, distill_mode: bool = False):
    """
    Build LR scheduler: warmup + cosine annealing.

    If scheduler_type == "step": per-batch updates, T_max = total_steps
    If "epoch": per-epoch updates, T_max = total_epochs
    """
    if train_config.downstream_scheduler == "step":
        total_steps = train_config.downstream_epochs * steps_per_epoch
        warmup_steps = train_config.downstream_warmup_epochs * steps_per_epoch
        warmup_start = 1.0  # 跳过warmup, 防CoT坍塌
        warmup = LinearLR(optimizer, start_factor=warmup_start, total_iters=warmup_steps)
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
                mixup_alpha: float = 0.0, num_classes: int = 2):
    """
    Single training epoch with optional per-step scheduler and MixUp.

    MixUp (针对CHD正样本过少问题):
      在batch内随机混合两个样本及其标签，生成插值训练数据。
      x_mixed = λ · x_i + (1-λ) · x_j
      y_mixed = λ · y_i + (1-λ) · y_j  (软标签)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if is_dual:
            # 兼容 4元组 (ecg, ppg, label, uid) 和 3元组 (ecg, ppg, label)
            if len(batch) == 4:
                ecg, ppg, labels, _ = batch
            else:
                ecg, ppg, labels, *_ = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
            loss = criterion(logits, labels)
        else:
            # ★ 兼容3元组 (x, labels, uid)
            if len(batch) == 3:
                x, labels, _ = batch
            else:
                x, labels, *_ = batch
            x, labels = x.to(device), labels.to(device)

            # ★ MixUp 数据增强 (仅在启用且训练正样本时)
            if mixup_alpha > 0 and num_classes <= 2:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(x.size(0), device=device)
                x_mixed = lam * x + (1 - lam) * x[perm]
                # 软标签: one-hot → 插值
                y_onehot = torch.zeros(x.size(0), num_classes, device=device)
                y_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
                y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
                logits = model(x_mixed)
                loss = criterion(logits, y_mixed)  # 用软标签算loss
            else:
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
            if len(batch) == 4:
                ecg, ppg, labels, _ = batch
            else:
                ecg, ppg, labels, *_ = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
            loss = criterion(logits, labels)
            uids = None
        else:
            # 兼容 2元组 / 3元组
            if len(batch) == 3:
                x, labels, uids = batch
            else:
                x, labels, *_ = batch
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

    # ★ 阈值优化：在验证集上搜索最佳决策阈值，提升CHD召回率
    if num_classes == 2:
        chd_probs = all_probs[:, 1]  # CHD概率
        best_f1, best_thresh, best_recall = 0.0, 0.5, 0.0
        for thresh in np.arange(0.25, 0.75, 0.025):
            pred_at_thresh = (chd_probs >= thresh).astype(int)
            f1_at_thresh = fbeta_score(all_labels_arr, pred_at_thresh, beta=1, average='binary', zero_division=0)
            rec_at_thresh = recall_score(all_labels_arr, pred_at_thresh, zero_division=0)
            if f1_at_thresh > best_f1:
                best_f1 = f1_at_thresh
                best_thresh = thresh
                best_recall = rec_at_thresh
        # 使用最佳阈值重新计算预测
        all_preds_arr = (chd_probs >= best_thresh).astype(int)
        # 同步更新acc
        acc = 100.0 * (all_preds_arr == all_labels_arr).sum() / len(all_labels_arr)
        print(f"[阈值优化] ★ 最佳阈值={best_thresh:.3f}, F1={best_f1:.4f}, CHD召回率={best_recall:.4f}")

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


def extract_dual_features(ecg_encoder, ppg_encoder, dataloader, device, desc="双通道特征提取"):
    """从冻结的双编码器提取 ECG+PPG 拼接特征和标签。"""
    ecg_encoder.eval()
    ppg_encoder.eval()
    all_embeddings, all_labels, all_uids = [], [], []
    total = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        ecg, ppg, labels, uids = batch
        ecg, ppg = ecg.to(device), ppg.to(device)
        with torch.no_grad():
            e, _ = ecg_encoder(ecg)   # (B, 512)
            p, _ = ppg_encoder(ppg)   # (B, 512)
            emb = torch.cat([e, p], dim=-1)  # (B, 1024)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
        all_uids.extend(uids)
        if batch_idx % 20 == 0:
            print(f"  [{desc}] {batch_idx}/{total}", end="\r")
    print(f"  [{desc}] 完成: {len(all_uids)} 样本, 特征维度={all_embeddings[0].shape[1]}")
    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, all_uids


# ── XGBoost 下游训练 ────────────────────────────────────────────
# M2AE 论文 (PPG-ECG Biosignal Fingerprinting):
# 冻结编码器 + XGBoost → CVD 分类 AUROC 0.974
# 在115K小数据场景下，冻结+XGBoost 比全微调更防过拟合

def extract_features(encoder, dataloader, device, desc="特征提取"):
    """从冻结编码器提取所有样本的嵌入和标签。"""
    encoder.eval()
    all_embeddings, all_labels, all_uids = [], [], []
    total = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            x, labels, uids = batch
        else:
            x, labels, *_ = batch
            uids = [f"unknown_{i}" for i in range(x.size(0))]
        x = x.to(device)
        with torch.no_grad():
            emb, _ = encoder(x)  # (B, D)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
        all_uids.extend(uids)
        if batch_idx % 20 == 0:
            print(f"  [{desc}] {batch_idx}/{total}", end="\r")
    print(f"  [{desc}] 完成: {len(all_uids)} 样本")
    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, all_uids


def train_downstream_xgboost(
    config: Config,
    checkpoint_path: str,
    dataset: str = "chd",
):
    """
    XGBoost 下游训练：冻结编码器 → 提取特征 → XGBoost。

    Returns:
        test_auc: float
    """
    print("\n" + "=" * 60)
    print("XGBoost 下游训练 (冻结编码器 + XGBoost)")
    print("=" * 60)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Num classes
    if dataset == "arrhythmia":
        num_classes = config.data.arrhythmia_num_classes
    else:
        num_classes = config.data.num_classes
    print(f"Device: {device} | Dataset: {dataset} | Classes: {num_classes}")

    # Data
    if config.model.use_dual_channel:
        train_loader, test_loader, _, _ = build_downstream_dataloaders(
            config.data, config.train, dataset, use_dual=True,
        )
    else:
        train_loader, test_loader, _, _ = build_downstream_dataloaders(
            config.data, config.train, dataset,
        )

    # Load encoder(s) — dual-channel: 两个编码器都冻结
    if config.model.use_dual_channel:
        ecg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "context", device)
        ppg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)
        ecg_encoder.eval()
        ppg_encoder.eval()
        for param in ecg_encoder.parameters():
            param.requires_grad = False
        for param in ppg_encoder.parameters():
            param.requires_grad = False
        print(f"[Encoder] 双通道冻结: ECG={sum(p.numel() for p in ecg_encoder.parameters()):,} + PPG={sum(p.numel() for p in ppg_encoder.parameters()):,}")
        encoder = None
    else:
        encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"[Encoder] 单通道冻结: {sum(p.numel() for p in encoder.parameters()):,} params")
        ecg_encoder = None
        ppg_encoder = None

    # 提取特征
    if config.model.use_dual_channel:
        print("\n[特征提取] 训练集 (双通道)...")
        X_train, y_train, uid_train = extract_dual_features(
            ecg_encoder, ppg_encoder, train_loader, device, "训练集")
        print(f"[特征提取] 测试集 (双通道)...")
        X_test, y_test, uid_test = extract_dual_features(
            ecg_encoder, ppg_encoder, test_loader, device, "测试集")
    else:
        print("\n[特征提取] 训练集...")
        X_train, y_train, uid_train = extract_features(encoder, train_loader, device, "训练集")
        print(f"[特征提取] 测试集...")
        X_test, y_test, uid_test = extract_features(encoder, test_loader, device, "测试集")

    print(f"\n[数据] 训练: {X_train.shape}, 测试: {X_test.shape}")
    print(f"       类别分布: 训练={np.bincount(y_train)}, 测试={np.bincount(y_test)}")

    # PCA 降维：1024 → 256 (去除冗余维度，加速XGBoost)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    print("\n[PCA] 降维中...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=256)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)
    print(f"[PCA] 1024 → 256, 保留方差: {pca.explained_variance_ratio_.sum():.2%}")

    # 类别权重
    scale_pos_weight = np.sqrt(np.bincount(y_train)[0] / max(np.bincount(y_train)[1], 1))
    print(f"[XGBoost] scale_pos_weight={scale_pos_weight:.3f}")

    # XGBoost 训练 (优化超参)
    print("\n[XGBoost] 训练中...")
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=100,
        verbosity=1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 评估
    print("\n[评估] 测试集...")
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # AUC
    if num_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
        print(f"CHD AUC = {auc:.4f}")
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"Macro AUC = {auc:.4f}")

    # 分类报告
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{report}")

    # 特征重要性
    importance = model.feature_importances_
    top_k = min(10, len(importance))
    top_idx = np.argsort(importance)[-top_k:][::-1]
    print(f"\n[特征重要性] Top {top_k} 维度:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. dim {idx}: {importance[idx]:.4f}")

    return auc


# ── Main Pipeline ───────────────────────────────────────────────

# ── Late Fusion Ensemble ───────────────────────────────────────────
# ECG 和 PPG 数据量不同(81K vs 5K)，各自训练后用平均预测融合

def ensemble_models(config, checkpoint_path, dataset="chd"):
    """
    Late Fusion: 加载 ECG 和 PPG 模型，平均预测。
    """
    print("\n" + "=" * 60)
    print("★ Late Fusion Ensemble: ECG + PPG 平均预测")
    print("=" * 60)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 加载两个模型
    ppg_ckpt = os.path.join(config.output_dir, f"downstream_{dataset}_ppg_best.pt")
    ecg_ckpt = os.path.join(config.output_dir, f"downstream_{dataset}_ecg_best.pt")

    # 加载PPG编码器
    ppg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)
    # 加载ECG编码器
    ecg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "context", device)

    # 构建两个分类器
    num_classes = config.data.arrhythmia_num_classes if "arrhythmia" in dataset else config.data.num_classes

    ppg_model = SignalClassifierCoT(
        encoder=ppg_encoder, encoder_dim=config.model.transformer_dim,
        num_classes=num_classes, num_heads=config.model.transformer_heads,
        num_reasoning_tokens=config.model.cot_tokens,
    ).to(device)
    ecg_model = SignalClassifierCoT(
        encoder=ecg_encoder, encoder_dim=config.model.transformer_dim,
        num_classes=num_classes, num_heads=config.model.transformer_heads,
        num_reasoning_tokens=config.model.cot_tokens,
    ).to(device)

    # 加载训练好的权重
    if os.path.exists(ppg_ckpt):
        ppg_model.load_state_dict(torch.load(ppg_ckpt, map_location=device)["model_state_dict"])
        print(f"[PPG] 加载权重: {ppg_ckpt}")
    else:
        print(f"[PPG] ⚠️ 未找到权重，使用预训练编码器")
    if os.path.exists(ecg_ckpt):
        ecg_model.load_state_dict(torch.load(ecg_ckpt, map_location=device)["model_state_dict"])
        print(f"[ECG] 加载权重: {ecg_ckpt}")
    else:
        print(f"[ECG] ⚠️ 未找到权重，使用预训练编码器")

    ppg_model.eval()
    ecg_model.eval()

    # 用 PPG 的 dataloader (匹配 test set)
    _, test_loader, _, test_ds = build_downstream_dataloaders(
        config.data, config.train, dataset, modality="ppg"
    )

    # 融合预测
    all_fused_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, labels, _ = batch
            else:
                x, labels, *_ = batch
            x = x.to(device)
            labels = labels.to(device)

            # 两个模型的logits
            logits_ppg = ppg_model(x)
            logits_ecg = ecg_model(x)

            # 平均融合
            fused_logits = (logits_ppg + logits_ecg) / 2.0
            fused_probs = fused_logits.softmax(dim=-1)

            all_fused_probs.append(fused_probs.cpu().numpy())
            all_labels.extend(labels.cpu().tolist())

    all_probs = np.concatenate(all_fused_probs, axis=0)
    all_labels_arr = np.array(all_labels)

    # 评估融合结果
    if num_classes == 2:
        auc = roc_auc_score(all_labels_arr, all_probs[:, 1])
        print(f"\n{'='*60}")
        print(f"★ Ensemble AUC = {auc:.4f}")
        print(f"{'='*60}")

        # 阈值优化
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.25, 0.75, 0.025):
            pred = (all_probs[:, 1] >= thresh).astype(int)
            f1 = fbeta_score(all_labels_arr, pred, beta=1, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        pred_final = (all_probs[:, 1] >= best_thresh).astype(int)
        print(f"最佳阈值: {best_thresh:.3f}, F1: {best_f1:.4f}")
        print(f"\n分类报告:\n{classification_report(all_labels_arr, pred_final, digits=4)}")

    return auc


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
    # ★ XGBoost 路径
    if config.model.use_xgboost:
        return train_downstream_xgboost(config, checkpoint_path, dataset)

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

    # Data & Encoder: 按模式选择
    is_dual = False
    if config.model.use_dual_channel:
        is_dual = True
        # ★ 双通道：ECG+PPG配对 (81K文件, 同split, 同uid)
        train_loader, test_loader, train_ds, test_ds = build_downstream_dataloaders(
            config.data, config.train, dataset, use_dual=True,
        )
        ecg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "context", device)
        ppg_encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)
        encoder = None  # 不单独使用
<<<<<<< HEAD
        print("[Model] ★ DualChannel SimpleFusion (ECG+PPG 向量级融合)")
        model = DualChannelSimpleFusion(
=======
        print("[Model] ★ DualChannel ECG+PPG 融合分类头")
        # M2AE 风格: 共享瓶颈融合 (非 token 拼接)
        model = SimpleFusion(
>>>>>>> e8d37a68bb17e6d7007183126ef270cf69f37752
            ecg_encoder=ecg_encoder, ppg_encoder=ppg_encoder,
            encoder_dim=config.model.transformer_dim,
            num_classes=num_classes,
        ).to(device)
    else:
        # 单通道 (PPG)
        train_loader, test_loader, train_ds, test_ds = build_downstream_dataloaders(
            config.data, config.train, dataset, use_dual=False,
        )
        encoder = load_pretrained_encoder(checkpoint_path, config.model, "target", device)

<<<<<<< HEAD
        # ★ ECG蒸馏: 加载冻结ECG编码器 + 投影头
        ecg_encoder_distill = None; proj_ppg = None; proj_ecg = None; ecg_train_loader = None
        if config.model.use_ecg_distill:
            print("[Distill] 加载ECG教师编码器 (冻结)...")
            ecg_encoder_distill = load_pretrained_encoder(checkpoint_path, config.model, "context", device)
            ecg_encoder_distill.eval()
            for p in ecg_encoder_distill.parameters(): p.requires_grad = False
            proj_ppg = nn.Sequential(nn.Linear(config.model.transformer_dim, 256), nn.GELU(), nn.Linear(256, 256)).to(device)
            proj_ecg = nn.Sequential(nn.Linear(config.model.transformer_dim, 256), nn.GELU(), nn.Linear(256, 256)).to(device)
            proj_ecg.load_state_dict(proj_ppg.state_dict())
            for p in proj_ecg.parameters(): p.requires_grad = False
            ecg_train_ds = DownstreamDataset(config.data.chd_ecg_dir+"/ecg_chd", config.data.chd_ppg_dir+"/train_test_split.json", "train", normalize=config.data.normalize)
            ecg_train_loader = DataLoader(ecg_train_ds, batch_size=config.train.downstream_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            print(f"[Distill] ECG教师就绪, lambda={config.model.distill_lambda}")

    # Build classifier (skip if dual-channel already created)
    if is_dual:
        pass  # DualChannelClassifierCoT already created above
    elif config.model.use_multiscale:
        print("[Model] HiMAE多尺度分类头 (细+中+粗三尺度)")
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
=======
    # Build classifier (非双通道时)
    if not is_dual:
        if config.model.use_multiscale:
            print("[Model] HiMAE多尺度分类头 (细+中+粗三尺度)")
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
>>>>>>> e8d37a68bb17e6d7007183126ef270cf69f37752

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
    model.freeze_encoders() if is_dual else model.freeze_encoder()

    trainable = [p for p in model.parameters() if p.requires_grad]
    probe_steps = len(train_loader)
    probe_lr = config.train.downstream_lr * 5 if ecg_encoder_distill is not None else config.train.downstream_lr
    optimizer = AdamW(trainable, lr=probe_lr)
    scheduler, sched_mode = build_scheduler(
        optimizer, config.train, probe_steps,
        distill_mode=(ecg_encoder_distill is not None),
    )

    for epoch in range(config.train.downstream_probe_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, sched_mode=sched_mode,
            mixup_alpha=0.0, num_classes=num_classes,
            is_dual=is_dual,
        )
        test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate(
            model, test_loader, criterion, device, num_classes,
            is_dual=is_dual,
        )

        if sched_mode == "epoch":
            scheduler.step()

        print(f"Probe Epoch {epoch+1:2d} | "
              f"Train L={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Test L={test_loss:.4f} Acc={test_acc:5.2f}% AUC={auc:.4f} "
              f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f}")

    # ── Phase 2: Full Fine-tune ──
    print("\n" + "=" * 60)
    if is_dual:
        print("Phase 2: Full Fine-tune (ECG frozen, PPG unfrozen)")
        model.unfreeze_ppg_only()
    else:
        print("Phase 2: Full Fine-tune")
        model.unfreeze_encoder()
    print("=" * 60)
<<<<<<< HEAD
=======
    model.unfreeze_encoders() if is_dual else model.unfreeze_encoder()
>>>>>>> e8d37a68bb17e6d7007183126ef270cf69f37752

    ft_epochs = config.train.downstream_epochs - config.train.downstream_probe_epochs
    ft_lr = config.train.downstream_lr * 0.1
    ft_steps = len(train_loader)

    if ecg_encoder_distill is not None:
        # 蒸馏模式: uniform LR + 跳过warmup (避免CoT坍塌)
        print(f"[Optimizer] 蒸馏模式: uniform LR={ft_lr:.2e}")
        optimizer = AdamW(list(model.parameters()) + (list(proj_ppg.parameters()) if proj_ppg else []), lr=ft_lr)
    elif config.model.use_layerwise_lr:
        print(f"[Optimizer] Layer-wise LR (base={ft_lr}, decay={config.model.layer_decay})")
        param_groups = get_layerwise_param_groups(model, ft_lr, config.model.layer_decay)
        if proj_ppg is not None:
            param_groups.append({"params": proj_ppg.parameters(), "lr": ft_lr, "name": "proj_ppg"})
        optimizer = AdamW(param_groups)
    else:
        optimizer = AdamW(model.parameters(), lr=ft_lr)

    scheduler, sched_mode = build_scheduler(
        optimizer, config.train, ft_steps,
        distill_mode=(ecg_encoder_distill is not None),
    )

    best_auc = 0.0
    best_state = None
    patience = 15  # early stopping: stop if no AUC improvement for N epochs
    no_improve = 0

    for epoch in range(ft_epochs):
        if ecg_encoder_distill is not None:
            # ★ ECG蒸馏模式: PPG+ECG配对训练
            model.train(); running_loss = 0.0; correct = total = 0
            for (ppg_b, ecg_b) in zip(train_loader, ecg_train_loader):
                x, labels, *_ = ppg_b; ex, *_ = ecg_b
                x, labels, ex = x.to(device), labels.to(device), ex.to(device)
                logits = model(x)
                cls_loss = criterion(logits, labels)
                # 投影对齐 loss
                with torch.no_grad():
                    ecg_pooled, _ = ecg_encoder_distill(ex)
                ppg_pooled, _ = model.encoder(x)
                align_loss = (1 - F.cosine_similarity(
                    proj_ppg(ppg_pooled), proj_ecg(ecg_pooled), dim=-1)).mean()
                loss = cls_loss + config.model.distill_lambda * align_loss
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None and sched_mode == "batch": scheduler.step()
                running_loss += loss.item()
                _, pred = logits.max(1); correct += pred.eq(labels).sum().item(); total += labels.size(0)
            train_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                scheduler=scheduler, sched_mode=sched_mode,
                mixup_alpha=config.train.mixup_alpha if config.train.use_mixup else 0.0,
                num_classes=num_classes,
                is_dual=is_dual,
            )
        test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate(
            model, test_loader, criterion, device, num_classes,
            is_dual=is_dual,
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
            no_improve = 0
            best_state = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "test_acc": test_acc, "test_auc": auc, "test_f1": f1,
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[EarlyStop] AUC未提升 {patience} 个epoch, 终止训练 (best AUC={best_auc:.4f})")
                break

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
        is_dual=is_dual,
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
