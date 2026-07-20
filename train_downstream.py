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
import hashlib
import copy
import pickle
import random
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, List

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
from dataset.data import (
    DownstreamDataset, DualDownstreamDataset,
    MultiDiseaseDataset, MultiDiseasePatientMILDataset,
)
from models.encoder import SignalEncoder
from models.classifier import (
    SignalClassifier, DualChannelClassifier,
    SignalClassifierCoT, DualChannelClassifierCoT,
    MultiScaleClassifier, PatientMILClassifier, DualStreamPatientMILClassifier,
)
from models.losses import build_criterion, compute_pos_weight


def seed_everything(seed: int):
    """Seed downstream initialization, sampling, shuffling, and CUDA RNGs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] {seed}")


def seed_dataloader_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def resolve_multidisease_split_file(split_file: str, data_dir: str) -> str:
    """Resolve a split manifest from the CLI/config, repo, or dataset directory."""
    split_file = os.path.expandvars(os.path.expanduser(split_file))
    candidates = [split_file]
    if not os.path.isabs(split_file):
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.join(repo_dir, split_file),
            os.path.join(data_dir, split_file),
            os.path.join(os.path.dirname(data_dir), split_file),
        ])

    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Multidisease development split was not found. Checked: "
        + ", ".join(checked)
    )


def multidisease_split_provenance(split_file: str, data_dir: str) -> dict:
    """Return portable split metadata for logs and downstream checkpoints."""
    resolved = resolve_multidisease_split_file(split_file, data_dir)
    with open(resolved, "rb") as f:
        payload = f.read()
    manifest = json.loads(payload.decode("utf-8"))
    return {
        "configured_path": split_file,
        "filename": os.path.basename(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "metadata": manifest.get("metadata", {}),
    }


def load_multidisease_named_split_manifest(
    split_file: str,
    data_dir: str,
    available_files: List[str],
    split_names: Sequence[str],
) -> Tuple[Dict[str, List[str]], str]:
    """Load and validate an exact patient-disjoint named split manifest."""
    split_names = tuple(split_names)
    if len(split_names) < 2 or len(set(split_names)) != len(split_names):
        raise ValueError("split_names must contain at least two unique names")
    resolved = resolve_multidisease_split_file(split_file, data_dir)
    with open(resolved, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Multidisease split must be a JSON object")

    split_lists = {}
    for split_name in split_names:
        files = manifest.get(split_name)
        if not isinstance(files, list) or not files:
            raise ValueError(f"Multidisease split '{split_name}' must be a non-empty list")
        if any(not isinstance(name, str) for name in files):
            raise ValueError(f"Multidisease split '{split_name}' contains a non-string filename")
        invalid = [
            name for name in files
            if "/" in name or "\\" in name
            or not name.endswith(".pkl")
            or not name.startswith(("train_", "test_"))
        ]
        if invalid:
            raise ValueError(
                f"Multidisease split '{split_name}' contains invalid filenames: "
                f"{invalid[:5]}"
            )
        duplicates = len(files) - len(set(files))
        if duplicates:
            raise ValueError(
                f"Multidisease split '{split_name}' contains {duplicates} duplicate files"
            )
        split_lists[split_name] = files

    file_sets = {name: set(files) for name, files in split_lists.items()}
    uid_sets = {
        name: {uid_from_filename(filename) for filename in files}
        for name, files in split_lists.items()
    }
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1:]:
            file_overlap = file_sets[left] & file_sets[right]
            if file_overlap:
                raise ValueError(
                    f"Multidisease {left}/{right} file leakage: "
                    f"{len(file_overlap)} overlapping files"
                )
            uid_overlap = uid_sets[left] & uid_sets[right]
            if uid_overlap:
                raise ValueError(
                    f"Multidisease {left}/{right} patient leakage: "
                    f"{len(uid_overlap)} overlapping UIDs; "
                    f"examples={sorted(uid_overlap)[:5]}"
                )

    available = set(available_files)
    requested = set().union(*file_sets.values())
    missing = requested - available
    if missing:
        raise FileNotFoundError(
            f"Multidisease split references {len(missing)} files missing from {data_dir}; "
            f"examples={sorted(missing)[:5]}"
        )

    ignored = available - requested
    summary = " | ".join(
        f"{name}={len(split_lists[name])} files/{len(uid_sets[name])} UIDs"
        for name in split_names
    )
    print(f"[DataSplit] manifest={resolved} | {summary} | ignored_files={len(ignored)}")
    return {name: sorted(split_lists[name]) for name in split_names}, resolved


def load_multidisease_split_manifest(
    split_file: str,
    data_dir: str,
    available_files: List[str],
) -> Tuple[List[str], List[str], List[str], str]:
    """Load and validate an exact patient-disjoint train/val/test manifest."""
    splits, resolved = load_multidisease_named_split_manifest(
        split_file, data_dir, available_files, ("train", "val", "test")
    )
    return splits["train"], splits["val"], splits["test"], resolved


def load_taskaware_multidisease_split_manifest(
    split_file: str,
    data_dir: str,
    available_files: List[str],
) -> Tuple[List[str], List[str], List[str], List[str], str]:
    """Load the four patient-disjoint splits used by task-aware pre-training."""
    names = ("feedback_train", "feedback_meta", "val", "test")
    splits, resolved = load_multidisease_named_split_manifest(
        split_file, data_dir, available_files, names
    )
    return (
        splits["feedback_train"],
        splits["feedback_meta"],
        splits["val"],
        splits["test"],
        resolved,
    )


# ── Data ────────────────────────────────────────────────────────

def dataloader_performance_kwargs(train_config: TrainConfig) -> dict:
    """Shared loader settings tuned for many small signal pickle files."""
    workers = max(0, int(getattr(train_config, "dataloader_workers", 4)))
    kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["worker_init_fn"] = seed_dataloader_worker
        kwargs["persistent_workers"] = bool(
            getattr(train_config, "dataloader_persistent_workers", True)
        )
        kwargs["prefetch_factor"] = max(
            1, int(getattr(train_config, "dataloader_prefetch_factor", 2))
        )
    return kwargs


def rebuild_train_loader(dataloader: DataLoader, batch_size: int,
                         train_config: TrainConfig) -> DataLoader:
    """Rebuild a training loader for a phase-specific batch size."""
    return DataLoader(
        dataloader.dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_performance_kwargs(train_config),
    )


def configure_cuda_performance(device: torch.device, train_config: TrainConfig):
    """Enable high-throughput CUDA kernels without changing model numerics materially."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    enable_tf32 = bool(getattr(train_config, "enable_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    print(f"[CUDA] cuDNN benchmark=on TF32={'on' if enable_tf32 else 'off'}")


def reset_gpu_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_memory_message(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = 100.0 * torch.cuda.max_memory_allocated(device) / total
    reserved = 100.0 * torch.cuda.max_memory_reserved(device) / total
    return f" GPU_peak={allocated:.1f}% reserved={reserved:.1f}%"


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
            channel=data_config.multidisease_channel,
            target_length=target_len,
        )
        test_dataset = MultiDiseaseDataset(
            data_dir=data_config.multidisease_dir,
            split="test",
            disease_labels=data_config.multidisease_labels,
            normalize=data_config.normalize,
            normalize_clip=data_config.normalize_clip,
            channel=data_config.multidisease_channel,
            target_length=target_len,
        )
        val_dataset = None
        split_manifest = getattr(
            data_config,
            "multidisease_split_file",
            getattr(data_config, "multidisease_development_split", ""),
        )
        if split_manifest:
            available_files = sorted(
                set(train_dataset.files) | set(test_dataset.files)
            )
            train_files, val_files, test_files, _ = load_multidisease_split_manifest(
                split_manifest,
                data_config.multidisease_dir,
                available_files,
            )
            val_dataset = copy.deepcopy(train_dataset)
            train_dataset.files = train_files
            val_dataset.files = val_files
            test_dataset.files = test_files
        elif data_config.val_split > 0:
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

        if data_config.multidisease_patient_mil:
            train_dataset = MultiDiseasePatientMILDataset(
                data_dir=data_config.multidisease_dir,
                split="train",
                disease_labels=data_config.multidisease_labels,
                normalize=data_config.normalize,
                normalize_clip=data_config.normalize_clip,
                channel=data_config.multidisease_channel,
                target_length=target_len,
                max_segments=data_config.multidisease_mil_segments,
                files=train_dataset.files,
                train=True,
            )
            if val_dataset is not None:
                val_dataset = MultiDiseasePatientMILDataset(
                    data_dir=data_config.multidisease_dir,
                    split="train",
                    disease_labels=data_config.multidisease_labels,
                    normalize=data_config.normalize,
                    normalize_clip=data_config.normalize_clip,
                    channel=data_config.multidisease_channel,
                    target_length=target_len,
                    max_segments=data_config.multidisease_mil_segments,
                    files=val_dataset.files,
                    train=False,
                )
            test_dataset = MultiDiseasePatientMILDataset(
                data_dir=data_config.multidisease_dir,
                split="test",
                disease_labels=data_config.multidisease_labels,
                normalize=data_config.normalize,
                normalize_clip=data_config.normalize_clip,
                channel=data_config.multidisease_channel,
                target_length=target_len,
                max_segments=data_config.multidisease_mil_segments,
                files=test_dataset.files,
                train=False,
            )

        batch_size = train_config.downstream_batch_size
        if data_config.multidisease_patient_mil:
            batch_size = getattr(train_config, "multidisease_mil_batch_size", batch_size)
            print(
                f"[Data] Patient-MIL batch_size={batch_size} "
                f"segments={data_config.multidisease_mil_segments} "
                f"(effective segments/step={batch_size * data_config.multidisease_mil_segments})"
            )

        loader_kwargs = dataloader_performance_kwargs(train_config)
        print(
            f"[Data] workers={loader_kwargs['num_workers']} "
            f"prefetch={loader_kwargs.get('prefetch_factor', 0)} "
            f"persistent={loader_kwargs.get('persistent_workers', False)} "
            f"pin_memory={loader_kwargs['pin_memory']}"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, **loader_kwargs,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, **loader_kwargs,
            )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, **loader_kwargs,
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

def build_encoder(model_config: ModelConfig, in_channels: Optional[int] = None) -> SignalEncoder:
    return SignalEncoder(
        in_channels=in_channels or model_config.in_channels,
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


def _select_pretrained_encoder_state(ckpt: dict, encoder_type: str):
    """Select the online modality state while preserving Phase 0 fallback."""
    # Phase 1 stores the directly optimized PPG branch separately. Prefer it
    # for the historical "target" role; Phase 0 checkpoints fall back to the
    # EMA target_encoder exactly as before.
    key = (
        "ppg_encoder"
        if encoder_type == "target" and "ppg_encoder" in ckpt
        else f"{encoder_type}_encoder"
    )
    if key in ckpt:
        state_dict = ckpt[key]
    else:
        msd = ckpt["model_state_dict"]
        prefix = (
            "ppg_encoder."
            if encoder_type == "target"
            and any(k.startswith("ppg_encoder.") for k in msd)
            else f"{encoder_type}_encoder."
        )
        state_dict = {
            k[len(prefix):]: v for k, v in msd.items()
            if k.startswith(prefix)
        }
    return state_dict, key


def load_pretrained_encoder(
    checkpoint_path: str, model_config: ModelConfig,
    encoder_type: str, device: torch.device,
    in_channels: Optional[int] = None,
) -> SignalEncoder:
    """Load a pre-trained encoder from JEPA checkpoint."""
    encoder = build_encoder(model_config, in_channels=in_channels).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict, key = _select_pretrained_encoder_state(ckpt, encoder_type)

    first_conv_key = "cnn.conv_blocks.0.0.weight"
    if first_conv_key in state_dict:
        old_w = state_dict[first_conv_key]
        new_w = encoder.state_dict()[first_conv_key]
        if old_w.shape != new_w.shape and old_w.dim() == 3 and new_w.dim() == 3:
            if old_w.size(1) == 1 and new_w.size(1) > 1:
                state_dict[first_conv_key] = old_w.repeat(1, new_w.size(1), 1) / new_w.size(1)
                print(f"[Encoder] Adapted first conv 1ch -> {new_w.size(1)}ch")
            elif new_w.size(1) == 1:
                state_dict[first_conv_key] = old_w.mean(dim=1, keepdim=True)
                print(f"[Encoder] Adapted first conv {old_w.size(1)}ch -> 1ch")

    encoder.load_state_dict(state_dict, strict=True)
    print(f"Loaded {key} for {encoder_type} role from {checkpoint_path}")
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

def pairwise_auc_margin_loss(logits: torch.Tensor, targets: torch.Tensor,
                             margin: float = 0.2) -> torch.Tensor:
    """Smooth pairwise ranking surrogate for patient-level ROC AUC."""
    positives = logits[targets > 0.5]
    negatives = logits[targets <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.sum() * 0.0
    score_diff = positives[:, None] - negatives[None, :]
    return F.softplus(float(margin) - score_diff).mean()


def compute_multidisease_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion,
    focus_label_index: int = 4,
    focus_loss_weight: float = 0.0,
    focus_pos_weight: Optional[torch.Tensor] = None,
    focus_auc_loss_weight: float = 0.0,
    focus_auc_margin: float = 0.2,
    base_loss: Optional[torch.Tensor] = None,
    return_components: bool = False,
):
    """Compose the shared multi-disease objective used by fine-tuning and feedback."""
    base = criterion(logits, targets) if base_loss is None else base_loss
    total = base
    zero = logits.sum() * 0.0
    focus_bce = zero
    focus_auc = zero
    valid_focus = 0 <= int(focus_label_index) < logits.size(1)

    if valid_focus and focus_loss_weight > 0:
        focus_logits = logits[:, focus_label_index].float()
        focus_targets = targets[:, focus_label_index].float()
        focus_bce = F.binary_cross_entropy_with_logits(
            focus_logits, focus_targets, pos_weight=focus_pos_weight
        )
        total = total + float(focus_loss_weight) * focus_bce

    if valid_focus and focus_auc_loss_weight > 0:
        focus_auc = pairwise_auc_margin_loss(
            logits[:, focus_label_index].float(),
            targets[:, focus_label_index].float(),
            margin=focus_auc_margin,
        )
        total = total + float(focus_auc_loss_weight) * focus_auc

    if not return_components:
        return total
    return total, {
        "base": base,
        "focus_bce": focus_bce,
        "focus_auc": focus_auc,
        "total": total,
    }


def train_epoch(model, dataloader, optimizer, criterion, device,
                scheduler=None, sched_mode="epoch", is_dual=False,
                distill_mode=False, ecg_encoder=None,
                proj_ppg=None, proj_ecg=None,
                ecg_loader=None, distill_lambda=0.5,
                cotrain_mode=False, ecg_model=None, classifier=None,
                multilabel=False, focus_label_index: int = 4,
                focus_loss_weight: float = 0.0,
                focus_pos_weight: Optional[torch.Tensor] = None,
                focus_auc_loss_weight: float = 0.0,
                focus_auc_margin: float = 0.2,
                use_amp: bool = False,
                scaler=None):
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
            ecg = ecg.to(device, non_blocking=True)
            ppg = ppg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(ecg, ppg)
            loss = criterion(logits, labels)
        else:
            if len(batch) >= 3:
                x, labels, *_ = batch
            else:
                x, labels = batch
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
                ex = ex.to(device, non_blocking=True)
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
                ex = ex.to(device, non_blocking=True)
                elabels = elabels.to(device, non_blocking=True)
                ecg_pooled, _ = ecg_encoder(ex)
                ecg_logits = classifier(ecg_pooled)
                cls_loss_ecg = criterion(ecg_logits, elabels)
                loss = cls_loss_ppg + cls_loss_ecg
            else:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(use_amp and device.type == "cuda"),
                ):
                    logits = model(x)
                loss = criterion(
                    logits.float() if multilabel else logits,
                    labels.float() if multilabel else labels,
                )

        if multilabel:
            loss = compute_multidisease_objective(
                logits,
                labels,
                criterion,
                focus_label_index=focus_label_index,
                focus_loss_weight=focus_loss_weight,
                focus_pos_weight=focus_pos_weight,
                focus_auc_loss_weight=focus_auc_loss_weight,
                focus_auc_margin=focus_auc_margin,
                base_loss=loss,
            )

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print("[Warn] non-finite loss detected; skipping this batch")
            continue
        all_params = list(model.parameters()) + (list(proj_ppg.parameters()) if distill_mode else [])
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
            ecg = ecg.to(device, non_blocking=True)
            ppg = ppg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(ecg, ppg)
            uids = batch[3] if len(batch) >= 4 else None
        else:
            if len(batch) >= 3:
                x, labels, *rest = batch
                uids = rest[0] if rest else None
            else:
                x, labels = batch
                uids = None
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
                        thresholds: Optional[np.ndarray] = None,
                        use_amp: bool = False):
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
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            logits = model(x)
        loss = criterion(logits.float(), labels.float())
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


def _threshold_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float((y_true == y_pred).mean())
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "f1":
        return float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0))
    return float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))


def tune_multilabel_thresholds_recall_floor(
    labels: np.ndarray,
    probs: np.ndarray,
    recall_floor: float = 0.60,
    opt_metric: str = "f05",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
) -> np.ndarray:
    """Tune thresholds with a minimum recall constraint per label."""
    grid = np.linspace(min_threshold, max_threshold, 91)
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)

    for c in range(labels.shape[1]):
        y_true = labels[:, c]
        if len(np.unique(y_true)) < 2:
            continue

        candidates = []
        fallback = (-1.0, -1.0, 0.5)  # recall, score, threshold
        for thr in grid:
            y_pred = (probs[:, c] >= thr).astype(np.float32)
            recall = recall_score(y_true, y_pred, zero_division=0)
            score = _threshold_score(y_true, y_pred, opt_metric)
            if recall >= recall_floor:
                candidates.append((score, thr))
            if recall > fallback[0] or (recall == fallback[0] and score > fallback[1]):
                fallback = (recall, score, float(thr))

        if candidates:
            best_score, best_thr = max(candidates, key=lambda x: x[0])
            thresholds[c] = float(best_thr)
        else:
            thresholds[c] = fallback[2]

    return thresholds


def tune_thresholds_from_config(labels: np.ndarray, probs: np.ndarray,
                                train_config: TrainConfig) -> np.ndarray:
    if train_config.threshold_strategy == "recall_floor":
        recall_thresholds = tune_multilabel_thresholds_recall_floor(
            labels, probs,
            recall_floor=train_config.threshold_recall_floor,
            opt_metric=train_config.threshold_opt_metric,
        )
        if train_config.threshold_recall_floor_all_labels:
            return recall_thresholds
        thresholds = tune_multilabel_thresholds(
            labels, probs, beta=train_config.threshold_beta,
        )
        focus_idx = train_config.chd_label_index
        if 0 <= focus_idx < thresholds.size:
            thresholds[focus_idx] = recall_thresholds[focus_idx]
        return thresholds
    return tune_multilabel_thresholds(
        labels, probs, beta=train_config.threshold_beta,
    )


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
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    configure_cuda_performance(device, config.train)
    use_amp = bool(
        device.type == "cuda"
        and getattr(config.train, "downstream_use_amp", True)
    )
    print(f"[AMP] downstream={'on' if use_amp else 'off'}")
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
    downstream_in_channels = (
        2 if multilabel and config.data.multidisease_channel == "both"
        else config.model.in_channels
    )
    if multilabel:
        print(
            f"[MultiDisease] channel={config.data.multidisease_channel} "
            f"in_channels={downstream_in_channels} "
            f"patient_mil={config.data.multidisease_patient_mil} "
            f"multiscale={config.data.multidisease_use_multiscale or config.model.use_multiscale}"
        )
    use_multidisease_dual_stream = bool(
        multilabel
        and config.data.multidisease_patient_mil
        and config.data.multidisease_channel == "both"
        and config.data.multidisease_dual_stream
    )

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
    split_provenance = None
    if multilabel:
        split_file = getattr(
            config.data,
            "multidisease_split_file",
            getattr(config.data, "multidisease_development_split", ""),
        )
        split_provenance = multidisease_split_provenance(
            split_file, config.data.multidisease_dir
        )
        metadata = split_provenance["metadata"]
        split_line = (
            f"[DataSplit] file={split_provenance['filename']} "
            f"sha256={split_provenance['sha256']} "
            f"patients={metadata.get('patient_counts', {})} "
            f"files={metadata.get('file_counts', {})}"
        )
        print(split_line)
        log_fh.write(split_line + "\n")
        log_fh.flush()

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
    if not use_dual and not use_multidisease_dual_stream:
        encoder_role = "target"
        if (
            multilabel
            and str(config.data.multidisease_channel)
            == str(config.data.multidisease_ecg_channel)
        ):
            encoder_role = "context"
        if multilabel:
            modality = "PPG" if encoder_role == "target" else "ECG"
            print(f"[Model] Single-stream {modality}: {encoder_role}_encoder")
        encoder = load_pretrained_encoder(
            checkpoint_path, config.model, encoder_role, device,
            in_channels=downstream_in_channels,
        )

    # Build classifier (skip if dual-channel already created above)
    if use_multidisease_dual_stream:
        print(
            "[Model] Dual-stream patient MIL: "
            "ECG=context_encoder + PPG=target_encoder + disease-conditioned attention"
        )
        ecg_encoder = load_pretrained_encoder(
            checkpoint_path, config.model, "context", device, in_channels=1,
        )
        ppg_encoder = load_pretrained_encoder(
            checkpoint_path, config.model, "target", device, in_channels=1,
        )
        model = DualStreamPatientMILClassifier(
            ecg_encoder=ecg_encoder,
            ppg_encoder=ppg_encoder,
            encoder_dim=config.model.transformer_dim,
            num_classes=num_classes,
            use_multiscale=(
                config.data.multidisease_use_multiscale or config.model.use_multiscale
            ),
            encoder_chunk_size=config.data.multidisease_mil_encoder_chunk_size,
            ppg_channel=config.data.multidisease_ppg_channel,
            ecg_channel=config.data.multidisease_ecg_channel,
        ).to(device)
    elif not use_dual and not use_distill and not use_cotrain:
        # Pure PPG single-channel
        use_multiscale_head = config.model.use_multiscale or (
            multilabel and config.data.multidisease_use_multiscale
        )
        if multilabel and config.data.multidisease_patient_mil:
            print("[Model] Patient-level MIL head"
                  f" (multiscale={use_multiscale_head}, "
                  f"encoder_chunk_size={config.data.multidisease_mil_encoder_chunk_size})")
            model = PatientMILClassifier(
                encoder=encoder,
                encoder_dim=config.model.transformer_dim,
                num_classes=num_classes,
                use_multiscale=use_multiscale_head,
                encoder_chunk_size=config.data.multidisease_mil_encoder_chunk_size,
            ).to(device)
        elif use_multiscale_head:
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
            f"auc_loss_weight={config.train.chd_auc_loss_weight} "
            f"best_metric={config.train.best_metric}"
        )

    # ── Phase 1: Linear Probe ──
    full_train_loader = train_loader
    full_encoder_chunk_size = getattr(model, "encoder_chunk_size", None)
    probe_train_loader = None

    n_probe = config.train.downstream_probe_epochs
    if use_distill or use_cotrain:
        n_probe = 1  # 快速初始化, 避免冻结下坍塌
    # dual-channel uses full probe (30 epochs) — 即使不稳定, FT能恢复
    if n_probe > 0:
        if multilabel and config.data.multidisease_patient_mil:
            probe_batch_size = max(
                config.train.multidisease_mil_batch_size,
                int(config.train.multidisease_probe_batch_size),
            )
            probe_train_loader = rebuild_train_loader(
                full_train_loader, probe_batch_size, config.train
            )
            train_loader = probe_train_loader
            model.encoder_chunk_size = max(
                1, int(config.train.multidisease_probe_encoder_chunk_size)
            )
            print(
                f"[Probe GPU] batch_size={probe_batch_size} "
                f"encoder_chunk_size={model.encoder_chunk_size} "
                f"steps/epoch={len(train_loader)}"
            )
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
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scheduler, sched_mode = build_scheduler(optimizer, config.train, probe_steps)

        for epoch in range(n_probe):
            reset_gpu_peak_memory(device)
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
                focus_auc_loss_weight=config.train.chd_auc_loss_weight if multilabel else 0.0,
                focus_auc_margin=config.train.chd_auc_margin,
                use_amp=use_amp,
                scaler=scaler,
            )
            eval_loader = val_loader if val_loader is not None else test_loader
            eval_name = "Val" if val_loader is not None else "Test"
            if multilabel:
                test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate_multilabel(
                    model, eval_loader, criterion, device, config.data.multidisease_labels,
                    use_amp=use_amp,
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
                        f"{eval_name} L={test_loss:.4f} Acc={test_acc:5.2f}% AUC={auc:.4f} "
                        f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} F0.5={f05:.4f}")
            if focus_auc is not None:
                log_line += f" CHD_AUC={focus_auc:.4f}"
            log_line += gpu_peak_memory_message(device)
            print(log_line)
            log_fh.write(log_line + "\n"); log_fh.flush()
    else:
        print("\n[Probe] Skipped → direct Full Fine-tune (signal aligned)")

    # ── Phase 2: Full Fine-tune ──
    if probe_train_loader is not None:
        train_loader = full_train_loader
        model.encoder_chunk_size = full_encoder_chunk_size
        del probe_train_loader
        print(
            f"[Full FT GPU] batch_size={train_loader.batch_size} "
            f"encoder_chunk_size={model.encoder_chunk_size} "
            f"steps/epoch={len(train_loader)}"
        )

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

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler, sched_mode = build_scheduler(optimizer, config.train, ft_steps)

    best_score = float("-inf")
    best_state = None
    no_improve = 0

    for epoch in range(ft_epochs):
        reset_gpu_peak_memory(device)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, sched_mode=sched_mode, is_dual=use_dual, distill_mode=use_distill, ecg_encoder=ecg_encoder, proj_ppg=proj_ppg, proj_ecg=proj_ecg, ecg_loader=ecg_train_loader, distill_lambda=distill_lambda,
            multilabel=multilabel,
            focus_label_index=focus_idx,
            focus_loss_weight=config.train.chd_focus_loss_weight if multilabel else 0.0,
            focus_pos_weight=focus_pos_weight,
            focus_auc_loss_weight=config.train.chd_auc_loss_weight if multilabel else 0.0,
            focus_auc_margin=config.train.chd_auc_margin,
            use_amp=use_amp,
            scaler=scaler,
        )
        eval_loader = val_loader if val_loader is not None else test_loader
        eval_name = "Val" if val_loader is not None else "Test"
        if multilabel:
            test_loss, test_acc, auc, auc_list, prec, rec, f1, f05, report, _, _, _ = evaluate_multilabel(
                model, eval_loader, criterion, device, config.data.multidisease_labels,
                use_amp=use_amp,
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
        log_line += gpu_peak_memory_message(device)
        print(log_line)
        log_fh.write(log_line + "\n"); log_fh.flush()

        if selected_metric > best_score:
            best_score = selected_metric
            best_state = {
                "epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                "val_acc": test_acc, "val_auc": auc, "val_f1": f1,
                "val_chd_auc": focus_auc, "val_best_metric": selected_metric,
                "best_metric": config.train.best_metric,
                "seed": config.seed,
                "multidisease_channel": (
                    config.data.multidisease_channel if multilabel else None
                ),
                "data_split": split_provenance,
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
                use_amp=use_amp,
            )
            thresholds = tune_thresholds_from_config(val_labels, val_probs, config.train)
            threshold_msg = (
                f"Tuned thresholds ({config.train.threshold_strategy}, "
                f"recall_floor={config.train.threshold_recall_floor}, "
                f"metric={config.train.threshold_opt_metric}): "
                f"{[round(float(t), 3) for t in thresholds]}"
            )
            print(threshold_msg)
            log_fh.write(threshold_msg + "\n")
            if best_state is not None:
                best_state["thresholds"] = thresholds.tolist()

        (_, test_acc, auc, auc_list,
         prec, rec, f1, f05, report, test_preds, test_labels, test_probs) = evaluate_multilabel(
            model, test_loader, criterion, device, config.data.multidisease_labels,
            thresholds=thresholds,
            use_amp=use_amp,
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
    parser.add_argument(
        "--multidisease_channel",
        choices=["both", "ppg", "ecg"],
        default=None,
        help="Multidisease modality ablation; default keeps config.py behavior",
    )
    parser.add_argument(
        "--multidisease_split",
        "--development_split",
        dest="multidisease_split",
        type=str,
        default=None,
        help=(
            "Exact patient-level train/val/test JSON manifest. Defaults to "
            "config.data.multidisease_split_file"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mil_batch_size", type=int, default=None,
        help="Patient-MIL patients per full fine-tune batch",
    )
    parser.add_argument(
        "--mil_chunk_size", type=int, default=None,
        help="Segments processed by each encoder call",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override downstream DataLoader worker count",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable downstream automatic mixed precision",
    )
    args = parser.parse_args()

    config = Config()
    config.seed = args.seed
    if args.multidisease_channel is not None:
        channel_map = {
            "both": "both",
            "ppg": str(config.data.multidisease_ppg_channel),
            "ecg": str(config.data.multidisease_ecg_channel),
        }
        config.data.multidisease_channel = channel_map[args.multidisease_channel]
        config.data.multidisease_dual_stream = args.multidisease_channel == "both"
    if args.multidisease_split is not None:
        config.data.multidisease_split_file = args.multidisease_split
    if args.mil_batch_size is not None:
        config.train.multidisease_mil_batch_size = args.mil_batch_size
    if args.mil_chunk_size is not None:
        config.data.multidisease_mil_encoder_chunk_size = args.mil_chunk_size
    if args.workers is not None:
        config.train.dataloader_workers = args.workers
    if args.no_amp:
        config.train.downstream_use_amp = False
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    train_downstream(config, args.checkpoint, args.dataset)
