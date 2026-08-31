"""Run a fair PPG-only JEPA downstream comparison on old and new datasets."""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from config import Config
import train_downstream as downstream
from dataset import data as dataset_data


COMMON_LABELS = [
    "高血压",
    "高血脂",
    "冠心病",
    "心律失常（房颤、频发早搏等）",
    "糖尿病",
    "颈动脉斑块",
]

VASCULAR7_LABELS = [
    *COMMON_LABELS,
    "脑卒中（中风）",
]


def configure_label_sources(label_schema):
    """Keep legacy six-label semantics while separating stroke in vascular7."""
    dataset_data.MULTIDISEASE_LABEL_SOURCES["高血脂"] = (
        "高脂血症（高胆固醇等）",
    )
    if label_schema == "vascular7":
        dataset_data.MULTIDISEASE_LABEL_SOURCES["其他疾病"] = (
            "下肢动脉硬化闭塞症",
        )
        dataset_data.MULTIDISEASE_LABEL_SOURCES["脑卒中（中风）"] = (
            "脑卒中",
            "中风",
        )
    else:
        dataset_data.MULTIDISEASE_LABEL_SOURCES["其他疾病"] = (
            "下肢动脉硬化闭塞症",
            "脑卒中（中风）",
        )
        dataset_data.MULTIDISEASE_LABEL_SOURCES.pop("脑卒中（中风）", None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--ft_epochs", type=int, default=30)
    parser.add_argument("--mil_batch_size", type=int, default=64)
    parser.add_argument("--mil_chunk_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--label_schema", choices=("common6", "vascular7"), default="common6"
    )
    parser.add_argument("--downstream_lr", type=float, default=5e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--sampler_mode", choices=("random", "multilabel_balanced"),
        default="random",
    )
    parser.add_argument("--sampler_exponent", type=float, default=0.5)
    parser.add_argument("--sampler_cap", type=float, default=4.0)
    parser.add_argument("--chd_focus_loss_weight", type=float, default=0.5)
    parser.add_argument("--chd_auc_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--best_metric", choices=("chd_auc", "macro_auc", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--best_metric_chd_alpha", type=float, default=0.7)
    parser.add_argument("--max_patients_per_split", type=int, default=0)
    parser.add_argument("--seal_test", action="store_true")
    return parser.parse_args()


def uid_from_filename(filename):
    parts = filename.split("_")
    if parts[0] in {"train", "val", "test"} and len(parts) >= 3:
        return parts[1]
    return parts[0]


def patient_subset(files, limit, seed):
    if limit <= 0:
        return sorted(files)
    by_uid = defaultdict(list)
    for filename in files:
        by_uid[uid_from_filename(filename)].append(filename)
    uids = sorted(by_uid)
    rng = random.Random(seed)
    rng.shuffle(uids)
    selected = set(uids[: min(limit, len(uids))])
    return sorted(
        filename
        for uid in selected
        for filename in by_uid[uid]
    )


def build_manifest(args, labels):
    if args.source_split:
        with open(args.source_split, "r", encoding="utf-8") as handle:
            source = json.load(handle)
        split_files = {
            name: list(source[name]) for name in ("train", "val", "test")
        }
        split_source = os.path.abspath(args.source_split)
    else:
        split_files = {"train": [], "val": [], "test": []}
        with os.scandir(args.data_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".pkl"):
                    continue
                prefix = entry.name.split("_", 1)[0]
                if prefix in split_files:
                    split_files[prefix].append(entry.name)
        split_source = "filename_prefix"

    for offset, name in enumerate(("train", "val", "test")):
        split_files[name] = patient_subset(
            split_files[name], args.max_patients_per_split, args.seed + offset
        )
        if not split_files[name]:
            raise ValueError(f"Empty split: {name}")

    uid_sets = {
        name: {uid_from_filename(filename) for filename in files}
        for name, files in split_files.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = uid_sets[left] & uid_sets[right]
        if overlap:
            raise ValueError(f"Patient leakage {left}/{right}: {len(overlap)}")

    manifest = {
        "metadata": {
            "version": 1,
            "dataset_id": args.dataset_id,
            "unit": "patient_uid",
            "seed": args.seed,
            "source_split": split_source,
            "disease_labels": labels,
            "patient_counts": {k: len(v) for k, v in uid_sets.items()},
            "file_counts": {k: len(v) for k, v in split_files.items()},
            "max_patients_per_split": args.max_patients_per_split,
        },
        **split_files,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "comparison_split.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return os.path.abspath(path), manifest


def flexible_split_loader(
    split_file,
    data_dir,
    available_files,
    expected_disease_labels=None,
):
    del available_files
    resolved = downstream.resolve_multidisease_split_file(split_file, data_dir)
    with open(resolved, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    labels = manifest.get("metadata", {}).get("disease_labels")
    if expected_disease_labels is not None and labels != list(expected_disease_labels):
        raise ValueError(
            f"Label schema mismatch: manifest={labels} expected={list(expected_disease_labels)}"
        )
    result = []
    uid_sets = {}
    for name in ("train", "val", "test"):
        files = manifest.get(name)
        if not isinstance(files, list) or not files:
            raise ValueError(f"Invalid or empty split: {name}")
        missing = [f for f in files if not os.path.isfile(os.path.join(data_dir, f))]
        if missing:
            raise FileNotFoundError(f"{name} has {len(missing)} missing files: {missing[:3]}")
        if len(files) != len(set(files)):
            raise ValueError(f"Duplicate files in {name}")
        uid_sets[name] = {uid_from_filename(f) for f in files}
        result.append(sorted(files))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = uid_sets[left] & uid_sets[right]
        if overlap:
            raise ValueError(f"Patient leakage {left}/{right}: {len(overlap)}")
    print(
        "[ComparisonSplit] "
        + " | ".join(
            f"{name}={len(manifest[name])} files/{len(uid_sets[name])} patients"
            for name in ("train", "val", "test")
        )
    )
    return result[0], result[1], result[2], resolved


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.source_split:
        args.source_split = os.path.abspath(args.source_split)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    labels = VASCULAR7_LABELS if args.label_schema == "vascular7" else COMMON_LABELS
    split_path, manifest = build_manifest(args, labels)
    downstream.load_multidisease_split_manifest = flexible_split_loader
    configure_label_sources(args.label_schema)

    config = Config()
    config.seed = args.seed
    config.output_dir = args.output_dir
    config.data.multidisease_dir = args.data_dir
    config.data.multidisease_split_file = split_path
    config.data.multidisease_labels = list(labels)
    config.data.multidisease_channel = "0"
    config.data.multidisease_dual_stream = False
    config.data.multidisease_ppg_channel = 0
    config.data.multidisease_patient_mil = True
    config.data.multidisease_use_multiscale = True
    config.data.multidisease_mil_encoder_chunk_size = args.mil_chunk_size
    config.model.use_multiscale = False
    config.model.downstream_shared_private_head = "off"
    config.train.downstream_probe_epochs = args.probe_epochs
    config.train.downstream_epochs = args.probe_epochs + args.ft_epochs
    config.train.multidisease_mil_batch_size = args.mil_batch_size
    config.train.multidisease_probe_batch_size = args.mil_batch_size
    config.train.multidisease_probe_encoder_chunk_size = args.mil_chunk_size
    config.train.dataloader_workers = args.workers
    config.train.chd_label_index = labels.index("冠心病")
    config.train.downstream_lr = args.downstream_lr
    config.train.downstream_gradient_accumulation_steps = args.grad_accum_steps
    config.train.multidisease_sampler_mode = args.sampler_mode
    config.train.multidisease_sampler_exponent = args.sampler_exponent
    config.train.multidisease_sampler_weight_cap = args.sampler_cap
    config.train.chd_focus_loss_weight = args.chd_focus_loss_weight
    config.train.chd_auc_loss_weight = args.chd_auc_loss_weight
    config.train.best_metric = args.best_metric
    config.train.best_metric_chd_alpha = args.best_metric_chd_alpha

    run_config = {
        "dataset_id": args.dataset_id,
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "labels": labels,
        "label_schema": args.label_schema,
        "seed": args.seed,
        "probe_epochs": args.probe_epochs,
        "ft_epochs": args.ft_epochs,
        "mil_batch_size": args.mil_batch_size,
        "mil_chunk_size": args.mil_chunk_size,
        "workers": args.workers,
        "downstream_lr": args.downstream_lr,
        "grad_accum_steps": args.grad_accum_steps,
        "sampler_mode": args.sampler_mode,
        "sampler_exponent": args.sampler_exponent,
        "sampler_cap": args.sampler_cap,
        "chd_focus_loss_weight": args.chd_focus_loss_weight,
        "chd_auc_loss_weight": args.chd_auc_loss_weight,
        "best_metric": args.best_metric,
        "best_metric_chd_alpha": args.best_metric_chd_alpha,
        "seal_test": args.seal_test,
        "split_metadata": manifest["metadata"],
    }
    with open(os.path.join(args.output_dir, "comparison_run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    downstream.train_downstream(
        config,
        args.checkpoint,
        "multidisease",
        seal_test=args.seal_test,
        encoder_init="pretrained",
        experiment_id=args.dataset_id,
    )


if __name__ == "__main__":
    main()
