"""Create a deterministic patient-level multilabel train/val/test split."""
import argparse
import json
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import Config
from dataset.data import multidisease_label_value


SPLIT_NAMES = ("train", "val", "test")
TASKAWARE_SPLIT_NAMES = ("feedback_train", "feedback_meta", "val", "test")


def uid_from_filename(filename: str) -> str:
    parts = filename.split("_")
    if len(parts) < 3 or parts[0] not in {"train", "test"}:
        raise ValueError(f"Unexpected multidisease filename: {filename}")
    return parts[1]


def target_split_sizes(num_patients: int, ratios: Sequence[float]) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.ndim != 1 or ratios.size < 2 or np.any(ratios <= 0):
        raise ValueError("ratios must contain at least two positive values")
    ratios = ratios / ratios.sum()
    raw = ratios * num_patients
    sizes = np.floor(raw).astype(np.int64)
    remainder = num_patients - int(sizes.sum())
    # Prefer test over val on an exact fractional tie.
    order = sorted(
        range(len(ratios)), key=lambda idx: (raw[idx] - sizes[idx], idx), reverse=True
    )
    for idx in order[:remainder]:
        sizes[idx] += 1
    return sizes


def iterative_multilabel_split(
    labels: np.ndarray,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> np.ndarray:
    """Assign multilabel patients while matching split sizes and label prevalence."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError("labels must have shape [patients, diseases]")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("multilabel targets must be binary")

    rng = np.random.default_rng(seed)
    split_sizes = target_split_sizes(labels.shape[0], ratios)
    normalized_ratios = split_sizes / split_sizes.sum()
    target_label_counts = normalized_ratios[:, None] * labels.sum(axis=0)[None, :]
    assigned_label_counts = np.zeros_like(target_label_counts)
    remaining_capacity = split_sizes.copy()
    assignments = np.full(labels.shape[0], -1, dtype=np.int64)
    unassigned = np.ones(labels.shape[0], dtype=bool)

    while True:
        remaining_counts = labels[unassigned].sum(axis=0)
        positive_labels = np.flatnonzero(remaining_counts > 0)
        if positive_labels.size == 0:
            break
        rarest_label = positive_labels[
            np.argmin(remaining_counts[positive_labels])
        ]
        candidates = np.flatnonzero(unassigned & (labels[:, rarest_label] == 1))
        rng.shuffle(candidates)

        for patient_idx in candidates:
            if not unassigned[patient_idx]:
                continue
            available_splits = np.flatnonzero(remaining_capacity > 0)
            primary_need = (
                target_label_counts[available_splits, rarest_label]
                - assigned_label_counts[available_splits, rarest_label]
            )
            best = available_splits[primary_need == primary_need.max()]
            if best.size > 1:
                patient_labels = labels[patient_idx]
                all_label_need = (
                    target_label_counts[best] - assigned_label_counts[best]
                ) @ patient_labels
                best = best[all_label_need == all_label_need.max()]
            if best.size > 1:
                capacities = remaining_capacity[best]
                best = best[capacities == capacities.max()]
            split_idx = int(rng.choice(best))
            assignments[patient_idx] = split_idx
            unassigned[patient_idx] = False
            remaining_capacity[split_idx] -= 1
            assigned_label_counts[split_idx] += labels[patient_idx]

    remaining_patients = np.flatnonzero(unassigned)
    rng.shuffle(remaining_patients)
    for patient_idx in remaining_patients:
        available_splits = np.flatnonzero(remaining_capacity > 0)
        capacities = remaining_capacity[available_splits]
        best = available_splits[capacities == capacities.max()]
        split_idx = int(rng.choice(best))
        assignments[patient_idx] = split_idx
        remaining_capacity[split_idx] -= 1

    if np.any(assignments < 0) or np.any(remaining_capacity != 0):
        raise RuntimeError("Failed to produce an exact patient split")
    return assignments


def discover_patient_files(data_dir: str) -> Dict[str, List[str]]:
    patient_files = defaultdict(list)
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".pkl") or not filename.startswith(("train_", "test_")):
            continue
        patient_files[uid_from_filename(filename)].append(filename)
    if not patient_files:
        raise RuntimeError(f"No train_*.pkl/test_*.pkl files found in {data_dir}")
    return dict(patient_files)


def load_patient_labels(
    data_dir: str,
    patient_files: Dict[str, List[str]],
    disease_labels: Sequence[str],
    validate_all_segments: bool = True,
    workers: int = 1,
) -> Tuple[List[str], np.ndarray]:
    uids = sorted(patient_files)
    targets = np.zeros((len(uids), len(disease_labels)), dtype=np.int64)

    def load_one(patient_idx: int):
        uid = uids[patient_idx]
        filenames = patient_files[uid]
        files_to_check = filenames if validate_all_segments else filenames[:1]
        expected = None
        for filename in files_to_check:
            with open(os.path.join(data_dir, filename), "rb") as f:
                sample = pickle.load(f)
            sample_uid = str(sample.get("uid", ""))
            if sample_uid != uid:
                raise ValueError(
                    f"UID mismatch: file={filename}, filename_uid={uid}, sample_uid={sample_uid}"
                )
            label_dict = sample.get("label")
            if not isinstance(label_dict, dict):
                raise ValueError(f"Missing label dictionary in {filename}")
            current = np.asarray(
                [
                    int(multidisease_label_value(label_dict, name))
                    for name in disease_labels
                ],
                dtype=np.int64,
            )
            if expected is None:
                expected = current
            elif not np.array_equal(current, expected):
                raise ValueError(
                    f"Inconsistent disease labels within UID {uid}: {filename}"
                )
        return patient_idx, expected

    patient_indices = range(len(uids))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(load_one, patient_indices)
            for completed, (patient_idx, expected) in enumerate(results, start=1):
                targets[patient_idx] = expected
                if completed % 500 == 0 or completed == len(uids):
                    print(f"[Labels] checked {completed}/{len(uids)} patients")
    else:
        for completed, patient_idx in enumerate(patient_indices, start=1):
            _, expected = load_one(patient_idx)
            targets[patient_idx] = expected
            if completed % 500 == 0 or completed == len(uids):
                print(f"[Labels] checked {completed}/{len(uids)} patients")
    return uids, targets


def build_manifest(
    patient_files: Dict[str, List[str]],
    uids: Sequence[str],
    labels: np.ndarray,
    assignments: np.ndarray,
    disease_labels: Sequence[str],
    ratios: Sequence[float],
    seed: int,
    label_validation: str,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> dict:
    split_names = tuple(split_names)
    if len(split_names) != len(ratios):
        raise ValueError("split_names and ratios must have the same length")
    split_files = {name: [] for name in split_names}
    split_patient_counts = {}
    split_label_counts = {}
    for split_idx, split_name in enumerate(split_names):
        patient_indices = np.flatnonzero(assignments == split_idx)
        split_patient_counts[split_name] = int(patient_indices.size)
        split_label_counts[split_name] = {
            name: int(labels[patient_indices, label_idx].sum())
            for label_idx, name in enumerate(disease_labels)
        }
        for patient_idx in patient_indices:
            split_files[split_name].extend(patient_files[uids[patient_idx]])
        split_files[split_name].sort()

    metadata = {
        "version": 1,
        "unit": "patient_uid",
        "seed": int(seed),
        "label_validation": label_validation,
        "ratios": {
            name: float(ratio) for name, ratio in zip(split_names, ratios)
        },
        "num_patients": len(uids),
        "num_files": int(sum(len(files) for files in patient_files.values())),
        "patient_counts": split_patient_counts,
        "file_counts": {name: len(files) for name, files in split_files.items()},
        "disease_labels": list(disease_labels),
        "positive_patient_counts": split_label_counts,
    }
    return {"metadata": metadata, **split_files}


def derive_downstream_manifest(taskaware_manifest: dict) -> dict:
    """Merge task-aware development roles into a sealed train/val/test split."""
    missing = [
        name for name in TASKAWARE_SPLIT_NAMES
        if not isinstance(taskaware_manifest.get(name), list)
    ]
    if missing:
        raise ValueError(
            "Task-aware manifest is missing list splits: " + ", ".join(missing)
        )

    metadata = taskaware_manifest.get("metadata", {})
    patient_counts = metadata.get("patient_counts", {})
    positive_counts = metadata.get("positive_patient_counts", {})
    disease_labels = list(metadata.get("disease_labels", []))
    if not disease_labels:
        raise ValueError("Task-aware manifest metadata has no disease_labels")

    file_sets = {
        name: set(taskaware_manifest[name])
        for name in TASKAWARE_SPLIT_NAMES
    }
    uid_sets = {
        name: {uid_from_filename(filename) for filename in files}
        for name, files in file_sets.items()
    }
    for left_index, left in enumerate(TASKAWARE_SPLIT_NAMES):
        for right in TASKAWARE_SPLIT_NAMES[left_index + 1:]:
            if file_sets[left] & file_sets[right]:
                raise ValueError(
                    f"Task-aware manifest has {left}/{right} file leakage"
                )
            if uid_sets[left] & uid_sets[right]:
                raise ValueError(
                    f"Task-aware manifest has {left}/{right} patient leakage"
                )
    for name in TASKAWARE_SPLIT_NAMES:
        recorded_count = patient_counts.get(name)
        if recorded_count is not None and int(recorded_count) != len(uid_sets[name]):
            raise ValueError(
                f"Task-aware {name} patient count mismatch: "
                f"metadata={recorded_count}, actual={len(uid_sets[name])}"
            )

    train_files = sorted(
        taskaware_manifest["feedback_train"]
        + taskaware_manifest["feedback_meta"]
    )
    val_files = sorted(taskaware_manifest["val"])
    test_files = sorted(taskaware_manifest["test"])

    train_positive_counts = {}
    for label in disease_labels:
        train_positive_counts[label] = int(
            positive_counts.get("feedback_train", {}).get(label, 0)
            + positive_counts.get("feedback_meta", {}).get(label, 0)
        )

    downstream_metadata = {
        **metadata,
        "version": max(2, int(metadata.get("version", 1))),
        "split_purpose": "downstream_development",
        "source_roles": {
            "train": ["feedback_train", "feedback_meta"],
            "val": ["val"],
            "test": ["test"],
        },
        "ratios": {
            "train": float(metadata.get("ratios", {}).get("feedback_train", 0.0))
            + float(metadata.get("ratios", {}).get("feedback_meta", 0.0)),
            "val": float(metadata.get("ratios", {}).get("val", 0.0)),
            "test": float(metadata.get("ratios", {}).get("test", 0.0)),
        },
        "patient_counts": {
            "train": len(uid_sets["feedback_train"])
            + len(uid_sets["feedback_meta"]),
            "val": len(uid_sets["val"]),
            "test": len(uid_sets["test"]),
        },
        "file_counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "positive_patient_counts": {
            "train": train_positive_counts,
            "val": {
                label: int(positive_counts.get("val", {}).get(label, 0))
                for label in disease_labels
            },
            "test": {
                label: int(positive_counts.get("test", {}).get(label, 0))
                for label in disease_labels
            },
        },
    }
    return {
        "metadata": downstream_metadata,
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def save_manifest_atomic(manifest: dict, output_path: str):
    """Write a split manifest atomically so interrupted jobs cannot corrupt it."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=Config().data.multidisease_dir)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--downstream_output",
        default=None,
        help=(
            "Derived train/val/test output for --taskaware. Defaults to "
            "splits/multidisease_taskaware_downstream.json"
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--feedback_train_ratio", type=float, default=0.55)
    parser.add_argument("--feedback_meta_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument(
        "--taskaware",
        action="store_true",
        help="Create feedback_train/feedback_meta/val/test patient splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--representative_only",
        action="store_true",
        help="Read one segment per UID instead of validating every segment label",
    )
    args = parser.parse_args()

    config = Config()
    if args.taskaware:
        split_names = TASKAWARE_SPLIT_NAMES
        ratios = (
            args.feedback_train_ratio,
            args.feedback_meta_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        default_output = "splits/multidisease_taskaware_split.json"
    else:
        split_names = SPLIT_NAMES
        ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
        default_output = "splits/multidisease_patient_split.json"
    patient_files = discover_patient_files(args.data_dir)
    uids, labels = load_patient_labels(
        args.data_dir,
        patient_files,
        config.data.multidisease_labels,
        validate_all_segments=not args.representative_only,
        workers=max(1, args.workers),
    )
    assignments = iterative_multilabel_split(labels, ratios=ratios, seed=args.seed)
    manifest = build_manifest(
        patient_files,
        uids,
        labels,
        assignments,
        config.data.multidisease_labels,
        ratios,
        args.seed,
        "representative_segment" if args.representative_only else "all_segments",
        split_names=split_names,
    )

    output_path = os.path.abspath(args.output or default_output)
    save_manifest_atomic(manifest, output_path)
    print(json.dumps(manifest["metadata"], ensure_ascii=False, indent=2))
    print(f"Saved patient-disjoint split to {output_path}")
    if args.taskaware:
        downstream_manifest = derive_downstream_manifest(manifest)
        downstream_output = os.path.abspath(
            args.downstream_output
            or "splits/multidisease_taskaware_downstream.json"
        )
        downstream_manifest["metadata"]["derived_from"] = os.path.basename(
            output_path
        )
        save_manifest_atomic(downstream_manifest, downstream_output)
        print(
            "Saved sealed downstream train/val/test split to "
            f"{downstream_output}"
        )


if __name__ == "__main__":
    main()
