"""Summarize paired Transport on/off runs across pre-training seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


CHD_LABEL = "冠心病"


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _split_content_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = {
        "train_files": sorted(payload["train_files"]),
        "val_files": sorted(payload["val_files"]),
        "train_uids": sorted(payload.get("train_uids", [])),
        "val_uids": sorted(payload.get("val_uids", [])),
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_predictions(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["uid"])
    if not rows:
        raise ValueError(f"No patient predictions in {path}")

    label_names = [
        column.split("::", 1)[1]
        for column in rows[0]
        if column.startswith("label::")
    ]
    uids = [row["uid"] for row in rows]
    split_roles = {row["split"] for row in rows}
    if split_roles != {"val"}:
        raise ValueError(f"Expected validation-only predictions: {path}")
    labels = np.asarray(
        [[float(row[f"label::{name}"]) for name in label_names] for row in rows],
        dtype=np.float64,
    )
    probabilities = np.asarray(
        [[float(row[f"prob::{name}"]) for name in label_names] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    return uids, label_names, labels, probabilities


def _auc_pair(labels: np.ndarray, probabilities: np.ndarray, chd_index: int):
    aucs = []
    for class_index in range(labels.shape[1]):
        target = labels[:, class_index]
        if np.unique(target).size < 2:
            continue
        aucs.append(roc_auc_score(target, probabilities[:, class_index]))
    if not aucs:
        raise ValueError("No label has both positive and negative patients")
    return float(np.mean(aucs)), float(
        roc_auc_score(labels[:, chd_index], probabilities[:, chd_index])
    )


def _paired_bootstrap(
    labels: np.ndarray,
    on_probabilities: np.ndarray,
    off_probabilities: np.ndarray,
    chd_index: int,
    iterations: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    patient_count = len(labels)
    macro_deltas = []
    chd_deltas = []
    for _ in range(iterations):
        indices = rng.integers(0, patient_count, patient_count)
        sampled_labels = labels[indices]
        try:
            on_macro, on_chd = _auc_pair(
                sampled_labels, on_probabilities[indices], chd_index
            )
            off_macro, off_chd = _auc_pair(
                sampled_labels, off_probabilities[indices], chd_index
            )
        except ValueError:
            continue
        macro_deltas.append(on_macro - off_macro)
        chd_deltas.append(on_chd - off_chd)
    if not macro_deltas:
        raise RuntimeError("All patient bootstrap samples were invalid")
    return {
        "delta_macro_auc_ci95_low": float(np.percentile(macro_deltas, 2.5)),
        "delta_macro_auc_ci95_high": float(np.percentile(macro_deltas, 97.5)),
        "delta_chd_auc_ci95_low": float(np.percentile(chd_deltas, 2.5)),
        "delta_chd_auc_ci95_high": float(np.percentile(chd_deltas, 97.5)),
        "valid_bootstrap_iterations": len(macro_deltas),
    }


def _mean_std(values):
    values = [float(value) for value in values]
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def _seed_t_interval(values):
    values = [float(value) for value in values]
    mean, std = _mean_std(values)
    if len(values) < 2:
        return mean, mean
    t_critical = {
        2: 12.7062047364,
        3: 4.3026527299,
        4: 3.1824463053,
        5: 2.7764451052,
    }.get(len(values), 1.96)
    half_width = t_critical * std / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_root", default="outputs_transport_pretrain_seed_study"
    )
    parser.add_argument(
        "--downstream_prefix",
        default="outputs_transport_pretrain_seed_study",
    )
    parser.add_argument(
        "--paper_dir",
        default=(
            "paper/ICASSP2027/03_experiments/"
            "P2_transport_pretrain_seeds/results"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    parser.add_argument("--downstream_seed", type=int, default=42)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root)
    paper_dir = Path(args.paper_dir)
    run_rows = []
    paired_rows = []
    split_hashes = set()

    for pretrain_seed in args.seeds:
        mode_data = {}
        for mode in ("on", "off"):
            study_dir = pretrain_root / f"{mode}_seed{pretrain_seed}"
            checkpoint_manifest = study_dir / "checkpoint_manifest.txt"
            split_manifest = study_dir / "pretrain_split.json"
            output_dir = Path(
                f"{args.downstream_prefix}_{mode}_preseed{pretrain_seed}"
                f"_ftseed{args.downstream_seed}"
            )
            predictions = output_dir / "validation_patient_predictions.csv"
            for required in (
                checkpoint_manifest,
                split_manifest,
                predictions,
            ):
                if not required.is_file():
                    raise FileNotFoundError(required)

            checkpoint_values = _read_key_values(checkpoint_manifest)
            if int(checkpoint_values["optimization_seed"]) != pretrain_seed:
                raise ValueError(
                    f"Optimization seed mismatch in {checkpoint_manifest}"
                )
            if int(checkpoint_values["data_split_seed"]) != 42:
                raise ValueError(
                    f"Pre-training split is not frozen to seed 42: "
                    f"{checkpoint_manifest}"
                )
            expected_transport = mode == "on"
            if (
                checkpoint_values["transport_enabled"].lower() == "true"
            ) != expected_transport:
                raise ValueError(f"Transport mode mismatch: {checkpoint_manifest}")

            split_hash = _split_content_hash(split_manifest)
            split_hashes.add(split_hash)
            uids, label_names, labels, probabilities = _read_predictions(
                predictions
            )
            if CHD_LABEL not in label_names:
                raise ValueError(f"{CHD_LABEL} is missing from {predictions}")
            chd_index = label_names.index(CHD_LABEL)
            macro_auc, chd_auc = _auc_pair(
                labels, probabilities, chd_index
            )
            row = {
                "pretrain_seed": pretrain_seed,
                "mode": mode,
                "downstream_seed": args.downstream_seed,
                "patients": len(uids),
                "macro_auc": macro_auc,
                "chd_auc": chd_auc,
                "pretrain_split_content_sha256": split_hash,
                "checkpoint_sha256": checkpoint_values["checkpoint_sha256"],
                "checkpoint": checkpoint_values["checkpoint"],
                "predictions": str(predictions.resolve()),
                "test_status": "sealed",
            }
            run_rows.append(row)
            mode_data[mode] = {
                "uids": uids,
                "labels": labels,
                "probabilities": probabilities,
                "macro_auc": macro_auc,
                "chd_auc": chd_auc,
                "chd_index": chd_index,
            }

        if mode_data["on"]["uids"] != mode_data["off"]["uids"]:
            raise ValueError(
                f"Patient UID mismatch for pre-training seed {pretrain_seed}"
            )
        if not np.array_equal(
            mode_data["on"]["labels"], mode_data["off"]["labels"]
        ):
            raise ValueError(
                f"Patient label mismatch for pre-training seed {pretrain_seed}"
            )
        bootstrap = _paired_bootstrap(
            mode_data["on"]["labels"],
            mode_data["on"]["probabilities"],
            mode_data["off"]["probabilities"],
            mode_data["on"]["chd_index"],
            args.bootstrap_iterations,
            args.bootstrap_seed + pretrain_seed,
        )
        paired_rows.append({
            "pretrain_seed": pretrain_seed,
            "downstream_seed": args.downstream_seed,
            "patients": len(mode_data["on"]["uids"]),
            "delta_macro_auc_on_minus_off": (
                mode_data["on"]["macro_auc"] - mode_data["off"]["macro_auc"]
            ),
            "delta_chd_auc_on_minus_off": (
                mode_data["on"]["chd_auc"] - mode_data["off"]["chd_auc"]
            ),
            **bootstrap,
            "test_status": "sealed",
        })

    if len(split_hashes) != 1:
        raise ValueError(
            "Pre-training patient split differs across modes or seeds: "
            f"{sorted(split_hashes)}"
        )

    aggregate_rows = []
    for metric in ("macro_auc", "chd_auc"):
        on_values = [
            row[metric] for row in run_rows if row["mode"] == "on"
        ]
        off_values = [
            row[metric] for row in run_rows if row["mode"] == "off"
        ]
        delta_key = f"delta_{metric}_on_minus_off"
        deltas = [row[delta_key] for row in paired_rows]
        on_mean, on_std = _mean_std(on_values)
        off_mean, off_std = _mean_std(off_values)
        delta_mean, delta_std = _mean_std(deltas)
        delta_low, delta_high = _seed_t_interval(deltas)
        aggregate_rows.append({
            "metric": metric,
            "pretrain_seeds": len(args.seeds),
            "on_mean": on_mean,
            "on_std": on_std,
            "off_mean": off_mean,
            "off_std": off_std,
            "delta_on_minus_off_mean": delta_mean,
            "delta_on_minus_off_std": delta_std,
            "delta_seed_t_ci95_low": delta_low,
            "delta_seed_t_ci95_high": delta_high,
            "downstream_seed": args.downstream_seed,
            "pretrain_split_content_sha256": next(iter(split_hashes)),
            "test_status": "sealed",
        })

    _write_csv(paper_dir / "transport_pretrain_seed_runs.csv", run_rows)
    _write_csv(
        paper_dir / "transport_pretrain_seed_paired_bootstrap.csv",
        paired_rows,
    )
    _write_csv(
        paper_dir / "transport_pretrain_seed_aggregate.csv",
        aggregate_rows,
    )

    chd_summary = next(
        row for row in aggregate_rows if row["metric"] == "chd_auc"
    )
    macro_summary = next(
        row for row in aggregate_rows if row["metric"] == "macro_auc"
    )
    report = [
        "# Transport 跨预训练种子配对实验",
        "",
        f"- 预训练种子：`{args.seeds}`",
        f"- 固定预训练患者划分 seed：`42`",
        f"- 固定下游微调 seed：`{args.downstream_seed}`",
        f"- 患者级 Bootstrap：`{args.bootstrap_iterations}` 次",
        "- 测试集：`sealed`",
        "",
        "| 指标 | Transport on | Transport off | on - off | seed 级 95% t CI |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Macro AUC | {macro_summary['on_mean']:.4f} ± "
            f"{macro_summary['on_std']:.4f} | "
            f"{macro_summary['off_mean']:.4f} ± "
            f"{macro_summary['off_std']:.4f} | "
            f"{macro_summary['delta_on_minus_off_mean']:+.4f} | "
            f"[{macro_summary['delta_seed_t_ci95_low']:+.4f}, "
            f"{macro_summary['delta_seed_t_ci95_high']:+.4f}] |"
        ),
        (
            f"| CHD AUC | {chd_summary['on_mean']:.4f} ± "
            f"{chd_summary['on_std']:.4f} | "
            f"{chd_summary['off_mean']:.4f} ± "
            f"{chd_summary['off_std']:.4f} | "
            f"{chd_summary['delta_on_minus_off_mean']:+.4f} | "
            f"[{chd_summary['delta_seed_t_ci95_low']:+.4f}, "
            f"{chd_summary['delta_seed_t_ci95_high']:+.4f}] |"
        ),
        "",
        "患者级置信区间见 `transport_pretrain_seed_paired_bootstrap.csv`。",
        "该实验只使用冻结验证集，不得据此解封测试集。",
    ]
    (paper_dir / "实验结果自动汇总.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
