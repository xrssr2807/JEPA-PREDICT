"""Summarize validation-only Transport constraint-composition ablations."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import torch


METRICS = ("val_macro_auc", "val_chd_auc", "val_macro_f1")


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _mean_std(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_metrics(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_status") != "sealed":
        raise ValueError(f"Test set was not sealed: {path}")
    validation = checkpoint.get("validation_metrics") or {}
    return {
        "val_macro_auc": float(
            validation.get("auc", checkpoint.get("val_auc", float("nan")))
        ),
        "val_chd_auc": float(checkpoint.get("val_chd_auc", float("nan"))),
        "val_macro_f1": float(
            validation.get("f1", checkpoint.get("val_f1", float("nan")))
        ),
        "split_sha256": (checkpoint.get("data_split") or {}).get("sha256", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_root", default="outputs_transport_constraint_ablation"
    )
    parser.add_argument(
        "--downstream_prefix", default="outputs_transport_constraint"
    )
    parser.add_argument(
        "--summary_dir", default="results/transport_constraint_ablation"
    )
    parser.add_argument(
        "--paper_dir",
        default=(
            "paper/ICASSP2027/03_experiments/"
            "P2_transport_constraint_ablation/results"
        ),
    )
    parser.add_argument("--downstream_seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "full",
            "static_delay",
            "fixed_prior",
            "zero_delay",
            "no_monotonic",
            "token_shuffled",
        ],
    )
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root)
    rows = []
    split_hashes = set()
    for seed in args.seeds:
        for mode in args.modes:
            manifest_path = (
                pretrain_root / f"{mode}_seed{seed}" / "checkpoint_manifest.txt"
            )
            output_dir = Path(
                f"{args.downstream_prefix}_{mode}_preseed{seed}"
                f"_ftseed{args.downstream_seed}"
            )
            downstream_path = output_dir / "downstream_multidisease_best.pt"
            for required in (manifest_path, downstream_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            manifest = _read_key_values(manifest_path)
            if manifest.get("transport_mode") != mode:
                raise ValueError(f"Transport mode mismatch: {manifest_path}")
            if int(manifest.get("optimization_seed", -1)) != seed:
                raise ValueError(f"Pretraining seed mismatch: {manifest_path}")
            metrics = _checkpoint_metrics(downstream_path)
            split_hashes.add(metrics.pop("split_sha256"))
            rows.append({
                "mode": mode,
                "pretrain_seed": seed,
                "downstream_seed": args.downstream_seed,
                **metrics,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "test_status": "sealed",
            })

    if "" in split_hashes or len(split_hashes) != 1:
        raise ValueError(
            f"Runs do not share one frozen downstream split: {split_hashes}"
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    aggregate_rows = []
    for mode in args.modes:
        mode_rows = grouped[mode]
        aggregate = {"mode": mode, "n": len(mode_rows)}
        for metric in METRICS:
            mean, std = _mean_std(row[metric] for row in mode_rows)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
        aggregate_rows.append(aggregate)

    indexed = {(row["mode"], row["pretrain_seed"]): row for row in rows}
    delta_rows = []
    for seed in args.seeds:
        control = indexed[("full", seed)]
        for mode in args.modes:
            if mode == "full":
                continue
            treatment = indexed[(mode, seed)]
            delta_rows.append({
                "mode": mode,
                "control": "full",
                "pretrain_seed": seed,
                "delta_val_macro_auc": (
                    treatment["val_macro_auc"] - control["val_macro_auc"]
                ),
                "delta_val_chd_auc": (
                    treatment["val_chd_auc"] - control["val_chd_auc"]
                ),
                "delta_val_macro_f1": (
                    treatment["val_macro_f1"] - control["val_macro_f1"]
                ),
                "test_status": "sealed",
            })

    summary_dir = Path(args.summary_dir)
    paper_dir = Path(args.paper_dir)
    runs_path = summary_dir / "transport_constraint_runs.csv"
    aggregate_path = summary_dir / "transport_constraint_aggregate.csv"
    delta_path = summary_dir / "transport_constraint_paired_deltas.csv"
    _write_csv(runs_path, rows)
    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(delta_path, delta_rows)

    paper_dir.mkdir(parents=True, exist_ok=True)
    for source in (runs_path, aggregate_path, delta_path):
        shutil.copy2(source, paper_dir / source.name)

    report = [
        "# Transport 约束组成消融自动汇总",
        "",
        "- 所有指标来自冻结验证集，测试集保持封存。",
        f"- 下游划分 SHA256：`{next(iter(split_hashes))}`",
        f"- 预训练种子：`{args.seeds}`",
        f"- 固定下游种子：`{args.downstream_seed}`",
        "",
        "| 模式 | N | Macro AUC | CHD AUC | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        report.append(
            f"| {row['mode']} | {row['n']} | "
            f"{row['val_macro_auc_mean']:.4f} ± "
            f"{row['val_macro_auc_std']:.4f} | "
            f"{row['val_chd_auc_mean']:.4f} ± "
            f"{row['val_chd_auc_std']:.4f} | "
            f"{row['val_macro_f1_mean']:.4f} ± "
            f"{row['val_macro_f1_std']:.4f} |"
        )
    report.extend([
        "",
        "论文解释时以 full 为对照。负差值表示移除或破坏该约束后性能下降。",
        "seed 42 只用于初筛；进入主文结论的保留变体需补跑 3407 和 2026。",
        "",
    ])
    (paper_dir / "实验结果自动汇总.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
