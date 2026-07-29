"""Summarize sealed ICASSP baseline runs from downstream checkpoints."""

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

import torch


METRICS = ("val_macro_auc", "val_chd_auc", "val_f1")


def summarize(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"n": 0, "mean": "", "std": ""}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", default="outputs_p3")
    parser.add_argument(
        "--paper_dir",
        default="paper/ICASSP2027/03_experiments/P3_baselines/results",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["transformer_scratch", "resnet1d", "contrastive"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 3407, 2026]
    )
    args = parser.parse_args()

    rows = []
    missing = []
    split_hashes = set()
    for experiment in args.experiments:
        for seed in args.seeds:
            checkpoint_path = os.path.join(
                f"{args.output_prefix}_{experiment}_seed{seed}",
                "downstream_multidisease_best.pt",
            )
            predictions_path = os.path.join(
                f"{args.output_prefix}_{experiment}_seed{seed}",
                "validation_patient_predictions.csv",
            )
            if not os.path.isfile(checkpoint_path):
                missing.append(checkpoint_path)
                continue
            if not os.path.isfile(predictions_path):
                missing.append(predictions_path)
                continue
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("test_status") != "sealed":
                raise ValueError(
                    f"Baseline is not test-sealed: {checkpoint_path}"
                )
            validation = checkpoint.get("validation_metrics") or {}
            split_hash = (checkpoint.get("data_split") or {}).get(
                "sha256", ""
            )
            split_hashes.add(split_hash)
            ablation = checkpoint.get("ablation_config") or {}
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "encoder_arch": ablation.get("encoder_arch", ""),
                "encoder_init": ablation.get("encoder_init", ""),
                "split_sha256": split_hash,
                "test_status": checkpoint.get("test_status"),
                "val_macro_auc": float(
                    validation.get("auc", checkpoint.get("val_auc"))
                ),
                "val_chd_auc": float(checkpoint.get("val_chd_auc")),
                "val_f1": float(
                    validation.get("f1", checkpoint.get("val_f1"))
                ),
                "checkpoint": os.path.abspath(checkpoint_path),
                "patient_predictions": os.path.abspath(predictions_path),
            })

    if missing:
        raise FileNotFoundError(
            "P3 baseline result set is incomplete:\n" + "\n".join(missing)
        )
    if not rows:
        raise FileNotFoundError("No P3 baseline results were found")
    if "" in split_hashes or len(split_hashes) != 1:
        raise ValueError(
            f"Baselines do not share one split SHA256: {split_hashes}"
        )

    os.makedirs(args.paper_dir, exist_ok=True)
    write_csv(os.path.join(args.paper_dir, "p3_baseline_runs.csv"), rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    aggregate_rows = []
    for experiment, experiment_rows in sorted(grouped.items()):
        result = {"experiment": experiment}
        for metric in METRICS:
            metric_summary = summarize(
                row[metric] for row in experiment_rows
            )
            for name, value in metric_summary.items():
                result[f"{metric}_{name}"] = value
        aggregate_rows.append(result)
    write_csv(
        os.path.join(args.paper_dir, "p3_baseline_aggregate.csv"),
        aggregate_rows,
    )

    lines = [
        "# P3 对比实验自动汇总",
        "",
        "> 所有指标来自冻结验证集；测试集保持封存。",
        "",
        f"- 划分 SHA256：`{next(iter(split_hashes))}`",
        f"- 完成运行：`{len(rows)}/{len(args.experiments)*len(args.seeds)}`",
        "",
        "| 模型 | N | 宏平均 AUC | 冠心病 AUC | 宏平均 F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['experiment']} | {row['val_macro_auc_n']} | "
            f"{float(row['val_macro_auc_mean']):.4f} ± "
            f"{float(row['val_macro_auc_std']):.4f} | "
            f"{float(row['val_chd_auc_mean']):.4f} ± "
            f"{float(row['val_chd_auc_std']):.4f} | "
            f"{float(row['val_f1_mean']):.4f} ± "
            f"{float(row['val_f1_std']):.4f} |"
        )
    with open(
        os.path.join(args.paper_dir, "实验结果自动汇总.md"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print(
        f"[Complete] P3 baseline summary | runs={len(rows)} "
        f"| test_set_sealed=True"
    )


if __name__ == "__main__":
    main()
