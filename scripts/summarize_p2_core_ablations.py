"""Summarize sealed P2 core ablations from downstream checkpoints."""

import argparse
import csv
import math
import os
import shutil
import statistics
from collections import defaultdict

import torch


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
METRICS = ("val_macro_auc", "val_chd_auc", "val_f1")


def finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize(values):
    values = finite(values)
    if not values:
        return {
            "n": 0,
            "mean": "",
            "std": "",
            "ci95_low": "",
            "ci95_high": "",
        }
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    std = statistics.stdev(values)
    critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
    half_width = critical * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def checkpoint_row(path, experiment, seed):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validation = checkpoint.get("validation_metrics") or {}
    if checkpoint.get("test_status") != "sealed":
        raise ValueError(f"Paper ablation is not test-sealed: {path}")
    ablation = checkpoint.get("ablation_config") or {}
    return {
        "experiment": experiment,
        "seed": int(seed),
        "checkpoint": os.path.abspath(path),
        "experiment_id": checkpoint.get("experiment_id", ""),
        "encoder_init": checkpoint.get(
            "encoder_init", ablation.get("encoder_init", "")
        ),
        "patient_mil": checkpoint.get(
            "patient_mil", ablation.get("patient_mil", "")
        ),
        "multiscale": checkpoint.get(
            "multiscale", ablation.get("multiscale", "")
        ),
        "channel": checkpoint.get(
            "multidisease_channel", ablation.get("channel", "")
        ),
        "source_checkpoint": checkpoint.get("pretrained_checkpoint") or "",
        "split_sha256": (checkpoint.get("data_split") or {}).get("sha256", ""),
        "test_status": checkpoint.get("test_status", ""),
        "val_macro_auc": float(
            validation.get("auc", checkpoint.get("val_auc", float("nan")))
        ),
        "val_chd_auc": float(
            checkpoint.get("val_chd_auc", float("nan"))
        ),
        "val_f1": float(
            validation.get("f1", checkpoint.get("val_f1", float("nan")))
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value):
    if value == "":
        return ""
    return f"{float(value):.4f}"


def write_paper_summary(path, rows, aggregate_rows, delta_rows, split_hash):
    aggregate_by_experiment = {
        row["experiment"]: row for row in aggregate_rows
    }
    lines = [
        "# P2 核心消融实验自动汇总",
        "",
        "> 本文件由 `scripts/summarize_p2_core_ablations.py` 自动生成。",
        "> 所有结果均来自验证集，测试集保持封存。",
        "",
        f"- 数据划分 SHA256：`{split_hash}`",
        f"- 完成运行数：`{len(rows)}`",
        "- 主要指标：冠心病 AUC",
        "- 次要指标：宏平均 AUC、宏平均 F1",
        "",
        "## 聚合结果",
        "",
        "| 实验 | N | 宏平均 AUC | 冠心病 AUC | 宏平均 F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment in sorted(aggregate_by_experiment):
        row = aggregate_by_experiment[experiment]
        lines.append(
            "| {experiment} | {n} | {macro} ± {macro_std} | "
            "{chd} ± {chd_std} | {f1} ± {f1_std} |".format(
                experiment=experiment,
                n=row["val_macro_auc_n"],
                macro=format_metric(row["val_macro_auc_mean"]),
                macro_std=format_metric(row["val_macro_auc_std"]),
                chd=format_metric(row["val_chd_auc_mean"]),
                chd_std=format_metric(row["val_chd_auc_std"]),
                f1=format_metric(row["val_f1_mean"]),
                f1_std=format_metric(row["val_f1_std"]),
            )
        )

    lines.extend([
        "",
        "## 相对完整 Phase 2 的配对差值",
        "",
        "差值定义为“消融组 - 完整 Phase 2”；负值表示移除该组件后性能下降。",
        "",
        "| 实验 | Seed | Δ宏平均 AUC | Δ冠心病 AUC | Δ宏平均 F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in delta_rows:
        lines.append(
            f"| {row['experiment']} | {row['seed']} | "
            f"{row['delta_val_macro_auc']:.4f} | "
            f"{row['delta_val_chd_auc']:.4f} | "
            f"{row['delta_val_f1']:.4f} |"
        )
    lines.extend([
        "",
        "## 论文使用约束",
        "",
        "1. seed 42 仅用于初筛；保留进入主表的比较需要补跑 seed 3407 和 2026。",
        "2. 所有模型与阈值只能根据验证集选择。",
        "3. 在核心消融和基线冻结前，不得解封测试集。",
        "4. Transport-off 必须来自重新训练的无 Transport Phase 2 权重，不能用 Phase 1 代替。",
        "",
    ])
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", default="outputs_p2_core")
    parser.add_argument("--summary_dir", default="results/p2_core")
    parser.add_argument(
        "--paper_dir",
        default=(
            "paper/ICASSP2027/03_experiments/"
            "P2_core_ablations/results"
        ),
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Write a partial exploratory summary instead of failing.",
    )
    parser.add_argument("--control", default="phase2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "random_init",
            "phase0",
            "phase1",
            "phase2",
            "mil_off",
            "multiscale_off",
        ],
    )
    args = parser.parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)
    rows = []
    missing = []
    for experiment in args.experiments:
        for seed in args.seeds:
            path = os.path.join(
                f"{args.output_prefix}_{experiment}_seed{seed}",
                "downstream_multidisease_best.pt",
            )
            if not os.path.isfile(path):
                missing.append(path)
                continue
            rows.append(checkpoint_row(path, experiment, seed))
    if not rows:
        raise FileNotFoundError("No P2 downstream checkpoints were found")
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "P2 result set is incomplete. Missing checkpoints:\n"
            + "\n".join(missing)
        )

    split_hashes = {row["split_sha256"] for row in rows}
    if "" in split_hashes or len(split_hashes) != 1:
        raise ValueError(
            f"P2 runs do not share one non-empty split SHA256: {split_hashes}"
        )

    runs_path = os.path.join(args.summary_dir, "p2_core_runs.csv")
    write_csv(runs_path, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    aggregate_rows = []
    for experiment, experiment_rows in sorted(grouped.items()):
        aggregate = {"experiment": experiment}
        for metric in METRICS:
            result = summarize([row[metric] for row in experiment_rows])
            for key, value in result.items():
                aggregate[f"{metric}_{key}"] = value
        aggregate_rows.append(aggregate)
    aggregate_path = os.path.join(
        args.summary_dir, "p2_core_aggregate.csv"
    )
    write_csv(aggregate_path, aggregate_rows)

    indexed = {
        (row["experiment"], row["seed"]): row for row in rows
    }
    delta_rows = []
    for experiment in args.experiments:
        if experiment == args.control:
            continue
        for seed in args.seeds:
            treatment = indexed.get((experiment, seed))
            control = indexed.get((args.control, seed))
            if treatment is None or control is None:
                continue
            delta_rows.append({
                "experiment": experiment,
                "control": args.control,
                "seed": seed,
                "delta_val_macro_auc": (
                    treatment["val_macro_auc"] - control["val_macro_auc"]
                ),
                "delta_val_chd_auc": (
                    treatment["val_chd_auc"] - control["val_chd_auc"]
                ),
                "delta_val_f1": treatment["val_f1"] - control["val_f1"],
            })
    delta_path = os.path.join(
        args.summary_dir, "p2_core_paired_deltas.csv"
    )
    write_csv(delta_path, delta_rows)

    os.makedirs(args.paper_dir, exist_ok=True)
    for source in (runs_path, aggregate_path, delta_path):
        shutil.copy2(source, os.path.join(args.paper_dir, os.path.basename(source)))
    paper_summary_path = os.path.join(args.paper_dir, "实验结果自动汇总.md")
    write_paper_summary(
        paper_summary_path,
        rows,
        aggregate_rows,
        delta_rows,
        next(iter(split_hashes)),
    )

    print(
        f"[Summary] runs={len(rows)} missing={len(missing)} "
        f"split_sha256={next(iter(split_hashes))}"
    )
    print(f"[Summary] per-run:   {runs_path}")
    print(f"[Summary] aggregate: {aggregate_path}")
    print(f"[Summary] paired:    {delta_path}")
    print(f"[Archive] paper:     {args.paper_dir}")
    for path in missing:
        print(f"[Missing] {path}")


if __name__ == "__main__":
    main()
