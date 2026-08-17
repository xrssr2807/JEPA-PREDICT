"""Summarize strict alpha=0.5 versus alpha=1.0 pretraining-seed repeats."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def alpha_tag(alpha):
    return f"a{round(float(alpha) * 100):03d}"


def load_metrics(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No patient predictions in {path}")

    labels = [
        column.split("::", 1)[1]
        for column in rows[0]
        if column.startswith("label::")
    ]
    aucs = []
    f1s = []
    chd_auc = math.nan
    for label in labels:
        y_true = np.asarray([int(float(row[f"label::{label}"])) for row in rows])
        y_prob = np.asarray([float(row[f"prob::{label}"]) for row in rows])
        y_pred = np.asarray([int(float(row[f"pred::{label}"])) for row in rows])
        if np.unique(y_true).size < 2:
            continue
        auc = float(roc_auc_score(y_true, y_prob))
        aucs.append(auc)
        f1s.append(float(f1_score(y_true, y_pred, zero_division=0)))
        if label == "冠心病":
            chd_auc = auc
    if not np.isfinite(chd_auc):
        raise ValueError(f"CHD label missing or invalid in {path}")
    return {
        "val_chd_auc": chd_auc,
        "val_macro_auc": float(np.mean(aucs)),
        "val_macro_f1": float(np.mean(f1s)),
        "num_patients": len(rows),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--pretrain_seeds", nargs="+", type=int, required=True)
    parser.add_argument("--downstream_seed", type=int, default=42)
    parser.add_argument("--downstream_prefix", default="outputs_direction_weight_repro")
    parser.add_argument("--paper_dir", required=True)
    parser.add_argument("--macro_noninferiority_margin", type=float, default=0.002)
    args = parser.parse_args()

    expected = {(float(alpha), seed) for alpha in args.alphas for seed in args.pretrain_seeds}
    runs = []
    missing = []
    for alpha, seed in sorted(expected):
        tag = alpha_tag(alpha)
        directory = (
            f"{args.downstream_prefix}_{tag}_preseed{seed}_ftseed{args.downstream_seed}"
        )
        predictions = os.path.join(directory, "validation_patient_predictions.csv")
        log_path = os.path.join(directory, "downstream_console.log")
        if not os.path.isfile(predictions) or not os.path.isfile(log_path):
            missing.append(directory)
            continue
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            log_text = handle.read()
        if "DEVELOPMENT COMPLETE (TEST SET SEALED)" not in log_text:
            missing.append(directory + " [not sealed-complete]")
            continue
        metrics = load_metrics(predictions)
        runs.append({
            "alpha": alpha,
            "pretrain_seed": seed,
            "downstream_seed": args.downstream_seed,
            **metrics,
            "test_status": "sealed",
            "predictions": os.path.abspath(predictions),
        })
    if missing:
        raise FileNotFoundError("Incomplete experiment matrix:\n" + "\n".join(missing))

    grouped = defaultdict(list)
    for row in runs:
        grouped[row["alpha"]].append(row)
    aggregate = []
    for alpha in sorted(grouped):
        group = grouped[alpha]
        chd_mean, chd_std = mean_std([row["val_chd_auc"] for row in group])
        macro_mean, macro_std = mean_std([row["val_macro_auc"] for row in group])
        f1_mean, f1_std = mean_std([row["val_macro_f1"] for row in group])
        aggregate.append({
            "alpha": alpha,
            "n_pretrain_seeds": len(group),
            "chd_auc_mean": chd_mean,
            "chd_auc_std": chd_std,
            "macro_auc_mean": macro_mean,
            "macro_auc_std": macro_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "test_status": "sealed",
        })

    by_key = {(row["alpha"], row["pretrain_seed"]): row for row in runs}
    paired = []
    for seed in args.pretrain_seeds:
        asymmetric = by_key[(0.5, seed)]
        symmetric = by_key[(1.0, seed)]
        paired.append({
            "pretrain_seed": seed,
            "chd_auc_delta_a050_minus_a100": asymmetric["val_chd_auc"] - symmetric["val_chd_auc"],
            "macro_auc_delta_a050_minus_a100": asymmetric["val_macro_auc"] - symmetric["val_macro_auc"],
            "macro_f1_delta_a050_minus_a100": asymmetric["val_macro_f1"] - symmetric["val_macro_f1"],
        })

    chd_deltas = [row["chd_auc_delta_a050_minus_a100"] for row in paired]
    macro_deltas = [row["macro_auc_delta_a050_minus_a100"] for row in paired]
    chd_positive = sum(delta > 0 for delta in chd_deltas)
    mean_chd_delta = float(np.mean(chd_deltas))
    mean_macro_delta = float(np.mean(macro_deltas))
    asymmetric_supported = chd_positive >= 2 and mean_chd_delta > 0
    macro_not_inferior = mean_macro_delta >= -args.macro_noninferiority_margin
    recommendation = (
        "alpha_0.5"
        if asymmetric_supported and macro_not_inferior
        else "alpha_1.0"
    )

    os.makedirs(args.paper_dir, exist_ok=True)
    write_csv(os.path.join(args.paper_dir, "direction_weight_independent_runs.csv"), runs)
    write_csv(os.path.join(args.paper_dir, "direction_weight_independent_aggregate.csv"), aggregate)
    write_csv(os.path.join(args.paper_dir, "direction_weight_independent_paired_deltas.csv"), paired)
    summary = {
        "scope": "validation-only independent pretraining-seed comparison",
        "alphas": args.alphas,
        "pretrain_seeds": args.pretrain_seeds,
        "downstream_seed": args.downstream_seed,
        "primary_rule": "alpha=0.5 wins CHD in at least 2/3 seeds and has positive mean paired delta",
        "co_primary_rule": f"mean macro-AUC delta >= -{args.macro_noninferiority_margin}",
        "chd_positive_seeds": chd_positive,
        "mean_chd_delta_a050_minus_a100": mean_chd_delta,
        "mean_macro_delta_a050_minus_a100": mean_macro_delta,
        "asymmetric_supported": asymmetric_supported,
        "macro_not_inferior": macro_not_inferior,
        "recommended_alpha": recommendation,
        "test_status": "sealed",
    }
    with open(os.path.join(args.paper_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.paper_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# 双向任务权重独立预训练三种子复现\n\n")
        handle.write(f"- CHD正向种子数：`{chd_positive}/{len(args.pretrain_seeds)}`\n")
        handle.write(f"- CHD平均配对差值（0.5-1.0）：`{mean_chd_delta:+.4f}`\n")
        handle.write(f"- Macro AUC平均配对差值（0.5-1.0）：`{mean_macro_delta:+.4f}`\n")
        handle.write(f"- 预设Macro非劣界值：`-{args.macro_noninferiority_margin:.4f}`\n")
        handle.write(f"- 推荐最终alpha：`{recommendation.replace('alpha_', '')}`\n")
        handle.write("- 测试集：封存，未参与选择。\n")
    print(
        "[Complete] direction-weight independent summary | "
        f"recommended={recommendation} | test_set_sealed=True"
    )


if __name__ == "__main__":
    main()
