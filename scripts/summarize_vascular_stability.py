#!/usr/bin/env python3
"""Summarize the validation-only 2x2 vascular training study."""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def safe_metrics(y, p):
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    variants = ("baseline7", "no_chd_bias", "stable_training", "combined")
    rows = []
    for variant in variants:
        run_dir = os.path.join(args.study_dir, variant)
        with open(os.path.join(run_dir, "comparison_run_config.json"), "r", encoding="utf-8") as handle:
            config = json.load(handle)
        frame = pd.read_csv(os.path.join(run_dir, "validation_patient_predictions.csv"))
        for label in config["labels"]:
            y = frame[f"label::{label}"].to_numpy(dtype=int)
            p = frame[f"prob::{label}"].to_numpy(dtype=float)
            auroc, auprc = safe_metrics(y, p)
            rows.append({
                "variant": variant,
                "label": label,
                "patients": len(frame),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "auroc": auroc,
                "auprc": auprc,
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(args.output_dir, "validation_metrics.csv"), index=False)
    macro = metrics.groupby("variant", as_index=False).agg(
        macro_auroc=("auroc", "mean"), macro_auprc=("auprc", "mean")
    )
    chd = metrics[metrics["label"] == "冠心病"][["variant", "auroc", "auprc"]].rename(
        columns={"auroc": "chd_auroc", "auprc": "chd_auprc"}
    )
    stroke = metrics[metrics["label"] == "脑卒中（中风）"][["variant", "auroc", "auprc"]].rename(
        columns={"auroc": "stroke_auroc", "auprc": "stroke_auprc"}
    )
    summary = macro.merge(chd, on="variant").merge(stroke, on="variant")
    summary.to_csv(os.path.join(args.output_dir, "variant_summary.csv"), index=False)
    baseline = summary.set_index("variant").loc["baseline7"]
    effects = summary.copy()
    for column in ("macro_auroc", "macro_auprc", "chd_auroc", "chd_auprc", "stroke_auroc", "stroke_auprc"):
        effects[f"delta_{column}"] = effects[column] - baseline[column]
    effects.to_csv(os.path.join(args.output_dir, "effects_vs_baseline7.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
