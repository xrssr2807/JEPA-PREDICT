"""Aggregate final-test metrics from frozen official-foundation-model heads."""

import argparse
import csv
import glob
import json
import os
import re
import statistics


METRICS = ("chd_auc", "chd_auprc", "macro_auc", "macro_auprc")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(os.path.abspath(args.input_dir), "*", "final_test_summary.json")))
    rows = []
    for path in paths:
        run_name = os.path.basename(os.path.dirname(path))
        match = re.fullmatch(r"(.+)_seed(\d+)", run_name)
        if not match:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not payload.get("test_set_used") or payload.get("threshold_tuning_on_test"):
            raise RuntimeError(f"Invalid final-test audit fields: {path}")
        metrics = payload["test_metrics"]
        rows.append({
            "model": match.group(1),
            "seed": int(match.group(2)),
            **{metric: float(metrics[metric]) for metric in METRICS},
        })
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 final-test runs, found {len(rows)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "official_fm_final_test_runs.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        aggregate = {"model": model, "n": len(model_rows)}
        for metric in METRICS:
            values = [row[metric] for row in model_rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sd"] = statistics.stdev(values)
        aggregates.append(aggregate)

    fields = ["model", "n"] + [field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_sd")]
    with open(os.path.join(args.output_dir, "official_fm_final_test_aggregate.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)

    markdown = os.path.join(args.output_dir, "official_fm_final_test_table.md")
    with open(markdown, "w", encoding="utf-8") as handle:
        handle.write("# One-time sealed test comparison\n\n")
        handle.write("All heads and hyperparameters were frozen using development validation only. No threshold or model selection used test labels.\n\n")
        handle.write("| Model | N | CHD AUROC | CHD AUPRC | Macro AUROC | Macro AUPRC |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in aggregates:
            values = [f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}" for metric in METRICS]
            handle.write(f"| {row['model']} | {row['n']} | " + " | ".join(values) + " |\n")
    print(markdown)


if __name__ == "__main__":
    main()
