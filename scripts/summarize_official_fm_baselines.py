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


def mean_sd(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def main():
    args = parse_args()
    pattern = os.path.join(os.path.abspath(args.input_dir), "*_summary.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        match = re.match(r"(.+)_seed(\d+)_summary\.json$", os.path.basename(path))
        if not match:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload["validation_metrics"]
        row = {"model": match.group(1), "seed": int(match.group(2))}
        row.update({name: float(metrics[name]) for name in METRICS})
        row["test_set_used"] = bool(payload.get("test_set_used", True))
        if row["test_set_used"]:
            raise RuntimeError(f"Sealed-test violation in {path}")
        runs.append(row)
    if not runs:
        raise RuntimeError(f"No summaries found under {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_path = os.path.join(args.output_dir, "official_fm_runs.csv")
    with open(run_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", *METRICS, "test_set_used"])
        writer.writeheader()
        writer.writerows(runs)

    aggregates = []
    for model in sorted({row["model"] for row in runs}):
        model_runs = [row for row in runs if row["model"] == model]
        row = {"model": model, "n": len(model_runs)}
        for metric in METRICS:
            mean, sd = mean_sd([item[metric] for item in model_runs])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        aggregates.append(row)

    fields = ["model", "n"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_sd")
    ]
    aggregate_path = os.path.join(args.output_dir, "official_fm_aggregate.csv")
    with open(aggregate_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)

    markdown_path = os.path.join(args.output_dir, "official_fm_validation_table.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("| Model | N | CHD AUROC | CHD AUPRC | Macro AUROC | Macro AUPRC |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in aggregates:
            values = []
            for metric in METRICS:
                values.append(f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}")
            handle.write(f"| {row['model']} | {row['n']} | " + " | ".join(values) + " |\n")
        handle.write("\nDevelopment validation only; patient split fixed; test set sealed.\n")
    print(markdown_path)


if __name__ == "__main__":
    main()
