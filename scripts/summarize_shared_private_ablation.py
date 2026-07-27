"""Summarize Shared-Private channel/seed ablations from saved checkpoints."""
import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

import torch


CHANNELS = ("ppg", "ecg", "both")
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


def checkpoint_row(path, mode, channel, seed):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validation = checkpoint.get("validation_metrics") or {}
    row = {
        "mode": mode,
        "channel": channel,
        "seed": int(seed),
        "checkpoint": os.path.abspath(path),
        "split_sha256": (checkpoint.get("data_split") or {}).get("sha256", ""),
        "test_status": checkpoint.get("test_status", "legacy"),
        "val_macro_auc": float(
            validation.get("auc", checkpoint.get("val_auc", float("nan")))
        ),
        "val_chd_auc": float(checkpoint.get("val_chd_auc", float("nan"))),
        "val_f1": float(validation.get("f1", checkpoint.get("val_f1", float("nan")))),
        "test_macro_auc": float(checkpoint.get("test_auc", float("nan"))),
        "test_chd_auc": float(checkpoint.get("test_chd_auc", float("nan"))),
        "test_f1": float(checkpoint.get("test_f1", float("nan"))),
    }
    return row


def finite(values):
    return [value for value in values if math.isfinite(value)]


def summarize(values):
    values = finite(values)
    if not values:
        return {"n": 0, "mean": "", "std": "", "ci95_low": "", "ci95_high": ""}
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


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", default="outputs_spv2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    parser.add_argument("--head_modes", nargs="+", default=["on", "off"])
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS))
    parser.add_argument("--summary_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)
    run_rows = []
    missing = []
    for seed in args.seeds:
        for mode in args.head_modes:
            for channel in args.channels:
                output_dir = (
                    f"{args.output_prefix}_{mode}_{channel}_seed{seed}"
                )
                path = os.path.join(
                    output_dir, "downstream_multidisease_best.pt"
                )
                if not os.path.isfile(path):
                    missing.append(path)
                    continue
                run_rows.append(checkpoint_row(path, mode, channel, seed))

    if not run_rows:
        raise FileNotFoundError(
            "No downstream checkpoints found for the requested experiment grid"
        )

    run_fields = list(run_rows[0])
    runs_path = os.path.join(
        args.summary_dir, "shared_private_ablation_runs.csv"
    )
    write_csv(runs_path, run_rows, run_fields)

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["mode"], row["channel"])].append(row)
    aggregate_rows = []
    metrics = ("val_macro_auc", "val_chd_auc", "val_f1")
    for (mode, channel), rows in sorted(grouped.items()):
        aggregate = {"mode": mode, "channel": channel}
        for metric in metrics:
            summary = summarize([row[metric] for row in rows])
            for name, value in summary.items():
                aggregate[f"{metric}_{name}"] = value
        aggregate_rows.append(aggregate)

    aggregate_path = os.path.join(
        args.summary_dir, "shared_private_ablation_aggregate.csv"
    )
    write_csv(
        aggregate_path,
        aggregate_rows,
        list(aggregate_rows[0]),
    )

    indexed = {
        (row["mode"], row["channel"], row["seed"]): row for row in run_rows
    }
    delta_rows = []
    for channel in args.channels:
        for seed in args.seeds:
            enabled = indexed.get(("on", channel, seed))
            disabled = indexed.get(("off", channel, seed))
            if enabled is None or disabled is None:
                continue
            delta_rows.append({
                "channel": channel,
                "seed": seed,
                "delta_val_macro_auc": (
                    enabled["val_macro_auc"] - disabled["val_macro_auc"]
                ),
                "delta_val_chd_auc": (
                    enabled["val_chd_auc"] - disabled["val_chd_auc"]
                ),
                "delta_val_f1": enabled["val_f1"] - disabled["val_f1"],
            })
    delta_path = os.path.join(
        args.summary_dir, "shared_private_ablation_paired_deltas.csv"
    )
    if delta_rows:
        write_csv(delta_path, delta_rows, list(delta_rows[0]))

    print(f"[Summary] runs={len(run_rows)} missing={len(missing)}")
    print(f"[Summary] per-run:   {runs_path}")
    print(f"[Summary] aggregate: {aggregate_path}")
    if delta_rows:
        print(f"[Summary] paired:    {delta_path}")
    for path in missing:
        print(f"[Missing] {path}")


if __name__ == "__main__":
    main()
