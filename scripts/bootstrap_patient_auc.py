"""Patient-level bootstrap confidence intervals for sealed validation runs."""

import argparse
import csv
import glob
import hashlib
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score


def load_predictions(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    label_columns = [
        column for column in rows[0] if column.startswith("label::")
    ]
    label_names = [column.split("::", 1)[1] for column in label_columns]
    probability_columns = [f"prob::{name}" for name in label_names]
    uids = np.asarray([row["uid"] for row in rows])
    labels = np.asarray([
        [float(row[column]) for column in label_columns] for row in rows
    ])
    probabilities = np.asarray([
        [float(row[column]) for column in probability_columns] for row in rows
    ])
    return uids, label_names, labels, probabilities


def auc_metrics(labels, probabilities, chd_index):
    per_class = []
    for class_index in range(labels.shape[1]):
        target = labels[:, class_index]
        if np.unique(target).size < 2:
            per_class.append(np.nan)
        else:
            per_class.append(
                roc_auc_score(target, probabilities[:, class_index])
            )
    return float(np.nanmean(per_class)), float(per_class[chd_index])


def discover_runs(p2_prefix, p3_prefix):
    patterns = [
        (p2_prefix, "p2"),
        (p3_prefix, "p3"),
    ]
    runs = {}
    for prefix, group in patterns:
        for path in glob.glob(
            f"{prefix}_*_seed*/validation_patient_predictions.csv"
        ):
            directory = os.path.dirname(path)
            match = re.match(
                rf"^{re.escape(prefix)}_(.+)_seed(\d+)$", directory
            )
            if match is None:
                continue
            experiment, seed = match.group(1), int(match.group(2))
            runs[(group, experiment, seed)] = path
    return runs


def aligned_pair(reference_path, treatment_path):
    ref_uids, ref_names, ref_labels, ref_probs = load_predictions(
        reference_path
    )
    trt_uids, trt_names, trt_labels, trt_probs = load_predictions(
        treatment_path
    )
    if ref_names != trt_names:
        raise ValueError("Prediction label schemas do not match")
    ref_index = {uid: index for index, uid in enumerate(ref_uids)}
    trt_index = {uid: index for index, uid in enumerate(trt_uids)}
    if set(ref_index) != set(trt_index):
        raise ValueError("Prediction files do not contain identical patient UIDs")
    ordered_uids = sorted(ref_index)
    ref_rows = [ref_index[uid] for uid in ordered_uids]
    trt_rows = [trt_index[uid] for uid in ordered_uids]
    labels = ref_labels[ref_rows]
    if not np.array_equal(labels, trt_labels[trt_rows]):
        raise ValueError("Patient labels differ between paired runs")
    return ref_names, labels, ref_probs[ref_rows], trt_probs[trt_rows]


def percentile_interval(values):
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_prefix", default="outputs_p2_core")
    parser.add_argument("--p3_prefix", default="outputs_p3")
    parser.add_argument("--reference", default="phase2")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output_dir",
        default="paper/ICASSP2027/04_statistics/results",
    )
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError("Bootstrap iterations must be at least 100")

    runs = discover_runs(args.p2_prefix, args.p3_prefix)
    if not runs:
        raise FileNotFoundError("No patient prediction files were discovered")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    paired_rows = []

    for (group, experiment, seed), path in sorted(runs.items()):
        uids, label_names, labels, probabilities = load_predictions(path)
        chd_index = label_names.index("冠心病")
        macro_auc, chd_auc = auc_metrics(labels, probabilities, chd_index)
        rng_seed = int.from_bytes(
            hashlib.sha256(
                f"{group}:{experiment}:{seed}:{args.seed}".encode()
            ).digest()[:4],
            "little",
        )
        rng = np.random.default_rng(rng_seed)
        macro_samples = []
        chd_samples = []
        for _ in range(args.iterations):
            indices = rng.integers(0, len(uids), size=len(uids))
            macro_value, chd_value = auc_metrics(
                labels[indices], probabilities[indices], chd_index
            )
            macro_samples.append(macro_value)
            chd_samples.append(chd_value)
        macro_low, macro_high = percentile_interval(macro_samples)
        chd_low, chd_high = percentile_interval(chd_samples)
        summary_rows.append({
            "group": group,
            "experiment": experiment,
            "seed": seed,
            "patients": len(uids),
            "macro_auc": macro_auc,
            "macro_auc_ci95_low": macro_low,
            "macro_auc_ci95_high": macro_high,
            "chd_auc": chd_auc,
            "chd_auc_ci95_low": chd_low,
            "chd_auc_ci95_high": chd_high,
            "predictions": os.path.abspath(path),
        })

        reference_key = ("p2", args.reference, seed)
        if experiment == args.reference and group == "p2":
            continue
        if reference_key not in runs:
            raise FileNotFoundError(
                f"Missing paired reference for seed {seed}: {reference_key}"
            )
        names, pair_labels, ref_probs, trt_probs = aligned_pair(
            runs[reference_key], path
        )
        pair_chd_index = names.index("冠心病")
        ref_macro, ref_chd = auc_metrics(
            pair_labels, ref_probs, pair_chd_index
        )
        trt_macro, trt_chd = auc_metrics(
            pair_labels, trt_probs, pair_chd_index
        )
        macro_deltas = []
        chd_deltas = []
        pair_rng = np.random.default_rng(rng_seed + 1)
        for _ in range(args.iterations):
            indices = pair_rng.integers(
                0, len(pair_labels), size=len(pair_labels)
            )
            sample_labels = pair_labels[indices]
            ref_values = auc_metrics(
                sample_labels, ref_probs[indices], pair_chd_index
            )
            trt_values = auc_metrics(
                sample_labels, trt_probs[indices], pair_chd_index
            )
            macro_deltas.append(trt_values[0] - ref_values[0])
            chd_deltas.append(trt_values[1] - ref_values[1])
        macro_delta_low, macro_delta_high = percentile_interval(macro_deltas)
        chd_delta_low, chd_delta_high = percentile_interval(chd_deltas)
        paired_rows.append({
            "group": group,
            "experiment": experiment,
            "reference": args.reference,
            "seed": seed,
            "delta_macro_auc": trt_macro - ref_macro,
            "delta_macro_auc_ci95_low": macro_delta_low,
            "delta_macro_auc_ci95_high": macro_delta_high,
            "delta_chd_auc": trt_chd - ref_chd,
            "delta_chd_auc_ci95_low": chd_delta_low,
            "delta_chd_auc_ci95_high": chd_delta_high,
        })

    for filename, rows in (
        ("patient_bootstrap_summary.csv", summary_rows),
        ("patient_bootstrap_paired_deltas.csv", paired_rows),
    ):
        with open(
            os.path.join(args.output_dir, filename),
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"[Complete] patient bootstrap | runs={len(summary_rows)} "
        f"paired={len(paired_rows)} iterations={args.iterations} "
        "| test_set_sealed=True"
    )


if __name__ == "__main__":
    main()
