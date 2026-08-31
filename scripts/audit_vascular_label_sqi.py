#!/usr/bin/env python3
"""Audit label definitions and validation SQI strata without reading test data."""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from dataset.data import compute_ppg_sqi, load_pickle_compat, multidisease_label_value


DEVICE_KEYS = ("device", "device_id", "device_model", "source_device", "vendor")
CENTER_KEYS = ("center", "site", "hospital", "institution", "center_id")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_sqi_windows", type=int, default=3)
    return parser.parse_args()


def uid_from_filename(filename):
    parts = filename.split("_")
    return parts[1] if parts[0] in {"train", "val", "test"} and len(parts) >= 3 else parts[0]


def read_sample(data_dir, filename):
    with open(os.path.join(data_dir, filename), "rb") as handle:
        return load_pickle_compat(handle)


def first_present(sample, keys):
    for key in keys:
        value = sample.get(key)
        if value not in (None, "", []):
            return str(value)
    return ""


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.split, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    labels = list(manifest.get("metadata", {}).get("disease_labels", []))
    if not labels:
        raise ValueError("Split manifest has no disease_labels")

    # Deliberately restrict every metadata read to development data.
    files_by_split = {name: list(manifest.get(name, [])) for name in ("train", "val")}
    by_uid = {name: defaultdict(list) for name in files_by_split}
    for split_name, files in files_by_split.items():
        for filename in files:
            by_uid[split_name][uid_from_filename(filename)].append(filename)

    rows = []
    raw_keys = Counter()
    metadata_coverage = Counter()
    val_sqi = []
    for split_name, patients in by_uid.items():
        positives = np.zeros(len(labels), dtype=np.int64)
        for uid, filenames in patients.items():
            sample = read_sample(args.data_dir, filenames[0])
            label_dict = sample.get("label", {})
            raw_keys.update(label_dict.keys())
            device = first_present(sample, DEVICE_KEYS)
            center = first_present(sample, CENTER_KEYS)
            metadata_coverage["device"] += int(bool(device))
            metadata_coverage["center"] += int(bool(center))
            values = np.asarray([
                multidisease_label_value(label_dict, label) for label in labels
            ], dtype=np.int64)
            positives += values

            if split_name == "val":
                scores = []
                for filename in filenames[: max(1, args.max_sqi_windows)]:
                    item = read_sample(args.data_dir, filename)
                    signal = np.asarray(item["data"], dtype=np.float32).squeeze()
                    if signal.ndim > 1:
                        signal = signal[0]
                    fs = int(round(float(np.asarray(item.get("sampling_rate", 100)).reshape(-1)[0])))
                    scores.append(compute_ppg_sqi(signal, fs=max(fs, 1)))
                val_sqi.append({"uid": uid, "sqi": float(np.mean(scores))})

        total = max(len(patients), 1)
        for label, positive in zip(labels, positives):
            rows.append({
                "split": split_name,
                "label": label,
                "patients": len(patients),
                "positive_patients": int(positive),
                "prevalence": float(positive / total),
            })

    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "label_prevalence_train_val.csv"), index=False
    )
    sqi_frame = pd.DataFrame(val_sqi)
    if not sqi_frame.empty:
        sqi_frame["sqi_stratum"] = pd.qcut(
            sqi_frame["sqi"], q=3, labels=("low", "medium", "high"), duplicates="drop"
        ).astype(str)
        sqi_frame.to_csv(os.path.join(args.output_dir, "validation_patient_sqi.csv"), index=False)

    coverage = {
        "protocol": "train_and_validation_only_test_sealed",
        "patients_scanned": sum(len(value) for value in by_uid.values()),
        "raw_label_keys": dict(sorted(raw_keys.items())),
        "device": {
            "candidate_keys": DEVICE_KEYS,
            "patients_with_metadata": metadata_coverage["device"],
            "status": "AVAILABLE" if metadata_coverage["device"] else "UNAVAILABLE",
        },
        "center": {
            "candidate_keys": CENTER_KEYS,
            "patients_with_metadata": metadata_coverage["center"],
            "status": "AVAILABLE" if metadata_coverage["center"] else "UNAVAILABLE",
        },
    }
    with open(os.path.join(args.output_dir, "metadata_coverage.json"), "w", encoding="utf-8") as handle:
        json.dump(coverage, handle, ensure_ascii=False, indent=2)

    if args.predictions:
        predictions = pd.read_csv(args.predictions, dtype={"uid": str})
        merged = predictions.merge(sqi_frame, on="uid", how="inner", validate="one_to_one")
        metric_rows = []
        for stratum, frame in merged.groupby("sqi_stratum", observed=True):
            for label in labels:
                y = frame[f"label::{label}"].to_numpy(dtype=int)
                p = frame[f"prob::{label}"].to_numpy(dtype=float)
                if np.unique(y).size < 2:
                    continue
                metric_rows.append({
                    "sqi_stratum": stratum,
                    "label": label,
                    "patients": len(frame),
                    "positives": int(y.sum()),
                    "auroc": roc_auc_score(y, p),
                    "auprc": average_precision_score(y, p),
                })
        pd.DataFrame(metric_rows).to_csv(
            os.path.join(args.output_dir, "validation_sqi_stratified_metrics.csv"), index=False
        )

    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
