"""Summarize sealed capacity-controlled pre-training objective runs."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict

import torch


METRICS = ("val_macro_auc", "val_chd_auc", "val_f1")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else "",
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", default="outputs_p0_objective")
    parser.add_argument("--pretrain_prefix", default="outputs_p0_pretrain")
    parser.add_argument(
        "--paper_dir",
        default=(
            "paper/ICASSP2027/03_experiments/P3_baselines/results/"
            "pretraining_objectives"
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["physio_v2", "multimodal_mae", "contrastive", "xmae"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--physio_template", required=True)
    args = parser.parse_args()

    pretrain_paths = {
        "physio_v2": lambda seed: args.physio_template.format(seed=seed),
        "multimodal_mae": lambda seed: os.path.join(
            f"{args.pretrain_prefix}_multimodal_mae_seed{seed}",
            "multimodal_mae_best.pt",
        ),
        "contrastive": lambda seed: os.path.join(
            f"{args.pretrain_prefix}_contrastive_seed{seed}",
            "contrastive_best.pt",
        ),
        "xmae": lambda seed: os.path.join(
            f"{args.pretrain_prefix}_xmae_seed{seed}",
            "xmae_objective_best.pt",
        ),
    }
    rows = []
    missing = []
    split_hashes = set()
    for experiment in args.experiments:
        for seed in args.seeds:
            downstream_path = os.path.join(
                f"{args.output_prefix}_{experiment}_seed{seed}",
                "downstream_multidisease_best.pt",
            )
            predictions_path = os.path.join(
                f"{args.output_prefix}_{experiment}_seed{seed}",
                "validation_patient_predictions.csv",
            )
            pretrain_path = pretrain_paths[experiment](seed)
            for path in (downstream_path, predictions_path, pretrain_path):
                if not os.path.isfile(path):
                    missing.append(path)
            if any(
                not os.path.isfile(path)
                for path in (downstream_path, predictions_path, pretrain_path)
            ):
                continue
            checkpoint = torch.load(
                downstream_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("test_status") != "sealed":
                raise ValueError(f"Test set is not sealed: {downstream_path}")
            validation = checkpoint.get("validation_metrics") or {}
            split_hash = (checkpoint.get("data_split") or {}).get("sha256", "")
            split_hashes.add(split_hash)
            pretrain = torch.load(
                pretrain_path, map_location="cpu", weights_only=False
            )
            manifest = pretrain.get("pretraining_manifest") or {}
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "split_sha256": split_hash,
                "pretrain_sha256": sha256(pretrain_path),
                "pretrain_objective": pretrain.get(
                    "pretraining_objective", manifest.get("experiment", "")
                ),
                "encoder_parameters": manifest.get("encoder_parameters", ""),
                "pretrain_steps_budget": manifest.get(
                    "optimizer_steps_budget", ""
                ),
                "test_status": "sealed",
                "val_macro_auc": float(
                    validation.get("auc", checkpoint.get("val_auc"))
                ),
                "val_chd_auc": float(checkpoint.get("val_chd_auc")),
                "val_f1": float(
                    validation.get("f1", checkpoint.get("val_f1"))
                ),
                "checkpoint": os.path.abspath(downstream_path),
                "patient_predictions": os.path.abspath(predictions_path),
            })
    if missing:
        raise FileNotFoundError(
            "P0 objective comparison is incomplete:\n" + "\n".join(missing)
        )
    if not rows:
        raise FileNotFoundError("No objective-comparison results found")
    if "" in split_hashes or len(split_hashes) != 1:
        raise ValueError(f"Runs do not share one split SHA256: {split_hashes}")

    os.makedirs(args.paper_dir, exist_ok=True)
    write_csv(os.path.join(args.paper_dir, "objective_runs.csv"), rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    aggregate = []
    for experiment, experiment_rows in sorted(grouped.items()):
        result = {"experiment": experiment}
        for metric in METRICS:
            summary = summarize(row[metric] for row in experiment_rows)
            for name, value in summary.items():
                result[f"{metric}_{name}"] = value
        aggregate.append(result)
    write_csv(os.path.join(args.paper_dir, "objective_aggregate.csv"), aggregate)
    with open(
        os.path.join(args.paper_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "split_sha256": next(iter(split_hashes)),
                "test_status": "sealed",
                "runs": rows,
                "aggregate": aggregate,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"[Complete] objective comparison summary | runs={len(rows)} "
        "| test_set_sealed=True"
    )


if __name__ == "__main__":
    main()

