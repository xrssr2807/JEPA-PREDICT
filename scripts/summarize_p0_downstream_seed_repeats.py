"""Summarize P0 downstream-seed repeats with a fixed pre-training checkpoint."""

from __future__ import annotations

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


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_checkpoints(values: list[str]) -> dict[str, str]:
    checkpoints = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --checkpoint value: {value}")
        experiment, path = value.split("=", 1)
        checkpoints[experiment] = path
    return checkpoints


def output_dir(args: argparse.Namespace, experiment: str, seed: int) -> str:
    if seed == args.pretrain_seed:
        return f"{args.legacy_output_prefix}_{experiment}_seed{seed}"
    return (
        f"{args.output_prefix}_{experiment}_preseed{args.pretrain_seed}"
        f"_ftseed{seed}"
    )


def summary(values: list[float]) -> tuple[int, float, float]:
    finite = [value for value in values if math.isfinite(value)]
    return (
        len(finite),
        statistics.fmean(finite),
        statistics.stdev(finite) if len(finite) > 1 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", default="outputs_p0_objective_ftrepeat")
    parser.add_argument("--legacy_output_prefix", default="outputs_p0_objective")
    parser.add_argument("--paper_dir", required=True)
    parser.add_argument("--pretrain_seed", type=int, default=42)
    parser.add_argument("--ft_seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    args = parser.parse_args()

    checkpoints = parse_checkpoints(args.checkpoint)
    experiments = ("physio_v2", "multimodal_mae", "contrastive", "xmae")
    missing_checkpoint = set(experiments).difference(checkpoints)
    if missing_checkpoint:
        raise ValueError(f"Missing checkpoint mappings: {sorted(missing_checkpoint)}")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    split_hashes: set[str] = set()
    pretrain_hashes = {
        experiment: sha256(path) if os.path.isfile(path) else ""
        for experiment, path in checkpoints.items()
    }
    for experiment in experiments:
        if not pretrain_hashes[experiment]:
            missing.append(checkpoints[experiment])
            continue
        for seed in args.ft_seeds:
            directory = output_dir(args, experiment, seed)
            model_path = os.path.join(directory, "downstream_multidisease_best.pt")
            predictions_path = os.path.join(
                directory, "validation_patient_predictions.csv"
            )
            for path in (model_path, predictions_path):
                if not os.path.isfile(path):
                    missing.append(path)
            if not os.path.isfile(model_path) or not os.path.isfile(predictions_path):
                continue

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if checkpoint.get("test_status") != "sealed":
                raise ValueError(f"Test set is not sealed: {model_path}")
            validation = checkpoint.get("validation_metrics") or {}
            split_hash = (checkpoint.get("data_split") or {}).get("sha256", "")
            split_hashes.add(split_hash)
            rows.append(
                {
                    "experiment": experiment,
                    "pretrain_seed": args.pretrain_seed,
                    "downstream_seed": seed,
                    "split_sha256": split_hash,
                    "pretrain_sha256": pretrain_hashes[experiment],
                    "test_status": "sealed",
                    "val_macro_auc": float(
                        validation.get("auc", checkpoint.get("val_auc"))
                    ),
                    "val_chd_auc": float(checkpoint.get("val_chd_auc")),
                    "val_f1": float(
                        validation.get("f1", checkpoint.get("val_f1"))
                    ),
                    "checkpoint": os.path.abspath(model_path),
                    "patient_predictions": os.path.abspath(predictions_path),
                }
            )

    if missing:
        raise FileNotFoundError(
            "P0 downstream seed repeats are incomplete:\n" + "\n".join(missing)
        )
    if not rows:
        raise FileNotFoundError("No P0 downstream repeat results found")
    if "" in split_hashes or len(split_hashes) != 1:
        raise ValueError(f"Runs do not share one split SHA256: {split_hashes}")

    os.makedirs(args.paper_dir, exist_ok=True)
    write_csv(os.path.join(args.paper_dir, "downstream_seed_runs.csv"), rows)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(row)
    aggregate = []
    for experiment in experiments:
        result: dict[str, object] = {"experiment": experiment}
        for metric in METRICS:
            n, mean, std = summary(
                [float(row[metric]) for row in grouped[experiment]]
            )
            result[f"{metric}_n"] = n
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        aggregate.append(result)
    write_csv(os.path.join(args.paper_dir, "downstream_seed_aggregate.csv"), aggregate)

    by_experiment_seed = {
        (str(row["experiment"]), int(row["downstream_seed"])): row
        for row in rows
    }
    paired = []
    for baseline in experiments[1:]:
        for metric in METRICS:
            differences = [
                float(by_experiment_seed[("physio_v2", seed)][metric])
                - float(by_experiment_seed[(baseline, seed)][metric])
                for seed in args.ft_seeds
            ]
            n, mean, std = summary(differences)
            paired.append(
                {
                    "comparison": f"physio_v2-{baseline}",
                    "metric": metric,
                    "n": n,
                    "mean_difference": mean,
                    "std_difference": std,
                    "seed_differences": json.dumps(differences),
                }
            )
    write_csv(os.path.join(args.paper_dir, "downstream_seed_paired.csv"), paired)

    with open(
        os.path.join(args.paper_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "design": "fixed_pretrain_checkpoint_downstream_seed_repeat",
                "pretrain_seed": args.pretrain_seed,
                "downstream_seeds": args.ft_seeds,
                "split_sha256": next(iter(split_hashes)),
                "test_status": "sealed",
                "runs": rows,
                "aggregate": aggregate,
                "paired": paired,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"[Complete] P0 downstream seed summary | runs={len(rows)} "
        "| test_set_sealed=True"
    )


if __name__ == "__main__":
    main()
