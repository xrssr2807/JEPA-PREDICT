"""Summarize independent PhysioV2 and MAE pretraining-seed comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


CHD_LABEL = "冠心病"


def read_predictions(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["uid"])
    if not rows:
        raise ValueError(f"No predictions in {path}")
    if {row["split"] for row in rows} != {"val"}:
        raise ValueError(f"Expected validation-only predictions: {path}")
    labels = [
        column.split("::", 1)[1]
        for column in rows[0]
        if column.startswith("label::")
    ]
    y_true = np.asarray(
        [[float(row[f"label::{label}"]) for label in labels] for row in rows]
    )
    y_prob = np.asarray(
        [[float(row[f"prob::{label}"]) for label in labels] for row in rows]
    )
    if not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    return [row["uid"] for row in rows], labels, y_true, y_prob


def aucs(y_true: np.ndarray, y_prob: np.ndarray, chd_index: int):
    per_class = []
    for index in range(y_true.shape[1]):
        if np.unique(y_true[:, index]).size < 2:
            continue
        per_class.append(roc_auc_score(y_true[:, index], y_prob[:, index]))
    return float(np.mean(per_class)), float(
        roc_auc_score(y_true[:, chd_index], y_prob[:, chd_index])
    )


def mean_std(values):
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def t_interval(values):
    mean, std = mean_std(values)
    if len(values) < 2:
        return mean, mean
    critical = {2: 12.7062, 3: 4.3027, 4: 3.1824}.get(len(values), 1.96)
    half = critical * std / math.sqrt(len(values))
    return mean - half, mean + half


def paired_bootstrap(y_true, physio_prob, mae_prob, chd_index, iterations, seed):
    rng = np.random.default_rng(seed)
    macro_delta = []
    chd_delta = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        try:
            physio_macro, physio_chd = aucs(
                y_true[indices], physio_prob[indices], chd_index
            )
            mae_macro, mae_chd = aucs(
                y_true[indices], mae_prob[indices], chd_index
            )
        except ValueError:
            continue
        macro_delta.append(physio_macro - mae_macro)
        chd_delta.append(physio_chd - mae_chd)
    return {
        "macro_ci95_low": float(np.percentile(macro_delta, 2.5)),
        "macro_ci95_high": float(np.percentile(macro_delta, 97.5)),
        "chd_ci95_low": float(np.percentile(chd_delta, 2.5)),
        "chd_ci95_high": float(np.percentile(chd_delta, 97.5)),
        "valid_bootstrap_iterations": len(macro_delta),
    }


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_dir(args, method, seed):
    if seed == 42:
        return Path(
            args.physio_seed42_dir
            if method == "physio_v2"
            else args.mae_seed42_dir
        )
    return Path(
        f"{args.downstream_prefix}_{method}_preseed{seed}"
        f"_ftseed{args.downstream_seed}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_seeds", nargs="+", type=int, required=True)
    parser.add_argument("--downstream_seed", type=int, default=42)
    parser.add_argument("--downstream_prefix", default="outputs_p0_independent")
    parser.add_argument(
        "--physio_seed42_dir", default="outputs_p0_objective_physio_v2_seed42"
    )
    parser.add_argument(
        "--mae_seed42_dir", default="outputs_p0_objective_multimodal_mae_seed42"
    )
    parser.add_argument("--paper_dir", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    args = parser.parse_args()

    run_rows = []
    paired_rows = []
    for seed in args.pretrain_seeds:
        method_data = {}
        for method in ("physio_v2", "multimodal_mae"):
            directory = result_dir(args, method, seed)
            predictions = directory / "validation_patient_predictions.csv"
            checkpoint = directory / "downstream_multidisease_best.pt"
            if not predictions.is_file() or not checkpoint.is_file():
                raise FileNotFoundError(directory)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("test_status") != "sealed":
                raise ValueError(f"Test set is not sealed: {checkpoint}")
            uids, labels, y_true, y_prob = read_predictions(predictions)
            if CHD_LABEL not in labels:
                raise ValueError(f"{CHD_LABEL} missing from {predictions}")
            chd_index = labels.index(CHD_LABEL)
            macro_auc, chd_auc = aucs(y_true, y_prob, chd_index)
            run_rows.append({
                "pretrain_seed": seed,
                "downstream_seed": args.downstream_seed,
                "method": method,
                "patients": len(uids),
                "macro_auc": macro_auc,
                "chd_auc": chd_auc,
                "test_status": "sealed",
                "predictions": str(predictions.resolve()),
            })
            method_data[method] = (uids, labels, y_true, y_prob, chd_index)

        physio = method_data["physio_v2"]
        mae = method_data["multimodal_mae"]
        if physio[0] != mae[0] or physio[1] != mae[1]:
            raise ValueError(f"Patient or label mismatch for seed {seed}")
        if not np.array_equal(physio[2], mae[2]):
            raise ValueError(f"Ground-truth mismatch for seed {seed}")
        physio_macro, physio_chd = aucs(physio[2], physio[3], physio[4])
        mae_macro, mae_chd = aucs(mae[2], mae[3], mae[4])
        paired_rows.append({
            "pretrain_seed": seed,
            "downstream_seed": args.downstream_seed,
            "delta_macro_auc_physio_minus_mae": physio_macro - mae_macro,
            "delta_chd_auc_physio_minus_mae": physio_chd - mae_chd,
            **paired_bootstrap(
                physio[2], physio[3], mae[3], physio[4],
                args.bootstrap_iterations, 2027 + seed,
            ),
            "test_status": "sealed",
        })

    aggregate_rows = []
    for metric in ("macro_auc", "chd_auc"):
        physio_values = [
            row[metric] for row in run_rows if row["method"] == "physio_v2"
        ]
        mae_values = [
            row[metric]
            for row in run_rows
            if row["method"] == "multimodal_mae"
        ]
        delta_key = f"delta_{metric}_physio_minus_mae"
        deltas = [row[delta_key] for row in paired_rows]
        physio_mean, physio_std = mean_std(physio_values)
        mae_mean, mae_std = mean_std(mae_values)
        delta_mean, delta_std = mean_std(deltas)
        ci_low, ci_high = t_interval(deltas)
        aggregate_rows.append({
            "metric": metric,
            "pretrain_seeds": len(args.pretrain_seeds),
            "physio_mean": physio_mean,
            "physio_std": physio_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "delta_seed_t_ci95_low": ci_low,
            "delta_seed_t_ci95_high": ci_high,
            "positive_seed_count": sum(value > 0 for value in deltas),
            "test_status": "sealed",
        })

    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    write_csv(paper_dir / "independent_pretrain_runs.csv", run_rows)
    write_csv(paper_dir / "independent_pretrain_paired.csv", paired_rows)
    write_csv(paper_dir / "independent_pretrain_aggregate.csv", aggregate_rows)
    summary = {
        "design": "paired independent pretraining seeds; fixed downstream seed",
        "pretrain_seeds": args.pretrain_seeds,
        "downstream_seed": args.downstream_seed,
        "test_status": "sealed",
        "runs": run_rows,
        "paired": paired_rows,
        "aggregate": aggregate_rows,
    }
    (paper_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "[Complete] independent PhysioV2 vs MAE summary | "
        f"pretrain_seeds={len(args.pretrain_seeds)} test_set_sealed=True"
    )


if __name__ == "__main__":
    main()
