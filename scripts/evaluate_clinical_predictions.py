#!/usr/bin/env python3
"""Create patient-level clinical evaluation figures and paired AUC tests.

The input contract is the validation/test prediction CSV written by
``train_downstream.py``. Multiple models must contain the same patient UIDs and
ground-truth labels so every comparison is paired at patient level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    output = np.empty(len(values), dtype=np.float64)
    output[order] = ranks
    return output


def _fast_delong(predictions: np.ndarray, positive_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return AUCs and covariance for rows of positive-first predictions."""

    models, total = predictions.shape
    m = int(positive_count)
    n = total - m
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both positive and negative patients")
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    tx = np.vstack([_midrank(row) for row in positive])
    ty = np.vstack([_midrank(row) for row in negative])
    tz = np.vstack([_midrank(row) for row in predictions])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / m + sy / n
    if models == 1:
        covariance = covariance.reshape(1, 1)
    return aucs, covariance


def paired_delong(
    labels: np.ndarray,
    reference_scores: np.ndarray,
    comparison_scores: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-labels, kind="stable")
    predictions = np.vstack([reference_scores, comparison_scores])[:, order]
    aucs, covariance = _fast_delong(predictions, int(labels.sum()))
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    delta = float(aucs[0] - aucs[1])
    if variance <= 0 or not math.isfinite(variance):
        z_value = float("nan")
        p_value = 1.0 if abs(delta) < 1e-12 else 0.0
    else:
        z_value = delta / math.sqrt(variance)
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {
        "reference_auc": float(aucs[0]),
        "comparison_auc": float(aucs[1]),
        "delta_reference_minus_comparison": delta,
        "delong_z": z_value,
        "delong_p_two_sided": float(p_value),
    }


def _auc_ci(labels: np.ndarray, scores: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    order = np.argsort(-labels, kind="stable")
    aucs, covariance = _fast_delong(scores[None, order], int(labels.sum()))
    standard_error = math.sqrt(max(0.0, float(covariance[0, 0])))
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return max(0.0, float(aucs[0] - z * standard_error)), min(
        1.0, float(aucs[0] + z * standard_error)
    )


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (scores >= lower) & (scores < upper if index < bins - 1 else scores <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return value


def calibration_parameters(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _read_predictions(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate patient UID in {path}")
    label_names = [key.split("::", 1)[1] for key in rows[0] if key.startswith("label::")]
    if not label_names:
        raise ValueError(f"No label::<disease> columns in {path}")
    ordered = sorted(rows, key=lambda row: row["uid"])
    return {
        "path": str(path.resolve()),
        "uids": [row["uid"] for row in ordered],
        "split": sorted({row.get("split", "") for row in ordered}),
        "labels": label_names,
        "y": np.asarray([[int(float(row[f"label::{name}"])) for name in label_names] for row in ordered]),
        "p": np.asarray([[float(row[f"prob::{name}"]) for name in label_names] for row in ordered]),
    }


def _parse_named_paths(values: Sequence[str]) -> Dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Use NAME=PATH for --predictions, got: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate model name: {name}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = path
    return output


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(output_dir: Path, datasets: Dict[str, dict], focus_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
    ):
        if candidate in available_fonts:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    colors = ["#174A7E", "#E07A5F", "#2A9D8F", "#6D597A", "#D4A017"]
    label_index = next(iter(datasets.values()))["labels"].index(focus_label)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for (name, data), color in zip(datasets.items(), colors * 10):
        y, p = data["y"][:, label_index], data["p"][:, label_index]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC: {focus_label}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "focus_roc.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for (name, data), color in zip(datasets.items(), colors * 10):
        y, p = data["y"][:, label_index], data["p"][:, label_index]
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, lw=2, color=color, label=f"{name} (AP={ap:.3f})")
    prevalence = float(next(iter(datasets.values()))["y"][:, label_index].mean())
    ax.axhline(prevalence, ls="--", color="#777777", lw=1, label=f"prevalence={prevalence:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall: {focus_label}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "focus_pr.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for (name, data), color in zip(datasets.items(), colors * 10):
        y, p = data["y"][:, label_index], data["p"][:, label_index]
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, "o-", lw=2, color=color, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1, label="ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"Calibration: {focus_label}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "focus_calibration.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="NAME=prediction.csv; repeat for each model")
    parser.add_argument("--reference", required=True, help="Reference model name for paired DeLong tests")
    parser.add_argument("--focus_label", default="冠心病")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    paths = _parse_named_paths(args.predictions)
    if args.reference not in paths:
        raise ValueError(f"Reference model not supplied: {args.reference}")
    datasets = {name: _read_predictions(path) for name, path in paths.items()}
    reference = datasets[args.reference]
    if args.focus_label not in reference["labels"]:
        raise ValueError(f"Focus label not found: {args.focus_label}")
    for name, data in datasets.items():
        if data["uids"] != reference["uids"] or data["labels"] != reference["labels"]:
            raise ValueError(f"{name} is not patient/label aligned with {args.reference}")
        if not np.array_equal(data["y"], reference["y"]):
            raise ValueError(f"Ground-truth labels differ for {name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[dict] = []
    for model_name, data in datasets.items():
        for index, label_name in enumerate(data["labels"]):
            y, p = data["y"][:, index], data["p"][:, index]
            if np.unique(y).size < 2:
                continue
            ci_low, ci_high = _auc_ci(y, p)
            intercept, slope = calibration_parameters(y, p)
            metric_rows.append({
                "model": model_name,
                "label": label_name,
                "patients": len(y),
                "positives": int(y.sum()),
                "auroc": float(roc_auc_score(y, p)),
                "auroc_ci95_low": ci_low,
                "auroc_ci95_high": ci_high,
                "average_precision": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "ece_10bin": expected_calibration_error(y, p),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            })

    comparisons = []
    focus_index = reference["labels"].index(args.focus_label)
    for name, data in datasets.items():
        if name == args.reference:
            continue
        result = paired_delong(reference["y"][:, focus_index], reference["p"][:, focus_index], data["p"][:, focus_index])
        comparisons.append({"label": args.focus_label, "reference": args.reference, "comparison": name, **result})

    _write_csv(output_dir / "clinical_metrics.csv", metric_rows)
    _write_csv(output_dir / "focus_paired_delong.csv", comparisons)
    _make_plots(output_dir, datasets, args.focus_label)
    summary = {
        "analysis_unit": "patient",
        "focus_label": args.focus_label,
        "reference": args.reference,
        "models": {name: data["path"] for name, data in datasets.items()},
        "patients": len(reference["uids"]),
        "split_values": reference["split"],
        "outputs": ["clinical_metrics.csv", "focus_paired_delong.csv", "focus_roc.png", "focus_pr.png", "focus_calibration.png"],
    }
    (output_dir / "clinical_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Complete] patient-level clinical report -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
