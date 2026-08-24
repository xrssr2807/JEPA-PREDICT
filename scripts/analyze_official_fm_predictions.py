import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


MODEL_LABELS = {
    "physiov2_ppg": "PhysioV2 (ours)",
    "moment_small": "MOMENT-small",
    "normwear": "NormWear",
    "papagei_s": "PaPaGei-S",
    "units_x128": "UniTS-x128",
    "pulse_ppg": "Pulse-PPG",
}
CHD_INDEX = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference", default="physiov2_ppg")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ensembles(prediction_dir):
    grouped = {}
    pattern = os.path.join(os.path.abspath(prediction_dir), "*_predictions.npz")
    for path in sorted(glob.glob(pattern)):
        match = re.match(r"(.+)_seed(\d+)_predictions\.npz$", os.path.basename(path))
        if match:
            grouped.setdefault(match.group(1), []).append(path)

    ensembles = {}
    for model, paths in grouped.items():
        reference_uids = None
        reference_labels = None
        probabilities = []
        for path in paths:
            payload = np.load(path, allow_pickle=False)
            order = np.argsort(payload["uid"])
            uids = payload["uid"][order]
            labels = payload["labels"][order]
            probs = payload["probabilities"][order]
            if reference_uids is None:
                reference_uids = uids
                reference_labels = labels
            elif not np.array_equal(reference_uids, uids):
                raise RuntimeError(f"UID mismatch for {model}: {path}")
            elif not np.array_equal(reference_labels, labels):
                raise RuntimeError(f"Label mismatch for {model}: {path}")
            probabilities.append(probs)
        ensembles[model] = {
            "uids": reference_uids,
            "labels": reference_labels,
            "probabilities": np.mean(probabilities, axis=0),
            "n_seeds": len(paths),
        }
    return ensembles


def paired_bootstrap(y, reference, comparator, iterations, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        sampled_y = y[indices]
        if np.unique(sampled_y).size != 2:
            continue
        deltas.append(
            roc_auc_score(sampled_y, reference[indices])
            - roc_auc_score(sampled_y, comparator[indices])
        )
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        raise RuntimeError("No valid paired bootstrap samples")
    non_positive = (np.count_nonzero(values <= 0) + 1) / (values.size + 1)
    non_negative = (np.count_nonzero(values >= 0) + 1) / (values.size + 1)
    return {
        "delta": float(roc_auc_score(y, reference) - roc_auc_score(y, comparator)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "p_two_sided": float(min(1.0, 2.0 * min(non_positive, non_negative))),
        "valid_bootstrap": int(values.size),
    }


def main():
    args = parse_args()
    ensembles = load_ensembles(args.prediction_dir)
    if args.reference not in ensembles:
        raise RuntimeError(f"Reference model missing: {args.reference}")

    reference = ensembles[args.reference]
    reference_uids = reference["uids"]
    y = reference["labels"][:, CHD_INDEX].astype(np.int64)
    metrics = {}
    for model, payload in ensembles.items():
        if not np.array_equal(reference_uids, payload["uids"]):
            raise RuntimeError(f"Cross-model UID mismatch: {model}")
        if not np.array_equal(reference["labels"], payload["labels"]):
            raise RuntimeError(f"Cross-model label mismatch: {model}")
        score = payload["probabilities"][:, CHD_INDEX]
        metrics[model] = {
            "score": score,
            "auc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "n_seeds": payload["n_seeds"],
        }

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, model in enumerate(sorted(metrics, key=lambda item: metrics[item]["auc"], reverse=True)):
        item = metrics[model]
        label = MODEL_LABELS.get(model, model)
        linewidth = 2.6 if model == args.reference else 1.6
        fpr, tpr, _ = roc_curve(y, item["score"])
        precision, recall, _ = precision_recall_curve(y, item["score"])
        axes[0].plot(fpr, tpr, color=colors(index), lw=linewidth, label=f"{label} ({item['auc']:.3f})")
        axes[1].plot(recall, precision, color=colors(index), lw=linewidth, label=f"{label} ({item['auprc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    axes[1].axhline(y.mean(), ls="--", color="0.6", lw=1, label=f"Prevalence ({y.mean():.3f})")
    axes[0].set(title="CHD ROC on development patients", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="CHD precision-recall", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=8, frameon=False)
    figure_path = os.path.join(args.output_dir, "official_fm_chd_roc_pr.png")
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    rows = []
    reference_score = metrics[args.reference]["score"]
    for model in sorted(metrics):
        if model == args.reference:
            continue
        result = paired_bootstrap(
            y,
            reference_score,
            metrics[model]["score"],
            args.bootstrap,
            args.seed,
        )
        rows.append({
            "reference": args.reference,
            "comparator": model,
            "reference_auc": metrics[args.reference]["auc"],
            "comparator_auc": metrics[model]["auc"],
            **result,
        })
    csv_path = os.path.join(args.output_dir, "official_fm_chd_paired_bootstrap.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = os.path.join(args.output_dir, "official_fm_chd_statistical_comparison.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Frozen PPG representation comparison\n\n")
        handle.write("Curves use the mean patient probability across three downstream seeds. ")
        handle.write("Confidence intervals use paired patient-level bootstrap on the development set; the test set remains sealed.\n\n")
        handle.write("Two-sided p-values are exploratory and uncorrected for five comparisons.\n\n")
        handle.write("| Comparator | PhysioV2 AUC | Comparator AUC | Delta | 95% CI | p (two-sided) |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda item: item["comparator_auc"], reverse=True):
            handle.write(
                f"| {MODEL_LABELS.get(row['comparator'], row['comparator'])} "
                f"| {row['reference_auc']:.4f} | {row['comparator_auc']:.4f} "
                f"| {row['delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                f"| {row['p_two_sided']:.4g} |\n"
            )
    print(figure_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
