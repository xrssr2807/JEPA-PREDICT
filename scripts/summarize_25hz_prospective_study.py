#!/usr/bin/env python3
"""Summarize validation and previously inspected prospective diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


CONDITIONS = (
    "native100_10s",
    "bridge25_10s",
    "native25_10s",
    "native25_30s_token",
)


def find_chd_columns(fieldnames: list[str]) -> tuple[str, str]:
    label_columns = [name for name in fieldnames if name.startswith("label::")]
    for label_column in label_columns:
        label_name = label_column.split("::", 1)[1]
        if label_name == "冠心病":
            return label_column, f"prob::{label_name}"
    if len(label_columns) <= 4:
        raise ValueError("Cannot locate CHD columns in prediction file")
    label_name = label_columns[4].split("::", 1)[1]
    return label_columns[4], f"prob::{label_name}"


def auc_from_predictions(path: Path) -> tuple[float, int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        label_column, probability_column = find_chd_columns(reader.fieldnames or [])
        rows = list(reader)
    labels = np.asarray([float(row[label_column]) for row in rows])
    probabilities = np.asarray([float(row[probability_column]) for row in rows])
    auc = float(roc_auc_score(labels, probabilities))
    return auc, len(rows), int(labels.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_root", required=True)
    args = parser.parse_args()
    root = Path(args.paper_root).resolve()
    rows = []
    for condition in CONDITIONS:
        validation_path = root / condition / "validation_patient_predictions.csv"
        external_path = root / condition / "external_patient_predictions.csv"
        if not validation_path.is_file() or not external_path.is_file():
            raise FileNotFoundError(f"Incomplete condition: {condition}")
        val_auc, val_n, val_positive = auc_from_predictions(validation_path)
        ext_auc, ext_n, ext_positive = auc_from_predictions(external_path)
        rows.append({
            "condition": condition,
            "validation_chd_auc": val_auc,
            "external_diagnostic_chd_auc": ext_auc,
            "generalization_gap": ext_auc - val_auc,
            "validation_patients": val_n,
            "validation_chd_positive": val_positive,
            "external_patients": ext_n,
            "external_chd_positive": ext_positive,
        })

    csv_path = root / "sampling_rate_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# 25 Hz 前瞻性诊断研究汇总",
        "",
        "| 条件 | 内部验证 CHD AUC | 前瞻诊断 CHD AUC | 泛化差值 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['condition']} | {row['validation_chd_auc']:.4f} | "
            f"{row['external_diagnostic_chd_auc']:.4f} | "
            f"{row['generalization_gap']:+.4f} |"
        )
    md_lines.extend([
        "",
        "> 外部队列已在既往实验中被查看，本表只能用于域偏移诊断，不能用于模型选择后再声明独立外部验证。",
    ])
    (root / "sampling_rate_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    (root / "sampling_rate_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
