#!/usr/bin/env python3
"""Summarize the sealed PPG continued-SSL validation comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run(name: str, directory: Path) -> dict:
    checkpoint = directory / "downstream_multidisease_best.pt"
    predictions = directory / "validation_patient_predictions.csv"
    log = directory / "downstream_console.log"
    for path in (checkpoint, predictions, log):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if "TEST SET SEALED" not in log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"Unsealed run: {directory}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "variant": name,
        "validation_macro_auc": float(payload["val_auc"]),
        "validation_chd_auc": float(payload["val_chd_auc"]),
        "validation_macro_f1": float(payload["val_f1"]),
        "validation_best_metric": float(payload["val_best_metric"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--adapted", required=True, type=Path)
    parser.add_argument("--adaptation_checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        read_run("baseline", args.baseline),
        read_run("continued_ssl", args.adapted),
    ]
    baseline = rows[0]
    for row in rows:
        row["delta_chd_auc_vs_baseline"] = (
            row["validation_chd_auc"] - baseline["validation_chd_auc"]
        )
        row["delta_macro_auc_vs_baseline"] = (
            row["validation_macro_auc"] - baseline["validation_macro_auc"]
        )
    adapted = rows[1]
    keep = (
        adapted["delta_chd_auc_vs_baseline"] >= 0.005
        and adapted["delta_macro_auc_vs_baseline"] >= -0.005
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ppg_continued_ssl_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "experiment": "P3_ppg_continued_ssl_seed42",
        "test_set_sealed": True,
        "decision": "replicate_three_seeds" if keep else "stop_after_seed42",
        "selection_rule": {
            "min_chd_auc_gain": 0.005,
            "max_macro_auc_drop": 0.005,
        },
        "adaptation_checkpoint": str(args.adaptation_checkpoint),
        "adaptation_checkpoint_sha256": sha256(args.adaptation_checkpoint),
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "runs": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# PPG继续自监督适配：验证集初筛",
        "",
        "全程仅使用冻结划分中的训练患者进行无标签适配，测试集保持封存。",
        "",
        "| 方案 | CHD AUC | Macro AUC | Macro F1 | CHD增量 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['validation_chd_auc']:.4f} | "
            f"{row['validation_macro_auc']:.4f} | "
            f"{row['validation_macro_f1']:.4f} | "
            f"{row['delta_chd_auc_vs_baseline']:+.4f} |"
        )
    lines.extend([
        "",
        f"预注册决策：**{manifest['decision']}**。",
        "只有 CHD AUC 至少提高 0.005 且 Macro AUC 下降不超过 0.005，才补三随机种子。",
    ])
    (args.output_dir / "analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[Complete] summary={csv_path} decision={manifest['decision']}")


if __name__ == "__main__":
    main()
