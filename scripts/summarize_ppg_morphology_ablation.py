#!/usr/bin/env python3
"""Summarize the sealed PPG morphology-head validation ablation."""

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


def parse_run(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIR")
    name, directory = value.split("=", 1)
    return name, Path(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--pretrain", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for name, directory in args.run:
        checkpoint = directory / "downstream_multidisease_best.pt"
        predictions = directory / "validation_patient_predictions.csv"
        log = directory / "downstream_console.log"
        for required in (checkpoint, predictions, log):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(required)
        if "TEST SET SEALED" not in log.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"Run is not sealed: {directory}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        rows.append({
            "variant": name,
            "validation_macro_auc": float(payload["val_auc"]),
            "validation_chd_auc": float(payload["val_chd_auc"]),
            "validation_macro_f1": float(payload["val_f1"]),
            "validation_best_metric": float(payload["val_best_metric"]),
            "ppg_morphology_head": bool(payload.get("ppg_morphology_head", False)),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        })

    baseline = next(row for row in rows if row["variant"] == "baseline")
    for row in rows:
        row["delta_chd_auc_vs_baseline"] = (
            row["validation_chd_auc"] - baseline["validation_chd_auc"]
        )
        row["delta_macro_auc_vs_baseline"] = (
            row["validation_macro_auc"] - baseline["validation_macro_auc"]
        )
    rows.sort(key=lambda row: row["validation_best_metric"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ppg_morphology_ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    morphology = next(row for row in rows if row["variant"] == "morphology")
    keep = (
        morphology["delta_chd_auc_vs_baseline"] >= 0.005
        and morphology["delta_macro_auc_vs_baseline"] >= -0.005
    )
    manifest = {
        "experiment": "P3_ppg_morphology_residual_seed42",
        "test_set_sealed": True,
        "selection_rule": {
            "min_chd_auc_gain": 0.005,
            "max_macro_auc_drop": 0.005,
        },
        "decision": "replicate_three_seeds" if keep else "stop_after_seed42",
        "pretrain": str(args.pretrain),
        "pretrain_sha256": sha256(args.pretrain),
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "runs": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# PPG形态学残差头：验证集初筛",
        "",
        "本阶段严格封存测试集，仅判断方案二是否值得进入三随机种子复现。",
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
        "本结果只用于开发集模型筛选，不能作为最终测试结论。",
    ])
    (args.output_dir / "analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"[Complete] summary={csv_path} decision={manifest['decision']}"
    )


if __name__ == "__main__":
    main()
