#!/usr/bin/env python3
"""Summarize validation-only selective dual-to-PPG distillation runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="variant=output_directory")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for specification in args.run:
        variant, directory = specification.split("=", 1)
        run_dir = Path(directory)
        checkpoint = run_dir / "downstream_multidisease_best.pt"
        predictions = run_dir / "validation_patient_predictions.csv"
        if not checkpoint.is_file() or not predictions.is_file():
            raise FileNotFoundError(f"Incomplete run: {variant} -> {run_dir}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("test_evaluated", False):
            raise RuntimeError(f"Test set was evaluated in {variant}")
        distillation = payload.get("distillation") or {}
        rows.append({
            "variant": variant,
            "validation_macro_auc": float(payload.get("val_auc", 0.5)),
            "validation_chd_auc": float(payload.get("val_chd_auc", 0.5)),
            "validation_macro_f1": float(payload.get("val_f1", 0.0)),
            "validation_best_metric": float(
                payload.get("val_best_metric", 0.0)
            ),
            "gate": distillation.get("gate", "none"),
            "logit_weight": float(distillation.get("logit_weight", 0.0)),
            "embedding_weight": float(
                distillation.get("embedding_weight", 0.0)
            ),
            "relation_weight": float(
                distillation.get("relation_weight", 0.0)
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        })

    rows.sort(key=lambda row: row["validation_chd_auc"], reverse=True)
    baseline = next(row for row in rows if row["variant"] == "baseline")
    for row in rows:
        row["delta_chd_auc_vs_baseline"] = (
            row["validation_chd_auc"] - baseline["validation_chd_auc"]
        )
        row["delta_macro_auc_vs_baseline"] = (
            row["validation_macro_auc"] - baseline["validation_macro_auc"]
        )

    csv_path = output_dir / "selective_ppg_distillation_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "protocol": "validation_only_test_sealed",
        "pretrain_checkpoint": args.pretrain,
        "pretrain_sha256": sha256(Path(args.pretrain)),
        "teacher_checkpoint": args.teacher,
        "teacher_sha256": sha256(Path(args.teacher)),
        "split": args.split,
        "split_sha256": sha256(Path(args.split)),
        "runs": rows,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    best = rows[0]
    lines = [
        "# 选择性双通道教师到PPG学生蒸馏：验证集初筛",
        "",
        "本阶段严格封存测试集，仅用于选择是否保留方案一。",
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
        f"当前验证集最佳方案为 **{best['variant']}**，相对PPG基线的 "
        f"CHD AUC变化为 **{best['delta_chd_auc_vs_baseline']:+.4f}**。",
        "",
        "只有在CHD AUC提高且Macro AUC没有明显下降时，才进入三随机种子复现。",
        "本表不能作为最终测试集结论。",
    ])
    (output_dir / "analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[Complete] summary={csv_path} best={best['variant']}")


if __name__ == "__main__":
    main()
