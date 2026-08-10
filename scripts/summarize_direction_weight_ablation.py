"""Summarize sealed Phase 2 asymmetric direction-weight experiments."""

import argparse
import csv
import hashlib
import json
import math
import os

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_tag(alpha):
    return f"a{round(float(alpha) * 100):03d}"


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrain_prefix", default="outputs_phase2_direction_weight"
    )
    parser.add_argument(
        "--downstream_prefix", default="outputs_direction_weight"
    )
    parser.add_argument(
        "--symmetric_checkpoint",
        default="outputs_phase2_physio_v2_seed42/jepa_best.pt",
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--paper_dir", required=True)
    args = parser.parse_args()

    expected_split_hash = sha256(args.split)
    rows = []
    missing = []
    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        pretrain_path = (
            args.symmetric_checkpoint
            if math.isclose(alpha, 1.0, abs_tol=1e-12)
            else os.path.join(
                f"{args.pretrain_prefix}_{tag}_seed{args.seed}",
                "jepa_best.pt",
            )
        )
        downstream_dir = (
            f"{args.downstream_prefix}_{tag}_both_seed{args.seed}"
        )
        downstream_path = os.path.join(
            downstream_dir, "downstream_multidisease_best.pt"
        )
        predictions_path = os.path.join(
            downstream_dir, "validation_patient_predictions.csv"
        )
        for path in (pretrain_path, downstream_path, predictions_path):
            if not os.path.isfile(path):
                missing.append(path)
        if any(
            not os.path.isfile(path)
            for path in (pretrain_path, downstream_path, predictions_path)
        ):
            continue

        pretrain = torch.load(
            pretrain_path, map_location="cpu", weights_only=False
        )
        phase2 = pretrain.get("phase2_config") or {}
        recorded_alpha = phase2.get("reverse_loss_weight")
        alpha_status = "recorded"
        if recorded_alpha is None and math.isclose(alpha, 1.0, abs_tol=1e-12):
            recorded_alpha = 1.0
            alpha_status = "legacy_symmetric_inferred"
        if recorded_alpha is None or not math.isclose(
            float(recorded_alpha), alpha, abs_tol=1e-12
        ):
            raise ValueError(
                f"Pretrain alpha mismatch for {pretrain_path}: "
                f"expected={alpha} recorded={recorded_alpha}"
            )

        downstream = torch.load(
            downstream_path, map_location="cpu", weights_only=False
        )
        if downstream.get("test_status") != "sealed":
            raise ValueError(f"Test set is not sealed: {downstream_path}")
        split_hash = (downstream.get("data_split") or {}).get("sha256", "")
        if split_hash != expected_split_hash:
            raise ValueError(
                f"Split mismatch for {downstream_path}: {split_hash}"
            )
        validation = downstream.get("validation_metrics") or {}
        rows.append({
            "alpha": alpha,
            "forward_weight": 1.0,
            "reverse_weight": alpha,
            "normalization_denominator": 1.0 + alpha,
            "seed": args.seed,
            "val_chd_auc": float(downstream["val_chd_auc"]),
            "val_macro_auc": float(
                validation.get("auc", downstream["val_auc"])
            ),
            "val_f1": float(validation.get("f1", downstream["val_f1"])),
            "test_status": "sealed",
            "split_sha256": split_hash,
            "pretrain_sha256": sha256(pretrain_path),
            "alpha_metadata": alpha_status,
            "pretrain_checkpoint": os.path.abspath(pretrain_path),
            "downstream_checkpoint": os.path.abspath(downstream_path),
            "validation_predictions": os.path.abspath(predictions_path),
        })

    if missing:
        raise FileNotFoundError(
            "Direction-weight ablation is incomplete:\n" + "\n".join(missing)
        )
    if len(rows) != len(args.alphas):
        raise RuntimeError("Not every requested alpha produced one result row")

    ranked = sorted(
        rows,
        key=lambda row: (row["val_chd_auc"], row["val_macro_auc"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, 1):
        row["chd_primary_rank"] = rank
    selected = ranked[0]

    os.makedirs(args.paper_dir, exist_ok=True)
    write_csv(os.path.join(args.paper_dir, "direction_weight_runs.csv"), rows)
    write_csv(
        os.path.join(args.paper_dir, "direction_weight_ranked.csv"), ranked
    )
    summary = {
        "selection_scope": "single-seed pilot; validation only",
        "selection_rule": "highest validation CHD AUC, macro AUC as tie-breaker",
        "selected_alpha": selected["alpha"],
        "selected_val_chd_auc": selected["val_chd_auc"],
        "selected_val_macro_auc": selected["val_macro_auc"],
        "split_sha256": expected_split_hash,
        "test_status": "sealed",
        "runs": rows,
    }
    with open(
        os.path.join(args.paper_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.paper_dir, "summary.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("# 双向任务非对称权重实验\n\n")
        handle.write(f"- 暂定最优 alpha：`{selected['alpha']}`\n")
        handle.write(
            f"- 验证集冠心病 AUC：`{selected['val_chd_auc']:.4f}`\n"
        )
        handle.write(
            f"- 验证集 Macro AUC：`{selected['val_macro_auc']:.4f}`\n"
        )
        handle.write("- 测试集：封存，未参与选择\n")
        handle.write("- 说明：该结果仅用于单种子初筛，后续需独立预训练种子复现。\n")
    print(
        "[Complete] asymmetric direction-weight summary | "
        f"selected_alpha={selected['alpha']} | test_set_sealed=True"
    )


if __name__ == "__main__":
    main()
