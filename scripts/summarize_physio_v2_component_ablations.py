"""Capture and aggregate sealed PhysioV2 component-ablation results."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(args):
    expected_split = sha256(args.split)
    pretrain = torch.load(args.pretrain, map_location="cpu", weights_only=False)
    downstream = torch.load(
        args.downstream, map_location="cpu", weights_only=False
    )
    if downstream.get("test_status") != "sealed":
        raise ValueError("Downstream checkpoint does not seal the test set")
    recorded_split = (downstream.get("data_split") or {}).get("sha256")
    if recorded_split != expected_split:
        raise ValueError(
            f"Split mismatch: expected={expected_split} actual={recorded_split}"
        )
    validation = downstream.get("validation_metrics") or {}
    result = {
        "variant": args.variant,
        "pretrain_seed": int(args.seed),
        "downstream_seed": int(args.downstream_seed),
        "val_chd_auc": float(downstream["val_chd_auc"]),
        "val_macro_auc": float(validation.get("auc", downstream["val_auc"])),
        "val_macro_f1": float(validation.get("f1", downstream["val_f1"])),
        "test_status": "sealed",
        "split_sha256": recorded_split,
        "pretrain_sha256": sha256(args.pretrain),
        "downstream_sha256": sha256(args.downstream),
        "phase2_config": pretrain.get("phase2_config") or {},
        "pretrain_epoch": pretrain.get("epoch"),
        "pretrain_val_loss": pretrain.get(
            "best_val_loss", pretrain.get("val_loss")
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[Capture] {args.variant} CHD={result['val_chd_auc']:.6f} "
        f"Macro={result['val_macro_auc']:.6f} test=sealed"
    )


def aggregate(args):
    paper_dir = Path(args.paper_dir)
    rows = []
    missing = []
    for variant in args.variants:
        path = paper_dir / variant / "result.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError(
            "Ablation matrix is incomplete:\n" + "\n".join(missing)
        )
    full = next((row for row in rows if row["variant"] == "full"), None)
    if full is None:
        raise ValueError("The full PhysioV2 reference is required")
    flat_rows = []
    for row in rows:
        flat_rows.append({
            "variant": row["variant"],
            "pretrain_seed": row["pretrain_seed"],
            "val_chd_auc": row["val_chd_auc"],
            "delta_chd_vs_full": row["val_chd_auc"] - full["val_chd_auc"],
            "val_macro_auc": row["val_macro_auc"],
            "delta_macro_vs_full": (
                row["val_macro_auc"] - full["val_macro_auc"]
            ),
            "val_macro_f1": row["val_macro_f1"],
            "test_status": row["test_status"],
            "split_sha256": row["split_sha256"],
        })
    with (paper_dir / "aggregate.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    ranked = sorted(
        flat_rows, key=lambda row: row["val_chd_auc"], reverse=True
    )
    payload = {
        "scope": "seed-42 validation-only component screening",
        "test_status": "sealed",
        "full_val_chd_auc": full["val_chd_auc"],
        "runs": flat_rows,
        "largest_chd_drops": sorted(
            [row for row in flat_rows if row["variant"] != "full"],
            key=lambda row: row["delta_chd_vs_full"],
        ),
    }
    (paper_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# PhysioV2-v2 component ablation",
        "",
        "Seed-42 validation screening; the sealed test set was not accessed.",
        "",
        "| Variant | CHD AUROC | Delta vs full | Macro AUROC |",
        "|---|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['variant']} | {row['val_chd_auc']:.4f} | "
            f"{row['delta_chd_vs_full']:+.4f} | {row['val_macro_auc']:.4f} |"
        )
    lines.extend([
        "",
        "Only the two largest reproducible CHD drops should advance to "
        "independent pretraining seeds 3407 and 2026.",
    ])
    (paper_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[Complete] PhysioV2 component summary | test_set_sealed=True")


def select_top(args):
    payload = json.loads(
        (Path(args.paper_dir) / "summary.json").read_text(encoding="utf-8")
    )
    selected = [
        row["variant"]
        for row in payload["largest_chd_drops"][: args.select_top_k]
    ]
    print(" ".join(selected))


def aggregate_multiseed(args):
    rows = []
    missing = []
    for directory in map(Path, args.multiseed_dirs):
        for variant in args.variants:
            path = directory / variant / "result.json"
            if not path.is_file():
                missing.append(str(path))
            else:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError(
            "Multi-seed matrix is incomplete:\n" + "\n".join(missing)
        )
    by_seed = {
        (int(row["pretrain_seed"]), row["variant"]): row for row in rows
    }
    seeds = sorted({int(row["pretrain_seed"]) for row in rows})
    summary_rows = []
    for variant in args.variants:
        values = [by_seed[(seed, variant)]["val_chd_auc"] for seed in seeds]
        macro = [by_seed[(seed, variant)]["val_macro_auc"] for seed in seeds]
        deltas = [
            by_seed[(seed, variant)]["val_chd_auc"]
            - by_seed[(seed, "full")]["val_chd_auc"]
            for seed in seeds
        ]
        summary_rows.append({
            "variant": variant,
            "n_seeds": len(seeds),
            "chd_auc_mean": statistics.mean(values),
            "chd_auc_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "macro_auc_mean": statistics.mean(macro),
            "macro_auc_sd": statistics.stdev(macro) if len(macro) > 1 else 0.0,
            "delta_chd_vs_full_mean": statistics.mean(deltas),
            "worse_than_full_seeds": sum(delta < 0 for delta in deltas),
        })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "aggregate_multiseed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# PhysioV2-v2 multi-seed component replication",
        "",
        f"Validation only; seeds={seeds}; sealed test set was not accessed.",
        "",
        "| Variant | CHD AUROC mean+/-SD | Delta vs full | Worse seeds |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['chd_auc_mean']:.4f}+/-"
            f"{row['chd_auc_sd']:.4f} | "
            f"{row['delta_chd_vs_full_mean']:+.4f} | "
            f"{row['worse_than_full_seeds']}/{row['n_seeds']} |"
        )
    (output / "summary_multiseed.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_multiseed.json").write_text(
        json.dumps(
            {"seeds": seeds, "test_status": "sealed", "runs": summary_rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[Complete] multi-seed component replication | test_set_sealed=True")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--pretrain")
    parser.add_argument("--downstream")
    parser.add_argument("--split")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downstream_seed", type=int, default=42)
    parser.add_argument("--paper_dir")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--select_top_k", type=int, default=0)
    parser.add_argument("--multiseed_dirs", nargs="+")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    if args.capture:
        required = (
            args.variant,
            args.pretrain,
            args.downstream,
            args.split,
            args.output,
        )
        if not all(required):
            parser.error("capture mode requires variant/pretrain/downstream/split/output")
        capture(args)
    elif args.select_top_k:
        if not args.paper_dir:
            parser.error("selection mode requires paper_dir")
        select_top(args)
    elif args.multiseed_dirs:
        if not args.variants or not args.output_dir:
            parser.error("multi-seed mode requires variants and output_dir")
        if "full" not in args.variants:
            parser.error("multi-seed mode requires the full variant")
        aggregate_multiseed(args)
    else:
        if not args.paper_dir or not args.variants:
            parser.error("aggregate mode requires paper_dir and variants")
        aggregate(args)


if __name__ == "__main__":
    main()
