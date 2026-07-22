#!/usr/bin/env python3
"""Collect final downstream metrics for phase and channel ablations."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
PATTERNS = {
    "test_acc": re.compile(rf"Best Test Acc:\s*{FLOAT}"),
    "macro_auc": re.compile(rf"Best Test AUC \(macro\):\s*{FLOAT}"),
    "chd_auc": re.compile(rf"CHD/[^\n]*?AUC:\s*{FLOAT}"),
    "macro_precision": re.compile(rf"Precision \(macro\):\s*{FLOAT}"),
    "macro_recall": re.compile(rf"Recall \(macro\):\s*{FLOAT}"),
    "macro_f1": re.compile(rf"F1 \(macro\):\s*{FLOAT}"),
}
CHD_PATTERN = re.compile(
    rf"CHD/[^\n]*?AUC:\s*{FLOAT}\s*"
    rf"\(P={FLOAT},\s*R={FLOAT},\s*F1={FLOAT},\s*support=([0-9]+)\)"
)


def last_match(pattern: re.Pattern[str], text: str) -> str:
    matches = pattern.findall(text)
    if not matches:
        return ""
    value = matches[-1]
    return value[0] if isinstance(value, tuple) else value


def read_experiment(root: Path, phase: str, channel: str) -> dict[str, str]:
    experiment = f"{phase}_{channel}"
    experiment_dir = root / experiment
    candidates = (
        experiment_dir / "downstream_log.txt",
        experiment_dir / "console.log",
    )
    log_path = next((path for path in candidates if path.is_file()), None)
    row = {
        "experiment": experiment,
        "phase": phase,
        "channel": channel,
        "log": str(log_path or ""),
    }
    if log_path is None:
        row.update({name: "" for name in PATTERNS})
        row.update(
            {
                "chd_precision": "",
                "chd_recall": "",
                "chd_f1": "",
                "chd_support": "",
                "status": "missing_log",
            }
        )
        return row

    text = log_path.read_text(encoding="utf-8", errors="replace")
    row.update({name: last_match(pattern, text) for name, pattern in PATTERNS.items()})

    chd_matches = CHD_PATTERN.findall(text)
    if chd_matches:
        _, precision, recall, f1, support = chd_matches[-1]
        row.update(
            {
                "chd_precision": precision,
                "chd_recall": recall,
                "chd_f1": f1,
                "chd_support": support,
            }
        )
    else:
        row.update(
            {
                "chd_precision": "",
                "chd_recall": "",
                "chd_f1": "",
                "chd_support": "",
            }
        )

    row["status"] = "complete" if row["chd_auc"] else "incomplete"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=["phase2", "phase3a"])
    parser.add_argument("--channels", nargs="+", default=["ppg", "ecg", "both"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        read_experiment(args.root, phase, channel)
        for phase in args.phases
        for channel in args.channels
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "phase",
        "channel",
        "test_acc",
        "macro_auc",
        "chd_auc",
        "chd_precision",
        "chd_recall",
        "chd_f1",
        "chd_support",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "status",
        "log",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Summary] wrote {args.output}")
    for row in rows:
        print(
            f"{row['experiment']:13s} status={row['status']:11s} "
            f"macro_auc={row['macro_auc'] or '-':>6s} "
            f"chd_auc={row['chd_auc'] or '-':>6s} "
            f"chd_recall={row['chd_recall'] or '-':>6s}"
        )


if __name__ == "__main__":
    main()
