#!/usr/bin/env python3
"""Collect final PPG-only downstream metrics from phase-ablation logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
PATTERNS = {
    "macro_auc": re.compile(rf"Best Test AUC \(macro\):\s*{FLOAT}"),
    "chd_auc": re.compile(rf"CHD/冠心病 AUC:\s*{FLOAT}"),
    "precision": re.compile(rf"Precision \(macro\):\s*{FLOAT}"),
    "recall": re.compile(rf"Recall \(macro\):\s*{FLOAT}"),
    "f1": re.compile(rf"F1 \(macro\):\s*{FLOAT}"),
}
PHASES = ("phase0", "phase1", "phase2", "phase3a")


def last_match(pattern: re.Pattern[str], text: str) -> str:
    matches = pattern.findall(text)
    return matches[-1] if matches else ""


def read_phase(root: Path, phase: str) -> dict[str, str]:
    phase_dir = root / phase
    candidates = (phase_dir / "downstream_log.txt", phase_dir / "console.log")
    log_path = next((path for path in candidates if path.is_file()), None)
    row = {"phase": phase, "log": str(log_path or "")}
    if log_path is None:
        row.update({name: "" for name in PATTERNS})
        row["status"] = "missing_log"
        return row

    text = log_path.read_text(encoding="utf-8", errors="replace")
    row.update({name: last_match(pattern, text) for name, pattern in PATTERNS.items()})
    row["status"] = "complete" if row["chd_auc"] else "incomplete"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [read_phase(args.root, phase) for phase in PHASES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phase", "macro_auc", "chd_auc", "precision", "recall", "f1", "status", "log"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Summary] wrote {args.output}")
    for row in rows:
        print(
            f"{row['phase']:7s} status={row['status']:11s} "
            f"macro_auc={row['macro_auc'] or '-':>6s} "
            f"chd_auc={row['chd_auc'] or '-':>6s} f1={row['f1'] or '-':>6s}"
        )


if __name__ == "__main__":
    main()
