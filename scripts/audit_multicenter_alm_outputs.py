#!/usr/bin/env python3
"""Validate ALM multicenter outputs and optionally compact arrays to float32."""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--convert-float32", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    manifest = root / "processed_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]

    files = sorted((root / "records").rglob("*.pkl"))
    fs_counts: collections.Counter[int] = collections.Counter()
    dtype_counts: collections.Counter[str] = collections.Counter()
    durations: list[float] = []
    invalid: list[dict[str, str]] = []
    patient_uids: set[str] = set()
    patient_record_pairs: set[tuple[str, str]] = set()

    for index, path in enumerate(files, 1):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            data = np.asarray(payload["data"])
            if data.ndim != 2 or data.shape[0] != 1:
                raise ValueError(f"shape={data.shape}")
            if data.shape[1] < 1 or not np.isfinite(data).all():
                raise ValueError("empty or non-finite signal")
            fs = int(payload["fs"])
            if args.convert_float32 and data.dtype != np.float32:
                payload["data"] = data.astype(np.float32)
                with path.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                data = payload["data"]
            fs_counts[fs] += 1
            dtype_counts[str(data.dtype)] += 1
            durations.append(float(data.shape[1] / fs))
            patient_uid = str(payload.get("patient_uid", ""))
            record_uid = str(payload.get("record_uid", ""))
            patient_uids.add(patient_uid)
            patient_record_pairs.add((patient_uid, record_uid))
        except Exception as exc:
            invalid.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if index % 200 == 0:
            print(f"[Audit] {index}/{len(files)}")

    complete_rows = [x for x in rows if x.get("status") in {"complete", "skipped"}]
    failed_rows = [x for x in rows if x.get("status") == "failed"]
    error_counts = collections.Counter(x.get("error", "unknown") for x in failed_rows)
    source_audit = {}
    if args.source:
        source = Path(args.source)
        source_audit = {
            "source_zip_files_after": len(list(source.rglob("*.zip"))),
            "source_partial_files_after": len(list(source.rglob("*.part"))),
        }

    duration_array = np.asarray(durations, dtype=np.float64)
    report = {
        "status": "PASS" if not invalid else "FAIL",
        "manifest_records": len(rows),
        "complete_records": len(complete_rows),
        "failed_records": len(failed_rows),
        "processed_patients": len(patient_uids - {""}),
        "processed_patient_record_pairs": len(
            {pair for pair in patient_record_pairs if pair[0] and pair[1]}
        ),
        "pkl_segments": len(files),
        "valid_pkl_segments": len(files) - len(invalid),
        "fs_counts": dict(fs_counts),
        "dtype_counts": dict(dtype_counts),
        "duration_seconds": {
            "min": float(duration_array.min()) if durations else None,
            "median": float(np.median(duration_array)) if durations else None,
            "p95": float(np.percentile(duration_array, 95)) if durations else None,
            "max": float(duration_array.max()) if durations else None,
            "total_hours": float(duration_array.sum() / 3600) if durations else 0.0,
        },
        "failure_reasons": dict(error_counts),
        "invalid_outputs": invalid,
        "source_modified": False,
        **source_audit,
    }
    (root / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
