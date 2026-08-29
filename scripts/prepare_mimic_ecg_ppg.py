#!/usr/bin/env python3
"""Convert local MIMIC-III WFDB ECG/PLETH segments to PhysioV2 .pt files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.data import compute_signal_stats


PREPROCESS_VERSION = 1
ECG_LEAD_PRIORITY = (
    "II", "MLII", "ECG II", "I", "V", "V1", "V2", "V3", "V4", "V5", "V6",
)
PPG_NAMES = ("PLETH", "PPG", "PULSE")


def _canonical_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(name).upper()).strip()


def choose_channels(signal_names):
    """Return ECG/PPG indices without silently mixing arbitrary channels."""
    canonical = [_canonical_name(name) for name in signal_names]
    ppg_index = next(
        (i for i, name in enumerate(canonical) if name in PPG_NAMES), None
    )
    ecg_index = None
    for preferred in ECG_LEAD_PRIORITY:
        target = _canonical_name(preferred)
        ecg_index = next(
            (i for i, name in enumerate(canonical) if name == target), None
        )
        if ecg_index is not None:
            break
    if ecg_index is None:
        ecg_index = next(
            (
                i for i, name in enumerate(canonical)
                if name.startswith("ECG") or name.startswith("ML")
            ),
            None,
        )
    return ecg_index, ppg_index


def infer_subject_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.fullmatch(r"p(\d{5,8})", part.lower())
        if match:
            return match.group(1)
    match = re.search(r"p(\d{5,8})", str(path).lower())
    if match:
        return match.group(1)
    return hashlib.sha1(str(path.parent).encode("utf-8")).hexdigest()[:12]


def _interpolate_small_gaps(values: np.ndarray, minimum_finite: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if finite.mean() < minimum_finite or finite.sum() < 2:
        raise ValueError("too many non-finite samples")
    if not finite.all():
        positions = np.arange(values.size)
        values = np.interp(positions, positions[finite], values[finite])
    return values


def _resample(values: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if math.isclose(source_hz, target_hz, rel_tol=0.0, abs_tol=1e-6):
        return values.astype(np.float64, copy=False)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("Install scipy from requirements-mimic.txt") from exc
    ratio = Fraction(target_hz / source_hz).limit_denominator(1000)
    return resample_poly(values, ratio.numerator, ratio.denominator)


def _zscore(values: np.ndarray, clip: float) -> np.ndarray:
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError("flat signal")
    normalized = (values - float(np.mean(values))) / std
    if clip > 0:
        normalized = np.clip(normalized, -clip, clip)
    return normalized.astype(np.float32)


def signal_quality(values: np.ndarray) -> dict:
    """Cheap modality-agnostic rejection metrics; no label information is used."""
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.std(values))
    span = float(np.ptp(values))
    if scale < 1e-6 or span < 1e-5:
        raise ValueError("flat signal")
    rounded = np.round(values, decimals=5)
    unique_ratio = float(np.unique(rounded).size / max(1, rounded.size))
    edge_fraction = float(
        np.mean((values == np.min(values)) | (values == np.max(values)))
    )
    if unique_ratio < 0.01:
        raise ValueError("low unique-value ratio")
    if edge_fraction > 0.20:
        raise ValueError("probable clipping")
    return {
        "std": scale,
        "range": span,
        "unique_ratio": unique_ratio,
        "edge_fraction": edge_fraction,
    }


def iter_physical_headers(root: Path):
    for header in sorted(root.rglob("*.hea")):
        if header.with_suffix(".dat").is_file():
            yield header


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare paired MIMIC ECG/PLETH windows for PhysioV2"
    )
    parser.add_argument("--mimic_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_seconds", type=float, default=30.0)
    parser.add_argument("--stride_seconds", type=float, default=30.0)
    parser.add_argument("--target_hz", type=float, default=100.0)
    parser.add_argument("--max_windows_per_patient", type=int, default=32)
    parser.add_argument("--max_patients", type=int, default=0)
    parser.add_argument("--minimum_finite", type=float, default=0.999)
    parser.add_argument("--normalize_clip", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import wfdb
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install -r requirements-mimic.txt") from exc

    root = Path(args.mimic_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"MIMIC root does not exist: {root}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "mimic_pretrain_manifest.csv"
    rows = []
    windows_per_patient = {}
    selected_patients = set()
    counts = {"headers": 0, "paired_headers": 0, "saved": 0, "rejected": 0}

    for header in iter_physical_headers(root):
        counts["headers"] += 1
        subject = infer_subject_id(header)
        if windows_per_patient.get(subject, 0) >= args.max_windows_per_patient:
            continue

        record_name = str(header.with_suffix(""))
        try:
            meta = wfdb.rdheader(record_name)
            ecg_index, ppg_index = choose_channels(meta.sig_name)
            if ecg_index is None or ppg_index is None:
                continue
            if subject not in selected_patients:
                if args.max_patients and len(selected_patients) >= args.max_patients:
                    continue
                selected_patients.add(subject)
            counts["paired_headers"] += 1
            source_hz = float(meta.fs)
            window_samples = int(round(args.window_seconds * source_hz))
            stride_samples = int(round(args.stride_seconds * source_hz))
            if window_samples < 2 or int(meta.sig_len) < window_samples:
                continue
        except Exception:
            counts["rejected"] += 1
            continue

        starts = range(0, int(meta.sig_len) - window_samples + 1, stride_samples)
        for start in starts:
            if windows_per_patient.get(subject, 0) >= args.max_windows_per_patient:
                break
            relative_record = header.relative_to(root).with_suffix("").as_posix()
            record_hash = hashlib.sha1(relative_record.encode("utf-8")).hexdigest()[:10]
            filename = (
                f"mimic_subject{subject}_rec{record_hash}_"
                f"seg{start:010d}.pt"
            )
            destination = output / filename
            if destination.exists() and not args.overwrite:
                windows_per_patient[subject] = windows_per_patient.get(subject, 0) + 1
                counts["saved"] += 1
                continue
            try:
                record = wfdb.rdrecord(
                    record_name,
                    sampfrom=start,
                    sampto=start + window_samples,
                    channels=[ecg_index, ppg_index],
                    physical=True,
                )
                signals = np.asarray(record.p_signal, dtype=np.float64)
                if signals.shape != (window_samples, 2):
                    raise ValueError(f"unexpected shape {signals.shape}")
                ecg_raw = _interpolate_small_gaps(signals[:, 0], args.minimum_finite)
                ppg_raw = _interpolate_small_gaps(signals[:, 1], args.minimum_finite)
                ecg_raw = _resample(ecg_raw, source_hz, args.target_hz)
                ppg_raw = _resample(ppg_raw, source_hz, args.target_hz)
                target_length = int(round(args.window_seconds * args.target_hz))
                ecg_raw = ecg_raw[:target_length]
                ppg_raw = ppg_raw[:target_length]
                if len(ecg_raw) != target_length or len(ppg_raw) != target_length:
                    raise ValueError("resampled window has the wrong length")
                ecg_quality = signal_quality(ecg_raw)
                ppg_quality = signal_quality(ppg_raw)
                payload = {
                    "ecg": torch.from_numpy(
                        _zscore(ecg_raw, args.normalize_clip)[None, :]
                    ),
                    "ppg": torch.from_numpy(
                        _zscore(ppg_raw, args.normalize_clip)[None, :]
                    ),
                    "ecg_stats": torch.from_numpy(
                        compute_signal_stats(ecg_raw).astype(np.float32)
                    ),
                    "uid": f"mimic:{subject}",
                    "source_id": 1,
                    "source": "mimic3wdb-matched",
                    "record_id": relative_record,
                    "lead_name": str(meta.sig_name[ecg_index]),
                    "source_fs": source_hz,
                    "target_fs": float(args.target_hz),
                    "window_start_sample": int(start),
                    "window_seconds": float(args.window_seconds),
                    "ecg_quality": ecg_quality,
                    "ppg_quality": ppg_quality,
                    "preprocess_version": PREPROCESS_VERSION,
                }
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, destination)
                rows.append({
                    "file": filename,
                    "subject_id": f"mimic:{subject}",
                    "record_id": relative_record,
                    "lead_name": str(meta.sig_name[ecg_index]),
                    "ppg_name": str(meta.sig_name[ppg_index]),
                    "source_fs": source_hz,
                    "target_fs": args.target_hz,
                    "start_sample": start,
                })
                windows_per_patient[subject] = windows_per_patient.get(subject, 0) + 1
                counts["saved"] += 1
            except Exception:
                counts["rejected"] += 1

    fieldnames = (
        "file", "subject_id", "record_id", "lead_name", "ppg_name",
        "source_fs", "target_fs", "start_sample",
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **counts,
        "patients": len(windows_per_patient),
        "mimic_root": str(root),
        "output_dir": str(output),
        "window_seconds": args.window_seconds,
        "target_hz": args.target_hz,
        "test_set_used": False,
    }
    (output / "mimic_pretrain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if counts["saved"] < 2 or len(windows_per_patient) < 2:
        raise SystemExit(f"Too few usable MIMIC windows: {summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
