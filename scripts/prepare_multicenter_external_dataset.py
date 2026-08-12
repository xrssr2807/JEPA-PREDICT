#!/usr/bin/env python3
"""Link multicenter diagnoses and build deterministic PPG-only MIL inputs."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = [
    "高血压",
    "高血糖",
    "高血脂",
    "其他疾病",
    "冠心病",
    "心律失常（房颤、频发早搏等）",
    "糖尿病",
    "颈动脉斑块",
]

POSITIVE_PATTERNS = {
    "高血压": (r"高血压",),
    "高血糖": (r"高血糖",),
    "高血脂": (r"高脂血症", r"高血脂", r"高胆固醇", r"高甘油三酯"),
    "其他疾病": (
        r"脑梗", r"脑卒中", r"中风", r"脑出血",
        r"下肢动脉闭塞", r"下肢动脉硬化闭塞",
    ),
    "冠心病": (
        r"冠心病", r"冠状动脉粥样硬化", r"CAD\s*-?\s*RADS",
        r"冠状动脉支架", r"冠脉支架", r"冠状动脉搭桥", r"不稳定心绞痛",
    ),
    "心律失常（房颤、频发早搏等）": (
        r"心律失常", r"房颤", r"心房颤动", r"房早", r"室早",
        r"房性早搏", r"室性早搏", r"室上早", r"频发早搏", r"偶发室上早",
    ),
    "糖尿病": (r"糖尿病",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--diagnosis-xlsx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows-per-patient", type=int, default=8)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Physical duration of each exported model window",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option: fixed samples per window. "
            "Prefer --window-seconds so duration is correct across rates"
        ),
    )
    return parser.parse_args()


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    result = str(value).strip().replace("\u00a0", "").replace(" ", "").replace("\n", "")
    if re.fullmatch(r"\d+\.0", result):
        result = result[:-2]
    return result


def canonical_hr_id(value: str) -> str:
    return value[2:] if re.fullmatch(r"HR\d+", value) else value


def diagnosis_text(row: pd.Series) -> str:
    values = []
    for value in row.iloc[5:]:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            values.append(text)
    return " | ".join(values)


def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def derive_labels(text: str) -> tuple[dict[str, int], dict[str, int], list[str]]:
    labels = {name: 0 for name in LABELS}
    valid = {name: int(bool(text)) for name in LABELS}
    notes: list[str] = []
    if not text:
        return labels, valid, ["diagnosis_missing"]

    for name, patterns in POSITIVE_PATTERNS.items():
        labels[name] = int(contains_any(text, patterns))

    labels["颈动脉斑块"] = int(
        "颈动脉斑块" in text
        or ("颈动脉" in text and ("粥样硬化" in text or "斑块" in text))
    )

    # A question mark denotes an unconfirmed diagnosis unless another explicit
    # coronary diagnosis is present in the same row.
    if re.search(r"冠心病\s*[？?]", text):
        definitive = re.sub(r"冠心病\s*[？?]", "", text)
        if not contains_any(definitive, POSITIVE_PATTERNS["冠心病"]):
            labels["冠心病"] = 0
            valid["冠心病"] = 0
            notes.append("chd_uncertain")
    return labels, valid, notes


def match_patients(
    diagnosis: pd.DataFrame, patient_map: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_by_exact: dict[str, dict[str, str]] = {}
    source_by_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in patient_map.to_dict("records"):
        source_id = normalize_id(row["source_patient_id"])
        item = {"patient_uid": str(row["patient_uid"]), "source_patient_id": source_id}
        source_by_exact[source_id] = item
        source_by_canonical[canonical_hr_id(source_id)].append(item)

    matched: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    for _, row in diagnosis.iterrows():
        diagnosis_id = normalize_id(row.iloc[0])
        match = source_by_exact.get(diagnosis_id)
        method = "exact"
        if match is None:
            candidates = source_by_canonical.get(canonical_hr_id(diagnosis_id), [])
            if len(candidates) == 1:
                match = candidates[0]
                method = "unique_hr_prefix_normalization"
        if match is None:
            unresolved.append({"diagnosis_id": diagnosis_id, "reason": "no_unique_match"})
            continue
        text = diagnosis_text(row)
        labels, valid, notes = derive_labels(text)
        matched.append(
            {
                **match,
                "diagnosis_id": diagnosis_id,
                "diagnosis_text": text,
                "match_method": method,
                "labels": labels,
                "label_valid": valid,
                "notes": ";".join(notes),
            }
        )
    return matched, unresolved


def _payload_rate(payload: dict[str, Any]) -> float:
    value = payload.get("fs", payload.get("sampling_rate", 25.0))
    rate = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid processed sampling rate: {value!r}")
    return rate


def select_patient_windows(
    paths: list[Path],
    count: int,
    window_seconds: float,
    window_samples_override: int | None = None,
) -> list[tuple[Path, int, int, float]]:
    descriptors: list[tuple[Path, int, int, float]] = []
    for path in sorted(paths):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        length = int(np.asarray(payload["data"]).shape[-1])
        rate = _payload_rate(payload)
        window_samples = (
            max(1, int(window_samples_override))
            if window_samples_override is not None
            else max(1, int(round(window_seconds * rate)))
        )
        for start in range(0, length - window_samples + 1, window_samples):
            descriptors.append((path, start, window_samples, rate))
    if len(descriptors) <= count:
        return descriptors
    chosen = np.linspace(0, len(descriptors) - 1, count).round().astype(int)
    return [descriptors[index] for index in chosen]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_root).resolve()
    output = Path(args.output).resolve()
    model_dir = output / "model_input"
    private_dir = output / "private"
    model_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    diagnosis = pd.read_excel(args.diagnosis_xlsx, sheet_name=0, dtype=object)
    patient_map = pd.read_csv(processed / "private" / "patient_id_map.csv", dtype=str)
    matched, unresolved = match_patients(diagnosis, patient_map)
    pkl_by_patient: dict[str, list[Path]] = defaultdict(list)
    for path in (processed / "records").rglob("*.pkl"):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        pkl_by_patient[str(payload.get("patient_uid", ""))].append(path)

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    positive_counts = Counter()
    valid_counts = Counter()
    patients_with_windows = 0
    patients_without_diagnosis = 0

    for item in matched:
        patient_uid = item["patient_uid"]
        external_uid = patient_uid.replace("_", "")
        paths = pkl_by_patient.get(patient_uid, [])
        has_diagnosis = bool(item["diagnosis_text"])
        if not has_diagnosis:
            patients_without_diagnosis += 1
        selected = (
            select_patient_windows(
                paths,
                args.windows_per_patient,
                args.window_seconds,
                args.window_samples,
            )
            if paths and has_diagnosis
            else []
        )
        if selected:
            patients_with_windows += 1
        for label in LABELS:
            if selected and item["label_valid"][label]:
                valid_counts[label] += 1
                positive_counts[label] += item["labels"][label]

        public = {
            "uid": external_uid,
            "match_method": item["match_method"],
            "has_diagnosis": int(has_diagnosis),
            "has_ppg": int(bool(paths)),
            "selected_windows": len(selected),
            "notes": item["notes"],
        }
        for label in LABELS:
            public[f"label::{label}"] = item["labels"][label]
            public[f"valid::{label}"] = item["label_valid"][label]
        public_rows.append(public)
        private_rows.append(
            {
                "uid": external_uid,
                "source_patient_id": item["source_patient_id"],
                "diagnosis_id": item["diagnosis_id"],
                "diagnosis_text": item["diagnosis_text"],
                "match_method": item["match_method"],
                "notes": item["notes"],
            }
        )

        cache: dict[Path, dict[str, Any]] = {}
        for segment_index, (path, start, window_samples, source_rate) in enumerate(selected):
            if path not in cache:
                with path.open("rb") as handle:
                    cache[path] = pickle.load(handle)
            source_payload = cache[path]
            data = np.asarray(source_payload["data"], dtype=np.float32)[
                :, start : start + window_samples
            ]
            filename = f"test_{external_uid}_seg{segment_index:03d}.pkl"
            payload = {
                "uid": external_uid,
                # Keep the portable artifact independent of the NumPy major
                # version used to create it. The downstream Dataset converts
                # this standard nested list back to float32 with np.asarray.
                "data": data.tolist(),
                "sampling_rate": source_rate,
                "device_origin_rate_hz": 25.0,
                "source_record_uid": source_payload.get("record_uid"),
                "source_start_sample": start,
                "label": item["labels"],
                "label_valid": item["label_valid"],
                "external_validation": True,
            }
            with (model_dir / filename).open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            window_rows.append(
                {
                    "uid": external_uid,
                    "filename": filename,
                    "source_processed_file": str(path.relative_to(processed)),
                    "source_start_sample": start,
                    "samples": window_samples,
                    "sampling_rate_hz": source_rate,
                    "duration_seconds": window_samples / source_rate,
                }
            )

    public_fields = [
        "uid", "match_method", "has_diagnosis", "has_ppg", "selected_windows", "notes",
        *[f"label::{name}" for name in LABELS],
        *[f"valid::{name}" for name in LABELS],
    ]
    write_csv(output / "external_patient_labels.csv", public_rows, public_fields)
    write_csv(
        private_dir / "label_linkage.csv",
        private_rows,
        ["uid", "source_patient_id", "diagnosis_id", "diagnosis_text", "match_method", "notes"],
    )
    write_csv(
        output / "window_manifest.csv",
        window_rows,
        [
            "uid", "filename", "source_processed_file", "source_start_sample",
            "samples", "sampling_rate_hz", "duration_seconds",
        ],
    )
    write_csv(
        private_dir / "unresolved_diagnosis_ids.csv",
        unresolved,
        ["diagnosis_id", "reason"],
    )

    summary = {
        "diagnosis_rows": len(diagnosis),
        "matched_patients": len(matched),
        "exact_matches": sum(x["match_method"] == "exact" for x in matched),
        "normalized_matches": sum(x["match_method"] != "exact" for x in matched),
        "unresolved_diagnosis_ids": len(unresolved),
        "matched_without_diagnosis_text": patients_without_diagnosis,
        "patients_with_model_windows": patients_with_windows,
        "model_windows": len(window_rows),
        "window_seconds": args.window_seconds,
        "legacy_window_samples": args.window_samples,
        "processed_sampling_rate_note": (
            "ALM exports are interpolated to their payload fs; source wearable "
            "acquisition is 25 Hz and is matched during retrospective training"
        ),
        "windows_per_patient_max": args.windows_per_patient,
        "positive_patient_counts": dict(positive_counts),
        "valid_patient_counts": dict(valid_counts),
        "channel": "ppg",
        "test_only_external_cohort": True,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
