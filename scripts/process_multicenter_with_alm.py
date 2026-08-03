#!/usr/bin/env python3
"""Safely run the supplied ALM PPG pipeline on multicenter ZIP exports.

The supplied alm.py deletes each source ZIP after processing and writes outputs
beside the raw data.  This runner imports its signal-processing functions but
keeps the source tree read-only, deduplicates records through the JSONL export,
and writes pseudonymized outputs plus auditable manifests to a separate folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import secrets
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Record:
    source_zip: str
    source_relative: str
    patient_uid: str
    record_uid: str
    externalid: str
    healthid: str
    device_type: str
    start_time_ms: int | None
    end_time_ms: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-destructive multicenter preprocessing using alm.py"
    )
    parser.add_argument("--source", required=True, help="Raw multicenter root")
    parser.add_argument("--output", required=True, help="Separate output root")
    parser.add_argument("--alm-script", required=True, help="Supplied alm.py")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--light-type", choices=("green", "ir", "fusion"), default="green")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all records")
    parser.add_argument("--smallest-first", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_alm(path: str, light_type: str):
    spec = importlib.util.spec_from_file_location("external_alm_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import ALM pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET_LIGHT_TYPE = light_type
    install_alm_compatibility_shims(module)
    return module


def install_alm_compatibility_shims(module) -> None:
    """Preserve the original algorithm on pandas 3 / NumPy 2."""

    def healthy_segments(frame, _time_col):
        if frame is None or frame.empty:
            return []
        frame = frame.sort_values("Timestamp").reset_index(drop=True)
        time_diffs = frame["Timestamp"].diff().dt.total_seconds()
        break_indices = time_diffs[time_diffs > module.MAX_ALLOWED_GAP_SECONDS].index
        bounds = [0, *[int(x) for x in break_indices], len(frame)]
        segments = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            segment = frame.iloc[start:end].copy()
            if len(segment) < 2:
                continue
            duration = (
                segment["Timestamp"].iloc[-1] - segment["Timestamp"].iloc[0]
            ).total_seconds()
            if (
                duration >= module.MIN_SEGMENT_DURATION_SECONDS
                and duration <= module.MAX_SEGMENT_DURATION_HOURS * 3600
            ):
                segments.append(segment)
        return segments

    def process(frame):
        acc_cols = ["ACC_X", "ACC_Y", "ACC_Z"]
        valid_acc_cols = [column for column in acc_cols if column in frame.columns]
        if len(valid_acc_cols) == 3:
            acc_data = frame[valid_acc_cols].to_numpy()
            acc_mag = np.linalg.norm(acc_data, axis=1, keepdims=True)
        else:
            acc_data = np.zeros((len(frame), 3))
            acc_mag = np.zeros((len(frame), 1))

        ppg_raw = frame["PPG"].ffill().bfill().to_numpy()
        if module.INVERT_SIGNAL:
            ppg_raw = -ppg_raw
        frame["PPG_Filtered"] = module.BasicSignalProcessor.butter_bandpass_filter(
            ppg_raw,
            module.PPG_LOWCUT,
            module.PPG_HIGHCUT,
            module.FILTERING_SAMPLING_RATE,
        )
        filtered_acc = [
            module.BasicSignalProcessor.butter_lowpass_filter(
                acc_data[:, index],
                module.ACC_LOWPASS,
                module.FILTERING_SAMPLING_RATE,
            )
            for index in range(3)
        ]
        frame["ACC_X_Filtered"] = filtered_acc[0]
        frame["ACC_Y_Filtered"] = filtered_acc[1]
        frame["ACC_Z_Filtered"] = filtered_acc[2]
        frame["ACC_Mag_Filtered"] = (
            module.BasicSignalProcessor.butter_lowpass_filter(
                acc_mag.ravel(),
                module.ACC_LOWPASS,
                module.FILTERING_SAMPLING_RATE,
            )
        )
        return frame

    module._get_healthy_segments = healthy_segments
    module.BasicSignalProcessor.process = staticmethod(process)


def load_jsonl_rows(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl_files = sorted(source.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL metadata found under {source}")
    for path in jsonl_files:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def index_zip_files(source: Path) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = {}
    for path in source.rglob("*.zip"):
        by_name.setdefault(path.name, []).append(path)
    return by_name


def choose_zip(paths: list[Path]) -> Path:
    # Prefer the canonical downloaded tree over the recovery/other copy.
    return sorted(paths, key=lambda p: ("other" in {x.lower() for x in p.parts}, len(p.parts), str(p)))[0]


def load_or_create_salt(private_dir: Path) -> str:
    private_dir.mkdir(parents=True, exist_ok=True)
    path = private_dir / "pseudonym_salt.txt"
    if path.exists():
        return path.read_text(encoding="ascii").strip()
    salt = secrets.token_hex(32)
    path.write_text(salt + "\n", encoding="ascii")
    return salt


def build_records(source: Path, output: Path) -> tuple[list[Record], dict[str, Any]]:
    rows = load_jsonl_rows(source)
    zip_index = index_zip_files(source)
    salt = load_or_create_salt(output / "private")
    records: list[Record] = []
    missing: list[str] = []
    seen_zip_names: set[str] = set()
    private_map: dict[str, str] = {}

    for row_index, row in enumerate(rows):
        zip_name = PurePosixPath(str(row.get("sensorData", ""))).name
        if not zip_name:
            missing.append(f"metadata_row_{row_index}: empty sensorData")
            continue
        if zip_name in seen_zip_names:
            continue
        seen_zip_names.add(zip_name)
        candidates = zip_index.get(zip_name, [])
        if not candidates:
            missing.append(zip_name)
            continue

        externalid = str(row.get("externalid") or "").strip()
        healthid = str(row.get("healthid") or "").strip()
        raw_patient_key = externalid or healthid or f"missing-id:{zip_name}"
        patient_uid = "mc_" + sha256_text(f"{salt}:{raw_patient_key}")[:16]
        record_key = str(row.get("uniqueid") or zip_name)
        record_uid = "rec_" + sha256_text(record_key)[:16]
        time_payload = row.get("timeStamp") or {}
        source_zip = choose_zip(candidates)
        private_map[patient_uid] = raw_patient_key
        records.append(
            Record(
                source_zip=str(source_zip),
                source_relative=str(source_zip.relative_to(source)),
                patient_uid=patient_uid,
                record_uid=record_uid,
                externalid=externalid,
                healthid=healthid,
                device_type=str(row.get("deviceType") or ""),
                start_time_ms=time_payload.get("startTime"),
                end_time_ms=time_payload.get("endTime"),
            )
        )

    private_path = output / "private" / "patient_id_map.csv"
    with private_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient_uid", "source_patient_id"))
        writer.writeheader()
        for patient_uid, raw_id in sorted(private_map.items()):
            writer.writerow({"patient_uid": patient_uid, "source_patient_id": raw_id})

    audit = {
        "metadata_rows": len(rows),
        "unique_metadata_zip_names": len(seen_zip_names),
        "physical_zip_files": sum(len(v) for v in zip_index.values()),
        "duplicate_physical_copies": sum(max(0, len(v) - 1) for v in zip_index.values()),
        "complete_records": len(records),
        "patients": len({r.patient_uid for r in records}),
        "missing_records": missing,
        "partial_downloads_ignored": len(list(source.rglob("*.part"))),
    }
    return records, audit


def update_output_pickle(path: Path, record: Record, txt_member: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    data = np.asarray(payload["data"], dtype=np.float32)
    if data.ndim != 2 or data.shape[0] != 1:
        raise ValueError(f"Unexpected data shape {data.shape} in {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in {path}")
    payload.update(
        {
            "data": data,
            "patient_uid": record.patient_uid,
            "record_uid": record.record_uid,
            "device_type": record.device_type,
            "source_member": txt_member,
            "source_start_time_ms": record.start_time_ms,
            "source_end_time_ms": record.end_time_ms,
            "preprocessing": "alm_green_ppg_v1",
        }
    )
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "output_file": str(path),
        "samples": int(data.shape[1]),
        "duration_seconds": float(data.shape[1] / float(payload["fs"])),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def read_sensor_member(alm, archive: zipfile.ZipFile, member: zipfile.ZipInfo):
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "latin1"):
        try:
            with archive.open(member, "r") as handle:
                return alm.pd.read_csv(
                    handle,
                    sep="\t",
                    on_bad_lines="warn",
                    engine="c",
                    encoding=encoding,
                )
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def process_record(
    record_payload: dict[str, Any],
    alm_script: str,
    output_root: str,
    light_type: str,
    overwrite: bool,
) -> dict[str, Any]:
    record = Record(**record_payload)
    started = time.time()
    record_dir = Path(output_root) / "records" / record.patient_uid / record.record_uid
    done_file = record_dir / "complete.json"
    if done_file.exists() and not overwrite:
        return json.loads(done_file.read_text(encoding="utf-8")) | {"status": "skipped"}
    record_dir.mkdir(parents=True, exist_ok=True)
    alm = load_alm(alm_script, light_type)
    outputs: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(record.source_zip, "r") as archive:
            txt_members = [x for x in archive.infolist() if x.filename.lower().endswith(".txt")]
            if not txt_members:
                raise ValueError("ZIP contains no TXT sensor file")
            for member_index, member in enumerate(txt_members):
                frame = read_sensor_member(alm, archive, member)
                uid = f"{record.patient_uid}_{record.record_uid}_m{member_index:02d}"
                base = record_dir / f"{uid}_processed.pkl"
                metadata = alm.process_dataframe_to_pkl(frame, str(base), uid)
                for item in metadata:
                    output_path = record_dir / item["output_filename"]
                    outputs.append(update_output_pickle(output_path, record, member.filename))
        result = {
            "status": "complete",
            "patient_uid": record.patient_uid,
            "record_uid": record.record_uid,
            "source_relative": record.source_relative,
            "device_type": record.device_type,
            "segments": len(outputs),
            "outputs": outputs,
            "elapsed_seconds": time.time() - started,
        }
        done_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "patient_uid": record.patient_uid,
            "record_uid": record.record_uid,
            "source_relative": record.source_relative,
            "device_type": record.device_type,
            "segments": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }


def write_manifest(output: Path, results: list[dict[str, Any]]) -> None:
    manifest_path = output / "processed_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    alm_script = Path(args.alm_script).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not alm_script.is_file():
        raise FileNotFoundError(f"ALM script does not exist: {alm_script}")
    if source == output or source in output.parents:
        raise ValueError("Output must not equal or be nested inside the raw source tree")
    output.mkdir(parents=True, exist_ok=True)

    records, audit = build_records(source, output)
    if args.smallest_first:
        records.sort(key=lambda x: os.path.getsize(x.source_zip))
    else:
        records.sort(key=lambda x: (x.patient_uid, x.record_uid))
    if args.limit > 0:
        records = records[: args.limit]
    audit.update(
        {
            "selected_records": len(records),
            "source": str(source),
            "output": str(output),
            "alm_script": str(alm_script),
            "light_type": args.light_type,
            "source_modified": False,
        }
    )
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        return 0

    results: list[dict[str, Any]] = []
    worker_count = max(1, min(args.workers, len(records)))

    def accept_result(result: dict[str, Any], index: int) -> None:
        results.append(result)
        print(
            f"[{index}/{len(records)}] {result['status']} "
            f"{result['record_uid']} segments={result.get('segments', 0)} "
            f"seconds={result.get('elapsed_seconds', 0):.1f}"
        )
        write_manifest(output, results)

    if worker_count == 1:
        for index, record in enumerate(records, 1):
            result = process_record(
                asdict(record),
                str(alm_script),
                str(output),
                args.light_type,
                args.overwrite,
            )
            accept_result(result, index)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_record,
                    asdict(record),
                    str(alm_script),
                    str(output),
                    args.light_type,
                    args.overwrite,
                ): record
                for record in records
            }
            for index, future in enumerate(as_completed(futures), 1):
                accept_result(future.result(), index)

    summary = {
        "records_selected": len(records),
        "records_complete": sum(x["status"] in {"complete", "skipped"} for x in results),
        "records_failed": sum(x["status"] == "failed" for x in results),
        "segments": sum(int(x.get("segments", 0)) for x in results),
        "source_modified": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["records_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
