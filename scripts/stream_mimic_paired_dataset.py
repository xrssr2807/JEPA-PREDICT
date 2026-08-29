#!/usr/bin/env python3
"""Stream paired MIMIC ECG/PLETH batches into PhysioV2 .pt windows.

Each raw batch is downloaded into a temporary directory, converted and
validated, then removed. The final output keeps only compact .pt windows,
batch manifests, and an aggregate manifest/summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_mimic_paired_fast.py"
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_mimic_ecg_ppg.py"


def run(command: list[str]) -> None:
    print("[Command] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--patients", type=int, default=1000)
    parser.add_argument("--batch_patients", type=int, default=100)
    parser.add_argument("--windows_per_patient", type=int, default=8)
    parser.add_argument("--window_seconds", type=float, default=30.0)
    parser.add_argument("--target_hz", type=float, default=100.0)
    parser.add_argument("--max_dat_mb", type=float, default=64.0)
    parser.add_argument("--max_records_per_patient", type=int, default=3)
    parser.add_argument("--scan_workers", type=int, default=16)
    parser.add_argument("--download_workers", type=int, default=6)
    parser.add_argument("--request_timeout", type=float, default=20.0)
    parser.add_argument("--keep_raw", action="store_true")
    args = parser.parse_args()
    if args.patients < 1 or args.batch_patients < 1:
        parser.error("patients and batch_patients must be positive")
    if args.windows_per_patient < 1:
        parser.error("windows_per_patient must be positive")
    return args


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def aggregate_manifests(paths: list[Path], destination: Path) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    seen_files: set[str] = set()
    for path in paths:
        current_fields, rows = read_manifest(path)
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise RuntimeError(f"manifest schema mismatch: {path}")
        for row in rows:
            if row["file"] in seen_files:
                raise RuntimeError(f"duplicate manifest file: {row['file']}")
            seen_files.add(row["file"])
            all_rows.append(row)
    if not fieldnames:
        raise RuntimeError("no completed batch manifests found")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    temporary.replace(destination)
    return all_rows


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    work = Path(args.work_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    batches_dir = output / "batch_metadata"
    batches_dir.mkdir(exist_ok=True)
    state_path = output / "stream_state.json"
    aggregate_manifest = output / "mimic_pretrain_manifest.csv"

    state = {
        "version": 1,
        "requested_patients": args.patients,
        "completed_patients": 0,
        "saved_windows": 0,
        "last_selected_patient": "",
        "batches": [],
    }
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("requested_patients") != args.patients:
            raise SystemExit(
                "Existing state uses a different --patients value. "
                "Use a new output directory or restore the original value."
            )

    while state["completed_patients"] < args.patients:
        batch_index = len(state["batches"])
        remaining = args.patients - state["completed_patients"]
        requested = min(args.batch_patients, remaining)
        raw_batch = work / f"raw_batch_{batch_index:04d}"
        processed_batch = work / f"pt_batch_{batch_index:04d}"
        shutil.rmtree(raw_batch, ignore_errors=True)
        shutil.rmtree(processed_batch, ignore_errors=True)
        raw_batch.mkdir(parents=True)
        processed_batch.mkdir(parents=True)

        download_command = [
            sys.executable,
            "-u",
            str(DOWNLOAD_SCRIPT),
            "--output_dir", str(raw_batch),
            "--patients", str(requested),
            "--windows_per_patient", str(args.windows_per_patient),
            "--window_seconds", str(args.window_seconds),
            "--max_dat_mb", str(args.max_dat_mb),
            "--max_records_per_patient", str(args.max_records_per_patient),
            "--scan_workers", str(args.scan_workers),
            "--download_workers", str(args.download_workers),
            "--request_timeout", str(args.request_timeout),
        ]
        if state["last_selected_patient"]:
            download_command.extend([
                "--start_after_patient",
                state["last_selected_patient"],
            ])
        run(download_command)
        download_summary = read_json(raw_batch / "download_summary.json")
        selected = int(download_summary["selected_patients"])
        if selected != requested:
            raise RuntimeError(
                f"download selected {selected} patients; expected {requested}"
            )

        run([
            sys.executable,
            "-u",
            str(PREPARE_SCRIPT),
            "--mimic_root", str(raw_batch),
            "--output_dir", str(processed_batch),
            "--window_seconds", str(args.window_seconds),
            "--stride_seconds", str(args.window_seconds),
            "--target_hz", str(args.target_hz),
            "--max_windows_per_patient", str(args.windows_per_patient),
        ])
        prepare_summary = read_json(
            processed_batch / "mimic_pretrain_summary.json"
        )
        prepared_patients = int(prepare_summary["patients"])
        saved_windows = int(prepare_summary["saved"])
        if prepared_patients < 1 or saved_windows < prepared_patients:
            raise RuntimeError(
                "batch conversion produced too few valid windows: "
                f"patients={prepared_patients}, windows={saved_windows}"
            )

        batch_manifest = processed_batch / "mimic_pretrain_manifest.csv"
        _, manifest_rows = read_manifest(batch_manifest)
        if len(manifest_rows) != saved_windows:
            raise RuntimeError(
                f"manifest rows {len(manifest_rows)} != saved windows "
                f"{saved_windows}"
            )
        archived_manifest = batches_dir / f"batch_{batch_index:04d}_manifest.csv"
        shutil.copy2(batch_manifest, archived_manifest)

        moved = 0
        for source in processed_batch.glob("*.pt"):
            destination = output / source.name
            if destination.exists():
                if sha256_file(source) != sha256_file(destination):
                    raise RuntimeError(f"conflicting output file: {destination}")
                source.unlink()
            else:
                source.replace(destination)
            moved += 1
        if moved != saved_windows:
            raise RuntimeError(f"moved {moved} .pt files; expected {saved_windows}")

        metadata = {
            "batch_index": batch_index,
            "requested_patients": requested,
            "selected_patients": selected,
            "prepared_patients": prepared_patients,
            "saved_windows": saved_windows,
            "downloaded_bytes": download_summary[
                "downloaded_bytes_including_existing"
            ],
            "first_patient": download_summary["patients"][0]["patient"],
            "last_patient": download_summary["last_selected_patient"],
        }
        write_json(batches_dir / f"batch_{batch_index:04d}.json", metadata)
        shutil.rmtree(processed_batch)
        if not args.keep_raw:
            shutil.rmtree(raw_batch)

        state["completed_patients"] += prepared_patients
        state["saved_windows"] += saved_windows
        state["last_selected_patient"] = download_summary[
            "last_selected_patient"
        ]
        state["batches"].append(metadata)
        write_json(state_path, state)
        print(
            f"[Batch complete] {state['completed_patients']}/{args.patients} "
            f"patients, {state['saved_windows']} windows; "
            f"raw_deleted={not args.keep_raw}",
            flush=True,
        )

    manifest_paths = [
        batches_dir / f"batch_{index:04d}_manifest.csv"
        for index in range(len(state["batches"]))
    ]
    rows = aggregate_manifests(manifest_paths, aggregate_manifest)
    subjects = {row["subject_id"] for row in rows}
    pt_files = list(output.glob("*.pt"))
    if len(pt_files) != len(rows):
        raise RuntimeError(
            f"final .pt count {len(pt_files)} != manifest rows {len(rows)}"
        )
    summary = {
        "source": "MIMIC-III Waveform Database Matched Subset",
        "streaming_download": True,
        "raw_deleted_after_verified_conversion": not args.keep_raw,
        "requested_patients": args.patients,
        "patients": len(subjects),
        "saved": len(pt_files),
        "window_seconds": args.window_seconds,
        "target_hz": args.target_hz,
        "windows_per_patient": args.windows_per_patient,
        "output_dir": str(output),
        "test_set_used": False,
    }
    write_json(output / "mimic_pretrain_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
