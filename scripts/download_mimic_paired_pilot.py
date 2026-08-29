#!/usr/bin/env python3
"""Selectively download MIMIC matched records containing ECG and PLETH."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_mimic_ecg_ppg import choose_channels


DEFAULT_BASE = (
    "https://archive.physionet.org/physiobank/database/"
    "mimic3wdb/matched/"
)


def fetch(
    url: str,
    *,
    binary: bool = False,
    attempts: int = 3,
    timeout: float = 60.0,
):
    request = urllib.request.Request(url, headers={"User-Agent": "PhysioV2-research/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            return payload if binary else payload.decode("utf-8", "replace")
        except (OSError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def remote_size(url: str, *, timeout: float = 60.0) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "PhysioV2-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers.get("Content-Length", 0))


def download(url: str, destination: Path) -> int:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "PhysioV2-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(temporary, destination)
    return destination.stat().st_size


def parse_master(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return [line.split()[0] for line in lines[1:] if line.split()[0] != "~"]


def parse_physical(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    first = lines[0].split()
    if "/" in first[0] or len(first) < 4:
        return None
    signal_count = int(first[1])
    fs = float(first[2].split("/")[0])
    signal_length = int(first[3])
    signal_lines = lines[1:1 + signal_count]
    names = [line.split()[-1] for line in signal_lines]
    ecg_index, ppg_index = choose_channels(names)
    if ecg_index is None or ppg_index is None:
        return None
    dat_files = sorted({line.split()[0] for line in signal_lines})
    return {
        "fs": fs,
        "signal_length": signal_length,
        "names": names,
        "dat_files": dat_files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--windows_per_patient", type=int, default=8)
    parser.add_argument("--window_seconds", type=float, default=30.0)
    parser.add_argument("--max_dat_mb", type=float, default=64.0)
    parser.add_argument("--max_records_per_patient", type=int, default=3)
    parser.add_argument("--base_url", default=DEFAULT_BASE)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/") + "/"
    index_text = fetch(base + "RECORDS-waveforms")
    (output / "RECORDS-waveforms").write_text(index_text, encoding="utf-8")

    patient_records = defaultdict(list)
    for line in index_text.splitlines():
        record = line.strip()
        if not record:
            continue
        parts = record.split("/")
        if len(parts) >= 3:
            patient_records[parts[1]].append(record)

    selected = []
    downloaded_bytes = 0
    scanned_records = 0
    rejected_large = 0
    failures = 0
    for patient in sorted(patient_records):
        if len(selected) >= args.patients:
            break
        patient_windows = 0
        patient_files = []
        for record in patient_records[patient][:args.max_records_per_patient]:
            scanned_records += 1
            parent = record.rsplit("/", 1)[0]
            try:
                master_text = fetch(base + record + ".hea")
                segments = parse_master(master_text)
                for segment in segments:
                    if segment.endswith("_layout"):
                        continue
                    segment_base = parent + "/" + segment
                    header_text = fetch(base + segment_base + ".hea")
                    metadata = parse_physical(header_text)
                    if metadata is None:
                        continue
                    possible = int(
                        metadata["signal_length"]
                        // round(args.window_seconds * metadata["fs"])
                    )
                    if possible < 1:
                        continue
                    sizes = []
                    for dat_name in metadata["dat_files"]:
                        sizes.append(remote_size(base + parent + "/" + dat_name))
                    if sum(sizes) > args.max_dat_mb * 1024 * 1024:
                        rejected_large += 1
                        continue
                    local_parent = output / parent
                    master_path = output / (record + ".hea")
                    master_path.parent.mkdir(parents=True, exist_ok=True)
                    master_path.write_text(master_text, encoding="utf-8")
                    header_path = output / (segment_base + ".hea")
                    header_path.parent.mkdir(parents=True, exist_ok=True)
                    header_path.write_text(header_text, encoding="utf-8")
                    for dat_name in metadata["dat_files"]:
                        dat_path = local_parent / dat_name
                        downloaded_bytes += download(
                            base + parent + "/" + dat_name, dat_path
                        )
                        patient_files.append(str(dat_path.relative_to(output)))
                    patient_windows += min(
                        possible, args.windows_per_patient - patient_windows
                    )
                    if patient_windows >= args.windows_per_patient:
                        break
                if patient_windows >= args.windows_per_patient:
                    break
            except Exception as exc:
                failures += 1
                print(f"[Skip] {record}: {exc}", flush=True)
        if patient_windows > 0:
            selected.append({
                "patient": patient,
                "estimated_windows": patient_windows,
                "files": sorted(set(patient_files)),
            })
            print(
                f"[Selected] {len(selected)}/{args.patients} {patient} "
                f"windows={patient_windows}",
                flush=True,
            )

    summary = {
        "source": "MIMIC-III Waveform Database Matched Subset",
        "base_url": base,
        "requested_patients": args.patients,
        "selected_patients": len(selected),
        "scanned_records": scanned_records,
        "rejected_large_segments": rejected_large,
        "download_failures": failures,
        "downloaded_bytes_including_existing": downloaded_bytes,
        "patients": selected,
    }
    (output / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "patients"}, indent=2))
    if not selected:
        raise SystemExit("No paired ECG/PLETH patients were downloaded")


if __name__ == "__main__":
    main()
