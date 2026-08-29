#!/usr/bin/env python3
"""Concurrently select and download paired ECG/PLETH MIMIC records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_mimic_paired_pilot import (
    DEFAULT_BASE,
    download,
    fetch,
    parse_master,
    parse_physical,
    remote_size,
)


def scan_patient(item, *, base, window_seconds, windows_per_patient,
                 max_dat_bytes, max_records_per_patient,
                 max_segments_per_record, request_timeout):
    patient, records = item
    candidates = []
    failures = []
    rejected_large = 0
    scanned_records = 0
    estimated_windows = 0
    for record in records[:max_records_per_patient]:
        scanned_records += 1
        parent = record.rsplit("/", 1)[0]
        try:
            master_text = fetch(
                base + record + ".hea", attempts=1, timeout=request_timeout
            )
            for segment in parse_master(master_text)[:max_segments_per_record]:
                if segment.endswith("_layout"):
                    continue
                segment_base = parent + "/" + segment
                header_text = fetch(
                    base + segment_base + ".hea",
                    attempts=1,
                    timeout=request_timeout,
                )
                metadata = parse_physical(header_text)
                if metadata is None:
                    continue
                possible = int(
                    metadata["signal_length"]
                    // round(window_seconds * metadata["fs"])
                )
                if possible < 1:
                    continue
                dat_entries = []
                total_bytes = 0
                for dat_name in metadata["dat_files"]:
                    size = remote_size(
                        base + parent + "/" + dat_name,
                        timeout=request_timeout,
                    )
                    total_bytes += size
                    dat_entries.append((dat_name, size))
                if total_bytes > max_dat_bytes:
                    rejected_large += 1
                    continue
                candidates.append({
                    "record": record,
                    "parent": parent,
                    "master_text": master_text,
                    "segment_base": segment_base,
                    "header_text": header_text,
                    "dat_entries": dat_entries,
                    "possible_windows": possible,
                })
                estimated_windows += min(
                    possible, windows_per_patient - estimated_windows
                )
                if estimated_windows >= windows_per_patient:
                    break
            if estimated_windows >= windows_per_patient:
                break
        except Exception as exc:
            failures.append(f"{record}: {exc}")
    return {
        "patient": patient,
        "estimated_windows": estimated_windows,
        "candidates": candidates,
        "scanned_records": scanned_records,
        "rejected_large": rejected_large,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--windows_per_patient", type=int, default=8)
    parser.add_argument("--window_seconds", type=float, default=30.0)
    parser.add_argument("--max_dat_mb", type=float, default=64.0)
    parser.add_argument("--max_records_per_patient", type=int, default=3)
    parser.add_argument("--max_segments_per_record", type=int, default=32)
    parser.add_argument("--scan_workers", type=int, default=16)
    parser.add_argument("--download_workers", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=15.0)
    parser.add_argument("--base_url", default=DEFAULT_BASE)
    parser.add_argument(
        "--start_after_patient",
        default="",
        help=(
            "Only scan patient IDs lexicographically after this ID. "
            "Used by the streaming downloader to resume at a batch boundary."
        ),
    )
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/") + "/"
    index_text = fetch(base + "RECORDS-waveforms")
    (output / "RECORDS-waveforms").write_text(index_text, encoding="utf-8")

    patient_records = defaultdict(list)
    for line in index_text.splitlines():
        record = line.strip()
        parts = record.split("/")
        if record and len(parts) >= 3:
            patient_records[parts[1]].append(record)

    items = sorted(patient_records.items())
    if args.start_after_patient:
        items = [
            item for item in items
            if item[0] > args.start_after_patient
        ]
    selected = []
    scanned_records = 0
    rejected_large = 0
    failures = []
    batch_size = max(args.scan_workers * 4, 32)
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        with ThreadPoolExecutor(max_workers=args.scan_workers) as pool:
            results = list(pool.map(
                lambda item: scan_patient(
                    item,
                    base=base,
                    window_seconds=args.window_seconds,
                    windows_per_patient=args.windows_per_patient,
                    max_dat_bytes=args.max_dat_mb * 1024 * 1024,
                    max_records_per_patient=args.max_records_per_patient,
                    max_segments_per_record=args.max_segments_per_record,
                    request_timeout=args.request_timeout,
                ),
                batch,
            ))
        for result in results:
            scanned_records += result["scanned_records"]
            rejected_large += result["rejected_large"]
            failures.extend(result["failures"])
            if result["estimated_windows"] > 0:
                selected.append(result)
                print(
                    f"[Selected] {len(selected)}/{args.patients} "
                    f"{result['patient']} windows={result['estimated_windows']}",
                    flush=True,
                )
                if len(selected) >= args.patients:
                    break
        if len(selected) >= args.patients:
            break

    if len(selected) < args.patients:
        raise SystemExit(
            f"Only {len(selected)} paired patients found; requested {args.patients}"
        )

    tasks = {}
    for result in selected:
        for candidate in result["candidates"]:
            master_path = output / (candidate["record"] + ".hea")
            master_path.parent.mkdir(parents=True, exist_ok=True)
            master_path.write_text(candidate["master_text"], encoding="utf-8")
            header_path = output / (candidate["segment_base"] + ".hea")
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(candidate["header_text"], encoding="utf-8")
            for dat_name, _ in candidate["dat_entries"]:
                destination = output / candidate["parent"] / dat_name
                tasks[str(destination)] = (
                    base + candidate["parent"] + "/" + dat_name,
                    destination,
                )

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        sizes = list(pool.map(lambda task: download(*task), tasks.values()))

    summary_patients = []
    for result in selected:
        files = []
        for candidate in result["candidates"]:
            files.extend(
                str((output / candidate["parent"] / name).relative_to(output))
                for name, _ in candidate["dat_entries"]
            )
        summary_patients.append({
            "patient": result["patient"],
            "estimated_windows": result["estimated_windows"],
            "files": sorted(set(files)),
        })

    summary = {
        "source": "MIMIC-III Waveform Database Matched Subset",
        "base_url": base,
        "requested_patients": args.patients,
        "selected_patients": len(selected),
        "scanned_records": scanned_records,
        "rejected_large_segments": rejected_large,
        "download_failures": len(failures),
        "downloaded_bytes_including_existing": sum(sizes),
        "scan_workers": args.scan_workers,
        "download_workers": args.download_workers,
        "start_after_patient": args.start_after_patient,
        "last_selected_patient": selected[-1]["patient"],
        "patients": summary_patients,
        "failure_examples": failures[:20],
    }
    (output / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"patients", "failure_examples"}}, indent=2))


if __name__ == "__main__":
    main()
