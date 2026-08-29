#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/mimic_physio_v2_1000_pt}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/mimic_stream_work}"
PATIENTS="${PATIENTS:-1000}"
BATCH_PATIENTS="${BATCH_PATIENTS:-100}"

mkdir -p "$OUTPUT_DIR" "$WORK_DIR" logs

python -u scripts/stream_mimic_paired_dataset.py \
  --output_dir "$OUTPUT_DIR" \
  --work_dir "$WORK_DIR" \
  --patients "$PATIENTS" \
  --batch_patients "$BATCH_PATIENTS" \
  --windows_per_patient 8 \
  --window_seconds 30 \
  --target_hz 100 \
  --max_dat_mb 64 \
  --max_records_per_patient 3 \
  --scan_workers 16 \
  --download_workers 6

echo "[Complete] MIMIC streaming dataset: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
