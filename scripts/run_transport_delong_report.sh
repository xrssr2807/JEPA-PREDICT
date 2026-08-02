#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ON_PREDICTIONS="${ON_PREDICTIONS:?Set ON_PREDICTIONS to Transport-on patient predictions}"
OFF_PREDICTIONS="${OFF_PREDICTIONS:?Set OFF_PREDICTIONS to Transport-off patient predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/ICASSP2027/04_statistics/transport_on_off}"

"$PYTHON_BIN" scripts/evaluate_clinical_predictions.py \
    --predictions "transport_on=$ON_PREDICTIONS" \
    --predictions "transport_off=$OFF_PREDICTIONS" \
    --reference transport_on \
    --focus_label 冠心病 \
    --output_dir "$OUTPUT_DIR"
