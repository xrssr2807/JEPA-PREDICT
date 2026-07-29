#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_transport_interpretability_seed42}"
MAX_SEGMENTS="${MAX_SEGMENTS:-0}"
MAX_SEGMENTS_PER_PATIENT="${MAX_SEGMENTS_PER_PATIENT:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
CPU_THREADS="${CPU_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] CPU_THREADS must be a positive integer: $CPU_THREADS" >&2
    exit 1
fi

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"

test -s "$CHECKPOINT" || {
    echo "[Error] checkpoint not found: $CHECKPOINT" >&2
    exit 1
}
test -s "$SPLIT" || {
    echo "[Error] split not found: $SPLIT" >&2
    exit 1
}
test -d "$DATA_DIR" || {
    echo "[Error] data directory not found: $DATA_DIR" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

echo "[Runtime] CPU_THREADS=$CPU_THREADS BATCH_SIZE=$BATCH_SIZE WORKERS=$WORKERS"
echo "[Sampling] MAX_SEGMENTS=$MAX_SEGMENTS MAX_SEGMENTS_PER_PATIENT=$MAX_SEGMENTS_PER_PATIENT"

"$PYTHON_BIN" -u analyze_transport_interpretability.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --split "$SPLIT" \
    --role val \
    --output_dir "$OUTPUT_DIR" \
    --max_segments "$MAX_SEGMENTS" \
    --max_segments_per_patient "$MAX_SEGMENTS_PER_PATIENT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/console.log"

echo "[Complete] $OUTPUT_DIR"
