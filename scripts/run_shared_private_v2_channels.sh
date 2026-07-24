#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_spv2}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
SHARED_PRIVATE_HEAD="${SHARED_PRIVATE_HEAD:-on}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -s "$CHECKPOINT" ]]; then
    echo "[Error] Checkpoint not found or empty: $CHECKPOINT" >&2
    exit 1
fi

if [[ ! -s "$SPLIT" ]]; then
    echo "[Error] Split manifest not found or empty: $SPLIT" >&2
    exit 1
fi

if ! python train_downstream.py --help 2>&1 | grep -q -- "--shared_private_head"; then
    echo "[Error] train_downstream.py is older than commit f4e244e." >&2
    echo "Update branch soft-dtw-token-align before running this script." >&2
    exit 1
fi

run_channel() {
    local channel="$1"
    local batch_size
    local chunk_size

    if [[ "$channel" == "both" ]]; then
        batch_size=32
        chunk_size=64
    else
        batch_size=64
        chunk_size=128
    fi

    local output_dir="${OUTPUT_PREFIX}_${channel}_seed${SEED}"
    mkdir -p "$output_dir"

    echo
    echo "============================================================"
    echo "[Run] channel=$channel"
    echo "[Run] checkpoint=$CHECKPOINT"
    echo "[Run] split=$SPLIT"
    echo "[Run] shared_private_head=$SHARED_PRIVATE_HEAD"
    echo "[Run] batch_size=$batch_size chunk_size=$chunk_size"
    echo "[Run] output=$output_dir"
    echo "============================================================"

    python -u train_downstream.py \
        --checkpoint "$CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel "$channel" \
        --multidisease_split "$SPLIT" \
        --shared_private_head "$SHARED_PRIVATE_HEAD" \
        --output_dir "$output_dir" \
        --mil_batch_size "$batch_size" \
        --mil_chunk_size "$chunk_size" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        2>&1 | tee "$output_dir/downstream_console.log"

    echo "[Done] channel=$channel"
}

echo "[Sequence] PPG -> ECG -> ECG+PPG"
run_channel ppg
run_channel ecg
run_channel both
echo "[Complete] All downstream channel experiments finished."
