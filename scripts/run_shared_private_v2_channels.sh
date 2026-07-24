#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_spv2}"
SEEDS="${SEEDS:-42 3407 2026}"
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
    local seed="$2"
    local batch_size
    local chunk_size

    if [[ "$channel" == "both" ]]; then
        batch_size=32
        chunk_size=64
    else
        batch_size=64
        chunk_size=128
    fi

    local output_dir="${OUTPUT_PREFIX}_${channel}_seed${seed}"
    mkdir -p "$output_dir"

    echo
    echo "============================================================"
    echo "[Run] seed=$seed"
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
        --seed "$seed" \
        2>&1 | tee "$output_dir/downstream_console.log"

    echo "[Done] seed=$seed channel=$channel"
}

read -r -a seed_values <<< "$SEEDS"
if [[ "${#seed_values[@]}" -eq 0 ]]; then
    echo "[Error] SEEDS must contain at least one integer." >&2
    exit 1
fi

for seed in "${seed_values[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "[Error] Invalid seed: $seed" >&2
        exit 1
    fi

    echo "[Sequence] seed=$seed | PPG -> ECG -> ECG+PPG"
    run_channel ppg "$seed"
    run_channel ecg "$seed"
    run_channel both "$seed"
done

echo "[Complete] All downstream experiments finished for seeds: $SEEDS"
