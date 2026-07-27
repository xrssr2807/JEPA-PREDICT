#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_spv2}"
SEEDS="${SEEDS:-42 3407 2026}"
HEAD_MODES="${HEAD_MODES:-on off}"
WORKERS="${WORKERS:-8}"
SEAL_TEST="${SEAL_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

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
if ! python train_downstream.py --help 2>&1 | grep -q -- "--seal_test"; then
    echo "[Error] train_downstream.py does not support sealed development." >&2
    exit 1
fi

run_channel() {
    local channel="$1"
    local seed="$2"
    local head_mode="$3"
    local batch_size
    local chunk_size

    if [[ "$channel" == "both" ]]; then
        batch_size=32
        chunk_size=64
    else
        batch_size=64
        chunk_size=128
    fi

    local output_dir="${OUTPUT_PREFIX}_${head_mode}_${channel}_seed${seed}"
    local saved_model="${output_dir}/downstream_multidisease_best.pt"
    if [[ "$SKIP_COMPLETED" == "1" && -s "$saved_model" ]]; then
        echo "[Skip] completed seed=$seed mode=$head_mode channel=$channel"
        return
    fi
    mkdir -p "$output_dir"

    echo
    echo "============================================================"
    echo "[Run] seed=$seed"
    echo "[Run] channel=$channel"
    echo "[Run] checkpoint=$CHECKPOINT"
    echo "[Run] split=$SPLIT"
    echo "[Run] shared_private_head=$head_mode"
    echo "[Run] seal_test=$SEAL_TEST"
    echo "[Run] batch_size=$batch_size chunk_size=$chunk_size"
    echo "[Run] output=$output_dir"
    echo "============================================================"

    local seal_args=()
    if [[ "$SEAL_TEST" == "1" ]]; then
        seal_args+=(--seal_test)
    fi

    python -u train_downstream.py \
        --checkpoint "$CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel "$channel" \
        --multidisease_split "$SPLIT" \
        --shared_private_head "$head_mode" \
        --output_dir "$output_dir" \
        --mil_batch_size "$batch_size" \
        --mil_chunk_size "$chunk_size" \
        --workers "$WORKERS" \
        --seed "$seed" \
        "${seal_args[@]}" \
        2>&1 | tee "$output_dir/downstream_console.log"

    echo "[Done] seed=$seed mode=$head_mode channel=$channel"
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

    read -r -a head_mode_values <<< "$HEAD_MODES"
    for head_mode in "${head_mode_values[@]}"; do
        if [[ "$head_mode" != "on" && "$head_mode" != "off" ]]; then
            echo "[Error] HEAD_MODES accepts only on/off, got: $head_mode" >&2
            exit 1
        fi
        echo "[Sequence] seed=$seed mode=$head_mode | PPG -> ECG -> ECG+PPG"
        run_channel ppg "$seed" "$head_mode"
        run_channel ecg "$seed" "$head_mode"
        run_channel both "$seed" "$head_mode"
    done
done

python scripts/summarize_shared_private_ablation.py \
    --output_prefix "$OUTPUT_PREFIX" \
    --seeds $SEEDS \
    --head_modes $HEAD_MODES

echo "[Complete] All downstream experiments finished."
echo "[Complete] seeds=$SEEDS head_modes=$HEAD_MODES seal_test=$SEAL_TEST"
