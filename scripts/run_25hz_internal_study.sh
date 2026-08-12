#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-outputs_phase2_physio_v2_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_25hz_prospective_study_seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"

for path in "$CHECKPOINT" "$SPLIT"; do
    if [[ ! -s "$path" ]]; then
        echo "[Error] required file missing or empty: $path" >&2
        exit 1
    fi
done

run_condition() {
    local name="$1"
    local canonical_hz="$2"
    local device_hz="$3"
    local token_seconds="$4"
    local output_dir="$OUTPUT_ROOT/$name/train"
    mkdir -p "$output_dir"

    if [[ -s "$output_dir/downstream_multidisease_best.pt" ]]; then
        echo "[Skip] completed condition: $name"
        return
    fi

    local sampling_args=(--canonical_rate_hz "$canonical_hz")
    if [[ "$device_hz" != "0" ]]; then
        sampling_args+=(--device_rate_hz "$device_hz")
    fi
    if [[ "$token_seconds" != "0" ]]; then
        sampling_args+=(--segment_token_seconds "$token_seconds")
    fi

    echo "============================================================"
    echo "[Run] $name | canonical=${canonical_hz}Hz device=${device_hz}Hz token=${token_seconds}s"
    echo "============================================================"
    "$PYTHON_BIN" -u train_downstream.py \
        --checkpoint "$CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel ppg \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --output_dir "$output_dir" \
        --mil_batch_size 32 \
        --mil_chunk_size 64 \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --seal_test \
        --experiment_id "25hz_${name}_seed${SEED}" \
        "${sampling_args[@]}" \
        2>&1 | tee "$output_dir/console.log"
}

run_condition native100_10s 100 0 0
run_condition bridge25_10s 100 25 0
run_condition native25_10s 25 0 0
run_condition native25_30s_token 25 0 30

echo "[Complete] all four internal 25 Hz conditions finished"
