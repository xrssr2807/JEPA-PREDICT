#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_spv2}"
SEEDS="${SEEDS:-42 3407 2026}"
HEAD_MODE="${HEAD_MODE:-on}"
CHANNEL="${CHANNEL:-both}"
WORKERS="${WORKERS:-8}"
CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST:-NO}"

if [[ "$CONFIRM_FINAL_TEST" != "YES" ]]; then
    echo "[Blocked] The test set remains sealed." >&2
    echo "Set CONFIRM_FINAL_TEST=YES only after mode/channel are frozen." >&2
    exit 2
fi
if [[ "$HEAD_MODE" != "on" && "$HEAD_MODE" != "off" ]]; then
    echo "[Error] HEAD_MODE must be on or off." >&2
    exit 1
fi
if [[ "$CHANNEL" != "ppg" && "$CHANNEL" != "ecg" && "$CHANNEL" != "both" ]]; then
    echo "[Error] CHANNEL must be ppg, ecg, or both." >&2
    exit 1
fi
if [[ ! -s "$CHECKPOINT" || ! -s "$SPLIT" ]]; then
    echo "[Error] Pretrained checkpoint or split manifest is missing." >&2
    exit 1
fi

if [[ "$CHANNEL" == "both" ]]; then
    batch_size=32
    chunk_size=64
else
    batch_size=64
    chunk_size=128
fi

read -r -a seed_values <<< "$SEEDS"
for seed in "${seed_values[@]}"; do
    output_dir="${OUTPUT_PREFIX}_${HEAD_MODE}_${CHANNEL}_seed${seed}"
    downstream_checkpoint="${output_dir}/downstream_multidisease_best.pt"
    if [[ ! -s "$downstream_checkpoint" ]]; then
        echo "[Error] Sealed checkpoint not found: $downstream_checkpoint" >&2
        exit 1
    fi
    test_status="$(
        python -c \
            "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('test_status', 'legacy'))" \
            "$downstream_checkpoint"
    )"
    if [[ "$test_status" == "evaluated" ]]; then
        echo "[Skip] Test already evaluated: $downstream_checkpoint"
        continue
    fi
    if [[ "$test_status" != "sealed" ]]; then
        echo "[Error] Expected sealed checkpoint, got status=$test_status" >&2
        exit 1
    fi

    echo "[FinalTest] seed=$seed mode=$HEAD_MODE channel=$CHANNEL"
    python -u train_downstream.py \
        --checkpoint "$CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel "$CHANNEL" \
        --multidisease_split "$SPLIT" \
        --shared_private_head "$HEAD_MODE" \
        --output_dir "$output_dir" \
        --mil_batch_size "$batch_size" \
        --mil_chunk_size "$chunk_size" \
        --workers "$WORKERS" \
        --seed "$seed" \
        --evaluate_checkpoint "$downstream_checkpoint" \
        2>&1 | tee "$output_dir/final_test_console.log"
done

python scripts/summarize_shared_private_ablation.py \
    --output_prefix "$OUTPUT_PREFIX" \
    --seeds $SEEDS \
    --head_modes "$HEAD_MODE" \
    --channels "$CHANNEL" \
    --summary_dir "results/final_${HEAD_MODE}_${CHANNEL}"

echo "[Complete] Final test evaluation finished for the frozen configuration."
