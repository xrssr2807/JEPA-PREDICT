#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_cross_device_seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
RATES="${RATES:-100 50 25}"
CHANNEL="${CHANNEL:-both}"

test -s "$PRETRAIN_CHECKPOINT" || {
    echo "[Error] pretraining checkpoint missing: $PRETRAIN_CHECKPOINT" >&2
    exit 1
}
test -s "$SPLIT" || {
    echo "[Error] split missing: $SPLIT" >&2
    exit 1
}
mkdir -p "$OUTPUT_ROOT"

train_one() {
    local mode="$1"
    local output_dir="$OUTPUT_ROOT/train_${mode}"
    mkdir -p "$output_dir"
    local extra=()
    if [[ "$mode" == "multirate" ]]; then
        extra+=(--multirate_train_hz 25,50,100 --multirate_probability 0.75)
    fi
    "$PYTHON_BIN" -u train_downstream.py \
        --checkpoint "$PRETRAIN_CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel "$CHANNEL" \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --output_dir "$output_dir" \
        --mil_batch_size 32 \
        --mil_chunk_size 64 \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --seal_test \
        --experiment_id "cross_device_${mode}_seed${SEED}" \
        "${extra[@]}" \
        2>&1 | tee "$output_dir/console.log"
}

evaluate_rates() {
    local mode="$1"
    local downstream_checkpoint="$OUTPUT_ROOT/train_${mode}/downstream_multidisease_best.pt"
    test -s "$downstream_checkpoint" || {
        echo "[Error] downstream checkpoint missing: $downstream_checkpoint" >&2
        exit 1
    }
    for rate in $RATES; do
        local output_dir="$OUTPUT_ROOT/eval_${mode}_${rate}hz"
        mkdir -p "$output_dir"
        "$PYTHON_BIN" -u train_downstream.py \
            --checkpoint "$PRETRAIN_CHECKPOINT" \
            --evaluate_checkpoint "$downstream_checkpoint" \
            --dataset multidisease \
            --multidisease_channel "$CHANNEL" \
            --multidisease_split "$SPLIT" \
            --shared_private_head off \
            --device_rate_hz "$rate" \
            --output_dir "$output_dir" \
            --mil_batch_size 32 \
            --mil_chunk_size 64 \
            --workers "$WORKERS" \
            --seed "$SEED" \
            --seal_test \
            2>&1 | tee "$output_dir/console.log"
    done
}

for mode in native multirate; do
    train_one "$mode"
    evaluate_rates "$mode"
done

report_args=(--reference native_100hz --focus_label 冠心病 --output_dir "$OUTPUT_ROOT/report")
for mode in native multirate; do
    for rate in $RATES; do
        report_args+=(
            --predictions
            "${mode}_${rate}hz=$OUTPUT_ROOT/eval_${mode}_${rate}hz/validation_patient_predictions.csv"
        )
    done
done
"$PYTHON_BIN" scripts/evaluate_clinical_predictions.py "${report_args[@]}"
echo "[Complete] cross-device validation-only experiment -> $OUTPUT_ROOT"
