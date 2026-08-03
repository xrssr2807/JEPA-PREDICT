#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
DOWNSTREAM_CHECKPOINT="${DOWNSTREAM_CHECKPOINT:-outputs_spv2_off_ppg_seed42/downstream_multidisease_best.pt}"
INTERNAL_SPLIT="${INTERNAL_SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXTERNAL_DATA_DIR="${EXTERNAL_DATA_DIR:-/root/autodl-tmp/multicenter_external_model_ready/model_input}"
EXTERNAL_ROOT="$(dirname "$EXTERNAL_DATA_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_external_multicenter_ppg_seed42}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P4_external_validation/results/jepa_ppg_seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"

for path in "$PRETRAIN_CHECKPOINT" "$DOWNSTREAM_CHECKPOINT" "$INTERNAL_SPLIT"; do
    test -s "$path" || { echo "[Error] required file missing: $path" >&2; exit 1; }
done
test -d "$EXTERNAL_DATA_DIR" || {
    echo "[Error] external model_input missing: $EXTERNAL_DATA_DIR" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR" "$PAPER_DIR"

"$PYTHON_BIN" -u train_downstream.py \
    --checkpoint "$PRETRAIN_CHECKPOINT" \
    --evaluate_checkpoint "$DOWNSTREAM_CHECKPOINT" \
    --external_data_dir "$EXTERNAL_DATA_DIR" \
    --dataset multidisease \
    --multidisease_channel ppg \
    --multidisease_split "$INTERNAL_SPLIT" \
    --shared_private_head off \
    --output_dir "$OUTPUT_DIR" \
    --mil_batch_size 64 \
    --mil_chunk_size 128 \
    --workers "$WORKERS" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/console.log"

"$PYTHON_BIN" scripts/evaluate_clinical_predictions.py \
    --predictions "jepa_ppg=$OUTPUT_DIR/external_patient_predictions.csv" \
    --reference jepa_ppg \
    --focus_label 冠心病 \
    --output_dir "$PAPER_DIR"

cp "$OUTPUT_DIR/external_patient_predictions.csv" "$PAPER_DIR/"
cp "$OUTPUT_DIR/external_evaluation_summary.json" "$PAPER_DIR/"
cp "$OUTPUT_DIR/console.log" "$PAPER_DIR/"
git rev-parse HEAD > "$PAPER_DIR/git_commit.txt"
sha256sum \
    "$PRETRAIN_CHECKPOINT" \
    "$DOWNSTREAM_CHECKPOINT" \
    "$INTERNAL_SPLIT" \
    > "$PAPER_DIR/input_sha256.txt"
for public_file in dataset_summary.json external_patient_labels.csv window_manifest.csv; do
    if [[ -s "$EXTERNAL_ROOT/$public_file" ]]; then
        cp "$EXTERNAL_ROOT/$public_file" "$PAPER_DIR/"
        sha256sum "$EXTERNAL_ROOT/$public_file" >> "$PAPER_DIR/input_sha256.txt"
    fi
done

echo "[Complete] frozen multicenter PPG validation -> $PAPER_DIR"
