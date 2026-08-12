#!/usr/bin/env bash
set -euo pipefail

# Diagnostic sampling-rate study. The prospective cohort has already been
# inspected and must not be used for threshold tuning or model selection.
PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-outputs_phase2_physio_v2_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXTERNAL_10S_DIR="${EXTERNAL_10S_DIR:-/root/autodl-tmp/multicenter_external_model_ready/model_input}"
EXTERNAL_30S_DIR="${EXTERNAL_30S_DIR:-/root/autodl-tmp/multicenter_external_model_ready_30s/model_input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_25hz_prospective_study_seed42}"
PAPER_ROOT="${PAPER_ROOT:-paper/ICASSP2027/03_experiments/P3_25hz_prospective_study/results/seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"

for path in "$PRETRAIN_CHECKPOINT" "$SPLIT"; do
    test -s "$path" || { echo "[Error] required file missing: $path" >&2; exit 1; }
done
test -d "$EXTERNAL_10S_DIR" || {
    echo "[Error] 10-second external model_input missing: $EXTERNAL_10S_DIR" >&2
    exit 1
}
if [[ ! -d "$EXTERNAL_30S_DIR" ]]; then
    if [[ -n "${EXTERNAL_PROCESSED_ROOT:-}" && -n "${EXTERNAL_DIAGNOSIS_XLSX:-}" ]]; then
        echo "[Prepare] contiguous 30-second external windows"
        "$PYTHON_BIN" scripts/prepare_multicenter_external_dataset.py \
            --processed-root "$EXTERNAL_PROCESSED_ROOT" \
            --diagnosis-xlsx "$EXTERNAL_DIAGNOSIS_XLSX" \
            --output "$(dirname "$EXTERNAL_30S_DIR")" \
            --windows-per-patient 8 \
            --window-seconds 30
    else
        cat >&2 <<EOF
[Error] 30-second external model_input missing: $EXTERNAL_30S_DIR
Set EXTERNAL_PROCESSED_ROOT and EXTERNAL_DIAGNOSIS_XLSX so this script can
build it, or create it first with prepare_multicenter_external_dataset.py.
EOF
        exit 1
    fi
fi
mkdir -p "$OUTPUT_ROOT" "$PAPER_ROOT"

run_condition() {
    local name="$1"
    local canonical_hz="$2"
    local device_hz="$3"
    local token_seconds="$4"
    local external_dir="$5"
    local train_dir="$OUTPUT_ROOT/${name}/train"
    local external_output="$OUTPUT_ROOT/${name}/external"
    local archive_dir="$PAPER_ROOT/${name}"
    mkdir -p "$train_dir" "$external_output" "$archive_dir"

    local sampling_args=(--canonical_rate_hz "$canonical_hz")
    if [[ "$device_hz" != "0" ]]; then
        sampling_args+=(--device_rate_hz "$device_hz")
    fi
    if [[ "$token_seconds" != "0" ]]; then
        sampling_args+=(--segment_token_seconds "$token_seconds")
    fi

    if [[ ! -s "$train_dir/downstream_multidisease_best.pt" ]]; then
        echo "[Train] $name"
        "$PYTHON_BIN" -u train_downstream.py \
            --checkpoint "$PRETRAIN_CHECKPOINT" \
            --dataset multidisease \
            --multidisease_channel ppg \
            --multidisease_split "$SPLIT" \
            --shared_private_head off \
            --output_dir "$train_dir" \
            --mil_batch_size 32 \
            --mil_chunk_size 64 \
            --workers "$WORKERS" \
            --seed "$SEED" \
            --seal_test \
            --experiment_id "25hz_${name}_seed${SEED}" \
            "${sampling_args[@]}" \
            2>&1 | tee "$train_dir/console.log"
    else
        echo "[Skip] completed training: $name"
    fi

    test -d "$external_dir" || {
        echo "[Error] external model_input missing for $name: $external_dir" >&2
        exit 1
    }
    echo "[External diagnostic] $name"
    "$PYTHON_BIN" -u train_downstream.py \
        --checkpoint "$PRETRAIN_CHECKPOINT" \
        --evaluate_checkpoint "$train_dir/downstream_multidisease_best.pt" \
        --external_data_dir "$external_dir" \
        --dataset multidisease \
        --multidisease_channel ppg \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --output_dir "$external_output" \
        --mil_batch_size 32 \
        --mil_chunk_size 64 \
        --workers "$WORKERS" \
        --seed "$SEED" \
        "${sampling_args[@]}" \
        2>&1 | tee "$external_output/console.log"

    cp "$train_dir/validation_patient_predictions.csv" "$archive_dir/"
    cp "$external_output/external_patient_predictions.csv" "$archive_dir/"
    cp "$external_output/external_evaluation_summary.json" "$archive_dir/"
    cp "$train_dir/console.log" "$archive_dir/train_console.log"
    cp "$external_output/console.log" "$archive_dir/external_console.log"
}

# Historical 100 Hz retrospective training control.
run_condition native100_10s 100 0 0 "$EXTERNAL_10S_DIR"

# Existing 100 Hz encoder geometry; only its observable frequency content is
# matched to the 25 Hz acquisition device.
run_condition bridge25_10s 100 25 0 "$EXTERNAL_10S_DIR"

# True 25 Hz tensors (250 samples per 10 seconds) passed to the same pretrained
# encoder weights. This tests whether native sample geometry helps.
run_condition native25_10s 25 0 0 "$EXTERNAL_10S_DIR"

# Consecutive 10-second retrospective windows form a 30-second segment token.
# The external directory must be regenerated with --window-seconds 30 so each
# token is contiguous rather than a concatenation of distant snapshots.
run_condition native25_30s_token 25 0 30 "$EXTERNAL_30S_DIR"

FOCUS_LABEL="${FOCUS_LABEL:-冠心病}"
validation_report_args=(
    --reference bridge25_10s
    --focus_label "$FOCUS_LABEL"
    --output_dir "$PAPER_ROOT/validation_report"
)
external_report_args=(
    --reference bridge25_10s
    --focus_label "$FOCUS_LABEL"
    --output_dir "$PAPER_ROOT/external_diagnostic_report"
)
for condition in native100_10s bridge25_10s native25_10s native25_30s_token; do
    validation_report_args+=(
        --predictions
        "$condition=$PAPER_ROOT/$condition/validation_patient_predictions.csv"
    )
    external_report_args+=(
        --predictions
        "$condition=$PAPER_ROOT/$condition/external_patient_predictions.csv"
    )
done
"$PYTHON_BIN" scripts/evaluate_clinical_predictions.py "${validation_report_args[@]}"
"$PYTHON_BIN" scripts/evaluate_clinical_predictions.py "${external_report_args[@]}"
"$PYTHON_BIN" scripts/summarize_25hz_prospective_study.py \
    --paper_root "$PAPER_ROOT"

git rev-parse HEAD > "$PAPER_ROOT/git_commit.txt"
sha256sum "$PRETRAIN_CHECKPOINT" "$SPLIT" > "$PAPER_ROOT/input_sha256.txt"
cat > "$PAPER_ROOT/protocol_status.txt" <<'EOF'
The external prospective cohort was previously inspected. These results are
diagnostic domain-shift analyses, not a new sealed external validation. No
external label, threshold, or metric may be used to tune these checkpoints.
EOF
echo "[Complete] 25 Hz prospective diagnostic study -> $PAPER_ROOT"
