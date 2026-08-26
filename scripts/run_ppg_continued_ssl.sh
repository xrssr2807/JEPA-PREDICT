#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/root/autodl-tmp/JEPA-PREDICT/outputs_phase2_physio_v2_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-/root/autodl-tmp/JEPA-PREDICT/splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
ADAPT_DIR="${ADAPT_DIR:-outputs_p3_ppg_continued_ssl_adaptation_seed42}"
BASELINE_DIR="${BASELINE_DIR:-outputs_p3_ppg_continued_ssl_baseline_seed42}"
ADAPTED_DIR="${ADAPTED_DIR:-outputs_p3_ppg_continued_ssl_adapted_seed42}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_ppg_continued_ssl/results/seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { echo "[Error] $*" >&2; exit 1; }

is_downstream_complete() {
    local output_dir="$1"
    [[ -s "${output_dir}/downstream_multidisease_best.pt" ]] \
        && [[ -s "${output_dir}/validation_patient_predictions.csv" ]] \
        && grep -Eq "DEVELOPMENT COMPLETE.*TEST SET SEALED" \
            "${output_dir}/downstream_console.log" 2>/dev/null
}

run_downstream() {
    local name="$1"
    local checkpoint="$2"
    local output_dir="$3"
    if is_downstream_complete "$output_dir"; then
        echo "[Skip] downstream=$name"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] downstream=$name checkpoint=$checkpoint"
    python -u train_downstream.py \
        --checkpoint "$checkpoint" \
        --dataset multidisease \
        --multidisease_channel ppg \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --ppg_morphology_head off \
        --patient_mil on \
        --multiscale on \
        --experiment_id "P3_ppg_continued_ssl_${name}_seed${SEED}" \
        --output_dir "$output_dir" \
        --mil_batch_size 64 \
        --mil_chunk_size 128 \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"
    is_downstream_complete "$output_dir" || die "Incomplete downstream: $name"
}

[[ -s "$SOURCE_CHECKPOINT" ]] || die "Missing checkpoint: $SOURCE_CHECKPOINT"
[[ -s "$SPLIT" ]] || die "Missing split: $SPLIT"
actual_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_sha"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[Error] CUDA unavailable; refusing CPU experiment")
print(f"[CUDA] {torch.cuda.get_device_name(0)}")
PY

adapted_checkpoint="${ADAPT_DIR}/ppg_continued_ssl_last.pt"
if [[ ! -s "$adapted_checkpoint" ]]; then
    echo "[Run] train-only PPG continued SSL"
    python -u train_ppg_continued_ssl.py \
        --checkpoint "$SOURCE_CHECKPOINT" \
        --split "$SPLIT" \
        --output_dir "$ADAPT_DIR" \
        --epochs 8 \
        --batch_size 192 \
        --workers "$WORKERS" \
        --seed "$SEED" \
        2>&1 | tee "${ADAPT_DIR}.log"
fi
[[ -s "$adapted_checkpoint" ]] || die "Adaptation checkpoint missing"

run_downstream baseline "$SOURCE_CHECKPOINT" "$BASELINE_DIR"
run_downstream continued_ssl "$adapted_checkpoint" "$ADAPTED_DIR"

python scripts/summarize_ppg_continued_ssl.py \
    --baseline "$BASELINE_DIR" \
    --adapted "$ADAPTED_DIR" \
    --adaptation_checkpoint "$adapted_checkpoint" \
    --split "$SPLIT" \
    --output_dir "$PAPER_DIR"

echo "[Complete] PPG continued SSL experiment | seed=$SEED | test=sealed"
