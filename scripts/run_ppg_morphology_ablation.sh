#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-/root/autodl-tmp/JEPA-PREDICT/outputs_phase2_physio_v2_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-/root/autodl-tmp/JEPA-PREDICT/splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p3_ppg_morphology}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_ppg_morphology/results/seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-64}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { echo "[Error] $*" >&2; exit 1; }

is_complete() {
    local output_dir="$1"
    [[ -s "${output_dir}/downstream_multidisease_best.pt" ]] \
        && [[ -s "${output_dir}/validation_patient_predictions.csv" ]] \
        && grep -Eq "DEVELOPMENT COMPLETE.*TEST SET SEALED" \
            "${output_dir}/downstream_console.log" 2>/dev/null
}

run_variant() {
    local variant="$1"
    local mode="$2"
    local output_dir="${OUTPUT_PREFIX}_${variant}_seed${SEED}"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$output_dir"; then
        echo "[Skip] variant=$variant"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] PPG morphology variant=$variant -> $output_dir"
    python -u train_downstream.py \
        --checkpoint "$PRETRAIN_CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel ppg \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --ppg_morphology_head "$mode" \
        --patient_mil on \
        --multiscale on \
        --experiment_id "P3_ppg_morphology_${variant}_seed${SEED}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"
    is_complete "$output_dir" || die "Incomplete run: $variant"
}

[[ -s "$PRETRAIN_CHECKPOINT" ]] || die "Missing checkpoint: $PRETRAIN_CHECKPOINT"
[[ -s "$SPLIT" ]] || die "Missing split: $SPLIT"
actual_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_sha"
python train_downstream.py --help 2>&1 | grep -q -- "--ppg_morphology_head" \
    || die "train_downstream.py lacks --ppg_morphology_head"
python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "[Error] CUDA GPU is unavailable; refusing to run this experiment on CPU"
    )
print(
    f"[CUDA] device={torch.cuda.get_device_name(0)} "
    f"vram_gb={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}"
)
PY

run_variant baseline off
run_variant morphology on

python scripts/summarize_ppg_morphology_ablation.py \
    --run "baseline=${OUTPUT_PREFIX}_baseline_seed${SEED}" \
    --run "morphology=${OUTPUT_PREFIX}_morphology_seed${SEED}" \
    --pretrain "$PRETRAIN_CHECKPOINT" \
    --split "$SPLIT" \
    --output_dir "$PAPER_DIR"

echo "[Complete] PPG morphology ablation | seed=$SEED | test=sealed"
