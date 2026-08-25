#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-outputs_phase2_physio_v2_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-outputs_p3_selective_kd_teacher_both_seed42}"
DUAL_TEACHER_CHECKPOINT="${DUAL_TEACHER_CHECKPOINT:-${TEACHER_OUTPUT}/downstream_multidisease_best.pt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p3_selective_kd_ppg}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_ppg_selective_distillation/results/seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-64}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-128}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-32}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:-64}"
VARIANTS="${VARIANTS:-baseline logit selective selective_relation}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

is_complete() {
    local output_dir="$1"
    [[ -s "${output_dir}/downstream_multidisease_best.pt" ]] \
        && [[ -s "${output_dir}/validation_patient_predictions.csv" ]] \
        && grep -Eq "DEVELOPMENT COMPLETE.*TEST SET SEALED" \
            "${output_dir}/downstream_console.log" 2>/dev/null
}

run_teacher() {
    local default_teacher_checkpoint="${TEACHER_OUTPUT}/downstream_multidisease_best.pt"
    if [[ "$DUAL_TEACHER_CHECKPOINT" != "$default_teacher_checkpoint" ]]; then
        [[ -s "$DUAL_TEACHER_CHECKPOINT" ]] || die \
            "Provided teacher checkpoint is missing: $DUAL_TEACHER_CHECKPOINT"
        echo "[Reuse] externally provided dual-channel teacher: $DUAL_TEACHER_CHECKPOINT"
        return
    fi
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$TEACHER_OUTPUT"; then
        echo "[Skip] sealed dual-channel teacher"
        return
    fi
    mkdir -p "$TEACHER_OUTPUT"
    echo "[Run] sealed dual-channel teacher -> $TEACHER_OUTPUT"
    python -u train_downstream.py \
        --checkpoint "$PRETRAIN_CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel both \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --patient_mil on \
        --multiscale on \
        --experiment_id "P3_selective_kd_teacher_both_seed${SEED}" \
        --output_dir "$TEACHER_OUTPUT" \
        --mil_batch_size "$TEACHER_BATCH_SIZE" \
        --mil_chunk_size "$TEACHER_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --seal_test \
        2>&1 | tee "${TEACHER_OUTPUT}/downstream_console.log"
    is_complete "$TEACHER_OUTPUT" || die "Teacher training is incomplete"
}

run_student() {
    local variant="$1"
    local output_dir="${OUTPUT_PREFIX}_${variant}_seed${SEED}"
    local -a distill_args=()
    case "$variant" in
        baseline)
            ;;
        logit)
            distill_args=(
                --dual_teacher_checkpoint "$DUAL_TEACHER_CHECKPOINT"
                --distill_logit_weight 0.30
                --distill_embedding_weight 0.00
                --distill_relation_weight 0.00
                --distill_temperature 2.0
                --distill_gate none
            )
            ;;
        selective)
            distill_args=(
                --dual_teacher_checkpoint "$DUAL_TEACHER_CHECKPOINT"
                --distill_logit_weight 0.30
                --distill_embedding_weight 0.10
                --distill_relation_weight 0.00
                --distill_temperature 2.0
                --distill_gate target_agreement
                --distill_confidence_threshold 0.60
                --distill_chd_weight 2.0
                --distill_balance_targets
                --distill_ramp_epochs 5
            )
            ;;
        selective_relation)
            distill_args=(
                --dual_teacher_checkpoint "$DUAL_TEACHER_CHECKPOINT"
                --distill_logit_weight 0.30
                --distill_embedding_weight 0.10
                --distill_relation_weight 0.05
                --distill_temperature 2.0
                --distill_gate target_agreement
                --distill_confidence_threshold 0.60
                --distill_chd_weight 2.0
                --distill_balance_targets
                --distill_ramp_epochs 5
            )
            ;;
        *) die "Unknown variant: $variant" ;;
    esac

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$output_dir"; then
        echo "[Skip] variant=$variant"
        return
    fi
    mkdir -p "$output_dir"
    printf '%q ' python -u train_downstream.py "${distill_args[@]}" \
        > "${output_dir}/command.txt"
    printf '\n' >> "${output_dir}/command.txt"
    echo "[Run] PPG student variant=$variant -> $output_dir"
    python -u train_downstream.py \
        --checkpoint "$PRETRAIN_CHECKPOINT" \
        --dataset multidisease \
        --multidisease_channel ppg \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --patient_mil on \
        --multiscale on \
        --experiment_id "P3_selective_kd_${variant}_seed${SEED}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --seal_test \
        "${distill_args[@]}" \
        2>&1 | tee "${output_dir}/downstream_console.log"
    is_complete "$output_dir" || die "Incomplete student run: $variant"
}

[[ -s "$PRETRAIN_CHECKPOINT" ]] || die \
    "Pretrained checkpoint missing: $PRETRAIN_CHECKPOINT"
[[ -s "$SPLIT" ]] || die "Frozen split missing: $SPLIT"
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_split_sha"

for option in dual_teacher_checkpoint distill_gate distill_relation_weight \
    distill_chd_weight distill_balance_targets distill_ramp_epochs seal_test; do
    python train_downstream.py --help 2>&1 | grep -q -- "--${option}" || die \
        "train_downstream.py does not support --${option}"
done

run_teacher
[[ -s "$DUAL_TEACHER_CHECKPOINT" ]] || die \
    "Teacher checkpoint missing: $DUAL_TEACHER_CHECKPOINT"

read -r -a variants <<< "$VARIANTS"
summary_args=()
for variant in "${variants[@]}"; do
    run_student "$variant"
    summary_args+=(
        --run "${variant}=${OUTPUT_PREFIX}_${variant}_seed${SEED}"
    )
done

python scripts/summarize_selective_ppg_distillation.py \
    "${summary_args[@]}" \
    --teacher "$DUAL_TEACHER_CHECKPOINT" \
    --pretrain "$PRETRAIN_CHECKPOINT" \
    --split "$SPLIT" \
    --output_dir "$PAPER_DIR"

echo "[Complete] selective dual-teacher PPG study | test=sealed"
