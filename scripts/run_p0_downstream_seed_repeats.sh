#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
FT_SEEDS="${FT_SEEDS:-42 3407 2026}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p0_objective_ftrepeat}"
LEGACY_OUTPUT_PREFIX="${LEGACY_OUTPUT_PREFIX:-outputs_p0_objective}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_baselines/results/pretraining_objectives/downstream_seed_repeats}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

PHYSIO_CHECKPOINT="${PHYSIO_CHECKPOINT:-outputs_phase2_physio_v2_seed42/jepa_best.pt}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-outputs_p0_pretrain_multimodal_mae_seed42/multimodal_mae_best.pt}"
CONTRASTIVE_CHECKPOINT="${CONTRASTIVE_CHECKPOINT:-outputs_p0_pretrain_contrastive_seed42/contrastive_best.pt}"
XMAE_CHECKPOINT="${XMAE_CHECKPOINT:-outputs_p0_pretrain_xmae_seed42/xmae_objective_best.pt}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

checkpoint_for() {
    case "$1" in
        physio_v2) echo "$PHYSIO_CHECKPOINT" ;;
        multimodal_mae) echo "$MAE_CHECKPOINT" ;;
        contrastive) echo "$CONTRASTIVE_CHECKPOINT" ;;
        xmae) echo "$XMAE_CHECKPOINT" ;;
        *) die "Unknown objective: $1" ;;
    esac
}

output_dir_for() {
    local experiment="$1"
    local seed="$2"
    if [[ "$seed" == "$PRETRAIN_SEED" ]]; then
        echo "${LEGACY_OUTPUT_PREFIX}_${experiment}_seed${seed}"
    else
        echo "${OUTPUT_PREFIX}_${experiment}_preseed${PRETRAIN_SEED}_ftseed${seed}"
    fi
}

is_complete() {
    local output_dir="$1"
    [[ -s "${output_dir}/downstream_multidisease_best.pt" ]] \
        && [[ -s "${output_dir}/validation_patient_predictions.csv" ]] \
        && grep -Eq "DEVELOPMENT COMPLETE.*TEST SET SEALED" \
            "${output_dir}/downstream_console.log" 2>/dev/null
}

run_downstream() {
    local experiment="$1"
    local seed="$2"
    local checkpoint output_dir
    checkpoint="$(checkpoint_for "$experiment")"
    output_dir="$(output_dir_for "$experiment" "$seed")"

    [[ -s "$checkpoint" ]] || die "Checkpoint missing: $checkpoint"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$output_dir"; then
        echo "[Skip] objective=$experiment pretrain_seed=$PRETRAIN_SEED ft_seed=$seed"
        return
    fi

    mkdir -p "$output_dir"
    echo "============================================================"
    echo "[Run] objective=$experiment pretrain_seed=$PRETRAIN_SEED ft_seed=$seed"
    echo "[Checkpoint] $checkpoint"
    echo "[Output] $output_dir"
    echo "============================================================"
    python -u train_downstream.py \
        --checkpoint "$checkpoint" \
        --encoder_init pretrained \
        --encoder_arch jepa_transformer \
        --dataset multidisease \
        --multidisease_channel both \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --patient_mil on \
        --multiscale on \
        --experiment_id "P0_${experiment}_preseed${PRETRAIN_SEED}_ftseed${seed}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$seed" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"

    is_complete "$output_dir" || die "Incomplete downstream run: $output_dir"
}

[[ -s "$SPLIT" ]] || die "Frozen split missing: $SPLIT"
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_split_sha"

for option in encoder_init encoder_arch seal_test experiment_id; do
    python train_downstream.py --help 2>&1 | grep -q -- "--${option}" || die \
        "train_downstream.py does not support --${option}"
done

experiments=(physio_v2 multimodal_mae contrastive xmae)
read -r -a seeds <<< "$FT_SEEDS"
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid downstream seed: $seed"
    for experiment in "${experiments[@]}"; do
        run_downstream "$experiment" "$seed"
    done
done

python scripts/summarize_p0_downstream_seed_repeats.py \
    --output_prefix "$OUTPUT_PREFIX" \
    --legacy_output_prefix "$LEGACY_OUTPUT_PREFIX" \
    --paper_dir "$PAPER_DIR" \
    --pretrain_seed "$PRETRAIN_SEED" \
    --ft_seeds "${seeds[@]}" \
    --checkpoint "physio_v2=$PHYSIO_CHECKPOINT" \
    --checkpoint "multimodal_mae=$MAE_CHECKPOINT" \
    --checkpoint "contrastive=$CONTRASTIVE_CHECKPOINT" \
    --checkpoint "xmae=$XMAE_CHECKPOINT"

echo "[Complete] P0 downstream seed repeats | pretrain_seed=$PRETRAIN_SEED ft_seeds=$FT_SEEDS test=sealed"
