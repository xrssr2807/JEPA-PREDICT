#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
SEEDS="${SEEDS:-42}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-80}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_ACCUM_STEPS="${PRETRAIN_ACCUM_STEPS:-3}"
PRETRAIN_LR="${PRETRAIN_LR:-2e-4}"
PRETRAIN_PREFIX="${PRETRAIN_PREFIX:-outputs_p0_pretrain}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p0_objective}"
PHYSIO_V2_TEMPLATE="${PHYSIO_V2_TEMPLATE:-outputs_phase2_physio_v2_seed{seed}/jepa_best.pt}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_baselines/results/pretraining_objectives}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

[[ -s "$SPLIT" ]] || die "Frozen downstream split is missing: $SPLIT"
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_split_sha"
[[ -d /root/autodl-tmp/split_processed ]] || die \
    "Preprocessed pre-training data is missing"

for option in encoder_init encoder_arch seal_test experiment_id; do
    python train_downstream.py --help 2>&1 | grep -q -- "--${option}" || die \
        "train_downstream.py does not support --${option}"
done

resolve_physio_checkpoint() {
    local seed="$1"
    echo "${PHYSIO_V2_TEMPLATE//\{seed\}/$seed}"
}

run_reconstruction_pretrain() {
    local experiment="$1"
    local objective="$2"
    local seed="$3"
    local output_dir="${PRETRAIN_PREFIX}_${experiment}_seed${seed}"
    local checkpoint="${output_dir}/${objective}_best.pt"
    local marker="[Complete] ${objective} baseline pre-training"
    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$checkpoint" ]] \
        && grep -Fq "$marker" "${output_dir}/console.log" 2>/dev/null; then
        echo "[Skip] pretrain=$experiment seed=$seed"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] pretrain=$experiment seed=$seed test=unused"
    python -u train_masked_reconstruction_pretrain.py \
        --objective "$objective" \
        --output_dir "$output_dir" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch_size "$PRETRAIN_BATCH_SIZE" \
        --accum_steps "$PRETRAIN_ACCUM_STEPS" \
        --workers "$WORKERS" \
        --learning_rate "$PRETRAIN_LR" \
        --warmup_epochs 10 \
        --patience 15 \
        --seed "$seed" \
        --data_split_seed "$DATA_SPLIT_SEED" \
        2>&1 | tee "${output_dir}/console.log"
    [[ -s "$checkpoint" ]] || die "Missing checkpoint: $checkpoint"
    grep -Fq "$marker" "${output_dir}/console.log" || die \
        "Completion marker missing: $output_dir"
}

run_contrastive_pretrain() {
    local seed="$1"
    local output_dir="${PRETRAIN_PREFIX}_contrastive_seed${seed}"
    local checkpoint="${output_dir}/contrastive_best.pt"
    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$checkpoint" ]] \
        && grep -Fq "[Complete] Cross-modal contrastive" \
            "${output_dir}/console.log" 2>/dev/null; then
        echo "[Skip] pretrain=contrastive seed=$seed"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] pretrain=contrastive seed=$seed test=unused"
    python -u train_contrastive_pretrain.py \
        --output_dir "$output_dir" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch_size "$PRETRAIN_BATCH_SIZE" \
        --accum_steps "$PRETRAIN_ACCUM_STEPS" \
        --workers "$WORKERS" \
        --learning_rate "$PRETRAIN_LR" \
        --warmup_epochs 10 \
        --patience 15 \
        --seed "$seed" \
        --data_split_seed "$DATA_SPLIT_SEED" \
        2>&1 | tee "${output_dir}/console.log"
    [[ -s "$checkpoint" ]] || die "Missing checkpoint: $checkpoint"
}

baseline_checkpoint() {
    local experiment="$1"
    local seed="$2"
    case "$experiment" in
        physio_v2)
            resolve_physio_checkpoint "$seed"
            ;;
        multimodal_mae)
            echo "${PRETRAIN_PREFIX}_multimodal_mae_seed${seed}/multimodal_mae_best.pt"
            ;;
        contrastive)
            echo "${PRETRAIN_PREFIX}_contrastive_seed${seed}/contrastive_best.pt"
            ;;
        xmae)
            echo "${PRETRAIN_PREFIX}_xmae_seed${seed}/xmae_objective_best.pt"
            ;;
        *)
            die "Unknown objective comparison: $experiment"
            ;;
    esac
}

run_downstream() {
    local experiment="$1"
    local seed="$2"
    local checkpoint
    checkpoint="$(baseline_checkpoint "$experiment" "$seed")"
    [[ -s "$checkpoint" ]] || die \
        "Pre-training checkpoint is missing: $checkpoint"
    local output_dir="${OUTPUT_PREFIX}_${experiment}_seed${seed}"
    local saved_model="${output_dir}/downstream_multidisease_best.pt"
    local predictions="${output_dir}/validation_patient_predictions.csv"
    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$saved_model" ]] \
        && [[ -s "$predictions" ]] \
        && grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${output_dir}/downstream_console.log" 2>/dev/null; then
        echo "[Skip] downstream=$experiment seed=$seed"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] downstream=$experiment seed=$seed channel=both test=sealed"
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
        --experiment_id "P0_objective_${experiment}_seed${seed}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$seed" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"
    [[ -s "$saved_model" ]] || die "Missing downstream model: $saved_model"
    [[ -s "$predictions" ]] || die "Missing predictions: $predictions"
    grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
        "${output_dir}/downstream_console.log" || die \
        "Sealed completion marker missing: $output_dir"
}

read -r -a seed_values <<< "$SEEDS"
experiments=(physio_v2 multimodal_mae contrastive xmae)
for seed in "${seed_values[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed: $seed"
    physio_checkpoint="$(resolve_physio_checkpoint "$seed")"
    [[ -s "$physio_checkpoint" ]] || die \
        "PhysioV2 checkpoint is missing: $physio_checkpoint"
    run_reconstruction_pretrain multimodal_mae multimodal_mae "$seed"
    run_contrastive_pretrain "$seed"
    run_reconstruction_pretrain xmae xmae_objective "$seed"
    for experiment in "${experiments[@]}"; do
        run_downstream "$experiment" "$seed"
    done
done

python scripts/summarize_p0_pretraining_objectives.py \
    --output_prefix "$OUTPUT_PREFIX" \
    --pretrain_prefix "$PRETRAIN_PREFIX" \
    --paper_dir "$PAPER_DIR" \
    --physio_template "$PHYSIO_V2_TEMPLATE" \
    --seeds "${seed_values[@]}" \
    --experiments "${experiments[@]}"

echo "[Complete] P0 pre-training objective comparison | test_set_sealed=True"

