#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
SEEDS="${SEEDS:-42 3407 2026}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p3}"
CONTRASTIVE_DIR="${CONTRASTIVE_DIR:-outputs_p3_contrastive_pretrain_seed42}"
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-30}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_baselines/results}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

[[ -s "$SPLIT" ]] || die "Split is missing: $SPLIT"
[[ -d /root/autodl-tmp/split_processed ]] \
    || die "Preprocessed pre-training data is missing"

for option in encoder_arch encoder_init seal_test experiment_id; do
    python train_downstream.py --help 2>&1 | grep -q -- "--${option}" \
        || die "train_downstream.py does not support --${option}"
done

contrastive_checkpoint="${CONTRASTIVE_DIR}/contrastive_best.pt"
if [[ ! -s "$contrastive_checkpoint" ]] \
    || ! grep -q "\[Complete\] Cross-modal contrastive" \
        "${CONTRASTIVE_DIR}/console.log" 2>/dev/null; then
    mkdir -p "$CONTRASTIVE_DIR"
    echo "[Run] Cross-modal ECG/PPG InfoNCE pre-training baseline"
    python -u train_contrastive_pretrain.py \
        --output_dir "$CONTRASTIVE_DIR" \
        --epochs "$CONTRASTIVE_EPOCHS" \
        --batch_size 32 \
        --accum_steps 4 \
        --workers "$WORKERS" \
        --seed 42 \
        2>&1 | tee "${CONTRASTIVE_DIR}/console.log"
fi
[[ -s "$contrastive_checkpoint" ]] \
    || die "Contrastive checkpoint was not produced"

run_downstream() {
    local experiment="$1"
    local seed="$2"
    local architecture="jepa_transformer"
    local initialization="random"
    local checkpoint_args=()
    local output_dir="${OUTPUT_PREFIX}_${experiment}_seed${seed}"
    local saved_model="${output_dir}/downstream_multidisease_best.pt"
    local predictions="${output_dir}/validation_patient_predictions.csv"

    case "$experiment" in
        transformer_scratch)
            ;;
        resnet1d)
            architecture="resnet1d"
            ;;
        contrastive)
            initialization="pretrained"
            checkpoint_args=(--checkpoint "$contrastive_checkpoint")
            ;;
        *)
            die "Unknown P3 baseline: $experiment"
            ;;
    esac

    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$saved_model" ]] \
        && [[ -s "$predictions" ]] \
        && grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${output_dir}/downstream_console.log" 2>/dev/null; then
        echo "[Skip] experiment=$experiment seed=$seed"
        return
    fi

    mkdir -p "$output_dir"
    echo "[Run] P3 experiment=$experiment seed=$seed test=sealed"
    python -u train_downstream.py \
        "${checkpoint_args[@]}" \
        --encoder_init "$initialization" \
        --encoder_arch "$architecture" \
        --dataset multidisease \
        --multidisease_channel both \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --patient_mil on \
        --multiscale on \
        --experiment_id "P3_${experiment}_seed${seed}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$seed" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"

    [[ -s "$saved_model" ]] || die "Missing checkpoint: $saved_model"
    [[ -s "$predictions" ]] || die "Missing patient predictions: $predictions"
    grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
        "${output_dir}/downstream_console.log" \
        || die "Sealed completion marker missing: $output_dir"
}

read -r -a seed_values <<< "$SEEDS"
experiments=(transformer_scratch resnet1d contrastive)
for seed in "${seed_values[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed: $seed"
    for experiment in "${experiments[@]}"; do
        run_downstream "$experiment" "$seed"
    done
done

python scripts/summarize_p3_baselines.py \
    --output_prefix "$OUTPUT_PREFIX" \
    --paper_dir "$PAPER_DIR" \
    --seeds "${seed_values[@]}" \
    --experiments "${experiments[@]}"

echo "[Complete] P3 internal baseline sequence | test_set_sealed=True"
