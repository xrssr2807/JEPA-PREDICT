#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-42}"
ALPHAS="${ALPHAS:-0 0.1 0.25 0.5 1.0}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SYMMETRIC_CHECKPOINT="${SYMMETRIC_CHECKPOINT:-outputs_phase2_physio_v2_seed42/jepa_best.pt}"
PRETRAIN_PREFIX="${PRETRAIN_PREFIX:-outputs_phase2_direction_weight}"
DOWNSTREAM_PREFIX="${DOWNSTREAM_PREFIX:-outputs_direction_weight}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_direction_weight/results/seed${SEED}}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-3}"
LR="${LR:-5e-5}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
KEEP_LAST="${KEEP_LAST:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

alpha_tag() {
    case "$1" in
        0|0.0|0.00) echo "a000" ;;
        0.1|0.10) echo "a010" ;;
        0.25) echo "a025" ;;
        0.5|0.50) echo "a050" ;;
        1|1.0|1.00) echo "a100" ;;
        *) die "Unsupported alpha '$1'; use 0 0.1 0.25 0.5 1.0" ;;
    esac
}

[[ -s "$INIT_CHECKPOINT" ]] || die "Initialization checkpoint missing: $INIT_CHECKPOINT"
[[ -s "$SYMMETRIC_CHECKPOINT" ]] || die "Symmetric alpha=1 checkpoint missing: $SYMMETRIC_CHECKPOINT"
[[ -s "$SPLIT" ]] || die "Frozen downstream split missing: $SPLIT"
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_split_sha"
python train_pretrain.py --help 2>&1 | grep -q -- "--reverse_loss_weight" || die \
    "Current train_pretrain.py does not support asymmetric direction weights"
python train_downstream.py --help 2>&1 | grep -q -- "--seal_test" || die \
    "Current train_downstream.py does not support sealed evaluation"

mkdir -p "$PAPER_DIR"
read -r -a alpha_values <<< "$ALPHAS"

for alpha in "${alpha_values[@]}"; do
    tag="$(alpha_tag "$alpha")"
    pretrain_dir="${PRETRAIN_PREFIX}_${tag}_seed${SEED}"
    pretrain_checkpoint="${pretrain_dir}/jepa_best.pt"
    if [[ "$tag" == "a100" ]]; then
        pretrain_checkpoint="$SYMMETRIC_CHECKPOINT"
        echo "[Reuse] alpha=1 symmetric checkpoint: $pretrain_checkpoint"
    elif [[ "$SKIP_COMPLETED" == "1" && -s "$pretrain_checkpoint" ]]; then
        echo "[Skip] pretrain alpha=$alpha checkpoint=$pretrain_checkpoint"
    else
        mkdir -p "$pretrain_dir"
        echo "============================================================"
        echo "[Run] pretrain alpha=$alpha ECG->PPG=1 PPG->ECG=$alpha"
        echo "============================================================"
        python -u train_pretrain.py \
            --phase 2 \
            --transport_mode physio_v2 \
            --shared_private \
            --reverse_loss_weight "$alpha" \
            --init_checkpoint "$INIT_CHECKPOINT" \
            --output_dir "$pretrain_dir" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --lr "$LR" \
            --transport_start_epoch 0 \
            --transport_ramp_epochs 10 \
            --counterfactual_weight 0.10 \
            --counterfactual_margin 0.10 \
            --sinkhorn_iters 20 \
            --checkpoint_interval 0 \
            --workers "$WORKERS" \
            --seed "$SEED" \
            2>&1 | tee "${pretrain_dir}/console.log"
        [[ -s "$pretrain_checkpoint" ]] || die \
            "Missing pretrain checkpoint: $pretrain_checkpoint"
        if [[ "$KEEP_LAST" != "1" ]]; then
            rm -f "${pretrain_dir}/jepa_last.pt"
        fi
    fi

    downstream_dir="${DOWNSTREAM_PREFIX}_${tag}_both_seed${SEED}"
    downstream_checkpoint="${downstream_dir}/downstream_multidisease_best.pt"
    predictions="${downstream_dir}/validation_patient_predictions.csv"
    if [[ "$SKIP_COMPLETED" == "1" \
        && -s "$downstream_checkpoint" \
        && -s "$predictions" \
        && -f "${downstream_dir}/downstream_console.log" \
        ]] && grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${downstream_dir}/downstream_console.log"; then
        echo "[Skip] downstream alpha=$alpha"
    else
        mkdir -p "$downstream_dir"
        echo "[Run] downstream alpha=$alpha channel=both test=sealed"
        python -u train_downstream.py \
            --checkpoint "$pretrain_checkpoint" \
            --encoder_init pretrained \
            --encoder_arch jepa_transformer \
            --dataset multidisease \
            --multidisease_channel both \
            --multidisease_split "$SPLIT" \
            --shared_private_head off \
            --patient_mil on \
            --multiscale on \
            --experiment_id "P0_direction_weight_${tag}_seed${SEED}" \
            --output_dir "$downstream_dir" \
            --mil_batch_size "$MIL_BATCH_SIZE" \
            --mil_chunk_size "$MIL_CHUNK_SIZE" \
            --workers "$WORKERS" \
            --seed "$SEED" \
            --seal_test \
            2>&1 | tee "${downstream_dir}/downstream_console.log"
        [[ -s "$downstream_checkpoint" ]] || die \
            "Missing downstream checkpoint: $downstream_checkpoint"
        [[ -s "$predictions" ]] || die \
            "Missing validation predictions: $predictions"
        grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${downstream_dir}/downstream_console.log" || die \
            "Sealed completion marker missing: $downstream_dir"
    fi

    run_dir="${PAPER_DIR}/${tag}"
    mkdir -p "$run_dir"
    {
        echo "git_sha=$(git rev-parse HEAD)"
        echo "seed=$SEED"
        echo "alpha=$alpha"
        echo "forward_weight=1.0"
        echo "reverse_weight=$alpha"
        echo "normalization_denominator=$(python -c "print(1.0 + float('$alpha'))")"
        echo "split=$SPLIT"
        echo "split_sha256=$actual_split_sha"
        echo "init_checkpoint=$INIT_CHECKPOINT"
        echo "init_sha256=$(sha256sum "$INIT_CHECKPOINT" | awk '{print $1}')"
        echo "pretrain_checkpoint=$pretrain_checkpoint"
        echo "pretrain_sha256=$(sha256sum "$pretrain_checkpoint" | awk '{print $1}')"
        echo "downstream_checkpoint=$downstream_checkpoint"
        echo "test_status=sealed"
    } > "${run_dir}/run_manifest.txt"
done

python scripts/summarize_direction_weight_ablation.py \
    --alphas "${alpha_values[@]}" \
    --seed "$SEED" \
    --pretrain_prefix "$PRETRAIN_PREFIX" \
    --downstream_prefix "$DOWNSTREAM_PREFIX" \
    --symmetric_checkpoint "$SYMMETRIC_CHECKPOINT" \
    --split "$SPLIT" \
    --paper_dir "$PAPER_DIR"

echo "[Complete] asymmetric direction-weight pilot | test_set_sealed=True"
