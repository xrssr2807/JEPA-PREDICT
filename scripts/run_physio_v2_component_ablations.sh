#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VARIANTS="${VARIANTS:-full no_content global_only fixed_delay hard_delay no_monotonic no_smoothness no_dustbin no_counterfactual}"
SEED="${SEED:-42}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
FULL_CHECKPOINT="${FULL_CHECKPOINT:-outputs_phase2_physio_v2_seed${SEED}/jepa_best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_physio_v2_component_ablation}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_physio_v2_component_ablation/results/seed${SEED}}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-3}"
LR="${LR:-5e-5}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
MIN_FREE_GB="${MIN_FREE_GB:-2}"
DELETE_LARGE_CHECKPOINTS="${DELETE_LARGE_CHECKPOINTS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { echo "[Error] $*" >&2; exit 1; }

free_gb() {
    df -Pk "$ROOT_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

require_space() {
    local available
    available="$(free_gb)"
    (( available >= MIN_FREE_GB )) || die \
        "Only ${available}GB free; require ${MIN_FREE_GB}GB"
}

variant_args() {
    local variant="$1"
    EXTRA_ARGS=()
    case "$variant" in
        full) ;;
        no_content) EXTRA_ARGS+=(--v2_content_weight 0) ;;
        global_only) EXTRA_ARGS+=(--v2_local_delay_weight 0) ;;
        fixed_delay)
            EXTRA_ARGS+=(--v2_delay_policy fixed_prior)
            ;;
        hard_delay) EXTRA_ARGS+=(--v2_delay_policy hard_argmax) ;;
        no_monotonic) EXTRA_ARGS+=(--monotonic_weight 0) ;;
        no_smoothness) EXTRA_ARGS+=(--delay_smoothness_weight 0) ;;
        no_dustbin) EXTRA_ARGS+=(--v2_dustbin off) ;;
        no_counterfactual) EXTRA_ARGS+=(--counterfactual_weight 0) ;;
        *) die "Unknown variant: $variant" ;;
    esac
}

for path in "$INIT_CHECKPOINT" "$FULL_CHECKPOINT" "$SPLIT"; do
    [[ -s "$path" ]] || die "Required input missing: $path"
done
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: $actual_split_sha"
for flag in v2_delay_policy v2_dustbin v2_content_weight \
    v2_local_delay_weight monotonic_weight delay_smoothness_weight; do
    python train_pretrain.py --help 2>&1 | grep -q -- "--${flag}" || die \
        "train_pretrain.py lacks --${flag}"
done
python train_downstream.py --help 2>&1 | grep -q -- "--seal_test" || die \
    "train_downstream.py lacks --seal_test"
require_space

mkdir -p "$OUTPUT_ROOT" "$PAPER_DIR" "${PAPER_DIR}/commands"
read -r -a variant_values <<< "$VARIANTS"

for variant in "${variant_values[@]}"; do
    run_dir="${PAPER_DIR}/${variant}"
    result_json="${run_dir}/result.json"
    if [[ -s "$result_json" ]]; then
        echo "[Skip] completed variant=$variant"
        continue
    fi
    require_space
    mkdir -p "$run_dir"
    variant_args "$variant"

    pretrain_dir="${OUTPUT_ROOT}/${variant}_seed${SEED}"
    pretrain_checkpoint="${pretrain_dir}/jepa_best.pt"
    if [[ "$variant" == "full" ]]; then
        pretrain_checkpoint="$FULL_CHECKPOINT"
        echo "[Reuse] full=$pretrain_checkpoint"
    else
        mkdir -p "$pretrain_dir"
        printf '%q ' python -u train_pretrain.py \
            --phase 2 --transport_mode physio_v2 --shared_private \
            --init_checkpoint "$INIT_CHECKPOINT" \
            --output_dir "$pretrain_dir" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" --accum_steps "$ACCUM_STEPS" \
            --lr "$LR" --transport_start_epoch 0 --transport_ramp_epochs 10 \
            --counterfactual_weight 0.10 --counterfactual_margin 0.10 \
            --sinkhorn_iters 20 --checkpoint_interval 0 \
            --workers "$WORKERS" --seed "$SEED" "${EXTRA_ARGS[@]}" \
            > "${PAPER_DIR}/commands/${variant}_pretrain.sh"
        printf '\n' >> "${PAPER_DIR}/commands/${variant}_pretrain.sh"
        python -u train_pretrain.py \
            --phase 2 --transport_mode physio_v2 --shared_private \
            --init_checkpoint "$INIT_CHECKPOINT" \
            --output_dir "$pretrain_dir" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" --accum_steps "$ACCUM_STEPS" \
            --lr "$LR" --transport_start_epoch 0 --transport_ramp_epochs 10 \
            --counterfactual_weight 0.10 --counterfactual_margin 0.10 \
            --sinkhorn_iters 20 --checkpoint_interval 0 \
            --workers "$WORKERS" --seed "$SEED" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${pretrain_dir}/console.log"
        [[ -s "$pretrain_checkpoint" ]] || die \
            "Pretrain checkpoint missing: $pretrain_checkpoint"
    fi

    downstream_dir="${OUTPUT_ROOT}/${variant}_preseed${SEED}_ftseed${DOWNSTREAM_SEED}"
    mkdir -p "$downstream_dir"
    printf '%q ' python -u train_downstream.py \
        --checkpoint "$pretrain_checkpoint" --encoder_init pretrained \
        --encoder_arch jepa_transformer --dataset multidisease \
        --multidisease_channel both --multidisease_split "$SPLIT" \
        --shared_private_head off --patient_mil on --multiscale on \
        --experiment_id "P2_physio_v2_component_${variant}_preseed${SEED}" \
        --output_dir "$downstream_dir" --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" --workers "$WORKERS" \
        --seed "$DOWNSTREAM_SEED" --seal_test \
        > "${PAPER_DIR}/commands/${variant}_downstream.sh"
    printf '\n' >> "${PAPER_DIR}/commands/${variant}_downstream.sh"
    python -u train_downstream.py \
        --checkpoint "$pretrain_checkpoint" --encoder_init pretrained \
        --encoder_arch jepa_transformer --dataset multidisease \
        --multidisease_channel both --multidisease_split "$SPLIT" \
        --shared_private_head off --patient_mil on --multiscale on \
        --experiment_id "P2_physio_v2_component_${variant}_preseed${SEED}" \
        --output_dir "$downstream_dir" --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" --workers "$WORKERS" \
        --seed "$DOWNSTREAM_SEED" --seal_test \
        2>&1 | tee "${downstream_dir}/downstream_console.log"

    downstream_checkpoint="${downstream_dir}/downstream_multidisease_best.pt"
    predictions="${downstream_dir}/validation_patient_predictions.csv"
    [[ -s "$downstream_checkpoint" && -s "$predictions" ]] || die \
        "Downstream outputs incomplete: $downstream_dir"
    grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
        "${downstream_dir}/downstream_console.log" || die \
        "Sealed completion marker missing: $downstream_dir"

    python scripts/summarize_physio_v2_component_ablations.py \
        --capture --variant "$variant" --pretrain "$pretrain_checkpoint" \
        --downstream "$downstream_checkpoint" --split "$SPLIT" \
        --seed "$SEED" --downstream_seed "$DOWNSTREAM_SEED" \
        --output "$result_json"
    cp "$predictions" "$run_dir/validation_patient_predictions.csv"
    cp "${downstream_dir}/downstream_console.log" "$run_dir/"
    if [[ "$variant" != "full" ]]; then
        cp "${pretrain_dir}/console.log" "$run_dir/pretrain_console.log"
        [[ -s "${pretrain_dir}/pretrain_log.txt" ]] && \
            cp "${pretrain_dir}/pretrain_log.txt" "$run_dir/"
        [[ -s "${pretrain_dir}/pretrain_split.json" ]] && \
            cp "${pretrain_dir}/pretrain_split.json" "$run_dir/"
    fi
    {
        echo "git_sha=$(git rev-parse HEAD)"
        echo "variant=$variant"
        echo "pretrain_seed=$SEED"
        echo "downstream_seed=$DOWNSTREAM_SEED"
        echo "split_sha256=$actual_split_sha"
        echo "test_status=sealed"
        echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "free_gb_after_capture=$(free_gb)"
    } > "$run_dir/run_manifest.txt"

    if [[ "$DELETE_LARGE_CHECKPOINTS" == "1" ]]; then
        rm -f -- "$downstream_checkpoint" "${downstream_dir}/downstream_multidisease_last.pt"
        if [[ "$variant" != "full" ]]; then
            rm -f -- "$pretrain_checkpoint" "${pretrain_dir}/jepa_last.pt"
        fi
    fi
    echo "[Done] variant=$variant free_gb=$(free_gb)"
done

python scripts/summarize_physio_v2_component_ablations.py \
    --paper_dir "$PAPER_DIR" --variants "${variant_values[@]}"
echo "[Complete] PhysioV2-v2 component ablation | test_set_sealed=True"
