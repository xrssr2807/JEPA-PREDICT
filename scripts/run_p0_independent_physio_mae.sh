#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-42 3407 2026}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"

PARENT_ROOT="${PARENT_ROOT:-outputs_p0_independent_physio_parent}"
PHYSIO_PREFIX="${PHYSIO_PREFIX:-outputs_phase2_physio_v2}"
MAE_PREFIX="${MAE_PREFIX:-outputs_p0_pretrain_multimodal_mae}"
DOWNSTREAM_PREFIX="${DOWNSTREAM_PREFIX:-outputs_p0_independent}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_baselines/results/physio_mae_independent_pretrain}"

BASE_EPOCHS="${BASE_EPOCHS:-80}"
SP_EPOCHS="${SP_EPOCHS:-40}"
PHYSIO_EPOCHS="${PHYSIO_EPOCHS:-40}"
MAE_EPOCHS="${MAE_EPOCHS:-80}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-192}"
BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-2}"
PHYSIO_BATCH_SIZE="${PHYSIO_BATCH_SIZE:-128}"
PHYSIO_ACCUM_STEPS="${PHYSIO_ACCUM_STEPS:-3}"
BASE_LR="${BASE_LR:-2e-4}"
SP_LR="${SP_LR:-1e-4}"
PHYSIO_LR="${PHYSIO_LR:-5e-5}"
MAE_LR="${MAE_LR:-2e-4}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PRUNE_INTERMEDIATE="${PRUNE_INTERMEDIATE:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"

PHYSIO_SEED42_CHECKPOINT="${PHYSIO_SEED42_CHECKPOINT:-outputs_phase2_physio_v2_seed42/jepa_best.pt}"
MAE_SEED42_CHECKPOINT="${MAE_SEED42_CHECKPOINT:-outputs_p0_pretrain_multimodal_mae_seed42/multimodal_mae_best.pt}"
PHYSIO_SEED42_DOWNSTREAM="${PHYSIO_SEED42_DOWNSTREAM:-outputs_p0_objective_physio_v2_seed42}"
MAE_SEED42_DOWNSTREAM="${MAE_SEED42_DOWNSTREAM:-outputs_p0_objective_multimodal_mae_seed42}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

free_gb() {
    df -Pk /root/autodl-tmp | awk 'NR==2 {printf "%d\n", $4 / 1024 / 1024}'
}

require_free_space() {
    local available
    available="$(free_gb)"
    (( available >= MIN_FREE_GB )) || die \
        "Only ${available}GB free; require at least ${MIN_FREE_GB}GB"
    echo "[Disk] available=${available}GB required=${MIN_FREE_GB}GB"
}

next_epoch() {
    python -c '
import sys
import torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(c.get("epoch", -1)) + 1)
' "$1"
}

has_full_model_state() {
    python -c '
import sys
import torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if "model_state_dict" in c else 1)
' "$1"
}

is_pretrain_complete() {
    local directory="$1"
    local checkpoint="$2"
    [[ -s "$checkpoint" ]] \
        && grep -Fq "Pre-training complete." "${directory}/console.log" 2>/dev/null
}

is_downstream_complete() {
    local directory="$1"
    [[ -s "${directory}/downstream_multidisease_best.pt" ]] \
        && [[ -s "${directory}/validation_patient_predictions.csv" ]] \
        && grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${directory}/downstream_console.log" 2>/dev/null
}

run_jepa_stage() {
    local stage="$1"
    local seed="$2"
    local output_dir="$3"
    local init_checkpoint="${4:-}"
    local epochs batch_size accum_steps learning_rate
    local command=()
    local stage_flags=()
    local schedule_flags=()
    local initial_args=()

    case "$stage" in
        base)
            epochs="$BASE_EPOCHS"
            batch_size="$BASE_BATCH_SIZE"
            accum_steps="$BASE_ACCUM_STEPS"
            learning_rate="$BASE_LR"
            stage_flags=(--transport_mode full)
            schedule_flags=(--early_stop_patience 0)
            ;;
        shared_private)
            [[ -s "$init_checkpoint" ]] || die \
                "Shared-Private initialization is missing: $init_checkpoint"
            epochs="$SP_EPOCHS"
            batch_size="$BASE_BATCH_SIZE"
            accum_steps="$BASE_ACCUM_STEPS"
            learning_rate="$SP_LR"
            stage_flags=(--transport_mode full --shared_private)
            initial_args=(--init_checkpoint "$init_checkpoint")
            schedule_flags=(
                --transport_start_epoch 0
                --transport_ramp_epochs 1
                --shared_private_start_epoch 0
                --shared_private_ramp_epochs 5
                --early_stop_patience 15
                --early_stop_min_delta 1e-4
            )
            ;;
        physio_v2)
            [[ -s "$init_checkpoint" ]] || die \
                "PhysioV2 initialization is missing: $init_checkpoint"
            epochs="$PHYSIO_EPOCHS"
            batch_size="$PHYSIO_BATCH_SIZE"
            accum_steps="$PHYSIO_ACCUM_STEPS"
            learning_rate="$PHYSIO_LR"
            stage_flags=(--transport_mode physio_v2 --shared_private)
            initial_args=(--init_checkpoint "$init_checkpoint")
            schedule_flags=(
                --transport_start_epoch 0
                --transport_ramp_epochs 10
                --counterfactual_weight 0.10
                --counterfactual_margin 0.10
                --sinkhorn_iters 20
                --early_stop_patience 15
                --early_stop_min_delta 1e-4
            )
            ;;
        *) die "Unknown JEPA stage: $stage" ;;
    esac

    local best_checkpoint="${output_dir}/jepa_best.pt"
    local last_checkpoint="${output_dir}/jepa_last.pt"
    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && is_pretrain_complete "$output_dir" "$best_checkpoint"; then
        echo "[Skip] stage=$stage pretrain_seed=$seed"
        return
    fi

    local log_mode=()
    local start_args=("${initial_args[@]}")
    if [[ "$SKIP_COMPLETED" == "1" && -s "$last_checkpoint" ]]; then
        local start
        start="$(next_epoch "$last_checkpoint")"
        if (( start > 0 && start < epochs )); then
            start_args=(--resume "$last_checkpoint" --start_epoch "$start")
            log_mode=(-a)
            echo "[Resume] stage=$stage pretrain_seed=$seed epoch=$start"
        fi
    fi

    command=(
        python -u train_pretrain.py
        --phase 2
        "${stage_flags[@]}"
        "${start_args[@]}"
        --output_dir "$output_dir"
        --epochs "$epochs"
        --batch_size "$batch_size"
        --accum_steps "$accum_steps"
        --lr "$learning_rate"
        "${schedule_flags[@]}"
        --checkpoint_interval 0
        --workers "$WORKERS"
        --prefetch_factor "$PREFETCH_FACTOR"
        --performance_mode
        --seed "$seed"
        --data_split_seed "$DATA_SPLIT_SEED"
    )

    mkdir -p "$output_dir"
    printf "%q " "${command[@]}" > "${output_dir}/command.txt"
    printf "\n" >> "${output_dir}/command.txt"
    echo "[Run] stage=$stage pretrain_seed=$seed"
    "${command[@]}" 2>&1 | tee "${log_mode[@]}" "${output_dir}/console.log"
    is_pretrain_complete "$output_dir" "$best_checkpoint" || die \
        "Incomplete JEPA pretraining: $output_dir"
}

resolve_or_build_parent() {
    local seed="$1"
    local candidates=(
        "outputs_transport_pretrain_seed_study/on_seed${seed}/shared_private/jepa_best.pt"
        "outputs_phase2_shared_private_seed${seed}/jepa_best.pt"
        "${PARENT_ROOT}_seed${seed}/shared_private/jepa_best.pt"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -s "$candidate" ]] && has_full_model_state "$candidate"; then
            RESULT_CHECKPOINT="$candidate"
            return
        fi
    done

    local run_root="${PARENT_ROOT}_seed${seed}"
    local base_dir="${run_root}/base"
    local sp_dir="${run_root}/shared_private"
    run_jepa_stage base "$seed" "$base_dir"
    run_jepa_stage shared_private "$seed" "$sp_dir" "${base_dir}/jepa_last.pt"
    RESULT_CHECKPOINT="${sp_dir}/jepa_best.pt"
}

run_physio_pretrain() {
    local seed="$1"
    if [[ "$seed" == "42" ]]; then
        [[ -s "$PHYSIO_SEED42_CHECKPOINT" ]] || die \
            "PhysioV2 seed42 checkpoint missing: $PHYSIO_SEED42_CHECKPOINT"
        RESULT_CHECKPOINT="$PHYSIO_SEED42_CHECKPOINT"
        return
    fi
    local output_dir="${PHYSIO_PREFIX}_seed${seed}"
    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && is_pretrain_complete "$output_dir" "${output_dir}/jepa_best.pt"; then
        echo "[Skip] stage=physio_v2 pretrain_seed=$seed"
        RESULT_CHECKPOINT="${output_dir}/jepa_best.pt"
        return
    fi
    local parent
    resolve_or_build_parent "$seed"
    parent="$RESULT_CHECKPOINT"
    run_jepa_stage physio_v2 "$seed" "$output_dir" "$parent"
    RESULT_CHECKPOINT="${output_dir}/jepa_best.pt"
}

run_mae_pretrain() {
    local seed="$1"
    if [[ "$seed" == "42" ]]; then
        [[ -s "$MAE_SEED42_CHECKPOINT" ]] || die \
            "MAE seed42 checkpoint missing: $MAE_SEED42_CHECKPOINT"
        RESULT_CHECKPOINT="$MAE_SEED42_CHECKPOINT"
        return
    fi
    local output_dir="${MAE_PREFIX}_seed${seed}"
    local checkpoint="${output_dir}/multimodal_mae_best.pt"
    if [[ "$SKIP_COMPLETED" == "1" && -s "$checkpoint" ]] \
        && grep -Fq "[Complete] multimodal_mae baseline pre-training" \
            "${output_dir}/console.log" 2>/dev/null; then
        RESULT_CHECKPOINT="$checkpoint"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] stage=multimodal_mae pretrain_seed=$seed"
    python -u train_masked_reconstruction_pretrain.py \
        --objective multimodal_mae \
        --output_dir "$output_dir" \
        --epochs "$MAE_EPOCHS" \
        --batch_size 128 \
        --accum_steps 3 \
        --workers "$WORKERS" \
        --learning_rate "$MAE_LR" \
        --warmup_epochs 10 \
        --patience 15 \
        --seed "$seed" \
        --data_split_seed "$DATA_SPLIT_SEED" \
        2>&1 | tee "${output_dir}/console.log"
    [[ -s "$checkpoint" ]] || die "MAE checkpoint missing: $checkpoint"
    RESULT_CHECKPOINT="$checkpoint"
}

downstream_dir_for() {
    local method="$1"
    local seed="$2"
    if [[ "$seed" == "42" ]]; then
        if [[ "$method" == "physio_v2" ]]; then
            echo "$PHYSIO_SEED42_DOWNSTREAM"
        else
            echo "$MAE_SEED42_DOWNSTREAM"
        fi
    else
        echo "${DOWNSTREAM_PREFIX}_${method}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    fi
}

run_downstream() {
    local method="$1"
    local seed="$2"
    local checkpoint="$3"
    local output_dir
    output_dir="$(downstream_dir_for "$method" "$seed")"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_downstream_complete "$output_dir"; then
        echo "[Skip] downstream method=$method pretrain_seed=$seed"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] downstream method=$method pretrain_seed=$seed ft_seed=$DOWNSTREAM_SEED"
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
        --experiment_id "P0_independent_${method}_preseed${seed}_ftseed${DOWNSTREAM_SEED}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$DOWNSTREAM_SEED" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"
    is_downstream_complete "$output_dir" || die \
        "Incomplete downstream run: $output_dir"
}

archive_seed() {
    local seed="$1"
    local physio_checkpoint="$2"
    local mae_checkpoint="$3"
    local seed_dir="${PAPER_DIR}/runs/pretrain_seed${seed}"
    mkdir -p "$seed_dir"
    {
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "pretrain_seed=$seed"
        echo "data_split_seed=$DATA_SPLIT_SEED"
        echo "downstream_seed=$DOWNSTREAM_SEED"
        echo "physio_checkpoint=$physio_checkpoint"
        echo "physio_sha256=$(sha256sum "$physio_checkpoint" | awk '{print $1}')"
        echo "mae_checkpoint=$mae_checkpoint"
        echo "mae_sha256=$(sha256sum "$mae_checkpoint" | awk '{print $1}')"
        echo "split_sha256=$(sha256sum "$SPLIT" | awk '{print $1}')"
        echo "test_status=sealed"
    } > "${seed_dir}/manifest.txt"
}

prune_seed() {
    local seed="$1"
    [[ "$PRUNE_INTERMEDIATE" == "1" && "$seed" != "42" ]] || return 0
    local run_root="${PARENT_ROOT}_seed${seed}"
    local physio_dir="${PHYSIO_PREFIX}_seed${seed}"
    local mae_dir="${MAE_PREFIX}_seed${seed}"
    local prune_log="${PAPER_DIR}/runs/pretrain_seed${seed}/pruned_artifacts.txt"
    {
        echo "pruned_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        for path in \
            "${run_root}/base/jepa_best.pt" \
            "${run_root}/base/jepa_last.pt" \
            "${run_root}/shared_private/jepa_best.pt" \
            "${run_root}/shared_private/jepa_last.pt" \
            "${physio_dir}/jepa_last.pt" \
            "${mae_dir}/multimodal_mae_last.pt"; do
            [[ -e "$path" ]] && stat -c "removed=%s_bytes %n" "$path"
        done
    } > "$prune_log"
    rm -f -- \
        "${run_root}/base/jepa_best.pt" \
        "${run_root}/base/jepa_last.pt" \
        "${run_root}/shared_private/jepa_best.pt" \
        "${run_root}/shared_private/jepa_last.pt" \
        "${physio_dir}/jepa_last.pt" \
        "${mae_dir}/multimodal_mae_last.pt"
    echo "[Prune] pretrain_seed=$seed kept final PhysioV2 and MAE best checkpoints"
}

[[ -s "$SPLIT" ]] || die "Frozen downstream split missing: $SPLIT"
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_split_sha"
[[ -d /root/autodl-tmp/split_processed ]] || die \
    "Preprocessed pretraining data missing: /root/autodl-tmp/split_processed"
[[ "$DATA_SPLIT_SEED" == "42" ]] || die \
    "This study requires frozen pretraining data split seed 42"
pretrain_help="$(python train_pretrain.py --help 2>&1)"
for option in transport_mode shared_private data_split_seed checkpoint_interval; do
    [[ "$pretrain_help" == *"--${option}"* ]] || die \
        "train_pretrain.py lacks --${option}"
done
downstream_help="$(python train_downstream.py --help 2>&1)"
for option in encoder_init encoder_arch seal_test experiment_id; do
    [[ "$downstream_help" == *"--${option}"* ]] || die \
        "train_downstream.py lacks --${option}"
done
require_free_space
mkdir -p "$PAPER_DIR/runs"

read -r -a seeds <<< "$PRETRAIN_SEEDS"
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid pretraining seed: $seed"
    [[ "$seed" == "42" ]] || require_free_space
    run_physio_pretrain "$seed"
    physio_checkpoint="$RESULT_CHECKPOINT"
    run_mae_pretrain "$seed"
    mae_checkpoint="$RESULT_CHECKPOINT"
    [[ -s "$physio_checkpoint" ]] || die "Physio checkpoint missing: $physio_checkpoint"
    [[ -s "$mae_checkpoint" ]] || die "MAE checkpoint missing: $mae_checkpoint"
    run_downstream physio_v2 "$seed" "$physio_checkpoint"
    run_downstream multimodal_mae "$seed" "$mae_checkpoint"
    archive_seed "$seed" "$physio_checkpoint" "$mae_checkpoint"
    prune_seed "$seed"
done

python scripts/summarize_p0_independent_physio_mae.py \
    --pretrain_seeds "${seeds[@]}" \
    --downstream_seed "$DOWNSTREAM_SEED" \
    --downstream_prefix "$DOWNSTREAM_PREFIX" \
    --physio_seed42_dir "$PHYSIO_SEED42_DOWNSTREAM" \
    --mae_seed42_dir "$MAE_SEED42_DOWNSTREAM" \
    --paper_dir "$PAPER_DIR"

cp "$SPLIT" "${PAPER_DIR}/multidisease_taskaware_downstream.json"
echo "[Complete] P0 independent PhysioV2 vs Multimodal MAE pretraining seeds"
echo "[Complete] pretrain_seeds=$PRETRAIN_SEEDS downstream_seed=$DOWNSTREAM_SEED test=sealed"
