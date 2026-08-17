#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-42 3407 2026}"
ALPHAS="${ALPHAS:-0.5 1.0}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716}"

PARENT_PREFIX="${PARENT_PREFIX:-outputs_direction_weight_parent}"
PRETRAIN_PREFIX="${PRETRAIN_PREFIX:-outputs_direction_weight_repro}"
DOWNSTREAM_PREFIX="${DOWNSTREAM_PREFIX:-outputs_direction_weight_repro}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_direction_weight/results/independent_pretrain_seeds}"

BASE_EPOCHS="${BASE_EPOCHS:-80}"
SP_EPOCHS="${SP_EPOCHS:-40}"
PHYSIO_EPOCHS="${PHYSIO_EPOCHS:-40}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-192}"
BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-2}"
PHYSIO_BATCH_SIZE="${PHYSIO_BATCH_SIZE:-128}"
PHYSIO_ACCUM_STEPS="${PHYSIO_ACCUM_STEPS:-3}"
BASE_LR="${BASE_LR:-2e-4}"
SP_LR="${SP_LR:-1e-4}"
PHYSIO_LR="${PHYSIO_LR:-5e-5}"
PRETRAIN_WORKERS="${PRETRAIN_WORKERS:-0}"
DOWNSTREAM_WORKERS="${DOWNSTREAM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PRUNE_PARENTS="${PRUNE_PARENTS:-1}"
SLIM_COMPLETED="${SLIM_COMPLETED:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
SEED42_PARENT="${SEED42_PARENT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

alpha_tag() {
    case "$1" in
        0.5|0.50) echo "a050" ;;
        1|1.0|1.00) echo "a100" ;;
        *) die "Unsupported alpha '$1'; expected 0.5 or 1.0" ;;
    esac
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

is_pretrain_complete() {
    local directory="$1"
    [[ -s "${directory}/jepa_best.pt" ]] \
        && grep -Fq "Pre-training complete." "${directory}/console.log" 2>/dev/null
}

is_downstream_complete() {
    local directory="$1"
    [[ -s "${directory}/downstream_multidisease_best.pt" ]] \
        && [[ -s "${directory}/validation_patient_predictions.csv" ]] \
        && grep -Fq \
            "strict_validation_only=true test_dataset_constructed=false" \
            "${directory}/downstream_console.log" 2>/dev/null \
        && grep -Fq "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${directory}/downstream_console.log" 2>/dev/null
}

run_base_stage() {
    local seed="$1"
    local output_dir="$2"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_pretrain_complete "$output_dir"; then
        echo "[Skip] base parent seed=$seed"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] base parent seed=$seed"
    python -u train_pretrain.py \
        --phase 2 \
        --transport_mode full \
        --output_dir "$output_dir" \
        --epochs "$BASE_EPOCHS" \
        --batch_size "$BASE_BATCH_SIZE" \
        --accum_steps "$BASE_ACCUM_STEPS" \
        --lr "$BASE_LR" \
        --early_stop_patience 0 \
        --checkpoint_interval 0 \
        --workers "$PRETRAIN_WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        --performance_mode \
        --seed "$seed" \
        --data_split_seed "$DATA_SPLIT_SEED" \
        2>&1 | tee "${output_dir}/console.log"
    is_pretrain_complete "$output_dir" || die "Incomplete base stage: $output_dir"
}

run_shared_private_stage() {
    local seed="$1"
    local init_checkpoint="$2"
    local output_dir="$3"
    [[ -s "$init_checkpoint" ]] || die "Missing base checkpoint: $init_checkpoint"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_pretrain_complete "$output_dir"; then
        echo "[Skip] shared-private parent seed=$seed"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] shared-private parent seed=$seed"
    python -u train_pretrain.py \
        --phase 2 \
        --transport_mode full \
        --shared_private \
        --init_checkpoint "$init_checkpoint" \
        --output_dir "$output_dir" \
        --epochs "$SP_EPOCHS" \
        --batch_size "$BASE_BATCH_SIZE" \
        --accum_steps "$BASE_ACCUM_STEPS" \
        --lr "$SP_LR" \
        --transport_start_epoch 0 \
        --transport_ramp_epochs 1 \
        --shared_private_start_epoch 0 \
        --shared_private_ramp_epochs 5 \
        --early_stop_patience 15 \
        --early_stop_min_delta 1e-4 \
        --checkpoint_interval 0 \
        --workers "$PRETRAIN_WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        --performance_mode \
        --seed "$seed" \
        --data_split_seed "$DATA_SPLIT_SEED" \
        2>&1 | tee "${output_dir}/console.log"
    is_pretrain_complete "$output_dir" || die \
        "Incomplete shared-private stage: $output_dir"
}

resolve_or_build_parent() {
    local seed="$1"
    local parent_root="${PARENT_PREFIX}_seed${seed}"
    local base_dir="${parent_root}/base"
    local sp_dir="${parent_root}/shared_private"

    if [[ "$seed" == "42" && -s "$SEED42_PARENT" ]]; then
        RESULT_CHECKPOINT="$SEED42_PARENT"
        RESULT_PARENT_ROOT=""
        echo "[Reuse] seed42 common parent: $RESULT_CHECKPOINT"
        return
    fi
    if is_pretrain_complete "$sp_dir"; then
        RESULT_CHECKPOINT="${sp_dir}/jepa_best.pt"
        RESULT_PARENT_ROOT="$parent_root"
        echo "[Reuse] strict common parent seed=$seed: $RESULT_CHECKPOINT"
        return
    fi

    require_free_space
    run_base_stage "$seed" "$base_dir"
    local base_checkpoint="${base_dir}/jepa_last.pt"
    [[ -s "$base_checkpoint" ]] || base_checkpoint="${base_dir}/jepa_best.pt"
    run_shared_private_stage "$seed" "$base_checkpoint" "$sp_dir"
    RESULT_CHECKPOINT="${sp_dir}/jepa_best.pt"
    RESULT_PARENT_ROOT="$parent_root"
}

run_physio_stage() {
    local seed="$1"
    local alpha="$2"
    local parent_checkpoint="$3"
    local tag output_dir
    tag="$(alpha_tag "$alpha")"
    output_dir="${PRETRAIN_PREFIX}_${tag}_preseed${seed}"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_pretrain_complete "$output_dir"; then
        echo "[Skip] PhysioV2 alpha=$alpha pretrain_seed=$seed"
        RESULT_CHECKPOINT="${output_dir}/jepa_best.pt"
        return
    fi
    require_free_space
    mkdir -p "$output_dir"
    echo "[Run] PhysioV2 alpha=$alpha pretrain_seed=$seed"
    python -u train_pretrain.py \
        --phase 2 \
        --transport_mode physio_v2 \
        --shared_private \
        --reverse_loss_weight "$alpha" \
        --init_checkpoint "$parent_checkpoint" \
        --output_dir "$output_dir" \
        --epochs "$PHYSIO_EPOCHS" \
        --batch_size "$PHYSIO_BATCH_SIZE" \
        --accum_steps "$PHYSIO_ACCUM_STEPS" \
        --lr "$PHYSIO_LR" \
        --transport_start_epoch 0 \
        --transport_ramp_epochs 10 \
        --counterfactual_weight 0.10 \
        --counterfactual_margin 0.10 \
        --sinkhorn_iters 20 \
        --early_stop_patience 15 \
        --early_stop_min_delta 1e-4 \
        --checkpoint_interval 0 \
        --workers "$PRETRAIN_WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        --performance_mode \
        --seed "$seed" \
        --data_split_seed "$DATA_SPLIT_SEED" \
        2>&1 | tee "${output_dir}/console.log"
    is_pretrain_complete "$output_dir" || die "Incomplete PhysioV2 stage: $output_dir"
    rm -f "${output_dir}/jepa_last.pt"
    RESULT_CHECKPOINT="${output_dir}/jepa_best.pt"
}

run_downstream() {
    local seed="$1"
    local alpha="$2"
    local checkpoint="$3"
    local tag output_dir
    tag="$(alpha_tag "$alpha")"
    output_dir="${DOWNSTREAM_PREFIX}_${tag}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_downstream_complete "$output_dir"; then
        echo "[Skip] downstream alpha=$alpha pretrain_seed=$seed"
        RESULT_DOWNSTREAM_DIR="$output_dir"
        return
    fi
    mkdir -p "$output_dir"
    echo "[Run] downstream alpha=$alpha pretrain_seed=$seed ft_seed=$DOWNSTREAM_SEED"
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
        --experiment_id "P2_direction_weight_${tag}_preseed${seed}_ftseed${DOWNSTREAM_SEED}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$DOWNSTREAM_WORKERS" \
        --seed "$DOWNSTREAM_SEED" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"
    is_downstream_complete "$output_dir" || die "Incomplete downstream run: $output_dir"
    RESULT_DOWNSTREAM_DIR="$output_dir"
}

archive_run() {
    local seed="$1"
    local alpha="$2"
    local parent_checkpoint="$3"
    local pretrain_checkpoint="$4"
    local downstream_dir="$5"
    local tag run_dir
    tag="$(alpha_tag "$alpha")"
    run_dir="${PAPER_DIR}/runs/${tag}_preseed${seed}"
    mkdir -p "$run_dir"
    {
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "alpha=$alpha"
        echo "forward_weight=1.0"
        echo "reverse_weight=$alpha"
        echo "pretrain_seed=$seed"
        echo "data_split_seed=$DATA_SPLIT_SEED"
        echo "downstream_seed=$DOWNSTREAM_SEED"
        echo "base_batch_size=$BASE_BATCH_SIZE"
        echo "base_accum_steps=$BASE_ACCUM_STEPS"
        echo "base_effective_batch=$((BASE_BATCH_SIZE * BASE_ACCUM_STEPS))"
        echo "physio_batch_size=$PHYSIO_BATCH_SIZE"
        echo "physio_accum_steps=$PHYSIO_ACCUM_STEPS"
        echo "physio_effective_batch=$((PHYSIO_BATCH_SIZE * PHYSIO_ACCUM_STEPS))"
        echo "pretrain_workers=$PRETRAIN_WORKERS"
        echo "mil_batch_size=$MIL_BATCH_SIZE"
        echo "mil_chunk_size=$MIL_CHUNK_SIZE"
        echo "downstream_workers=$DOWNSTREAM_WORKERS"
        echo "parent_checkpoint=$parent_checkpoint"
        echo "parent_sha256=$(sha256sum "$parent_checkpoint" | awk '{print $1}')"
        echo "pretrain_checkpoint=$pretrain_checkpoint"
        echo "pretrain_sha256=$(sha256sum "$pretrain_checkpoint" | awk '{print $1}')"
        echo "downstream_dir=$downstream_dir"
        echo "split=$SPLIT"
        echo "split_sha256=$(sha256sum "$SPLIT" | awk '{print $1}')"
        echo "test_status=sealed"
    } > "${run_dir}/manifest.txt"
}

slim_checkpoint() {
    local checkpoint="$1"
    [[ "$SLIM_COMPLETED" == "1" ]] || return 0
    python scripts/slim_pretrain_checkpoint.py --input "$checkpoint"
}

[[ -s "$SPLIT" ]] || die "Frozen downstream split missing: $SPLIT"
actual_split_sha="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split_sha" == "$EXPECTED_SPLIT_SHA" ]] || die \
    "Split SHA mismatch: expected=$EXPECTED_SPLIT_SHA actual=$actual_split_sha"
python train_pretrain.py --help 2>&1 | grep -q -- "--reverse_loss_weight" || die \
    "train_pretrain.py does not support asymmetric direction weights"
python train_downstream.py --help 2>&1 | grep -q -- "--seal_test" || die \
    "train_downstream.py does not support sealed evaluation"
mkdir -p "$PAPER_DIR"

read -r -a seeds <<< "$PRETRAIN_SEEDS"
read -r -a alpha_values <<< "$ALPHAS"

for seed in "${seeds[@]}"; do
    echo "################################################################"
    echo "[Seed] strict independent pretraining seed=$seed"
    echo "################################################################"
    resolve_or_build_parent "$seed"
    parent_checkpoint="$RESULT_CHECKPOINT"
    parent_root="$RESULT_PARENT_ROOT"

    for alpha in "${alpha_values[@]}"; do
        run_physio_stage "$seed" "$alpha" "$parent_checkpoint"
        pretrain_checkpoint="$RESULT_CHECKPOINT"
        run_downstream "$seed" "$alpha" "$pretrain_checkpoint"
        downstream_dir="$RESULT_DOWNSTREAM_DIR"
        slim_checkpoint "$pretrain_checkpoint"
        archive_run "$seed" "$alpha" "$parent_checkpoint" \
            "$pretrain_checkpoint" "$downstream_dir"
    done

    if [[ "$PRUNE_PARENTS" == "1" && -n "$parent_root" ]]; then
        case "$parent_root" in
            ${PARENT_PREFIX}_seed*)
                parent_abs="$(realpath -m -- "$parent_root")"
                root_abs="$(realpath -m -- "$ROOT_DIR")"
                [[ "$parent_abs" == "$root_abs"/* ]] || die \
                    "Refusing to prune parent outside repository: $parent_abs"
                echo "[Prune] completed generated parent seed=$seed: $parent_abs"
                rm -rf -- "$parent_abs"
                ;;
            *) die "Refusing to prune unexpected parent path: $parent_root" ;;
        esac
    fi
done

python scripts/summarize_direction_weight_independent_seeds.py \
    --alphas "${alpha_values[@]}" \
    --pretrain_seeds "${seeds[@]}" \
    --downstream_seed "$DOWNSTREAM_SEED" \
    --downstream_prefix "$DOWNSTREAM_PREFIX" \
    --paper_dir "$PAPER_DIR"

echo "[Complete] direction-weight independent pretraining seeds | test_set_sealed=True"
