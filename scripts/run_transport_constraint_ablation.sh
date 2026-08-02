#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODES="${MODES:-full static_delay fixed_prior zero_delay no_monotonic token_shuffled}"
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-42}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-outputs_transport_constraint_ablation}"
DOWNSTREAM_PREFIX="${DOWNSTREAM_PREFIX:-outputs_transport_constraint}"
SUMMARY_DIR="${SUMMARY_DIR:-results/transport_constraint_ablation}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_transport_constraint_ablation/results}"

BASE_EPOCHS="${BASE_EPOCHS:-80}"
SP_EPOCHS="${SP_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-192}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
BASE_LR="${BASE_LR:-2e-4}"
SP_LR="${SP_LR:-1e-4}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PRUNE_INTERMEDIATE="${PRUNE_INTERMEDIATE:-1}"
DRY_RUN="${DRY_RUN:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

is_mode() {
    case "$1" in
        full|static_delay|fixed_prior|zero_delay|no_monotonic|token_shuffled)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

next_epoch() {
    python -c '
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)) + 1)
' "$1"
}

run_logged() {
    local log_path="$1"
    shift
    if [[ "$DRY_RUN" == "1" ]]; then
        printf "[DryRun] "
        printf "%q " "$@"
        printf "\n"
        return
    fi
    "$@" 2>&1 | tee "$log_path"
}

require_interfaces() {
    python train_pretrain.py --help 2>&1 | grep -q -- "--transport_mode" \
        || die "train_pretrain.py lacks --transport_mode; update the repository"
    for option in shared_private_head patient_mil multiscale seal_test; do
        python train_downstream.py --help 2>&1 | grep -q -- "--${option}" \
            || die "train_downstream.py lacks --${option}"
    done
    [[ -s "$SPLIT" ]] || die "Downstream split is missing: $SPLIT"
}

write_pretrain_manifest() {
    local mode="$1"
    local seed="$2"
    local checkpoint="$3"
    local run_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local split_file="${run_dir}/shared_private/pretrain_split.json"
    [[ -s "$checkpoint" ]] || die "Final checkpoint is missing: $checkpoint"
    [[ -s "$split_file" ]] || die "Pretrain split is missing: $split_file"
    {
        echo "experiment=Transport_constraint_${mode}_pretrain_seed${seed}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$GIT_COMMIT"
        echo "transport_enabled=true"
        echo "transport_mode=$mode"
        echo "optimization_seed=$seed"
        echo "data_split_seed=$DATA_SPLIT_SEED"
        echo "shared_private_enabled=true"
        echo "checkpoint=$checkpoint"
        echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
        echo "pretrain_split=$split_file"
        echo "pretrain_split_sha256=$(sha256sum "$split_file" | awk '{print $1}')"
        echo "test_status=sealed"
    } > "${run_dir}/checkpoint_manifest.txt"
}

prune_pretrain() {
    local mode="$1"
    local seed="$2"
    [[ "$PRUNE_INTERMEDIATE" == "1" ]] || return
    local run_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local final_checkpoint="${run_dir}/shared_private/jepa_best.pt"
    [[ -s "$final_checkpoint" ]] \
        || die "Refusing to prune without final best checkpoint: $run_dir"
    local prune_log="${run_dir}/pruned_artifacts.txt"
    {
        echo "pruned_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "kept=$final_checkpoint"
        for path in \
            "${run_dir}/base/jepa_best.pt" \
            "${run_dir}/base/jepa_last.pt" \
            "${run_dir}/shared_private/jepa_last.pt"; do
            if [[ -e "$path" ]]; then
                stat -c "removed=%s_bytes %n" "$path"
            fi
        done
    } > "$prune_log"
    rm -f -- \
        "${run_dir}/base/jepa_best.pt" \
        "${run_dir}/base/jepa_last.pt" \
        "${run_dir}/shared_private/jepa_last.pt"
}

run_pretrain_stage() {
    local mode="$1"
    local seed="$2"
    local stage="$3"
    local run_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local output_dir="${run_dir}/${stage}"
    local epochs checkpoint
    local command=()

    if [[ "$stage" == "base" ]]; then
        epochs="$BASE_EPOCHS"
        checkpoint="${output_dir}/jepa_last.pt"
        command=(
            python -u train_pretrain.py
            --phase 2
            --transport_mode "$mode"
            --output_dir "$output_dir"
            --epochs "$epochs"
            --batch_size "$BATCH_SIZE"
            --accum_steps "$ACCUM_STEPS"
            --lr "$BASE_LR"
            --early_stop_patience 0
            --checkpoint_interval 0
            --workers "$WORKERS"
            --prefetch_factor "$PREFETCH_FACTOR"
            --performance_mode
            --seed "$seed"
            --data_split_seed "$DATA_SPLIT_SEED"
        )
    else
        epochs="$SP_EPOCHS"
        checkpoint="${output_dir}/jepa_last.pt"
        local base_checkpoint="${run_dir}/base/jepa_last.pt"
        if [[ "$DRY_RUN" != "1" && ! -s "$base_checkpoint" ]]; then
            die "Base checkpoint missing before Shared-Private stage"
        fi
        command=(
            python -u train_pretrain.py
            --phase 2
            --transport_mode "$mode"
            --shared_private
            --init_checkpoint "$base_checkpoint"
            --output_dir "$output_dir"
            --epochs "$epochs"
            --batch_size "$BATCH_SIZE"
            --accum_steps "$ACCUM_STEPS"
            --lr "$SP_LR"
            --transport_start_epoch 0
            --transport_ramp_epochs 1
            --shared_private_start_epoch 0
            --shared_private_ramp_epochs 5
            --early_stop_patience "$EARLY_STOP_PATIENCE"
            --early_stop_min_delta "$EARLY_STOP_MIN_DELTA"
            --checkpoint_interval 0
            --workers "$WORKERS"
            --prefetch_factor "$PREFETCH_FACTOR"
            --performance_mode
            --seed "$seed"
            --data_split_seed "$DATA_SPLIT_SEED"
        )
    fi

    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "${output_dir}/jepa_best.pt" ]] \
        && grep -q "Pre-training complete." "${output_dir}/console.log" 2>/dev/null; then
        echo "[Skip] pretrain mode=$mode seed=$seed stage=$stage"
        return
    fi

    if [[ "$SKIP_COMPLETED" == "1" && -s "$checkpoint" ]]; then
        local start
        start="$(next_epoch "$checkpoint")"
        if (( start > 0 && start < epochs )); then
            if [[ "$stage" == "shared_private" ]]; then
                command=(
                    python -u train_pretrain.py
                    --phase 2
                    --transport_mode "$mode"
                    --shared_private
                    --resume "$checkpoint"
                    --start_epoch "$start"
                    --output_dir "$output_dir"
                    --epochs "$epochs"
                    --batch_size "$BATCH_SIZE"
                    --accum_steps "$ACCUM_STEPS"
                    --lr "$SP_LR"
                    --transport_start_epoch 0
                    --transport_ramp_epochs 1
                    --shared_private_start_epoch 0
                    --shared_private_ramp_epochs 5
                    --early_stop_patience "$EARLY_STOP_PATIENCE"
                    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA"
                    --checkpoint_interval 0
                    --workers "$WORKERS"
                    --prefetch_factor "$PREFETCH_FACTOR"
                    --performance_mode
                    --seed "$seed"
                    --data_split_seed "$DATA_SPLIT_SEED"
                )
            else
                command+=(--resume "$checkpoint" --start_epoch "$start")
            fi
            echo "[Resume] mode=$mode seed=$seed stage=$stage epoch=$start"
        fi
    fi

    mkdir -p "$output_dir"
    printf "%q " "${command[@]}" > "${output_dir}/command.txt"
    printf "\n" >> "${output_dir}/command.txt"
    echo "[Run] pretrain mode=$mode seed=$seed stage=$stage"
    run_logged "${output_dir}/console.log" "${command[@]}"
    [[ "$DRY_RUN" == "1" ]] && return
    [[ -s "${output_dir}/jepa_last.pt" ]] \
        || die "Pretraining did not produce jepa_last.pt: $output_dir"
}

run_pretraining() {
    local mode="$1"
    local seed="$2"
    local run_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local final_checkpoint="${run_dir}/shared_private/jepa_best.pt"

    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$final_checkpoint" ]] \
        && [[ -s "${run_dir}/checkpoint_manifest.txt" ]]; then
        echo "[Skip] completed pretraining mode=$mode seed=$seed"
        return
    fi
    run_pretrain_stage "$mode" "$seed" base
    run_pretrain_stage "$mode" "$seed" shared_private
    [[ "$DRY_RUN" == "1" ]] && return
    [[ -s "$final_checkpoint" ]] \
        || die "Shared-Private best checkpoint missing: $final_checkpoint"
    write_pretrain_manifest "$mode" "$seed" "$final_checkpoint"
    prune_pretrain "$mode" "$seed"
}

archive_run() {
    local mode="$1"
    local seed="$2"
    local run_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local output_dir="${DOWNSTREAM_PREFIX}_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    local archive_dir="${PAPER_DIR}/runs/${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    mkdir -p "$archive_dir"
    cp "${run_dir}/checkpoint_manifest.txt" "$archive_dir/"
    for path in \
        "${run_dir}/base/command.txt" \
        "${run_dir}/shared_private/command.txt" \
        "${output_dir}/command.txt" \
        "${output_dir}/experiment_manifest.txt" \
        "${output_dir}/validation_patient_predictions.csv"; do
        [[ -s "$path" ]] && cp "$path" "$archive_dir/"
    done
    for stage in base shared_private; do
        [[ -s "${run_dir}/${stage}/console.log" ]] \
            && tail -n 200 "${run_dir}/${stage}/console.log" \
                > "${archive_dir}/${stage}_console_tail.txt"
    done
    [[ -s "${output_dir}/downstream_console.log" ]] \
        && tail -n 200 "${output_dir}/downstream_console.log" \
            > "${archive_dir}/downstream_console_tail.txt"
}

run_downstream() {
    local mode="$1"
    local seed="$2"
    local checkpoint="${PRETRAIN_ROOT}/${mode}_seed${seed}/shared_private/jepa_best.pt"
    local output_dir="${DOWNSTREAM_PREFIX}_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    local saved_model="${output_dir}/downstream_multidisease_best.pt"
    if [[ "$DRY_RUN" != "1" && ! -s "$checkpoint" ]]; then
        die "Checkpoint missing: $checkpoint"
    fi

    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$saved_model" ]] \
        && grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${output_dir}/downstream_console.log" 2>/dev/null; then
        echo "[Skip] downstream mode=$mode pretrain_seed=$seed"
        archive_run "$mode" "$seed"
        return
    fi

    mkdir -p "$output_dir"
    local command=(
        python -u train_downstream.py
        --checkpoint "$checkpoint"
        --dataset multidisease
        --multidisease_channel both
        --multidisease_split "$SPLIT"
        --shared_private_head off
        --patient_mil on
        --multiscale on
        --experiment_id "Transport_constraint_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
        --output_dir "$output_dir"
        --mil_batch_size "$MIL_BATCH_SIZE"
        --mil_chunk_size "$MIL_CHUNK_SIZE"
        --workers "$WORKERS"
        --seed "$DOWNSTREAM_SEED"
        --seal_test
    )
    local checkpoint_sha256="DRY_RUN"
    if [[ "$DRY_RUN" != "1" ]]; then
        checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
    fi
    {
        echo "experiment=Transport_constraint_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$GIT_COMMIT"
        echo "transport_mode=$mode"
        echo "pretrain_seed=$seed"
        echo "downstream_seed=$DOWNSTREAM_SEED"
        echo "checkpoint=$checkpoint"
        echo "checkpoint_sha256=$checkpoint_sha256"
        echo "split=$SPLIT"
        echo "split_sha256=$SPLIT_SHA256"
        echo "channel=both"
        echo "shared_private_head=off"
        echo "patient_mil=on"
        echo "multiscale=on"
        echo "test_status=sealed"
    } > "${output_dir}/experiment_manifest.txt"
    printf "%q " "${command[@]}" > "${output_dir}/command.txt"
    printf "\n" >> "${output_dir}/command.txt"

    echo "[Run] downstream mode=$mode pretrain_seed=$seed"
    run_logged "${output_dir}/downstream_console.log" "${command[@]}"
    [[ "$DRY_RUN" == "1" ]] && return
    [[ -s "$saved_model" ]] || die "Downstream checkpoint missing: $saved_model"
    grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
        "${output_dir}/downstream_console.log" \
        || die "Sealed completion marker missing: $output_dir"
    archive_run "$mode" "$seed"
}

require_interfaces
GIT_COMMIT="$(git rev-parse HEAD)"
SPLIT_SHA256="$(sha256sum "$SPLIT" | awk '{print $1}')"
mkdir -p "$PRETRAIN_ROOT" "$SUMMARY_DIR" "$PAPER_DIR/runs"

read -r -a mode_values <<< "$MODES"
read -r -a seed_values <<< "$PRETRAIN_SEEDS"
for mode in "${mode_values[@]}"; do
    is_mode "$mode" || die "Unknown Transport constraint mode: $mode"
done
for seed in "${seed_values[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid pretraining seed: $seed"
    for mode in "${mode_values[@]}"; do
        run_pretraining "$mode" "$seed"
        run_downstream "$mode" "$seed"
    done
done

if [[ "$DRY_RUN" != "1" ]]; then
    python scripts/summarize_transport_constraint_ablation.py \
        --pretrain_root "$PRETRAIN_ROOT" \
        --downstream_prefix "$DOWNSTREAM_PREFIX" \
        --summary_dir "$SUMMARY_DIR" \
        --paper_dir "$PAPER_DIR" \
        --downstream_seed "$DOWNSTREAM_SEED" \
        --seeds "${seed_values[@]}" \
        --modes "${mode_values[@]}"
fi

echo "[Complete] Transport constraint-composition ablation"
echo "[Complete] modes=$MODES pretrain_seeds=$PRETRAIN_SEEDS test=sealed"
echo "[Archive] $PAPER_DIR"
