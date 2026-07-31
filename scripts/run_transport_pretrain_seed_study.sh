#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-42 3407 2026}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-outputs_transport_pretrain_seed_study}"
DOWNSTREAM_PREFIX="${DOWNSTREAM_PREFIX:-outputs_transport_pretrain_seed_study}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_transport_pretrain_seeds/results}"
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
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
REUSE_SEED42="${REUSE_SEED42:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PRUNE_INTERMEDIATE="${PRUNE_INTERMEDIATE:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

resolve_existing_checkpoint() {
    local label="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -s "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "$label checkpoint was not found"
}

for option in data_split_seed checkpoint_interval disable_transport shared_private; do
    python train_pretrain.py --help 2>&1 | grep -q -- "--${option}" \
        || die "train_pretrain.py does not support --${option}"
done
for option in seal_test experiment_id patient_mil multiscale; do
    python train_downstream.py --help 2>&1 | grep -q -- "--${option}" \
        || die "train_downstream.py does not support --${option}"
done
[[ -s "$SPLIT" ]] || die "Frozen downstream split is missing: $SPLIT"
[[ "$DATA_SPLIT_SEED" == "42" ]] \
    || die "This preregistered study requires DATA_SPLIT_SEED=42"

mkdir -p "$PRETRAIN_ROOT" "$PAPER_DIR/runs"
split_sha256="$(sha256sum "$SPLIT" | awk '{print $1}')"
git_commit="$(git rev-parse HEAD)"

seed42_checkpoint() {
    local mode="$1"
    if [[ "$mode" == "on" ]]; then
        resolve_existing_checkpoint "Transport-on seed 42" \
            "${ON_SEED42_CKPT:-}" \
            outputs_phase2_shared_private_seed42/jepa_best.pt \
            /root/autodl-tmp/JEPA-PREDICT/outputs_phase2_shared_private_seed42/jepa_best.pt
    else
        resolve_existing_checkpoint "Transport-off seed 42" \
            "${OFF_SEED42_CKPT:-}" \
            outputs_phase2_shared_private_no_transport_seed42/jepa_best.pt \
            /root/autodl-tmp/JEPA-PREDICT/outputs_phase2_shared_private_no_transport_seed42/jepa_best.pt
    fi
}

seed42_split_manifest() {
    local mode="$1"
    local checkpoint="$2"
    local candidates=()
    if [[ "$mode" == "on" ]]; then
        candidates=(
            "${ON_SEED42_SPLIT:-}"
            "$(dirname "$checkpoint")/pretrain_split.json"
            outputs_phase2_seed42_bs192/pretrain_split.json
            /root/autodl-tmp/JEPA-PREDICT/outputs_phase2_seed42_bs192/pretrain_split.json
        )
    else
        candidates=(
            "${OFF_SEED42_SPLIT:-}"
            "$(dirname "$checkpoint")/pretrain_split.json"
            outputs_phase2_no_transport_seed42/pretrain_split.json
            /root/autodl-tmp/JEPA-PREDICT/outputs_phase2_no_transport_seed42/pretrain_split.json
        )
    fi

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -s "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "Pre-training split manifest was not found for Transport-$mode seed 42"
}

write_checkpoint_manifest() {
    local mode="$1"
    local seed="$2"
    local checkpoint="$3"
    local study_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local split_source=""

    mkdir -p "$study_dir"
    if [[ "$seed" == "42" && "$REUSE_SEED42" == "1" ]]; then
        split_source="$(seed42_split_manifest "$mode" "$checkpoint")"
    elif [[ -s "$(dirname "$checkpoint")/pretrain_split.json" ]]; then
        split_source="$(dirname "$checkpoint")/pretrain_split.json"
    elif [[ -s "${study_dir}/pretrain_split.json" ]]; then
        split_source="${study_dir}/pretrain_split.json"
    fi

    if [[ -s "$split_source" ]]; then
        if [[ "$split_source" != "${study_dir}/pretrain_split.json" ]]; then
            cp "$split_source" "${study_dir}/pretrain_split.json"
        fi
    else
        die "Pre-training split manifest is missing for $checkpoint"
    fi
    {
        echo "experiment=Transport_${mode}_pretrain_seed${seed}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$git_commit"
        echo "optimization_seed=$seed"
        echo "data_split_seed=$DATA_SPLIT_SEED"
        echo "transport_enabled=$([[ "$mode" == "on" ]] && echo true || echo false)"
        echo "shared_private_enabled=true"
        echo "checkpoint=$checkpoint"
        echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
        echo "test_status=sealed"
    } > "${study_dir}/checkpoint_manifest.txt"
}

prune_pretraining_artifacts() {
    local mode="$1"
    local seed="$2"
    local study_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local prune_log="${study_dir}/pruned_artifacts.txt"
    local candidates=(
        "${study_dir}/base/jepa_best.pt"
        "${study_dir}/base/jepa_last.pt"
        "${study_dir}/shared_private/jepa_last.pt"
        "${study_dir}/base/jepa_best.pt.tmp"
        "${study_dir}/base/jepa_last.pt.tmp"
        "${study_dir}/shared_private/jepa_best.pt.tmp"
        "${study_dir}/shared_private/jepa_last.pt.tmp"
    )

    [[ "$PRUNE_INTERMEDIATE" == "1" ]] || return
    [[ "$seed" != "42" || "$REUSE_SEED42" != "1" ]] || return
    [[ -s "${study_dir}/shared_private/jepa_best.pt" ]] \
        || die "Refusing to prune before final checkpoint exists: $study_dir"

    {
        echo "pruned_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "kept=${study_dir}/shared_private/jepa_best.pt"
        local candidate
        for candidate in "${candidates[@]}"; do
            if [[ -e "$candidate" ]]; then
                stat -c 'removed=%s_bytes %n' "$candidate"
            fi
        done
    } > "$prune_log"
    rm -f -- "${candidates[@]}"
    echo "[Prune] kept final best only | transport=$mode pretrain_seed=$seed"
}

checkpoint_next_epoch() {
    local checkpoint="$1"
    python -c '
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("epoch", -1))
if epoch < 0:
    raise ValueError(f"Missing checkpoint epoch: {sys.argv[1]}")
print(epoch + 1)
' "$checkpoint" 2>/dev/null
}

run_pretraining() {
    local mode="$1"
    local seed="$2"
    local study_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"

    if [[ "$seed" == "42" && "$REUSE_SEED42" == "1" ]]; then
        local reused_checkpoint
        reused_checkpoint="$(seed42_checkpoint "$mode")"
        echo "[Reuse] mode=$mode pretrain_seed=42 checkpoint=$reused_checkpoint"
        write_checkpoint_manifest "$mode" "$seed" "$reused_checkpoint"
        return
    fi

    local base_dir="${study_dir}/base"
    local sp_dir="${study_dir}/shared_private"
    local base_checkpoint="${base_dir}/jepa_last.pt"
    local final_checkpoint="${sp_dir}/jepa_best.pt"
    local transport_args=()
    [[ "$mode" == "off" ]] && transport_args=(--disable_transport)

    if [[ "$SKIP_COMPLETED" == "1" && -s "$final_checkpoint" ]] \
        && grep -q "Pre-training complete." "${sp_dir}/console.log" 2>/dev/null; then
        echo "[Skip] completed full pre-training | transport=$mode pretrain_seed=$seed"
        cp "${sp_dir}/pretrain_split.json" "${study_dir}/pretrain_split.json"
        write_checkpoint_manifest "$mode" "$seed" "$final_checkpoint"
        prune_pretraining_artifacts "$mode" "$seed"
        return
    fi

    if [[ "$SKIP_COMPLETED" != "1" || ! -s "$base_checkpoint" ]] \
        || ! grep -q "Pre-training complete." "${base_dir}/console.log" 2>/dev/null; then
        mkdir -p "$base_dir"
        local base_start_args=()
        local base_tee_args=()
        local base_next_epoch=0
        if [[ "$SKIP_COMPLETED" == "1" && -s "$base_checkpoint" ]]; then
            base_next_epoch="$(checkpoint_next_epoch "$base_checkpoint" || echo 0)"
        fi
        if (( base_next_epoch > 0 && base_next_epoch <= BASE_EPOCHS )); then
            base_start_args=(
                --resume "$base_checkpoint"
                --start_epoch "$base_next_epoch"
            )
            base_tee_args=(-a)
            echo "[Resume] Phase 2 base | transport=$mode pretrain_seed=$seed start_epoch=$base_next_epoch"
        else
            echo "[Run] Phase 2 base | transport=$mode pretrain_seed=$seed"
        fi
        python -u train_pretrain.py \
            --phase 2 \
            "${transport_args[@]}" \
            "${base_start_args[@]}" \
            --output_dir "$base_dir" \
            --epochs "$BASE_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --lr "$BASE_LR" \
            --early_stop_patience 0 \
            --checkpoint_interval 0 \
            --workers "$WORKERS" \
            --prefetch_factor "$PREFETCH_FACTOR" \
            --performance_mode \
            --seed "$seed" \
            --data_split_seed "$DATA_SPLIT_SEED" \
            2>&1 | tee "${base_tee_args[@]}" "${base_dir}/console.log"
    else
        echo "[Skip] completed base | transport=$mode pretrain_seed=$seed"
    fi
    [[ -s "$base_checkpoint" ]] || die "Base checkpoint missing: $base_checkpoint"

    if [[ "$SKIP_COMPLETED" != "1" || ! -s "$final_checkpoint" ]] \
        || ! grep -q "Pre-training complete." "${sp_dir}/console.log" 2>/dev/null; then
        mkdir -p "$sp_dir"
        local sp_last_checkpoint="${sp_dir}/jepa_last.pt"
        local sp_start_args=(--init_checkpoint "$base_checkpoint")
        local sp_tee_args=()
        local sp_next_epoch=0
        if [[ "$SKIP_COMPLETED" == "1" && -s "$sp_last_checkpoint" ]]; then
            sp_next_epoch="$(
                checkpoint_next_epoch "$sp_last_checkpoint" || echo 0
            )"
        fi
        if (( sp_next_epoch > 0 && sp_next_epoch <= SP_EPOCHS )); then
            sp_start_args=(
                --resume "$sp_last_checkpoint"
                --start_epoch "$sp_next_epoch"
            )
            sp_tee_args=(-a)
            echo "[Resume] Shared-Private | transport=$mode pretrain_seed=$seed start_epoch=$sp_next_epoch"
        else
            echo "[Run] Shared-Private | transport=$mode pretrain_seed=$seed"
        fi
        python -u train_pretrain.py \
            --phase 2 \
            "${transport_args[@]}" \
            --shared_private \
            "${sp_start_args[@]}" \
            --output_dir "$sp_dir" \
            --epochs "$SP_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --lr "$SP_LR" \
            --transport_start_epoch 0 \
            --transport_ramp_epochs 1 \
            --shared_private_start_epoch 0 \
            --shared_private_ramp_epochs 5 \
            --early_stop_patience 15 \
            --early_stop_min_delta 1e-4 \
            --checkpoint_interval 0 \
            --workers "$WORKERS" \
            --prefetch_factor "$PREFETCH_FACTOR" \
            --performance_mode \
            --seed "$seed" \
            --data_split_seed "$DATA_SPLIT_SEED" \
            2>&1 | tee "${sp_tee_args[@]}" "${sp_dir}/console.log"
    else
        echo "[Skip] completed Shared-Private | transport=$mode pretrain_seed=$seed"
    fi
    [[ -s "$final_checkpoint" ]] \
        || die "Shared-Private checkpoint missing: $final_checkpoint"
    cp "${sp_dir}/pretrain_split.json" "${study_dir}/pretrain_split.json"
    write_checkpoint_manifest "$mode" "$seed" "$final_checkpoint"
    prune_pretraining_artifacts "$mode" "$seed"
}

checkpoint_for() {
    local mode="$1"
    local seed="$2"
    local manifest="${PRETRAIN_ROOT}/${mode}_seed${seed}/checkpoint_manifest.txt"
    [[ -s "$manifest" ]] || die "Checkpoint manifest missing: $manifest"
    local checkpoint
    checkpoint="$(grep '^checkpoint=' "$manifest" | cut -d= -f2-)"
    [[ -s "$checkpoint" ]] || die "Checkpoint missing: $checkpoint"
    printf '%s\n' "$checkpoint"
}

archive_run() {
    local mode="$1"
    local seed="$2"
    local study_dir="${PRETRAIN_ROOT}/${mode}_seed${seed}"
    local output_dir="${DOWNSTREAM_PREFIX}_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    local pretrain_archive="${PAPER_DIR}/runs/pretrain_${mode}_seed${seed}"
    local downstream_archive="${PAPER_DIR}/runs/downstream_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"

    mkdir -p "$pretrain_archive" "$downstream_archive"
    cp "${study_dir}/checkpoint_manifest.txt" "$pretrain_archive/"
    cp "${study_dir}/pretrain_split.json" "$pretrain_archive/"
    for stage_name in base shared_private; do
        if [[ -s "${study_dir}/${stage_name}/console.log" ]]; then
            tail -n 200 "${study_dir}/${stage_name}/console.log" \
                > "${pretrain_archive}/${stage_name}_console_tail.txt"
        fi
        if [[ -s "${study_dir}/${stage_name}/pretrain_log.txt" ]]; then
            cp "${study_dir}/${stage_name}/pretrain_log.txt" \
                "${pretrain_archive}/${stage_name}_pretrain_log.txt"
        fi
    done
    for filename in \
        downstream_console.log downstream_log.txt experiment_manifest.txt \
        validation_patient_predictions.csv; do
        if [[ -s "${output_dir}/${filename}" ]]; then
            cp "${output_dir}/${filename}" "$downstream_archive/"
        fi
    done
}

run_downstream() {
    local mode="$1"
    local seed="$2"
    local checkpoint
    checkpoint="$(checkpoint_for "$mode" "$seed")"
    local output_dir="${DOWNSTREAM_PREFIX}_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
    local saved_model="${output_dir}/downstream_multidisease_best.pt"
    local predictions="${output_dir}/validation_patient_predictions.csv"

    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$saved_model" && -s "$predictions" ]] \
        && grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${output_dir}/downstream_console.log" 2>/dev/null; then
        echo "[Skip] downstream | transport=$mode pretrain_seed=$seed"
        archive_run "$mode" "$seed"
        return
    fi

    mkdir -p "$output_dir"
    {
        echo "experiment=Transport_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$git_commit"
        echo "pretrain_seed=$seed"
        echo "pretrain_data_split_seed=$DATA_SPLIT_SEED"
        echo "downstream_seed=$DOWNSTREAM_SEED"
        echo "transport_enabled=$([[ "$mode" == "on" ]] && echo true || echo false)"
        echo "checkpoint=$checkpoint"
        echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
        echo "split=$SPLIT"
        echo "split_sha256=$split_sha256"
        echo "channel=both"
        echo "shared_private_head=off"
        echo "patient_mil=on"
        echo "multiscale=on"
        echo "test_status=sealed"
    } > "${output_dir}/experiment_manifest.txt"

    echo "[Run] downstream | transport=$mode pretrain_seed=$seed ft_seed=$DOWNSTREAM_SEED"
    python -u train_downstream.py \
        --checkpoint "$checkpoint" \
        --dataset multidisease \
        --multidisease_channel both \
        --multidisease_split "$SPLIT" \
        --shared_private_head off \
        --patient_mil on \
        --multiscale on \
        --experiment_id "Transport_${mode}_preseed${seed}_ftseed${DOWNSTREAM_SEED}" \
        --output_dir "$output_dir" \
        --mil_batch_size "$MIL_BATCH_SIZE" \
        --mil_chunk_size "$MIL_CHUNK_SIZE" \
        --workers "$WORKERS" \
        --seed "$DOWNSTREAM_SEED" \
        --seal_test \
        2>&1 | tee "${output_dir}/downstream_console.log"

    [[ -s "$saved_model" && -s "$predictions" ]] \
        || die "Downstream artifacts incomplete: $output_dir"
    grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
        "${output_dir}/downstream_console.log" \
        || die "Test-sealed completion marker missing: $output_dir"
    archive_run "$mode" "$seed"
}

read -r -a seed_values <<< "$PRETRAIN_SEEDS"
for seed in "${seed_values[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid pre-training seed: $seed"
    for mode in on off; do
        run_pretraining "$mode" "$seed"
        run_downstream "$mode" "$seed"
    done
done

python scripts/summarize_transport_pretrain_seed_study.py \
    --pretrain_root "$PRETRAIN_ROOT" \
    --downstream_prefix "$DOWNSTREAM_PREFIX" \
    --paper_dir "$PAPER_DIR" \
    --seeds "${seed_values[@]}" \
    --downstream_seed "$DOWNSTREAM_SEED" \
    --bootstrap_iterations "$BOOTSTRAP_ITERATIONS"

cp "$SPLIT" "${PAPER_DIR}/multidisease_taskaware_downstream.json"
{
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$git_commit"
    echo "pretrain_seeds=$PRETRAIN_SEEDS"
    echo "data_split_seed=$DATA_SPLIT_SEED"
    echo "downstream_seed=$DOWNSTREAM_SEED"
    echo "split_sha256=$split_sha256"
    echo "test_status=sealed"
} > "${PAPER_DIR}/completion_manifest.txt"

echo "[Complete] Transport independent pre-training seed study"
echo "[Complete] test_set_sealed=True"
echo "[Archive] $PAPER_DIR"
