#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-42}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs_phase2_no_transport_seed42}"
SP_OUTPUT="${SP_OUTPUT:-outputs_phase2_shared_private_no_transport_seed42}"
BASE_EPOCHS="${BASE_EPOCHS:-80}"
SP_EPOCHS="${SP_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-192}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
BASE_LR="${BASE_LR:-2e-4}"
SP_LR="${SP_LR:-1e-4}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

if ! python train_pretrain.py --help 2>&1 | grep -q -- "--disable_transport"; then
    die "train_pretrain.py does not support --disable_transport"
fi

run_base() {
    if [[ "$SKIP_COMPLETED" == "1" && -s "${BASE_OUTPUT}/jepa_epoch_${BASE_EPOCHS}.pt" ]]; then
        echo "[Skip] no-transport base checkpoint already exists"
        return
    fi
    mkdir -p "$BASE_OUTPUT"
    python -u train_pretrain.py \
        --phase 2 \
        --disable_transport \
        --output_dir "$BASE_OUTPUT" \
        --epochs "$BASE_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --accum_steps "$ACCUM_STEPS" \
        --lr "$BASE_LR" \
        --workers "$WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        --performance_mode \
        --seed "$SEED" \
        2>&1 | tee "${BASE_OUTPUT}/console.log"
}

run_shared_private() {
    local init_checkpoint="${BASE_OUTPUT}/jepa_epoch_${BASE_EPOCHS}.pt"
    [[ -s "$init_checkpoint" ]] || init_checkpoint="${BASE_OUTPUT}/jepa_last.pt"
    [[ -s "$init_checkpoint" ]] || die "No no-transport base checkpoint"
    if [[ "$SKIP_COMPLETED" == "1" && -s "${SP_OUTPUT}/jepa_best.pt" ]]; then
        echo "[Skip] no-transport Shared-Private checkpoint already exists"
        return
    fi
    mkdir -p "$SP_OUTPUT"
    python -u train_pretrain.py \
        --phase 2 \
        --disable_transport \
        --shared_private \
        --init_checkpoint "$init_checkpoint" \
        --output_dir "$SP_OUTPUT" \
        --epochs "$SP_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --accum_steps "$ACCUM_STEPS" \
        --lr "$SP_LR" \
        --transport_start_epoch 0 \
        --transport_ramp_epochs 1 \
        --shared_private_start_epoch 0 \
        --shared_private_ramp_epochs 5 \
        --workers "$WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        --performance_mode \
        --seed "$SEED" \
        2>&1 | tee "${SP_OUTPUT}/console.log"
}

run_base
run_shared_private

[[ -s "${SP_OUTPUT}/jepa_best.pt" ]] || die "Final no-transport checkpoint is missing"
{
    echo "experiment=P2_transport_off_pretrain"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "seed=$SEED"
    echo "transport_enabled=false"
    echo "shared_private_enabled=true"
    echo "base_epochs=$BASE_EPOCHS"
    echo "shared_private_epochs=$SP_EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "accum_steps=$ACCUM_STEPS"
    echo "base_lr=$BASE_LR"
    echo "shared_private_lr=$SP_LR"
    echo "checkpoint=${SP_OUTPUT}/jepa_best.pt"
    echo "checkpoint_sha256=$(sha256sum "${SP_OUTPUT}/jepa_best.pt" | awk '{print $1}')"
} > "${SP_OUTPUT}/transport_ablation_manifest.txt"

echo "[Complete] Phase 2 no-transport pretraining finished."
echo "[Complete] checkpoint=${SP_OUTPUT}/jepa_best.pt"
