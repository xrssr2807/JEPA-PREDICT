#!/usr/bin/env bash
set -Eeuo pipefail

# Priority 2: initialize Shared-Private JEPA from a converged Phase 2 model.
# This is a new optimization run: encoder/predictor/transport weights are
# loaded, while the optimizer, scheduler, best-loss state, and new private
# modules start fresh.

PYTHON_BIN="${PYTHON_BIN:-python}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs_phase2_seed42_bs192/jepa_epoch_80.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_phase2_shared_private_seed42}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-3}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
OMP_THREADS="${OMP_THREADS:-8}"

PRIVATE_DIM="${PRIVATE_DIM:-128}"
PRIVATE_LOSS_WEIGHT="${PRIVATE_LOSS_WEIGHT:-0.50}"
ORTHOGONALITY_WEIGHT="${ORTHOGONALITY_WEIGHT:-0.05}"
SHARED_PRIVATE_RAMP_EPOCHS="${SHARED_PRIVATE_RAMP_EPOCHS:-5}"

export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_THREADS"
export PYTHONHASHSEED="$SEED"
export PYTHONIOENCODING="utf-8"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -f train_pretrain.py ]] || die "Run this script from the JEPA-PREDICT repository root."
[[ -s "$INIT_CHECKPOINT" ]] || die "Phase 2 checkpoint is missing or empty: $INIT_CHECKPOINT"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

mkdir -p "$OUTPUT_DIR"
{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "init_checkpoint=$INIT_CHECKPOINT"
  echo "init_checkpoint_sha256=$(sha256sum "$INIT_CHECKPOINT" | awk '{print $1}')"
  echo "seed=$SEED"
  echo "epochs=$EPOCHS"
  echo "batch_size=$BATCH_SIZE"
  echo "accum_steps=$ACCUM_STEPS"
  echo "lr=$LR"
  echo "private_dim=$PRIVATE_DIM"
  echo "private_loss_weight=$PRIVATE_LOSS_WEIGHT"
  echo "orthogonality_weight=$ORTHOGONALITY_WEIGHT"
  echo "shared_private_ramp_epochs=$SHARED_PRIVATE_RAMP_EPOCHS"
} | tee "$OUTPUT_DIR/run_manifest.txt"

"$PYTHON_BIN" -u train_pretrain.py \
  --phase 2 \
  --shared_private \
  --init_checkpoint "$INIT_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --accum_steps "$ACCUM_STEPS" \
  --lr "$LR" \
  --transport_start_epoch 0 \
  --transport_ramp_epochs 1 \
  --shared_private_start_epoch 0 \
  --shared_private_ramp_epochs "$SHARED_PRIVATE_RAMP_EPOCHS" \
  --private_dim "$PRIVATE_DIM" \
  --private_loss_weight "$PRIVATE_LOSS_WEIGHT" \
  --orthogonality_weight "$ORTHOGONALITY_WEIGHT" \
  --early_stop_patience 15 \
  --early_stop_min_delta 1e-4 \
  --workers "$WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --performance_mode \
  --seed "$SEED" \
  2>&1 | tee "$OUTPUT_DIR/console.log"

echo "[Done] Best checkpoint: $OUTPUT_DIR/jepa_best.pt"

