#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/JEPA-PREDICT-priority1}"
SEED="${SEED:-42}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_phase2_physio_v2_seed${SEED}}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_physio_transport_v2/results/seed${SEED}}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-3}"
LR="${LR:-5e-5}"
WORKERS="${WORKERS:-8}"

cd "$ROOT"
test -s "$INIT_CHECKPOINT" || {
    echo "[Error] initialization checkpoint missing: $INIT_CHECKPOINT" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR" "$PAPER_DIR"
{
    echo "git_sha=$(git rev-parse HEAD)"
    echo "seed=$SEED"
    echo "init_checkpoint=$INIT_CHECKPOINT"
    echo "init_sha256=$(sha256sum "$INIT_CHECKPOINT" | awk '{print $1}')"
    echo "transport_mode=physio_v2"
    echo "epochs=$EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "accum_steps=$ACCUM_STEPS"
    echo "lr=$LR"
} > "$PAPER_DIR/run_manifest.txt"

python -u train_pretrain.py \
    --phase 2 \
    --transport_mode physio_v2 \
    --shared_private \
    --init_checkpoint "$INIT_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --accum_steps "$ACCUM_STEPS" \
    --lr "$LR" \
    --transport_start_epoch 0 \
    --transport_ramp_epochs 10 \
    --counterfactual_weight 0.10 \
    --counterfactual_margin 0.10 \
    --sinkhorn_iters 20 \
    --workers "$WORKERS" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/console.log"

cp "$OUTPUT_DIR/console.log" "$PAPER_DIR/console.log"
find "$OUTPUT_DIR" -maxdepth 1 -type f \
    \( -name 'pretrain_log.txt' -o -name 'jepa_best.pt' -o -name 'jepa_last.pt' \) \
    -printf '%f\t%s bytes\n' | sort > "$PAPER_DIR/output_manifest.txt"
echo "[Complete] Physio Transport v2 seed=$SEED"
