#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OFFICIAL_REPO="${NORMWEAR_REPO:-/root/autodl-tmp/official_models/src/NormWear}"
CHECKPOINT="${NORMWEAR_CHECKPOINT:-/root/autodl-tmp/official_models/weights/normwear_pretrain_ckpt.pth}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/official_fm_cache/normwear}"
SEEDS="${SEEDS:-42 3407 2026}"

test -d "$DATA_DIR" || { echo "[Error] data missing: $DATA_DIR"; exit 1; }
test -s "$SPLIT" || { echo "[Error] split missing: $SPLIT"; exit 1; }
test -d "$OFFICIAL_REPO" || { echo "[Error] official repo missing: $OFFICIAL_REPO"; exit 1; }
test -s "$CHECKPOINT" || { echo "[Error] checkpoint missing: $CHECKPOINT"; exit 1; }

echo "[Protocol] official=NormWear modality=PPG target_rate=64Hz test_set=sealed"
python -u -m official_fm_baselines.extract_embeddings \
  --model normwear \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --official_repo "$OFFICIAL_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$CACHE_DIR" \
  --batch_size "${EXTRACT_BATCH_SIZE:-8}" \
  --workers "${WORKERS:-4}"

for seed in $SEEDS; do
  out="outputs_official_fm_normwear_seed${seed}"
  echo "[Run] NormWear cached MIL seed=$seed"
  python -u -m official_fm_baselines.train_cached_mil \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$out" \
    --seed "$seed"
done

echo "[Complete] NormWear official-weight validation experiments"
