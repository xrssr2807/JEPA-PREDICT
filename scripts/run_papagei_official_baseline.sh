#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OFFICIAL_REPO="${PAPAGEI_REPO:-/root/autodl-tmp/official_models/src/papagei}"
CHECKPOINT="${PAPAGEI_CHECKPOINT:-/root/autodl-tmp/official_models/weights/papagei_s.pt}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/official_fm_cache/papagei_s}"
SEEDS="${SEEDS:-42 3407 2026}"

test -d "$DATA_DIR" || { echo "[Error] data missing: $DATA_DIR"; exit 1; }
test -s "$SPLIT" || { echo "[Error] split missing: $SPLIT"; exit 1; }
test -d "$OFFICIAL_REPO" || { echo "[Error] official repo missing: $OFFICIAL_REPO"; exit 1; }
test -s "$CHECKPOINT" || { echo "[Error] checkpoint missing: $CHECKPOINT"; exit 1; }

echo "[Protocol] official=PaPaGei-S modality=PPG target_rate=125Hz test_set=sealed"
python -u -m official_fm_baselines.extract_embeddings \
  --model papagei_s \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --official_repo "$OFFICIAL_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$CACHE_DIR" \
  --batch_size "${EXTRACT_BATCH_SIZE:-256}" \
  --workers "${WORKERS:-8}"

for seed in $SEEDS; do
  out="outputs_official_fm_papagei_s_seed${seed}"
  echo "[Run] PaPaGei-S cached MIL seed=$seed"
  python -u -m official_fm_baselines.train_cached_mil \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$out" \
    --seed "$seed"
done

echo "[Complete] PaPaGei-S official-weight validation experiments"
