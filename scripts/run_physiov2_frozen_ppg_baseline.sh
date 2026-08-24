#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
CHECKPOINT="${PHYSIOV2_CHECKPOINT:-/root/autodl-tmp/JEPA-PREDICT/outputs_phase2_physio_v2_seed42/jepa_best.pt}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/official_fm_cache/physiov2_ppg}"
SEEDS="${SEEDS:-42 3407 2026}"

test -d "$DATA_DIR" || { echo "[Error] data missing: $DATA_DIR"; exit 1; }
test -s "$SPLIT" || { echo "[Error] split missing: $SPLIT"; exit 1; }
test -s "$CHECKPOINT" || { echo "[Error] checkpoint missing: $CHECKPOINT"; exit 1; }

echo "[Protocol] model=PhysioV2 modality=PPG frozen_encoder=true test_set=sealed"
python -u -m official_fm_baselines.extract_embeddings \
  --model physiov2_ppg \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$CACHE_DIR" \
  --batch_size "${EXTRACT_BATCH_SIZE:-512}" \
  --workers "${WORKERS:-8}"

for seed in $SEEDS; do
  out="outputs_official_fm_physiov2_ppg_seed${seed}"
  echo "[Run] PhysioV2 frozen PPG cached MIL seed=$seed"
  python -u -m official_fm_baselines.train_cached_mil \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$out" \
    --seed "$seed"
done

echo "[Complete] PhysioV2 frozen-PPG validation experiments"
