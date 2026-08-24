#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OFFICIAL_REPO="${PULSEPPG_REPO:-/root/autodl-tmp/official_models/src/pulseppg}"
CHECKPOINT="${PULSEPPG_CHECKPOINT:-/root/autodl-tmp/official_models/weights/pulseppg/checkpoint_best.pkl}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/official_fm_cache/pulse_ppg}"
SEEDS="${SEEDS:-42 3407 2026}"

test -d "$DATA_DIR" || { echo "[Error] data missing: $DATA_DIR"; exit 1; }
test -s "$SPLIT" || { echo "[Error] split missing: $SPLIT"; exit 1; }
test -d "$OFFICIAL_REPO" || { echo "[Error] official repo missing: $OFFICIAL_REPO"; exit 1; }
test -s "$CHECKPOINT" || { echo "[Error] checkpoint missing: $CHECKPOINT"; exit 1; }

echo "[Protocol] official=Pulse-PPG modality=PPG target_rate=50Hz test_set=sealed"
python -u -m official_fm_baselines.extract_embeddings \
  --model pulse_ppg \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --official_repo "$OFFICIAL_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$CACHE_DIR" \
  --batch_size "${EXTRACT_BATCH_SIZE:-512}" \
  --workers "${WORKERS:-8}"

for seed in $SEEDS; do
  out="outputs_official_fm_pulse_ppg_seed${seed}"
  echo "[Run] Pulse-PPG cached MIL seed=$seed"
  python -u -m official_fm_baselines.train_cached_mil \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$out" \
    --seed "$seed"
done

echo "[Complete] Pulse-PPG official-weight validation experiments"
