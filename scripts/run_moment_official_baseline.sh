#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/JEPA-PREDICT-official-fm}"
DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/official_fm_cache/moment_small}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_official_fm_moment_small}"
PYTHON="${PYTHON:-/root/miniconda3/envs/JEPA/bin/python}"
SEEDS="${SEEDS:-42 3407 2026}"
WORKERS="${WORKERS:-8}"

cd "$REPO_DIR"
mkdir -p "$CACHE_DIR" "$OUTPUT_ROOT" logs
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[Protocol] official=MOMENT-small modality=PPG test_set=sealed"
"$PYTHON" -u -m official_fm_baselines.extract_embeddings \
  --model moment_small \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --output_dir "$CACHE_DIR" \
  --batch_size 192 \
  --workers "$WORKERS" \
  --seed 42

for seed in $SEEDS; do
  output_dir="${OUTPUT_ROOT}_seed${seed}"
  if [[ -f "$output_dir/DEVELOPMENT_COMPLETE" ]]; then
    echo "[Skip] completed seed=$seed"
    continue
  fi
  echo "[Run] MOMENT-small cached MIL seed=$seed"
  "$PYTHON" -u -m official_fm_baselines.train_cached_mil \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$output_dir" \
    --seed "$seed" \
    --workers "$WORKERS"
done

echo "[Complete] MOMENT-small official-weight validation experiments"

