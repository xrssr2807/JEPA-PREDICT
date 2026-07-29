#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/JEPA-PREDICT-priority1

PYTHON_BIN=/root/miniconda3/envs/JEPA/bin/python \
CHECKPOINT=outputs_phase2_shared_private_seed42/jepa_best.pt \
SPLIT=splits/multidisease_taskaware_downstream.json \
DATA_DIR=/root/ppgchd/ppgchd/data_updated \
OUTPUT_DIR=outputs_transport_interpretability_strict_seed42 \
MAX_SEGMENTS=512 \
MAX_SEGMENTS_PER_PATIENT=2 \
BATCH_SIZE=128 \
WORKERS=4 \
CPU_THREADS=16 \
SEED=42 \
bash scripts/run_transport_interpretability.sh
