#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
TASKAWARE_SPLIT="${TASKAWARE_SPLIT:-splits/multidisease_taskaware_split.json}"
DOWNSTREAM_SPLIT="${DOWNSTREAM_SPLIT:-splits/multidisease_taskaware_downstream.json}"
REPRESENTATIVE_ONLY="${REPRESENTATIVE_ONLY:-0}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "[Error] Multidisease data directory not found: $DATA_DIR" >&2
    exit 1
fi

representative_args=()
if [[ "$REPRESENTATIVE_ONLY" == "1" ]]; then
    representative_args+=(--representative_only)
fi

python generate_multidisease_patient_split.py \
    --data_dir "$DATA_DIR" \
    --taskaware \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --output "$TASKAWARE_SPLIT" \
    --downstream_output "$DOWNSTREAM_SPLIT" \
    "${representative_args[@]}"

python scripts/audit_multidisease_split.py \
    --split "$TASKAWARE_SPLIT" \
    --data_dir "$DATA_DIR"

python scripts/audit_multidisease_split.py \
    --split "$DOWNSTREAM_SPLIT" \
    --data_dir "$DATA_DIR"

sha256sum "$TASKAWARE_SPLIT" "$DOWNSTREAM_SPLIT"
echo "[Complete] Eight-label patient-disjoint manifests are ready."
