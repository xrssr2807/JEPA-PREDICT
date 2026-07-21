#!/usr/bin/env bash
set -Eeuo pipefail

# Strict PPG-only comparison across the four pre-training phases. All downstream
# settings stay fixed; only the pre-trained checkpoint changes.

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_FILE="${SPLIT_FILE:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_ppg_phase_ablation_seed42}"
SEED="${SEED:-42}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-64}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-128}"
WORKERS="${WORKERS:-8}"
OMP_THREADS="${OMP_THREADS:-4}"

PHASE0_CKPT="${PHASE0_CKPT:-outputs/jepa_best.pt}"
PHASE1_CKPT="${PHASE1_CKPT:-outputs_phase1_perf/jepa_best.pt}"
PHASE2_CKPT="${PHASE2_CKPT:-outputs_phase2_seed42_bs192/jepa_epoch_80.pt}"
PHASE3A_CKPT="${PHASE3A_CKPT:-outputs_taskaware_scratch_v2/jepa_taskaware_best.pt}"
export PHASE0_CKPT PHASE1_CKPT PHASE2_CKPT PHASE3A_CKPT

# Space-separated subset is allowed, for example: PHASES="phase2 phase3a".
PHASES="${PHASES:-phase0 phase1 phase2 phase3a}"

# Replace malformed inherited values such as an empty OMP_NUM_THREADS.
export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_THREADS"
export PYTHONHASHSEED="$SEED"

declare -A CHECKPOINTS=(
  [phase0]="$PHASE0_CKPT"
  [phase1]="$PHASE1_CKPT"
  [phase2]="$PHASE2_CKPT"
  [phase3a]="$PHASE3A_CKPT"
)

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -f train_downstream.py ]] || die "Run this script from the JEPA-PREDICT repository root."
[[ -f "$SPLIT_FILE" ]] || die "Split manifest not found: $SPLIT_FILE"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

for phase in $PHASES; do
  [[ -n "${CHECKPOINTS[$phase]+x}" ]] || die "Unknown phase '$phase'. Use phase0 phase1 phase2 phase3a."
  checkpoint="${CHECKPOINTS[$phase]}"
  [[ -s "$checkpoint" ]] || die "$phase checkpoint is missing or empty: $checkpoint"
done

# Validate the JSON schema and catch truncated modern PyTorch zip checkpoints
# before starting the multi-hour experiment.
"$PYTHON_BIN" - "$SPLIT_FILE" $PHASES <<'PY'
import json
import os
import sys
import zipfile

split_path = sys.argv[1]
phase_names = sys.argv[2:]
checkpoint_env = {
    "phase0": os.environ.get("PHASE0_CKPT", "outputs/jepa_best.pt"),
    "phase1": os.environ.get("PHASE1_CKPT", "outputs_phase1_perf/jepa_best.pt"),
    "phase2": os.environ.get("PHASE2_CKPT", "outputs_phase2_seed42_bs192/jepa_epoch_80.pt"),
    "phase3a": os.environ.get("PHASE3A_CKPT", "outputs_taskaware_scratch_v2/jepa_taskaware_best.pt"),
}

with open(split_path, "r", encoding="utf-8") as handle:
    split = json.load(handle)
missing = [name for name in ("train", "val", "test") if not isinstance(split.get(name), list)]
if missing:
    raise SystemExit(f"Split manifest lacks list fields: {missing}")
if any(len(split[name]) == 0 for name in ("train", "val", "test")):
    raise SystemExit("Split manifest contains an empty train/val/test set")

for phase in phase_names:
    path = checkpoint_env[phase]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise SystemExit(f"Checkpoint archive is empty: {path}")
    else:
        print(f"[Preflight] {phase}: legacy/non-zip checkpoint; torch.load will validate it: {path}")

print(
    "[Preflight] split files: "
    + ", ".join(f"{name}={len(split[name])}" for name in ("train", "val", "test"))
)
PY

mkdir -p "$OUTPUT_ROOT"
MANIFEST="$OUTPUT_ROOT/experiment_manifest.txt"
{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "split=$SPLIT_FILE"
  echo "split_sha256=$(sha256sum "$SPLIT_FILE" | awk '{print $1}')"
  echo "channel=ppg"
  echo "seed=$SEED"
  echo "mil_batch_size=$MIL_BATCH_SIZE"
  echo "mil_chunk_size=$MIL_CHUNK_SIZE"
  echo "workers=$WORKERS"
  echo "omp_threads=$OMP_THREADS"
  for phase in $PHASES; do
    checkpoint="${CHECKPOINTS[$phase]}"
    echo "$phase.checkpoint=$checkpoint"
    echo "$phase.sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
  done
} | tee "$MANIFEST"

for phase in $PHASES; do
  checkpoint="${CHECKPOINTS[$phase]}"
  output_dir="$OUTPUT_ROOT/$phase"
  mkdir -p "$output_dir"

  echo
  echo "============================================================"
  echo "Running $phase | checkpoint=$checkpoint | channel=PPG-only"
  echo "============================================================"

  "$PYTHON_BIN" -u train_downstream.py \
    --checkpoint "$checkpoint" \
    --dataset multidisease \
    --multidisease_channel ppg \
    --multidisease_split "$SPLIT_FILE" \
    --output_dir "$output_dir" \
    --mil_batch_size "$MIL_BATCH_SIZE" \
    --mil_chunk_size "$MIL_CHUNK_SIZE" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    2>&1 | tee "$output_dir/console.log"
done

"$PYTHON_BIN" scripts/summarize_ppg_phase_ablation.py \
  --root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.csv"

echo "[Done] Results: $OUTPUT_ROOT/summary.csv"
