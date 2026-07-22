#!/usr/bin/env bash
set -Eeuo pipefail

# Run the strict 2 x 3 downstream comparison:
#   Phase 2 / Phase 3A x PPG-only / ECG-only / ECG+PPG.
# All experiments use the same patient split, seed, and downstream code.

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_FILE="${SPLIT_FILE:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_phase2_phase3a_channel_seed42}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
OMP_THREADS="${OMP_THREADS:-8}"

SINGLE_BATCH_SIZE="${SINGLE_BATCH_SIZE:-64}"
SINGLE_CHUNK_SIZE="${SINGLE_CHUNK_SIZE:-128}"
DUAL_BATCH_SIZE="${DUAL_BATCH_SIZE:-32}"
DUAL_CHUNK_SIZE="${DUAL_CHUNK_SIZE:-64}"

PHASE2_CKPT="${PHASE2_CKPT:-outputs_phase2_seed42_bs192/jepa_epoch_80.pt}"
PHASE3A_CKPT="${PHASE3A_CKPT:-outputs_taskaware_scratch_v2/jepa_taskaware_best.pt}"
export PHASE2_CKPT PHASE3A_CKPT

# Space-separated subsets are supported, for example:
#   PHASES="phase3a" CHANNELS="ppg both" bash scripts/run_phase2_phase3a_channel_ablation.sh
PHASES="${PHASES:-phase2 phase3a}"
CHANNELS="${CHANNELS:-ppg ecg both}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_THREADS"
export PYTHONHASHSEED="$SEED"
export PYTHONIOENCODING="utf-8"

declare -A CHECKPOINTS=(
  [phase2]="$PHASE2_CKPT"
  [phase3a]="$PHASE3A_CKPT"
)

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_complete() {
  local output_dir="$1"
  [[ -s "$output_dir/downstream_multidisease_best.pt" ]] &&
    grep -q "FINAL EVALUATION" "$output_dir/console.log" 2>/dev/null
}

[[ -f train_downstream.py ]] || die "Run this script from the JEPA-PREDICT repository root."
[[ -f scripts/summarize_channel_ablation.py ]] || die "Missing scripts/summarize_channel_ablation.py"
[[ -s "$SPLIT_FILE" ]] || die "Split manifest not found or empty: $SPLIT_FILE"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

for phase in $PHASES; do
  [[ -n "${CHECKPOINTS[$phase]+x}" ]] || die "Unknown phase '$phase'. Use phase2 or phase3a."
  [[ -s "${CHECKPOINTS[$phase]}" ]] || die "$phase checkpoint is missing or empty: ${CHECKPOINTS[$phase]}"
done

for channel in $CHANNELS; do
  case "$channel" in
    ppg|ecg|both) ;;
    *) die "Unknown channel '$channel'. Use ppg, ecg, or both." ;;
  esac
done

# Check split integrity and catch truncated modern PyTorch zip checkpoints
# before committing GPU time to the six experiments.
"$PYTHON_BIN" - "$SPLIT_FILE" $PHASES <<'PY'
import json
import os
import sys
import zipfile

split_path = sys.argv[1]
phases = sys.argv[2:]
checkpoints = {
    "phase2": os.environ["PHASE2_CKPT"],
    "phase3a": os.environ["PHASE3A_CKPT"],
}

with open(split_path, "r", encoding="utf-8") as handle:
    split = json.load(handle)

required = ("train", "val", "test")
missing = [name for name in required if not isinstance(split.get(name), list)]
if missing:
    raise SystemExit(f"Split manifest lacks list fields: {missing}")
if any(not split[name] for name in required):
    raise SystemExit("Split manifest contains an empty train/val/test set")

def uid(filename: str) -> str:
    parts = filename.split("_")
    return parts[1] if parts[0] in {"train", "val", "test"} and len(parts) >= 3 else parts[0]

uid_sets = {name: {uid(item) for item in split[name]} for name in required}
overlaps = {
    "train-val": uid_sets["train"] & uid_sets["val"],
    "train-test": uid_sets["train"] & uid_sets["test"],
    "val-test": uid_sets["val"] & uid_sets["test"],
}
leaks = {name: len(values) for name, values in overlaps.items() if values}
if leaks:
    raise SystemExit(f"Patient UID leakage detected: {leaks}")

for phase in phases:
    path = checkpoints[phase]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise SystemExit(f"Checkpoint archive is empty: {path}")
    else:
        print(f"[Preflight] {phase}: legacy/non-zip checkpoint; torch.load will validate it: {path}")

print(
    "[Preflight] patients: "
    + ", ".join(f"{name}={len(uid_sets[name])}" for name in required)
)
print("[Preflight] patient UID overlap: 0")
PY

mkdir -p "$OUTPUT_ROOT"
MANIFEST="$OUTPUT_ROOT/experiment_manifest.txt"
{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "split=$SPLIT_FILE"
  echo "split_sha256=$(sha256sum "$SPLIT_FILE" | awk '{print $1}')"
  echo "phases=$PHASES"
  echo "channels=$CHANNELS"
  echo "seed=$SEED"
  echo "single_batch_size=$SINGLE_BATCH_SIZE"
  echo "single_chunk_size=$SINGLE_CHUNK_SIZE"
  echo "dual_batch_size=$DUAL_BATCH_SIZE"
  echo "dual_chunk_size=$DUAL_CHUNK_SIZE"
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

  for channel in $CHANNELS; do
    output_dir="$OUTPUT_ROOT/${phase}_${channel}"
    mkdir -p "$output_dir"

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$output_dir"; then
      echo "[Skip] completed experiment: ${phase}_${channel}"
      continue
    fi

    if [[ "$channel" == "both" ]]; then
      batch_size="$DUAL_BATCH_SIZE"
      chunk_size="$DUAL_CHUNK_SIZE"
    else
      batch_size="$SINGLE_BATCH_SIZE"
      chunk_size="$SINGLE_CHUNK_SIZE"
    fi

    {
      echo "phase=$phase"
      echo "channel=$channel"
      echo "checkpoint=$checkpoint"
      echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
      echo "split=$SPLIT_FILE"
      echo "seed=$SEED"
      echo "mil_batch_size=$batch_size"
      echo "mil_chunk_size=$chunk_size"
    } > "$output_dir/run_config.txt"

    echo
    echo "============================================================"
    echo "Running ${phase}_${channel}"
    echo "checkpoint=$checkpoint"
    echo "batch=$batch_size chunk=$chunk_size seed=$SEED"
    echo "============================================================"

    "$PYTHON_BIN" -u train_downstream.py \
      --checkpoint "$checkpoint" \
      --dataset multidisease \
      --multidisease_channel "$channel" \
      --multidisease_split "$SPLIT_FILE" \
      --output_dir "$output_dir" \
      --mil_batch_size "$batch_size" \
      --mil_chunk_size "$chunk_size" \
      --workers "$WORKERS" \
      --seed "$SEED" \
      2>&1 | tee "$output_dir/console.log"
  done
done

"$PYTHON_BIN" scripts/summarize_channel_ablation.py \
  --root "$OUTPUT_ROOT" \
  --phases $PHASES \
  --channels $CHANNELS \
  --output "$OUTPUT_ROOT/summary.csv"

echo "[Done] Comparison table: $OUTPUT_ROOT/summary.csv"

