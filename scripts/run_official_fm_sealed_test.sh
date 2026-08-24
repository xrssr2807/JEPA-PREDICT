#!/usr/bin/env bash
set -euo pipefail

AUTHORIZATION="${UNSEAL_TEST:-}"
[[ "$AUTHORIZATION" == "FINAL_ICASSP_2027" ]] || {
  echo "[Error] Set UNSEAL_TEST=FINAL_ICASSP_2027 for the one-time final evaluation"
  exit 1
}

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/JEPA-PREDICT-official-fm}"
DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
SPLIT="${SPLIT:-$REPO_DIR/splits/multidisease_taskaware_downstream.json}"
EXPECTED_SPLIT="e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716"
PYTHON="${PYTHON:-/root/miniconda3/envs/JEPA/bin/python}"
MOMENT_PYTHON="${MOMENT_PYTHON:-/root/miniconda3/envs/official-moment/bin/python}"
WORKERS="${WORKERS:-8}"
SEEDS=(42 3407 2026)
MODELS=(physiov2_ppg moment_small normwear papagei_s units_x128 pulse_ppg)
FINAL_ROOT="${FINAL_ROOT:-outputs_official_fm_final_test}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_official_foundation_models/results/final_test}"

cd "$REPO_DIR"
mkdir -p "$FINAL_ROOT/predictions" "$PAPER_DIR" logs
if [[ -f "$FINAL_ROOT/FINAL_TEST_COMPLETE" ]]; then
  echo "[Complete] final test already completed; refusing a second unsealing run"
  exit 0
fi

actual_split="$(sha256sum "$SPLIT" | awk '{print $1}')"
[[ "$actual_split" == "$EXPECTED_SPLIT" ]] || {
  echo "[Error] split hash mismatch: $actual_split"
  exit 1
}

declare -A CACHE=(
  [physiov2_ppg]="/root/autodl-tmp/official_fm_cache/physiov2_ppg"
  [moment_small]="/root/autodl-tmp/official_fm_cache/moment_small"
  [normwear]="/root/autodl-tmp/official_fm_cache/normwear"
  [papagei_s]="/root/autodl-tmp/official_fm_cache/papagei_s"
  [units_x128]="/root/autodl-tmp/official_fm_cache/units_x128"
  [pulse_ppg]="/root/autodl-tmp/official_fm_cache/pulse_ppg"
)
declare -A REPO=(
  [physiov2_ppg]=""
  [moment_small]=""
  [normwear]="/root/autodl-tmp/official_models/src/NormWear"
  [papagei_s]="/root/autodl-tmp/official_models/src/papagei"
  [units_x128]="/root/autodl-tmp/official_models/src/UniTS"
  [pulse_ppg]="/root/autodl-tmp/official_models/src/pulseppg"
)
declare -A CKPT=(
  [physiov2_ppg]="/root/autodl-tmp/JEPA-PREDICT/outputs_phase2_physio_v2_seed42/jepa_best.pt"
  [moment_small]=""
  [normwear]="/root/autodl-tmp/official_models/weights/normwear_pretrain_ckpt.pth"
  [papagei_s]="/root/autodl-tmp/official_models/weights/papagei_s.pt"
  [units_x128]="/root/autodl-tmp/official_models/weights/units_x128_pretrain_checkpoint.pth"
  [pulse_ppg]="/root/autodl-tmp/official_models/weights/pulseppg/checkpoint_best.pkl"
)
declare -A BATCH=(
  [physiov2_ppg]=512 [moment_small]=192 [normwear]=8
  [papagei_s]=256 [units_x128]=256 [pulse_ppg]=512
)

manifest="$FINAL_ROOT/frozen_protocol_manifest.tsv"
printf "kind\tmodel\tseed\tpath\tsha256\n" > "$manifest"
printf "split\tall\t-\t%s\t%s\n" "$SPLIT" "$actual_split" >> "$manifest"
for model in "${MODELS[@]}"; do
  if [[ -n "${CKPT[$model]}" ]]; then
    test -s "${CKPT[$model]}" || { echo "[Error] official checkpoint missing: ${CKPT[$model]}"; exit 1; }
    printf "encoder\t%s\t-\t%s\t%s\n" "$model" "${CKPT[$model]}" "$(sha256sum "${CKPT[$model]}" | awk '{print $1}')" >> "$manifest"
  fi
  for seed in "${SEEDS[@]}"; do
    run_dir="outputs_official_fm_${model}_seed${seed}"
    head="$run_dir/best_validation_model.pt"
    summary="$run_dir/summary.json"
    test -s "$head" || { echo "[Error] frozen head missing: $head"; exit 1; }
    test -s "$summary" || { echo "[Error] validation summary missing: $summary"; exit 1; }
    printf "head\t%s\t%s\t%s\t%s\n" "$model" "$seed" "$head" "$(sha256sum "$head" | awk '{print $1}')" >> "$manifest"
    printf "selection\t%s\t%s\t%s\t%s\n" "$model" "$seed" "$summary" "$(sha256sum "$summary" | awk '{print $1}')" >> "$manifest"
  done
done

extract_test() {
  local model="$1"
  local python="$PYTHON"
  [[ "$model" == "moment_small" ]] && python="$MOMENT_PYTHON"
  local command=("$python" -u -m official_fm_baselines.extract_embeddings
    --model "$model" --data_dir "$DATA_DIR" --split "$SPLIT"
    --output_dir "${CACHE[$model]}" --batch_size "${BATCH[$model]}"
    --workers "$WORKERS" --seed 42 --unseal_test
    --test_authorization "$AUTHORIZATION")
  [[ -n "${REPO[$model]}" ]] && command+=(--official_repo "${REPO[$model]}")
  [[ -n "${CKPT[$model]}" ]] && command+=(--checkpoint "${CKPT[$model]}")
  echo "[Extract test] $model"
  "${command[@]}"
}

for model in "${MODELS[@]}"; do
  extract_test "$model"
  for seed in "${SEEDS[@]}"; do
    output="$FINAL_ROOT/${model}_seed${seed}"
    if [[ -f "$output/FINAL_TEST_COMPLETE" ]]; then
      echo "[Skip] final test already present: $model seed=$seed"
      continue
    fi
    echo "[Evaluate test] $model seed=$seed"
    "$PYTHON" -u -m official_fm_baselines.evaluate_cached_test \
      --cache_dir "${CACHE[$model]}" \
      --run_dir "outputs_official_fm_${model}_seed${seed}" \
      --output_dir "$output" \
      --authorization "$AUTHORIZATION" \
      --workers "$WORKERS"
    cp "$output/test_patient_predictions.npz" \
      "$FINAL_ROOT/predictions/${model}_seed${seed}_predictions.npz"
  done
done

"$PYTHON" scripts/summarize_official_fm_final_test.py \
  --input_dir "$FINAL_ROOT" --output_dir "$PAPER_DIR"
"$PYTHON" scripts/analyze_official_fm_predictions.py \
  --prediction_dir "$FINAL_ROOT/predictions" \
  --output_dir "$PAPER_DIR" \
  --reference physiov2_ppg --bootstrap 5000 --seed 2027 \
  --cohort_label "sealed test patients" --analysis_scope final_test

cp "$manifest" "$PAPER_DIR/frozen_protocol_manifest.tsv"
printf "test_set_used=true\ncompleted_utc=%s\n" "$(date -u +%FT%TZ)" > "$FINAL_ROOT/FINAL_TEST_COMPLETE"
echo "[Complete] one-time official-FM sealed test evaluation"
