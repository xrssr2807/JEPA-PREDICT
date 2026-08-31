#!/usr/bin/env bash
set -Eeuo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROOT="${ROOT:-/root/autodl-tmp/JEPA-PREDICT}"
PYTHON="${PYTHON:-/root/miniconda3/envs/JEPA/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/vascular_ppg_1ch_full_v1_no_upstream_split}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/outputs_phase2_physio_v2_seed42/jepa_best.pt}"
SOURCE_SPLIT="${SOURCE_SPLIT:-${ROOT}/outputs_huawei_best_new_ppg_formal_seed42/comparison_split.json}"
OLD_DATA_DIR="${OLD_DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
OLD_SPLIT="${OLD_SPLIT:-${ROOT}/splits/multidisease_taskaware_downstream.json}"
SEED="${SEED:-42}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
FT_EPOCHS="${FT_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

STUDY_DIR="${ROOT}/outputs_vascular_multidisease_stability_seed${SEED}"
ARCHIVE_DIR="${ROOT}/paper/ICASSP2027/03_experiments/P4_vascular_new_downstream/results/multidisease_stability_seed${SEED}"
LOG_DIR="${ROOT}/logs"

mkdir -p "${STUDY_DIR}" "${ARCHIVE_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/vascular_multidisease_stability_seed${SEED}.log") 2>&1

fail() { echo "[Error] $*" >&2; exit 1; }
for path in "${PYTHON}" "${DATA_DIR}" "${CHECKPOINT}" "${SOURCE_SPLIT}"; do
  [[ -e "${path}" ]] || fail "Missing required input: ${path}"
done

cd "${ROOT}"
echo "[Protocol] validation-only; test remains sealed"
echo "[Labels] common six diseases + 脑卒中（中风）"
sha256sum "${CHECKPOINT}" "${SOURCE_SPLIT}" | tee "${ARCHIVE_DIR}/input_sha256.txt"

run_variant() {
  local name="$1" lr="$2" accumulation="$3" sampler="$4"
  local chd_bce="$5" chd_auc="$6" metric="$7" alpha="$8"
  local output="${STUDY_DIR}/${name}"
  if [[ -s "${output}/FORMAL_COMPLETE" ]]; then
    echo "[Skip] ${name} already complete"
    return
  fi
  echo "[Run] ${name}: lr=${lr} accum=${accumulation} sampler=${sampler} metric=${metric}"
  "${PYTHON}" scripts/run_jepa_downstream_comparison.py \
    --dataset_id "vascular7_${name}_seed${SEED}" \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${output}" \
    --source_split "${SOURCE_SPLIT}" \
    --label_schema vascular7 \
    --seed "${SEED}" \
    --probe_epochs "${PROBE_EPOCHS}" \
    --ft_epochs "${FT_EPOCHS}" \
    --mil_batch_size 32 \
    --mil_chunk_size 64 \
    --workers "${WORKERS}" \
    --downstream_lr "${lr}" \
    --grad_accum_steps "${accumulation}" \
    --sampler_mode "${sampler}" \
    --sampler_exponent 0.5 \
    --sampler_cap 4.0 \
    --chd_focus_loss_weight "${chd_bce}" \
    --chd_auc_loss_weight "${chd_auc}" \
    --best_metric "${metric}" \
    --best_metric_chd_alpha "${alpha}" \
    --seal_test
  for required in comparison_run_config.json comparison_split.json validation_patient_predictions.csv downstream_log.txt; do
    [[ -s "${output}/${required}" ]] || fail "${name} missing ${required}"
  done
  printf 'status=COMPLETE\ntest_set_sealed=true\ncompleted_at=%s\n' \
    "$(date --iso-8601=seconds)" > "${output}/FORMAL_COMPLETE"
}

# 2x2 design: CHD-specific objective/selection x stable optimization.
run_variant baseline7       5e-4   1 random               0.5 0.1 hybrid    0.7
run_variant no_chd_bias     5e-4   1 random               0.0 0.0 macro_auc 0.0
run_variant stable_training 2.5e-4 2 multilabel_balanced  0.5 0.1 hybrid    0.7
run_variant combined        2.5e-4 2 multilabel_balanced  0.0 0.0 macro_auc 0.0

"${PYTHON}" scripts/summarize_vascular_stability.py \
  --study_dir "${STUDY_DIR}" --output_dir "${ARCHIVE_DIR}"

BASELINE_PRED="${STUDY_DIR}/baseline7/validation_patient_predictions.csv"
"${PYTHON}" scripts/audit_vascular_label_sqi.py \
  --data_dir "${DATA_DIR}" \
  --split "${STUDY_DIR}/baseline7/comparison_split.json" \
  --predictions "${BASELINE_PRED}" \
  --output_dir "${ARCHIVE_DIR}/label_and_sqi_audit/new_vascular7"

if [[ -d "${OLD_DATA_DIR}" && -s "${OLD_SPLIT}" ]]; then
  "${PYTHON}" scripts/audit_vascular_label_sqi.py \
    --data_dir "${OLD_DATA_DIR}" \
    --split "${OLD_SPLIT}" \
    --output_dir "${ARCHIVE_DIR}/label_and_sqi_audit/old_original8"
else
  echo "[Audit] old cohort inputs unavailable; old/new label comparison deferred"
fi

for name in baseline7 no_chd_bias stable_training combined; do
  mkdir -p "${ARCHIVE_DIR}/${name}"
  cp -f "${STUDY_DIR}/${name}/comparison_run_config.json" \
    "${STUDY_DIR}/${name}/validation_patient_predictions.csv" \
    "${STUDY_DIR}/${name}/downstream_log.txt" \
    "${STUDY_DIR}/${name}/FORMAL_COMPLETE" \
    "${ARCHIVE_DIR}/${name}/"
  gzip -c "${STUDY_DIR}/${name}/comparison_split.json" \
    > "${ARCHIVE_DIR}/${name}/comparison_split.json.gz"
done
printf 'status=COMPLETE\ntest_set_sealed=true\ncompleted_at=%s\n' \
  "$(date --iso-8601=seconds)" > "${ARCHIVE_DIR}/study_status.txt"

echo "[Complete] ${ARCHIVE_DIR}"
sync
if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
  shutdown -h now
fi
