#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
SEEDS="${SEEDS:-42 3407 2026}"
WORKERS="${WORKERS:-8}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/icassp_remaining}"
TIME_SHIFT_ROOT="${TIME_SHIFT_ROOT:-paper/ICASSP2027/03_experiments/P2_transport_time_shift/results}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$MASTER_LOG_DIR"

die() {
    echo "[Error] $*" >&2
    exit 1
}

stage() {
    echo
    echo "################################################################"
    echo "[Stage] $1"
    echo "################################################################"
}

resolve_checkpoint() {
    local label="$1"
    local explicit="$2"
    shift 2
    local candidate

    if [[ -n "$explicit" ]]; then
        [[ -s "$explicit" ]] \
            || die "$label checkpoint is missing or empty: $explicit"
        printf '%s\n' "$explicit"
        return
    fi

    for candidate in "$@"; do
        if [[ -s "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "$label checkpoint was not found in any known project directory"
}

PHASE0_CKPT_RESOLVED="$(resolve_checkpoint \
    "Phase 0" "${PHASE0_CKPT:-}" \
    outputs/jepa_best.pt \
    /root/autodl-tmp/JEPA-PREDICT/outputs/jepa_best.pt)"
PHASE1_CKPT_RESOLVED="$(resolve_checkpoint \
    "Phase 1" "${PHASE1_CKPT:-}" \
    outputs_phase1_perf/jepa_best.pt \
    /root/autodl-tmp/JEPA-PREDICT/outputs_phase1_perf/jepa_best.pt)"
PHASE2_CKPT_RESOLVED="$(resolve_checkpoint \
    "Phase 2" "${PHASE2_CKPT:-}" \
    outputs_phase2_shared_private_seed42/jepa_best.pt \
    /root/autodl-tmp/JEPA-PREDICT/outputs_phase2_shared_private_seed42/jepa_best.pt)"
PHASE2_NO_TRANSPORT_CKPT_RESOLVED="$(resolve_checkpoint \
    "Phase 2 no-Transport" "${PHASE2_NO_TRANSPORT_CKPT:-}" \
    outputs_phase2_shared_private_no_transport_seed42/jepa_best.pt \
    /root/autodl-tmp/JEPA-PREDICT/outputs_phase2_shared_private_no_transport_seed42/jepa_best.pt)"

[[ -s "$SPLIT" ]] || die "Frozen downstream split is missing: $SPLIT"
echo "[Checkpoint] Phase 0: $PHASE0_CKPT_RESOLVED"
echo "[Checkpoint] Phase 1: $PHASE1_CKPT_RESOLVED"
echo "[Checkpoint] Phase 2: $PHASE2_CKPT_RESOLVED"
echo "[Checkpoint] Phase 2 no-Transport: $PHASE2_NO_TRANSPORT_CKPT_RESOLVED"

stage "0/7 preflight and leakage audit"
python scripts/audit_multidisease_split.py \
    --split "$SPLIT" \
    --data_dir /root/ppgchd/ppgchd/data_updated \
    2>&1 | tee "${MASTER_LOG_DIR}/stage0_audit.log"
grep -q '"status": "PASS"' "${MASTER_LOG_DIR}/stage0_audit.log" \
    || die "Split audit did not pass"

stage "1/7 Transport time-shift smoke"
smoke_dir="${TIME_SHIFT_ROOT}/smoke"
smoke_summary="${smoke_dir}/teacher_tokens_seed42/transport_time_shift_summary.json"
if [[ "$SKIP_COMPLETED" == "1" && -s "$smoke_summary" ]]; then
    echo "[Skip] completed Transport time-shift smoke"
else
    mkdir -p "$smoke_dir"
    env \
        CHECKPOINT="$PHASE2_CKPT_RESOLVED" \
        SPLIT="$SPLIT" \
        DOMAINS=teacher_tokens \
        MAX_SEGMENTS=128 \
        MAX_SEGMENTS_PER_PATIENT=2 \
        BATCH_SIZE=16 \
        WORKERS="$WORKERS" \
        BOOTSTRAP_ITERATIONS=200 \
        SEED=42 \
        PAPER_DIR="$smoke_dir" \
        bash scripts/run_transport_time_shift.sh \
        2>&1 | tee "${smoke_dir}/run.log"
fi
[[ -s "$smoke_summary" ]] || die "Time-shift smoke did not complete"

stage "2/7 Transport time-shift formal validation experiment"
formal_dir="${TIME_SHIFT_ROOT}/formal"
formal_teacher="${formal_dir}/teacher_tokens_seed42/transport_time_shift_summary.json"
formal_waveform="${formal_dir}/waveform_seed42/transport_time_shift_summary.json"
if [[ "$SKIP_COMPLETED" == "1" ]] \
    && [[ -s "$formal_teacher" ]] \
    && [[ -s "$formal_waveform" ]]; then
    echo "[Skip] completed formal Transport time-shift experiment"
else
    mkdir -p "$formal_dir"
    env \
        CHECKPOINT="$PHASE2_CKPT_RESOLVED" \
        SPLIT="$SPLIT" \
        DOMAINS="teacher_tokens waveform" \
        MAX_SEGMENTS=512 \
        MAX_SEGMENTS_PER_PATIENT=2 \
        BATCH_SIZE=16 \
        WORKERS="$WORKERS" \
        BOOTSTRAP_ITERATIONS=2000 \
        SEED=42 \
        PAPER_DIR="$formal_dir" \
        bash scripts/run_transport_time_shift.sh \
        2>&1 | tee "${formal_dir}/run.log"
fi
[[ -s "$formal_teacher" && -s "$formal_waveform" ]] \
    || die "Formal Transport time-shift outputs are incomplete"

stage "3/7 complete three-seed pretraining-value and core ablations"
env \
    SPLIT="$SPLIT" \
    SEEDS="$SEEDS" \
    EXPERIMENTS="random_init phase0 phase1 phase2 mil_off multiscale_off transport_off" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    WORKERS="$WORKERS" \
    PHASE0_CKPT="$PHASE0_CKPT_RESOLVED" \
    PHASE1_CKPT="$PHASE1_CKPT_RESOLVED" \
    PHASE2_CKPT="$PHASE2_CKPT_RESOLVED" \
    PHASE2_NO_TRANSPORT_CKPT="$PHASE2_NO_TRANSPORT_CKPT_RESOLVED" \
    bash scripts/run_p2_core_ablations.sh \
    2>&1 | tee "${MASTER_LOG_DIR}/stage3_p2.log"

stage "4/7 export validation patient predictions for paired inference"
env \
    SPLIT="$SPLIT" \
    SEEDS="$SEEDS" \
    EXPERIMENTS="random_init phase0 phase1 phase2 mil_off multiscale_off transport_off" \
    WORKERS="$WORKERS" \
    PHASE0_CKPT="$PHASE0_CKPT_RESOLVED" \
    PHASE1_CKPT="$PHASE1_CKPT_RESOLVED" \
    PHASE2_CKPT="$PHASE2_CKPT_RESOLVED" \
    PHASE2_NO_TRANSPORT_CKPT="$PHASE2_NO_TRANSPORT_CKPT_RESOLVED" \
    bash scripts/export_p2_validation_predictions.sh \
    2>&1 | tee "${MASTER_LOG_DIR}/stage4_predictions.log"

stage "5/7 supervised and contrastive baselines"
env \
    SPLIT="$SPLIT" \
    SEEDS="$SEEDS" \
    WORKERS="$WORKERS" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    bash scripts/run_p3_baselines.sh \
    2>&1 | tee "${MASTER_LOG_DIR}/stage5_baselines.log"

stage "6/7 patient-level bootstrap confidence intervals"
python scripts/bootstrap_patient_auc.py \
    --p2_prefix outputs_p2_core \
    --p3_prefix outputs_p3 \
    --reference phase2 \
    --iterations 2000 \
    --output_dir paper/ICASSP2027/04_statistics/results \
    2>&1 | tee "${MASTER_LOG_DIR}/stage6_bootstrap.log"

stage "7/7 archive integrity report"
{
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "split=$SPLIT"
    echo "split_sha256=$(sha256sum "$SPLIT" | awk '{print $1}')"
    echo "seeds=$SEEDS"
    echo "test_status=sealed"
    find "$TIME_SHIFT_ROOT/formal" -name transport_time_shift_summary.json -type f
    find paper/ICASSP2027/03_experiments/P2_core_ablations/results \
        -maxdepth 1 -type f
    find paper/ICASSP2027/03_experiments/P3_baselines/results \
        -maxdepth 1 -type f
} > "${MASTER_LOG_DIR}/completion_manifest.txt"

echo
echo "[Complete] ICASSP remaining experiment pipeline finished"
echo "[Complete] test_set_sealed=True"
echo "[Archive] ${MASTER_LOG_DIR}/completion_manifest.txt"
