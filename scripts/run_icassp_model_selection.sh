#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DIRECTION_PAPER_DIR="${DIRECTION_PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_direction_weight/results/independent_pretrain_seeds}"
BASELINE_PAPER_DIR="${BASELINE_PAPER_DIR:-paper/ICASSP2027/03_experiments/P3_baselines/results/physio_mae_independent_pretrain}"
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-42 3407 2026}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

[[ -f scripts/run_direction_weight_independent_pretrain_seeds.sh ]] || die \
    "Direction-weight runner is missing"
[[ -f scripts/run_p0_independent_physio_mae.sh ]] || die \
    "PhysioV2-vs-MAE runner is missing"
grep -Fq "test_dataset_constructed=false" train_downstream.py || die \
    "Strict validation-only loader protocol is not installed"

echo "################################################################"
echo "[ICASSP Stage 1/3] select asymmetric direction weight"
echo "################################################################"
PRETRAIN_SEEDS="$PRETRAIN_SEEDS" \
DOWNSTREAM_SEED="$DOWNSTREAM_SEED" \
PAPER_DIR="$DIRECTION_PAPER_DIR" \
bash scripts/run_direction_weight_independent_pretrain_seeds.sh

DECISION_JSON="${DIRECTION_PAPER_DIR}/summary.json"
[[ -s "$DECISION_JSON" ]] || die "Direction decision is missing: $DECISION_JSON"
selected_alpha="$(python -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
print(str(result["recommended_alpha"]).replace("alpha_", ""))
' "$DECISION_JSON")"
case "$selected_alpha" in
    0.5) alpha_tag="a050" ;;
    1.0|1) alpha_tag="a100" ;;
    *) die "Unexpected selected alpha: $selected_alpha" ;;
esac
checkpoint_template="outputs_direction_weight_repro_${alpha_tag}_preseed{seed}/jepa_best.pt"

echo "[Decision] selected_alpha=$selected_alpha"
echo "[Decision] checkpoint_template=$checkpoint_template"

echo "################################################################"
echo "[ICASSP Stage 2/3] independent PhysioV2 vs Multimodal MAE"
echo "################################################################"
PRETRAIN_SEEDS="$PRETRAIN_SEEDS" \
DOWNSTREAM_SEED="$DOWNSTREAM_SEED" \
PHYSIO_CHECKPOINT_TEMPLATE="$checkpoint_template" \
REUSE_LEGACY_SEED42=0 \
PAPER_DIR="$BASELINE_PAPER_DIR" \
bash scripts/run_p0_independent_physio_mae.sh

echo "################################################################"
echo "[ICASSP Stage 3/3] protocol stop"
echo "################################################################"
cat <<EOF
[Complete] model selection and independent baseline validation are complete.
[Sealed] the internal test set remains untouched.
[Next] review validation summaries, freeze the final configuration, and only
       then run the separately authorized one-time test-set evaluation.
EOF
