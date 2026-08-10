#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Default: run only the immediate, preregistered evidence gap. Later stages
# are intentionally opt-in because their configuration depends on the result
# of the previous stage and can consume several GPU-days.
STAGES="${STAGES:-p0_independent}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/evidence_pipeline}"
mkdir -p "$MASTER_LOG_DIR"

stage() {
    echo
    echo "################################################################"
    echo "[Stage] $1"
    echo "################################################################"
}

run_stage() {
    local name="$1"
    shift
    local marker="${MASTER_LOG_DIR}/${name}.complete"
    if [[ -s "$marker" ]]; then
        echo "[Skip] completed stage=$name"
        return
    fi
    stage "$name"
    "$@" 2>&1 | tee "${MASTER_LOG_DIR}/${name}.log"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"
}

for requested in $STAGES; do
    case "$requested" in
        p0_independent)
            run_stage p0_independent \
                bash scripts/run_p0_independent_physio_mae.sh
            ;;
        direction_pilot)
            run_stage direction_pilot \
                bash scripts/run_phase2_direction_weight_ablation.sh
            ;;
        constraint_completion)
            run_stage constraint_completion \
                env MODES="full static_delay zero_delay no_monotonic" \
                PRETRAIN_SEEDS="42" \
                bash scripts/run_transport_constraint_ablation.sh
            ;;
        time_shift)
            run_stage time_shift \
                env PAPER_DIR="paper/ICASSP2027/03_experiments/P2_transport_time_shift/results/pipeline" \
                bash scripts/run_transport_time_shift.sh
            ;;
        cross_device)
            run_stage cross_device bash scripts/run_cross_device_robustness.sh
            ;;
        *)
            echo "[Error] unknown stage: $requested" >&2
            exit 2
            ;;
    esac
done

echo "[Complete] requested evidence stages: $STAGES"
