#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p2_core}"
SEEDS="${SEEDS:-42 3407 2026}"
EXPERIMENTS="${EXPERIMENTS:-random_init phase0 phase1 phase2 mil_off multiscale_off transport_off}"
PHASE0_CKPT="${PHASE0_CKPT:-outputs/jepa_best.pt}"
PHASE1_CKPT="${PHASE1_CKPT:-outputs_phase1_perf/jepa_best.pt}"
PHASE2_CKPT="${PHASE2_CKPT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
PHASE2_NO_TRANSPORT_CKPT="${PHASE2_NO_TRANSPORT_CKPT:-outputs_phase2_shared_private_no_transport_seed42/jepa_best.pt}"
WORKERS="${WORKERS:-8}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

checkpoint_for() {
    case "$1" in
        phase0) printf '%s\n' "$PHASE0_CKPT" ;;
        phase1) printf '%s\n' "$PHASE1_CKPT" ;;
        phase2|mil_off|multiscale_off) printf '%s\n' "$PHASE2_CKPT" ;;
        transport_off) printf '%s\n' "$PHASE2_NO_TRANSPORT_CKPT" ;;
        random_init) printf '\n' ;;
        *) die "Unknown experiment: $1" ;;
    esac
}

read -r -a seed_values <<< "$SEEDS"
read -r -a experiment_values <<< "$EXPERIMENTS"
for seed in "${seed_values[@]}"; do
    for experiment in "${experiment_values[@]}"; do
        output_dir="${OUTPUT_PREFIX}_${experiment}_seed${seed}"
        downstream_checkpoint="${output_dir}/downstream_multidisease_best.pt"
        predictions="${output_dir}/validation_patient_predictions.csv"
        [[ -s "$downstream_checkpoint" ]] \
            || die "Missing downstream checkpoint: $downstream_checkpoint"
        if [[ -s "$predictions" ]]; then
            echo "[Skip] validation predictions exist: $experiment seed=$seed"
            continue
        fi

        source_checkpoint="$(checkpoint_for "$experiment")"
        initialization="pretrained"
        patient_mil="on"
        multiscale="on"
        checkpoint_args=()
        if [[ "$experiment" == "random_init" ]]; then
            initialization="random"
        else
            [[ -s "$source_checkpoint" ]] \
                || die "Missing source checkpoint: $source_checkpoint"
            checkpoint_args=(--checkpoint "$source_checkpoint")
        fi
        [[ "$experiment" == "mil_off" ]] && patient_mil="off"
        [[ "$experiment" == "multiscale_off" ]] && multiscale="off"

        echo "[Export] experiment=$experiment seed=$seed test=sealed"
        python -u train_downstream.py \
            "${checkpoint_args[@]}" \
            --evaluate_checkpoint "$downstream_checkpoint" \
            --encoder_init "$initialization" \
            --encoder_arch jepa_transformer \
            --dataset multidisease \
            --multidisease_channel both \
            --multidisease_split "$SPLIT" \
            --shared_private_head off \
            --patient_mil "$patient_mil" \
            --multiscale "$multiscale" \
            --output_dir "$output_dir" \
            --workers "$WORKERS" \
            --seed "$seed" \
            --seal_test \
            2>&1 | tee "${output_dir}/validation_export.log"
        [[ -s "$predictions" ]] \
            || die "Validation predictions were not produced: $predictions"
    done
done

echo "[Complete] P2 validation predictions exported | test_set_sealed=True"
