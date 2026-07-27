#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_p2_core}"
SUMMARY_DIR="${SUMMARY_DIR:-results/p2_core}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_core_ablations/results}"
SEEDS="${SEEDS:-42}"
EXPERIMENTS="${EXPERIMENTS:-random_init phase0 phase1 phase2 mil_off multiscale_off}"
PHASE0_CKPT="${PHASE0_CKPT:-outputs/jepa_best.pt}"
PHASE1_CKPT="${PHASE1_CKPT:-outputs_phase1_perf/jepa_best.pt}"
PHASE2_CKPT="${PHASE2_CKPT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
PHASE2_NO_TRANSPORT_CKPT="${PHASE2_NO_TRANSPORT_CKPT:-outputs_phase2_shared_private_no_transport_seed42/jepa_best.pt}"
WORKERS="${WORKERS:-8}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-32}"
MIL_CHUNK_SIZE="${MIL_CHUNK_SIZE:-64}"
SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-256}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "[Error] $*" >&2
    exit 1
}

[[ -s "$SPLIT" ]] || die "Split manifest is missing or empty: $SPLIT"

for option in encoder_init patient_mil multiscale experiment_id seal_test; do
    if ! python train_downstream.py --help 2>&1 | grep -q -- "--${option}"; then
        die "train_downstream.py does not support --${option}"
    fi
done

split_sha256="$(sha256sum "$SPLIT" | awk '{print $1}')"
git_commit="$(git rev-parse HEAD)"
mkdir -p "$SUMMARY_DIR"
mkdir -p "$PAPER_DIR/runs"

archive_run() {
    local experiment="$1"
    local seed="$2"
    local output_dir="${OUTPUT_PREFIX}_${experiment}_seed${seed}"
    local archive_dir="${PAPER_DIR}/runs/P2_${experiment}_seed${seed}"

    mkdir -p "$archive_dir"
    for filename in experiment_manifest.txt command.txt; do
        if [[ -s "${output_dir}/${filename}" ]]; then
            cp "${output_dir}/${filename}" "${archive_dir}/${filename}"
        fi
    done
    if [[ -s "${output_dir}/downstream_console.log" ]]; then
        tail -n 160 "${output_dir}/downstream_console.log" \
            > "${archive_dir}/日志末尾.txt"
    fi
}

checkpoint_for() {
    case "$1" in
        phase0) printf '%s\n' "$PHASE0_CKPT" ;;
        phase1) printf '%s\n' "$PHASE1_CKPT" ;;
        phase2|mil_off|multiscale_off) printf '%s\n' "$PHASE2_CKPT" ;;
        transport_off) printf '%s\n' "$PHASE2_NO_TRANSPORT_CKPT" ;;
        random_init) printf '\n' ;;
        *) die "Unknown P2 experiment: $1" ;;
    esac
}

validate_experiment() {
    local experiment="$1"
    local checkpoint
    checkpoint="$(checkpoint_for "$experiment")"
    if [[ "$experiment" != "random_init" && ! -s "$checkpoint" ]]; then
        die "$experiment checkpoint is missing or empty: $checkpoint"
    fi
}

run_experiment() {
    local experiment="$1"
    local seed="$2"
    local checkpoint
    local patient_mil="on"
    local multiscale="on"
    local encoder_init="pretrained"
    local output_dir="${OUTPUT_PREFIX}_${experiment}_seed${seed}"
    local saved_model="${output_dir}/downstream_multidisease_best.pt"

    checkpoint="$(checkpoint_for "$experiment")"
    case "$experiment" in
        random_init)
            encoder_init="random"
            ;;
        mil_off)
            patient_mil="off"
            ;;
        multiscale_off)
            multiscale="off"
            ;;
    esac

    if [[ "$SKIP_COMPLETED" == "1" ]] \
        && [[ -s "$saved_model" ]] \
        && grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
            "${output_dir}/downstream_console.log" 2>/dev/null; then
        echo "[Skip] completed experiment=$experiment seed=$seed"
        archive_run "$experiment" "$seed"
        return
    fi

    mkdir -p "$output_dir"
    local command=(
        python -u train_downstream.py
        --dataset multidisease
        --multidisease_channel both
        --multidisease_split "$SPLIT"
        --shared_private_head off
        --encoder_init "$encoder_init"
        --patient_mil "$patient_mil"
        --multiscale "$multiscale"
        --experiment_id "P2_${experiment}_seed${seed}"
        --output_dir "$output_dir"
        --workers "$WORKERS"
        --seed "$seed"
        --seal_test
    )
    if [[ "$encoder_init" == "pretrained" ]]; then
        command+=(--checkpoint "$checkpoint")
    fi
    if [[ "$patient_mil" == "on" ]]; then
        command+=(
            --mil_batch_size "$MIL_BATCH_SIZE"
            --mil_chunk_size "$MIL_CHUNK_SIZE"
        )
    else
        command+=(--batch_size "$SEGMENT_BATCH_SIZE")
    fi

    {
        echo "experiment=P2_${experiment}_seed${seed}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$git_commit"
        echo "split=$SPLIT"
        echo "split_sha256=$split_sha256"
        echo "seed=$seed"
        echo "encoder_init=$encoder_init"
        echo "patient_mil=$patient_mil"
        echo "multiscale=$multiscale"
        echo "channel=both"
        echo "shared_private_head=off"
        echo "test_status=sealed"
        if [[ -n "$checkpoint" ]]; then
            echo "checkpoint=$checkpoint"
            echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
        else
            echo "checkpoint=none"
            echo "checkpoint_sha256=none"
        fi
    } > "${output_dir}/experiment_manifest.txt"
    printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
    printf '\n' >> "${output_dir}/command.txt"

    echo
    echo "============================================================"
    echo "[Run] experiment=$experiment seed=$seed"
    echo "[Run] encoder_init=$encoder_init patient_mil=$patient_mil multiscale=$multiscale"
    echo "[Run] checkpoint=${checkpoint:-none}"
    echo "[Run] split_sha256=$split_sha256"
    echo "[Run] output=$output_dir"
    echo "[Run] test=sealed"
    echo "============================================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[DryRun] '
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi

    "${command[@]}" 2>&1 | tee "${output_dir}/downstream_console.log"
    [[ -s "$saved_model" ]] || die "No downstream checkpoint: $saved_model"
    grep -q "DEVELOPMENT COMPLETE (TEST SET SEALED)" \
        "${output_dir}/downstream_console.log" \
        || die "Sealed completion marker is missing for $experiment seed=$seed"
    sha256sum "$saved_model" >> "${output_dir}/experiment_manifest.txt"
    archive_run "$experiment" "$seed"
    echo "[Done] experiment=$experiment seed=$seed"
}

read -r -a experiment_values <<< "$EXPERIMENTS"
read -r -a seed_values <<< "$SEEDS"
[[ "${#experiment_values[@]}" -gt 0 ]] || die "EXPERIMENTS is empty"
[[ "${#seed_values[@]}" -gt 0 ]] || die "SEEDS is empty"

for experiment in "${experiment_values[@]}"; do
    validate_experiment "$experiment"
done

for seed in "${seed_values[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed: $seed"
    for experiment in "${experiment_values[@]}"; do
        run_experiment "$experiment" "$seed"
    done
done

if [[ "$DRY_RUN" != "1" ]]; then
    python scripts/summarize_p2_core_ablations.py \
        --output_prefix "$OUTPUT_PREFIX" \
        --summary_dir "$SUMMARY_DIR" \
        --paper_dir "$PAPER_DIR" \
        --control phase2 \
        --seeds $SEEDS \
        --experiments $EXPERIMENTS
fi

echo "[Complete] P2 core ablation sequence finished."
echo "[Complete] experiments=$EXPERIMENTS seeds=$SEEDS test=sealed"
