#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
DATA_DIR="${DATA_DIR:-/root/ppgchd/ppgchd/data_updated}"
PAPER_DIR="${PAPER_DIR:-paper/ICASSP2027/03_experiments/P2_transport_time_shift/results}"
DOMAINS="${DOMAINS:-teacher_tokens waveform}"
MAX_SEGMENTS="${MAX_SEGMENTS:-512}"
MAX_SEGMENTS_PER_PATIENT="${MAX_SEGMENTS_PER_PATIENT:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-8}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
SEED="${SEED:-42}"
CPU_THREADS="${CPU_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] CPU_THREADS must be a positive integer: $CPU_THREADS" >&2
    exit 1
fi
if ! [[ "$MAX_SEGMENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] MAX_SEGMENTS must be a positive integer: $MAX_SEGMENTS" >&2
    exit 1
fi

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"

test -s "$CHECKPOINT" || {
    echo "[Error] checkpoint not found: $CHECKPOINT" >&2
    exit 1
}
test -s "$SPLIT" || {
    echo "[Error] split not found: $SPLIT" >&2
    exit 1
}
test -d "$DATA_DIR" || {
    echo "[Error] data directory not found: $DATA_DIR" >&2
    exit 1
}

mkdir -p "$PAPER_DIR"

for domain in $DOMAINS; do
    case "$domain" in
        teacher_tokens|waveform) ;;
        *)
            echo "[Error] unsupported shift domain: $domain" >&2
            exit 1
            ;;
    esac

    output_dir="${PAPER_DIR}/${domain}_seed${SEED}"
    mkdir -p "$output_dir"
    command_file="$output_dir/command.txt"

    {
        echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "checkpoint=$CHECKPOINT"
        echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
        echo "split=$SPLIT"
        echo "split_sha256=$(sha256sum "$SPLIT" | awk '{print $1}')"
        echo "domain=$domain"
        echo "seed=$SEED"
        echo "max_segments=$MAX_SEGMENTS"
        echo "max_segments_per_patient=$MAX_SEGMENTS_PER_PATIENT"
        echo "batch_size=$BATCH_SIZE"
        echo "workers=$WORKERS"
    } > "$command_file"

    echo "============================================================"
    echo "[Run] Transport time shift | domain=$domain | seed=$SEED"
    echo "[Output] $output_dir"
    echo "============================================================"

    "$PYTHON_BIN" -u analyze_transport_time_shift.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir "$DATA_DIR" \
        --split "$SPLIT" \
        --role val \
        --shift_domain "$domain" \
        --output_dir "$output_dir" \
        --max_segments "$MAX_SEGMENTS" \
        --max_segments_per_patient "$MAX_SEGMENTS_PER_PATIENT" \
        --batch_size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --bootstrap_iterations "$BOOTSTRAP_ITERATIONS" \
        --seed "$SEED" \
        2>&1 | tee "$output_dir/console.log"

    grep -q "\[Complete\] Transport time-shift intervention" \
        "$output_dir/console.log" || {
        echo "[Error] completion marker missing: $output_dir" >&2
        exit 1
    }
    test -s "$output_dir/transport_time_shift_summary.json" || {
        echo "[Error] summary missing: $output_dir" >&2
        exit 1
    }
    echo "[Done] domain=$domain"
done

echo "[Complete] all Transport time-shift experiments finished"
