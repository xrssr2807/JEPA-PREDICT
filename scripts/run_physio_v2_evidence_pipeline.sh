#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ALL_VARIANTS="${ALL_VARIANTS:-full no_content global_only fixed_delay hard_delay no_monotonic no_smoothness no_dustbin no_counterfactual}"
REPLICATION_SEEDS="${REPLICATION_SEEDS:-3407 2026}"
TOP_K="${TOP_K:-2}"
SPLIT="${SPLIT:-splits/multidisease_taskaware_downstream.json}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs_phase2_shared_private_seed42/jepa_best.pt}"
FULL_TEMPLATE="${FULL_TEMPLATE:-outputs_phase2_physio_v2_seed{seed}/jepa_best.pt}"
RESULT_ROOT="${RESULT_ROOT:-paper/ICASSP2027/03_experiments/P2_physio_v2_component_ablation/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_physio_v2_component_ablation}"

resolve_full() {
    local seed="$1"
    local path="${FULL_TEMPLATE//\{seed\}/$seed}"
    [[ -s "$path" ]] || {
        echo "[Error] full reference missing for seed=$seed: $path" >&2
        exit 1
    }
    printf '%s\n' "$path"
}

mkdir -p logs "$RESULT_ROOT"
echo "[Stage 1/3] seed42 complete component screen"
SEED=42 VARIANTS="$ALL_VARIANTS" SPLIT="$SPLIT" \
INIT_CHECKPOINT="$INIT_CHECKPOINT" FULL_CHECKPOINT="$(resolve_full 42)" \
OUTPUT_ROOT="$OUTPUT_ROOT" PAPER_DIR="$RESULT_ROOT/seed42" \
bash scripts/run_physio_v2_component_ablations.sh

selected="$(python scripts/summarize_physio_v2_component_ablations.py \
    --paper_dir "$RESULT_ROOT/seed42" --select_top_k "$TOP_K")"
[[ -n "$selected" ]] || {
    echo "[Error] no component selected for replication" >&2
    exit 1
}
echo "$selected" > "$RESULT_ROOT/selected_variants.txt"
echo "[Selection] largest CHD drops: $selected"

echo "[Stage 2/3] independent-seed replication"
for seed in $REPLICATION_SEEDS; do
    SEED="$seed" VARIANTS="full $selected" SPLIT="$SPLIT" \
    INIT_CHECKPOINT="$INIT_CHECKPOINT" FULL_CHECKPOINT="$(resolve_full "$seed")" \
    OUTPUT_ROOT="$OUTPUT_ROOT" PAPER_DIR="$RESULT_ROOT/seed${seed}" \
    bash scripts/run_physio_v2_component_ablations.sh
done

echo "[Stage 3/3] strict multi-seed summary"
read -r -a selected_values <<< "$selected"
read -r -a replication_values <<< "$REPLICATION_SEEDS"
seed_dirs=("$RESULT_ROOT/seed42")
for seed in "${replication_values[@]}"; do
    seed_dirs+=("$RESULT_ROOT/seed${seed}")
done
python scripts/summarize_physio_v2_component_ablations.py \
    --multiseed_dirs "${seed_dirs[@]}" \
    --variants full "${selected_values[@]}" \
    --output_dir "$RESULT_ROOT/multiseed"

{
    echo "git_sha=$(git rev-parse HEAD)"
    echo "all_variants=$ALL_VARIANTS"
    echo "selected_variants=$selected"
    echo "replication_seeds=42 $REPLICATION_SEEDS"
    echo "test_status=sealed"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RESULT_ROOT/pipeline_manifest.txt"
echo "[Complete] PhysioV2-v2 evidence pipeline | test_set_sealed=True"
