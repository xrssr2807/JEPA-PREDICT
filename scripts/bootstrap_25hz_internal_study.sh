#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${SOURCE_REPO:-/root/autodl-tmp/JEPA-PREDICT}"
WORKTREE="${WORKTREE:-/root/autodl-tmp/JEPA-PREDICT-25hz}"
BRANCH="${BRANCH:-feature/25hz-prospective-long-context}"
SOURCE_SPLIT="${SOURCE_SPLIT:-$SOURCE_REPO/splits/multidisease_taskaware_downstream.json}"
SOURCE_SSH_HOST="${SOURCE_SSH_HOST:-connect.nmb1.seetacloud.com}"
SOURCE_SSH_PORT="${SOURCE_SSH_PORT:-35228}"
SOURCE_SSH_USER="${SOURCE_SSH_USER:-root}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/root/autodl-tmp/JEPA-PREDICT/outputs_phase2_physio_v2_seed42/jepa_best.pt}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-6a80dad30446d73348501eb2d9ca50b1afbfc9d55cc925ce7759da0fe5989714}"
LOG_FILE="${LOG_FILE:-logs/25hz_internal_training_seed42.log}"
PID_FILE="${PID_FILE:-logs/25hz_internal_training_seed42.pid}"

if [[ ! -d "$SOURCE_REPO/.git" ]]; then
    echo "[Error] source repository not found: $SOURCE_REPO" >&2
    exit 1
fi
if [[ ! -s "$SOURCE_SPLIT" ]]; then
    echo "[Error] patient split not found: $SOURCE_SPLIT" >&2
    exit 1
fi
if ! git -C "$SOURCE_REPO" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    echo "[Error] origin/$BRANCH is unavailable. Fetch the branch first." >&2
    exit 1
fi

git -C "$SOURCE_REPO" worktree prune
if [[ ! -e "$WORKTREE/.git" ]]; then
    echo "[Setup] creating isolated worktree: $WORKTREE"
    git -C "$SOURCE_REPO" worktree add --detach "$WORKTREE" "origin/$BRANCH"
else
    if [[ -n "$(git -C "$WORKTREE" status --porcelain --untracked-files=no)" ]]; then
        echo "[Error] tracked files are modified in existing worktree: $WORKTREE" >&2
        echo "Commit or preserve those changes before updating the worktree." >&2
        exit 1
    fi
    echo "[Setup] updating existing worktree to origin/$BRANCH"
    git -C "$WORKTREE" switch --detach "origin/$BRANCH"
fi

mkdir -p \
    "$WORKTREE/splits" \
    "$WORKTREE/outputs_phase2_physio_v2_seed42" \
    "$WORKTREE/logs"
target_split="$WORKTREE/splits/multidisease_taskaware_downstream.json"
if [[ -e "$target_split" ]] \
    && [[ "$(readlink -f "$SOURCE_SPLIT")" == "$(readlink -f "$target_split")" ]]; then
    echo "[Setup] reusing existing patient split link: $target_split"
else
    cp -f "$SOURCE_SPLIT" "$target_split"
fi

checkpoint="$WORKTREE/outputs_phase2_physio_v2_seed42/jepa_best.pt"
if [[ ! -s "$checkpoint" ]]; then
    echo "[Transfer] copying PhysioV2 checkpoint from ${SOURCE_SSH_HOST}:${SOURCE_SSH_PORT}"
    scp -P "$SOURCE_SSH_PORT" \
        "${SOURCE_SSH_USER}@${SOURCE_SSH_HOST}:${SOURCE_CHECKPOINT}" \
        "$checkpoint.part"
    mv -f "$checkpoint.part" "$checkpoint"
fi

actual_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
    echo "[Error] checkpoint SHA256 mismatch" >&2
    echo "expected=$EXPECTED_CHECKPOINT_SHA256" >&2
    echo "actual=$actual_sha256" >&2
    exit 1
fi

if [[ ! -d /root/ppgchd/ppgchd/data_updated ]]; then
    echo "[Error] downstream dataset missing: /root/ppgchd/ppgchd/data_updated" >&2
    exit 1
fi

cd "$WORKTREE"
if [[ -s "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "[Running] study is already active: PID=$old_pid"
        echo "tail -f $WORKTREE/$LOG_FILE"
        exit 0
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[Launch] starting four-condition internal study"
nohup bash scripts/run_25hz_internal_study.sh > "$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "[Started] PID=$pid"
echo "[Log] $WORKTREE/$LOG_FILE"
echo "[Results] $WORKTREE/outputs_25hz_prospective_study_seed42"
echo "Run: tail -f $WORKTREE/$LOG_FILE"
