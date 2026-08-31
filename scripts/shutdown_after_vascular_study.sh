#!/usr/bin/env bash
set -u

MASTER_PID_FILE="${1:-logs/vascular7_master.pid}"
STATUS_FILE="${2:-paper/ICASSP2027/03_experiments/P4_vascular_new_downstream/results/multidisease_stability_seed42/study_status.txt}"
WATCH_LOG="${3:-logs/vascular7_shutdown_watcher.log}"

MASTER_PID="$(cat "${MASTER_PID_FILE}")"
echo "[$(date --iso-8601=seconds)] watching master PID ${MASTER_PID}" >> "${WATCH_LOG}"
while kill -0 "${MASTER_PID}" 2>/dev/null; do
  sleep 60
done

if grep -q '^status=COMPLETE$' "${STATUS_FILE}" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] study complete; shutting down" >> "${WATCH_LOG}"
  sync
  shutdown -h now
else
  echo "[$(date --iso-8601=seconds)] study incomplete; server kept online" >> "${WATCH_LOG}"
  exit 1
fi
