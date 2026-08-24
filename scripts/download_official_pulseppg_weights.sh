#!/usr/bin/env bash
set -euo pipefail

URL="${PULSEPPG_URL:-https://zenodo.org/api/records/17345536/files/pulseppg_model_weights.zip/content}"
DEST="${PULSEPPG_ZIP:-/root/autodl-tmp/official_models/weights/pulseppg/pulseppg_model_weights.zip}"
EXPECTED_SIZE=315300817
EXPECTED_MD5=2d0ebda9afeb9648674464098a698c37
PARTS="${PARTS:-8}"
WORK="${DEST}.parts"

mkdir -p "$(dirname "$DEST")" "$WORK"
chunk=$(( (EXPECTED_SIZE + PARTS - 1) / PARTS ))
pids=()

for ((i=0; i<PARTS; i++)); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  if ((end >= EXPECTED_SIZE)); then end=$((EXPECTED_SIZE - 1)); fi
  part=$(printf "%s/part_%02d" "$WORK" "$i")
  expected=$((end - start + 1))
  if [[ -f "$part" ]] && [[ $(stat -c %s "$part") -eq $expected ]]; then
    echo "[Skip] part=$i bytes=$expected"
    continue
  fi
  echo "[Download] part=$i range=$start-$end"
  curl -L --fail --retry 8 --retry-delay 2 \
    -r "$start-$end" -o "${part}.tmp" "$URL" \
    > "${part}.log" 2>&1 &
  pids+=("$!:$part:$expected")
done

for item in "${pids[@]}"; do
  IFS=: read -r pid part expected <<< "$item"
  wait "$pid"
  actual=$(stat -c %s "${part}.tmp")
  [[ "$actual" -eq "$expected" ]] || {
    echo "[Error] bad part size: $part expected=$expected actual=$actual"
    exit 1
  }
  mv "${part}.tmp" "$part"
done

cat "$WORK"/part_* > "${DEST}.tmp"
actual_size=$(stat -c %s "${DEST}.tmp")
actual_md5=$(md5sum "${DEST}.tmp" | awk '{print $1}')
[[ "$actual_size" -eq "$EXPECTED_SIZE" ]] || {
  echo "[Error] archive size expected=$EXPECTED_SIZE actual=$actual_size"
  exit 1
}
[[ "$actual_md5" == "$EXPECTED_MD5" ]] || {
  echo "[Error] archive md5 expected=$EXPECTED_MD5 actual=$actual_md5"
  exit 1
}
mv "${DEST}.tmp" "$DEST"
echo "[Complete] Pulse-PPG official archive: $DEST md5=$actual_md5"
