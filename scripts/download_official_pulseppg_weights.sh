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

download_part() {
  local index="$1"
  local start="$2"
  local end="$3"
  local part="$4"
  local expected="$5"
  local tmp="${part}.tmp"
  local log="${part}.log"
  local attempt=0

  if [[ -f "$part" ]] && [[ $(stat -c %s "$part") -eq $expected ]]; then
    echo "[Skip] part=$index bytes=$expected"
    return 0
  fi
  if [[ -f "$part" ]]; then
    rm -f "$part"
  fi
  touch "$tmp"

  while true; do
    local actual
    actual=$(stat -c %s "$tmp")
    if [[ "$actual" -eq "$expected" ]]; then
      mv "$tmp" "$part"
      echo "[Complete] part=$index bytes=$actual"
      return 0
    fi
    if [[ "$actual" -gt "$expected" ]]; then
      echo "[Error] oversized partial part=$index expected=$expected actual=$actual" | tee -a "$log"
      return 1
    fi

    attempt=$((attempt + 1))
    if [[ "$attempt" -gt 60 ]]; then
      echo "[Error] retry limit reached for part=$index bytes=$actual/$expected" | tee -a "$log"
      return 1
    fi
    local from=$((start + actual))
    local piece="${part}.piece.$$"
    echo "[Resume] part=$index attempt=$attempt range=$from-$end bytes=$actual/$expected" | tee -a "$log"
    set +e
    curl -L --fail --retry 5 --retry-all-errors --retry-delay 2 \
      --connect-timeout 20 --max-time 600 \
      -r "$from-$end" -o "$piece" "$URL" >> "$log" 2>&1
    local rc=$?
    set -e
    if [[ -s "$piece" ]]; then
      cat "$piece" >> "$tmp"
    fi
    rm -f "$piece"
    if [[ "$rc" -ne 0 ]]; then
      sleep 2
    fi
  done
}

for ((i=0; i<PARTS; i++)); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  if ((end >= EXPECTED_SIZE)); then end=$((EXPECTED_SIZE - 1)); fi
  part=$(printf "%s/part_%02d" "$WORK" "$i")
  expected=$((end - start + 1))
  download_part "$i" "$start" "$end" "$part" "$expected" &
  pids+=("$!:$part:$expected")
done

for item in "${pids[@]}"; do
  IFS=: read -r pid part expected <<< "$item"
  wait "$pid"
  actual=$(stat -c %s "$part")
  [[ "$actual" -eq "$expected" ]] || {
    echo "[Error] bad part size: $part expected=$expected actual=$actual"
    exit 1
  }
done

: > "${DEST}.tmp"
for ((i=0; i<PARTS; i++)); do
  part=$(printf "%s/part_%02d" "$WORK" "$i")
  cat "$part" >> "${DEST}.tmp"
done
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
