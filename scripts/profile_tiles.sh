#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/out/profile_tiles"
mkdir -p "${OUT_DIR}"

if [ "$#" -gt 0 ]; then
  ROWS_LIST=("$@")
else
  ROWS_LIST=(16 32 64 96)
fi

TIME_CMD=""
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_CMD="/usr/bin/time -l"
  TIME_MODE="macos"
elif /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_CMD="/usr/bin/time -v"
  TIME_MODE="linux"
else
  echo "No supported /usr/bin/time flavor found."
  exit 1
fi

REPORT="${OUT_DIR}/tile_profile_report.md"
echo "| tile_rows | elapsed_s | max_rss_mb | output |" > "${REPORT}"
echo "|---:|---:|---:|---|" >> "${REPORT}"

for rows in "${ROWS_LIST[@]}"; do
  LOG="${OUT_DIR}/rows_${rows}.log"
  OUT="${OUT_DIR}/rows_${rows}.png"
  echo "Profiling tile_rows=${rows}..."
  set +e
  ${TIME_CMD} bash "${ROOT_DIR}/scripts/run_kerrtrace.sh" \
    --width 640 --height 360 \
    --spin 0.7 --observer-radius 30 --observer-inclination 70 \
    --background-mode hdri \
    --hdri-path /Users/iman.rosignoli/Documents/KerrTrace/assets/backgrounds/downloads_imported/sfondo3.jpg \
    --disk-outer-radius 12 --disk-emission-gain 30 \
    --enable-disk-segmented-palette \
    --disk-segmented-palette-mode accretion_warm \
    --disk-segmented-rings 16 --disk-segmented-sectors 64 \
    --disk-segmented-sigma 0.7 --disk-segmented-mix 0.75 \
    --max-steps 320 --step-size 0.11 \
    --device cpu --render-tile-rows "${rows}" \
    --output "${OUT}" >"${LOG}" 2>&1
  rc=$?
  set -e
  if [ ${rc} -ne 0 ]; then
    echo "| ${rows} | ERR | ERR | ${OUT} |" >> "${REPORT}"
    continue
  fi

  elapsed="$(awk '/  Time:/{print $(NF)}' "${LOG}" | tail -n1 | sed 's/s$//')"
  if [ "${TIME_MODE}" = "macos" ]; then
    maxrss_raw="$(awk '/maximum resident set size/{print $1}' "${LOG}" | tail -n1)"
  else
    maxrss_raw="$(awk -F': ' '/Maximum resident set size/{print $2}' "${LOG}" | tail -n1)"
  fi
  if [ -z "${maxrss_raw}" ]; then
    maxrss_raw=0
  fi
  maxrss_mb="$(python3 - <<PY
raw=float("${maxrss_raw}")
mode="${TIME_MODE}"
if mode == "macos":
    # /usr/bin/time -l reports bytes on macOS.
    mb = raw / (1024.0 * 1024.0)
else:
    # /usr/bin/time -v reports kB on Linux.
    mb = raw / 1024.0
print(f"{mb:.1f}")
PY
)"
  echo "| ${rows} | ${elapsed:-0} | ${maxrss_mb} | ${OUT} |" >> "${REPORT}"
done

echo "Saved report: ${REPORT}"
