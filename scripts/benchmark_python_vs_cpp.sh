#!/usr/bin/env bash
set -euo pipefail

ROOT_CPP="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_PY="${ROOT_CPP}/../KerrTrace"
OUT_DIR="${ROOT_CPP}/out/benchmarks"
mkdir -p "${OUT_DIR}"

PY_BIN="${ROOT_PY}/.venv/bin/python"
if [ ! -x "${PY_BIN}" ]; then
  PY_BIN="python3"
fi

CPP_LOG="${OUT_DIR}/cpp_frame.log"
PY_LOG="${OUT_DIR}/python_frame.log"
CPP_OUT="${OUT_DIR}/cpp_frame.png"
PY_OUT="${OUT_DIR}/python_frame.png"
REPORT="${OUT_DIR}/python_vs_cpp_report.txt"

COMMON_ARGS=(
  --width 320
  --height 180
  --coordinate-system kerr_schild
  --metric-model kerr
  --spin 0.7
  --observer-radius 30
  --observer-inclination-deg 70
  --disk-outer-radius 12
  --disk-emission-gain 30
  --background-mode hdri
  --hdri-path /Users/iman.rosignoli/Documents/KerrTrace/assets/backgrounds/downloads_imported/sfondo3.jpg
  --hdri-exposure 1.2
  --hdri-rotation-deg 20
  --enable-disk-segmented-palette
  --disk-segmented-palette-mode accretion_warm
  --disk-segmented-rings 12
  --disk-segmented-sectors 36
  --disk-segmented-sigma 0.6
  --disk-segmented-mix 0.9
  --max-steps 220
  --step-size 0.12
  --device cpu
)

echo "Running Python benchmark..."
(
  cd "${ROOT_PY}"
  "${PY_BIN}" -m kerrtrace \
    --frames 1 \
    "${COMMON_ARGS[@]}" \
    --output "${PY_OUT}"
) >"${PY_LOG}" 2>&1

echo "Running C++ benchmark..."
(
  cd "${ROOT_CPP}"
  bash scripts/run_kerrtrace.sh \
    --output "${CPP_OUT}" \
    "${COMMON_ARGS[@]/--observer-inclination-deg/--observer-inclination}"
) >"${CPP_LOG}" 2>&1

py_time="$(awk '/Time:/{print $(NF)}' "${PY_LOG}" | tail -n1 | sed 's/s$//')"
cpp_time="$(awk '/  Time:/{print $(NF)}' "${CPP_LOG}" | tail -n1 | sed 's/s$//')"

python3 - <<PY > "${REPORT}"
py_t = float("${py_time:-0}")
cpp_t = float("${cpp_time:-0}")
if py_t > 0 and cpp_t > 0:
    speedup = py_t / cpp_t
else:
    speedup = 0.0
winner = "cpp" if cpp_t < py_t else "python"
print(f"python_time_s={py_t:.4f}")
print(f"cpp_time_s={cpp_t:.4f}")
print(f"speedup_py_over_cpp={speedup:.2f}x")
print(f"winner={winner}")
PY

cat "${REPORT}"
echo "Python log: ${PY_LOG}"
echo "C++ log:    ${CPP_LOG}"
echo "Python out: ${PY_OUT}"
echo "C++ out:    ${CPP_OUT}"
