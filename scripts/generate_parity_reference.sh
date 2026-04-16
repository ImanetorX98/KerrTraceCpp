#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY_ROOT_DEFAULT="${ROOT_DIR}/../KerrTrace"
PY_ROOT="${PY_ROOT:-${PY_ROOT_DEFAULT}}"
OUT_DIR="${1:-${ROOT_DIR}/out/parity_reference}"

WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-180}"
SPIN="${SPIN:-0.7}"
OBSERVER_RADIUS="${OBSERVER_RADIUS:-30}"
OBSERVER_INCLINATION="${OBSERVER_INCLINATION:-70}"
DISK_OUTER_RADIUS="${DISK_OUTER_RADIUS:-12}"
DISK_EMISSION_GAIN="${DISK_EMISSION_GAIN:-30}"
MAX_STEPS="${MAX_STEPS:-220}"
STEP_SIZE="${STEP_SIZE:-0.12}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "${OUT_DIR}"

PY_OUT="${OUT_DIR}/python_reference.png"
CPP_OUT="${OUT_DIR}/cpp_reference.png"
SIDE_BY_SIDE_OUT="${OUT_DIR}/python_vs_cpp_side_by_side.png"
METRICS_OUT="${OUT_DIR}/metrics.txt"

if [ -x "${PY_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${PY_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[1/3] Rendering Python reference frame..."
(
  cd "${PY_ROOT}"
  "${PYTHON_BIN}" -m kerrtrace \
    --frames 1 \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --coordinate-system kerr_schild \
    --metric-model kerr \
    --spin "${SPIN}" \
    --observer-radius "${OBSERVER_RADIUS}" \
    --observer-inclination-deg "${OBSERVER_INCLINATION}" \
    --disk-outer-radius "${DISK_OUTER_RADIUS}" \
    --disk-emission-gain "${DISK_EMISSION_GAIN}" \
    --enable-disk-segmented-palette \
    --disk-segmented-palette-mode accretion_warm \
    --disk-segmented-rings 8 \
    --disk-segmented-sectors 24 \
    --disk-segmented-sigma 0.6 \
    --disk-segmented-mix 0.9 \
    --background-mode darkspace \
    --max-steps "${MAX_STEPS}" \
    --step-size "${STEP_SIZE}" \
    --device "${DEVICE}" \
    --output "${PY_OUT}"
)

echo "[2/3] Rendering C++ reference frame..."
(
  cd "${ROOT_DIR}"
  ./scripts/run_kerrtrace.sh \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --coordinate-system kerr_schild \
    --metric-model kerr \
    --spin "${SPIN}" \
    --observer-radius "${OBSERVER_RADIUS}" \
    --observer-inclination "${OBSERVER_INCLINATION}" \
    --disk-outer-radius "${DISK_OUTER_RADIUS}" \
    --disk-emission-gain "${DISK_EMISSION_GAIN}" \
    --enable-disk-segmented-palette \
    --disk-segmented-palette-mode accretion_warm \
    --disk-segmented-rings 8 \
    --disk-segmented-sectors 24 \
    --disk-segmented-sigma 0.6 \
    --disk-segmented-mix 0.9 \
    --background-mode darkspace \
    --max-steps "${MAX_STEPS}" \
    --step-size "${STEP_SIZE}" \
    --device "${DEVICE}" \
    --output "${CPP_OUT}"
)

echo "[3/3] Building side-by-side and metrics..."
python3 - <<PY
from pathlib import Path
import numpy as np
from PIL import Image

py = Path("${PY_OUT}")
cpp = Path("${CPP_OUT}")
side = Path("${SIDE_BY_SIDE_OUT}")
metrics = Path("${METRICS_OUT}")

img_py = Image.open(py).convert("RGB")
img_cpp = Image.open(cpp).convert("RGB")
if img_py.size != img_cpp.size:
    img_cpp = img_cpp.resize(img_py.size)

arr_py = np.asarray(img_py, dtype=np.float32)
arr_cpp = np.asarray(img_cpp, dtype=np.float32)
diff = np.abs(arr_py - arr_cpp)

mae = float(diff.mean())
rmse = float(np.sqrt(np.mean((arr_py - arr_cpp) ** 2)))
psnr = float(20.0 * np.log10(255.0 / max(rmse, 1e-8)))

w, h = img_py.size
canvas = Image.new("RGB", (w * 2, h))
canvas.paste(img_py, (0, 0))
canvas.paste(img_cpp, (w, 0))
canvas.save(side)

metrics.write_text(
    "\\n".join(
        [
            f"width={w}",
            f"height={h}",
            f"mae={mae:.4f}",
            f"rmse={rmse:.4f}",
            f"psnr_db={psnr:.2f}",
        ]
    )
    + "\\n",
    encoding="utf-8",
)

print(f"Saved: {side}")
print(f"Saved: {metrics}")
PY

echo "Done."
echo "  Python frame: ${PY_OUT}"
echo "  C++ frame:    ${CPP_OUT}"
echo "  Compare:      ${SIDE_BY_SIDE_OUT}"
echo "  Metrics:      ${METRICS_OUT}"
