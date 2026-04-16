#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/build_arm64/kerrtrace"
OUT_DIR="${ROOT_DIR}/tests/fixtures/reference"

if [ ! -x "${BIN}" ]; then
  echo "Missing binary: ${BIN}"
  echo "Build first in ${ROOT_DIR}/build_arm64"
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[1/2] Rendering baseline fixture..."
"${ROOT_DIR}/scripts/run_kerrtrace.sh" \
  --config "${ROOT_DIR}/tests/fixtures/config_regression_baseline.json" \
  --output "${OUT_DIR}/baseline_160x90.png"

echo "[2/2] Rendering advanced fixture..."
"${ROOT_DIR}/scripts/run_kerrtrace.sh" \
  --config "${ROOT_DIR}/tests/fixtures/config_regression_advanced.json" \
  --output "${OUT_DIR}/advanced_160x90.png"

echo "Saved fixtures in ${OUT_DIR}"
