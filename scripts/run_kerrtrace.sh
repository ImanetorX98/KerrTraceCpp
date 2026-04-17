#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/build_arm64/kerrtrace"
LIBTORCH_LIB="${ROOT_DIR}/libtorch/lib"

if [ ! -x "${BIN}" ]; then
  echo "Missing binary: ${BIN}"
  echo "Build first:"
  echo "  cd ${ROOT_DIR}/build_arm64 && cmake --build . --parallel 8"
  exit 1
fi

if [ ! -d "${LIBTORCH_LIB}" ]; then
  echo "Missing libtorch libs at ${LIBTORCH_LIB}"
  echo "Run:"
  echo "  bash ${ROOT_DIR}/scripts/download_libtorch.sh"
  exit 1
fi

OS="$(uname -s)"

if [ "${OS}" = "Darwin" ]; then
  # macOS: try a few common libomp locations
  OMP_CANDIDATES=(
    "${ROOT_DIR}/libtorch/lib/libomp.dylib"
    "${ROOT_DIR}/../KerrTrace/.venv/lib/python3.13/site-packages/torch/lib/libomp.dylib"
    "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    "/usr/local/opt/libomp/lib/libomp.dylib"
  )
  OMP_DIR=""
  for p in "${OMP_CANDIDATES[@]}"; do
    if [ -f "$p" ]; then
      OMP_DIR="$(dirname "$p")"
      break
    fi
  done
  if [ -n "${OMP_DIR}" ]; then
    export DYLD_LIBRARY_PATH="${LIBTORCH_LIB}:${OMP_DIR}:${DYLD_LIBRARY_PATH:-}"
  else
    export DYLD_LIBRARY_PATH="${LIBTORCH_LIB}:${DYLD_LIBRARY_PATH:-}"
  fi
else
  # Linux
  export LD_LIBRARY_PATH="${LIBTORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

exec "${BIN}" "$@"
