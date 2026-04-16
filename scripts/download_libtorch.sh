#!/usr/bin/env bash
# Download LibTorch for your platform and place it in ./libtorch/
# Run once before cmake.

set -e
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
LIBTORCH_FLAVOR="${LIBTORCH_FLAVOR:-auto}"  # auto|cpu|cu121
DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${DIR}/libtorch"

if [ -d "$TARGET" ]; then
    echo "LibTorch already present at $TARGET"
    exit 0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
    # macOS — CPU+MPS build (same package, MPS auto-detected at runtime)
    URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${TORCH_VERSION}.zip"
    if [ "$ARCH" = "x86_64" ]; then
        URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-${TORCH_VERSION}.zip"
    fi
elif [ "$OS" = "Linux" ]; then
    if [ "${LIBTORCH_FLAVOR}" = "cpu" ] || [ "${LIBTORCH_FLAVOR}" = "auto" ]; then
        URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
    else
        URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu121.zip"
    fi
else
    echo "Unsupported OS: $OS. Download LibTorch manually from https://pytorch.org/get-started/locally/"
    exit 1
fi

echo "Downloading LibTorch ${TORCH_VERSION} from:"
echo "  $URL"
cd "$DIR"
curl -L -o libtorch.zip "$URL"
unzip -q libtorch.zip
rm libtorch.zip
echo "Done. LibTorch installed at: $TARGET"
