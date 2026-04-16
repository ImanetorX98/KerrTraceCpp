# KerrTrace C++ — Build Instructions

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| CMake | ≥ 3.18 | |
| C++ compiler | C++17 | clang++ on Mac, MSVC/GCC on Win/Linux |
| LibTorch | ≥ 2.0 | see below |
| ffmpeg | any | for video encoding (optional) |

## 1 — Download LibTorch

```bash
bash scripts/download_libtorch.sh
```

Or download manually from https://pytorch.org/get-started/locally/ (C++/LibTorch tab)
and extract to `KerrTraceCpp/libtorch/`.

## 2 — Build

### macOS (Apple Silicon — MPS GPU)

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=../libtorch \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(sysctl -n hw.logicalcpu)
```

### Linux (CUDA)

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=../libtorch \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

### Windows (CUDA / MSVC)

```powershell
mkdir build; cd build
cmake .. -DCMAKE_PREFIX_PATH="..\libtorch" `
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 3 — Run

### Single frame

```bash
./build_arm64/kerrtrace --disk-model riaf --riaf-color-mode gargantua \
            --enable-gargantua-look --spin 0.998 \
            --observer-inclination 80 --width 1280 --height 720 \
            --output out/render.png
```

If macOS cannot find `libomp.dylib`, use the helper launcher:

```bash
./scripts/run_kerrtrace.sh --disk-model riaf --riaf-color-mode gargantua \
  --enable-gargantua-look --spin 0.998 \
  --observer-inclination 80 --width 1280 --height 720 \
  --output out/render.png
```

### Animation

```bash
./build_arm64/kerrtrace --animate \
            --animation-parameter observer_azimuth_deg \
            --animation-start 0 --animation-end 360 \
            --animation-duration 3 --animation-fps 24 \
            --width 640 --height 360 \
            --output out/orbit.mp4
```

### Advanced disk controls (layered + differential + volume)

```bash
./scripts/run_kerrtrace.sh \
  --width 1280 --height 720 \
  --spin 0.85 --observer-radius 30 --observer-inclination 70 \
  --background-mode hdri --hdri-path ../KerrTrace/assets/backgrounds/downloads_imported/sfondo3.jpg \
  --enable-disk-layered-palette --disk-layer-count 24 --disk-layer-mix 0.75 \
  --disk-layer-pattern-count 18 --disk-layer-pattern-contrast 0.55 \
  --enable-disk-differential-rotation --disk-diffrot-visual-mode hybrid --disk-diffrot-strength 1.0 \
  --enable-disk-volume-emission --disk-volume-samples 7 --disk-volume-density-scale 1.4 --disk-volume-strength 1.1 \
  --enable-disk-segmented-palette --disk-segmented-palette-mode accretion_warm \
  --output out/advanced_disk.png
```

### From JSON config

```bash
./build_arm64/kerrtrace --config my_config.json --output out/result.png
```

### Python vs C++ parity reference (single command)

```bash
bash scripts/generate_parity_reference.sh out/parity_reference
```

Outputs:
- `python_reference.png`
- `cpp_reference.png`
- `python_vs_cpp_side_by_side.png`
- `metrics.txt` (`mae`, `rmse`, `psnr_db`)

## Parameters (mirrors Python KerrTrace)

| Flag | Python equivalent | Default |
|------|-------------------|---------|
| `--spin` | `spin` | 0.85 |
| `--disk-model` | `disk_model` | `physical_nt` |
| `--riaf-color-mode` | `riaf_color_mode` | `blackbody` |
| `--riaf-alpha-n/t/b` | `riaf_alpha_*` | 1.1/0.84/1.25 |
| `--enable-gargantua-look` | `gargantua_look_preset` | off |
| `--disk-beaming-strength` | `disk_beaming_strength` | 1.0 |
| `--device` | `device` | `auto` |
| `--render-tile-rows` | `render_tile_rows` | 64 |

Full parameter list: `./kerrtrace --help`

## GPU support

| Platform | GPU backend |
|----------|-------------|
| macOS (Apple Silicon) | MPS (Metal Performance Shaders) |
| Linux / Windows — NVIDIA | CUDA |
| All platforms | CPU (fallback) |

Device is selected automatically with `--device auto`.
Force with `--device cuda`, `--device mps`, or `--device cpu`.

## Performance vs Python

The C++ build eliminates Python interpreter overhead and enables more aggressive
memory reuse. On CPU, OpenMP tile parallelism is used. On GPU, the same LibTorch
tensor kernels run as in the Python version — the GPU compute time is identical,
but startup and frame-to-frame overhead is lower.

For maximum GPU performance, ensure you downloaded the CUDA or MPS-enabled
LibTorch package (not the CPU-only one).
