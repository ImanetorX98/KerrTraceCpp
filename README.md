# KerrTrace C++

GPU-accelerated ray tracer for Kerr black holes with accretion disk emission, written in C++17 with [LibTorch](https://pytorch.org/cppdocs/) as the tensor backend.

This is a C++ reimplementation of the [Python KerrTrace](https://github.com/ImanRosignoli/KerrTrace) project, sharing the same physical model, JSON config format, and CLI flags — with lower startup overhead and tighter memory control.

---

## Features

- **Spacetime models** — Kerr, Kerr-Newman, Schwarzschild; Boyer-Lindquist and Kerr-Schild coordinates
- **Accretion disk** — Novikov-Thorne (`physical_nt`), legacy power-law (`legacy`), RIAF with blackbody/plasma/gargantua color modes
- **Disk palette system** — layered turbulence, segmented color-wheel (`accretion_warm`, `rainbow`), differential rotation, volume emission
- **Background** — dark space, HDRI equirectangular (with exposure and rotation), procedural star field
- **Animation** — azimuth orbit, inclination sweep, arbitrary parameter sweep; stream and frame-file encoding; resume/keep semantics
- **Video encoding** — h264 and h265/10-bit via ffmpeg
- **GPU support** — MPS (Apple Silicon), CUDA (NVIDIA), CPU fallback; auto-detected at runtime
- **Config file** — JSON input/output, fully compatible with Python KerrTrace config files
- **Post-processing** — Gargantua cinematic look preset; ACES filmic tone-map + gamma 2.2

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| CMake | ≥ 3.18 | |
| C++ compiler | C++17 | clang++ (macOS), GCC or clang (Linux), MSVC (Windows) |
| LibTorch | ≥ 2.0 | downloaded via helper script |
| OpenMP | any | optional; used for CPU tile parallelism |
| ffmpeg | any | optional; required for video output |

---

## Build

### Step 1 — Download LibTorch

**macOS / Linux**
```bash
bash scripts/download_libtorch.sh
```

**Windows (PowerShell)**
```powershell
.\scripts\download_libtorch.ps1
```

Or download manually from https://pytorch.org/get-started/locally/ (select C++ / LibTorch) and extract to `libtorch/` inside this directory.

### Step 2 — Compile

**macOS (Apple Silicon — MPS)**

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=../libtorch \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(sysctl -n hw.logicalcpu)
```

**Linux (CUDA)**

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=../libtorch \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

**Windows (CUDA / MSVC)**

```powershell
mkdir build; cd build
cmake .. -DCMAKE_PREFIX_PATH="..\libtorch" `
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

> **macOS — library path note.** If the binary fails with a missing `libomp.dylib` error, use the wrapper script instead of calling the binary directly:
> ```bash
> bash scripts/run_kerrtrace.sh [args...]
> ```

---

## Quick Start

> **Path conventions**
> - macOS / Linux: `./build/kerrtrace`
> - Windows: `.\build\Release\kerrtrace.exe`

### Single frame

**macOS / Linux**
```bash
./build/kerrtrace \
  --spin 0.85 \
  --observer-inclination 75 \
  --width 1280 --height 720 \
  --output out/render.png
```

**Windows**
```powershell
.\build\Release\kerrtrace.exe `
  --spin 0.85 `
  --observer-inclination 75 `
  --width 1280 --height 720 `
  --output out\render.png
```

### Gargantua preset

```bash
./build/kerrtrace \
  --disk-model riaf --riaf-color-mode gargantua \
  --enable-gargantua-look --spin 0.998 \
  --observer-inclination 80 \
  --width 1920 --height 1080 \
  --output out/gargantua.png
```

### Animation (azimuth orbit)

```bash
./build/kerrtrace \
  --animate \
  --animation-parameter observer_azimuth_deg \
  --animation-start 0 --animation-end 360 \
  --animation-duration 4 --animation-fps 24 \
  --width 1280 --height 720 \
  --output out/orbit.mp4
```

### From JSON config

```bash
./build/kerrtrace --config my_config.json --output out/result.png
```

JSON configs exported from the Python version are accepted without modification.

---

## CLI Reference

```
./build/kerrtrace --help
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--width` / `--height` | 1280 / 720 | Output resolution |
| `--spin` | 0.85 | Black hole spin \|a\| ≤ 1 |
| `--charge` | 0.0 | Black hole charge \|Q\| ≤ 1 (Kerr-Newman) |
| `--metric-model` | `kerr` | `kerr` \| `kerr_newman` \| `schwarzschild` |
| `--coordinate-system` | `kerr_schild` | `kerr_schild` \| `boyer_lindquist` |
| `--disk-model` | `physical_nt` | `physical_nt` \| `legacy` \| `riaf` |
| `--disk-palette` | `default` | `default` \| `interstellar_warm` |
| `--enable-disk-layered-palette` | off | Enable layered turbulence palette |
| `--enable-disk-segmented-palette` | off | Enable segmented color-wheel palette |
| `--enable-disk-volume-emission` | off | Enable volumetric above-plane emission |
| `--background-mode` | `darkspace` | `darkspace` \| `procedural` \| `hdri` |
| `--hdri-path` | — | Path to equirectangular HDR/image file |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--animate` | off | Enable animation mode |
| `--animation-parameter` | `observer_azimuth_deg` | Config key to animate |
| `--animation-duration` | 2.0 | Duration in seconds |
| `--animation-fps` | 24 | Frames per second |
| `--video-codec` | `h264` | `h264` \| `h265` |
| `--resume-frames` | off | Skip already-rendered frames |
| `--keep-frames` | off | Keep per-frame PNGs after encoding |
| `--frames-dir` | auto temp | Directory for per-frame PNGs |
| `--progress-file` | — | Write JSON progress updates to file |
| `--enable-gargantua-look` | off | Apply Gargantua cinematic post-process |
| `--max-steps` | 500 | Geodesic integration steps |
| `--step-size` | 0.11 | Base integration step size |
| `--adaptive-integrator` | on | RK4/5 adaptive step control |
| `--render-tile-rows` | 64 | Rows per compute tile (memory control) |
| `--config` | — | Load parameters from JSON file |
| `--output` | `out/render.png` | Output path (`.png`, `.mp4`, `.mov`, `.mkv`, `.gif`) |

---

## GPU Support

| Platform | Backend |
|----------|---------|
| macOS — Apple Silicon | MPS (Metal Performance Shaders) |
| Linux / Windows — NVIDIA | CUDA |
| All platforms | CPU (fallback) |

Device is selected automatically with `--device auto`.  
Force a specific backend: `--device mps`, `--device cuda`, `--device cpu`.

---

## Performance

C++ eliminates Python interpreter and import overhead (cold start < 0.5 s vs ~15 s in Python).  
At small-to-medium resolutions (up to ~1080p), the C++ build is significantly faster.  
At 4K and above, the bottleneck shifts to GPU tensor kernels — which are identical between C++ and Python since both use LibTorch — so the relative speedup narrows.

| Resolution | C++ | Python | Speedup |
|------------|-----|--------|---------|
| 320×180 (CPU) | ~8 s | ~320 s | ~40× |
| 1280×720 (MPS) | ~22 s | ~60 s | ~2.7× |
| 3840×2160 (MPS) | ~362 s | ~448 s | ~1.2× |

*Measured on Apple M-series; actual times vary by model and disk complexity.*

---

## Validation

> The shell scripts below require bash. On Windows, run them via **Git Bash** or **WSL**.

Generate deterministic regression fixtures:

```bash
bash scripts/generate_regression_fixtures.sh
```

Run regression checks (PSNR/SSIM + Python config acceptance):

```bash
python3 scripts/run_regression_suite.py
```

On Windows (PowerShell):

```powershell
python scripts\run_regression_suite.py
```

Generate side-by-side Python vs C++ parity reference:

```bash
bash scripts/generate_parity_reference.sh out/parity_reference
# outputs: python_reference.png, cpp_reference.png, side_by_side.png, metrics.txt
```

Profile tile-size / memory behavior:

```bash
bash scripts/profile_tiles.sh
```

---

## Known Differences vs Python KerrTrace

- **Tone-mapping**: C++ uses ACES filmic + gamma 2.2; Python uses Reinhard + gamma 2.2. Images are visually comparable but not pixel-identical (~30% brightness difference in disk midtones).
- **No desktop progress UI**: C++ provides terminal progress and `--progress-file` JSON output; the Tk/web progress window is Python-only.
- **No wormhole or web UI**: those workflows remain Python-only for now.
- **Metric-for-metric visual parity**: PSNR ≈ 25–30 dB on reference scenes at matched parameters.

---

## Project Structure

```
KerrTraceCpp/
├── src/
│   ├── main.cpp          # CLI entry point
│   ├── config.cpp        # RenderConfig JSON serialization + validation
│   ├── geometry.cpp      # Geodesic integration, Doppler/beaming physics
│   ├── raytracer.cpp     # Tile-based render loop, disk shading
│   ├── palette.cpp       # Layered / segmented disk color systems
│   ├── animation.cpp     # Frame scheduling + ffmpeg encoding
│   └── image_io.cpp      # PNG load/save, tone-map operators
├── include/kerrtrace/    # Public headers
├── tests/fixtures/       # Reference images for regression
├── scripts/              # Build helpers, benchmark, regression scripts
├── CMakeLists.txt
└── README_BUILD.md       # Detailed build notes
```

---

## License

MIT — see [LICENSE](LICENSE).
