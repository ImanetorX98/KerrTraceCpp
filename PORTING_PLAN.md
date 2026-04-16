# KerrTrace C++ Porting Plan

## Goal
Bring `KerrTraceCpp` to practical feature parity with the Python `kerrtrace` pipeline in incremental, testable phases.

## Current Snapshot
- Core tracer loop exists (`geometry`, `raytracer`, `palette`, `animation`).
- Build system exists (CMake + LibTorch), with helper download script.
- CLI coverage for core Kerr workflows is broad (single frame + animation + advanced disk controls).
- Regression and CI infrastructure is available.

## Phase 1: Foundation and Visual Parity Baseline (in progress)
- [x] Add segmented disk palette controls to C++ config/CLI/JSON:
  - `disk_segmented_mix`
  - `disk_segmented_hue_offset`
  - `disk_segmented_palette_mode`
- [x] Port segmented color blending logic into C++ raytracer:
  - gaussian ring/sector blending
  - warm natural mode
  - rainbow mode
- [x] Build and run smoke test on local machine with LibTorch present.
- [x] Save side-by-side reference frames (`Python vs C++`) for a fixed seed/config.
  - Script: `scripts/generate_parity_reference.sh`
  - Last local run (CPU, 320x180): `MAE=55.3322`, `RMSE=92.7303`, `PSNR=8.79 dB`

## Phase 2: Feature Gap Closure
- [x] Port missing background features:
  - HDRI equirectangular loading/sampling
  - exposure/rotation controls
- [x] Port additional disk options used in Python workflows:
  - layered palette controls
  - differential rotation visual options
  - disk volume emission controls
- [x] Port output/runtime controls:
  - progress status file behavior (`--progress-file`) and terminal progress
  - frame resume/keep semantics (`--resume-frames`, `--keep-frames`, `--frames-dir`)
  - stream encoding settings parity (`--video-codec h264|h265`)

## Phase 3: Robustness and Validation
- [x] Add deterministic regression fixtures (small resolutions).
- [x] Add image similarity checks (PSNR/SSIM) for reference scenes.
- [x] Add config parity tests (Python JSON -> C++ parse/validate).
- [x] Profile CPU/GPU memory and tile behavior for long renders.

## Phase 4: Packaging and Developer Experience
- [x] Make `KerrTraceCpp` an independent git repo (optional but recommended).
- [x] Add CI build matrix (macOS/Linux, CPU at minimum).
- [x] Add release docs with known differences and migration notes.

## Working Rules
- Implement one focused slice at a time.
- Keep every step runnable (or clearly mark what dependency is missing).
- Record feature parity status in this document as each step lands.
