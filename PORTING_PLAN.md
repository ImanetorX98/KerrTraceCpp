# KerrTrace C++ Porting Plan

## Goal
Bring `KerrTraceCpp` to practical feature parity with the Python `kerrtrace` pipeline in incremental, testable phases.

## Current Snapshot
- Core tracer loop exists (`geometry`, `raytracer`, `palette`, `animation`).
- Build system exists (CMake + LibTorch), but `libtorch/` is not checked in.
- CLI coverage is partial compared with Python.
- Background HDRI sampling is still TODO.

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
- [ ] Port additional disk options used in Python workflows:
  - layered palette controls
  - differential rotation visual options
  - disk volume emission controls
- [ ] Port output/runtime controls:
  - progress window behavior
  - frame resume/keep semantics
  - stream encoding settings parity

## Phase 3: Robustness and Validation
- [ ] Add deterministic regression fixtures (small resolutions).
- [ ] Add image similarity checks (PSNR/SSIM) for reference scenes.
- [ ] Add config parity tests (Python JSON -> C++ parse/validate).
- [ ] Profile CPU/GPU memory and tile behavior for long renders.

## Phase 4: Packaging and Developer Experience
- [ ] Make `KerrTraceCpp` an independent git repo (optional but recommended).
- [ ] Add CI build matrix (macOS/Linux, CPU at minimum).
- [ ] Add release docs with known differences and migration notes.

## Working Rules
- Implement one focused slice at a time.
- Keep every step runnable (or clearly mark what dependency is missing).
- Record feature parity status in this document as each step lands.
