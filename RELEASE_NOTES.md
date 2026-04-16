# KerrTraceCpp Release Notes

## v1.3.0 (Porting Milestone)

### What Is Implemented
- Kerr/Kerr-Newman tracer core in C++ (LibTorch backend).
- Segmented disk palette (`accretion_warm`, `rainbow`).
- Layered disk palette controls.
- Differential-rotation visual modulation controls.
- Disk volume emission controls.
- HDRI equirectangular background sampling with exposure/rotation.
- Animation rendering and ffmpeg encoding (`h264`/`h265`).
- Resume/keep semantics for animation frame directories.
- Progress status file output (`--progress-file`).

### Validation Tooling
- Regression fixture generation:
  - `bash scripts/generate_regression_fixtures.sh`
- Regression suite (PSNR/SSIM + Python-style config acceptance):
  - `python scripts/run_regression_suite.py`
- Tile/memory profiling helper:
  - `bash scripts/profile_tiles.sh`

### Known Differences vs Python `kerrtrace`
- No native desktop progress window UI in C++ yet (text progress + progress-file provided).
- Python still has broader feature coverage in non-core areas (wormhole-specific and web UI workflows).
- Metric-for-metric visual parity is close but not byte-identical; use PSNR/SSIM thresholds for regression.

### Migration Notes
- Existing JSON config files are accepted; unknown keys are ignored by C++ parser.
- For long jobs, use:
  - `--resume-frames`
  - `--keep-frames`
  - `--frames-dir out/my_frames`
  - `--progress-file out/progress.txt`
