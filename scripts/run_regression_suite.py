#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1.0e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    # Lightweight global SSIM (channel-averaged), sufficient for regression gates.
    k1 = 0.01
    k2 = 0.03
    L = 255.0
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    vals = []
    for ch in range(3):
        x = a[..., ch]
        y = b[..., ch]
        mux = float(np.mean(x))
        muy = float(np.mean(y))
        sigx2 = float(np.var(x))
        sigy2 = float(np.var(y))
        sigxy = float(np.mean((x - mux) * (y - muy)))
        num = (2.0 * mux * muy + c1) * (2.0 * sigxy + c2)
        den = (mux * mux + muy * muy + c1) * (sigx2 + sigy2 + c2)
        vals.append(num / den if den > 1.0e-12 else 1.0)
    return float(np.mean(vals))


def run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def render_config(root: Path, config_path: Path, output_path: Path) -> None:
    run_cmd(
        [
            "bash",
            str(root / "scripts" / "run_kerrtrace.sh"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=root,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run KerrTraceCpp regression checks.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--psnr-min", type=float, default=35.0)
    ap.add_argument("--ssim-min", type=float, default=0.95)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    fixture_dir = root / "tests" / "fixtures"
    ref_dir = fixture_dir / "reference"
    tmp_dir = root / "out" / "regression_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("baseline", fixture_dir / "config_regression_baseline.json", ref_dir / "baseline_160x90.png"),
        ("advanced", fixture_dir / "config_regression_advanced.json", ref_dir / "advanced_160x90.png"),
    ]

    missing = [str(ref) for _, _, ref in cases if not ref.exists()]
    if missing:
        print("Missing reference fixtures:")
        for p in missing:
            print(f"  - {p}")
        print("Run: bash scripts/generate_regression_fixtures.sh")
        return 2

    failures: list[str] = []
    for name, cfg, ref in cases:
        out = tmp_dir / f"{name}_render.png"
        print(f"[render] {name}")
        render_config(root, cfg, out)
        a = load_rgb(out)
        b = load_rgb(ref)
        if a.shape != b.shape:
            failures.append(f"{name}: shape mismatch {a.shape} vs {b.shape}")
            continue
        m_psnr = psnr(a, b)
        m_ssim = ssim_global(a, b)
        print(f"[metrics] {name}: PSNR={m_psnr:.3f} dB  SSIM={m_ssim:.6f}")
        if m_psnr < args.psnr_min:
            failures.append(f"{name}: PSNR {m_psnr:.3f} < {args.psnr_min:.3f}")
        if m_ssim < args.ssim_min:
            failures.append(f"{name}: SSIM {m_ssim:.6f} < {args.ssim_min:.6f}")

    # Config parity check: C++ must accept Python-style JSON config and render.
    parity_cfg = fixture_dir / "config_python_style_sample.json"
    parity_out = tmp_dir / "python_style_config_render.png"
    print("[config-parity] Rendering Python-style config fixture")
    try:
        render_config(root, parity_cfg, parity_out)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"config parity failed: {exc}")

    if failures:
        print("\nRegression FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    if not args.keep_temp:
        for p in tmp_dir.glob("*.png"):
            p.unlink(missing_ok=True)
    print("\nRegression PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
