#pragma once
#include <torch/torch.h>

namespace kerrtrace {

// ── Metric components at a batch of points ────────────────────────────────────
// All tensors have shape (N,) unless noted.
// Convention: M=1, G=c=1, signature (-+++).

struct KerrMetric {
    torch::Tensor Sigma;   // r² + a²cos²θ
    torch::Tensor Delta;   // r² - 2r + a²
    torch::Tensor A_val;   // (r²+a²)² - a²Δsin²θ

    // Contravariant metric components
    torch::Tensor gtt_up;   // g^{tt}
    torch::Tensor grr_up;   // g^{rr}
    torch::Tensor gthth_up; // g^{θθ}
    torch::Tensor gpp_up;   // g^{φφ}
    torch::Tensor gtp_up;   // g^{tφ}

    // Covariant metric components
    torch::Tensor gtt;      // g_{tt}
    torch::Tensor grr;      // g_{rr}
    torch::Tensor gthth;    // g_{θθ}
    torch::Tensor gpp;      // g_{φφ}
    torch::Tensor gtp;      // g_{tφ}
};

// Compute all metric components at (r, theta) for Kerr with spin a.
KerrMetric compute_kerr_metric(
    const torch::Tensor& r,
    const torch::Tensor& theta,
    float a,
    float charge = 0.0f
);

// ── Geodesic RHS ──────────────────────────────────────────────────────────────
// state: (N, 6) columns = [r, θ, φ, t, p_r, p_θ]
// E:     (N,)  conserved energy  (-p_t)
// L:     (N,)  conserved ang-mom (p_φ)
// Returns dstate/dλ with same shape.
torch::Tensor geodesic_rhs(
    const torch::Tensor& state,
    const torch::Tensor& E,
    const torch::Tensor& L,
    float a,
    float charge = 0.0f
);

// ── Camera ray initialisation ─────────────────────────────────────────────────
// Pixel grid → initial (state, E, L) for backward ray tracing.
// Returns {state (N,6), E (N,), L (N,)} where N = width*height.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
init_camera_rays(
    int width, int height, float fov_deg,
    float r_obs, float theta_obs_deg, float phi_obs_deg, float roll_deg,
    float a, float charge,
    const torch::TensorOptions& opts
);

// ── Keplerian angular velocity at r (equatorial, prograde) ───────────────────
// Ω_K = 1/(r^{3/2} + a)  (M=1)
torch::Tensor keplerian_omega(const torch::Tensor& r, float a);

// ── Relativistic redshift/Doppler factor for Keplerian emitters ───────────────
// ξ = (E - L*Ω_K) / sqrt(-(g_tt + 2g_tφ*Ω_K + g_φφ*Ω_K²))
torch::Tensor relativistic_factor(
    const torch::Tensor& r,
    const torch::Tensor& E,
    const torch::Tensor& L,
    float a,
    float beaming_strength = 1.0f
);

} // namespace kerrtrace
