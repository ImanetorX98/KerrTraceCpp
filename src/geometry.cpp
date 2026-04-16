#include "kerrtrace/geometry.hpp"
#include "kerrtrace/config.hpp"
#include <cmath>
#include <stdexcept>

namespace kerrtrace {

// ── Metric ────────────────────────────────────────────────────────────────────

KerrMetric compute_kerr_metric(
    const torch::Tensor& r,
    const torch::Tensor& theta,
    float a,
    float charge)
{
    const float a2 = a * a;
    const float Q2 = charge * charge;

    auto sin_t  = torch::sin(theta);
    auto cos_t  = torch::cos(theta);
    auto sin2   = sin_t * sin_t;
    auto cos2   = cos_t * cos_t;

    auto Sigma  = r * r + a2 * cos2;
    // Kerr-Newman: Δ = r²-2r+a²+Q²
    auto Delta  = r * r - 2.0f * r + a2 + Q2;
    auto A_val  = (r * r + a2).pow(2) - a2 * Delta * sin2;

    constexpr float EPS = 1e-7f;
    auto Sig    = Sigma.clamp_min(EPS);
    auto Del    = Delta.abs().clamp_min(EPS) * Delta.sign();
    auto sin2e  = sin2.clamp_min(EPS);
    auto SigDel = Sig * Del;

    KerrMetric m;
    m.Sigma = Sigma;
    m.Delta = Delta;
    m.A_val = A_val;

    // Contravariant
    m.gtt_up   = -A_val / SigDel;
    m.grr_up   = Del / Sig;
    m.gthth_up = 1.0f / Sig;
    m.gpp_up   = (Del - a2 * sin2) / (SigDel * sin2e);
    m.gtp_up   = -2.0f * a * r / SigDel;

    // Covariant
    m.gtt   = -(Sig - 2.0f * r) / Sig;        // Kerr-Schild: -(1 - 2r/Σ) approx
    m.grr   = Sig / Del;
    m.gthth = Sig;
    m.gpp   = sin2 * (r * r + a2 + 2.0f * r * a2 * sin2 / Sig.clamp_min(EPS));
    m.gtp   = -2.0f * a * r * sin2 / Sig;

    return m;
}

// ── Geodesic RHS ──────────────────────────────────────────────────────────────

torch::Tensor geodesic_rhs(
    const torch::Tensor& state,
    const torch::Tensor& E,
    const torch::Tensor& L,
    float a,
    float charge)
{
    const float a2 = a * a;
    const float Q2 = charge * charge;
    constexpr float EPS = 1e-7f;

    auto r      = state.select(1, 0);
    auto theta  = state.select(1, 1);
    auto p_r    = state.select(1, 4);
    auto p_th   = state.select(1, 5);

    auto sin_t  = torch::sin(theta);
    auto cos_t  = torch::cos(theta);
    auto sin2   = sin_t * sin_t;
    auto cos2   = cos_t * cos_t;
    auto sin2e  = sin2.clamp_min(EPS);

    auto Sigma  = r * r + a2 * cos2;
    auto Delta  = r * r - 2.0f * r + a2 + Q2;
    auto A_val  = (r * r + a2).pow(2) - a2 * Delta * sin2;

    auto Sig    = Sigma.clamp_min(EPS);
    auto Del    = Delta.abs().clamp_min(EPS) * Delta.sign();
    auto SigDel = Sig * Del;
    auto Sig2   = Sig * Sig;
    auto SigDel2 = SigDel * SigDel;

    // ── Position derivatives (contravariant velocity) ──────────────────────
    // dr/dλ = g^{rr} p_r
    auto dr = (Del / Sig) * p_r;
    // dθ/dλ = g^{θθ} p_θ
    auto dth = p_th / Sig;
    // dφ/dλ = g^{φφ} L - E g^{tφ}
    auto gpp = (Del - a2 * sin2) / (SigDel * sin2e);
    auto gtp = -2.0f * a * r / SigDel;
    auto dphi = gpp * L - E * gtp;
    // dt/dλ = -E g^{tt} + L g^{tφ}
    auto gtt = -A_val / SigDel;
    auto dt  = -E * gtt + L * gtp;

    // ── Momentum derivatives: dp_r/dλ = -½ ∂g^{μν}/∂r · p_μ p_ν ──────────
    auto dSig_dr  = 2.0f * r;
    auto dDel_dr  = 2.0f * r - 2.0f;   // M=1
    auto dA_dr    = 4.0f * r * (r * r + a2) - 2.0f * a2 * (r - 1.0f) * sin2;
    auto dSD_dr   = dSig_dr * Del + Sig * dDel_dr;   // ∂(Σ·Δ)/∂r

    auto dgtt_dr  = -(dA_dr * SigDel - A_val * dSD_dr) / SigDel2;
    auto dgrr_dr  = (dDel_dr * Sig - Del * dSig_dr) / Sig2;
    auto dgthth_dr = -dSig_dr / Sig2;

    // g^{φφ} = num_φ / den_φ  (den_φ = Σ·Δ·sin²θ)
    auto num_p    = Del - a2 * sin2;
    auto den_p    = SigDel * sin2e;
    auto dnum_p_dr = dDel_dr;                          // a²sin²θ independent of r
    auto dden_p_dr = dSD_dr * sin2e;
    auto dgpp_dr  = (dnum_p_dr * den_p - num_p * dden_p_dr) / (den_p * den_p);

    auto dgtp_dr  = -2.0f * a * (SigDel - r * dSD_dr) / SigDel2;

    auto dp_r = -0.5f * (
        dgtt_dr   * E * E
      - 2.0f * E * L * dgtp_dr
      + dgrr_dr  * p_r * p_r
      + dgthth_dr * p_th * p_th
      + dgpp_dr  * L * L
    );

    // ── dp_θ/dλ = -½ ∂g^{μν}/∂θ · p_μ p_ν ────────────────────────────────
    auto sin2t    = torch::sin(2.0f * theta);
    auto dSig_dth = -a2 * sin2t;                      // ∂Σ/∂θ = -a²sin2θ
    // ∂Δ/∂θ = 0
    auto dA_dth   = -a2 * Del * sin2t;                // ∂A/∂θ = -a²Δ sin2θ
    auto dSD_dth  = Del * dSig_dth;                   // ∂(ΣΔ)/∂θ, Δ const in θ

    auto dgtt_dth  = -(dA_dth * SigDel - A_val * dSD_dth) / SigDel2;
    auto dgrr_dth  = -Del * dSig_dth / Sig2;
    auto dgthth_dth = -dSig_dth / Sig2;

    // ∂(num_φ)/∂θ = ∂(Δ-a²sin²θ)/∂θ = -a²sin2θ = dSig_dth (same value)
    auto dnum_p_dth = dSig_dth;
    // ∂(den_φ)/∂θ = ∂(ΣΔsin²θ)/∂θ = Δ(sin²θ ∂Σ/∂θ + Σ sin2θ)
    auto dden_p_dth = Del * (sin2e * dSig_dth + Sig * sin2t);
    auto dgpp_dth  = (dnum_p_dth * den_p - num_p * dden_p_dth) / (den_p * den_p);

    // ∂g^{tφ}/∂θ = -2ar · ∂(1/(ΣΔ))/∂θ = -2ar · (-dSig_dth/(Σ²Δ))
    auto dgtp_dth  = 2.0f * a * r * dSig_dth / (Sig2 * Del.abs().clamp_min(EPS));

    auto dp_th = -0.5f * (
        dgtt_dth   * E * E
      - 2.0f * E * L * dgtp_dth
      + dgrr_dth  * p_r * p_r
      + dgthth_dth * p_th * p_th
      + dgpp_dth  * L * L
    );

    // State: [r, θ, φ, t, p_r, p_θ]
    return torch::stack({dr, dth, dphi, dt, dp_r, dp_th}, /*dim=*/1);
}

// ── Camera ray initialisation ─────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
init_camera_rays(
    int W, int H, float fov_deg,
    float r_obs, float theta_obs_deg, float phi_obs_deg, float roll_deg,
    float a, float charge,
    const torch::TensorOptions& opts)
{
    const float a2 = a * a;
    const float Q2 = charge * charge;
    constexpr float EPS = 1e-7f;
    const float PI = static_cast<float>(M_PI);

    float theta_obs = theta_obs_deg * PI / 180.0f;
    float phi_obs   = phi_obs_deg   * PI / 180.0f;
    float fov_half  = fov_deg * PI / 180.0f * 0.5f;
    float aspect    = static_cast<float>(W) / static_cast<float>(H);

    int N = W * H;

    // ── Pixel grid → angular offsets ─────────────────────────────────────────
    // u ∈ [-1,1] (right), v ∈ [-1,1] (up)
    auto col_idx = torch::arange(W, opts);
    auto row_idx = torch::arange(H, opts);
    auto grids = torch::meshgrid({row_idx, col_idx}, "ij");
    auto grid_r = grids[0];
    auto grid_c = grids[1];
    auto u = ((grid_c.flatten().to(opts.dtype()) + 0.5f) / W - 0.5f) * 2.0f;  // right +
    auto v = (0.5f - (grid_r.flatten().to(opts.dtype()) + 0.5f) / H) * 2.0f;  // up +

    // Ray direction in local (r, θ, φ) frame
    // Forward = -∂_r, up = -∂_θ, right = +∂_φ
    float t_fov = std::tan(fov_half);
    auto dx = u * (aspect * t_fov);  // local φ component
    auto dy = v * t_fov;             // local -θ component
    // z = 1 (toward BH, -r direction)
    auto len = torch::sqrt(1.0f + dx * dx + dy * dy);
    auto dir_r   = -(1.0f / len);           // -r
    auto dir_th  = -(dy / len);             // θ (down = +θ)
    auto dir_phi = (dx / len);              // +φ

    // ── Metric at observer ────────────────────────────────────────────────────
    auto r_t  = torch::full({N}, r_obs, opts);
    auto th_t = torch::full({N}, theta_obs, opts);

    auto sin_t  = torch::sin(th_t);
    auto cos_t  = torch::cos(th_t);
    auto sin2   = sin_t * sin_t;
    auto cos2   = cos_t * cos_t;
    auto Sigma  = r_t * r_t + a2 * cos2;
    auto Delta  = r_t * r_t - 2.0f * r_t + a2 + Q2;
    auto A_val  = (r_t * r_t + a2).pow(2) - a2 * Delta * sin2;
    auto sin2e  = sin2.clamp_min(EPS);
    auto Sig    = Sigma.clamp_min(EPS);
    auto Del    = Delta.abs().clamp_min(EPS) * Delta.sign();
    auto SigDel = Sig * Del;

    // sqrt(g_rr) = sqrt(Σ/Δ)
    auto sqrt_grr  = torch::sqrt(Sig / Del.clamp_min(EPS));
    // sqrt(g_θθ) = sqrt(Σ)
    auto sqrt_gthth = torch::sqrt(Sig);
    // sqrt(g_φφ)
    auto g_pp      = sin2e * (r_t * r_t + a2 + 2.0f * r_t * a2 * sin2e / Sig);
    auto sqrt_gpp  = torch::sqrt(g_pp.clamp_min(EPS));

    // ── Covariant momenta from local direction ────────────────────────────────
    // p_r = dir_r * sqrt(g_rr)
    // p_θ = dir_θ * sqrt(g_θθ)
    // L = p_φ = dir_φ * sqrt(g_φφ)
    auto p_r  = dir_r  * sqrt_grr;
    auto p_th = dir_th * sqrt_gthth;
    auto L    = dir_phi * sqrt_gpp;

    // ── p_t from null condition H=0 ───────────────────────────────────────────
    // g^{tt}p_t² + 2g^{tφ}L·p_t + (g^{rr}p_r² + g^{θθ}p_θ² + g^{φφ}L²) = 0
    auto gtt_up  = -A_val / SigDel;
    auto gtp_up  = -2.0f * a * r_t / SigDel;
    auto grr_up  = Del / Sig;
    auto gthth_up = 1.0f / Sig;
    auto gpp_up  = (Del - a2 * sin2) / (SigDel * sin2e);

    auto rhs_term = grr_up * p_r * p_r + gthth_up * p_th * p_th + gpp_up * L * L;
    // discriminant: (g^{tφ}L)² - g^{tt} * rhs
    auto discr = (gtp_up * L).pow(2) - gtt_up * rhs_term;
    discr = discr.clamp_min(0.0f);

    // p_t = (-g^{tφ}L + sqrt(discr)) / g^{tt}   → gives p_t < 0 (E > 0)
    auto p_t = (-gtp_up * L + torch::sqrt(discr)) / gtt_up;
    auto E   = -p_t;  // E > 0

    // ── Initial state tensor [r, θ, φ, t, p_r, p_θ] ──────────────────────────
    auto phi_t = torch::full({N}, phi_obs, opts);
    auto t_coord = torch::zeros({N}, opts);
    auto state = torch::stack({r_t, th_t, phi_t, t_coord, p_r, p_th}, /*dim=*/1);

    return {state, E, L};
}

// ── Keplerian omega ────────────────────────────────────────────────────────────

torch::Tensor keplerian_omega(const torch::Tensor& r, float a)
{
    // Ω_K = ±1 / (r^{3/2} ± a)  prograde (+)
    return 1.0f / (torch::pow(r.clamp_min(1e-3f), 1.5f) + a);
}

// ── Relativistic factor ────────────────────────────────────────────────────────

torch::Tensor relativistic_factor(
    const torch::Tensor& r_disk,
    const torch::Tensor& E,
    const torch::Tensor& L,
    float a,
    float beaming_strength)
{
    constexpr float EPS = 1e-7f;
    const float a2 = a * a;

    // Equatorial plane (θ=π/2, cos θ=0, sin θ=1)
    auto Sigma = r_disk * r_disk;  // Σ at eq.
    auto Delta = r_disk * r_disk - 2.0f * r_disk + a2;
    auto g_tt  = -(Sigma - 2.0f * r_disk) / Sigma.clamp_min(EPS);
    auto g_tp  = -2.0f * a * r_disk / Sigma.clamp_min(EPS);
    auto g_pp  = r_disk * r_disk + a2 + 2.0f * a2 * r_disk / Sigma.clamp_min(EPS);

    auto Omega_K = keplerian_omega(r_disk, a);
    auto norm2   = -(g_tt + 2.0f * g_tp * Omega_K + g_pp * Omega_K * Omega_K);
    auto u_t     = 1.0f / torch::sqrt(norm2.clamp_min(EPS));

    // ξ = (E - L*Ω_K) * u_t
    auto xi = (E - L * Omega_K) * u_t;
    xi = xi.clamp_min(EPS);

    // Observed intensity ∝ ξ^(3+beaming)
    return torch::pow(xi, 3.0f + beaming_strength);
}

} // namespace kerrtrace
