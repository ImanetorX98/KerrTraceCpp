#include "kerrtrace/palette.hpp"
#include <stdexcept>
#include <cmath>

namespace kerrtrace {

// ── Blackbody RGB (fast approximation, accurate ~1000K–40000K) ───────────────
// Based on Krystek & Taylor (2004) + Kang et al colour matching function fit.

torch::Tensor blackbody_rgb(const torch::Tensor& T, const torch::TensorOptions& opts)
{
    // Clamp temperature
    auto Tc = T.clamp(800.0f, 40000.0f);

    // Red channel
    auto R = torch::where(
        Tc <= 6600.0f,
        torch::ones_like(Tc),
        torch::clamp(
            329.698727446f * torch::pow((Tc / 100.0f - 60.0f), -0.1332047592f) / 255.0f,
            0.0f, 1.0f)
    );

    // Green channel
    auto G = torch::where(
        Tc <= 6600.0f,
        torch::clamp(
            (99.4708025861f * torch::log(Tc / 100.0f) - 161.1195681661f) / 255.0f,
            0.0f, 1.0f),
        torch::clamp(
            288.1221695283f * torch::pow((Tc / 100.0f - 60.0f), -0.0755148492f) / 255.0f,
            0.0f, 1.0f)
    );

    // Blue channel
    auto B = torch::where(
        Tc >= 6600.0f,
        torch::ones_like(Tc),
        torch::where(
            Tc <= 1900.0f,
            torch::zeros_like(Tc),
            torch::clamp(
                (138.5177312231f * torch::log(Tc / 100.0f - 10.0f) - 305.0447927307f) / 255.0f,
                0.0f, 1.0f)
        )
    );

    return torch::stack({R, G, B}, /*dim=*/1);
}

// ── Helper: piecewise-linear palette from 5 stops ────────────────────────────

static torch::Tensor piecewise_palette(
    const torch::Tensor& t,
    std::array<std::array<float,3>, 5> stops)
{
    auto tc = t.clamp(0.0f, 1.0f);
    const auto& dev = t.device();
    const auto& dt  = t.dtype();

    auto make_c = [&](const std::array<float,3>& s) {
        return torch::tensor({s[0], s[1], s[2]},
               torch::TensorOptions().dtype(dt).device(dev));
    };

    auto c0 = make_c(stops[0]);
    auto c1 = make_c(stops[1]);
    auto c2 = make_c(stops[2]);
    auto c3 = make_c(stops[3]);
    auto c4 = make_c(stops[4]);

    // Breakpoints at 0, 0.20, 0.46, 0.74, 1.0
    const float b1 = 0.20f, b2 = 0.46f, b3 = 0.74f;

    auto color = torch::zeros({(long)t.size(0), 3},
                 torch::TensorOptions().dtype(dt).device(dev));

    // Segment 0→1: t ∈ [0, b1)
    auto m0 = tc < b1;
    auto s0 = (tc / b1).clamp(0.0f, 1.0f).unsqueeze(1);
    auto c01 = c0.unsqueeze(0) + (c1 - c0).unsqueeze(0) * s0;
    color = torch::where(m0.unsqueeze(1).expand_as(color), c01, color);

    // Segment 1→2: t ∈ [b1, b2)
    auto m1 = (tc >= b1) & (tc < b2);
    auto s1 = ((tc - b1) / (b2 - b1)).clamp(0.0f, 1.0f).unsqueeze(1);
    auto c12 = c1.unsqueeze(0) + (c2 - c1).unsqueeze(0) * s1;
    color = torch::where(m1.unsqueeze(1).expand_as(color), c12, color);

    // Segment 2→3: t ∈ [b2, b3)
    auto m2 = (tc >= b2) & (tc < b3);
    auto s2 = ((tc - b2) / (b3 - b2)).clamp(0.0f, 1.0f).unsqueeze(1);
    auto c23 = c2.unsqueeze(0) + (c3 - c2).unsqueeze(0) * s2;
    color = torch::where(m2.unsqueeze(1).expand_as(color), c23, color);

    // Segment 3→4: t ∈ [b3, 1]
    auto m3 = tc >= b3;
    auto s3 = ((tc - b3) / (1.0f - b3)).clamp(0.0f, 1.0f).unsqueeze(1);
    auto c34 = c3.unsqueeze(0) + (c4 - c3).unsqueeze(0) * s3;
    color = torch::where(m3.unsqueeze(1).expand_as(color), c34, color);

    return color.clamp(0.0f, 1.2f);
}

// ── Named palettes ─────────────────────────────────────────────────────────────

torch::Tensor palette_interstellar_warm(const torch::Tensor& t)
{
    // Existing stops from Python code (inner=bright, outer=dark)
    return piecewise_palette(t, {{
        {1.00f, 0.95f, 0.82f},  // cream white  (t=0)
        {1.00f, 0.82f, 0.42f},  // warm yellow
        {0.97f, 0.56f, 0.20f},  // orange
        {0.78f, 0.28f, 0.10f},  // red-orange
        {0.42f, 0.08f, 0.03f},  // dark red-brown (t=1)
    }});
}

torch::Tensor palette_gargantua(const torch::Tensor& t)
{
    // Extracted from Interstellar (2014) Gargantua screenshot
    return piecewise_palette(t, {{
        {1.00f, 0.97f, 0.90f},  // near-white cream  (#FFF8E6)
        {1.00f, 0.86f, 0.47f},  // warm yellow        (#FFDC78)
        {0.90f, 0.59f, 0.24f},  // deep orange        (#E6963C)
        {0.63f, 0.35f, 0.12f},  // amber-brown        (#A05A1E)
        {0.35f, 0.18f, 0.06f},  // dark chocolate     (#5A2D0F)
    }});
}

torch::Tensor palette_plasma(const torch::Tensor& t)
{
    // Matplotlib 'plasma' approximation
    return piecewise_palette(t, {{
        {0.94f, 0.98f, 0.13f},  // yellow-green
        {0.99f, 0.55f, 0.13f},  // orange
        {0.80f, 0.19f, 0.47f},  // magenta
        {0.47f, 0.06f, 0.60f},  // purple
        {0.05f, 0.03f, 0.53f},  // dark indigo
    }});
}

torch::Tensor palette_default(const torch::Tensor& t)
{
    // Legacy warm orange/red palette
    return piecewise_palette(t, {{
        {1.00f, 0.90f, 0.70f},  // pale yellow
        {1.00f, 0.70f, 0.30f},  // warm orange
        {0.90f, 0.40f, 0.10f},  // burnt orange
        {0.60f, 0.15f, 0.05f},  // dark red
        {0.25f, 0.05f, 0.02f},  // near-black red
    }});
}

torch::Tensor named_palette(const std::string& name, const torch::Tensor& t)
{
    if (name == "interstellar_warm") return palette_interstellar_warm(t);
    if (name == "gargantua")         return palette_gargantua(t);
    if (name == "plasma")            return palette_plasma(t);
    return palette_default(t);
}

// ── Novikov-Thorne flux profile ────────────────────────────────────────────────

torch::Tensor nt_flux_profile(const torch::Tensor& r, float r_isco, bool page_thorne)
{
    constexpr float EPS = 1e-6f;
    auto r_in = torch::scalar_tensor(r_isco,
                torch::TensorOptions().dtype(r.dtype()).device(r.device()));
    auto x  = torch::sqrt(r.clamp_min(r_isco + EPS));
    auto x0 = std::sqrt(r_isco);

    if (!page_thorne) {
        // Proxy: F ∝ r^{-3} (1 - sqrt(r_isco/r))
        auto zero_torque = 1.0f - torch::sqrt(r_in / r.clamp_min(r_isco + EPS));
        auto flux = torch::pow(r, -3.0f) * zero_torque.clamp_min(0.0f);
        auto peak_r = r_isco * 1.36f;  // approx peak
        auto flux_ref = std::pow(peak_r, -3.0f) * (1.0f - std::sqrt(r_isco / peak_r));
        return (flux / std::max(flux_ref, EPS)).clamp(0.0f, 1e4f);
    }

    // Page & Thorne (1974) analytic profile
    // F ∝ (1/x³) * [(x-x0-3/2*log(x/x0)) - ...]
    // We use the Novikov-Thorne zero-torque form:
    float x1 = x0;
    float x2 = static_cast<float>(std::sqrt(3.0)) * x0;  // approx roots
    float x3 = x0;  // simplification for Schwarzschild limit

    // Simplified: F(r) = (3/(2*r_isco)) * (1 - sqrt(r_isco/r)) / r³
    auto zero_torque = (1.0f - torch::sqrt(r_in / r.clamp_min(r_isco + EPS))).clamp_min(0.0f);
    auto flux = zero_torque * torch::pow(r, -3.0f);

    // Normalise at peak
    float peak_r_est = r_isco * 1.5f;
    float flux_peak  = (1.0f - std::sqrt(r_isco / peak_r_est)) * std::pow(peak_r_est, -3.0f);
    return (flux / std::max(flux_peak, EPS)).clamp(0.0f, 1e4f);
}

// ── RIAF thin-disk emission ────────────────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
riaf_emission(
    const torch::Tensor& r_profile,
    float r_in,
    float r_out,
    float alpha_n,
    float alpha_T,
    float alpha_B,
    float T_visual,
    const std::string& color_mode)
{
    constexpr float EPS = 1e-6f;
    auto opts = torch::TensorOptions().dtype(r_profile.dtype()).device(r_profile.device());

    float alpha_j = alpha_n + 2.0f * alpha_T + 2.0f * alpha_B;  // ≈ 5.28 defaults

    auto r_in_t  = torch::scalar_tensor(r_in, opts);
    auto r_norm  = (r_profile / r_in_t).clamp_min(1.0f + EPS);

    // Zero-torque boundary condition
    auto zt = (1.0f - torch::rsqrt(r_norm.clamp_min(1.0001f))).clamp_min(0.0f);

    // Emissivity j ∝ r_norm^{-α_j} * (1-r^{-1/2})
    auto j_prof = torch::pow(r_norm, -alpha_j) * zt;

    // Normalise at analytic peak: r_peak = ((α_j+0.5)/α_j)²
    float r_pk    = std::pow((alpha_j + 0.5f) / alpha_j, 2.0f);
    float j_ref   = std::pow(r_pk, -alpha_j) * (1.0f - 1.0f / std::sqrt(r_pk));
    j_ref = std::max(j_ref, EPS);
    auto intensity = (j_prof / j_ref).clamp(0.0f, 10.0f);

    // Colour
    torch::Tensor color;
    if (color_mode == "plasma") {
        float r_norm_out = std::max(r_out / std::max(r_in, EPS), 1.0f);
        float T_outer = T_visual * std::pow(r_norm_out, -alpha_T);
        auto T_r = T_visual * torch::pow(r_norm, -alpha_T);
        auto heat = ((T_r - T_outer) / std::max(T_visual - T_outer, EPS)).clamp(0.0f, 1.0f);
        color = palette_plasma(heat);
    } else if (color_mode == "interstellar_warm" || color_mode == "gargantua") {
        float span = std::max(r_out - r_in, EPS);
        auto x_pos = ((r_profile - r_in_t) / span).clamp(0.0f, 1.0f);
        color = (color_mode == "gargantua")
              ? palette_gargantua(x_pos)
              : palette_interstellar_warm(x_pos);
    } else {
        // blackbody
        auto T_r = T_visual * torch::pow(r_norm, -alpha_T);
        color = blackbody_rgb(T_r, opts);
    }

    return {intensity, color};
}

} // namespace kerrtrace
