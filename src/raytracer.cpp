#include "kerrtrace/raytracer.hpp"
#include "kerrtrace/geometry.hpp"
#include "kerrtrace/palette.hpp"
#include "kerrtrace/config.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace kerrtrace {

// ── Helpers ───────────────────────────────────────────────────────────────────

static torch::Device resolve_device(const std::string& dev_str)
{
    if (dev_str == "cpu")  return torch::kCPU;
    if (dev_str == "cuda") return torch::kCUDA;
    if (dev_str == "mps")  return torch::Device("mps");
    // auto
    if (torch::cuda::is_available()) return torch::kCUDA;
#ifdef __APPLE__
    // MPS availability check (LibTorch 2.0+)
    try {
        auto x = torch::zeros({1}, torch::device(torch::Device("mps")));
        return torch::Device("mps");
    } catch (...) {}
#endif
    return torch::kCPU;
}

static torch::Dtype resolve_dtype(const std::string& s)
{
    return (s == "float64") ? torch::kFloat64 : torch::kFloat32;
}

static torch::Tensor fract_tensor(const torch::Tensor& x)
{
    return x - torch::floor(x);
}

static torch::Tensor hsv_to_rgb(const torch::Tensor& h, const torch::Tensor& s, const torch::Tensor& v)
{
    auto h6 = torch::remainder(h, 1.0f) * 6.0f;
    auto i = torch::floor(h6).to(torch::kInt64);
    auto f = h6 - torch::floor(h6);
    auto p = v * (1.0f - s);
    auto q = v * (1.0f - f * s);
    auto t = v * (1.0f - (1.0f - f) * s);

    auto cond0 = (i == 0);
    auto cond1 = (i == 1);
    auto cond2 = (i == 2);
    auto cond3 = (i == 3);
    auto cond4 = (i == 4);

    auto r = torch::where(cond0, v,
             torch::where(cond1, q,
             torch::where(cond2, p,
             torch::where(cond3, p,
             torch::where(cond4, t, v)))));
    auto g = torch::where(cond0, t,
             torch::where(cond1, v,
             torch::where(cond2, v,
             torch::where(cond3, q,
             torch::where(cond4, p, p)))));
    auto b = torch::where(cond0, p,
             torch::where(cond1, p,
             torch::where(cond2, t,
             torch::where(cond3, v,
             torch::where(cond4, v, q)))));
    return torch::stack({r, g, b}, 1);
}

// ── Constructor ───────────────────────────────────────────────────────────────

Raytracer::Raytracer(const RenderConfig& cfg)
    : cfg_(cfg)
    , device_(resolve_device(cfg.device))
    , dtype_(resolve_dtype(cfg.dtype))
    , a_(cfg.spin)
{
    r_isco_    = isco_radius(a_);
    r_horizon_ = event_horizon_radius(a_, cfg.charge);
}

// ── Main render ───────────────────────────────────────────────────────────────

torch::Tensor Raytracer::render(ProgressFn progress) const
{
    int H = cfg_.height, W = cfg_.width;
    int tile_rows = std::max(1, std::min(cfg_.render_tile_rows, H));

    auto result = torch::zeros({H, W, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    int n_tiles = (H + tile_rows - 1) / tile_rows;
    int done    = 0;

    for (int tile = 0; tile < n_tiles; ++tile) {
        int r0 = tile * tile_rows;
        int r1 = std::min(r0 + tile_rows, H);

        auto tile_img = render_tile(r0, r1);  // (r1-r0, W, 3) on device
        result.slice(0, r0, r1) = tile_img.to(torch::kCPU);

        done += (r1 - r0);
        if (progress) progress(done, H);
    }

    return result;
}

// ── Tile render ───────────────────────────────────────────────────────────────

torch::Tensor Raytracer::render_tile(int row_start, int row_end) const
{
    const float PI = static_cast<float>(M_PI);
    constexpr float EPS = 1e-5f;
    int H = cfg_.height, W = cfg_.width;
    int N = (row_end - row_start) * W;

    auto opts  = torch::TensorOptions().dtype(dtype_).device(device_);
    auto bopts = torch::TensorOptions().dtype(torch::kBool).device(device_);

    // ── Init camera rays for this tile ────────────────────────────────────────
    // Re-use full init_camera_rays then slice the tile rows
    auto [state_all, E_all, L_all] = init_camera_rays(
        W, H, cfg_.fov_deg,
        cfg_.observer_radius,
        cfg_.observer_inclination_deg,
        cfg_.observer_azimuth_deg,
        cfg_.observer_roll_deg,
        a_, cfg_.charge, opts);

    // Slice tile rows
    int start_idx = row_start * W;
    int end_idx   = row_end   * W;
    auto state = state_all.slice(0, start_idx, end_idx).contiguous();
    auto E     = E_all.slice(0, start_idx, end_idx).contiguous();
    auto L     = L_all.slice(0, start_idx, end_idx).contiguous();

    // ── Buffers ───────────────────────────────────────────────────────────────
    auto active      = torch::ones({N}, bopts);
    auto accumulated = torch::zeros({N, 3}, opts);
    auto disk_order  = torch::zeros({N}, opts);
    auto prev_theta  = state.select(1, 1).clone();

    float r_in   = (cfg_.disk_inner_radius > 0.f)
                   ? cfg_.disk_inner_radius : r_isco_;
    float r_out  = cfg_.disk_outer_radius;
    float r_esc  = cfg_.observer_radius * 3.0f;
    float h      = cfg_.step_size;
    float h_over6 = h / 6.0f;
    const int max_order = 5;  // max disk crossings per ray

    // ── Integration loop ──────────────────────────────────────────────────────
    for (int step = 0; step < cfg_.max_steps; ++step) {
        if (!active.any().item<bool>()) break;

        // RK4
        auto k1 = geodesic_rhs(state, E, L, a_, cfg_.charge);
        auto k2 = geodesic_rhs(state + (h * 0.5f) * k1, E, L, a_, cfg_.charge);
        auto k3 = geodesic_rhs(state + (h * 0.5f) * k2, E, L, a_, cfg_.charge);
        auto k4 = geodesic_rhs(state + h * k3, E, L, a_, cfg_.charge);
        auto new_state = state + h_over6 * (k1 + 2.0f * k2 + 2.0f * k3 + k4);

        // ── Disk crossing detection ────────────────────────────────────────
        auto theta_new = new_state.select(1, 1);
        auto half_pi   = torch::scalar_tensor(PI * 0.5f, opts);

        auto sign_old  = (prev_theta - PI * 0.5f).sign();
        auto sign_new  = (theta_new  - PI * 0.5f).sign();
        auto crossed   = (sign_old * sign_new < 0.0f) & active;

        if (crossed.any().item<bool>()) {
            // Linearly interpolate r at disk crossing
            auto r_old = state.select(1, 0);
            auto r_new = new_state.select(1, 0);
            auto t_frac = ((PI * 0.5f - prev_theta)
                         / (theta_new - prev_theta + EPS)).clamp(0.0f, 1.0f);
            auto r_cross = r_old + t_frac * (r_new - r_old);

            // φ at crossing
            auto phi_cross = state.select(1, 2)
                           + t_frac * (new_state.select(1, 2) - state.select(1, 2));

            // Check radial bounds and order cap
            auto in_disk = crossed
                & (r_cross >= r_in)
                & (r_cross <= r_out)
                & (disk_order < float(max_order));

            if (in_disk.any().item<bool>()) {
                auto emission = disk_emission(r_cross, phi_cross, E, L, in_disk, disk_order);
                accumulated = accumulated + emission;
                disk_order  = disk_order + in_disk.to(dtype_);
            }
        }

        // ── Advance ───────────────────────────────────────────────────────
        prev_theta = theta_new.clone();
        state = new_state;

        // ── Termination ───────────────────────────────────────────────────
        auto r_curr = state.select(1, 0);
        auto horizon_hit = (r_curr < r_horizon_ + 0.05f) & active;
        auto escaped     = (r_curr > r_esc)              & active;

        // Background for escaped rays
        if (escaped.any().item<bool>()) {
            auto bg = background_color(state, escaped);
            accumulated = accumulated + bg;
        }

        active = active & ~horizon_hit & ~escaped;
    }

    // ── Apply post-processing gain ─────────────────────────────────────────────
    accumulated = accumulated * cfg_.disk_emission_gain;

    // Reshape to tile image
    int tile_H = row_end - row_start;
    return accumulated.reshape({tile_H, W, 3}).to(torch::kFloat32);
}

// ── Disk emission ─────────────────────────────────────────────────────────────

torch::Tensor Raytracer::disk_emission(
    const torch::Tensor& r_cross,
    const torch::Tensor& phi_cross,
    const torch::Tensor& E,
    const torch::Tensor& L,
    const torch::Tensor& in_disk,
    const torch::Tensor& order) const
{
    constexpr float EPS = 1e-6f;
    auto opts = torch::TensorOptions().dtype(dtype_).device(device_);

    float r_in  = (cfg_.disk_inner_radius > 0.f) ? cfg_.disk_inner_radius : r_isco_;
    float r_out = cfg_.disk_outer_radius;

    // ── Relativistic factor ────────────────────────────────────────────────
    auto rel_gain = relativistic_factor(
        r_cross.clamp_min(r_in + EPS), E, L, a_, cfg_.disk_beaming_strength);
    // For photon rings (order > 0) intensity falls off
    auto order_gain = torch::exp(-order * 0.6f);

    torch::Tensor intensity, color;

    // ── Disk model ────────────────────────────────────────────────────────
    if (cfg_.disk_model == "riaf") {
        auto [inten, col] = riaf_emission(
            r_cross, r_in, r_out,
            cfg_.riaf_alpha_n,
            cfg_.riaf_alpha_T,
            cfg_.riaf_alpha_B,
            cfg_.riaf_T_visual,
            cfg_.riaf_color_mode);
        intensity = inten;
        color     = col;
    } else if (cfg_.disk_model == "physical_nt") {
        bool page_thorne = (cfg_.disk_radial_profile == "nt_page_thorne");
        auto flux = nt_flux_profile(r_cross, r_in, page_thorne);
        intensity = flux;

        // Blackbody colour from effective temperature
        // T ∝ F^{1/4} * T0, T0 tuned so peak ~10000K
        float T0 = 15000.0f;
        auto T_eff = T0 * torch::pow(flux.clamp_min(EPS), 0.25f);
        if (cfg_.disk_palette == "interstellar_warm") {
            float span = std::max(r_out - r_in, EPS);
            auto x = ((r_cross - r_in) / span).clamp(0.0f, 1.0f);
            color = palette_interstellar_warm(x);
        } else {
            color = blackbody_rgb(T_eff, opts);
        }
    } else {
        // legacy: simple power law r^{-3}
        auto x   = ((r_cross - r_in) / std::max(r_out - r_in, EPS)).clamp(0.0f, 1.0f);
        intensity = torch::pow(1.0f - x, 3.0f);
        color     = named_palette(cfg_.disk_palette, 1.0f - x);
    }

    // Optional segmented palette (rings/sectors) blended over base color.
    if (cfg_.disk_segmented_palette) {
        const float pi2 = 2.0f * static_cast<float>(M_PI);
        const int n_rings = std::max(1, cfg_.disk_segmented_rings);
        const int n_sectors = std::max(2, cfg_.disk_segmented_sectors);
        const float sigma = std::max(0.05f, cfg_.disk_segmented_sigma);
        const float hue_offset = cfg_.disk_segmented_hue_offset;
        const float span = std::max(r_out - r_in, EPS);

        auto x = ((r_cross - r_in) / span).clamp(0.0f, 1.0f - 1.0e-7f);
        auto phase = torch::remainder(phi_cross, pi2);
        auto r_float = x * static_cast<float>(n_rings);
        auto s_float = phase / pi2 * static_cast<float>(n_sectors);

        auto color_acc = torch::zeros_like(color);
        auto weight_acc = torch::zeros_like(r_cross);
        const int K = std::max(1, static_cast<int>(std::ceil(2.5f * sigma)));

        for (int dr = -K; dr <= K; ++dr) {
            for (int ds = -K; ds <= K; ++ds) {
                auto ring_j = torch::floor(r_float) + static_cast<float>(dr);
                auto sec_j = torch::floor(s_float) + static_cast<float>(ds);
                auto ring_center = ring_j + 0.5f;
                auto sec_center = sec_j + 0.5f;

                auto d_ring = r_float - ring_center;
                auto d_sec_raw = s_float - sec_center;
                auto d_sec = d_sec_raw - torch::round(d_sec_raw / static_cast<float>(n_sectors)) * static_cast<float>(n_sectors);

                auto w = torch::exp(-0.5f * (d_ring * d_ring + d_sec * d_sec) / (sigma * sigma));
                w = torch::where(
                    (ring_j >= 0.0f) & (ring_j < static_cast<float>(n_rings)),
                    w,
                    torch::zeros_like(w)
                );

                auto ring_idx = torch::clamp(ring_j, 0.0f, static_cast<float>(n_rings - 1));
                auto sec_idx = torch::remainder(sec_j, static_cast<float>(n_sectors));
                auto ring_norm = ring_idx / std::max(1.0f, static_cast<float>(n_rings - 1));

                torch::Tensor hue, sat, val;
                if (cfg_.disk_segmented_palette_mode == "rainbow") {
                    auto cell_id = ring_idx * static_cast<float>(n_sectors) + sec_idx;
                    auto rnd = fract_tensor(torch::sin(cell_id * 17.237f + 47.991f) * 29341.7712f);
                    hue = torch::remainder(hue_offset + sec_idx / static_cast<float>(n_sectors) + 0.14f * (rnd - 0.5f), 1.0f);
                    sat = torch::clamp(0.78f - 0.24f * ring_norm + 0.20f * (rnd - 0.5f), 0.18f, 1.0f);
                    val = torch::clamp(0.74f + 0.22f * ring_norm + 0.24f * (rnd - 0.5f), 0.18f, 1.0f);
                } else {
                    auto cell_id = ring_idx * static_cast<float>(n_sectors) + sec_idx;
                    auto rnd_a = fract_tensor(torch::sin(cell_id * 12.9898f + 78.233f) * 43758.5453f);
                    auto rnd_b = fract_tensor(torch::sin(cell_id * 19.9137f + 11.135f) * 24634.6345f);
                    auto rnd_c = fract_tensor(torch::sin(cell_id * 7.1231f + 93.531f) * 35412.2381f);
                    auto hotspot = torch::clamp((rnd_c - 0.70f) * 2.8f, 0.0f, 1.0f)
                                 * torch::clamp(1.25f - ring_norm, 0.0f, 1.0f);
                    hue = torch::remainder(hue_offset + 0.01f + 0.13f * rnd_a, 1.0f);
                    sat = torch::clamp(0.90f + 0.08f * ring_norm + 0.22f * (rnd_b - 0.5f) - 0.65f * hotspot, 0.08f, 1.0f);
                    val = torch::clamp(0.82f - 0.30f * ring_norm + 0.28f * (rnd_c - 0.5f) + 0.55f * hotspot, 0.18f, 1.0f);
                }

                auto cell_rgb = hsv_to_rgb(hue, sat, val);
                color_acc = color_acc + w.unsqueeze(1) * cell_rgb;
                weight_acc = weight_acc + w;
            }
        }

        auto seg_color = color_acc / torch::clamp(weight_acc, 1.0e-8f).unsqueeze(1);
        float seg_mix = std::clamp(cfg_.disk_segmented_mix, 0.0f, 1.0f);
        color = (1.0f - seg_mix) * color + seg_mix * seg_color;
    }

    // ── Inner / outer edge boost ───────────────────────────────────────────
    float span = std::max(r_out - r_in, EPS);
    auto x_norm = ((r_cross - r_in) / span).clamp(0.0f, 1.0f);
    auto inner_rim = torch::exp(-x_norm * 20.0f);        // sharp inner glow
    auto outer_rim = torch::exp(-(1.0f - x_norm) * 8.0f);

    auto edge_w = 0.5f
                + intensity
                + cfg_.inner_edge_boost * inner_rim
                + cfg_.outer_edge_boost * outer_rim;

    // ── Combine ───────────────────────────────────────────────────────────
    auto final_intensity = (intensity * edge_w * rel_gain * order_gain)
                           .clamp(0.0f, 100.0f)
                           .unsqueeze(1);

    auto emission = color * final_intensity;

    // Zero out non-disk rays
    emission = emission * in_disk.to(dtype_).unsqueeze(1);
    return emission;
}

// ── Background color ──────────────────────────────────────────────────────────

torch::Tensor Raytracer::background_color(
    const torch::Tensor& state,
    const torch::Tensor& escaped_mask) const
{
    auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
    int N = static_cast<int>(state.size(0));
    auto bg = torch::zeros({N, 3}, opts);

    if (cfg_.background_mode == "darkspace") {
        if (!cfg_.enable_star_background) {
            return bg * escaped_mask.to(dtype_).unsqueeze(1);
        }
        // Procedural star field: hash-based pseudo-random stars
        auto phi   = state.select(1, 2);          // φ coordinate
        auto theta = state.select(1, 1);          // θ coordinate

        // Map to 2D position in [0,1]²
        const float PI = static_cast<float>(M_PI);
        auto u = (phi / (2.0f * PI)).fmod(1.0f).abs();
        auto v = (theta / PI).clamp(0.0f, 1.0f);

        // Simple star density: probability proportional to star_density
        // Use sin(large_freq * phi + theta) as a deterministic "random" value
        auto star_val = 0.5f * (1.0f
            + torch::sin(u * 1237.0f + v * 3171.0f)
            * torch::cos(u * 2341.0f - v * 1723.0f));
        float thresh = 1.0f - cfg_.star_density * 0.001f;
        auto is_star = star_val > thresh;

        // Star brightness ~ exponential distribution
        auto brightness = torch::exp((star_val - 1.0f) / (1.0f - thresh + 1e-8f))
                        * cfg_.star_brightness;
        // White stars with slight colour variation
        auto R = (1.0f + 0.1f * torch::sin(u * 5431.0f)).clamp(0.5f, 1.0f) * brightness;
        auto G = (1.0f + 0.1f * torch::cos(v * 3271.0f)).clamp(0.5f, 1.0f) * brightness;
        auto B = (1.0f + 0.05f * torch::sin((u+v) * 7891.0f)).clamp(0.6f, 1.1f) * brightness;

        auto star_color = torch::stack({R, G, B}, 1);
        bg = torch::where(is_star.unsqueeze(1).expand_as(star_color), star_color, bg);
    }
    // hdri: TODO (load and sample equirectangular image)

    return bg * escaped_mask.to(dtype_).unsqueeze(1);
}

// ── Post-processing ───────────────────────────────────────────────────────────

torch::Tensor Raytracer::postprocess(const torch::Tensor& rgb) const
{
    if (cfg_.postprocess_pipeline == "gargantua"
        && cfg_.gargantua_look_strength > 0.0f) {
        return postprocess_gargantua(rgb, cfg_.gargantua_look_strength);
    }
    return rgb;
}

torch::Tensor Raytracer::postprocess_gargantua(
    const torch::Tensor& img, float strength) const
{
    // img: (H, W, 3)
    auto out = img.clone();

    // 1. Bloom: Gaussian blur on overbright regions, add back
    auto bright = (img - 0.8f).clamp_min(0.0f);
    // Simple 5×5 box blur approximation via unfold
    // For full quality use a proper separable Gaussian
    if (bright.max().item<float>() > 0.0f) {
        auto b = bright.permute({2, 0, 1}).unsqueeze(0); // (1,3,H,W)
        // 3 passes of box blur (approximates Gaussian)
        auto pad = torch::nn::functional::pad(b,
            torch::nn::functional::PadFuncOptions({2,2,2,2}).mode(torch::kReflect));
        auto blurred = torch::avg_pool2d(pad, {5,5}, {1,1});
        // Match spatial dims
        if (blurred.size(2) == b.size(2) && blurred.size(3) == b.size(3)) {
            auto bloom = blurred.squeeze(0).permute({1, 2, 0}) * strength * 2.0f;
            out = out + bloom;
        }
    }

    // 2. Warm colour grading: boost reds/oranges slightly
    auto R = out.select(2, 0);
    auto B = out.select(2, 2);
    out.select(2, 0) = (R * (1.0f + 0.08f * strength)).clamp_max(20.0f);
    out.select(2, 2) = (B * (1.0f - 0.06f * strength)).clamp_min(0.0f);

    // 3. ACES filmic tone-map
    // x*(x*2.51+0.03) / (x*(x*2.43+0.59)+0.14)
    auto x = out.clamp_min(0.0f);
    out = (x * (x * 2.51f + 0.03f)) / (x * (x * 2.43f + 0.59f) + 0.14f).clamp_min(1e-5f);
    out = out.clamp(0.0f, 1.0f);

    return out;
}

} // namespace kerrtrace
