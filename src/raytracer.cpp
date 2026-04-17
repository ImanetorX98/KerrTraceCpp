#include "kerrtrace/raytracer.hpp"
#include "kerrtrace/geometry.hpp"
#include "kerrtrace/image_io.hpp"
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

    if (cfg_.background_mode == "hdri" && !cfg_.hdri_path.empty()) {
        try {
            auto tex_opts = torch::TensorOptions().dtype(dtype_).device(device_);
            hdri_tex_ = load_png(cfg_.hdri_path, tex_opts).contiguous();
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load HDRI '" << cfg_.hdri_path
                      << "': " << e.what() << "\n";
        }
    }
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
            auto p_theta_cross = state.select(1, 5)
                               + t_frac * (new_state.select(1, 5) - state.select(1, 5));
            auto p_theta_abs = torch::abs(p_theta_cross);

            // Check radial bounds and order cap
            auto in_disk = crossed
                & (r_cross >= r_in)
                & (r_cross <= r_out)
                & (disk_order < float(max_order));

            if (in_disk.any().item<bool>()) {
                auto emission = disk_emission(r_cross, phi_cross, p_theta_abs, E, L, in_disk, disk_order);
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
    const torch::Tensor& p_theta_abs,
    const torch::Tensor& E,
    const torch::Tensor& L,
    const torch::Tensor& in_disk,
    const torch::Tensor& order) const
{
    constexpr float EPS = 1e-6f;
    auto opts = torch::TensorOptions().dtype(dtype_).device(device_);

    float r_in  = (cfg_.disk_inner_radius > 0.f) ? cfg_.disk_inner_radius : r_isco_;
    float r_out = cfg_.disk_outer_radius;

    // ── Relativistic factor (g^(3+b)) ─────────────────────────────────────
    auto rel_gain = relativistic_factor(
        r_cross.clamp_min(r_in + EPS), E, L, a_, cfg_.disk_beaming_strength);

    // Self-occlusion: grazing rays (small p_θ) are attenuated
    // Python: mu = p_θ / (p_θ + 0.20),  occlusion = (1-s) + s*sqrt(mu)
    {
        auto mu = p_theta_abs / (p_theta_abs + 0.20f).clamp_min(1.0e-6f);
        float s = cfg_.disk_self_occlusion_strength;
        auto occlusion = (1.0f - s) + s * torch::sqrt(mu.clamp(0.0f, 1.0f));
        rel_gain = rel_gain * occlusion;
    }

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

    // Optional layered palette blended over base color.
    if (cfg_.disk_layered_palette) {
        const float pi2 = 2.0f * static_cast<float>(M_PI);
        const int n_layers = std::max(2, cfg_.disk_layer_count);
        const float n_layers_f = static_cast<float>(n_layers);
        const float span_safe = std::max(r_out - r_in, EPS);
        auto x = ((r_cross - r_in) / span_safe).clamp(0.0f, 1.0f - 1.0e-7f);

        auto layer_pos = torch::clamp(x * n_layers_f, 0.0f, n_layers_f - 1.0e-6f);
        auto layer_idx = torch::floor(layer_pos);
        auto layer_idx_next = torch::minimum(layer_idx + 1.0f, torch::full_like(layer_idx, n_layers_f - 1.0f));
        auto layer_frac = torch::clamp(layer_pos - layer_idx, 0.0f, 1.0f);

        auto u0 = (layer_idx + 0.5f) / n_layers_f;
        auto u1 = (layer_idx_next + 0.5f) / n_layers_f;
        auto c0 = palette_interstellar_warm(u0);
        auto c1 = palette_interstellar_warm(u1);
        auto base_layer_color = c0 + (c1 - c0) * layer_frac.unsqueeze(1);

        auto flow_phi = phi_cross * cfg_.disk_layer_time_scale - cfg_.disk_layer_global_phase;
        auto tile_gain = torch::ones_like(flow_phi);
        if (cfg_.enable_disk_differential_rotation && cfg_.disk_diffrot_strength > 0.0f) {
            float strength = cfg_.disk_diffrot_strength;
            float seed = static_cast<float>(cfg_.disk_diffrot_seed);
            auto radial_key = u0;
            auto raw = torch::sin(
                (layer_idx + 1.0f) * 12.9898f
                + (radial_key + 1.0f) * 78.233f
                + seed * 37.719f
                + 0.5f
            ) * 43758.5453f;
            auto phase_offset = (fract_tensor(raw) - 0.5f) * pi2;
            flow_phi = torch::remainder(flow_phi + 0.36f * strength * phase_offset, pi2);

            if (cfg_.disk_diffrot_visual_mode != "layer_phase") {
                auto tile_drive = flow_phi * (1.0f + 0.15f * strength)
                                + (layer_idx + 1.0f) * 0.37f
                                + radial_key * 3.1f;
                auto tile_wave = 0.5f + 0.5f * torch::sin(tile_drive);
                if (cfg_.disk_diffrot_visual_mode == "hybrid") {
                    auto fine_wave = 0.5f + 0.5f * torch::sin(1.73f * tile_drive + 0.5f * flow_phi);
                    tile_wave = 0.62f * tile_wave + 0.38f * fine_wave;
                }
                tile_gain = torch::clamp(0.82f + 0.36f * tile_wave, 0.55f, 1.45f);
            }
        }

        auto phase = cfg_.disk_layer_pattern_count * flow_phi + pi2 * layer_idx / n_layers_f;
        auto wave = 0.5f + 0.5f * torch::sin(phase);
        auto wave2 = 0.5f + 0.5f * torch::sin(0.73f * phase + 8.0f * u0);
        auto contrast = torch::clamp(torch::scalar_tensor(cfg_.disk_layer_pattern_contrast, opts), 0.0f, 1.0f);
        auto band = (1.0f - contrast) + contrast * (0.28f + 0.72f * wave * (0.82f + 0.18f * wave2));
        band = band * tile_gain;

        if (cfg_.disk_layer_accident_strength > 0.0f) {
            auto accident_strength = torch::scalar_tensor(cfg_.disk_layer_accident_strength, opts);
            auto accident_count = torch::scalar_tensor(cfg_.disk_layer_accident_count, opts);
            auto accident_sharp = torch::scalar_tensor(cfg_.disk_layer_accident_sharpness, opts);
            auto inner_weight = torch::pow(torch::clamp(1.0f - u0, 0.0f, 1.0f), 0.35f);
            auto seg_count = torch::clamp(2.0f + 4.0f * accident_count, 2.0f, 512.0f);
            auto phi_unit = torch::remainder(flow_phi / pi2, 1.0f);
            auto seg_pos = phi_unit * seg_count;
            auto seg_idx = torch::floor(seg_pos);
            auto seg_frac = seg_pos - seg_idx;
            auto edge = torch::clamp(0.10f / torch::clamp_min(accident_sharp, 1.0f), 0.004f, 0.08f);
            auto duty = torch::clamp(0.78f - 0.055f * torch::clamp_min(accident_sharp - 1.0f, 0.0f), 0.14f, 0.88f);
            auto rise = torch::clamp(seg_frac / torch::clamp_min(edge, 1.0e-4f), 0.0f, 1.0f);
            auto fall = torch::clamp((duty - seg_frac) / torch::clamp_min(edge, 1.0e-4f), 0.0f, 1.0f);
            auto in_rect = torch::where(seg_frac < duty, torch::minimum(rise, fall), torch::zeros_like(seg_frac));
            auto seg_hash = fract_tensor(torch::sin((layer_idx + 1.0f) * 127.1f + (seg_idx + 1.0f) * 311.7f + 17.0f * u0) * 43758.5453f);
            auto accident_profile = torch::clamp(in_rect * (0.35f + 0.65f * seg_hash), 0.0f, 1.0f);
            auto accident_boost = torch::clamp(
                1.0f + accident_strength * inner_weight * (1.60f * accident_profile - 0.30f),
                0.45f,
                3.2f
            );
            band = band * accident_boost;
        }

        auto odd = torch::remainder(layer_idx, 2.0f);
        auto tint_shift = (odd * 2.0f - 1.0f) * 0.03f;
        auto tint = torch::stack({
            torch::clamp(1.0f + 0.5f * tint_shift, 0.85f, 1.15f),
            torch::clamp(1.0f - 0.2f * tint_shift, 0.85f, 1.15f),
            torch::clamp(1.0f - 0.8f * tint_shift, 0.80f, 1.20f)
        }, 1);
        auto layered_color = torch::clamp(base_layer_color * tint * band.unsqueeze(1), 0.0f, 1.5f);
        float layer_mix = std::clamp(cfg_.disk_layer_mix, 0.0f, 1.0f);
        color = (1.0f - layer_mix) * color + layer_mix * layered_color;
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

        auto seg_color = color_acc / torch::clamp_min(weight_acc, 1.0e-8f).unsqueeze(1);
        float seg_mix = std::clamp(cfg_.disk_segmented_mix, 0.0f, 1.0f);
        color = (1.0f - seg_mix) * color + seg_mix * seg_color;
    }

    // ── Edge weights — matches Python exactly ─────────────────────────────
    float span = std::max(r_out - r_in, EPS);
    auto x_norm = ((r_cross - r_in) / span).clamp(0.0f, 1.0f);

    // Gaussian rim glows (Python: exp(-((r-r_in)/width)^2))
    float inner_width = 0.05f * span + 1.0e-3f;
    float outer_width = 0.10f * span + 1.0e-3f;
    auto dr_inner = (r_cross - r_in) / inner_width;
    auto dr_outer = (r_out - r_cross) / outer_width;
    auto inner_rim = torch::exp(-(dr_inner * dr_inner));
    auto outer_rim = torch::exp(-(dr_outer * dr_outer));

    // body = pow(1-x, 0.35) — smooth radial weight (Python)
    auto body = torch::pow((1.0f - x_norm).clamp(0.0f, 1.0f), 0.35f);

    auto edge_w = 0.22f + body
                + cfg_.inner_edge_boost * inner_rim
                + cfg_.outer_edge_boost * outer_rim;

    // Optional disk volume emission factor.
    auto volume_factor = torch::ones_like(intensity);
    if (cfg_.disk_volume_emission && cfg_.disk_volume_strength > 0.0f) {
        auto h_eff = 0.015f * torch::clamp_min(r_cross, 1.0f);
        auto mu_los = p_theta_abs / torch::clamp_min(p_theta_abs + 0.10f, 1.0e-6f);
        auto path_len = (2.0f * h_eff) / torch::clamp_min(mu_los, 0.04f);
        auto tau = cfg_.disk_volume_density_scale * path_len / torch::clamp_min(r_cross, 1.0e-6f);
        auto trans = 1.0f - torch::exp(-torch::clamp(tau, 0.0f, 40.0f));

        int ns = std::max(1, cfg_.disk_volume_samples);
        auto temp_factor = torch::ones_like(intensity);
        if (ns > 1) {
            auto zeta = torch::linspace(-1.0f, 1.0f, ns, opts).view({1, ns});
            auto sigma = torch::scalar_tensor(0.48f, opts);
            auto rho = torch::exp(-0.5f * torch::square(zeta / sigma));
            auto temp_mod = 1.0f - cfg_.disk_volume_temperature_drop * torch::abs(zeta);
            auto angle_mod = 0.85f + 0.15f * mu_los.unsqueeze(1);
            auto kernel = rho * temp_mod * angle_mod;
            auto denom = torch::clamp(torch::sum(rho, 1, true), 1.0e-8f);
            temp_factor = torch::sum(kernel, 1) / denom.squeeze(1);
        }
        volume_factor = 1.0f + cfg_.disk_volume_strength * trans * torch::clamp_min(temp_factor, 0.0f);
    }

    // ── Combine ───────────────────────────────────────────────────────────
    auto final_intensity = (intensity * edge_w * rel_gain * order_gain * volume_factor)
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
    using namespace torch::indexing;
    auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
    int N = static_cast<int>(state.size(0));
    auto bg = torch::zeros({N, 3}, opts);

    if (cfg_.background_mode == "hdri") {
        if (hdri_tex_.defined() && hdri_tex_.numel() > 0) {
            const float PI = static_cast<float>(M_PI);
            auto phi = state.select(1, 2);
            auto theta = state.select(1, 1);
            auto rot = cfg_.hdri_rotation_deg * (PI / 180.0f);

            auto u = torch::remainder(phi + rot, 2.0f * PI) / (2.0f * PI);
            auto v = (theta / PI).clamp(0.0f, 1.0f);

            int tex_h = static_cast<int>(hdri_tex_.size(0));
            int tex_w = static_cast<int>(hdri_tex_.size(1));

            auto x = torch::clamp(u, 0.0f, 1.0f) * static_cast<float>(tex_w - 1);
            auto y = torch::clamp(v, 0.0f, 1.0f) * static_cast<float>(tex_h - 1);

            auto x0 = torch::floor(x).to(torch::kLong);
            auto y0 = torch::floor(y).to(torch::kLong);
            auto x1 = torch::remainder(x0 + 1, tex_w);
            auto y1 = torch::clamp(y0 + 1, 0, tex_h - 1);

            auto tx = (x - x0.to(dtype_)).unsqueeze(1);
            auto ty = (y - y0.to(dtype_)).unsqueeze(1);

            auto c00 = hdri_tex_.index({y0, x0});
            auto c10 = hdri_tex_.index({y0, x1});
            auto c01 = hdri_tex_.index({y1, x0});
            auto c11 = hdri_tex_.index({y1, x1});

            auto c0 = c00 * (1.0f - tx) + c10 * tx;
            auto c1 = c01 * (1.0f - tx) + c11 * tx;
            bg = (c0 * (1.0f - ty) + c1 * ty) * cfg_.hdri_exposure;
        }
    } else if (cfg_.background_mode == "darkspace") {
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
