#include "kerrtrace/config.hpp"
#include "kerrtrace/raytracer.hpp"
#include "kerrtrace/animation.hpp"
#include "kerrtrace/image_io.hpp"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>

using namespace kerrtrace;

int main(int argc, char** argv)
{
    CLI::App app{"KerrTrace C++ — Kerr black hole ray tracer", "kerrtrace"};
    app.set_version_flag("--version", "1.2.0");

    RenderConfig cfg;
    std::string config_file;
    bool enable_gargantua = false;
    bool disable_gargantua = false;

    // ── Config file ────────────────────────────────────────────────────────────
    app.add_option("--config", config_file, "JSON config file");

    // ── Resolution & Camera ────────────────────────────────────────────────────
    app.add_option("--width",  cfg.width,  "Image width (px)");
    app.add_option("--height", cfg.height, "Image height (px)");
    app.add_option("--fov",    cfg.fov_deg,"Field of view (deg)");

    // ── Spacetime ─────────────────────────────────────────────────────────────
    app.add_option("--metric-model",       cfg.metric_model,      "Metric: kerr|kerr_newman|schwarzschild");
    app.add_option("--coordinate-system",  cfg.coordinate_system, "Coordinates: kerr_schild|boyer_lindquist");
    app.add_option("--spin",               cfg.spin,              "Black hole spin a ∈ [-1,1]");
    app.add_option("--charge",             cfg.charge,            "Black hole charge Q ∈ [0,1]");

    // ── Observer ──────────────────────────────────────────────────────────────
    app.add_option("--observer-radius",        cfg.observer_radius,           "Observer distance (M)");
    app.add_option("--observer-inclination",   cfg.observer_inclination_deg,  "Inclination angle (deg)");
    app.add_option("--observer-azimuth",       cfg.observer_azimuth_deg,      "Azimuth angle (deg)");
    app.add_option("--observer-roll",          cfg.observer_roll_deg,         "Camera roll (deg)");

    // ── Disk ──────────────────────────────────────────────────────────────────
    app.add_option("--disk-model",          cfg.disk_model,          "Disk: physical_nt|legacy|riaf");
    app.add_option("--disk-inner-radius",   cfg.disk_inner_radius,   "Inner disk radius (<0=ISCO)");
    app.add_option("--disk-outer-radius",   cfg.disk_outer_radius,   "Outer disk radius (M)");
    app.add_option("--disk-emission-gain",  cfg.disk_emission_gain,  "Emission brightness multiplier");
    app.add_option("--disk-palette",        cfg.disk_palette,        "Palette: default|interstellar_warm");
    app.add_option("--disk-radial-profile", cfg.disk_radial_profile, "Profile: nt_page_thorne|nt_proxy");
    app.add_option("--inner-edge-boost",    cfg.inner_edge_boost,    "Inner rim brightness boost");
    app.add_option("--outer-edge-boost",    cfg.outer_edge_boost,    "Outer rim brightness boost");
    app.add_option("--disk-beaming-strength", cfg.disk_beaming_strength, "Relativistic beaming exponent");
    app.add_flag  ("--enable-disk-layered-palette", cfg.disk_layered_palette, "Enable layered disk palette");
    app.add_option("--disk-layer-count", cfg.disk_layer_count, "Layered palette layer count");
    app.add_option("--disk-layer-mix", cfg.disk_layer_mix, "Layered palette blend [0,1]");
    app.add_option("--disk-layer-pattern-count", cfg.disk_layer_pattern_count, "Layer pattern frequency");
    app.add_option("--disk-layer-pattern-contrast", cfg.disk_layer_pattern_contrast, "Layer pattern contrast [0,1]");
    app.add_option("--disk-layer-time-scale", cfg.disk_layer_time_scale, "Layer flow time scale");
    app.add_option("--disk-layer-global-phase", cfg.disk_layer_global_phase, "Layer global phase offset (rad)");
    app.add_option("--disk-layer-accident-strength", cfg.disk_layer_accident_strength, "Layer inhomogeneity strength");
    app.add_option("--disk-layer-accident-count", cfg.disk_layer_accident_count, "Layer inhomogeneity density");
    app.add_option("--disk-layer-accident-sharpness", cfg.disk_layer_accident_sharpness, "Layer inhomogeneity sharpness");
    app.add_flag  ("--enable-disk-differential-rotation", cfg.enable_disk_differential_rotation, "Enable differential rotation visual modulation");
    app.add_option("--disk-diffrot-visual-mode", cfg.disk_diffrot_visual_mode, "Differential rotation visual mode: layer_phase|annular_tiles|hybrid");
    app.add_option("--disk-diffrot-strength", cfg.disk_diffrot_strength, "Differential rotation visual strength");
    app.add_option("--disk-diffrot-seed", cfg.disk_diffrot_seed, "Differential rotation random seed");
    app.add_flag  ("--enable-disk-segmented-palette",  cfg.disk_segmented_palette, "Enable segmented disk palette");
    app.add_option("--disk-segmented-rings",   cfg.disk_segmented_rings,   "Segmented palette rings");
    app.add_option("--disk-segmented-sectors", cfg.disk_segmented_sectors, "Segmented palette sectors");
    app.add_option("--disk-segmented-sigma",   cfg.disk_segmented_sigma,   "Segmented palette Gaussian smoothing");
    app.add_option("--disk-segmented-mix",     cfg.disk_segmented_mix,     "Segmented palette blend [0,1]");
    app.add_option("--disk-segmented-hue-offset", cfg.disk_segmented_hue_offset, "Segmented palette hue offset [0,1]");
    app.add_option("--disk-segmented-palette-mode", cfg.disk_segmented_palette_mode, "Segmented palette mode: accretion_warm|rainbow");

    // ── RIAF ──────────────────────────────────────────────────────────────────
    app.add_option("--riaf-alpha-n",   cfg.riaf_alpha_n,    "RIAF n_e power-law index");
    app.add_option("--riaf-alpha-t",   cfg.riaf_alpha_T,    "RIAF T_e power-law index");
    app.add_option("--riaf-alpha-b",   cfg.riaf_alpha_B,    "RIAF B power-law index");
    app.add_option("--riaf-t-visual",  cfg.riaf_T_visual,   "RIAF visual temp at ISCO (K)");
    app.add_option("--riaf-color-mode",cfg.riaf_color_mode, "RIAF colour: blackbody|plasma|interstellar_warm|gargantua");
    app.add_flag  ("--enable-disk-volume-emission", cfg.disk_volume_emission, "Enable disk volume emission boost");
    app.add_option("--disk-volume-samples", cfg.disk_volume_samples, "Vertical samples for volume emission");
    app.add_option("--disk-volume-density-scale", cfg.disk_volume_density_scale, "Volume density scale");
    app.add_option("--disk-volume-temperature-drop", cfg.disk_volume_temperature_drop, "Vertical temperature drop [0,1]");
    app.add_option("--disk-volume-strength", cfg.disk_volume_strength, "Volume emission strength");

    // ── Integration ───────────────────────────────────────────────────────────
    app.add_option("--max-steps",  cfg.max_steps,  "Max geodesic integration steps");
    app.add_option("--step-size",  cfg.step_size,  "Integration step size λ");

    // ── Compute ───────────────────────────────────────────────────────────────
    app.add_option("--device",          cfg.device,         "Device: auto|cpu|cuda|mps");
    app.add_option("--dtype",           cfg.dtype,          "Precision: float32|float64");
    app.add_option("--render-tile-rows",cfg.render_tile_rows,"Rows per tile (memory control)");

    // ── Background ────────────────────────────────────────────────────────────
    app.add_option("--background-mode", cfg.background_mode,       "darkspace|hdri|procedural");
    app.add_option("--hdri-path",       cfg.hdri_path,             "Path to HDRI panorama");
    app.add_option("--hdri-exposure",   cfg.hdri_exposure,         "HDRI exposure multiplier");
    app.add_option("--hdri-rotation-deg", cfg.hdri_rotation_deg,   "HDRI azimuth rotation (deg)");
    app.add_option("--star-density",    cfg.star_density,          "Star field density");
    app.add_option("--star-brightness", cfg.star_brightness,       "Star brightness");

    // ── Post-processing ───────────────────────────────────────────────────────
    app.add_option("--postprocess-pipeline",   cfg.postprocess_pipeline,   "off|gargantua");
    app.add_option("--gargantua-look-strength",cfg.gargantua_look_strength,"Gargantua bloom strength");
    app.add_flag  ("--enable-gargantua-look",  enable_gargantua,           "Apply gargantua preset");
    app.add_flag  ("--disable-gargantua-look", disable_gargantua,          "Disable gargantua preset");

    // ── Output ────────────────────────────────────────────────────────────────
    app.add_option("--output,-o", cfg.output, "Output PNG path");

    // ── Animation ─────────────────────────────────────────────────────────────
    app.add_flag  ("--animate",                cfg.animate,              "Render animation");
    app.add_option("--animation-duration",     cfg.animation_duration,   "Duration (s)");
    app.add_option("--animation-fps",          cfg.animation_fps,        "Frames per second");
    app.add_option("--animation-parameter",    cfg.animation_parameter,  "Parameter to animate");
    app.add_option("--animation-start",        cfg.animation_start,      "Parameter start value");
    app.add_option("--animation-end",          cfg.animation_end,        "Parameter end value");
    app.add_option("--animation-workers",      cfg.animation_workers,    "Parallel render workers");

    CLI11_PARSE(app, argc, argv);

    // ── Load config file (overridden by explicit CLI flags) ───────────────────
    if (!config_file.empty()) {
        std::ifstream f(config_file);
        if (!f) {
            std::cerr << "Cannot open config file: " << config_file << "\n";
            return 1;
        }
        nlohmann::json j;
        f >> j;
        from_json(j, cfg);
        // Re-parse so CLI flags override file
        CLI11_PARSE(app, argc, argv);
    }

    // ── Apply gargantua preset ─────────────────────────────────────────────────
    if (enable_gargantua)  cfg = apply_gargantua_preset(cfg);

    // ── Validate ──────────────────────────────────────────────────────────────
    try {
        validate(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    // ── Print summary ─────────────────────────────────────────────────────────
    std::cout << "KerrTrace C++  v1.2.0\n"
              << "  Metric:    " << cfg.metric_model << " (a=" << cfg.spin << ")\n"
              << "  Disk:      " << cfg.disk_model << "\n"
              << "  Observer:  r=" << cfg.observer_radius
                                   << " θ=" << cfg.observer_inclination_deg << "°\n"
              << "  Device:    " << cfg.device << "\n"
              << "  Output:    " << cfg.output << "\n";
    if (cfg.animate) {
        int nframes = static_cast<int>(
            std::round(cfg.animation_duration * cfg.animation_fps));
        std::cout << "  Animation: " << nframes << " frames @ "
                  << cfg.animation_fps << " fps\n";
    }
    std::cout << std::flush;

    auto t0 = std::chrono::steady_clock::now();

    if (!cfg.animate) {
        // ── Single frame ──────────────────────────────────────────────────────
        Raytracer rt(cfg);

        int rows_done = 0;
        auto progress = [&](int done, int total) {
            float pct = 100.0f * done / total;
            // Simple progress bar
            int bar_w = 40;
            int filled = static_cast<int>(pct / 100.0f * bar_w);
            std::cout << "\r  [";
            for (int i = 0; i < bar_w; ++i)
                std::cout << (i < filled ? '#' : ' ');
            std::cout << "] " << std::fixed << std::setprecision(1) << pct << "%  ";
            std::cout.flush();
        };

        auto img = rt.render(progress);
        img = rt.postprocess(img);
        std::cout << "\n";

        save_png(img, cfg.output);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        int H = img.size(0), W = img.size(1);
        long total_rays = static_cast<long>(H) * W;
        std::cout << "  Saved: " << cfg.output << "\n"
                  << "  Rays:  " << total_rays << "\n"
                  << "  Time:  " << std::fixed << std::setprecision(2)
                  << elapsed << "s\n";

    } else {
        // ── Animation ─────────────────────────────────────────────────────────
        std::filesystem::path out_path(cfg.output);
        auto stem = out_path.stem().string();
        auto frames_dir = out_path.parent_path() / (stem + "_frames");

        int total_frames = static_cast<int>(
            std::round(cfg.animation_duration * cfg.animation_fps));
        std::cout << "  Frames dir: " << frames_dir << "\n";

        render_animation(cfg, frames_dir,
            [&](int done, int total) {
                std::cout << "\r  Frame " << done << "/" << total << "  ";
                std::cout.flush();
            });
        std::cout << "\n";

        // Encode video
        std::cout << "  Encoding video → " << cfg.output << " ...\n";
        bool ok = encode_video(frames_dir, cfg.output, cfg.animation_fps);
        if (!ok) {
            std::cerr << "  Warning: ffmpeg encoding failed. "
                         "Frames saved to: " << frames_dir << "\n";
        } else {
            std::cout << "  Video saved: " << cfg.output << "\n";
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "  Total time: " << elapsed << "s\n";
    }

    return 0;
}
