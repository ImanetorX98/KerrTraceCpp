#include "kerrtrace/config.hpp"
#include <cmath>
#include <stdexcept>

namespace kerrtrace {

// ── ISCO ──────────────────────────────────────────────────────────────────────

float isco_radius(float spin)
{
    float a = std::max(-0.999f, std::min(0.999f, spin));
    float a2 = a * a;
    float z1 = 1.0f + std::cbrt(1.0f - a2)
             * (std::cbrt(1.0f + a) + std::cbrt(1.0f - a));
    float z2 = std::sqrt(3.0f * a2 + z1 * z1);
    float sign = (a >= 0.0f) ? 1.0f : -1.0f;
    return 3.0f + z2 - sign * std::sqrt((3.0f - z1) * (3.0f + z1 + 2.0f * z2));
}

float event_horizon_radius(float spin, float charge)
{
    // r_+ = 1 + sqrt(1 - a² - Q²)
    float inner = 1.0f - spin * spin - charge * charge;
    if (inner < 0.0f) inner = 0.0f;
    return 1.0f + std::sqrt(inner);
}

// ── Gargantua preset ──────────────────────────────────────────────────────────

RenderConfig apply_gargantua_preset(RenderConfig cfg)
{
    // Mirror the Python gargantua_look_preset logic
    auto set_if_default = [](auto& field, const auto& dflt, const auto& val) {
        if (field == dflt) field = val;
    };

    // Disk palette → interstellar_warm if still default
    set_if_default(cfg.disk_palette, std::string("default"), std::string("interstellar_warm"));
    // Force gargantua post-process
    set_if_default(cfg.postprocess_pipeline, std::string("off"), std::string("gargantua"));
    set_if_default(cfg.gargantua_look_strength, 0.0f, 1.0f);
    // High spin
    set_if_default(cfg.spin, 0.85f, 0.998f);
    // Beaming
    set_if_default(cfg.disk_beaming_strength, 1.0f, 1.5f);

    return cfg;
}

// ── Validation ────────────────────────────────────────────────────────────────

void validate(const RenderConfig& cfg)
{
    if (cfg.width  < 8 || cfg.width  > 16384) throw std::invalid_argument("width out of range");
    if (cfg.height < 8 || cfg.height > 16384) throw std::invalid_argument("height out of range");
    if (cfg.fov_deg <= 0.0f || cfg.fov_deg >= 180.0f)
        throw std::invalid_argument("fov_deg must be in (0, 180)");
    if (std::abs(cfg.spin) > 1.0f)
        throw std::invalid_argument("spin must be in [-1, 1]");
    if (cfg.spin * cfg.spin + cfg.charge * cfg.charge > 1.0f)
        throw std::invalid_argument("spin²+charge² must be <= 1");
    if (cfg.observer_radius < 2.0f)
        throw std::invalid_argument("observer_radius must be >= 2M");
    if (cfg.observer_inclination_deg < 0.0f || cfg.observer_inclination_deg > 180.0f)
        throw std::invalid_argument("observer_inclination_deg must be in [0, 180]");
    if (cfg.disk_inner_radius > 0.0f && cfg.disk_inner_radius >= cfg.disk_outer_radius)
        throw std::invalid_argument("disk_inner_radius must be < disk_outer_radius");
    if (cfg.disk_emission_gain <= 0.0f)
        throw std::invalid_argument("disk_emission_gain must be > 0");
    if (cfg.disk_segmented_rings < 1 || cfg.disk_segmented_rings > 256)
        throw std::invalid_argument("disk_segmented_rings must be in [1, 256]");
    if (cfg.disk_segmented_sectors < 2 || cfg.disk_segmented_sectors > 1024)
        throw std::invalid_argument("disk_segmented_sectors must be in [2, 1024]");
    if (cfg.disk_segmented_sigma <= 0.0f || cfg.disk_segmented_sigma > 4.0f)
        throw std::invalid_argument("disk_segmented_sigma must be in (0, 4]");
    if (cfg.disk_segmented_mix < 0.0f || cfg.disk_segmented_mix > 1.0f)
        throw std::invalid_argument("disk_segmented_mix must be in [0, 1]");
    if (cfg.disk_segmented_palette_mode != "accretion_warm" && cfg.disk_segmented_palette_mode != "rainbow")
        throw std::invalid_argument("disk_segmented_palette_mode must be 'accretion_warm' or 'rainbow'");
    if (cfg.disk_layer_count < 2 || cfg.disk_layer_count > 512)
        throw std::invalid_argument("disk_layer_count must be in [2, 512]");
    if (cfg.disk_layer_mix < 0.0f || cfg.disk_layer_mix > 1.0f)
        throw std::invalid_argument("disk_layer_mix must be in [0, 1]");
    if (cfg.disk_layer_pattern_count <= 0.0f || cfg.disk_layer_pattern_count > 1024.0f)
        throw std::invalid_argument("disk_layer_pattern_count must be in (0, 1024]");
    if (cfg.disk_layer_pattern_contrast < 0.0f || cfg.disk_layer_pattern_contrast > 1.0f)
        throw std::invalid_argument("disk_layer_pattern_contrast must be in [0, 1]");
    if (cfg.disk_layer_time_scale <= 0.0f || cfg.disk_layer_time_scale > 64.0f)
        throw std::invalid_argument("disk_layer_time_scale must be in (0, 64]");
    if (cfg.disk_layer_accident_strength < 0.0f || cfg.disk_layer_accident_strength > 4.0f)
        throw std::invalid_argument("disk_layer_accident_strength must be in [0, 4]");
    if (cfg.disk_layer_accident_count < 0.0f || cfg.disk_layer_accident_count > 128.0f)
        throw std::invalid_argument("disk_layer_accident_count must be in [0, 128]");
    if (cfg.disk_layer_accident_sharpness < 1.0f || cfg.disk_layer_accident_sharpness > 32.0f)
        throw std::invalid_argument("disk_layer_accident_sharpness must be in [1, 32]");
    if (cfg.disk_diffrot_visual_mode != "layer_phase"
        && cfg.disk_diffrot_visual_mode != "annular_tiles"
        && cfg.disk_diffrot_visual_mode != "hybrid")
        throw std::invalid_argument("disk_diffrot_visual_mode must be 'layer_phase', 'annular_tiles', or 'hybrid'");
    if (cfg.disk_diffrot_strength < 0.0f || cfg.disk_diffrot_strength > 3.0f)
        throw std::invalid_argument("disk_diffrot_strength must be in [0, 3]");
    if (cfg.disk_diffrot_seed < 0)
        throw std::invalid_argument("disk_diffrot_seed must be >= 0");
    if (cfg.disk_volume_samples < 1 || cfg.disk_volume_samples > 64)
        throw std::invalid_argument("disk_volume_samples must be in [1, 64]");
    if (cfg.disk_volume_density_scale < 0.0f || cfg.disk_volume_density_scale > 1000.0f)
        throw std::invalid_argument("disk_volume_density_scale must be in [0, 1000]");
    if (cfg.disk_volume_temperature_drop < 0.0f || cfg.disk_volume_temperature_drop > 1.0f)
        throw std::invalid_argument("disk_volume_temperature_drop must be in [0, 1]");
    if (cfg.disk_volume_strength < 0.0f || cfg.disk_volume_strength > 10.0f)
        throw std::invalid_argument("disk_volume_strength must be in [0, 10]");
    if (cfg.max_steps < 10 || cfg.max_steps > 100000)
        throw std::invalid_argument("max_steps must be in [10, 100000]");
    if (cfg.step_size <= 0.0f)
        throw std::invalid_argument("step_size must be > 0");
    if (cfg.background_mode == "hdri" && cfg.hdri_path.empty())
        throw std::invalid_argument("hdri_path is required when background_mode='hdri'");
    if (cfg.hdri_exposure <= 0.0f)
        throw std::invalid_argument("hdri_exposure must be > 0");
    if (cfg.video_codec != "h264" && cfg.video_codec != "h265")
        throw std::invalid_argument("video_codec must be 'h264' or 'h265'");

    // RIAF
    if (cfg.disk_model == "riaf") {
        if (cfg.riaf_alpha_n <= 0.0f || cfg.riaf_alpha_n > 4.0f)
            throw std::invalid_argument("riaf_alpha_n must be in (0, 4]");
        if (cfg.riaf_alpha_T <= 0.0f || cfg.riaf_alpha_T > 4.0f)
            throw std::invalid_argument("riaf_alpha_T must be in (0, 4]");
        if (cfg.riaf_alpha_B <= 0.0f || cfg.riaf_alpha_B > 4.0f)
            throw std::invalid_argument("riaf_alpha_B must be in (0, 4]");
        if (cfg.riaf_T_visual < 1000.0f || cfg.riaf_T_visual > 100000.0f)
            throw std::invalid_argument("riaf_T_visual must be in [1000, 100000] K");
        static const std::array<std::string, 4> riaf_modes{
            "blackbody","plasma","interstellar_warm","gargantua"};
        if (std::find(riaf_modes.begin(), riaf_modes.end(), cfg.riaf_color_mode)
            == riaf_modes.end())
            throw std::invalid_argument("unknown riaf_color_mode: " + cfg.riaf_color_mode);
    }
}

// ── JSON serialisation ────────────────────────────────────────────────────────

void to_json(nlohmann::json& j, const RenderConfig& c)
{
    j = {
        {"width",                   c.width},
        {"height",                  c.height},
        {"fov_deg",                 c.fov_deg},
        {"metric_model",            c.metric_model},
        {"coordinate_system",       c.coordinate_system},
        {"spin",                    c.spin},
        {"charge",                  c.charge},
        {"observer_radius",         c.observer_radius},
        {"observer_inclination_deg",c.observer_inclination_deg},
        {"observer_azimuth_deg",    c.observer_azimuth_deg},
        {"observer_roll_deg",       c.observer_roll_deg},
        {"disk_model",              c.disk_model},
        {"disk_inner_radius",       c.disk_inner_radius},
        {"disk_outer_radius",       c.disk_outer_radius},
        {"disk_emission_gain",      c.disk_emission_gain},
        {"disk_palette",            c.disk_palette},
        {"disk_radial_profile",     c.disk_radial_profile},
        {"inner_edge_boost",        c.inner_edge_boost},
        {"outer_edge_boost",        c.outer_edge_boost},
        {"disk_beaming_strength",   c.disk_beaming_strength},
        {"riaf_alpha_n",            c.riaf_alpha_n},
        {"riaf_alpha_T",            c.riaf_alpha_T},
        {"riaf_alpha_B",            c.riaf_alpha_B},
        {"riaf_T_visual",           c.riaf_T_visual},
        {"riaf_color_mode",         c.riaf_color_mode},
        {"disk_layered_palette",    c.disk_layered_palette},
        {"disk_layer_count",        c.disk_layer_count},
        {"disk_layer_mix",          c.disk_layer_mix},
        {"disk_layer_pattern_count", c.disk_layer_pattern_count},
        {"disk_layer_pattern_contrast", c.disk_layer_pattern_contrast},
        {"disk_layer_time_scale",   c.disk_layer_time_scale},
        {"disk_layer_accident_strength", c.disk_layer_accident_strength},
        {"disk_layer_accident_count", c.disk_layer_accident_count},
        {"disk_layer_accident_sharpness", c.disk_layer_accident_sharpness},
        {"disk_layer_global_phase", c.disk_layer_global_phase},
        {"disk_layer_phase_rate_hz", c.disk_layer_phase_rate_hz},
        {"enable_disk_differential_rotation", c.enable_disk_differential_rotation},
        {"disk_diffrot_visual_mode", c.disk_diffrot_visual_mode},
        {"disk_diffrot_strength", c.disk_diffrot_strength},
        {"disk_diffrot_seed", c.disk_diffrot_seed},
        {"disk_segmented_palette",  c.disk_segmented_palette},
        {"disk_segmented_rings",    c.disk_segmented_rings},
        {"disk_segmented_sectors",  c.disk_segmented_sectors},
        {"disk_segmented_sigma",    c.disk_segmented_sigma},
        {"disk_segmented_mix",      c.disk_segmented_mix},
        {"disk_segmented_hue_offset", c.disk_segmented_hue_offset},
        {"disk_segmented_palette_mode", c.disk_segmented_palette_mode},
        {"disk_volume_emission", c.disk_volume_emission},
        {"disk_volume_samples", c.disk_volume_samples},
        {"disk_volume_density_scale", c.disk_volume_density_scale},
        {"disk_volume_temperature_drop", c.disk_volume_temperature_drop},
        {"disk_volume_strength", c.disk_volume_strength},
        {"max_steps",               c.max_steps},
        {"step_size",               c.step_size},
        {"adaptive_integrator",     c.adaptive_integrator},
        {"device",                  c.device},
        {"dtype",                   c.dtype},
        {"render_tile_rows",        c.render_tile_rows},
        {"background_mode",         c.background_mode},
        {"enable_star_background",  c.enable_star_background},
        {"star_density",            c.star_density},
        {"star_brightness",         c.star_brightness},
        {"hdri_path",               c.hdri_path},
        {"hdri_exposure",           c.hdri_exposure},
        {"hdri_rotation_deg",       c.hdri_rotation_deg},
        {"postprocess_pipeline",    c.postprocess_pipeline},
        {"gargantua_look_strength", c.gargantua_look_strength},
        {"gargantua_look_preset",   c.gargantua_look_preset},
        {"output",                  c.output},
        {"keep_frames",             c.keep_frames},
        {"resume_frames",           c.resume_frames},
        {"frames_dir",              c.frames_dir},
        {"video_codec",             c.video_codec},
        {"progress_file",           c.progress_file},
        {"animate",                 c.animate},
        {"animation_duration",      c.animation_duration},
        {"animation_fps",           c.animation_fps},
        {"animation_parameter",     c.animation_parameter},
        {"animation_start",         c.animation_start},
        {"animation_end",           c.animation_end},
        {"animation_workers",       c.animation_workers},
    };
}

#define JG(key) if (j.contains(#key)) c.key = j[#key].get<decltype(c.key)>()

void from_json(const nlohmann::json& j, RenderConfig& c)
{
    JG(width); JG(height); JG(fov_deg);
    JG(metric_model); JG(coordinate_system);
    JG(spin); JG(charge);
    JG(observer_radius); JG(observer_inclination_deg);
    JG(observer_azimuth_deg); JG(observer_roll_deg);
    JG(disk_model); JG(disk_inner_radius); JG(disk_outer_radius);
    JG(disk_emission_gain); JG(disk_palette); JG(disk_radial_profile);
    JG(inner_edge_boost); JG(outer_edge_boost); JG(disk_beaming_strength);
    JG(riaf_alpha_n); JG(riaf_alpha_T); JG(riaf_alpha_B);
    JG(riaf_T_visual); JG(riaf_color_mode);
    JG(disk_layered_palette); JG(disk_layer_count); JG(disk_layer_mix);
    JG(disk_layer_pattern_count); JG(disk_layer_pattern_contrast); JG(disk_layer_time_scale);
    JG(disk_layer_accident_strength); JG(disk_layer_accident_count); JG(disk_layer_accident_sharpness);
    JG(disk_layer_global_phase); JG(disk_layer_phase_rate_hz);
    JG(enable_disk_differential_rotation); JG(disk_diffrot_visual_mode);
    JG(disk_diffrot_strength); JG(disk_diffrot_seed);
    JG(disk_segmented_palette); JG(disk_segmented_rings); JG(disk_segmented_sectors);
    JG(disk_segmented_sigma); JG(disk_segmented_mix); JG(disk_segmented_hue_offset);
    JG(disk_segmented_palette_mode);
    JG(disk_volume_emission); JG(disk_volume_samples); JG(disk_volume_density_scale);
    JG(disk_volume_temperature_drop); JG(disk_volume_strength);
    JG(max_steps); JG(step_size); JG(adaptive_integrator);
    JG(device); JG(dtype); JG(render_tile_rows);
    JG(background_mode); JG(enable_star_background);
    JG(star_density); JG(star_brightness); JG(hdri_path);
    JG(hdri_exposure); JG(hdri_rotation_deg);
    JG(postprocess_pipeline); JG(gargantua_look_strength); JG(gargantua_look_preset);
    JG(output); JG(keep_frames); JG(resume_frames); JG(frames_dir); JG(video_codec); JG(progress_file);
    JG(animate); JG(animation_duration); JG(animation_fps);
    JG(animation_parameter); JG(animation_start); JG(animation_end);
    JG(animation_workers);
}

} // namespace kerrtrace
