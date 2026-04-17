#pragma once
#include <string>
#include <optional>
#include <nlohmann/json.hpp>

namespace kerrtrace {

struct RenderConfig {
    // ── Resolution & Camera ────────────────────────────────────────────────
    int   width                      = 1280;
    int   height                     = 720;
    float fov_deg                    = 38.0f;

    // ── Spacetime ─────────────────────────────────────────────────────────
    std::string metric_model         = "kerr";       // kerr | kerr_newman | schwarzschild
    std::string coordinate_system    = "kerr_schild"; // kerr_schild | boyer_lindquist
    float spin                       = 0.85f;         // |a| <= 1
    float charge                     = 0.0f;          // |Q| <= 1

    // ── Observer ───────────────────────────────────────────────────────────
    float observer_radius            = 30.0f;   // in units of M
    float observer_inclination_deg   = 80.0f;
    float observer_azimuth_deg       = 0.0f;
    float observer_roll_deg          = 0.0f;

    // ── Accretion disk ─────────────────────────────────────────────────────
    std::string disk_model           = "physical_nt"; // physical_nt | legacy | riaf
    float disk_inner_radius          = -1.0f;          // <0 → auto (ISCO)
    float disk_outer_radius          = 12.0f;
    float disk_emission_gain         = 1.0f;   // matches Python default
    std::string disk_palette         = "default";      // default | interstellar_warm
    std::string disk_radial_profile  = "nt_page_thorne"; // nt_page_thorne | nt_proxy

    // Disk edge boosts
    float inner_edge_boost             = 1.0f;
    float outer_edge_boost             = 0.5f;
    float disk_beaming_strength        = 0.45f;   // matches Python default
    float disk_self_occlusion_strength = 0.35f;

    // ── RIAF model ─────────────────────────────────────────────────────────
    float riaf_alpha_n               = 1.1f;   // n_e ∝ r^{-α_n}
    float riaf_alpha_T               = 0.84f;  // T_e ∝ r^{-α_T}
    float riaf_alpha_B               = 1.25f;  // B   ∝ r^{-α_B}
    float riaf_T_visual              = 18000.f;
    std::string riaf_color_mode      = "blackbody"; // blackbody|plasma|interstellar_warm|gargantua

    // ── Layered disk palette ───────────────────────────────────────────────
    bool  disk_layered_palette              = true;
    int   disk_layer_count                  = 30;
    float disk_layer_mix                    = 0.55f;
    float disk_layer_pattern_count          = 14.0f;
    float disk_layer_pattern_contrast       = 0.45f;
    float disk_layer_time_scale             = 1.0f;
    float disk_layer_accident_strength      = 0.42f;
    float disk_layer_accident_count         = 3.8f;
    float disk_layer_accident_sharpness     = 7.0f;
    float disk_layer_global_phase           = 0.0f;
    float disk_layer_phase_rate_hz          = 0.35f;
    bool  enable_disk_differential_rotation = false;
    std::string disk_diffrot_visual_mode    = "layer_phase"; // layer_phase|annular_tiles|hybrid
    float disk_diffrot_strength             = 1.0f;
    int   disk_diffrot_seed                 = 7;

    // ── Segmented disk palette ─────────────────────────────────────────────
    bool  disk_segmented_palette     = false;
    int   disk_segmented_rings       = 3;
    int   disk_segmented_sectors     = 12;
    float disk_segmented_sigma       = 0.5f;
    float disk_segmented_mix         = 1.0f;
    float disk_segmented_hue_offset  = 0.0f;
    std::string disk_segmented_palette_mode = "accretion_warm"; // accretion_warm|rainbow

    // ── Disk volume emission ───────────────────────────────────────────────
    bool  disk_volume_emission              = false;
    int   disk_volume_samples               = 5;
    float disk_volume_density_scale         = 1.0f;
    float disk_volume_temperature_drop      = 0.28f;
    float disk_volume_strength              = 0.85f;

    // ── Integration ────────────────────────────────────────────────────────
    int   max_steps                  = 500;
    float step_size                  = 0.11f;
    bool  adaptive_integrator        = true;

    // ── GPU / compute ──────────────────────────────────────────────────────
    std::string device               = "auto";   // auto | cpu | cuda | mps
    std::string dtype                = "float32"; // float32 | float64
    int   render_tile_rows           = 64;        // rows per tile (memory control)

    // ── Background ─────────────────────────────────────────────────────────
    std::string background_mode      = "darkspace"; // darkspace | hdri | procedural
    bool  enable_star_background     = true;
    float star_density               = 0.5f;
    float star_brightness            = 1.0f;
    std::string hdri_path            = "";
    float hdri_exposure              = 1.0f;
    float hdri_rotation_deg          = 0.0f;

    // ── Post-processing ────────────────────────────────────────────────────
    std::string postprocess_pipeline = "off";   // off | gargantua
    float gargantua_look_strength    = 0.0f;
    bool  gargantua_look_preset      = false;

    // ── Output ─────────────────────────────────────────────────────────────
    std::string output               = "out/render.png";
    bool  keep_frames                = false;
    bool  resume_frames              = false;
    std::string frames_dir           = "";
    std::string video_codec          = "h264"; // h264|h265
    std::string progress_file        = "";

    // ── Animation ──────────────────────────────────────────────────────────
    bool  animate                    = false;
    float animation_duration         = 2.0f;
    float animation_fps              = 24.0f;
    std::string animation_parameter  = "observer_azimuth_deg";
    float animation_start            = 0.0f;
    float animation_end              = 360.0f;
    int   animation_workers          = 1;
};

// ── JSON serialization ────────────────────────────────────────────────────────
void to_json(nlohmann::json& j, const RenderConfig& c);
void from_json(const nlohmann::json& j, RenderConfig& c);

// Apply gargantua preset (modifies defaults like the Python version)
RenderConfig apply_gargantua_preset(RenderConfig cfg);

// Validate config; throws std::invalid_argument on error
void validate(const RenderConfig& cfg);

// Compute ISCO for the given spin (prograde, M=1)
float isco_radius(float spin);

// Compute outer event horizon radius
float event_horizon_radius(float spin, float charge = 0.0f);

} // namespace kerrtrace
