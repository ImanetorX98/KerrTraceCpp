#pragma once
#include "kerrtrace/config.hpp"
#include "kerrtrace/raytracer.hpp"
#include <filesystem>
#include <functional>

namespace kerrtrace {

struct FrameSpec {
    int   frame_index;
    float t_norm;      // [0,1] normalised time
    RenderConfig cfg;  // per-frame config
};

// Build frame schedule by interpolating the animation parameter.
std::vector<FrameSpec> build_frame_schedule(const RenderConfig& base_cfg);

// Render all frames to <output_dir>/frame_XXXXX.png
// Returns the frames directory path.
std::filesystem::path render_animation(
    const RenderConfig& base_cfg,
    const std::filesystem::path& frames_dir,
    std::function<void(int /*frame*/, int /*total*/)> progress = nullptr
);

// Encode PNG frames to MP4 using system ffmpeg.
// Returns true on success.
bool encode_video(
    const std::filesystem::path& frames_dir,
    const std::string& output_path,
    float fps,
    const std::string& codec = "h264"  // h264 | h265
);

} // namespace kerrtrace
