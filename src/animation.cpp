#include "kerrtrace/animation.hpp"
#include "kerrtrace/image_io.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

namespace kerrtrace {

// ── Frame schedule ────────────────────────────────────────────────────────────

std::vector<FrameSpec> build_frame_schedule(const RenderConfig& base)
{
    int total_frames = static_cast<int>(
        std::round(base.animation_duration * base.animation_fps));
    total_frames = std::max(1, total_frames);

    std::vector<FrameSpec> schedule;
    schedule.reserve(total_frames);

    for (int i = 0; i < total_frames; ++i) {
        float t = (total_frames > 1)
                ? static_cast<float>(i) / static_cast<float>(total_frames - 1)
                : 0.0f;
        float val = base.animation_start
                  + t * (base.animation_end - base.animation_start);

        RenderConfig cfg = base;
        // Apply animated parameter
        const auto& p = base.animation_parameter;
        if      (p == "observer_azimuth_deg")    cfg.observer_azimuth_deg    = val;
        else if (p == "observer_inclination_deg") cfg.observer_inclination_deg = val;
        else if (p == "observer_radius")          cfg.observer_radius          = val;
        else if (p == "spin")                     cfg.spin                     = val;
        else if (p == "disk_emission_gain")       cfg.disk_emission_gain       = val;

        schedule.push_back({i, t, cfg});
    }
    return schedule;
}

// ── Render all frames ─────────────────────────────────────────────────────────

std::filesystem::path render_animation(
    const RenderConfig& base_cfg,
    const std::filesystem::path& frames_dir,
    bool resume_existing,
    std::function<void(int, int)> progress)
{
    std::filesystem::create_directories(frames_dir);
    auto schedule = build_frame_schedule(base_cfg);
    int total = static_cast<int>(schedule.size());
    int workers = std::max(1, std::min(base_cfg.animation_workers, total));

    std::atomic<int> next_frame{0};
    std::atomic<int> done_frames{0};
    std::mutex progress_mtx;

    auto worker = [&]() {
        while (true) {
            int idx = next_frame.fetch_add(1);
            if (idx >= total) break;

            const auto& spec = schedule[idx];

            std::ostringstream ss;
            ss << "frame_" << std::setw(5) << std::setfill('0') << spec.frame_index << ".png";
            auto path = frames_dir / ss.str();

            if (resume_existing && std::filesystem::exists(path)) {
                int completed = ++done_frames;
                if (progress) progress(completed, total);
                continue;
            }

            Raytracer rt(spec.cfg);
            auto img = rt.render();
            img = rt.postprocess(img);

            save_png(img, path.string());

            int completed = ++done_frames;
            if (progress) progress(completed, total);
        }
    };

    if (workers == 1) {
        worker();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int w = 0; w < workers; ++w)
            threads.emplace_back(worker);
        for (auto& t : threads) t.join();
    }

    return frames_dir;
}

// ── Video encoding ────────────────────────────────────────────────────────────

bool encode_video(
    const std::filesystem::path& frames_dir,
    const std::string& output_path,
    float fps,
    const std::string& codec)
{
    std::string vcodec = (codec == "h265") ? "libx265" : "libx264";
    std::string pix_fmt = (codec == "h265") ? "yuv420p10le" : "yuv420p";
    std::string crf = (codec == "h265") ? "18" : "23";

    std::ostringstream cmd;
    cmd << "ffmpeg -y"
        << " -framerate " << fps
        << " -i \"" << (frames_dir / "frame_%05d.png").string() << "\""
        << " -c:v " << vcodec
        << " -crf " << crf
        << " -pix_fmt " << pix_fmt
        << " -movflags +faststart"
        << " \"" << output_path << "\""
        << " 2>&1";

    int ret = std::system(cmd.str().c_str());
    return (ret == 0);
}

} // namespace kerrtrace
