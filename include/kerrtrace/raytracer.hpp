#pragma once
#include "kerrtrace/config.hpp"
#include <torch/torch.h>
#include <functional>

namespace kerrtrace {

// Progress callback: (completed_rows, total_rows)
using ProgressFn = std::function<void(int, int)>;

class Raytracer {
public:
    explicit Raytracer(const RenderConfig& cfg);

    // Render a single frame. Returns (H, W, 3) float32 RGB tensor in [0, ∞).
    torch::Tensor render(ProgressFn progress = nullptr) const;

    // Apply gargantua post-processing to a (H, W, 3) image.
    torch::Tensor postprocess(const torch::Tensor& rgb) const;

private:
    // Core geodesic integration loop for a tile.
    // pixel_rows: flat indices [row_start, row_end) of this tile.
    torch::Tensor render_tile(int row_start, int row_end) const;

    // Compute disk emission at disk crossing points.
    // r_cross: (N,), E: (N,), L: (N,), in_disk mask: (N,), order: (N,)
    torch::Tensor disk_emission(
        const torch::Tensor& r_cross,
        const torch::Tensor& phi_cross,
        const torch::Tensor& E,
        const torch::Tensor& L,
        const torch::Tensor& in_disk_mask,
        const torch::Tensor& order
    ) const;

    // Compute background color for escaped rays.
    torch::Tensor background_color(
        const torch::Tensor& state,
        const torch::Tensor& escaped_mask
    ) const;

    // Apply gargantua bloom + tone-map.
    torch::Tensor postprocess_gargantua(const torch::Tensor& rgb, float strength) const;

    const RenderConfig cfg_;
    torch::Device device_;
    torch::Dtype  dtype_;
    torch::Tensor hdri_tex_;
    float a_;       // spin
    float r_isco_;
    float r_horizon_;
};

} // namespace kerrtrace
