#pragma once
#include <torch/torch.h>
#include <string>
#include <filesystem>

namespace kerrtrace {

// Write (H, W, 3) float32 RGB tensor to PNG.
// Values are tone-mapped (Reinhard) and clamped to [0,255].
void save_png(const torch::Tensor& rgb, const std::string& path);

// Simple Reinhard tone-map: out = x / (1 + x)
torch::Tensor tonemap_reinhard(const torch::Tensor& x);

// ACES filmic tone-map approximation
torch::Tensor tonemap_aces(const torch::Tensor& x);

// Load PNG into (H, W, 3) float32 tensor, values in [0,1]
torch::Tensor load_png(const std::string& path, const torch::TensorOptions& opts);

} // namespace kerrtrace
