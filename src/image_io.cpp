#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include "kerrtrace/image_io.hpp"
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace kerrtrace {

torch::Tensor tonemap_reinhard(const torch::Tensor& x)
{
    return x / (1.0f + x);
}

torch::Tensor tonemap_aces(const torch::Tensor& x)
{
    auto xc = x.clamp_min(0.0f);
    return (xc * (xc * 2.51f + 0.03f))
         / (xc * (xc * 2.43f + 0.59f) + 0.14f).clamp_min(1e-5f);
}

void save_png(const torch::Tensor& rgb, const std::string& path)
{
    // Ensure output directory exists
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    // Move to CPU, float32
    auto img = rgb.to(torch::kCPU).to(torch::kFloat32).contiguous();
    int H = static_cast<int>(img.size(0));
    int W = static_cast<int>(img.size(1));

    // Tone-map: ACES filmic
    auto toned = tonemap_aces(img).clamp(0.0f, 1.0f);

    // Convert to uint8
    auto u8 = (toned * 255.0f).round().clamp(0.0f, 255.0f).to(torch::kUInt8);
    auto u8_c = u8.contiguous();

    int ret = stbi_write_png(
        path.c_str(), W, H, 3,
        u8_c.data_ptr<uint8_t>(),
        W * 3);

    if (ret == 0)
        throw std::runtime_error("Failed to write PNG: " + path);
}

torch::Tensor load_png(const std::string& path, const torch::TensorOptions& opts)
{
    int W, H, C;
    unsigned char* data = stbi_load(path.c_str(), &W, &H, &C, 3);
    if (!data)
        throw std::runtime_error("Failed to load image: " + path);

    auto tensor = torch::from_blob(data, {H, W, 3}, torch::kUInt8).clone().to(opts);
    stbi_image_free(data);
    return tensor.to(torch::kFloat32) / 255.0f;
}

} // namespace kerrtrace
