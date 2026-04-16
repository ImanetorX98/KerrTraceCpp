#pragma once
#include <torch/torch.h>
#include <string>

namespace kerrtrace {

// All palette functions take t in [0,1] or temperatures,
// shape (N,), and return (N,3) RGB in [0,1..1.2].

// ── Blackbody RGB (CIE 1931 approximation) ───────────────────────────────────
// T in Kelvin
torch::Tensor blackbody_rgb(const torch::Tensor& T, const torch::TensorOptions& opts);

// ── Named palettes (t: 0=inner/hot, 1=outer/cool) ────────────────────────────
torch::Tensor palette_interstellar_warm(const torch::Tensor& t);
torch::Tensor palette_gargantua(const torch::Tensor& t);
torch::Tensor palette_plasma(const torch::Tensor& t);
torch::Tensor palette_default(const torch::Tensor& t);  // warm orange legacy

// Dispatch by name
torch::Tensor named_palette(const std::string& name, const torch::Tensor& t);

// ── Novikov-Thorne flux profile ───────────────────────────────────────────────
// Returns normalised flux profile F(r)/F_peak for NT disk.
// r_profile: (N,)  in-disk hit radii
// r_isco:    float
torch::Tensor nt_flux_profile(const torch::Tensor& r_profile, float r_isco, bool page_thorne = true);

// ── RIAF thin-disk emission ───────────────────────────────────────────────────
// Returns {intensity (N,), color (N,3)}
std::pair<torch::Tensor, torch::Tensor>
riaf_emission(
    const torch::Tensor& r_profile,
    float r_in,
    float r_out,
    float alpha_n,
    float alpha_T,
    float alpha_B,
    float T_visual,
    const std::string& color_mode
);

} // namespace kerrtrace
