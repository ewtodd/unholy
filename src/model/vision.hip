#include "model/vision.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernels/gemm_wmma.hpp"
#include "kernels/vision.hpp"

namespace unholy::model {

namespace k = unholy::kernels;

VisionScratch make_vision_scratch(const Dsv4Config& c, int dev) {
  const int r = c.vision_downsample, D = c.vision_dim;
  const size_t N = size_t(c.vision_max_tokens) * r * r + 64;  // patches (grids are padded to r)
  const size_t L = size_t(c.vision_max_tokens);
  const size_t kpad = (size_t(3) * c.vision_patch * c.vision_patch + 31) / 32 * 32;
  VisionScratch s;
  s.max_patches = int(N);
  s.patches = rt::DeviceBuffer(dev, N * kpad * 2);
  s.x = rt::DeviceBuffer(dev, N * D * 2);
  s.h = rt::DeviceBuffer(dev, N * D * 2);
  s.qkv = rt::DeviceBuffer(dev, N * 3 * D * 2);
  s.o = rt::DeviceBuffer(dev, N * D * 2);
  s.f32 = rt::DeviceBuffer(dev, std::max(N * 2 * c.vision_inter, L * c.dim) * 4);
  s.mlp = rt::DeviceBuffer(dev, N * c.vision_inter * 2);
  s.unfold = rt::DeviceBuffer(dev, L * D * r * r * 2);
  s.gelu = rt::DeviceBuffer(dev, L * c.dim * 2);
  s.aligned = rt::DeviceBuffer(dev, L * c.dim * 2);
  s.cos = rt::DeviceBuffer(dev, N * 32 * 4);
  s.sin = rt::DeviceBuffer(dev, N * 32 * 4);
  s.src = rt::DeviceBuffer(dev, (L * 2 + 64) * 4);
  return s;
}

void vision_encode(const Dsv4Config& c, const VisionWeights& w, VisionScratch& s, const vision::Patches& p,
                   const vision::ImageBlock& block, uint16_t* out, hipStream_t st) {
  const int N = p.n_vit_h * p.n_vit_w, D = c.vision_dim, H = c.vision_heads, I = c.vision_inter;
  const int r = c.vision_downsample, K0 = 3 * c.vision_patch * c.vision_patch;
  if (N <= 0 || N > s.max_patches) throw std::runtime_error("vision: image larger than the scratch");
  if (D / H != 64) throw std::runtime_error("vision: head dim 64 only");
  // Patches [N][K0] -> [N][kpad] (zero tail), staged from the host.
  {
    std::vector<uint16_t> padded(size_t(N) * w.kpad, 0);
    for (int i = 0; i < N; ++i) std::copy_n(&p.data[size_t(i) * K0], K0, &padded[size_t(i) * w.kpad]);
    UH_CHECK(hipMemcpyAsync(s.patches.data(), padded.data(), padded.size() * 2, hipMemcpyHostToDevice, st));
    // 2-D rope: inv_freq over rope dim 32 (theta^(-2i/32)), freqs [h * inv (16), w * inv (16)] in fp32.
    std::vector<float> cs(size_t(N) * 32), sn(size_t(N) * 32);
    float inv[16];
    for (int i = 0; i < 16; ++i) inv[i] = 1.0f / std::pow(float(c.vision_rope_theta), float(2 * i) / 32.0f);
    for (int y = 0; y < p.n_vit_h; ++y)
      for (int x = 0; x < p.n_vit_w; ++x)
        for (int j = 0; j < 32; ++j) {
          const float f = float(j < 16 ? y : x) * inv[j % 16];
          cs[(size_t(y) * p.n_vit_w + x) * 32 + j] = float(std::cos(double(f)));
          sn[(size_t(y) * p.n_vit_w + x) * 32 + j] = float(std::sin(double(f)));
        }
    UH_CHECK(hipMemcpyAsync(s.cos.data(), cs.data(), cs.size() * 4, hipMemcpyHostToDevice, st));
    UH_CHECK(hipMemcpyAsync(s.sin.data(), sn.data(), sn.size() * 4, hipMemcpyHostToDevice, st));
    // Block rows: image slots take the aligner rows in the N-layout order, the rest their type's row.
    std::vector<int32_t> src(block.types.size());
    size_t k = 0;
    for (size_t i = 0; i < block.types.size(); ++i)
      src[i] = block.types[i] == vision::kImage ? block.perm.at(k++) : -1 - block.types[i];
    UH_CHECK(hipMemcpyAsync(s.src.data(), src.data(), src.size() * 4, hipMemcpyHostToDevice, st));
    UH_CHECK(hipStreamSynchronize(st));  // the host vectors go out of scope
  }
  auto* x = s.x.as<uint16_t>();
  auto* h = s.h.as<uint16_t>();
  auto* qkv = s.qkv.as<uint16_t>();
  auto* o = s.o.as<uint16_t>();
  auto* acc = s.f32.as<float>();
  k::gemm_bf16_f32out_wmma(s.patches.as<uint16_t>(), w.kpad, w.patch_w.as<uint16_t>(), acc, D, N, D, w.kpad, st);
  k::vit_bias(acc, N, D, w.patch_b.p<uint16_t>(), x, st);
  for (const auto& b : w.blocks) {
    k::vit_rmsnorm(x, N, D, b.norm1.p<uint16_t>(), 1e-6f, h, st);
    k::gemm_bf16_f32out_wmma(h, D, b.wqkv_w.p<uint16_t>(), acc, 3 * D, N, 3 * D, D, st);
    k::vit_bias(acc, N, 3 * D, b.wqkv_b.p<uint16_t>(), qkv, st);
    k::vit_rope(qkv, N, H, s.cos.as<float>(), s.sin.as<float>(), st);
    k::vit_attention(qkv, N, H, o, st);
    k::gemm_bf16_f32out_wmma(o, D, b.wo_w.p<uint16_t>(), acc, D, N, D, D, st);
    k::vit_bias(acc, N, D, b.wo_b.p<uint16_t>(), h, st);
    k::vit_add(x, h, size_t(N) * D, st);
    k::vit_rmsnorm(x, N, D, b.norm2.p<uint16_t>(), 1e-6f, h, st);
    k::gemm_bf16_f32out_wmma(h, D, b.w1.p<uint16_t>(), acc, 2 * I, N, 2 * I, D, st);
    k::vit_silu_mul(acc, N, I, s.mlp.as<uint16_t>(), st);
    k::gemm_bf16_f32out_wmma(s.mlp.as<uint16_t>(), I, b.w2.p<uint16_t>(), acc, D, N, D, I, st);
    k::vit_bias(acc, N, D, nullptr, h, st);
    k::vit_add(x, h, size_t(N) * D, st);
  }
  k::vit_rmsnorm(x, N, D, w.norm.p<uint16_t>(), 1e-6f, h, st);
  // Aligner: 3 x 3 unfold -> w1 -> GELU -> w2.
  const int L = ((p.n_vit_h + r - 1) / r) * ((p.n_vit_w + r - 1) / r), KU = D * r * r;
  k::vit_unfold(h, p.n_vit_h, p.n_vit_w, D, r, s.unfold.as<uint16_t>(), st);
  k::gemm_bf16_f32out_wmma(s.unfold.as<uint16_t>(), KU, w.al_w1.p<uint16_t>(), acc, c.dim, L, c.dim, KU, st);
  k::vit_bias_gelu(acc, L, c.dim, w.al_b1.p<uint16_t>(), s.gelu.as<uint16_t>(), st);
  k::gemm_bf16_f32out_wmma(s.gelu.as<uint16_t>(), c.dim, w.al_w2.p<uint16_t>(), acc, c.dim, L, c.dim, c.dim, st);
  k::vit_bias(acc, L, c.dim, w.al_b2.p<uint16_t>(), s.aligned.as<uint16_t>(), st);
  k::vit_assemble(s.aligned.as<uint16_t>(), w.params.as<uint16_t>(), s.src.as<int32_t>(), int(block.types.size()),
                  c.dim, out, st);
}

}  // namespace unholy::model
