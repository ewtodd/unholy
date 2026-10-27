#include "model/glm5_vision.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/gemm_wmma.hpp"
#include "kernels/vision.hpp"
#include "model/loader.hpp"

namespace unholy::model {

namespace k = unholy::kernels;

namespace {

constexpr int kRows = 2048;              // rows per chunk of the row-wise layers
constexpr float kLayerNormEps = 1e-5f;  // the merger's nn.LayerNorm(dim) default

struct Sizes {
  size_t patches, act, qkv, acc, mlp, rope;  // elements
};
Sizes sizes(const Glm5VisionConfig& c, int max_tokens) {
  const size_t L = size_t(max_tokens), N = L * c.merge * c.merge;
  const size_t kpad = (size_t(c.in_channels) * c.temporal * c.patch * c.patch + 31) / 32 * 32;
  const size_t widest = std::max({3 * c.dim, 2 * c.inter, c.out_dim, 2 * c.proj_inter});
  return {N * kpad, std::max(N * c.dim, L * c.out_dim), N * 3 * c.dim, kRows * widest,
          kRows * size_t(std::max(c.inter, c.proj_inter)), N * 32};
}

}  // namespace

Glm5VisionConfig Glm5VisionConfig::from_hf(const gufo::json::Value& top) {
  Glm5VisionConfig c;
  const auto* v = top.find("vision_config");
  if (!v) return c;
  auto num = [&](const char* key) {
    const auto* x = v->find(key);
    if (!x || !x->is_number()) throw std::runtime_error(std::string("vision_config: missing ") + key);
    return x->as_double();
  };
  c.depth = int(num("depth"));
  c.dim = int(num("hidden_size"));
  c.heads = int(num("num_heads"));
  c.inter = int(num("intermediate_size"));
  c.out_dim = int(num("out_hidden_size"));
  c.proj_inter = int(num("projection_intermediate_size"));
  c.in_channels = int(num("in_channels"));
  c.patch = int(num("patch_size"));
  c.temporal = int(num("temporal_patch_size"));
  c.merge = int(num("spatial_merge_size"));
  c.eps = float(num("rms_norm_eps"));
  c.swiglu_limit = float(num("swiglu_limit"));
  c.rope_theta = 10000.0f;  // Glm5NextVisionConfig's default rope_parameters
  if (const auto* rp = v->find("rope_parameters"); rp && rp->find("rope_theta"))
    c.rope_theta = float(rp->find("rope_theta")->as_double());
  return c;
}

vision::GlmVisionConfig Glm5VisionConfig::preprocess(int max_tokens) const {
  vision::GlmVisionConfig v;
  v.patch = patch;
  v.merge = merge;
  v.temporal = temporal;
  v.max_tokens = max_tokens;
  return v;
}

Glm5VisionWeights load_glm5_vision(const Checkpoint& ck, const Glm5VisionConfig& c, int dev, hipStream_t st) {
  if (c.depth <= 0) throw std::runtime_error("load_glm5_vision: the config has no vision tower");
  if (!ck.find("vision.patch_embed.proj.weight"))
    throw std::runtime_error("load_glm5_vision: no vision.* tensors (vision.safetensors beside the quant: "
                             "tools/box/extract-glm-vision.py)");
  if (c.dim / c.heads != 64) throw std::runtime_error("load_glm5_vision: head dim 64 only");
  rt::DeviceGuard g(dev);
  std::vector<Uploader::Item> items;
  auto bf16 = [&](const std::string& name) -> const TensorInfo& {
    const TensorInfo& t = ck.at(name);
    if (t.dtype != DType::BF16) throw std::runtime_error(name + ": expected BF16");
    return t;
  };
  auto tensor = [&](const std::string& name) {
    const TensorInfo& t = bf16(name);
    DevTensor d;
    d.dtype = t.dtype;
    d.shape = t.shape;
    d.buf = rt::DeviceBuffer(dev, t.nbytes);
    d.ptr = d.buf.data();
    items.push_back({&t, d.ptr});
    return d;
  };
  // Two tensors of the same shape, the second's rows after the first's.
  auto stacked = [&](const std::string& a, const std::string& b) {
    const TensorInfo &ta = bf16(a), &tb = bf16(b);
    if (ta.shape != tb.shape) throw std::runtime_error(a + ", " + b + ": shapes differ");
    rt::DeviceBuffer buf(dev, ta.nbytes + tb.nbytes);
    items.push_back({&ta, buf.data()});
    items.push_back({&tb, static_cast<uint8_t*>(buf.data()) + ta.nbytes});
    return buf;
  };
  Glm5VisionWeights w;
  w.patch_b = tensor("vision.patch_embed.proj.bias");
  for (int i = 0; i < c.depth; ++i) {
    const std::string p = "vision.blocks." + std::to_string(i) + ".";
    Glm5VisionWeights::Block b;
    b.norm1 = tensor(p + "norm1.weight");
    b.qkv_w = tensor(p + "attn.qkv.weight");
    b.qkv_b = tensor(p + "attn.qkv.bias");
    b.q_norm = tensor(p + "attn.q_norm.weight");
    b.k_norm = tensor(p + "attn.k_norm.weight");
    b.proj_w = tensor(p + "attn.proj.weight");
    b.proj_b = tensor(p + "attn.proj.bias");
    b.norm2 = tensor(p + "norm2.weight");
    b.gate_up_w = stacked(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight");
    b.gate_up_b = stacked(p + "mlp.gate_proj.bias", p + "mlp.up_proj.bias");
    b.down_w = tensor(p + "mlp.down_proj.weight");
    b.down_b = tensor(p + "mlp.down_proj.bias");
    w.blocks.push_back(std::move(b));
  }
  w.post_norm = tensor("vision.post_layernorm.weight");
  w.ds_b = tensor("vision.downsample.bias");
  w.m_proj = tensor("vision.merger.proj.weight");
  w.m_norm_w = tensor("vision.merger.post_projection_norm.weight");
  w.m_norm_b = tensor("vision.merger.post_projection_norm.bias");
  w.m_gate_up = stacked("vision.merger.gate_proj.weight", "vision.merger.up_proj.weight");
  w.m_down = tensor("vision.merger.down_proj.weight");
  Uploader up(ck);
  up.upload_many(std::move(items), st);
  up.sync();

  // Patch embedding Conv3d [dim][3][temporal][p][p] (the patches' own layout) -> [dim][kpad].
  const TensorInfo& pw = bf16("vision.patch_embed.proj.weight");
  const int K = c.in_channels * c.temporal * c.patch * c.patch;
  if (pw.nbytes != size_t(c.dim) * K * 2) throw std::runtime_error("vision.patch_embed.proj.weight: shape");
  w.kpad = (K + 31) / 32 * 32;
  const auto src = ck.as<uint16_t>(pw);
  std::vector<uint16_t> padded(size_t(c.dim) * w.kpad, 0);
  for (int r = 0; r < c.dim; ++r) std::copy_n(&src[size_t(r) * K], K, &padded[size_t(r) * w.kpad]);
  w.patch_w = rt::DeviceBuffer(dev, padded.size() * 2);
  UH_CHECK(hipMemcpy(w.patch_w.data(), padded.data(), padded.size() * 2, hipMemcpyHostToDevice));
  // Downsample Conv2d [out][dim][i][j] -> [out][(i * merge + j) * dim + c]: a group's patches are rows of the input.
  const TensorInfo& dw = bf16("vision.downsample.weight");
  const int m2 = c.merge * c.merge;
  if (dw.nbytes != size_t(c.out_dim) * c.dim * m2 * 2) throw std::runtime_error("vision.downsample.weight: shape");
  const auto ds = ck.as<uint16_t>(dw);
  std::vector<uint16_t> re(size_t(c.out_dim) * m2 * c.dim);
  for (int o = 0; o < c.out_dim; ++o)
    for (int ch = 0; ch < c.dim; ++ch)
      for (int q = 0; q < m2; ++q) re[(size_t(o) * m2 + q) * c.dim + ch] = ds[(size_t(o) * c.dim + ch) * m2 + q];
  w.ds_w = rt::DeviceBuffer(dev, re.size() * 2);
  UH_CHECK(hipMemcpy(w.ds_w.data(), re.data(), re.size() * 2, hipMemcpyHostToDevice));
  return w;
}

size_t glm5_vision_scratch_bytes(const Glm5VisionConfig& c, int max_tokens) {
  const Sizes z = sizes(c, max_tokens);
  return (z.patches + 3 * z.act + z.qkv + z.mlp) * 2 + (z.acc + 2 * z.rope) * 4;
}

Glm5VisionScratch make_glm5_vision_scratch(const Glm5VisionConfig& c, int max_tokens, int dev) {
  const Sizes z = sizes(c, max_tokens);
  Glm5VisionScratch s;
  s.max_patches = max_tokens * c.merge * c.merge;
  s.patches = rt::DeviceBuffer(dev, z.patches * 2);
  s.x = rt::DeviceBuffer(dev, z.act * 2);
  s.h = rt::DeviceBuffer(dev, z.act * 2);
  s.o = rt::DeviceBuffer(dev, z.act * 2);
  s.qkv = rt::DeviceBuffer(dev, z.qkv * 2);
  s.acc = rt::DeviceBuffer(dev, z.acc * 4);
  s.mlp = rt::DeviceBuffer(dev, z.mlp * 2);
  s.cos = rt::DeviceBuffer(dev, z.rope * 4);
  s.sin = rt::DeviceBuffer(dev, z.rope * 4);
  return s;
}

void glm5_vision_encode(const Glm5VisionConfig& c, const Glm5VisionWeights& w, Glm5VisionScratch& s,
                        const vision::GlmPatches& p, uint16_t* out, hipStream_t st) {
  const int m = c.merge, m2 = m * m, N = p.grid_h * p.grid_w, L = N / m2;
  const int D = c.dim, H = c.heads, I = c.inter, O = c.out_dim, PI = c.proj_inter;
  const int K0 = c.in_channels * c.temporal * c.patch * c.patch;
  if (N <= 0 || N > s.max_patches) throw std::runtime_error("glm5 vision: image larger than the scratch");
  if (p.grid_h % m || p.grid_w % m) throw std::runtime_error("glm5 vision: grid not a multiple of the merge size");
  if (p.data.size() != size_t(N) * K0) throw std::runtime_error("glm5 vision: patch data size");
  {
    std::vector<uint16_t> padded(size_t(N) * w.kpad, 0);
    for (int i = 0; i < N; ++i) std::copy_n(&p.data[size_t(i) * K0], K0, &padded[size_t(i) * w.kpad]);
    UH_CHECK(hipMemcpyAsync(s.patches.data(), padded.data(), padded.size() * 2, hipMemcpyHostToDevice, st));
    // Rope angles by each patch's row and column, the patches in merge-group order (get_vision_position_ids):
    // inv_freq theta^(-2i / 32), i < 16, the row's 16 then the column's, in fp32.
    std::vector<float> cs(size_t(N) * 32), sn(size_t(N) * 32);
    float inv[16];
    for (int i = 0; i < 16; ++i) inv[i] = 1.0f / std::pow(c.rope_theta, float(2 * i) / 32.0f);
    size_t idx = 0;
    for (int by = 0; by < p.grid_h / m; ++by)
      for (int bx = 0; bx < p.grid_w / m; ++bx)
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j, ++idx)
            for (int f = 0; f < 32; ++f) {
              const float a = float(f < 16 ? by * m + i : bx * m + j) * inv[f % 16];
              cs[idx * 32 + f] = float(std::cos(double(a)));
              sn[idx * 32 + f] = float(std::sin(double(a)));
            }
    UH_CHECK(hipMemcpyAsync(s.cos.data(), cs.data(), cs.size() * 4, hipMemcpyHostToDevice, st));
    UH_CHECK(hipMemcpyAsync(s.sin.data(), sn.data(), sn.size() * 4, hipMemcpyHostToDevice, st));
    UH_CHECK(hipStreamSynchronize(st));  // the host vectors go out of scope
  }
  auto* x = s.x.as<uint16_t>();
  auto* h = s.h.as<uint16_t>();
  auto* o = s.o.as<uint16_t>();
  auto* qkv = s.qkv.as<uint16_t>();
  auto* mlp = s.mlp.as<uint16_t>();
  auto* acc = s.acc.as<float>();
  // a [M][K] (row stride lda) x wt [n][K]^T in chunks of kRows rows; epi(m0, rows) consumes acc.
  auto linear = [&](const uint16_t* a, int lda, int M, const uint16_t* wt, int n, int K, const auto& epi) {
    for (int m0 = 0; m0 < M; m0 += kRows) {
      const int r = std::min(kRows, M - m0);
      k::gemm_bf16_f32out_wmma(a + size_t(m0) * lda, lda, wt, acc, n, r, n, K, st);
      epi(m0, r);
    }
  };
  linear(s.patches.as<uint16_t>(), w.kpad, N, w.patch_w.as<uint16_t>(), D, w.kpad,
         [&](int m0, int r) { k::vit_bias(acc, r, D, w.patch_b.p<uint16_t>(), x + size_t(m0) * D, st); });
  for (const auto& b : w.blocks) {
    k::vit_rmsnorm_hf(x, N, D, b.norm1.p<uint16_t>(), c.eps, h, st);
    linear(h, D, N, b.qkv_w.p<uint16_t>(), 3 * D, D, [&](int m0, int r) {
      k::vit_bias(acc, r, 3 * D, b.qkv_b.p<uint16_t>(), qkv + size_t(m0) * 3 * D, st);
    });
    k::vit_qk_norm_rope(qkv, N, H, b.q_norm.p<uint16_t>(), b.k_norm.p<uint16_t>(), c.eps, s.cos.as<float>(),
                        s.sin.as<float>(), st);
    k::vit_attention_wmma(qkv, N, H, o, st);
    linear(o, D, N, b.proj_w.p<uint16_t>(), D, D,
           [&](int m0, int r) { k::vit_bias(acc, r, D, b.proj_b.p<uint16_t>(), h + size_t(m0) * D, st); });
    k::vit_add(x, h, size_t(N) * D, st);
    k::vit_rmsnorm_hf(x, N, D, b.norm2.p<uint16_t>(), c.eps, h, st);
    linear(h, D, N, b.gate_up_w.as<uint16_t>(), 2 * I, D, [&](int m0, int r) {
      k::vit_swiglu_clamp(acc, r, I, b.gate_up_b.as<uint16_t>(), c.swiglu_limit, mlp, st);
      k::gemm_bf16_f32out_wmma(mlp, I, b.down_w.p<uint16_t>(), acc, D, r, D, I, st);
      k::vit_bias(acc, r, D, b.down_b.p<uint16_t>(), o + size_t(m0) * D, st);
    });
    k::vit_add(x, o, size_t(N) * D, st);
  }
  k::vit_rmsnorm_hf(x, N, D, w.post_norm.p<uint16_t>(), c.eps, h, st);
  // Downsample: merge group l is rows l * m2 .. l * m2 + m2 - 1 of h, read as one row of m2 * D.
  linear(h, m2 * D, L, w.ds_w.as<uint16_t>(), O, m2 * D,
         [&](int m0, int r) { k::vit_bias(acc, r, O, w.ds_b.p<uint16_t>(), x + size_t(m0) * O, st); });
  linear(x, O, L, w.m_proj.p<uint16_t>(), O, O,
         [&](int m0, int r) { k::vit_bias(acc, r, O, nullptr, o + size_t(m0) * O, st); });
  k::vit_layernorm_gelu(o, L, O, w.m_norm_w.p<uint16_t>(), w.m_norm_b.p<uint16_t>(), kLayerNormEps, h, st);
  linear(h, O, L, w.m_gate_up.as<uint16_t>(), 2 * PI, O, [&](int m0, int r) {
    k::vit_swiglu_clamp(acc, r, PI, nullptr, c.swiglu_limit, mlp, st);
    k::gemm_bf16_f32out_wmma(mlp, PI, w.m_down.p<uint16_t>(), acc, O, r, O, PI, st);
    k::vit_bias(acc, r, O, nullptr, out + size_t(m0) * O, st);
  });
}

}  // namespace unholy::model
