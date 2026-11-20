#include "model/hy4_vision.hpp"

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

constexpr int kRows = 2048;    // rows per chunk of the row-wise layers (the fp32 accumulator's height)
constexpr int kHeadPad = 80;   // heads padded to 80 dims for the WMMA attention
constexpr int kPatchK = 768;   // 3 x 16 x 16

int pad32(int n) { return (n + 31) / 32 * 32; }

struct Sizes {
  size_t patches, act, qkv, acc, mlp, unfold, merged;  // elements
};
Sizes sizes(const Hy4VisionConfig& c, int max_patches) {
  const size_t N = size_t(max_patches), m2 = size_t(c.merge) * c.merge, L = N / m2;
  const size_t kconv = m2 * c.dim, ip = size_t(pad32(c.inter));
  const size_t widest = std::max({size_t(3) * c.heads * kHeadPad, ip, kconv, size_t(c.out_dim)});
  return {N * kPatchK, N * c.dim, N * 3 * c.heads * kHeadPad, kRows * widest, kRows * ip, L * kconv,
          (L + N / c.merge) * c.out_dim};
}

}  // namespace

Hy4VisionConfig Hy4VisionConfig::from_hf(const gufo::json::Value& top) {
  Hy4VisionConfig c;
  const auto* v = top.find("vision_config");
  if (!v || !v->is_object()) return c;
  const auto num = [&](const char* key) {
    const auto* x = v->find(key);
    if (!x) throw std::runtime_error(std::string("vision_config: missing ") + key);
    return x->as_double();
  };
  c.depth = int(num("num_hidden_layers"));
  c.dim = int(num("hidden_size"));
  c.heads = int(num("num_attention_heads"));
  c.head_dim = c.dim / c.heads;
  c.inter = int(num("intermediate_size"));
  c.out_dim = int(num("out_hidden_size"));
  c.patch = int(num("patch_size"));
  c.merge = int(num("spatial_merge_size"));
  c.edge = int(num("max_image_size")) / c.patch;
  c.max_patches = int(num("max_vit_seq_len"));
  c.eps = float(num("rms_norm_eps"));
  if (c.patch != 16 || c.merge != 2 || c.head_dim > kHeadPad || c.dim % 32)
    throw std::runtime_error("hy4 vision: unexpected shape (patch 16, merge 2, heads <= 80 dims)");
  return c;
}

size_t hy4_vision_weight_bytes(const Checkpoint& ck) {
  size_t b = 0;
  for (const auto& t : ck.tensors())
    if (t.name.rfind("vit.", 0) == 0) b += t.nbytes;
  return b + b / 8;  // the padded copies (qkv heads to 80, MLP to a multiple of 32)
}

Hy4VisionWeights load_hy4_vision(const Checkpoint& ck, const Hy4VisionConfig& c, int dev, hipStream_t st) {
  if (c.depth <= 0) throw std::runtime_error("load_hy4_vision: the config has no vision tower");
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
  // Host-side repacks into zero-padded layouts.
  auto upload = [&](const std::vector<uint16_t>& v) {
    rt::DeviceBuffer b(dev, v.size() * 2);
    UH_CHECK(hipMemcpy(b.data(), v.data(), v.size() * 2, hipMemcpyHostToDevice));
    return b;
  };
  const int D = c.dim, H = c.heads, hd = c.head_dim, I = c.inter;
  Hy4VisionWeights w;
  w.inter_pad = pad32(I);
  const int IP = w.inter_pad;
  w.patch_w = tensor("vit.embeddings.patch_embedding.weight");  // [dim][3][16][16] = [dim][768]
  if (w.patch_w.buf.size() != size_t(D) * kPatchK * 2) throw std::runtime_error("vit patch embedding: shape");
  w.patch_b = tensor("vit.embeddings.patch_embedding.bias");
  w.pos = tensor("vit.embeddings.position_embedding.weight");  // [1 + edge^2][dim], row 0 unused
  if (w.pos.buf.size() != size_t(1 + c.edge * c.edge) * D * 2) throw std::runtime_error("vit position table: shape");
  for (int i = 0; i < c.depth; ++i) {
    const std::string p = "vit.layers." + std::to_string(i) + ".";
    Hy4VisionWeights::Block b;
    b.ln1_w = tensor(p + "input_layernorm.weight");
    b.ln1_b = tensor(p + "input_layernorm.bias");
    b.o_w = tensor(p + "self_attn.o_proj.weight");
    b.o_b = tensor(p + "self_attn.o_proj.bias");
    b.ln2_w = tensor(p + "post_attention_layernorm.weight");
    b.ln2_b = tensor(p + "post_attention_layernorm.bias");
    b.fc2_b = tensor(p + "mlp.dense_4h_to_h.bias");
    {  // q, k, v: [3][H][80][D] with zero rows for dims hd..79
      std::vector<uint16_t> wq(size_t(3) * H * kHeadPad * D, 0), bq(size_t(3) * H * kHeadPad, 0);
      const char* names[3] = {"q_proj", "k_proj", "v_proj"};
      for (int j = 0; j < 3; ++j) {
        const auto ws = ck.as<uint16_t>(bf16(p + "self_attn." + names[j] + ".weight"));
        const auto bs = ck.as<uint16_t>(bf16(p + "self_attn." + names[j] + ".bias"));
        if (ws.size() != size_t(D) * D || bs.size() != size_t(D)) throw std::runtime_error(p + names[j] + ": shape");
        for (int h = 0; h < H; ++h)
          for (int d = 0; d < hd; ++d) {
            const size_t dst = (size_t(j) * H + h) * kHeadPad + d, src = size_t(h) * hd + d;
            std::copy_n(&ws[src * D], D, &wq[dst * D]);
            bq[dst] = bs[src];
          }
      }
      b.qkv_w = upload(wq);
      b.qkv_b = upload(bq);
    }
    {  // MLP padded to IP: fc1 rows and fc2 columns past I are zero
      const auto w1 = ck.as<uint16_t>(bf16(p + "mlp.dense_h_to_4h.weight"));
      const auto b1 = ck.as<uint16_t>(bf16(p + "mlp.dense_h_to_4h.bias"));
      const auto w2 = ck.as<uint16_t>(bf16(p + "mlp.dense_4h_to_h.weight"));
      if (w1.size() != size_t(I) * D || b1.size() != size_t(I) || w2.size() != size_t(D) * I)
        throw std::runtime_error(p + "mlp: shape");
      std::vector<uint16_t> f1(size_t(IP) * D, 0), fb(size_t(IP), 0), f2(size_t(D) * IP, 0);
      std::copy(w1.begin(), w1.end(), f1.begin());
      std::copy(b1.begin(), b1.end(), fb.begin());
      for (int r = 0; r < D; ++r) std::copy_n(&w2[size_t(r) * I], I, &f2[size_t(r) * IP]);
      b.fc1_w = upload(f1);
      b.fc1_b = upload(fb);
      b.fc2_w = upload(f2);
    }
    w.blocks.push_back(std::move(b));
  }
  w.norm = tensor("vit.perceive.norm.weight");
  w.conv1_w = tensor("vit.perceive.proj.0.weight");  // [4 dim][dim][2][2]: column c * 4 + i * 2 + j (vit_unfold's)
  w.conv1_b = tensor("vit.perceive.proj.0.bias");
  w.conv2_w = tensor("vit.perceive.proj.2.weight");  // [out][4 dim][1][1]
  w.conv2_b = tensor("vit.perceive.proj.2.bias");
  w.newline = tensor("vit.perceive.image_newline");
  if (w.conv1_w.buf.size() != size_t(4) * D * 4 * D * 2 || w.conv2_w.buf.size() != size_t(c.out_dim) * 4 * D * 2)
    throw std::runtime_error("vit perceive: shapes");
  Uploader up(ck);
  up.upload_many(std::move(items), st);
  up.sync();
  return w;
}

size_t hy4_vision_scratch_bytes(const Hy4VisionConfig& c, int max_patches) {
  const Sizes z = sizes(c, max_patches);
  return (z.patches + 3 * z.act + z.qkv + z.mlp + 2 * z.unfold + z.merged) * 2 + z.acc * 4 + size_t(max_patches) * 4;
}

Hy4VisionScratch make_hy4_vision_scratch(const Hy4VisionConfig& c, int max_patches, int dev) {
  const Sizes z = sizes(c, max_patches);
  Hy4VisionScratch s;
  s.max_patches = max_patches;
  rt::DeviceGuard g(dev);
  s.patches = rt::DeviceBuffer(dev, z.patches * 2);
  s.x = rt::DeviceBuffer(dev, z.act * 2);
  s.h = rt::DeviceBuffer(dev, z.act * 2);
  s.o = rt::DeviceBuffer(dev, z.act * 2);
  s.qkv = rt::DeviceBuffer(dev, z.qkv * 2);
  s.acc = rt::DeviceBuffer(dev, z.acc * 4);
  s.mlp = rt::DeviceBuffer(dev, z.mlp * 2);
  s.unfold = rt::DeviceBuffer(dev, z.unfold * 2);
  s.conv = rt::DeviceBuffer(dev, z.unfold * 2);
  s.merged = rt::DeviceBuffer(dev, z.merged * 2);
  s.src = rt::DeviceBuffer(dev, size_t(max_patches) * 4);
  return s;
}

void hy4_vision_encode(const Hy4VisionConfig& c, const Hy4VisionWeights& w, Hy4VisionScratch& s, int gh, int gw,
                       const std::vector<uint16_t>& patches, uint16_t* out, hipStream_t st) {
  const int m = c.merge, N = gh * gw, L = N / (m * m);
  const int D = c.dim, H = c.heads, IP = w.inter_pad, O = c.out_dim, QKV = 3 * H * kHeadPad, KC = m * m * D;
  if (N <= 0 || N > s.max_patches) throw std::runtime_error("hy4 vision: image larger than the scratch");
  if (gh % m || gw % m) throw std::runtime_error("hy4 vision: grid not a multiple of the merge size");
  if (patches.size() != size_t(N) * kPatchK) throw std::runtime_error("hy4 vision: patch data size");
  const int rows = gh / m, cols = gw / m + 1, T = rows * cols;
  {
    UH_CHECK(hipMemcpyAsync(s.patches.data(), patches.data(), patches.size() * 2, hipMemcpyHostToDevice, st));
    // The output rows: each merged row's cells, then the newline (params row 0).
    std::vector<int32_t> src(static_cast<size_t>(T));
    for (int i = 0; i < T; ++i) src[size_t(i)] = i % cols < cols - 1 ? (i / cols) * (cols - 1) + i % cols : -1;
    UH_CHECK(hipMemcpyAsync(s.src.data(), src.data(), src.size() * 4, hipMemcpyHostToDevice, st));
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
  linear(s.patches.as<uint16_t>(), kPatchK, N, w.patch_w.p<uint16_t>(), D, kPatchK,
         [&](int m0, int r) { k::vit_bias(acc, r, D, w.patch_b.p<uint16_t>(), x + size_t(m0) * D, st); });
  k::vit_add_pos_bilinear(x, w.pos.p<uint16_t>() + D, c.edge, gh, gw, D, st);
  const float scale = 1.0f / std::sqrt(float(c.head_dim));
  for (const auto& b : w.blocks) {
    k::vit_layernorm(x, N, D, b.ln1_w.p<uint16_t>(), b.ln1_b.p<uint16_t>(), c.eps, h, st);
    linear(h, D, N, b.qkv_w.as<uint16_t>(), QKV, D, [&](int m0, int r) {
      k::vit_bias(acc, r, QKV, b.qkv_b.as<uint16_t>(), qkv + size_t(m0) * QKV, st);
    });
    k::vit_attention_wmma80(qkv, N, H, c.head_dim, scale, o, st);
    linear(o, D, N, b.o_w.p<uint16_t>(), D, D,
           [&](int m0, int r) { k::vit_bias(acc, r, D, b.o_b.p<uint16_t>(), h + size_t(m0) * D, st); });
    k::vit_add(x, h, size_t(N) * D, st);
    k::vit_layernorm(x, N, D, b.ln2_w.p<uint16_t>(), b.ln2_b.p<uint16_t>(), c.eps, h, st);
    linear(h, D, N, b.fc1_w.as<uint16_t>(), IP, D, [&](int m0, int r) {
      k::vit_bias_gelu(acc, r, IP, b.fc1_b.as<uint16_t>(), mlp, st);
      k::gemm_bf16_f32out_wmma(mlp, IP, b.fc2_w.as<uint16_t>(), acc, D, r, D, IP, st);
      k::vit_bias(acc, r, D, b.fc2_b.p<uint16_t>(), o + size_t(m0) * D, st);
    });
    k::vit_add(x, o, size_t(N) * D, st);
  }
  // The projector: RMSNorm (transformers' order), the 2 x 2 stride-2 conv as a GEMM over unfolded cells, GELU,
  // the 1 x 1 conv, then the rows with a newline after each merged row.
  k::vit_rmsnorm_hf(x, N, D, w.norm.p<uint16_t>(), c.eps, h, st);
  k::vit_unfold(h, gh, gw, D, m, s.unfold.as<uint16_t>(), st);
  linear(s.unfold.as<uint16_t>(), KC, L, w.conv1_w.p<uint16_t>(), KC, KC, [&](int m0, int r) {
    k::vit_bias_gelu(acc, r, KC, w.conv1_b.p<uint16_t>(), s.conv.as<uint16_t>() + size_t(m0) * KC, st);
  });
  linear(s.conv.as<uint16_t>(), KC, L, w.conv2_w.p<uint16_t>(), O, KC, [&](int m0, int r) {
    k::vit_bias(acc, r, O, w.conv2_b.p<uint16_t>(), s.merged.as<uint16_t>() + size_t(m0) * O, st);
  });
  k::vit_assemble(s.merged.as<uint16_t>(), w.newline.p<uint16_t>(), s.src.as<int32_t>(), T, O, out, st);
}

}  // namespace unholy::model
