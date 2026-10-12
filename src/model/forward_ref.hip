#include "model/forward_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/attn_wmma.hpp"
#include "kernels/exl3_moe.hpp"
#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/hc_fast.hpp"
#include "kernels/indexer.hpp"
#include "kernels/ref.hpp"

namespace unholy::model {

namespace k = unholy::kernels;
using bf16 = uint16_t;

// ---------------------------------------------------------------------------------------------------
// Caches and scratch

LayerCache make_layer_cache(const Dsv4Config& c, const LayerWeights& w, const CacheGeometry& g, int dev,
                            hipStream_t st) {
  LayerCache lc;
  lc.layer = w.layer;
  lc.ratio = w.ratio;
  lc.indexer = w.has_indexer;
  lc.slots = g.max_slots;
  lc.window = c.window;
  rt::DeviceGuard guard(dev);
  lc.window_rows = rt::DeviceBuffer(dev, size_t(g.max_slots) * c.window * c.head_dim * 2);
  UH_CHECK(hipMemsetAsync(lc.window_rows.data(), 0, lc.window_rows.size(), st));
  auto alloc_zero = [&](size_t bytes) {
    rt::DeviceBuffer b(dev, bytes);
    UH_CHECK(hipMemsetAsync(b.data(), 0, bytes, st));
    return b;
  };
  if (w.ratio) {
    lc.ring = w.comp->ring();
    lc.width = w.comp->width();
    lc.ring_kv = alloc_zero(size_t(g.max_slots) * lc.ring * lc.width * 4);
    lc.ring_sc = alloc_zero(size_t(g.max_slots) * lc.ring * lc.width * 4);
    lc.comp = rt::DeviceBuffer(dev, size_t(g.num_pages) * (kPageTokens / w.ratio) * c.head_dim * 2);
  }
  if (w.has_indexer) {
    lc.iring = w.idx_comp->ring();
    lc.iwidth = w.idx_comp->width();
    lc.iring_kv = alloc_zero(size_t(g.max_slots) * lc.iring * lc.iwidth * 4);
    lc.iring_sc = alloc_zero(size_t(g.max_slots) * lc.iring * lc.iwidth * 4);
    lc.idx = rt::DeviceBuffer(dev, size_t(g.num_pages) * k::kFp4PageBytes);  // packed fp4 keys
  }
  UH_CHECK(hipStreamSynchronize(st));
  return lc;
}

namespace {

enum Buf : size_t {
  kY, kA, kPost, kComb, kQBuf, kQSc, kQA, kQR, kQ, kKV, kAF, kPK, kPS, kRowsF, kRowsB, kRowPos, kQI, kWI,
  kISc, kIdx, kO, kOA, kAO, kLG, kSel, kWt, kYF, kG, kH1, kH3, kHH, kPairOut, kPairIdx, kShared, kEIdx, kEWt,
  kY2F, kPairIn, kHcPart, kQC, kQCS, kNumBufs
};

template <typename T>
T* B(const Scratch& s, Buf b) {
  return static_cast<T*>(s.get(b));
}

}  // namespace

namespace {
constexpr size_t kIndexerScoreBytes = size_t(256) << 20;
std::vector<size_t> scratch_sizes(const Dsv4Config& c, int max_tokens, int max_seq) {
  const size_t N = size_t(max_tokens);
  const size_t rows = N / 4 + 2;
  const size_t keys = size_t(max_seq / 4 + 1);
  const size_t K = size_t(c.n_active);
  std::vector<size_t> sz(kNumBufs, 0);
  sz[kY] = N * c.dim * 2;
  sz[kA] = N * c.dim * 2;
  sz[kPost] = N * c.hc_mult * 4;
  sz[kComb] = N * c.hc_mult * c.hc_mult * 4;
  sz[kQBuf] = N * size_t(c.o_groups * c.o_lora);  // widest quantized activation (8192)
  sz[kQSc] = N * size_t(c.o_groups * c.o_lora / 32);
  sz[kQA] = N * c.q_lora * 2;
  sz[kQR] = N * c.q_lora * 2;
  sz[kQ] = N * c.n_heads * c.head_dim * 2;
  sz[kKV] = N * c.head_dim * 2;
  sz[kAF] = N * c.dim * 4;
  sz[kPK] = N * 2 * c.head_dim * 4;
  sz[kPS] = N * 2 * c.head_dim * 4;
  sz[kRowsF] = rows * c.head_dim * 4;
  sz[kRowsB] = rows * c.head_dim * 2;
  sz[kRowPos] = rows * 4;
  sz[kQI] = N * c.idx_heads * c.idx_dim * 2;
  sz[kWI] = N * c.idx_heads * 2;
  // Indexer scores for a sub-batch of a chunk's queries (fp32 rows on the reference path, bf16 on the
  // fast one): capped, the chunk's queries go through it in turns.
  sz[kISc] = std::max(keys * 4, std::min(N * keys * 4, kIndexerScoreBytes));
  sz[kQC] = N * c.idx_heads * c.idx_dim;
  sz[kQCS] = N * c.idx_heads * (c.idx_dim / 32);
  sz[kIdx] = N * c.idx_topk * 4;
  sz[kO] = N * c.n_heads * c.head_dim * 2;
  sz[kOA] = N * c.o_groups * c.o_lora * 2;
  sz[kAO] = N * c.dim * 2;
  sz[kLG] = N * c.n_experts * 4;
  sz[kSel] = N * K * 4;
  sz[kWt] = N * K * 4;
  sz[kYF] = N * c.dim * 4;
  sz[kG] = N * c.dim * 2;
  sz[kH1] = N * c.moe_inter * 2;
  sz[kH3] = N * c.moe_inter * 2;
  sz[kHH] = N * c.moe_inter * 2;
  sz[kPairOut] = N * K * c.dim * 2;
  sz[kPairIdx] = N * K * 4;
  sz[kShared] = N * c.dim * 2;
  sz[kEIdx] = N * K * 4;
  sz[kEWt] = N * K * 4;
  sz[kY2F] = N * c.dim * 4;
  sz[kPairIn] = N * K * c.dim * 2;
  sz[kHcPart] = size_t(32) * 32 * 25 * 4;
  return sz;
}
}  // namespace

size_t expert_scratch_bytes(const Dsv4Config& c, int rows) {
  const size_t R = size_t(std::max(rows, 1)), W = size_t(std::max(c.dim, c.moe_inter));
  return R * W + R * W / 32 + 3 * R * c.moe_inter * 2 + 2 * R * W * 4;
}

size_t exl3_moe_scratch_bytes(const Dsv4Config& c, int rows) {
  const size_t R = size_t(std::max(rows, 1));
  return R * c.dim * 4 + R * c.moe_inter * 6 + R * 4 + size_t(rows / 64 + c.n_experts + 1) * 12;
}

size_t scratch_bytes(const Dsv4Config& c, int max_tokens, int max_seq) {
  size_t total = expert_scratch_bytes(c, max_tokens) + exl3_moe_scratch_bytes(c, max_tokens * c.n_active);
  for (size_t b : scratch_sizes(c, max_tokens, max_seq)) total += std::max<size_t>(b, 256);
  return total;
}

size_t layer_cache_bytes(const Dsv4Config& c, int ratio, bool indexer, const CacheGeometry& g) {
  size_t b = size_t(g.max_slots) * c.window * c.head_dim * 2;
  if (ratio) {
    const size_t width = size_t(ratio == 4 ? 2 : 1) * c.head_dim, ring = size_t(ratio == 4 ? 8 : ratio);
    b += 2 * size_t(g.max_slots) * ring * width * 4;
    b += size_t(g.num_pages) * (kPageTokens / ratio) * c.head_dim * 2;
  }
  if (indexer) {
    b += 2 * size_t(g.max_slots) * 8 * (2 * c.idx_dim) * 4;
    b += size_t(g.num_pages) * k::kFp4PageBytes;
  }
  return b;
}

Scratch make_scratch(const Dsv4Config& c, int dev, int max_tokens, int max_seq) {
  const std::vector<size_t> sz = scratch_sizes(c, max_tokens, max_seq);
  Scratch s;
  s.max_tokens = max_tokens;
  s.max_keys = max_seq / 4 + 1;
  for (size_t i = 0; i < kNumBufs; ++i) s.bufs.emplace_back(dev, std::max<size_t>(sz[i], 256));
  s.expert = make_expert_scratch(c, dev, max_tokens);
  s.exl3 = make_exl3_moe_scratch(c, dev, max_tokens * c.n_active);
  s.dispatch.max_tiles = s.exl3.max_tiles;
  s.dispatch.count = rt::DeviceBuffer(dev, size_t(c.n_experts) * 4);
  s.dispatch.off = rt::DeviceBuffer(dev, size_t(c.n_experts) * 4);
  s.dispatch.fill = rt::DeviceBuffer(dev, size_t(c.n_experts) * 4);
  s.dispatch.plan = rt::DeviceBuffer(dev, sizeof(kernels::MoePlan));
  s.dispatch.ltiles = rt::DeviceBuffer(dev, size_t(s.dispatch.max_tiles) * sizeof(kernels::Exl3Tile));
  s.dispatch.rtiles = rt::DeviceBuffer(dev, size_t(s.dispatch.max_tiles) * sizeof(kernels::Exl3Tile));
  return s;
}

ExpertScratch make_expert_scratch(const Dsv4Config& c, int dev, int rows) {
  const size_t R = size_t(std::max(rows, 1)), W = size_t(std::max(c.dim, c.moe_inter));
  ExpertScratch s;
  s.rows = rows;
  s.qb = rt::DeviceBuffer(dev, R * W);
  s.qs = rt::DeviceBuffer(dev, R * W / 32);
  s.h1 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.h3 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.hh = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.xf = rt::DeviceBuffer(dev, R * W * 4);
  s.y2 = rt::DeviceBuffer(dev, R * W * 4);
  return s;
}

void expert_forward(const Dsv4Config& c, const LayerWeights& w, int e, uint16_t* in, const float* rw, int cnt,
                    uint16_t* out, const ExpertScratch& s, hipStream_t st) {
  const int D = c.dim, I = c.moe_inter;
  if (cnt > s.rows) throw std::runtime_error("expert_forward: more rows than scratch");
  bf16* h1 = s.h1.as<bf16>();
  bf16* h3 = s.h3.as<bf16>();
  bf16* hh = s.hh.as<bf16>();
  if (w.exl3_experts) {
    // EXL3 weights on the fp8-quantized activations the experts were trained with (A8 fake quant).
    float* xf = s.xf.as<float>();
    float* y2 = s.y2.as<float>();
    auto lin = [&](const Exl3Weights& xw, const bf16* x, int in_dim, bf16* y, int out_dim) {
      k::exl3_in(x, in_dim, xw.suh.p<uint16_t>(), xf, cnt, in_dim, st);
      k::gemm_exl3_inner(xf, xw.trellis.p<uint16_t>(), xw.bits, y2, cnt, out_dim, in_dim, st);
      k::exl3_out(y2, xw.svh.p<uint16_t>(), y, out_dim, cnt, out_dim, st);
    };
    k::fake_quant_fp8(in, D, cnt, D, 128, st);
    lin(w.x1[size_t(e)], in, D, h1, I);
    lin(w.x3[size_t(e)], in, D, h3, I);
    k::swiglu(h1, h3, rw, hh, cnt, I, c.swiglu_limit, st);
    k::fake_quant_fp8(hh, I, cnt, I, 128, st);
    lin(w.x2[size_t(e)], hh, I, out, D);
  } else {
    uint8_t* qb = s.qb.as<uint8_t>();
    uint8_t* qs = s.qs.as<uint8_t>();
    k::act_quant_fp8(in, D, qb, qs, cnt, D, 128, st);
    k::gemm_a8w4(qb, qs, w.w1[size_t(e)].w.p<uint8_t>(), w.w1[size_t(e)].s.p<uint8_t>(), h1, I, cnt, I, D, st);
    k::gemm_a8w4(qb, qs, w.w3[size_t(e)].w.p<uint8_t>(), w.w3[size_t(e)].s.p<uint8_t>(), h3, I, cnt, I, D, st);
    k::swiglu(h1, h3, rw, hh, cnt, I, c.swiglu_limit, st);
    k::act_quant_fp8(hh, I, qb, qs, cnt, I, 128, st);
    k::gemm_a8w4(qb, qs, w.w2[size_t(e)].w.p<uint8_t>(), w.w2[size_t(e)].s.p<uint8_t>(), out, D, cnt, D, I, st);
  }
}

Exl3MoeScratch make_exl3_moe_scratch(const Dsv4Config& c, int dev, int rows) {
  const size_t R = size_t(std::max(rows, 1));
  Exl3MoeScratch s;
  s.rows = rows;
  s.max_tiles = rows / 64 + c.n_experts + 1;
  s.xh1 = rt::DeviceBuffer(dev, R * c.dim * 2);
  s.xh3 = rt::DeviceBuffer(dev, R * c.dim * 2);
  s.h1 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.h3 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.xh2 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.tiles = rt::DeviceBuffer(dev, size_t(s.max_tiles) * sizeof(kernels::Exl3Tile));
  s.row_expert = rt::DeviceBuffer(dev, R * 4);
  s.yp = rt::DeviceBuffer(dev, R * size_t(std::max(c.dim, c.moe_inter)) * 4);
  return s;
}

bool exl3_fast_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_REF_EXL3");
    return !v || std::string(v) == "0";
  }();
  return on;
}

std::vector<kernels::Exl3Tile> exl3_tiles(const std::vector<std::array<int, 3>>& runs) {
  std::vector<kernels::Exl3Tile> t;
  for (const auto& r : runs)
    for (int i = 0; i < r[2]; i += 64) t.push_back({r[0], r[1] + i, std::min(64, r[2] - i)});
  return t;
}

void exl3_moe_forward(const Dsv4Config& c, const LayerWeights& w, const uint16_t* in, const float* rw,
                      const int32_t* row_expert, const std::vector<kernels::Exl3Tile>& tiles, int P,
                      const Exl3MoeScratch& s, uint16_t* out, hipStream_t st) {
  if (tiles.empty()) return;
  if (int(tiles.size()) > s.max_tiles || P > s.rows) throw std::runtime_error("exl3_moe_forward: scratch too small");
  const int D = c.dim, I = c.moe_inter, E = c.n_experts;
  auto* dt = s.tiles.as<kernels::Exl3Tile>();
  UH_CHECK(hipMemcpyAsync(dt, tiles.data(), tiles.size() * sizeof(kernels::Exl3Tile), hipMemcpyHostToDevice, st));
  const int nt = int(tiles.size());
  uint16_t* xh1 = s.xh1.as<uint16_t>();
  uint16_t* xh3 = s.xh3.as<uint16_t>();
  uint16_t* xh2 = s.xh2.as<uint16_t>();
  bf16* h1 = s.h1.as<bf16>();
  bf16* h3 = s.h3.as<bf16>();
  k::exl3_xform_in(in, D, row_expert, w.exl3_ptrs(0, 1, E), xh1, P, D, true, st);
  k::exl3_xform_in(in, D, row_expert, w.exl3_ptrs(1, 1, E), xh3, P, D, true, st);
  k::exl3_grouped_gemm(xh1, D, dt, nt, w.exl3_ptrs(0, 0, E), w.exl3_ptrs(0, 2, E), w.exl3_bits[0], h1, I, I, st);
  k::exl3_grouped_gemm(xh3, D, dt, nt, w.exl3_ptrs(1, 0, E), w.exl3_ptrs(1, 2, E), w.exl3_bits[1], h3, I, I, st);
  k::exl3_swiglu_xform(h1, h3, rw, row_expert, w.exl3_ptrs(2, 1, E), xh2, P, I, c.swiglu_limit, st);
  k::exl3_grouped_gemm(xh2, I, dt, nt, w.exl3_ptrs(2, 0, E), w.exl3_ptrs(2, 2, E), w.exl3_bits[2], out, D, D, st);
}

void exl3_moe_forward_dev(const Dsv4Config& c, const LayerWeights& w, const uint16_t* in, const float* rw,
                          const int32_t* row_expert, const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev,
                          int P, const Exl3MoeScratch& s, uint16_t* out, hipStream_t st) {
  if (P > s.rows) throw std::runtime_error("exl3_moe_forward_dev: scratch too small");
  const int D = c.dim, I = c.moe_inter, E = c.n_experts;
  max_tiles = std::min(max_tiles, P);  // every tile has >= 1 row: bounds the grids at small batches
  uint16_t* xh1 = s.xh1.as<uint16_t>();
  uint16_t* xh3 = s.xh3.as<uint16_t>();
  uint16_t* xh2 = s.xh2.as<uint16_t>();
  bf16* h1 = s.h1.as<bf16>();
  bf16* h3 = s.h3.as<bf16>();
  k::exl3_xform_in(in, D, row_expert, w.exl3_ptrs(0, 1, E), xh1, P, D, true, st);
  k::exl3_xform_in(in, D, row_expert, w.exl3_ptrs(1, 1, E), xh3, P, D, true, st);
  if (P / c.n_active <= 8) {  // <= 8 tokens: every expert has <= 8 rows -> GEMV kernels
    float* yp = s.yp.as<float>();
    k::exl3_grouped_gemv(xh1, D, tiles, max_tiles, w.exl3_ptrs(0, 0, E), w.exl3_ptrs(0, 2, E), w.exl3_bits[0], yp, h1,
                         I, I, st, ntiles_dev);
    k::exl3_grouped_gemv(xh3, D, tiles, max_tiles, w.exl3_ptrs(1, 0, E), w.exl3_ptrs(1, 2, E), w.exl3_bits[1], yp, h3,
                         I, I, st, ntiles_dev);
    k::exl3_swiglu_xform(h1, h3, rw, row_expert, w.exl3_ptrs(2, 1, E), xh2, P, I, c.swiglu_limit, st);
    k::exl3_grouped_gemv(xh2, I, tiles, max_tiles, w.exl3_ptrs(2, 0, E), w.exl3_ptrs(2, 2, E), w.exl3_bits[2], yp, out,
                         D, D, st, ntiles_dev);
    return;
  }
  k::exl3_grouped_gemm(xh1, D, tiles, max_tiles, w.exl3_ptrs(0, 0, E), w.exl3_ptrs(0, 2, E), w.exl3_bits[0], h1, I, I,
                       st, ntiles_dev);
  k::exl3_grouped_gemm(xh3, D, tiles, max_tiles, w.exl3_ptrs(1, 0, E), w.exl3_ptrs(1, 2, E), w.exl3_bits[1], h3, I, I,
                       st, ntiles_dev);
  k::exl3_swiglu_xform(h1, h3, rw, row_expert, w.exl3_ptrs(2, 1, E), xh2, P, I, c.swiglu_limit, st);
  k::exl3_grouped_gemm(xh2, I, tiles, max_tiles, w.exl3_ptrs(2, 0, E), w.exl3_ptrs(2, 2, E), w.exl3_bits[2], out, D, D,
                       st, ntiles_dev);
}

namespace {
std::vector<std::vector<uint64_t>>* g_expert_hist = nullptr;
}
void set_expert_histogram(std::vector<std::vector<uint64_t>>* hist) { g_expert_hist = hist; }

// ---------------------------------------------------------------------------------------------------
// Forward

namespace {

// Dense dispatch: WMMA kernels unless UNHOLY_REF_DENSE=1 (fp8 WMMA needs gfx12).
bool fast_dense() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_REF_DENSE");
    return !v || std::string(v) == "0";
  }();
  return on;
}
bool on_gfx12() {
  static int arch[64] = {};
  int dev = 0;
  UH_CHECK(hipGetDevice(&dev));
  if (!arch[dev]) {
    hipDeviceProp_t p{};
    UH_CHECK(hipGetDeviceProperties(&p, dev));
    arch[dev] = std::string(p.gcnArchName).rfind("gfx12", 0) == 0 ? 2 : 1;
  }
  return arch[dev] == 2;
}
void dense_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
                int N, int K, hipStream_t st) {
  if (fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) k::gemv_w8a8(a, sa, w, sw, c, ldc, M, N, K, st);
  else if (fast_dense() && on_gfx12()) k::gemm_w8a8_wmma(a, sa, w, sw, c, ldc, M, N, K, st);
  else k::gemm_w8a8(a, sa, w, sw, c, ldc, M, N, K, st);
}
// The reference's fp32 linears on x.float() of a bf16 tensor: bf16 WMMA on the bf16 source is exact in
// the products; the reference kernel takes the fp32 copy.
void dense_f32(const bf16* a_bf16, const float* a_f32, int lda, const bf16* w, float* c, int ldc, int M, int N,
               int K, hipStream_t st) {
  if (fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) k::gemv_bf16_f32out(a_bf16, lda, w, c, ldc, M, N, K, st);
  else if (fast_dense()) k::gemm_bf16_f32out_wmma(a_bf16, lda, w, c, ldc, M, N, K, st);
  else k::gemm_f32(a_f32, lda, w, c, ldc, M, N, K, st);
}
void dense_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) k::gemv_bf16(a, lda, w, c, ldc, M, N, K, st);
  else if (fast_dense()) k::gemm_bf16_wmma(a, lda, w, c, ldc, M, N, K, st);
  else k::gemm_bf16(a, lda, w, c, ldc, M, N, K, st);
}
void dense_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                        hipStream_t st) {
  if (fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) k::gemv_grouped_fp8w(a, w, sw, c, M, G, R, K, st);
  else if (fast_dense()) k::gemm_grouped_fp8w_wmma(a, w, sw, c, M, G, R, K, st);
  else k::gemm_grouped_fp8w(a, w, sw, c, M, G, R, K, st);
}

// Attention dispatch: WMMA kernel on gfx12 unless UNHOLY_REF_ATTN=1.
void attention(const bf16* q, const int32_t* pos, k::AttnKv kv, const float* sink, bf16* o, int T, int H, int D,
               float scale, hipStream_t st) {
  static const bool fast = [] {
    const char* v = std::getenv("UNHOLY_REF_ATTN");
    return !v || std::string(v) == "0";
  }();
  if (fast && on_gfx12()) k::sparse_attn_wmma(q, pos, kv, sink, o, T, H, D, scale, st);
  else k::sparse_attn(q, pos, kv, sink, o, T, H, D, scale, st);
}

// Hyper-Connection pre-mix dispatch: token-tiled kernel unless UNHOLY_REF_HC=1.
void hc_pre(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post, float* comb,
            float* part, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st) {
  static const bool fast = [] {
    const char* v = std::getenv("UNHOLY_REF_HC");
    return !v || std::string(v) == "0";
  }();
  if (fast && T <= k::kHcSmallMaxTokens)
    k::hc_pre_small(x, fn, scale, base, y, post, comb, part, T, hc, dim, iters, norm_eps, hc_eps, st);
  else if (fast) k::hc_pre_tiled(x, fn, scale, base, y, post, comb, T, hc, dim, iters, norm_eps, hc_eps, st);
  else k::hc_pre(x, fn, scale, base, y, post, comb, T, hc, dim, iters, norm_eps, hc_eps, st);
}

// UNHOLY_DUMP=dir: raw intermediates of batches whose first chunk starts at position 0, named like
// the oracle's captures (tools/ref/oracle/run.py --capture-layer), for tools/ref/compare_dump.py.
void dump(int layer, const BatchDesc& b, const char* name, const void* dptr, size_t bytes, hipStream_t st) {
  static const char* dir = std::getenv("UNHOLY_DUMP");
  if (!dir || b.chunks.empty() || b.chunks[0].s0 != 0) return;
  std::vector<char> h(bytes);
  UH_CHECK(hipStreamSynchronize(st));
  UH_CHECK(hipMemcpy(h.data(), dptr, bytes, hipMemcpyDeviceToHost));
  char path[512];
  std::snprintf(path, sizeof path, "%s/L%03d_%s.bin", dir, layer, name);
  if (FILE* f = std::fopen(path, "wb")) {
    std::fwrite(h.data(), 1, bytes, f);
    std::fclose(f);
  }
}

k::PagedRows paged(rt::DeviceBuffer& pool, const BatchDesc& b, int slot, int rows_per_page, int d) {
  return k::PagedRows{pool.as<bf16>(), b.block_tables + size_t(slot) * b.max_pages_per_seq, rows_per_page, d};
}
k::PagedFp4 paged_fp4(rt::DeviceBuffer& pool, const BatchDesc& b, int slot) {
  return k::PagedFp4{pool.as<uint8_t>(), b.block_tables + size_t(slot) * b.max_pages_per_seq};
}

// Indexer dispatch: int8-WMMA scores over fp4 codes on gfx12 unless UNHOLY_REF_INDEXER=1.
bool fast_indexer() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_REF_INDEXER");
    return !v || std::string(v) == "0";
  }();
  return on && on_gfx12();
}

// Compressor rows for one chunk, given the batch-wide projections (pk/ps already include ape):
// emitted rows are normed, roped at their block start, quant-simulated and stored to the pool.
void compress_chunk(const Dsv4Config& c, const CompressorWeights& cw, const float* pk, const float* ps,
                    float* ring_kv, float* ring_sc, k::PagedRows dst, k::PagedFp4 dst4, const Scratch& scr,
                    const RopeTables& rope, const ChunkDesc& ch, hipStream_t st) {
  const int width = cw.width(), d = cw.d;
  const float* cpk = pk + size_t(ch.t0) * width;
  const float* cps = ps + size_t(ch.t0) * width;
  const int row0 = ch.s0 / cw.ratio, nrows = (ch.s0 + ch.n) / cw.ratio - row0;
  if (nrows > 0) {
    float* rf = B<float>(scr, kRowsF);
    bf16* rb = B<bf16>(scr, kRowsB);
    k::compress_rows(cpk, cps, ring_kv, ring_sc, rf, ch.s0, ch.n, row0, nrows, cw.ratio, d, cw.overlap, cw.ring(),
                     st);
    k::cast_f32_bf16(rf, rb, size_t(nrows) * d, st);
    k::rmsnorm(rb, d, cw.norm.p<bf16>(), rb, d, nrows, d, c.norm_eps, st);
    int32_t* dpos = B<int32_t>(scr, kRowPos);
    k::positions(dpos, nrows, row0 * cw.ratio, cw.ratio, st);
    k::rope(rb, d, nrows, 1, d, d - c.rope_dim, c.rope_dim / 2, dpos, rope.yarn.as<float2>(), false, st);
    if (cw.rotate) {  // indexer keys: Hadamard, then fp4 codes + scales (the reference's fake quant, stored)
      k::hadamard128(rb, size_t(nrows) * d / 128, st);
      k::store_rows_fp4(rb, nrows, dst4, row0, st);
    } else {
      k::fake_quant_fp8(rb, d, nrows, d - c.rope_dim, 64, st);
      k::store_rows(rb, nrows, d, dst, row0, st);
    }
  }
  k::ring_update(cpk, cps, ring_kv, ring_sc, ch.s0, ch.n, cw.ring(), width, st);
}

// Batch-wide compressor projections: x_f32 [T, dim] -> pk/ps [T, width] (+ ape by position).
void compressor_proj(const Dsv4Config& c, const CompressorWeights& cw, const Scratch& scr, const BatchDesc& b,
                     hipStream_t st) {
  const int width = cw.width();
  float* pk = B<float>(scr, kPK);
  float* ps = B<float>(scr, kPS);
  dense_f32(B<bf16>(scr, kA), B<float>(scr, kAF), c.dim, cw.wkv.p<bf16>(), pk, width, b.T, width, c.dim, st);
  dense_f32(B<bf16>(scr, kA), B<float>(scr, kAF), c.dim, cw.wgate.p<bf16>(), ps, width, b.T, width, c.dim, st);
  for (const auto& ch : b.chunks) k::add_ape(ps + size_t(ch.t0) * width, cw.ape.p<float>(), ch.s0, ch.n, cw.ratio, width, st);
}

void attention_half(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                    const RopeTables& rope, bf16* x, const BatchDesc& b, hipStream_t st) {
  const int D = c.dim, H = c.n_heads, HD = c.head_dim, rd = c.rope_dim, T = b.T, L = w.layer;
  const float2* freqs = (w.ratio ? rope.yarn : rope.plain).as<float2>();
  const int32_t* pos = b.pos;
  bf16* y = B<bf16>(scr, kY);
  bf16* a = B<bf16>(scr, kA);
  float* post = B<float>(scr, kPost);
  float* comb = B<float>(scr, kComb);
  uint8_t* qb = B<uint8_t>(scr, kQBuf);
  uint8_t* qs = B<uint8_t>(scr, kQSc);

  hc_pre(x, w.hc_attn_fn.p<float>(), w.hc_attn_scale.p<float>(), w.hc_attn_base.p<float>(), y, post, comb, B<float>(scr, kHcPart), T,
            c.hc_mult, D, c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  dump(L, b, "hcpre_attn_y", y, size_t(T) * D * 2, st);
  dump(L, b, "hcpre_attn_post", post, size_t(T) * c.hc_mult * 4, st);
  dump(L, b, "hcpre_attn_comb", comb, size_t(T) * c.hc_mult * c.hc_mult * 4, st);
  k::rmsnorm(y, D, w.attn_norm.p<bf16>(), a, D, T, D, c.norm_eps, st);
  dump(L, b, "attn_in", a, size_t(T) * D * 2, st);

  // q: wq_a -> q_norm (qr) -> wq_b -> per-head rms -> rope; kv: wkv from the same quantized input.
  bf16* qa = B<bf16>(scr, kQA);
  bf16* qr = B<bf16>(scr, kQR);
  bf16* q = B<bf16>(scr, kQ);
  bf16* kv = B<bf16>(scr, kKV);
  k::act_quant_fp8(a, D, qb, qs, T, D, 128, st);
  dense_w8a8(qb, qs, w.wq_a.w.p<uint8_t>(), w.wq_a.s.p<uint8_t>(), qa, c.q_lora, T, c.q_lora, D, st);
  dump(L, b, "wq_a_out", qa, size_t(T) * c.q_lora * 2, st);
  dense_w8a8(qb, qs, w.wkv.w.p<uint8_t>(), w.wkv.s.p<uint8_t>(), kv, HD, T, HD, D, st);
  dump(L, b, "wkv_out", kv, size_t(T) * HD * 2, st);
  k::rmsnorm(qa, c.q_lora, w.q_norm.p<bf16>(), qr, c.q_lora, T, c.q_lora, c.norm_eps, st);
  k::act_quant_fp8(qr, c.q_lora, qb, qs, T, c.q_lora, 128, st);
  dense_w8a8(qb, qs, w.wq_b.w.p<uint8_t>(), w.wq_b.s.p<uint8_t>(), q, H * HD, T, H * HD, c.q_lora, st);
  dump(L, b, "wq_b_out", q, size_t(T) * H * HD * 2, st);
  k::head_rms_bf16(q, T, H, HD, c.norm_eps, st);
  k::rope(q, H * HD, T, H, HD, HD - rd, rd / 2, pos, freqs, false, st);
  k::rmsnorm(kv, HD, w.kv_norm.p<bf16>(), kv, HD, T, HD, c.norm_eps, st);
  k::rope(kv, HD, T, 1, HD, HD - rd, rd / 2, pos, freqs, false, st);
  k::fake_quant_fp8(kv, HD, T, HD - rd, 64, st);

  // Indexer inputs (token-parallel); per-chunk scores/top-k below.
  bf16* qi = B<bf16>(scr, kQI);
  bf16* wi = B<bf16>(scr, kWI);
  const int IH = c.idx_heads, ID = c.idx_dim;
  if (w.ratio) k::cast_bf16_f32(a, B<float>(scr, kAF), size_t(T) * D, st);
  if (w.has_indexer) {
    dense_w8a8(qb, qs, w.idx_wq_b.w.p<uint8_t>(), w.idx_wq_b.s.p<uint8_t>(), qi, IH * ID, T, IH * ID, c.q_lora,
                 st);
    k::rope(qi, IH * ID, T, IH, ID, ID - rd, rd / 2, pos, freqs, false, st);
    k::hadamard128(qi, size_t(T) * IH * ID / 128, st);
    if (fast_indexer()) k::quant_q_fp4(qi, T, IH, B<int8_t>(scr, kQC), B<uint8_t>(scr, kQCS), st);
    else k::fake_quant_fp4(qi, IH * ID, T, IH * ID, 32, st);
    dense_bf16(a, D, w.idx_wproj.p<bf16>(), wi, IH, T, IH, D, st);
    // weights_proj(x) * (softmax_scale * n_heads^-0.5): the Python double, applied as fp32.
    const double wscale = (1.0 / std::sqrt(double(ID))) * (1.0 / std::sqrt(double(IH)));
    k::scale_bf16(wi, float(wscale), size_t(T) * IH, st);
    // Indexer compressor (projections batch-wide, rows per chunk).
    compressor_proj(c, *w.idx_comp, scr, b, st);
    for (const auto& ch : b.chunks)
      compress_chunk(c, *w.idx_comp, B<float>(scr, kPK), B<float>(scr, kPS),
                     cache.iring_kv.as<float>() + size_t(ch.slot) * cache.iring * cache.iwidth,
                     cache.iring_sc.as<float>() + size_t(ch.slot) * cache.iring * cache.iwidth,
                     k::PagedRows{}, paged_fp4(cache.idx, b, ch.slot), scr, rope, ch, st);
  }
  if (w.ratio) {
    compressor_proj(c, *w.comp, scr, b, st);
    for (const auto& ch : b.chunks)
      compress_chunk(c, *w.comp, B<float>(scr, kPK), B<float>(scr, kPS),
                     cache.ring_kv.as<float>() + size_t(ch.slot) * cache.ring * cache.width,
                     cache.ring_sc.as<float>() + size_t(ch.slot) * cache.ring * cache.width,
                     paged(cache.comp, b, ch.slot, kPageTokens / w.ratio, HD), k::PagedFp4{}, scr, rope, ch, st);
  }

  // Per chunk: indexer top-k, sparse attention, window ring.
  bf16* o = B<bf16>(scr, kO);
  const float scale = float(1.0 / std::sqrt(double(HD)));
  dump(L, b, "sa_q", q, size_t(T) * H * HD * 2, st);
  dump(L, b, "sa_kv", kv, size_t(T) * HD * 2, st);
  for (const auto& ch : b.chunks) {
    const int32_t* cpos = pos + ch.t0;
    bf16* window = cache.window_rows.as<bf16>() + size_t(ch.slot) * c.window * HD;
    k::AttnKv akv;
    akv.chunk_kv = kv + size_t(ch.t0) * HD;
    akv.ring = window;
    akv.s0 = ch.s0;
    akv.ratio = w.ratio;
    if (w.ratio) akv.comp = paged(cache.comp, b, ch.slot, kPageTokens / w.ratio, HD);
    if (w.has_indexer) {
      const int nkeys = (ch.s0 + ch.n) / 4;
      const int kk = std::min(c.idx_topk, nkeys);
      int32_t* idx = B<int32_t>(scr, kIdx);
      if (kk > 0) {
        // The chunk's queries in sub-batches that fit the score buffer.
        const bool fast = fast_indexer();
        const size_t row_bytes = size_t(nkeys) * (fast ? 2 : 4);
        const int sub = int(std::max<size_t>(1, std::min<size_t>(size_t(ch.n), scr.bufs[kISc].size() / row_bytes)));
        const k::PagedFp4 keys = paged_fp4(cache.idx, b, ch.slot);
        for (int t = 0; t < ch.n; t += sub) {
          const int nt = std::min(sub, ch.n - t), tt = ch.t0 + t;
          if (fast) {
            uint16_t* sc = B<uint16_t>(scr, kISc);
            k::indexer_scores_fp4(B<int8_t>(scr, kQC) + size_t(tt) * IH * ID,
                                  B<uint8_t>(scr, kQCS) + size_t(tt) * IH * (ID / 32), wi + size_t(tt) * IH, keys,
                                  pos + tt, sc, nt, IH, nkeys, 4, st);
            k::topk_bf16(sc, nt, nkeys, kk, idx + size_t(t) * kk, st);
          } else {
            float* sc = B<float>(scr, kISc);
            k::indexer_scores(qi + size_t(tt) * IH * ID, wi + size_t(tt) * IH, keys, pos + tt, sc, nt, IH, ID, nkeys,
                              4, st);
            k::topk_rows(sc, nt, nkeys, kk, idx + size_t(t) * kk, st);
          }
        }
        if (b.chunks.size() == 1) dump(L, b, "indexer_out", idx, size_t(ch.n) * kk * 4, st);
      }
      akv.comp_idx = idx;
      akv.ncomp = kk;
    }
    attention(q + size_t(ch.t0) * H * HD, cpos, akv, w.sink.p<float>(), o + size_t(ch.t0) * H * HD, ch.n, H, HD,
                   scale, st);
    k::window_update(kv + size_t(ch.t0) * HD, window, ch.s0, ch.n, HD, st);
  }
  dump(L, b, "sa_o", o, size_t(T) * H * HD * 2, st);
  k::rope(o, H * HD, T, H, HD, HD - rd, rd / 2, pos, freqs, true, st);

  // o: grouped wo_a (bf16 einsum on dequantized fp8 weights), then wo_b.
  bf16* oa = B<bf16>(scr, kOA);
  bf16* ao = B<bf16>(scr, kAO);
  const int G = c.o_groups, R = c.o_lora, GK = H * HD / G;
  dense_grouped_fp8w(o, w.wo_a.w.p<uint8_t>(), w.wo_a.s.p<uint8_t>(), oa, T, G, R, GK, st);
  dump(L, b, "wo_b_in", oa, size_t(T) * G * R * 2, st);
  k::act_quant_fp8(oa, G * R, qb, qs, T, G * R, 128, st);
  dense_w8a8(qb, qs, w.wo_b.w.p<uint8_t>(), w.wo_b.s.p<uint8_t>(), ao, D, T, D, G * R, st);
  dump(L, b, "attn_out", ao, size_t(T) * D * 2, st);
  k::hc_post(ao, x, post, comb, x, T, c.hc_mult, D, st);
}

// Routing -> pair rows -> experts entirely on the device (EXL3 fast path). Pair rows are laid out
// resident experts first, then remote ones; the remote block goes to the remote side through pinned
// memory with stream-ordered flags, so the host only enqueues.
void moe_device_dispatch(const Dsv4Config& c, const LayerWeights& w, const Scratch& scr, const bf16* f,
                         const int32_t* sel, const float* wt, int T, hipStream_t st, RemoteExperts* remote) {
  const int D = c.dim, E = c.n_experts, K = c.n_active, P = T * K;
  const auto& ds = scr.dispatch;
  int32_t* count = ds.count.as<int32_t>();
  int32_t* off = ds.off.as<int32_t>();
  int32_t* fill = ds.fill.as<int32_t>();
  auto* plan = ds.plan.as<k::MoePlan>();
  auto* lt = ds.ltiles.as<k::Exl3Tile>();
  auto* rtl = ds.rtiles.as<k::Exl3Tile>();
  int32_t* eidx = B<int32_t>(scr, kEIdx);
  float* ewt = B<float>(scr, kEWt);
  int32_t* pidx = B<int32_t>(scr, kPairIdx);
  int32_t* row_expert = scr.exl3.row_expert.as<int32_t>();
  bf16* pair_in = B<bf16>(scr, kPairIn);
  bf16* pair_out = B<bf16>(scr, kPairOut);
  k::moe_count(sel, T, K, count, E, st);
  k::moe_plan(count, w.resident.as<uint8_t>(), E, P, plan, lt, rtl, off, fill, ds.max_tiles, st);
  k::moe_scatter(sel, wt, T, K, off, fill, eidx, ewt, row_expert, pidx, st);
  k::gather_rows(f, eidx, pair_in, P, D, st);
  const bool any_remote = w.n_resident < E;
  if (any_remote) {
    if (!remote || !remote->can_enqueue()) throw std::runtime_error("moe: remote experts need an enqueue backend");
    if (size_t(P) > remote->capacity_rows || ds.max_tiles > remote->max_tiles)
      throw std::runtime_error("moe: remote staging too small");
    UH_CHECK(hipMemcpyAsync(remote->plan_host, plan, sizeof(k::MoePlan), hipMemcpyDeviceToHost, st));
    UH_CHECK(hipMemcpyAsync(remote->tiles_host, rtl, size_t(ds.max_tiles) * sizeof(k::Exl3Tile), hipMemcpyDeviceToHost,
                            st));
    UH_CHECK(hipMemcpyAsync(remote->e_host, row_expert, size_t(P) * 4, hipMemcpyDeviceToHost, st));
    UH_CHECK(hipMemcpyAsync(remote->w_host, ewt, size_t(P) * 4, hipMemcpyDeviceToHost, st));
    k::copy_remote_rows(pair_in, remote->in_host, plan, D, P, st);
    remote->enqueue(w.layer, st, P);
  }
  exl3_moe_forward_dev(c, w, pair_in, ewt, row_expert, lt, ds.max_tiles, &plan->n_local_tiles, P, scr.exl3, pair_out,
                       st);
  if (any_remote) {
    remote->join(st);
    k::copy_remote_rows(remote->out_host, pair_out, plan, D, P, st);
  }
}

void moe_half(const Dsv4Config& c, const LayerWeights& w, const Scratch& scr, bf16* x, const BatchDesc& b,
              hipStream_t st, RemoteExperts* remote) {
  const int D = c.dim, I = c.moe_inter, E = c.n_experts, K = c.n_active, T = b.T, L = w.layer;
  bf16* y = B<bf16>(scr, kY);
  bf16* f = B<bf16>(scr, kA);
  float* post = B<float>(scr, kPost);
  float* comb = B<float>(scr, kComb);
  uint8_t* qb = B<uint8_t>(scr, kQBuf);
  uint8_t* qs = B<uint8_t>(scr, kQSc);
  hc_pre(x, w.hc_ffn_fn.p<float>(), w.hc_ffn_scale.p<float>(), w.hc_ffn_base.p<float>(), y, post, comb, B<float>(scr, kHcPart), T,
            c.hc_mult, D, c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  dump(L, b, "hcpre_ffn_y", y, size_t(T) * D * 2, st);
  dump(L, b, "hcpre_ffn_post", post, size_t(T) * c.hc_mult * 4, st);
  dump(L, b, "hcpre_ffn_comb", comb, size_t(T) * c.hc_mult * c.hc_mult * 4, st);
  k::rmsnorm(y, D, w.ffn_norm.p<bf16>(), f, D, T, D, c.norm_eps, st);
  dump(L, b, "ffn_in", f, size_t(T) * D * 2, st);

  // Router (fp32 linear on x.float()).
  float* af = B<float>(scr, kAF);
  float* lg = B<float>(scr, kLG);
  int32_t* sel = B<int32_t>(scr, kSel);
  float* wt = B<float>(scr, kWt);
  k::cast_bf16_f32(f, af, size_t(T) * D, st);
  dense_f32(f, af, D, w.gate_w.p<bf16>(), lg, E, T, E, D, st);
  (fast_dense() ? k::route_fast : k::route)(lg, w.gate_bias.p<float>(), w.hash ? w.tid2eid.p<int32_t>() : nullptr, b.ids,
                                           sel, wt, T, E, K, c.route_scale, st);
  dump(L, b, "gate_out0", wt, size_t(T) * K * 4, st);
  dump(L, b, "gate_out1", sel, size_t(T) * K * 4, st);

  const bool device_path = w.exl3_experts && exl3_fast_enabled() && !g_expert_hist &&
                           (w.n_resident == E || (remote && remote->can_enqueue()));
  int32_t* dpidx = B<int32_t>(scr, kPairIdx);
  bf16* pair_out = B<bf16>(scr, kPairOut);
  if (device_path) {
    moe_device_dispatch(c, w, scr, f, sel, wt, T, st, remote);
  } else {
    // Expert-major pairs (ascending expert id): pair p computes expert e on token t into row p of the
    // pair-output buffer; pidx[t][j] lists token t's pairs in ascending expert order for the combine.
    std::vector<int32_t> hsel(size_t(T) * K);
    std::vector<float> hwt(size_t(T) * K);
    UH_CHECK(hipMemcpyAsync(hsel.data(), sel, hsel.size() * 4, hipMemcpyDeviceToHost, st));
    UH_CHECK(hipMemcpyAsync(hwt.data(), wt, hwt.size() * 4, hipMemcpyDeviceToHost, st));
    UH_CHECK(hipStreamSynchronize(st));
    std::vector<int> count(size_t(E), 0), off(size_t(E) + 1, 0);
    for (int32_t e : hsel) ++count[size_t(e)];
    if (g_expert_hist) {
      if (g_expert_hist->size() <= size_t(L)) g_expert_hist->resize(size_t(L) + 1);
      auto& hl = (*g_expert_hist)[size_t(L)];
      hl.resize(size_t(E), 0);
      for (int e = 0; e < E; ++e) hl[size_t(e)] += uint64_t(count[size_t(e)]);
    }
    // Pair rows: this device's experts first, then the remote ones (each group in ascending expert
    // order), so the remote rows are one contiguous block [remote0, P) to ship each way. The combine
    // follows pidx (each token's pairs by expert id), so the placement does not change the sum order.
    int remote0 = 0;
    {
      int next = 0;
      for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) remote0 = next;
        for (int e = 0; e < E; ++e)
          if (w.has_expert(e) == (pass == 0)) {
            off[size_t(e)] = next;
            next += count[size_t(e)];
          }
      }
    }
    std::vector<int32_t> eidx(size_t(T) * K), pidx(size_t(T) * K);
    std::vector<float> ewt(size_t(T) * K);
    std::vector<int> fill(off.begin(), off.begin() + E);
    for (int t = 0; t < T; ++t) {
      // Token t's picks sorted by expert id (selection order is by score).
      std::vector<std::pair<int32_t, float>> picks;
      for (int j = 0; j < K; ++j) picks.push_back({hsel[size_t(t) * K + j], hwt[size_t(t) * K + j]});
      std::sort(picks.begin(), picks.end(), [](auto& a, auto& b2) { return a.first < b2.first; });
      for (int j = 0; j < K; ++j) {
        const int p = fill[size_t(picks[size_t(j)].first)]++;
        eidx[size_t(p)] = t;
        ewt[size_t(p)] = picks[size_t(j)].second;
        pidx[size_t(t) * K + j] = p;
      }
    }
    int32_t* deidx = B<int32_t>(scr, kEIdx);
    float* dewt = B<float>(scr, kEWt);
    UH_CHECK(hipMemcpyAsync(deidx, eidx.data(), eidx.size() * 4, hipMemcpyHostToDevice, st));
    UH_CHECK(hipMemcpyAsync(dewt, ewt.data(), ewt.size() * 4, hipMemcpyHostToDevice, st));
    UH_CHECK(hipMemcpyAsync(dpidx, pidx.data(), pidx.size() * 4, hipMemcpyHostToDevice, st));

    bf16* pair_in = B<bf16>(scr, kPairIn);
    const int P = T * K;
    k::gather_rows(f, deidx, pair_in, P, D, st);
    const bool fast = w.exl3_experts && exl3_fast_enabled();
    std::vector<int32_t> row_expert;
    if (fast) {
      row_expert.resize(size_t(P));
      for (int e = 0; e < E; ++e)
        for (int p = off[size_t(e)]; p < off[size_t(e)] + count[size_t(e)]; ++p) row_expert[size_t(p)] = e;
      UH_CHECK(hipMemcpyAsync(scr.exl3.row_expert.data(), row_expert.data(), size_t(P) * 4, hipMemcpyHostToDevice, st));
    }
    // Experts this device does not hold: ship every pair's input once (the remote side only reads the
    // rows of its jobs); local experts are queued meanwhile and overlap the remote computation.
    std::vector<RemoteExperts::Job> jobs;
    for (int e = 0; e < E; ++e)
      if (count[size_t(e)] && !w.has_expert(e)) jobs.push_back({e, off[size_t(e)], count[size_t(e)]});
    hipEvent_t shipped = nullptr;
    if (!jobs.empty()) {
      if (!remote) throw std::runtime_error("moe: experts missing on this device and no remote backend");
      if (size_t(P) > remote->capacity_rows) throw std::runtime_error("moe: remote staging too small");
      UH_CHECK(hipMemcpyAsync(remote->in_host + size_t(remote0) * D, pair_in + size_t(remote0) * D,
                              size_t(P - remote0) * D * 2, hipMemcpyDeviceToHost, st));
      std::copy(ewt.begin(), ewt.end(), remote->w_host);
      if (fast) std::copy(row_expert.begin(), row_expert.end(), remote->e_host);
      UH_CHECK(hipEventCreateWithFlags(&shipped, hipEventDisableTiming));
      UH_CHECK(hipEventRecord(shipped, st));
    }
    if (fast) {
      std::vector<std::array<int, 3>> runs;
      for (int e = 0; e < E; ++e)
        if (count[size_t(e)] && w.has_expert(e)) runs.push_back({e, off[size_t(e)], count[size_t(e)]});
      exl3_moe_forward(c, w, pair_in, dewt, scr.exl3.row_expert.as<int32_t>(), exl3_tiles(runs), P, scr.exl3, pair_out,
                       st);
    } else {
      for (int e = 0; e < E; ++e) {
        const int cnt = count[size_t(e)];
        if (!cnt || !w.has_expert(e)) continue;
        const size_t o = size_t(off[size_t(e)]);
        expert_forward(c, w, e, pair_in + o * D, dewt + o, cnt, pair_out + o * D, scr.expert, st);
      }
    }
    if (!jobs.empty()) {
      UH_CHECK(hipEventSynchronize(shipped));
      UH_CHECK(hipEventDestroy(shipped));
      remote->run(L, jobs);
      UH_CHECK(hipMemcpyAsync(pair_out + size_t(remote0) * D, remote->out_host + size_t(remote0) * D,
                              size_t(P - remote0) * D * 2, hipMemcpyHostToDevice, st));
    }
  }
  bf16* h1 = B<bf16>(scr, kH1);
  bf16* h3 = B<bf16>(scr, kH3);
  bf16* hh = B<bf16>(scr, kHH);
  // Shared expert (fp8 weights, no routing weight).
  bf16* shared = B<bf16>(scr, kShared);
  k::act_quant_fp8(f, D, qb, qs, T, D, 128, st);
  dense_w8a8(qb, qs, w.sw1.w.p<uint8_t>(), w.sw1.s.p<uint8_t>(), h1, I, T, I, D, st);
  dense_w8a8(qb, qs, w.sw3.w.p<uint8_t>(), w.sw3.s.p<uint8_t>(), h3, I, T, I, D, st);
  k::swiglu(h1, h3, nullptr, hh, T, I, c.swiglu_limit, st);
  k::act_quant_fp8(hh, I, qb, qs, T, I, 128, st);
  dense_w8a8(qb, qs, w.sw2.w.p<uint8_t>(), w.sw2.s.p<uint8_t>(), shared, D, T, D, I, st);
  dump(L, b, "shared_out", shared, size_t(T) * D * 2, st);
  bf16* fo = B<bf16>(scr, kAO);
  k::moe_combine(pair_out, dpidx, shared, fo, T, K, D, st);
  dump(L, b, "ffn_out", fo, size_t(T) * D * 2, st);
  k::hc_post(fo, x, post, comb, x, T, c.hc_mult, D, st);
  if (!device_path) UH_CHECK(hipStreamSynchronize(st));  // the host path's vectors
}

}  // namespace

void layer_forward(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                   const RopeTables& rope, uint16_t* x, const BatchDesc& b, hipStream_t st, RemoteExperts* remote) {
  if (b.T > scr.max_tokens) throw std::runtime_error("layer_forward: batch larger than scratch");
  for (const auto& ch : b.chunks) {
    if ((ch.s0 + ch.n) / 4 > scr.max_keys) throw std::runtime_error("layer_forward: context larger than scratch");
    if (ch.slot < 0 || ch.slot >= cache.slots) throw std::runtime_error("layer_forward: bad slot");
  }
  attention_half(c, w, cache, scr, rope, x, b, st);
  moe_half(c, w, scr, x, b, st, remote);
}

}  // namespace unholy::model
