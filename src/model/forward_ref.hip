#include "model/forward_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/attn_wmma.hpp"
#include "kernels/dspark.hpp"
#include "kernels/elem_fast.hpp"
#include "kernels/exl3_moe.hpp"
#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/hc_fast.hpp"
#include "kernels/indexer.hpp"
#include "kernels/mx_moe.hpp"
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
  kISc, kIdx, kO, kOA, kAO, kLG, kSel, kWt, kH1, kH3, kHH, kPairOut, kPairIdx, kShared, kEIdx, kEWt,
  kPairIn, kHcPart, kQC, kQCS, kRTok, kNumBufs
};

template <typename T>
T* B(const Scratch& s, Buf b) {
  return static_cast<T*>(s.get(b));
}

}  // namespace

bool exl3_fast_enabled();
bool mx_fast_enabled();
namespace {
std::vector<std::vector<uint64_t>>* g_expert_hist = nullptr;
}
// The reference expert path (expert_forward and its scratch) runs only with UNHOLY_REF_EXL3 / UNHOLY_REF_MX
// or a host-side expert histogram (set_expert_histogram before the scratch is made).
bool ref_experts_needed() { return !exl3_fast_enabled() || !mx_fast_enabled() || g_expert_hist != nullptr; }

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
  sz[kQCS] = N * c.idx_heads;
  sz[kIdx] = N * c.idx_topk * 4;
  sz[kO] = N * c.n_heads * c.head_dim * 2;
  sz[kOA] = N * c.o_groups * c.o_lora * 2;
  sz[kAO] = N * c.dim * 2;
  sz[kLG] = N * c.n_experts * 4;
  sz[kSel] = N * K * 4;
  sz[kWt] = N * K * 4;
  sz[kH1] = N * c.moe_inter * 2;
  sz[kH3] = N * c.moe_inter * 2;
  sz[kHH] = N * c.moe_inter * 2;
  sz[kPairOut] = N * K * c.dim * 2;
  sz[kPairIdx] = N * K * 4;
  sz[kShared] = N * c.dim * 2;
  sz[kEIdx] = N * K * 4;
  sz[kEWt] = N * K * 4;
  sz[kPairIn] = N * K * c.dim * 2;
  sz[kHcPart] = size_t(32) * 32 * 25 * 4;
  sz[kRTok] = N * 2 * 4;  // remote_tokens: tok2u [T], u2t [T]
  return sz;
}
}  // namespace

size_t expert_scratch_bytes(const Dsv4Config& c, int rows) {
  const size_t R = size_t(std::max(rows, 1)), W = size_t(std::max(c.dim, c.moe_inter));
  return R * W + R * W / 32 + 3 * R * c.moe_inter * 2 + 2 * R * W * 4;
}

// Tiles of up to `rows` pair rows: <= 128-row tiles (the GEMMs) and, for steps of up to
// mx_gemv_max_tokens() tokens, kMxGemvTileRows-row tiles (the GEMVs).
int moe_max_tiles(const Dsv4Config& c, int rows) {
  return rows / 64 + c.n_experts + 1 + std::min(rows, k::mx_gemv_max_tokens() * c.n_active) / k::kMxGemvTileRows;
}

size_t exl3_moe_scratch_bytes(const Dsv4Config& c, int rows, bool exl3) {
  const size_t R = size_t(std::max(rows, 1));
  const size_t only_exl3 = exl3 ? R * c.dim * 2 + R * size_t(std::max(c.dim, c.moe_inter)) * 4 : 0;  // xh3, yp
  return R * c.dim * 2 + only_exl3 + R * c.moe_inter * 6 + R * 4 + size_t(moe_max_tiles(c, rows)) * 12;
}

size_t scratch_bytes(const Dsv4Config& c, int max_tokens, int max_seq, bool exl3) {
  size_t total = (ref_experts_needed() ? expert_scratch_bytes(c, max_tokens) : 0) +
                 exl3_moe_scratch_bytes(c, max_tokens * c.n_active, exl3);
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

Scratch make_scratch(const Dsv4Config& c, int dev, int max_tokens, int max_seq, bool exl3) {
  const std::vector<size_t> sz = scratch_sizes(c, max_tokens, max_seq);
  Scratch s;
  s.max_tokens = max_tokens;
  s.max_keys = max_seq / 4 + 1;
  for (size_t i = 0; i < kNumBufs; ++i) s.bufs.emplace_back(dev, std::max<size_t>(sz[i], 256));
  if (ref_experts_needed()) s.expert = make_expert_scratch(c, dev, max_tokens);
  s.exl3 = make_exl3_moe_scratch(c, dev, max_tokens * c.n_active, exl3);
  s.dispatch.max_tiles = s.exl3.max_tiles;
  // Every layer the MoE runs for: the target's and the DSpark stages (n_layers + n_mtp).
  s.route_hist = rt::DeviceBuffer(dev, std::max(size_t(c.n_layers + c.n_mtp), c.compress_ratios.size()) * c.n_experts * 4);
  UH_CHECK(hipMemset(s.route_hist.data(), 0, s.route_hist.size()));
  s.dispatch.count = rt::DeviceBuffer(dev, size_t(c.n_experts) * 4);
  UH_CHECK(hipMemset(s.dispatch.count.data(), 0, s.dispatch.count.size()));  // then kept zero by moe_plan
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
    auto lin = [&](const Fp4Linear& l, bf16* y, int out_dim, int in_dim) {
      (l.tiled ? k::gemm_a8w4_tiled : k::gemm_a8w4)(qb, qs, l.w.p<uint8_t>(), l.s.p<uint8_t>(), y, out_dim, cnt, out_dim,
                                                    in_dim, st);
    };
    k::act_quant_fp8(in, D, qb, qs, cnt, D, 128, st);
    lin(w.w1[size_t(e)], h1, I, D);
    lin(w.w3[size_t(e)], h3, I, D);
    k::swiglu(h1, h3, rw, hh, cnt, I, c.swiglu_limit, st);
    k::act_quant_fp8(hh, I, qb, qs, cnt, I, 128, st);
    lin(w.w2[size_t(e)], out, D, I);
  }
}

Exl3MoeScratch make_exl3_moe_scratch(const Dsv4Config& c, int dev, int rows, bool exl3) {
  const size_t R = size_t(std::max(rows, 1));
  Exl3MoeScratch s;
  s.rows = rows;
  s.max_tiles = moe_max_tiles(c, rows);
  s.xh1 = rt::DeviceBuffer(dev, R * c.dim * 2);
  if (exl3) s.xh3 = rt::DeviceBuffer(dev, R * c.dim * 2);  // the EXL3 path's second transformed input
  s.h1 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.h3 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.xh2 = rt::DeviceBuffer(dev, R * c.moe_inter * 2);
  s.tiles = rt::DeviceBuffer(dev, size_t(s.max_tiles) * sizeof(kernels::Exl3Tile));
  s.row_expert = rt::DeviceBuffer(dev, R * 4);
  if (exl3) s.yp = rt::DeviceBuffer(dev, R * size_t(std::max(c.dim, c.moe_inter)) * 4);  // the EXL3 GEMV's partials
  return s;
}

bool exl3_fast_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_REF_EXL3");
    return !v || std::string(v) == "0";
  }();
  return on;
}

// Native MXFP4 experts on the grouped kernels (kernels/mx_moe.hpp) unless UNHOLY_REF_MX=1.
bool mx_fast_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_REF_MX");
    return !v || std::string(v) == "0";
  }();
  return on;
}
bool routed_fast(const LayerWeights& w) { return w.exl3_experts ? exl3_fast_enabled() : mx_fast_enabled(); }

std::vector<kernels::Exl3Tile> exl3_tiles(const std::vector<std::array<int, 3>>& runs) {
  std::vector<kernels::Exl3Tile> t;
  for (const auto& r : runs)
    for (int i = 0; i < r[2]; i += k::kExl3TileRows) t.push_back({r[0], r[1] + i, std::min(k::kExl3TileRows, r[2] - i)});
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

void mx_moe_forward_dev_q(const Dsv4Config& c, const LayerWeights& w, const uint8_t* q, const uint8_t* s,
                          const int32_t* p2u, const kernels::MoePlan* plan, const float* rw, const int32_t* row_expert,
                          const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev, int P,
                          const Exl3MoeScratch& sc, uint16_t* out, hipStream_t st, const int32_t* u2p) {
  if (P > sc.rows) throw std::runtime_error("mx_moe_forward_dev_q: scratch too small");
  if (w.exl3_experts) throw std::runtime_error("mx_moe_forward_dev_q: native MXFP4 experts only");
  const int D = c.dim, I = c.moe_inter, E = c.n_experts;
  max_tiles = std::min(max_tiles, P);
  const auto tab = [&](int m, int f) { return reinterpret_cast<const uint8_t* const*>(w.exl3_ptrs(m, f, E)); };
  bf16* xa = sc.xh1.as<bf16>();
  bf16* hh = sc.xh2.as<bf16>();
  bf16* g = sc.h1.as<bf16>();
  bf16* u = sc.h3.as<bf16>();
  k::mx_xform_in_q(q, s, p2u, plan, row_expert, tab(0, 1), xa, P, D, st);
  const bool gemv = k::mx_use_gemv(P / c.n_active);
  const auto lin = gemv ? k::mx_grouped_gemv : k::mx_grouped_gemm;
  lin(xa, D, tiles, max_tiles, tab(0, 0), tab(0, 1), g, I, I, st, ntiles_dev, tab(1, 0), tab(1, 1), u);  // w1, w3
  k::mx_swiglu(g, u, rw, row_expert, tab(2, 1), hh, P, I, c.swiglu_limit, st);
  if (!u2p) {
    lin(hh, I, tiles, max_tiles, tab(2, 0), tab(2, 1), out, D, D, st, ntiles_dev, nullptr, nullptr, nullptr);
    return;
  }
  lin(hh, I, tiles, max_tiles, tab(2, 0), tab(2, 1), xa, D, D, st, ntiles_dev, nullptr, nullptr, nullptr);  // xa is free after the up GEMMs
  k::remote_combine(xa, u2p, plan, out, c.n_active, D, P / c.n_active, st);
}

void exl3_moe_forward_dev(const Dsv4Config& c, const LayerWeights& w, const uint16_t* in, const float* rw,
                          const int32_t* row_expert, const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev,
                          int P, const Exl3MoeScratch& s, uint16_t* out, hipStream_t st) {
  if (P > s.rows) throw std::runtime_error("exl3_moe_forward_dev: scratch too small");
  const int D = c.dim, I = c.moe_inter, E = c.n_experts;
  max_tiles = std::min(max_tiles, P);  // every tile has >= 1 row: bounds the grids at small batches
  if (!w.exl3_experts) {  // native MXFP4 experts: tables [w1, w3, w2] x [weights, scales]
    const auto tab = [&](int m, int f) { return reinterpret_cast<const uint8_t* const*>(w.exl3_ptrs(m, f, E)); };
    bf16* xa = s.xh1.as<bf16>();
    bf16* hh = s.xh2.as<bf16>();
    bf16* g = s.h1.as<bf16>();
    bf16* u = s.h3.as<bf16>();
    k::mx_xform_in(in, D, row_expert, tab(0, 1), xa, P, D, st);
    const bool gemv = k::mx_use_gemv(P / c.n_active);
    const auto lin = gemv ? k::mx_grouped_gemv : k::mx_grouped_gemm;
    lin(xa, D, tiles, max_tiles, tab(0, 0), tab(0, 1), g, I, I, st, ntiles_dev, tab(1, 0), tab(1, 1), u);  // w1, w3
    k::mx_swiglu(g, u, rw, row_expert, tab(2, 1), hh, P, I, c.swiglu_limit, st);
    lin(hh, I, tiles, max_tiles, tab(2, 0), tab(2, 1), out, D, D, st, ntiles_dev, nullptr, nullptr, nullptr);
    return;
  }
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
// Two W8A8 linears on the same quantized input: one GEMV launch for both when the GEMV applies (every
// column computed as by its own GEMV), else the two GEMMs.
bool gemv_parts() {
  static const bool on = !std::getenv("UNHOLY_NO_GEMV_PARTS");
  return on;
}
void dense_w8a8_2(const uint8_t* a, const uint8_t* sa, const Fp8Linear& l1, bf16* c1, int ldc1, const Fp8Linear& l2,
                  bf16* c2, int ldc2, int M, int K, hipStream_t st) {
  if (gemv_parts() && fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) {
    const k::GemvPart parts[2] = {{l1.w.p<uint8_t>(), l1.s.p<uint8_t>(), c1, ldc1, l1.out},
                                  {l2.w.p<uint8_t>(), l2.s.p<uint8_t>(), c2, ldc2, l2.out}};
    k::gemv_w8a8_parts(a, sa, parts, 2, M, K, st);
    return;
  }
  dense_w8a8(a, sa, l1.w.p<uint8_t>(), l1.s.p<uint8_t>(), c1, ldc1, M, l1.out, K, st);
  dense_w8a8(a, sa, l2.w.p<uint8_t>(), l2.s.p<uint8_t>(), c2, ldc2, M, l2.out, K, st);
}
// The reference's fp32 linears on x.float() of a bf16 tensor: bf16 WMMA on the bf16 source is exact in
// the products; the reference kernel takes the fp32 copy.
void dense_f32(const bf16* a_bf16, const float* a_f32, int lda, const bf16* w, float* c, int ldc, int M, int N,
               int K, hipStream_t st) {
  if (fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) k::gemv_bf16_f32out(a_bf16, lda, w, c, ldc, M, N, K, st);
  else if (fast_dense()) k::gemm_bf16_f32out_wmma(a_bf16, lda, w, c, ldc, M, N, K, st);
  else k::gemm_f32(a_f32, lda, w, c, ldc, M, N, K, st);
}
// Two fp32-output linears (bf16 weights [N, K]) on the same input: one GEMV launch when it applies.
void dense_f32_2(const bf16* a_bf16, const float* a_f32, int lda, const bf16* w1, float* c1, int ldc1, int N1,
                 const bf16* w2, float* c2, int ldc2, int N2, int M, int K, hipStream_t st) {
  if (gemv_parts() && fast_dense() && M <= k::kGemvMaxRows && K % 512 == 0) {
    const k::GemvPart parts[2] = {{w1, nullptr, c1, ldc1, N1}, {w2, nullptr, c2, ldc2, N2}};
    k::gemv_bf16_f32out_parts(a_bf16, lda, parts, 2, M, K, st);
    return;
  }
  dense_f32(a_bf16, a_f32, lda, w1, c1, ldc1, M, N1, K, st);
  dense_f32(a_bf16, a_f32, lda, w2, c2, ldc2, M, N2, K, st);
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

// Elementwise kernels over (row, slice) workgroups unless UNHOLY_REF_ELEM=1 (bit-identical; a decode
// step's rows otherwise get one workgroup each). rmsnorm only out of place.
bool fast_elem() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_REF_ELEM");
    return !v || std::string(v) == "0";
  }();
  return on;
}
void norm(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps, hipStream_t st) {
  if (fast_elem() && x != y) k::rmsnorm_slices(x, ldx, w, y, ldy, rows, dim, eps, st);
  else k::rmsnorm(x, ldx, w, y, ldy, rows, dim, eps, st);
}
// norm() then act_quant_fp8 (block 128) of its output into q / qs: one launch when the fused kernel
// applies (UNHOLY_REF_ELEM unset, out of place, 256-value slices).
void norm_quant(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, uint8_t* q, uint8_t* qs, int rows, int dim,
                float eps, hipStream_t st) {
  if (fast_elem() && k::rmsnorm_quant_slices(x, ldx, w, y, ldy, q, qs, rows, dim, eps, st)) return;
  norm(x, ldx, w, y, ldy, rows, dim, eps, st);
  k::act_quant_fp8(y, ldy, q, qs, rows, dim, 128, st);
}
void hc_post(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc, int dim,
             hipStream_t st) {
  if (fast_elem()) k::hc_post_slices(x, res, post, comb, out, T, hc, dim, st);
  else k::hc_post(x, res, post, comb, out, T, hc, dim, st);
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

// All chunks of a layer in one attention launch (the WMMA kernel; same per-query computation) unless
// UNHOLY_REF_ATTN=1 or UNHOLY_ATTN_PER_CHUNK=1.
bool batched_attention() {
  static const bool on = [] {
    const char* r = std::getenv("UNHOLY_REF_ATTN");
    const char* v = std::getenv("UNHOLY_ATTN_PER_CHUNK");
    return (!r || std::string(r) == "0") && (!v || std::string(v) == "0");
  }();
  return on && on_gfx12();
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
  dense_f32_2(B<bf16>(scr, kA), B<float>(scr, kAF), c.dim, cw.wkv.p<bf16>(), pk, width, width, cw.wgate.p<bf16>(), ps,
              width, width, b.T, c.dim, st);
  for (const auto& ch : b.chunks) k::add_ape(ps + size_t(ch.t0) * width, cw.ape.p<float>(), ch.s0, ch.n, cw.ratio, width, st);
}

// draft != null: a DSpark stage (ratio 0) over draft blocks instead of sequence chunks.
void attention_half(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                    const RopeTables& rope, bf16* x, const BatchDesc& b, hipStream_t st,
                    const DraftDesc* draft = nullptr) {
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
  norm_quant(y, D, w.attn_norm.p<bf16>(), a, D, qb, qs, T, D, c.norm_eps, st);  // a and its fp8 codes
  dump(L, b, "attn_in", a, size_t(T) * D * 2, st);

  // q: wq_a -> q_norm (qr) -> wq_b -> per-head rms -> rope; kv: wkv from the same quantized input.
  bf16* qa = B<bf16>(scr, kQA);
  bf16* qr = B<bf16>(scr, kQR);
  bf16* q = B<bf16>(scr, kQ);
  bf16* kv = B<bf16>(scr, kKV);
  bf16* qi = B<bf16>(scr, kQI);
  bf16* wi = B<bf16>(scr, kWI);
  const int IH = c.idx_heads, ID = c.idx_dim;
  dense_w8a8_2(qb, qs, w.wq_a, qa, c.q_lora, w.wkv, kv, HD, T, D, st);
  dump(L, b, "wq_a_out", qa, size_t(T) * c.q_lora * 2, st);
  dump(L, b, "wkv_out", kv, size_t(T) * HD * 2, st);
  norm_quant(qa, c.q_lora, w.q_norm.p<bf16>(), qr, c.q_lora, qb, qs, T, c.q_lora, c.norm_eps, st);
  // wq_b and the indexer's wq_b take the same quantized qr.
  if (w.has_indexer) dense_w8a8_2(qb, qs, w.wq_b, q, H * HD, w.idx_wq_b, qi, IH * ID, T, c.q_lora, st);
  else dense_w8a8(qb, qs, w.wq_b.w.p<uint8_t>(), w.wq_b.s.p<uint8_t>(), q, H * HD, T, H * HD, c.q_lora, st);
  dump(L, b, "wq_b_out", q, size_t(T) * H * HD * 2, st);
  if (fast_elem()) {  // the same three-kernel chains, fused per (row, head) / row
    k::head_rms_rope(q, T, H, HD, c.norm_eps, pos, freqs, rd / 2, st);
    k::kv_chain(kv, w.kv_norm.p<bf16>(), T, HD, c.norm_eps, pos, freqs, rd / 2, st);
  } else {
    k::head_rms_bf16(q, T, H, HD, c.norm_eps, st);
    k::rope(q, H * HD, T, H, HD, HD - rd, rd / 2, pos, freqs, false, st);
    k::rmsnorm(kv, HD, w.kv_norm.p<bf16>(), kv, HD, T, HD, c.norm_eps, st);
    k::rope(kv, HD, T, 1, HD, HD - rd, rd / 2, pos, freqs, false, st);
    k::fake_quant_fp8(kv, HD, T, HD - rd, 64, st);
  }

  // Indexer inputs (token-parallel; qi from the wq_b launch above); per-chunk scores/top-k below.
  if (w.ratio && !fast_dense()) k::cast_bf16_f32(a, B<float>(scr, kAF), size_t(T) * D, st);  // reference GEMMs' input
  if (w.has_indexer) {
    if (fast_indexer() && fast_elem() && ID == 128) {  // rope, Hadamard and the fp4 codes in one launch
      k::index_q_chain(qi, T, IH, pos, freqs, rd / 2, B<uint8_t>(scr, kQC), B<uint8_t>(scr, kQCS), st);
    } else {
      k::rope(qi, IH * ID, T, IH, ID, ID - rd, rd / 2, pos, freqs, false, st);
      k::hadamard128(qi, size_t(T) * IH * ID / 128, st);
      if (fast_indexer()) k::quant_q_fp4(qi, T, IH, B<uint8_t>(scr, kQC), B<uint8_t>(scr, kQCS), st);
      else k::fake_quant_fp4(qi, IH * ID, T, IH * ID, 32, st);
    }
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
  if (draft) {  // DSparkAttention: the request's window (slot order), then its whole block
    const int B = c.dspark_block;
    for (size_t r = 0; r < draft->slots.size(); ++r) {
      k::AttnKv akv;
      akv.chunk_kv = kv + r * B * HD;
      akv.ring = cache.window_rows.as<bf16>() + size_t(draft->slots[r]) * c.window * HD;
      akv.dspark_slots = std::min(c.window, draft->start[r] + 1);
      akv.dspark_chunk = B;
      attention(q + r * B * H * HD, pos + r * B, akv, w.sink.p<float>(), o + r * B * H * HD, B, H, HD, scale, st);
    }
  }
  // Indexer top-k of every chunk (chunk c's rows at idx + t0 * idx_topk, stride kk), then the attention of
  // all chunks, then their window rings. Chunks are of different slots, so no chunk reads another's ring
  // update; a step with two chunks of one slot keeps the chunk-by-chunk order.
  int32_t* idx = B<int32_t>(scr, kIdx);
  const auto topk_of = [&](const ChunkDesc& ch) { return std::min(c.idx_topk, (ch.s0 + ch.n) / 4); };
  bool distinct = true;
  for (size_t i = 0; i < b.chunks.size() && distinct; ++i)
    for (size_t j = 0; j < i; ++j) distinct &= b.chunks[i].slot != b.chunks[j].slot;
  const auto indexer = [&](const ChunkDesc* only) {
    // Score rows of several chunks (or sub-batches of one) share the score buffer; their top-k runs in
    // one launch when it is full and at the end.
    const bool fast = fast_indexer();
    const size_t cap = scr.bufs[kISc].size();
    k::TopkBatch tb;
    size_t used = 0;
    const auto flush = [&]() {
      // The score buffer past the pending rows is the top-k's scratch (several workgroups per row for a
      // few long rows: decode at long context).
      if (tb.ngroups)
        k::topk_bf16_batch(B<uint16_t>(scr, kISc), tb, idx, st, reinterpret_cast<uint32_t*>(B<uint8_t>(scr, kISc) + used),
                           (cap - used) / 4);
      tb.ngroups = 0;
      used = 0;
    };
    for (const auto& ch : b.chunks) {
      if (only && &ch != only) continue;
      const int nkeys = (ch.s0 + ch.n) / 4, kk = topk_of(ch);
      if (kk <= 0) continue;
      int32_t* cidx = idx + size_t(ch.t0) * c.idx_topk;
      const size_t row_bytes = size_t(nkeys) * (fast ? 2 : 4);
      const int sub = int(std::max<size_t>(1, std::min<size_t>(size_t(ch.n), cap / row_bytes)));
      const k::PagedFp4 keys = paged_fp4(cache.idx, b, ch.slot);
      for (int t = 0; t < ch.n; t += sub) {
        const int nt = std::min(sub, ch.n - t), tt = ch.t0 + t;
        if (fast) {
          const size_t bytes = (size_t(nt) * row_bytes + 255) / 256 * 256;
          if (used + bytes > cap || tb.ngroups == k::TopkBatch::kMax) flush();
          k::indexer_scores_fp4(B<uint8_t>(scr, kQC) + size_t(tt) * IH * ID, B<uint8_t>(scr, kQCS) + size_t(tt) * IH,
                                wi + size_t(tt) * IH, keys, pos + tt, B<uint16_t>(scr, kISc) + used / 2, nt, IH, nkeys,
                                4, st);
          tb.g[tb.ngroups++] = {used / 2, nt, nkeys, kk, int(size_t(ch.t0) * c.idx_topk + size_t(t) * kk)};
          used += bytes;
        } else {
          float* sc = B<float>(scr, kISc);
          k::indexer_scores(qi + size_t(tt) * IH * ID, wi + size_t(tt) * IH, keys, pos + tt, sc, nt, IH, ID, nkeys,
                            4, st);
          k::topk_rows(sc, nt, nkeys, kk, cidx + size_t(t) * kk, st);
        }
      }
    }
    flush();
    if (b.chunks.size() == 1 && topk_of(b.chunks[0]) > 0)
      dump(L, b, "indexer_out", idx, size_t(b.chunks[0].n) * topk_of(b.chunks[0]) * 4, st);
  };
  const auto chunk_kv = [&](const ChunkDesc& ch) {
    k::AttnKv akv;
    akv.chunk_kv = kv + size_t(ch.t0) * HD;
    akv.ring = cache.window_rows.as<bf16>() + size_t(ch.slot) * c.window * HD;
    akv.s0 = ch.s0;
    akv.ratio = w.ratio;
    if (b.vis) akv.vis = b.vis + size_t(ch.t0) * 2;
    if (w.ratio) akv.comp = paged(cache.comp, b, ch.slot, kPageTokens / w.ratio, HD);
    if (w.has_indexer) {
      akv.comp_idx = idx + size_t(ch.t0) * c.idx_topk;
      akv.ncomp = topk_of(ch);
    }
    return akv;
  };
  if (distinct && batched_attention()) {
    if (w.has_indexer) indexer(nullptr);
    k::AttnBatch ab;
    ab.kv = kv;
    ab.ring = cache.window_rows.as<bf16>();
    ab.ring_stride = size_t(c.window) * HD;
    if (w.ratio) {
      ab.comp_pool = cache.comp.as<bf16>();
      ab.rows_per_page = kPageTokens / w.ratio;
      ab.row_stride = HD;
    }
    ab.tables = b.block_tables;
    ab.max_pages = b.max_pages_per_seq;
    ab.ratio = w.ratio;
    ab.comp_idx = w.has_indexer ? idx : nullptr;
    ab.vis = b.vis;
    for (size_t i = 0; i < b.chunks.size(); ++i) {
      const auto& ch = b.chunks[i];
      ab.c[ab.nchunks++] = {ch.t0, ch.n, ch.s0, ch.slot, w.has_indexer ? topk_of(ch) : 0, int(size_t(ch.t0) * c.idx_topk)};
      if (ab.nchunks == k::AttnBatch::kMax || i + 1 == b.chunks.size()) {
        k::sparse_attn_wmma_batch(q, pos, ab, w.sink.p<float>(), o, scale, st);
        ab.nchunks = 0;
      }
    }
    for (const auto& ch : b.chunks)
      k::window_update(kv + size_t(ch.t0) * HD, cache.window_rows.as<bf16>() + size_t(ch.slot) * c.window * HD, ch.s0,
                       ch.n, HD, st);
  } else {
    for (const auto& ch : b.chunks) {
      if (w.has_indexer) indexer(&ch);
      attention(q + size_t(ch.t0) * H * HD, pos + ch.t0, chunk_kv(ch), w.sink.p<float>(), o + size_t(ch.t0) * H * HD,
                ch.n, H, HD, scale, st);
      k::window_update(kv + size_t(ch.t0) * HD, cache.window_rows.as<bf16>() + size_t(ch.slot) * c.window * HD, ch.s0,
                       ch.n, HD, st);
    }
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
  hc_post(ao, x, post, comb, x, T, c.hc_mult, D, st);
}

// Routing -> pair rows -> experts entirely on the device (EXL3 fast path). Pair rows are laid out
// resident experts first, then remote ones; the remote block goes to the remote side through pinned
// memory with stream-ordered flags, so the host only enqueues.
// The remote experts are only joined by moe_device_join, so the card can queue independent work (the
// shared expert) while they run.
struct MoeDispatch {
  bool remote = false;    // remote experts were enqueued
  bool combined = false;  // their outputs come back combined per token (moe_combine_partial)
};
MoeDispatch moe_device_dispatch(const Dsv4Config& c, const LayerWeights& w, const Scratch& scr, const bf16* f,
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
  uint32_t* hist = scr.route_hist.as<uint32_t>() + size_t(w.layer) * E;
  const int tile_rows = !w.exl3_experts && k::mx_use_gemv(T) ? k::kMxGemvTileRows : k::kExl3TileRows;
  const bool any_remote = w.n_resident < E;
  bool qin = false, combined = false;
  if (any_remote) {
    if (!remote || !remote->can_enqueue()) throw std::runtime_error("moe: remote experts need an enqueue backend");
    if (size_t(P) > remote->capacity_rows || ds.max_tiles > remote->max_tiles)
      throw std::runtime_error("moe: remote staging too small");
    // Native experts: one fp8 row per token with remote pairs instead of a bf16 row per remote pair (the
    // remote side rebuilds the same expert inputs; ~5x fewer bytes over the host link, which both cards
    // share behind the PCIe switch).
    qin = !w.exl3_experts && remote->quant_in();
    combined = qin && remote->combine_out();
  }
  // The remote side's staging is written by kernels, not async copies: the copies were not reliably
  // ordered after the previous layer's stream wait on the remote side's done flag (DECISIONS 2026-10-12),
  // so they could overwrite the staging the remote side was still reading.
  int32_t* tok2u = B<int32_t>(scr, kRTok);
  if (T <= k::kMoeSmallMaxTokens && K <= 8 && E <= 256) {  // decode-sized: one block for the bookkeeping
    k::MoeSmallArgs ma{sel, wt, T, K, hist, w.resident.as<uint8_t>(), E, P, ds.max_tiles, tile_rows, plan, lt, rtl, eidx,
                       ewt, row_expert, pidx, any_remote, qin, tok2u, tok2u + T,
                       any_remote ? remote->p2u_host : nullptr, combined ? remote->u2p_host : nullptr,
                       any_remote ? remote->plan_host : nullptr, any_remote ? remote->tiles_host : nullptr,
                       any_remote ? remote->w_host : nullptr, any_remote ? remote->e_host : nullptr};
    k::moe_dispatch_small(ma, st);
    k::gather_rows(f, eidx, pair_in, P, D, st);
    if (qin) k::quant_remote_tokens(f, D, tok2u + T, plan, remote->inq_host, remote->ins_host, T, st);
  } else {
    k::moe_count(sel, T, K, count, E, st, hist);
    k::moe_plan(count, w.resident.as<uint8_t>(), E, P, plan, lt, rtl, off, fill, ds.max_tiles, st, tile_rows);
    k::moe_scatter(sel, wt, T, K, off, fill, eidx, ewt, row_expert, pidx, st);
    k::gather_rows(f, eidx, pair_in, P, D, st);
    if (any_remote) {
      if (qin) {  // before stage_remote: it copies the plan with the token count
        k::remote_tokens(pidx, eidx, T, K, plan, tok2u, tok2u + T, remote->p2u_host,
                         combined ? remote->u2p_host : nullptr, st);
        k::quant_remote_tokens(f, D, tok2u + T, plan, remote->inq_host, remote->ins_host, T, st);
      }
      k::stage_remote(plan, rtl, ewt, row_expert, remote->plan_host, remote->tiles_host, remote->w_host,
                      remote->e_host, P, st);
    }
  }
  if (any_remote) {
    if (!qin) k::copy_remote_rows(pair_in, remote->in_host, plan, D, P, st);
    static const bool stats = std::getenv("UNHOLY_MOE_STATS") != nullptr;
    if (stats) {  // diagnostics only: synchronizes the stream
      k::MoePlan hp;
      UH_CHECK(hipMemcpyAsync(&hp, plan, sizeof hp, hipMemcpyDeviceToHost, st));
      UH_CHECK(hipStreamSynchronize(st));
      std::fprintf(stderr, "moe L%d: %d pairs, %d remote (%.1f%%)\n", w.layer, hp.P, hp.P - hp.remote0,
                   100.0 * (hp.P - hp.remote0) / std::max(1, hp.P));
    }
    remote->enqueue(w.layer, st, P);
  }
  exl3_moe_forward_dev(c, w, pair_in, ewt, row_expert, lt, ds.max_tiles, &plan->n_local_tiles, P, scr.exl3, pair_out,
                       st);
  return {any_remote, combined};
}

// The card stream waits for the remote experts and takes their outputs into the pair-output rows.
void moe_device_join(const Dsv4Config& c, const Scratch& scr, const MoeDispatch& md, int T, hipStream_t st,
                     RemoteExperts* remote) {
  if (!md.remote) return;
  const auto* plan = scr.dispatch.plan.as<k::MoePlan>();
  bf16* pair_out = B<bf16>(scr, kPairOut);
  remote->join(st);
  if (md.combined) k::copy_remote_partial(remote->out_host, pair_out, plan, c.dim, T, st);
  else k::copy_remote_rows(remote->out_host, pair_out, plan, c.dim, T * c.n_active, st);
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
  norm_quant(y, D, w.ffn_norm.p<bf16>(), f, D, qb, qs, T, D, c.norm_eps, st);  // f, and its fp8 codes for the shared expert
  dump(L, b, "ffn_in", f, size_t(T) * D * 2, st);

  // Router (fp32 linear on x.float()).
  float* af = B<float>(scr, kAF);
  float* lg = B<float>(scr, kLG);
  int32_t* sel = B<int32_t>(scr, kSel);
  float* wt = B<float>(scr, kWt);
  if (!fast_dense()) k::cast_bf16_f32(f, af, size_t(T) * D, st);  // the reference GEMM's input
  dense_f32(f, af, D, w.gate_w.p<bf16>(), lg, E, T, E, D, st);
  (fast_dense() ? k::route_fast : k::route)(lg, w.gate_bias.p<float>(), w.hash ? w.tid2eid.p<int32_t>() : nullptr, b.ids,
                                           sel, wt, T, E, K, c.route_scale, st,
                                           w.gate_bias_vl ? w.gate_bias_vl.p<float>() : nullptr, c.vocab);
  dump(L, b, "gate_out0", wt, size_t(T) * K * 4, st);
  dump(L, b, "gate_out1", sel, size_t(T) * K * 4, st);

  const bool device_path = routed_fast(w) && !g_expert_hist &&
                           (w.n_resident == E || (remote && remote->can_enqueue()));
  int32_t* dpidx = B<int32_t>(scr, kPairIdx);
  bf16* pair_out = B<bf16>(scr, kPairOut);
  MoeDispatch md;
  if (device_path) {
    md = moe_device_dispatch(c, w, scr, f, sel, wt, T, st, remote);
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
  // Shared expert (fp8 weights, no routing weight), queued while the remote experts run: its buffers are
  // not the routed path's.
  bf16* shared = B<bf16>(scr, kShared);
  dense_w8a8_2(qb, qs, w.sw1, h1, I, w.sw3, h3, I, T, D, st);  // qb / qs: f's codes (norm_quant)
  k::swiglu(h1, h3, nullptr, hh, T, I, c.swiglu_limit, st);
  k::act_quant_fp8(hh, I, qb, qs, T, I, 128, st);
  dense_w8a8(qb, qs, w.sw2.w.p<uint8_t>(), w.sw2.s.p<uint8_t>(), shared, D, T, D, I, st);
  dump(L, b, "shared_out", shared, size_t(T) * D * 2, st);
  moe_device_join(c, scr, md, T, st, remote);
  bf16* fo = B<bf16>(scr, kAO);
  if (md.combined)
    k::moe_combine_partial(pair_out, dpidx, scr.dispatch.plan.as<k::MoePlan>(), B<int32_t>(scr, kRTok), shared, fo, T, K,
                           D, st);
  else
    k::moe_combine(pair_out, dpidx, shared, fo, T, K, D, st);
  dump(L, b, "ffn_out", fo, size_t(T) * D * 2, st);
  hc_post(fo, x, post, comb, x, T, c.hc_mult, D, st);
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

// ---- DSpark ---------------------------------------------------------------------------------------

void dspark_main_kv(const Dsv4Config& c, const DSparkWeights& d, std::vector<LayerCache>& stages,
                    const Scratch& scr, const RopeTables& rope, const bf16* main_hidden, const int32_t* slots,
                    const int32_t* pos, int R, hipStream_t st) {
  if (R <= 0) return;
  const int D = c.dim, HD = c.head_dim, rd = c.rope_dim, MD = int(c.dspark_target_layers.size()) * D;
  uint8_t* qb = B<uint8_t>(scr, kQBuf);
  uint8_t* qs = B<uint8_t>(scr, kQSc);
  bf16* mx = B<bf16>(scr, kY);
  bf16* kv = B<bf16>(scr, kKV);
  if (size_t(R) * MD > scr.bufs[kQBuf].size()) throw std::runtime_error("dspark_main_kv: too many rows");
  // main_x = main_norm(main_proj(main_hidden)): an fp8 linear (act_quant per 128, fp8 GEMM).
  k::act_quant_fp8(main_hidden, MD, qb, qs, R, MD, 128, st);
  dense_w8a8(qb, qs, d.main_proj.w.p<uint8_t>(), d.main_proj.s.p<uint8_t>(), mx, D, R, D, MD, st);
  k::rmsnorm(mx, D, d.main_norm.p<bf16>(), mx, D, R, D, c.norm_eps, st);
  // Each stage's main_kv = kv_norm(wkv(main_x)), roped (plain table: ratio 0) and fp8-simulated.
  k::act_quant_fp8(mx, D, qb, qs, R, D, 128, st);
  for (size_t i = 0; i < stages.size(); ++i) {
    const LayerWeights& w = d.stages[i];
    dense_w8a8(qb, qs, w.wkv.w.p<uint8_t>(), w.wkv.s.p<uint8_t>(), kv, HD, R, HD, D, st);
    k::rmsnorm(kv, HD, w.kv_norm.p<bf16>(), kv, HD, R, HD, c.norm_eps, st);
    k::rope(kv, HD, R, 1, HD, HD - rd, rd / 2, pos, rope.plain.as<float2>(), false, st);
    k::fake_quant_fp8(kv, HD, R, HD - rd, 64, st);
    k::scatter_window(kv, R, HD, slots, pos, stages[i].window_rows.as<bf16>(), st);
  }
}

void dspark_stage(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                  const RopeTables& rope, bf16* x, const DraftDesc& d, hipStream_t st, RemoteExperts* remote) {
  BatchDesc b;
  b.T = d.T;
  b.ids = d.ids;
  b.pos = d.pos;
  attention_half(c, w, cache, scr, rope, x, b, st, &d);
  moe_half(c, w, scr, x, b, st, remote);
}

void dspark_head(const Dsv4Config& c, const DSparkWeights& d, const Scratch& scr, const bf16* x, const bf16* head_w,
                 int R, float* logits, int32_t* tok, float* conf, hipStream_t st) {
  const int D = c.dim, V = c.vocab, Bk = c.dspark_block, K = c.dspark_markov_rank, T = R * Bk;
  bf16* xh = B<bf16>(scr, kY);
  bf16* xn = B<bf16>(scr, kA);
  bf16* emb = B<bf16>(scr, kQA);
  void* part = B<float>(scr, kAF);
  if (size_t(T) * K * 2 > scr.bufs[kQA].size() || k::markov_scratch_bytes(V, R) > scr.bufs[kAF].size())
    throw std::runtime_error("dspark_head: scratch too small");
  k::hc_head(x, d.hc_head_fn.p<float>(), d.hc_head_scale.p<float>(), d.hc_head_base.p<float>(), xh, T, c.hc_mult, D,
             c.norm_eps, c.hc_eps, st);
  k::rmsnorm(xh, D, d.norm.p<bf16>(), xn, D, T, D, c.norm_eps, st);
  if (T <= k::kGemvMaxRows) k::gemv_bf16_f32out(xn, D, head_w, logits, V, T, V, D, st);
  else k::gemm_bf16_f32out_wmma(xn, D, head_w, logits, V, T, V, D, st);
  for (int i = 0; i < Bk; ++i)
    k::markov_step(d.markov_w1.p<bf16>(), d.markov_w2.p<bf16>(), V, K, logits, Bk, tok, emb, R, i, part, st);
  k::dspark_confidence(xh, D, emb, K, d.conf_proj.p<bf16>(), conf, T, st);
}

}  // namespace unholy::model
