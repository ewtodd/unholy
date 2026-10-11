#include "model/forward_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/exl3_moe.hpp"
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
    lc.idx = rt::DeviceBuffer(dev, size_t(g.num_pages) * (kPageTokens / 4) * c.idx_dim * 2);
  }
  UH_CHECK(hipStreamSynchronize(st));
  return lc;
}

namespace {

enum Buf : size_t {
  kY, kA, kPost, kComb, kQBuf, kQSc, kQA, kQR, kQ, kKV, kAF, kPK, kPS, kRowsF, kRowsB, kRowPos, kQI, kWI,
  kISc, kIdx, kO, kOA, kAO, kLG, kSel, kWt, kYF, kG, kH1, kH3, kHH, kPairOut, kPairIdx, kShared, kEIdx, kEWt,
  kY2F, kPairIn, kNumBufs
};

template <typename T>
T* B(const Scratch& s, Buf b) {
  return static_cast<T*>(s.get(b));
}

}  // namespace

namespace {
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
  sz[kISc] = N * keys * 4;
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
    b += size_t(g.num_pages) * (kPageTokens / 4) * c.idx_dim * 2;
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

namespace {
std::vector<std::vector<uint64_t>>* g_expert_hist = nullptr;
}
void set_expert_histogram(std::vector<std::vector<uint64_t>>* hist) { g_expert_hist = hist; }

// ---------------------------------------------------------------------------------------------------
// Forward

namespace {

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

// Compressor rows for one chunk, given the batch-wide projections (pk/ps already include ape):
// emitted rows are normed, roped at their block start, quant-simulated and stored to the pool.
void compress_chunk(const Dsv4Config& c, const CompressorWeights& cw, const float* pk, const float* ps,
                    float* ring_kv, float* ring_sc, k::PagedRows dst, const Scratch& scr, const RopeTables& rope,
                    const ChunkDesc& ch, hipStream_t st) {
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
    std::vector<int32_t> rpos(static_cast<size_t>(nrows));
    for (int j = 0; j < nrows; ++j) rpos[size_t(j)] = (row0 + j) * cw.ratio;
    int32_t* dpos = B<int32_t>(scr, kRowPos);
    UH_CHECK(hipMemcpyAsync(dpos, rpos.data(), rpos.size() * 4, hipMemcpyHostToDevice, st));
    k::rope(rb, d, nrows, 1, d, d - c.rope_dim, c.rope_dim / 2, dpos, rope.yarn.as<float2>(), false, st);
    if (cw.rotate) {
      k::hadamard128(rb, size_t(nrows) * d / 128, st);
      k::fake_quant_fp4(rb, d, nrows, d, 32, st);
    } else {
      k::fake_quant_fp8(rb, d, nrows, d - c.rope_dim, 64, st);
    }
    k::store_rows(rb, nrows, d, dst, row0, st);
    UH_CHECK(hipStreamSynchronize(st));  // rpos is on the host stack
  }
  k::ring_update(cpk, cps, ring_kv, ring_sc, ch.s0, ch.n, cw.ring(), width, st);
}

// Batch-wide compressor projections: x_f32 [T, dim] -> pk/ps [T, width] (+ ape by position).
void compressor_proj(const Dsv4Config& c, const CompressorWeights& cw, const Scratch& scr, const BatchDesc& b,
                     hipStream_t st) {
  const int width = cw.width();
  float* pk = B<float>(scr, kPK);
  float* ps = B<float>(scr, kPS);
  k::gemm_f32(B<float>(scr, kAF), c.dim, cw.wkv.p<bf16>(), pk, width, b.T, width, c.dim, st);
  k::gemm_f32(B<float>(scr, kAF), c.dim, cw.wgate.p<bf16>(), ps, width, b.T, width, c.dim, st);
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

  k::hc_pre(x, w.hc_attn_fn.p<float>(), w.hc_attn_scale.p<float>(), w.hc_attn_base.p<float>(), y, post, comb, T,
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
  k::gemm_w8a8(qb, qs, w.wq_a.w.p<uint8_t>(), w.wq_a.s.p<uint8_t>(), qa, c.q_lora, T, c.q_lora, D, st);
  dump(L, b, "wq_a_out", qa, size_t(T) * c.q_lora * 2, st);
  k::gemm_w8a8(qb, qs, w.wkv.w.p<uint8_t>(), w.wkv.s.p<uint8_t>(), kv, HD, T, HD, D, st);
  dump(L, b, "wkv_out", kv, size_t(T) * HD * 2, st);
  k::rmsnorm(qa, c.q_lora, w.q_norm.p<bf16>(), qr, c.q_lora, T, c.q_lora, c.norm_eps, st);
  k::act_quant_fp8(qr, c.q_lora, qb, qs, T, c.q_lora, 128, st);
  k::gemm_w8a8(qb, qs, w.wq_b.w.p<uint8_t>(), w.wq_b.s.p<uint8_t>(), q, H * HD, T, H * HD, c.q_lora, st);
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
    k::gemm_w8a8(qb, qs, w.idx_wq_b.w.p<uint8_t>(), w.idx_wq_b.s.p<uint8_t>(), qi, IH * ID, T, IH * ID, c.q_lora,
                 st);
    k::rope(qi, IH * ID, T, IH, ID, ID - rd, rd / 2, pos, freqs, false, st);
    k::hadamard128(qi, size_t(T) * IH * ID / 128, st);
    k::fake_quant_fp4(qi, IH * ID, T, IH * ID, 32, st);
    k::gemm_bf16(a, D, w.idx_wproj.p<bf16>(), wi, IH, T, IH, D, st);
    // weights_proj(x) * (softmax_scale * n_heads^-0.5): the Python double, applied as fp32.
    const double wscale = (1.0 / std::sqrt(double(ID))) * (1.0 / std::sqrt(double(IH)));
    k::scale_bf16(wi, float(wscale), size_t(T) * IH, st);
    // Indexer compressor (projections batch-wide, rows per chunk).
    compressor_proj(c, *w.idx_comp, scr, b, st);
    for (const auto& ch : b.chunks)
      compress_chunk(c, *w.idx_comp, B<float>(scr, kPK), B<float>(scr, kPS),
                     cache.iring_kv.as<float>() + size_t(ch.slot) * cache.iring * cache.iwidth,
                     cache.iring_sc.as<float>() + size_t(ch.slot) * cache.iring * cache.iwidth,
                     paged(cache.idx, b, ch.slot, kPageTokens / 4, ID), scr, rope, ch, st);
  }
  if (w.ratio) {
    compressor_proj(c, *w.comp, scr, b, st);
    for (const auto& ch : b.chunks)
      compress_chunk(c, *w.comp, B<float>(scr, kPK), B<float>(scr, kPS),
                     cache.ring_kv.as<float>() + size_t(ch.slot) * cache.ring * cache.width,
                     cache.ring_sc.as<float>() + size_t(ch.slot) * cache.ring * cache.width,
                     paged(cache.comp, b, ch.slot, kPageTokens / w.ratio, HD), scr, rope, ch, st);
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
        float* sc = B<float>(scr, kISc);
        k::indexer_scores(qi + size_t(ch.t0) * IH * ID, wi + size_t(ch.t0) * IH,
                          paged(cache.idx, b, ch.slot, kPageTokens / 4, ID), cpos, sc, ch.n, IH, ID, nkeys, 4, st);
        k::topk_rows(sc, ch.n, nkeys, kk, idx, st);
        if (b.chunks.size() == 1) dump(L, b, "indexer_out", idx, size_t(ch.n) * kk * 4, st);
      }
      akv.comp_idx = idx;
      akv.ncomp = kk;
    }
    k::sparse_attn(q + size_t(ch.t0) * H * HD, cpos, akv, w.sink.p<float>(), o + size_t(ch.t0) * H * HD, ch.n, H, HD,
                   scale, st);
    k::window_update(kv + size_t(ch.t0) * HD, window, ch.s0, ch.n, HD, st);
  }
  dump(L, b, "sa_o", o, size_t(T) * H * HD * 2, st);
  k::rope(o, H * HD, T, H, HD, HD - rd, rd / 2, pos, freqs, true, st);

  // o: grouped wo_a (bf16 einsum on dequantized fp8 weights), then wo_b.
  bf16* oa = B<bf16>(scr, kOA);
  bf16* ao = B<bf16>(scr, kAO);
  const int G = c.o_groups, R = c.o_lora, GK = H * HD / G;
  k::gemm_grouped_fp8w(o, w.wo_a.w.p<uint8_t>(), w.wo_a.s.p<uint8_t>(), oa, T, G, R, GK, st);
  dump(L, b, "wo_b_in", oa, size_t(T) * G * R * 2, st);
  k::act_quant_fp8(oa, G * R, qb, qs, T, G * R, 128, st);
  k::gemm_w8a8(qb, qs, w.wo_b.w.p<uint8_t>(), w.wo_b.s.p<uint8_t>(), ao, D, T, D, G * R, st);
  dump(L, b, "attn_out", ao, size_t(T) * D * 2, st);
  k::hc_post(ao, x, post, comb, x, T, c.hc_mult, D, st);
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
  k::hc_pre(x, w.hc_ffn_fn.p<float>(), w.hc_ffn_scale.p<float>(), w.hc_ffn_base.p<float>(), y, post, comb, T,
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
  k::gemm_f32(af, D, w.gate_w.p<bf16>(), lg, E, T, E, D, st);
  k::route(lg, w.gate_bias.p<float>(), w.hash ? w.tid2eid.p<int32_t>() : nullptr, b.ids, sel, wt, T, E, K,
           c.route_scale, st);
  dump(L, b, "gate_out0", wt, size_t(T) * K * 4, st);
  dump(L, b, "gate_out1", sel, size_t(T) * K * 4, st);

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
  for (int e = 0; e < E; ++e) off[size_t(e) + 1] = off[size_t(e)] + count[size_t(e)];
  std::vector<int32_t> eidx(size_t(T) * K), pidx(size_t(T) * K);
  std::vector<float> ewt(size_t(T) * K);
  std::vector<int> fill(off.begin(), off.end() - 1);
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
  int32_t* dpidx = B<int32_t>(scr, kPairIdx);
  UH_CHECK(hipMemcpyAsync(deidx, eidx.data(), eidx.size() * 4, hipMemcpyHostToDevice, st));
  UH_CHECK(hipMemcpyAsync(dewt, ewt.data(), ewt.size() * 4, hipMemcpyHostToDevice, st));
  UH_CHECK(hipMemcpyAsync(dpidx, pidx.data(), pidx.size() * 4, hipMemcpyHostToDevice, st));

  bf16* pair_out = B<bf16>(scr, kPairOut);
  bf16* pair_in = B<bf16>(scr, kPairIn);
  const int P = T * K;
  k::gather_rows(f, deidx, pair_in, P, D, st);
  const bool fast = w.exl3_experts && exl3_fast_enabled();
  std::vector<int32_t> row_expert;
  if (fast) {
    row_expert.resize(size_t(P));
    for (int e = 0; e < E; ++e)
      for (int p = off[size_t(e)]; p < off[size_t(e) + 1]; ++p) row_expert[size_t(p)] = e;
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
    UH_CHECK(hipMemcpyAsync(remote->in_host, pair_in, size_t(P) * D * 2, hipMemcpyDeviceToHost, st));
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
    for (const auto& j : jobs)
      UH_CHECK(hipMemcpyAsync(pair_out + size_t(j.row0) * D, remote->out_host + size_t(j.row0) * D,
                              size_t(j.cnt) * D * 2, hipMemcpyHostToDevice, st));
  }
  bf16* h1 = B<bf16>(scr, kH1);
  bf16* h3 = B<bf16>(scr, kH3);
  bf16* hh = B<bf16>(scr, kHH);
  // Shared expert (fp8 weights, no routing weight).
  bf16* shared = B<bf16>(scr, kShared);
  k::act_quant_fp8(f, D, qb, qs, T, D, 128, st);
  k::gemm_w8a8(qb, qs, w.sw1.w.p<uint8_t>(), w.sw1.s.p<uint8_t>(), h1, I, T, I, D, st);
  k::gemm_w8a8(qb, qs, w.sw3.w.p<uint8_t>(), w.sw3.s.p<uint8_t>(), h3, I, T, I, D, st);
  k::swiglu(h1, h3, nullptr, hh, T, I, c.swiglu_limit, st);
  k::act_quant_fp8(hh, I, qb, qs, T, I, 128, st);
  k::gemm_w8a8(qb, qs, w.sw2.w.p<uint8_t>(), w.sw2.s.p<uint8_t>(), shared, D, T, D, I, st);
  dump(L, b, "shared_out", shared, size_t(T) * D * 2, st);
  bf16* fo = B<bf16>(scr, kAO);
  k::moe_combine(pair_out, dpidx, shared, fo, T, K, D, st);
  dump(L, b, "ffn_out", fo, size_t(T) * D * 2, st);
  k::hc_post(fo, x, post, comb, x, T, c.hc_mult, D, st);
  UH_CHECK(hipStreamSynchronize(st));  // host vectors above
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
