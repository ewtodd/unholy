// GLM-5.3-Flash fast path (model/glm5.hpp "fast path"): the reference layer's structure on the fast kernels.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "kernels/elem_fast.hpp"
#include "kernels/exl3_moe.hpp"
#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/glm_fast.hpp"
#include "kernels/hc_fast.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/ref.hpp"
#include "kernels/ref_glm.hpp"
#include "model/glm5.hpp"

namespace unholy::model {

namespace k = unholy::kernels;
using bf16 = uint16_t;

namespace {

__global__ void f16_to_bf16_kernel(const uint16_t* x, uint16_t* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const float f = __half2float(__ushort_as_half(x[i]));
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even (finite weights)
    y[i] = uint16_t(u >> 16);
  }
}

// Tiles of a dense EXL3 linear over `rows` rows: expert 0, <= kExl3TileRows rows each.
__global__ void dense_tiles_kernel(k::Exl3Tile* t, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r0 = i * k::kExl3TileRows;
  if (r0 < rows) t[i] = k::Exl3Tile{0, r0, min(k::kExl3TileRows, rows - r0)};
}

// kv_b [H (dn + dv)][r] -> uk_t [H][r][dn] (each head's key rows, transposed) and uv [H][dv][r].
__global__ void split_kv_b_kernel(const bf16* kvb, bf16* uk_t, bf16* uv, int H, int dn, int dv, int r) {
  const size_t n = size_t(H) * (dn + dv) * r;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int col = int(i % r), row = int(i / r), h = row / (dn + dv), j = row % (dn + dv);
    if (j < dn) uk_t[(size_t(h) * r + col) * dn + j] = kvb[i];
    else uv[(size_t(h) * dv + j - dn) * r + col] = kvb[i];
  }
}

// fp16 weights -> bf16 in place of the tensor (a new buffer).
void to_bf16(DevTensor& t, int dev, hipStream_t st) {
  if (t.dtype != DType::F16) return;
  size_t n = 1;
  for (auto d : t.shape) n *= size_t(d);
  rt::DeviceBuffer b(dev, n * 2);
  const int blocks = int(std::min<size_t>((n + 255) / 256, 4096));
  f16_to_bf16_kernel<<<blocks, 256, 0, st>>>(t.p<uint16_t>(), b.as<uint16_t>(), n);
  UH_CHECK_LAUNCH();
  UH_CHECK(hipStreamSynchronize(st));
  t.buf = std::move(b);
  t.ptr = t.buf.data();
  t.dtype = DType::BF16;
}

void prepare_linear(GLinear& l, int dev, hipStream_t st) {
  if (!l) return;
  if (l.exl3()) {
    const void* p[3] = {l.x.trellis.ptr, l.x.suh.ptr, l.x.svh.ptr};
    l.tab = rt::DeviceBuffer(dev, sizeof p);
    UH_CHECK(hipMemcpy(l.tab.data(), p, sizeof p, hipMemcpyHostToDevice));
  } else {
    to_bf16(l.w, dev, st);
    if (l.w.dtype != DType::BF16) throw std::runtime_error("glm5 fast path: dense weights must be bf16, fp16 or EXL3");
  }
}

// Set once per forward (every dense linear of a forward has its rows): the dense tiles for `rows`.
void set_rows(const Glm5Scratch& s, int rows, hipStream_t st) {
  const int n = (rows + k::kExl3TileRows - 1) / k::kExl3TileRows;
  dense_tiles_kernel<<<(n + 63) / 64, 64, 0, st>>>(s.at<k::Exl3Tile>(kDenseTiles), rows);
  UH_CHECK_LAUNCH();
}

// xh (fp16, the input transform done) -> out through EXL3 linear l, over `rows` rows (tiles from set_rows).
void exl3_mm(const GLinear& l, const uint16_t* xh, bf16* out, int ldo, int rows, const Glm5Scratch& s, hipStream_t st) {
  const auto* tab = l.tab.as<const uint16_t* const>();
  const auto* tiles = s.at<k::Exl3Tile>(kDenseTiles);
  if (rows <= 8)
    k::exl3_grouped_gemv(xh, l.in, tiles, 1, tab, tab + 2, l.x.bits, s.at<float>(kYp), out, ldo, l.out, st);
  else
    k::exl3_grouped_gemm(xh, l.in, tiles, (rows + k::kExl3TileRows - 1) / k::kExl3TileRows, tab, tab + 2, l.x.bits, out,
                         ldo, l.out, st);
}

void linear(const GLinear& l, const bf16* x, int ldx, bf16* out, int ldo, int rows, const Glm5Scratch& s,
            hipStream_t st) {
  if (rows <= 0) return;
  if (l.exl3()) {
    uint16_t* xh = s.at<uint16_t>(kExl3X);
    const auto* tab = l.tab.as<const uint16_t* const>();
    k::exl3_xform_in(x, ldx, s.at<int32_t>(kZeroRows), tab + 1, xh, rows, l.in, false, st);
    exl3_mm(l, xh, out, ldo, rows, s, st);
  } else if (rows <= k::kGemvRowsCap && l.in % 512 == 0) {
    k::gemv_bf16(x, ldx, l.w.p<bf16>(), out, ldo, rows, l.out, l.in, st);
  } else {
    k::gemm_bf16_wmma(x, ldx, l.w.p<bf16>(), out, ldo, rows, l.out, l.in, st);
  }
}

// SwiGLU MLP on EXL3 gate / up / down (dense layers 0-2, the shared expert): the SwiGLU fused into the down
// projection's input transform.
void mlp(const Glm5Config& c, const GLinear& gate, const GLinear& up, const GLinear& down, const bf16* x, int rows,
         bf16* out, const Glm5Scratch& s, hipStream_t st) {
  const int I = gate.out;
  bf16* g = s.at<bf16>(kExpGate);
  bf16* u = s.at<bf16>(kExpUp);
  linear(gate, x, c.dim, g, I, rows, s, st);
  linear(up, x, c.dim, u, I, rows, s, st);
  if (!down.exl3()) throw std::runtime_error("glm5 fast path: MLP down projection must be EXL3");
  uint16_t* xh = s.at<uint16_t>(kXh2);
  k::exl3_swiglu_xform(g, u, nullptr, s.at<int32_t>(kZeroRows), down.tab.as<const uint16_t* const>() + 1, xh, rows, I,
                       c.swiglu_limit, st, false);
  exl3_mm(down, xh, out, c.dim, rows, s, st);
}

void norm(const bf16* x, const bf16* w, bf16* y, int rows, int dim, float eps, hipStream_t st) {
  if (x != y) k::rmsnorm_slices(x, dim, w, y, dim, rows, dim, eps, st);
  else k::rmsnorm_hf(x, dim, w, y, dim, rows, dim, eps, st);
}

template <typename T>
T* slot_rows(const rt::DeviceBuffer& b, int slot, size_t per_slot) {
  return b.as<T>() + size_t(slot) * per_slot;
}

void kda_fast(const Glm5Config& c, const Glm5LayerWeights& w, Glm5PagedCache& cache, const Glm5Scratch& s,
              const bf16* hn, bf16* out, const Glm5Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, H = c.lin_heads, Dh = c.lin_dim, W = c.lin_width(), C3 = 3 * W;
  bf16* qkv = s.at<bf16>(kQkv);
  bf16* qkvc = s.at<bf16>(kQkvConv);
  if (w.qkv) {
    linear(w.qkv, hn, D, qkv, C3, T, s, st);
  } else {
    linear(w.q, hn, D, qkv, C3, T, s, st);
    linear(w.k, hn, D, qkv + W, C3, T, s, st);
    linear(w.v, hn, D, qkv + 2 * W, C3, T, s, st);
  }
  const size_t conv_slot = size_t(c.conv_kernel - 1) * C3;
  for (const auto& ch : b.chunks)  // a new sequence in the slot: zero state
    if (ch.s0 == 0) {
      UH_CHECK(hipMemsetAsync(slot_rows<bf16>(cache.conv, ch.slot, conv_slot), 0, conv_slot * 2, st));
      const size_t ss = size_t(H) * Dh * Dh;
      UH_CHECK(hipMemsetAsync(slot_rows<float>(cache.state, ch.slot, ss), 0, ss * 4, st));
    }
  // The chunks in launches of up to GlmKdaBatch::kMax (backups: verify chunks' state after save_after).
  const auto batched = [&](auto&& launch) {
    k::GlmKdaBatch kb;
    for (size_t i = 0; i < b.chunks.size(); ++i) {
      const auto& ch = b.chunks[i];
      kb.c[kb.nchunks++] = {ch.t0, ch.n, ch.slot, ch.save_after};
      if (kb.nchunks == k::GlmKdaBatch::kMax || i + 1 == b.chunks.size()) {
        launch(kb);
        kb.nchunks = 0;
      }
    }
  };
  batched([&](const k::GlmKdaBatch& kb) {
    k::glm_kda_conv(qkv, C3, cache.conv.as<bf16>(), w.conv.p<bf16>(), qkvc, kb, C3, st, cache.conv_bak.as<bf16>());
  });
  bf16* low = s.at<bf16>(kLowRank);
  bf16* row = s.at<bf16>(kGateRow);
  float* g = s.at<float>(kForget);
  linear(w.f_a, hn, D, low, Dh, T, s, st);
  linear(w.f_b, low, Dh, row, W, T, s, st);
  k::kda_gate(row, w.dt_bias.p<float>(), w.a_log.p<float>(), g, T, H, Dh, c.gate_lower_bound, st);
  bf16* beta = s.at<bf16>(kBeta);
  linear(w.b, hn, D, beta, H, T, s, st);
  k::sigmoid_bf16(beta, beta, size_t(T) * H, st);
  bf16* core = s.at<bf16>(kCore);
  float* qn = s.at<float>(kKdaQ);
  float* kn = s.at<float>(kKdaK);
  float* eg = s.at<float>(kKdaE);
  k::glm_kda_prep(qkvc, qkvc + W, C3, g, qn, kn, eg, T, H, Dh, st);
  batched([&](const k::GlmKdaBatch& kb) {
    k::glm_kda_recurrent(qn, kn, eg, qkvc + 2 * W, C3, beta, cache.state.as<float>(), core, kb, H, Dh, st,
                         cache.state_bak.as<float>());
  });
  linear(w.g_a, hn, D, low, Dh, T, s, st);
  linear(w.g_b, low, Dh, row, W, T, s, st);
  bf16* on = s.at<bf16>(kOutNorm);
  k::kda_out_norm(core, row, w.o_norm.p<bf16>(), on, T, H, Dh, c.norm_eps, st);
  linear(w.o, on, W, out, D, T, s, st);
}

// A DSA layer's cache rows for the batch: the latent rows (kv_a, its norm; stored in fp8) and the indexer's keys and
// gate scores (pooled once a pool completes).
void dsa_write(const Glm5Config& c, const Glm5LayerWeights& w, Glm5PagedCache& cache, const Glm5Scratch& s,
               const bf16* hn, const Glm5Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, R = c.kv_lora, ID = c.idx_dim;
  if (!b.tables) throw std::runtime_error("glm5 fast path: DSA layers need block tables");
  bf16* lat = s.at<bf16>(kLatent);
  linear(w.kv_a, hn, D, lat, R, T, s, st);
  k::rmsnorm_hf(lat, R, w.kv_a_norm.p<bf16>(), lat, R, T, R, c.norm_eps, st);
  bf16* ik = s.at<bf16>(kIdxK);
  bf16* ig = s.at<bf16>(kIdxGate);
  linear(w.idx_wk, hn, D, ik, ID, T, s, st);
  k::layernorm(ik, w.idx_knorm_w.ptr, w.idx_knorm_b.ptr, w.idx_knorm_w.dtype == DType::F16, ik, T, ID, 1e-6f, st);
  linear(w.idx_gate, hn, D, ig, ID, T, s, st);
  for (const auto& ch : b.chunks) {
    if ((ch.s0 + ch.n + k::kGlmPageRows - 1) / k::kGlmPageRows > b.max_pages)
      throw std::runtime_error("glm5: sequence longer than the block table");
    const k::GlmPaged pg{cache.lat.as<uint8_t>(), cache.lat_s.as<uint8_t>(), cache.pk.as<bf16>(),
                         b.tables + size_t(ch.slot) * b.max_pages};
    k::glm_store_latent(lat + size_t(ch.t0) * R, pg, ch.s0, ch.n, st);
    k::glm_pool_keys(ik + size_t(ch.t0) * ID, ig + size_t(ch.t0) * ID, slot_rows<bf16>(cache.tail_k, ch.slot, 4 * ID),
                     slot_rows<bf16>(cache.tail_g, ch.slot, 4 * ID), w.idx_ape.p<float>(), pg, ch.s0, ch.n, st);
  }
}

void dsa_fast(const Glm5Config& c, const Glm5LayerWeights& w, Glm5PagedCache& cache, const Glm5Scratch& s,
              const bf16* hn, bf16* out, const Glm5Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, H = c.n_heads, R = c.kv_lora;
  const int IH = c.idx_heads, ID = c.idx_dim, pool = c.idx_pool;
  const int kpools = c.idx_topk / pool, width = c.idx_topk + pool - 1;
  dsa_write(c, w, cache, s, hn, b, st);
  bf16* qr = s.at<bf16>(kQResid);
  linear(w.q_a, hn, D, qr, c.q_lora, T, s, st);
  k::rmsnorm_hf(qr, c.q_lora, w.q_a_norm.p<bf16>(), qr, c.q_lora, T, c.q_lora, c.norm_eps, st);
  bf16* q = s.at<bf16>(kQ);
  linear(w.q_b, qr, c.q_lora, q, H * c.nope_dim, T, s, st);
  bf16* iq = s.at<bf16>(kIdxQ);
  bf16* iw = s.at<bf16>(kIdxW);
  linear(w.idx_wq_b, qr, c.q_lora, iq, IH * ID, T, s, st);
  linear(w.idx_wproj, hn, D, iw, IH, T, s, st);
  // Absorbed queries (bf16), per head: qa[t][h] = q[t][h] . W_uk(h).
  bf16* qa = s.at<bf16>(kQAbs);
  bf16* ol = s.at<bf16>(kLatentOut);
  k::gemm_grouped_bf16_wmma(q, w.w_ukT.p<bf16>(), qa, T, H, R, c.nope_dim, st);
  float* isc = s.at<float>(kIdxScores);
  int32_t* pools = s.at<int32_t>(kIdxPools);
  int32_t* sel = s.at<int32_t>(kIdxSel);
  const float iscale = 1.0f / std::sqrt(float(ID)), ascale = 1.0f / std::sqrt(float(c.nope_dim));
  for (const auto& ch : b.chunks) {
    const k::GlmPaged pg{cache.lat.as<uint8_t>(), cache.lat_s.as<uint8_t>(), cache.pk.as<bf16>(),
                         b.tables + size_t(ch.slot) * b.max_pages};
    const int npool = (ch.s0 + ch.n) / pool;
    int32_t* cpools = pools + size_t(ch.t0) * kpools;
    if (npool == 0) {
      UH_CHECK(hipMemsetAsync(cpools, 0xff, size_t(ch.n) * kpools * 4, st));
    } else {
      for (int q0 = 0; q0 < ch.n; q0 += kIdxBlock) {
        const int m = std::min(kIdxBlock, ch.n - q0), t = ch.t0 + q0;
        k::glm_idx_scores(iq + size_t(t) * IH * ID, iw + size_t(t) * IH, pg, b.pos + t, isc, npool, m, npool, iscale,
                          st);
        k::glm_topk(isc, npool, m, npool, kpools, cpools + size_t(q0) * kpools, s.get(kTopkCand), st);
      }
    }
    int32_t* csel = sel + size_t(ch.t0) * width;
    k::glm_idx_expand(cpools, b.pos + ch.t0, csel, ch.n, kpools, pool, st);
    if (ch.n <= kAttnSplitRows)  // decode / verify chunks: the keys over more workgroups
      k::glm_mla_attn_split(qa + size_t(ch.t0) * H * R, csel, width, b.pos + ch.t0, pg, ol + size_t(ch.t0) * H * R,
                            ch.n, ascale, kAttnSplits, s.at<float>(kAttnPart), st);
    else
      k::glm_mla_attn(qa + size_t(ch.t0) * H * R, csel, width, b.pos + ch.t0, pg, ol + size_t(ch.t0) * H * R, ch.n,
                      ascale, st);
  }
  bf16* o = s.at<bf16>(kAttnOut);
  k::gemm_grouped_bf16_wmma(ol, w.w_uv.p<bf16>(), o, T, H, c.v_dim, R, st);
  linear(w.o, o, H * c.v_dim, out, D, T, s, st);
}

// Routed experts on the device, in two calls so that other work can run between them. moe_start: router -> dispatch
// (kernels/moe_dispatch) -> the APU's (remote) share staged and started through `remote`; moe_post: the grouped EXL3
// experts on this card's tiles and the shared expert meanwhile -> join -> combine. A step's rows [r0, r0 + T) use
// pair rows [r0 * K, (r0 + T) * K) of the MoE buffers, plan `idx` and the tile lists from `tile0` (glm5_layers_fast's
// parts); everything else is used within one call.
struct MoePart {
  int T = 0, r0 = 0, idx = 0, tile0 = 0;
  RemoteExperts* remote = nullptr;
};

int moe_cap_tiles(const Glm5Config& c, int T) { return T * c.n_active / k::kExl3TileRows + c.n_experts + 1; }

void moe_start(const Glm5Config& c, const Glm5LayerWeights& w, const Glm5Scratch& s, const bf16* hn,
               const MoePart& m, hipStream_t st) {
  const int D = c.dim, E = c.n_experts, K = c.n_active, T = m.T, P = T * K, P0 = m.r0 * K;
  RemoteExperts* remote = m.remote;
  const bool any_remote = w.n_resident < E;
  if (any_remote && (!remote || !remote->can_enqueue()))
    throw std::runtime_error("glm5: remote experts need a backend");
  float* logits = s.at<float>(kF32Rows);
  int32_t* sel = s.at<int32_t>(kSel) + P0;
  float* rw = s.at<float>(kRouteW) + P0;
  if (T <= k::kGemvRowsCap) k::gemv_bf16_f32out(hn, D, w.router.w.p<bf16>(), logits, E, T, E, D, st);
  else k::gemm_bf16_f32out_wmma(hn, D, w.router.w.p<bf16>(), logits, E, T, E, D, st);
  k::glm_route(logits, w.router_bias.p<float>(), sel, rw, T, E, K, c.route_scale, st);

  auto* plan = s.at<k::MoePlan>(kMoePlan) + m.idx;
  auto* lt = s.at<k::Exl3Tile>(kMoeLTiles) + m.tile0;
  auto* rtl = s.at<k::Exl3Tile>(kMoeRTiles) + m.tile0;
  int32_t* eidx = s.at<int32_t>(kEIdx) + P0;
  float* ewt = s.at<float>(kEWt) + P0;
  int32_t* row_expert = s.at<int32_t>(kRowExpert) + P0;
  int32_t* pidx = s.at<int32_t>(kPairIdx) + P0;
  // The APU's inputs go as compact token rows (remote_tokens: tok2u [T], u2t [T], in the reference's pair -> token
  // buffer, which the fast path does not use otherwise).
  int32_t* tok2u = s.at<int32_t>(kPairTok) + 2 * m.r0;
  int32_t* u2t = tok2u + T;
  uint32_t* hist = s.at<uint32_t>(kRouteHist) + size_t(w.layer) * E;
  const int cap_tiles = moe_cap_tiles(c, T);
  if (any_remote && (size_t(P) > remote->capacity_rows || cap_tiles > remote->max_tiles))
    throw std::runtime_error("glm5: remote staging too small");
  if (T <= k::kMoeSmallMaxTokens) {  // decode-sized: the bookkeeping (and the remote staging) in one block
    k::MoeSmallArgs ma{sel, rw, T, K, hist, w.resident.as<uint8_t>(), E, P, cap_tiles, k::kExl3TileRows, plan, lt, rtl,
                       eidx, ewt, row_expert, pidx, any_remote, any_remote, tok2u, u2t,
                       any_remote ? remote->p2u_host : nullptr, any_remote ? remote->u2p_host : nullptr,
                       any_remote ? remote->plan_host : nullptr, any_remote ? remote->tiles_host : nullptr,
                       any_remote ? remote->w_host : nullptr, any_remote ? remote->e_host : nullptr};
    k::moe_dispatch_small(ma, st);
  } else {
    k::moe_count(sel, T, K, s.at<int32_t>(kMoeCount), E, st, hist);
    k::moe_plan(s.at<int32_t>(kMoeCount), w.resident.as<uint8_t>(), E, P, plan, lt, rtl, s.at<int32_t>(kMoeOff),
                s.at<int32_t>(kMoeFill), cap_tiles, st);
    k::moe_scatter(sel, rw, T, K, s.at<int32_t>(kMoeOff), s.at<int32_t>(kMoeFill), eidx, ewt, row_expert, pidx, st);
    if (any_remote) {  // remote_tokens first: stage_remote copies the plan with the token count
      k::remote_tokens(pidx, eidx, T, K, plan, tok2u, u2t, remote->p2u_host, remote->u2p_host, st);
      k::stage_remote(plan, rtl, ewt, row_expert, remote->plan_host, remote->tiles_host, remote->w_host,
                      remote->e_host, P, st);
    }
  }
  k::gather_rows(hn, eidx, s.at<bf16>(kExpIn) + size_t(P0) * D, P, D, st);
  if (any_remote) {  // the remote tokens' rows to the APU, which starts once the card stream gets here
    k::copy_remote_tokens(hn, D, u2t, plan, remote->in_host, T, st);
    remote->enqueue(w.layer, st, P);
  }
}

void moe_post(const Glm5Config& c, const Glm5LayerWeights& w, const Glm5Scratch& s, const bf16* hn, bf16* out,
              const MoePart& m, hipStream_t st) {
  const int D = c.dim, E = c.n_experts, K = c.n_active, T = m.T, P = T * K, P0 = m.r0 * K;
  RemoteExperts* remote = m.remote;
  const bool any_remote = w.n_resident < E;
  auto* plan = s.at<k::MoePlan>(kMoePlan) + m.idx;
  const int max_tiles = std::min(P, moe_cap_tiles(c, T));
  const float* ewt = s.at<float>(kEWt) + P0;
  const int32_t* row_expert = s.at<int32_t>(kRowExpert) + P0;
  const int32_t* pidx = s.at<int32_t>(kPairIdx) + P0;
  const int32_t* tok2u = s.at<int32_t>(kPairTok) + 2 * m.r0;
  bf16* pin = s.at<bf16>(kExpIn) + size_t(P0) * D;
  bf16* pout = s.at<bf16>(kExpOut) + size_t(P0) * D;
  const Glm5ExpertBufs eb{s.at<uint16_t>(kXh1), s.at<uint16_t>(kXh3), s.at<uint16_t>(kXh2), s.at<bf16>(kExpGate),
                          s.at<bf16>(kExpUp), s.at<float>(kYp)};
  if (w.n_resident > 0)  // (the MTP layer's experts are all on the APU)
    glm5_experts_dev(c, w, pin, ewt, row_expert, s.at<k::Exl3Tile>(kMoeLTiles) + m.tile0, max_tiles,
                     &plan->n_local_tiles, P, eb, pout, st, nullptr, &plan->remote0);
  bf16* sh = s.at<bf16>(kShared);
  mlp(c, w.sh_gate, w.sh_up, w.sh_down, hn, T, sh, s, st);
  if (!any_remote) {
    k::glm_moe_combine(pout, pidx, sh, out, T, K, D, st);
    return;
  }
  // The APU's share: one fp32 row per token (its pairs' sum), into the fp16 gate inputs' room (the local experts
  // are done with it).
  float* partial = s.at<float>(kXh1);
  remote->join(st);
  k::glm_copy_partial(reinterpret_cast<const float*>(remote->out_host), partial, &plan->n_remote_tokens, D, T, st);
  k::glm_moe_combine_partial(pout, pidx, &plan->remote0, tok2u, partial, sh, out, T, K, D, st);
}

void moe_fast(const Glm5Config& c, const Glm5LayerWeights& w, const Glm5Scratch& s, const bf16* hn, bf16* out, int T,
              hipStream_t st, RemoteExperts* remote) {
  const MoePart m{T, 0, 0, 0, remote};
  moe_start(c, w, s, hn, m, st);
  moe_post(c, w, s, hn, out, m, st);
}

void hc_pre(const Glm5Config& c, const DevTensor& fn, const DevTensor& scale, const DevTensor& base, const bf16* x,
            bf16* y, float* post, float* comb, int T, const Glm5Scratch& s, hipStream_t st) {
  if (T <= k::kHcSmallMaxTokens)
    k::hc_pre_small(x, fn.p<float>(), scale.p<float>(), base.p<float>(), y, post, comb, s.at<float>(kHcPart), T,
                    c.hc_mult, c.dim, c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  else
    k::hc_pre_tiled(x, fn.p<float>(), scale.p<float>(), base.p<float>(), y, post, comb, T, c.hc_mult, c.dim,
                    c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
}

}  // namespace

void prepare_glm5_fast(const Glm5Config& c, Glm5LayerWeights& w, int dev, hipStream_t st) {
  rt::DeviceGuard g(dev);
  for (GLinear* l : {&w.o, &w.q, &w.k, &w.v, &w.qkv, &w.f_a, &w.f_b, &w.b, &w.g_a, &w.g_b, &w.q_a, &w.q_b, &w.kv_a,
                     &w.kv_b, &w.idx_wq_b, &w.idx_wk, &w.idx_wproj, &w.idx_gate, &w.gate, &w.up, &w.down, &w.router,
                     &w.sh_gate, &w.sh_up, &w.sh_down})
    prepare_linear(*l, dev, st);
  if (!w.kda && w.kv_b) {  // the absorbed attention's per-head weights (not on an experts-only copy)
    const int H = c.n_heads, dn = c.nope_dim, dv = c.v_dim, r = c.kv_lora;
    for (DevTensor* t : {&w.w_ukT, &w.w_uv}) {
      const size_t n = size_t(H) * (t == &w.w_ukT ? dn : dv) * r;
      t->buf = rt::DeviceBuffer(dev, n * 2);
      t->ptr = t->buf.data();
      t->dtype = DType::BF16;
    }
    split_kv_b_kernel<<<4096, 256, 0, st>>>(w.kv_b.w.p<bf16>(), w.w_ukT.p<bf16>(), w.w_uv.p<bf16>(), H, dn, dv, r);
    UH_CHECK_LAUNCH();
    UH_CHECK(hipStreamSynchronize(st));
  }
  if (w.dense) return;
  const size_t E = size_t(c.n_experts);
  std::vector<const void*> tab(9 * E, nullptr);
  std::vector<uint8_t> resident(E, 0);
  w.n_resident = 0;
  for (size_t e = 0; e < E; ++e) {
    if (e >= w.e_gate.size() || !w.e_gate[e]) continue;
    const GLinear* m3[3] = {&w.e_gate[e], &w.e_up[e], &w.e_down[e]};
    for (int m = 0; m < 3; ++m) {
      if (!m3[m]->exl3()) throw std::runtime_error("glm5 fast path: routed experts must be EXL3");
      if (w.exl3_bits[m] && w.exl3_bits[m] != m3[m]->x.bits)
        throw std::runtime_error("glm5 fast path: mixed EXL3 bit widths in one expert projection");
      w.exl3_bits[m] = m3[m]->x.bits;
      tab[size_t(m * 3 + 0) * E + e] = m3[m]->x.trellis.ptr;
      tab[size_t(m * 3 + 1) * E + e] = m3[m]->x.suh.ptr;
      tab[size_t(m * 3 + 2) * E + e] = m3[m]->x.svh.ptr;
    }
    resident[e] = 1;
    ++w.n_resident;
  }
  w.exl3_tab = rt::DeviceBuffer(dev, tab.size() * sizeof(void*));
  UH_CHECK(hipMemcpy(w.exl3_tab.data(), tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice));
  w.resident = rt::DeviceBuffer(dev, E);
  UH_CHECK(hipMemcpy(w.resident.data(), resident.data(), E, hipMemcpyHostToDevice));
}

void prepare_glm5_fast_head(Glm5HeadWeights& h, int dev, hipStream_t st) {
  rt::DeviceGuard g(dev);
  prepare_linear(h.head, dev, st);
}

namespace {

// A part of a step: rows [m.r0, m.r0 + b.T) (b: their chunks, t0 relative to m.r0, pos from row m.r0).
struct LayerPart {
  Glm5Batch b;
  MoePart m;
};

// Attention, then the FFN side's mixing and norm (the FFN's input in the part's rows of hn).
void layer_pre(const Glm5Config& c, const Glm5LayerWeights& w, Glm5PagedCache& cache, const Glm5Scratch& s,
               uint16_t* x, const LayerPart& p, hipStream_t st) {
  const int T = p.b.T, D = c.dim, hc = c.hc_mult, r0 = p.m.r0;
  bf16* h = s.at<bf16>(kH) + size_t(r0) * D;
  bf16* hn = s.at<bf16>(kHn) + size_t(r0) * D;
  bf16* out = s.at<bf16>(kOut) + size_t(r0) * D;
  float* post = s.at<float>(kPost) + size_t(r0) * hc;
  float* comb = s.at<float>(kComb) + size_t(r0) * hc * hc;
  uint16_t* xr = x + size_t(r0) * hc * D;
  set_rows(s, T, st);
  hc_pre(c, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, xr, h, post, comb, T, s, st);
  norm(h, w.input_norm.p<bf16>(), hn, T, D, c.norm_eps, st);
  if (w.kda) kda_fast(c, w, cache, s, hn, out, p.b, st);
  else dsa_fast(c, w, cache, s, hn, out, p.b, st);
  k::hc_post_slices(out, xr, post, comb, xr, T, hc, D, st);
  hc_pre(c, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, xr, h, post, comb, T, s, st);
  norm(h, w.post_norm.p<bf16>(), hn, T, D, c.norm_eps, st);
}

void layer_start(const Glm5Config& c, const Glm5LayerWeights& w, const Glm5Scratch& s, const LayerPart& p,
                 hipStream_t st) {
  if (!w.dense) moe_start(c, w, s, s.at<bf16>(kHn) + size_t(p.m.r0) * c.dim, p.m, st);
}

// The FFN (the dense MLP, or the MoE layer_start began) and its mixing back into x. tiles: the dense tile list was
// last set for another part's rows.
void layer_post(const Glm5Config& c, const Glm5LayerWeights& w, const Glm5Scratch& s, uint16_t* x,
                const LayerPart& p, hipStream_t st, bool tiles) {
  const int T = p.b.T, D = c.dim, hc = c.hc_mult, r0 = p.m.r0;
  bf16* hn = s.at<bf16>(kHn) + size_t(r0) * D;
  bf16* out = s.at<bf16>(kOut) + size_t(r0) * D;
  uint16_t* xr = x + size_t(r0) * hc * D;
  if (tiles) set_rows(s, T, st);
  if (w.dense) mlp(c, w.gate, w.up, w.down, hn, T, out, s, st);
  else moe_post(c, w, s, hn, out, p.m, st);
  k::hc_post_slices(out, xr, s.at<float>(kPost) + size_t(r0) * hc, s.at<float>(kComb) + size_t(r0) * hc * hc, xr, T,
                    hc, D, st);
}

constexpr int kSplitMinRows = 2048;  // steps this large run as two parts (1024-row steps: 8% slower split)
constexpr int kSplitMinPiece = 64;   // a chunk is cut only into pieces this long (attention's decode kernels: <= 4)

// Where a step splits into two parts: near the middle, between chunks or where a chunk crosses a page boundary;
// both parts >= a quarter of the step, the second within the second channel's capacity. 0: no split.
int split_cut(const Glm5Config& c, const Glm5Batch& b, const RemoteExperts& second) {
  const int T = b.T, cap = int(second.capacity_rows / size_t(c.n_active));
  if (T < kSplitMinRows) return 0;
  int best = 0;
  const auto consider = [&](int cut) {
    if (cut < T / 4 || T - cut < T / 4 || T - cut > cap) return;
    if (best == 0 || std::abs(cut - T / 2) < std::abs(best - T / 2)) best = cut;
  };
  for (const auto& ch : b.chunks) {
    if (ch.t0 > 0) consider(ch.t0);
    for (int q = (ch.s0 / k::kGlmPageRows + 1) * k::kGlmPageRows; q < ch.s0 + ch.n; q += k::kGlmPageRows)
      if (q - ch.s0 >= kSplitMinPiece && ch.s0 + ch.n - q >= kSplitMinPiece) consider(ch.t0 + q - ch.s0);
  }
  return best;
}

void split_batch(const Glm5Batch& b, int cut, Glm5Batch& lo, Glm5Batch& hi) {
  for (Glm5Batch* p : {&lo, &hi}) {
    p->tables = b.tables;
    p->max_pages = b.max_pages;
  }
  lo.T = cut;
  lo.pos = b.pos;
  hi.T = b.T - cut;
  hi.pos = b.pos + cut;
  for (const Glm5Chunk& ch : b.chunks) {
    if (ch.t0 + ch.n <= cut) {
      lo.chunks.push_back(ch);
    } else if (ch.t0 >= cut) {
      Glm5Chunk h = ch;
      h.t0 -= cut;
      hi.chunks.push_back(h);
    } else {  // pieces >= kSplitMinPiece: never a verify chunk (save_after)
      const int n1 = cut - ch.t0;
      lo.chunks.push_back({ch.slot, ch.s0, n1, ch.t0});
      hi.chunks.push_back({ch.slot, ch.s0 + n1, ch.n - n1, 0});
    }
  }
}

}  // namespace

void glm5_layer_forward_fast(const Glm5Config& c, const Glm5LayerWeights& w, Glm5PagedCache& cache,
                             const Glm5Scratch& s, uint16_t* x, const Glm5Batch& b, hipStream_t st,
                             RemoteExperts* remote) {
  if (b.T > s.max_tokens) throw std::runtime_error("glm5_layer_forward_fast: batch larger than the scratch");
  const LayerPart p{b, MoePart{b.T, 0, 0, 0, remote}};
  layer_pre(c, w, cache, s, x, p, st);
  layer_start(c, w, s, p, st);
  layer_post(c, w, s, x, p, st, false);
}

void glm5_layers_fast(const Glm5Config& c, std::span<const Glm5LayerWeights> w, std::span<Glm5PagedCache> cache,
                      const Glm5Scratch& s, uint16_t* x, const Glm5Batch& b, hipStream_t st, RemoteExperts* remote,
                      RemoteExperts* remote2) {
  if (b.T > s.max_tokens) throw std::runtime_error("glm5_layers_fast: batch larger than the scratch");
  const int n = int(w.size()), cut = remote2 ? split_cut(c, b, *remote2) : 0;
  if (cut == 0) {
    for (int i = 0; i < n; ++i) glm5_layer_forward_fast(c, w[size_t(i)], cache[size_t(i)], s, x, b, st, remote);
    return;
  }
  LayerPart p[2];
  split_batch(b, cut, p[0].b, p[1].b);
  p[0].m = {cut, 0, 0, 0, remote};
  p[1].m = {b.T - cut, cut, 1, moe_cap_tiles(c, cut), remote2};
  // Layer i's second part and layer i + 1's first part run on the card while the APU has the other part's experts.
  layer_pre(c, w[0], cache[0], s, x, p[0], st);
  layer_start(c, w[0], s, p[0], st);
  for (int i = 0; i < n; ++i) {
    const auto& wi = w[size_t(i)];
    layer_pre(c, wi, cache[size_t(i)], s, x, p[1], st);
    layer_start(c, wi, s, p[1], st);
    layer_post(c, wi, s, x, p[0], st, true);
    if (i + 1 < n) {
      layer_pre(c, w[size_t(i) + 1], cache[size_t(i) + 1], s, x, p[0], st);
      layer_start(c, w[size_t(i) + 1], s, p[0], st);
    }
    layer_post(c, wi, s, x, p[1], st, true);
  }
}

void glm5_head_forward_fast(const Glm5Config& c, const Glm5HeadWeights& hw, const Glm5Scratch& s, const uint16_t* x,
                            int T, float* logits, hipStream_t st) {
  const int D = c.dim, V = c.vocab, G = s.head_rows;
  bf16* h = s.at<bf16>(kH);
  bf16* hn = s.at<bf16>(kHn);
  bf16* lb = s.at<bf16>(kHeadOut);
  for (int r0 = 0; r0 < T; r0 += G) {
    const int n = std::min(G, T - r0);
    set_rows(s, n, st);
    k::hc_mean(x + size_t(r0) * c.hc_mult * D, h, n, c.hc_mult, D, st);
    norm(h, hw.norm.p<bf16>(), hn, n, D, c.norm_eps, st);
    linear(hw.head, hn, D, lb, V, n, s, st);
    k::cast_bf16_f32(lb, logits + size_t(r0) * V, size_t(n) * V, st);
  }
}

void glm5_head_logits(const Glm5Config& c, const Glm5HeadWeights& hw, const Glm5Scratch& s, const uint16_t* hn, int T,
                      float* logits, hipStream_t st) {
  const int D = c.dim, V = c.vocab, G = s.head_rows;
  bf16* lb = s.at<bf16>(kHeadOut);
  for (int r0 = 0; r0 < T; r0 += G) {
    const int n = std::min(G, T - r0);
    set_rows(s, n, st);
    linear(hw.head, hn + size_t(r0) * D, D, lb, V, n, s, st);
    k::cast_bf16_f32(lb, logits + size_t(r0) * V, size_t(n) * V, st);
  }
}

void glm5_mtp_in(const Glm5Config& c, const Glm5MtpWeights& m, const uint16_t* emb, const uint16_t* h, uint16_t* x0,
                 int rows, const Glm5Scratch& s, hipStream_t st) {
  if (rows <= 0) return;
  bf16* cat = s.at<bf16>(kExpIn);  // [rows, 2 dim]
  k::glm_mtp_cat(emb, h, m.enorm.p<bf16>(), m.hnorm.p<bf16>(), cat, rows, c.dim, c.norm_eps, st);
  set_rows(s, rows, st);
  linear(m.eh, cat, 2 * c.dim, x0, c.dim, rows, s, st);
}

void glm5_mtp_cache(const Glm5Config& c, const Glm5MtpWeights& m, Glm5PagedCache& cache, const Glm5Scratch& s,
                    const uint16_t* x0, const Glm5Batch& b, hipStream_t st) {
  if (b.T <= 0) return;
  set_rows(s, b.T, st);
  bf16* hn = s.at<bf16>(kHn);
  norm(x0, m.layer.input_norm.p<bf16>(), hn, b.T, c.dim, c.norm_eps, st);
  dsa_write(c, m.layer, cache, s, hn, b, st);
}

void glm5_mtp_forward(const Glm5Config& c, const Glm5MtpWeights& m, Glm5PagedCache& cache, const Glm5Scratch& s,
                      uint16_t* x, uint16_t* hn_out, const Glm5Batch& b, hipStream_t st, RemoteExperts* remote) {
  const int T = b.T, D = c.dim;
  if (T <= 0) return;
  if (T > s.max_tokens) throw std::runtime_error("glm5_mtp_forward: batch larger than the scratch");
  set_rows(s, T, st);
  bf16* hn = s.at<bf16>(kHn);
  bf16* out = s.at<bf16>(kOut);
  norm(x, m.layer.input_norm.p<bf16>(), hn, T, D, c.norm_eps, st);
  dsa_fast(c, m.layer, cache, s, hn, out, b, st);
  k::glm_add_bf16(x, out, size_t(T) * D, st);
  norm(x, m.layer.post_norm.p<bf16>(), hn, T, D, c.norm_eps, st);
  moe_fast(c, m.layer, s, hn, out, T, st, remote);
  k::glm_add_bf16(x, out, size_t(T) * D, st);
  norm(x, m.head_norm.p<bf16>(), hn_out, T, D, c.norm_eps, st);
}

}  // namespace unholy::model

namespace unholy::model {

size_t glm5_paged_cache_bytes(const Glm5Config& c, bool kda, int slots, int num_pages, bool backups) {
  if (kda)
    return (backups ? 2 : 1) * size_t(slots) *
           ((c.conv_kernel - 1) * 3 * size_t(c.lin_width()) * 2 + size_t(c.lin_heads) * c.lin_dim * c.lin_dim * 4);
  const size_t rows = size_t(num_pages) * kernels::kGlmPageRows;
  return rows * c.kv_lora + rows * (c.kv_lora / 64) + rows / c.idx_pool * c.idx_dim * 2 +
         size_t(slots) * 2 * c.idx_pool * c.idx_dim * 2;
}

Glm5PagedCache make_glm5_paged_cache(const Glm5Config& c, int layer, int slots, int num_pages, int dev,
                                     hipStream_t st, bool backups) {
  if (c.kv_lora != 512 || c.idx_dim != 128 || c.idx_pool != 4 || c.n_heads != 64 || c.idx_heads != 32)
    throw std::runtime_error("glm5 fast path: kernels/glm_fast.hip is built for GLM-5.3-Flash's shapes");
  Glm5PagedCache k;
  k.layer = layer;
  k.slots = slots;
  k.num_pages = num_pages;
  k.kda = c.is_kda(layer);
  rt::DeviceGuard g(dev);
  auto zeroed = [&](size_t bytes) {
    rt::DeviceBuffer b(dev, bytes);
    UH_CHECK(hipMemsetAsync(b.data(), 0, bytes, st));
    return b;
  };
  if (k.kda) {
    k.conv = zeroed(size_t(slots) * (c.conv_kernel - 1) * 3 * c.lin_width() * 2);
    k.state = zeroed(size_t(slots) * c.lin_heads * c.lin_dim * c.lin_dim * 4);
    if (backups) {
      k.conv_bak = zeroed(k.conv.size());
      k.state_bak = zeroed(k.state.size());
    }
  } else {
    const size_t rows = size_t(num_pages) * kernels::kGlmPageRows;
    k.lat = zeroed(rows * c.kv_lora);
    k.lat_s = zeroed(rows * (c.kv_lora / 64));
    k.pk = zeroed(rows / c.idx_pool * c.idx_dim * 2);
    k.tail_k = zeroed(size_t(slots) * c.idx_pool * c.idx_dim * 2);
    k.tail_g = zeroed(size_t(slots) * c.idx_pool * c.idx_dim * 2);
  }
  UH_CHECK(hipStreamSynchronize(st));
  return k;
}

}  // namespace unholy::model

namespace unholy::model {

void glm5_experts_dev(const Glm5Config& c, const Glm5LayerWeights& w, const uint16_t* in, const float* rw,
                      const int32_t* row_expert, const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev,
                      int P, const Glm5ExpertBufs& b, uint16_t* out, hipStream_t st, const int* row0_dev,
                      const int* row1_dev, const int32_t* x_rows) {
  const int D = c.dim, I = c.moe_inter, E = c.n_experts;
  max_tiles = std::min(max_tiles, P);  // every tile has >= 1 row
  k::exl3_xform_in(in, D, row_expert, w.tab(0, 1, E), b.xh1, P, D, false, st, row0_dev, row1_dev, x_rows);
  k::exl3_xform_in(in, D, row_expert, w.tab(1, 1, E), b.xh3, P, D, false, st, row0_dev, row1_dev, x_rows);
  if (P / c.n_active <= 8) {  // <= 8 tokens: every expert has <= 8 rows -> the GEMVs
    const int* nt = ntiles_dev;
    k::exl3_grouped_gemv(b.xh1, D, tiles, max_tiles, w.tab(0, 0, E), w.tab(0, 2, E), w.exl3_bits[0], b.yp, b.h1, I, I,
                         st, nt);
    k::exl3_grouped_gemv(b.xh3, D, tiles, max_tiles, w.tab(1, 0, E), w.tab(1, 2, E), w.exl3_bits[1], b.yp, b.h3, I, I,
                         st, nt);
    k::exl3_swiglu_xform(b.h1, b.h3, rw, row_expert, w.tab(2, 1, E), b.xh2, P, I, c.swiglu_limit, st, false,
                         row0_dev, row1_dev);
    k::exl3_grouped_gemv(b.xh2, I, tiles, max_tiles, w.tab(2, 0, E), w.tab(2, 2, E), w.exl3_bits[2], b.yp, out, D, D,
                         st, nt);
    return;
  }
  k::exl3_grouped_gemm(b.xh1, D, tiles, max_tiles, w.tab(0, 0, E), w.tab(0, 2, E), w.exl3_bits[0], b.h1, I, I, st,
                       ntiles_dev);
  k::exl3_grouped_gemm(b.xh3, D, tiles, max_tiles, w.tab(1, 0, E), w.tab(1, 2, E), w.exl3_bits[1], b.h3, I, I, st,
                       ntiles_dev);
  k::exl3_swiglu_xform(b.h1, b.h3, rw, row_expert, w.tab(2, 1, E), b.xh2, P, I, c.swiglu_limit, st, false,
                       row0_dev, row1_dev);
  k::exl3_grouped_gemm(b.xh2, I, tiles, max_tiles, w.tab(2, 0, E), w.tab(2, 2, E), w.exl3_bits[2], out, D, D, st,
                       ntiles_dev);
}

}  // namespace unholy::model

namespace unholy::model {

void glm5_restore_kda(Glm5PagedCache& cache, int slot, hipStream_t st) {
  if (!cache.kda) return;
  if (!cache.state_bak.data()) throw std::runtime_error("glm5_restore_kda: the cache has no backups");
  const size_t cb = cache.conv.size() / cache.slots, sb = cache.state.size() / cache.slots;
  UH_CHECK(hipMemcpyAsync(cache.conv.as<uint8_t>() + slot * cb, cache.conv_bak.as<uint8_t>() + slot * cb, cb,
                          hipMemcpyDeviceToDevice, st));
  UH_CHECK(hipMemcpyAsync(cache.state.as<uint8_t>() + slot * sb, cache.state_bak.as<uint8_t>() + slot * sb, sb,
                          hipMemcpyDeviceToDevice, st));
}

}  // namespace unholy::model

namespace unholy::model {

void prepare_glm5_fast_mtp(const Glm5Config& c, Glm5MtpWeights& m, int dev, hipStream_t st) {
  prepare_glm5_fast(c, m.layer, dev, st);
  rt::DeviceGuard g(dev);
  prepare_linear(m.eh, dev, st);
}

}  // namespace unholy::model
