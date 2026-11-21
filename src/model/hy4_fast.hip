// Hy4 flash layer forward (model/hy4.hpp): transformers' HYV4FlashDecoderLayer on the fast kernels.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/elem_fast.hpp"
#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/glm_fast.hpp"
#include "kernels/hy4.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/mx_moe.hpp"
#include "kernels/ref.hpp"
#include "kernels/ref_glm.hpp"
#include "model/hy4.hpp"

namespace unholy::model {

namespace k = unholy::kernels;
using bf16 = uint16_t;

namespace {

std::vector<std::vector<uint64_t>>* g_hist = nullptr;

bool gemv_ok(const Hy4Linear& l, int rows) {
  return rows <= k::kGemvRowsCap && (l.in % 512 == 0 || (l.in < 512 && l.in % 64 == 0)) && l.out % 16 == 0;
}

// The current forward's dense tile lists (one matrix, the batch's rows): 8-row tiles for the GEMV, 64-row ones for the
// GEMM; set by set_dense_tiles once per layer forward (every linear of a layer has the batch's rows).
struct DenseTiles {
  const k::Exl3Tile* t8 = nullptr;
  const k::Exl3Tile* t64 = nullptr;
  int rows = 0;
};
DenseTiles g_tiles;

// A tile buffer's lists depend only on the row count: written when it changes (*cached: the rows the buffer holds,
// kept with the buffer's owner; every buffer is written and read on one stream).
void set_dense_tiles_at(k::Exl3Tile* t, int rows, hipStream_t st, int* cached = nullptr) {
  const int n8 = (rows + k::kHy4GemvRows - 1) / k::kHy4GemvRows;
  g_tiles = {t, t + n8, rows};
  if (cached && *cached == rows) return;
  k::hy4_dense_tiles(t, rows, k::kHy4GemvRows, st);
  k::hy4_dense_tiles(t + n8, rows, k::kHy4TileRows, st);
  if (cached) *cached = rows;
}

void set_dense_tiles(const Hy4Scratch& s, int rows, hipStream_t st) {
  set_dense_tiles_at(s.at<k::Exl3Tile>(kBufDenseTiles), rows, st, &s.dense_tile_rows);
}

void linear(const Hy4Linear& l, const bf16* x, int ldx, bf16* out, int ldo, int rows, hipStream_t st) {
  if (rows <= 0) return;
  if (l.fp8()) {
    if (rows != g_tiles.rows) throw std::runtime_error("hy4: fp8 linear outside its forward's rows");
    const auto* tw = l.tab.as<const uint8_t* const>();
    const auto* ts = reinterpret_cast<const uint16_t* const*>(tw + 1);
    if (rows <= kHy4GemvTokens) {  // the cards: the few-row WMMA kernel (memory-bound at any row count); else GEMV
      const int n8 = (rows + k::kHy4GemvRows - 1) / k::kHy4GemvRows;
      if (!k::hy4_wmma_gemv(0, x, ldx, l.in, g_tiles.t8, n8, tw, reinterpret_cast<const void* const*>(ts), out, ldo,
                            l.out, st))
        k::hy4_fp8_grouped_gemv(x, ldx, l.in, g_tiles.t8, n8, tw, ts, out, ldo, l.out, st);
    } else {
      if (ldx != l.in) throw std::runtime_error("hy4: fp8 GEMM rows must be packed");
      k::hy4_fp8_grouped_gemm(x, l.in, g_tiles.t64, (rows + k::kHy4TileRows - 1) / k::kHy4TileRows, tw, ts, out, ldo,
                              l.out, st);
    }
    return;
  }
  if (rows <= kHy4GemvTokens && rows == g_tiles.rows &&
      k::hy4_wmma_gemv(2, x, ldx, l.in, g_tiles.t8, (rows + k::kHy4GemvRows - 1) / k::kHy4GemvRows, nullptr, nullptr,
                       out, ldo, l.out, st, nullptr, l.w.ptr))
    return;
  if (gemv_ok(l, rows)) k::gemv_bf16(x, ldx, l.w.p<bf16>(), out, ldo, rows, l.out, l.in, st);
  else k::gemm_bf16_wmma(x, ldx, l.w.p<bf16>(), out, ldo, rows, l.out, l.in, st);
}

void linear_f32(const Hy4Linear& l, const bf16* x, int ldx, float* out, int ldo, int rows, hipStream_t st) {
  if (rows <= 0) return;
  if (gemv_ok(l, rows)) k::gemv_bf16_f32out(x, ldx, l.w.p<bf16>(), out, ldo, rows, l.out, l.in, st);
  else k::gemm_bf16_f32out_wmma(x, ldx, l.w.p<bf16>(), out, ldo, rows, l.out, l.in, st);
}

void norm(bf16* x, int ld, const DevTensor& w, int rows, int dim, float eps, hipStream_t st) {
  k::rmsnorm_hf(x, ld, w.p<bf16>(), x, ld, rows, dim, eps, st);
}

// ---- attention ----------------------------------------------------------------------------------------------------

void attention(const Hy4Config& c, const Hy4LayerWeights& w, Hy4Cache& cache, const Hy4Scratch& s, const bf16* hn,
               bf16* out, const Hy4Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, H = c.heads(w.layer), R = k::kHy4Row;
  const float theta = c.theta(w.layer);
  bf16* qr = s.at<bf16>(kBufQr);
  linear(w.q_a, hn, D, qr, c.q_lora, T, st);
  norm(qr, c.q_lora, w.q_a_norm, T, c.q_lora, c.eps, st);
  bf16* q = s.at<bf16>(kBufQ);
  linear(w.q_b, qr, c.q_lora, q, H * 256, T, st);
  k::hy4_rope(q, H * 256, T, H, 256, c.nope, b.pos, theta, st, b.pos3);
  bf16* kv = s.at<bf16>(kBufKv);
  linear(w.kv_a, hn, D, kv, R, T, st);
  norm(kv, R, w.kv_a_norm, T, c.kv_lora, c.eps, st);
  k::hy4_rope(kv, R, T, 1, R, c.kv_lora, b.pos, theta, st, b.pos3);
  bf16* g = s.at<bf16>(kBufG);
  linear(w.gate, hn, D, g, H * 256, T, st);
  bf16* qn = s.at<bf16>(kBufQn);
  bf16* qa = s.at<bf16>(kBufQa);
  bf16* qp = s.at<bf16>(kBufQp);
  k::hy4_split_q(q, qn, T, H, st);
  k::gemm_grouped_bf16_wmma(qn, w.w_ukT.p<bf16>(), qa, T, H, c.kv_lora, c.nope, st);
  k::hy4_pack_q(qa, q, qp, T, H, st);
  bf16* ol = s.at<bf16>(kBufOl);
  const float scale = 1.0f / std::sqrt(float(c.nope + c.rope));
  float* ws = s.at<float>(kBufAttnSplit);  // few-query attention: keys split over workgroups
  const size_t ws_bytes = s.bufs[size_t(kBufAttnSplit)].size();

  if (w.sparse) {
    if (!b.tables) throw std::runtime_error("hy4: sparse layers need block tables");
    const int IH = c.idx_heads, ID = c.idx_dim, kg = c.idx_groups(), width = c.idx_width();
    bf16* tk = s.at<bf16>(kBufTk);
    linear(w.idx_k, hn, D, tk, ID, T, st);
    norm(tk, ID, w.idx_tok_norm, T, ID, c.eps, st);
    bf16* iq = s.at<bf16>(kBufIq);
    linear(w.idx_q, qr, c.q_lora, iq, IH * ID, T, st);
    k::hy4_rope(iq, IH * ID, T, IH, ID, ID - c.rope, b.pos, c.theta_sparse, st, b.pos3);
    bf16* iw = s.at<bf16>(kBufIw);
    linear(w.idx_w, hn, D, iw, IH, T, st);
    const float wscale = 1.0f / std::sqrt(float(IH)) / std::sqrt(float(ID));
    float* isc = s.at<float>(kBufIScores);
    int32_t* groups = s.at<int32_t>(kBufIGroups);
    int32_t* sel = s.at<int32_t>(kBufISel);
    for (const auto& ch : b.chunks) {
      if ((ch.s0 + ch.n + k::kHy4PageRows - 1) / k::kHy4PageRows > b.max_pages)
        throw std::runtime_error("hy4: sequence longer than the block table");
      const int32_t* table = b.tables + size_t(ch.slot) * b.max_pages;
      bf16* rows = cache.rows.as<bf16>();
      k::hy4_page_store(kv + size_t(ch.t0) * R, rows, table, ch.s0, ch.n, st);
      const k::Hy4Pooled pg{cache.pooled.as<bf16>(), table};
      if (ch.verify) {  // the tail and the chunk's token keys, for restoring the tail after rejected drafts
        if (ch.n > kHy4MaxVerify) throw std::runtime_error("hy4: verify chunk longer than the backups");
        const size_t T4 = size_t(c.idx_pool) * ID;
        UH_CHECK(hipMemcpyAsync(cache.tail_bak.as<bf16>() + ch.slot * T4, cache.tail.as<bf16>() + ch.slot * T4, T4 * 2,
                                hipMemcpyDeviceToDevice, st));
        UH_CHECK(hipMemcpyAsync(cache.vkeys.as<bf16>() + size_t(ch.slot) * kHy4MaxVerify * ID, tk + size_t(ch.t0) * ID,
                                size_t(ch.n) * ID * 2, hipMemcpyDeviceToDevice, st));
      }
      k::Hy4GroupRope gr;
      if (b.pos3) {
        gr.rows = b.pos3 + size_t(ch.t0) * 3;
        std::copy_n(ch.g0, 3, gr.g0);
      }
      k::hy4_idx_pool(tk + size_t(ch.t0) * ID, cache.tail.as<bf16>() + size_t(ch.slot) * c.idx_pool * ID,
                      w.idx_wv.p<float>(), w.idx_wg.p<float>(), w.idx_ape.p<float>(), w.idx_knorm.p<float>(), c.eps,
                      c.theta_sparse, pg, ch.s0, ch.n, st, gr);
      const k::Hy4Keys keys{rows, table, nullptr, nullptr, 0};
      const int last = ch.s0 + ch.n - 1;
      const int nsave = std::min(ch.n, kHy4MaxVerify);  // DFly2: the recalled positions of the chunk's last rows
      const auto save = [&](const int32_t* sl) {
        if (cache.dsel.data())
          k::hy4_dfly_save_sel(sl, width, b.pos, ch.t0 + ch.n - nsave, nsave,
                               cache.dsel.as<int32_t>() + size_t(ch.slot) * kHy4MaxVerify * 2048,
                               cache.dcount.as<int32_t>() + size_t(ch.slot) * kHy4MaxVerify, kg, st);
      };
      if (last / c.idx_pool <= kg) {  // every query recalls all of its groups: dense causal attention
        k::hy4_mla_attn(qp + size_t(ch.t0) * H * R, H, keys, nullptr, 0, 0, b.pos + ch.t0, w.sinks.p<float>(),
                        ol + size_t(ch.t0) * H * c.kv_lora, ch.n, scale, st, nullptr, ws, ws_bytes);
        save(nullptr);
        continue;
      }
      const int npool = (ch.s0 + ch.n) / c.idx_pool;
      for (int q0 = 0; q0 < ch.n; q0 += kHy4IdxBlock) {
        const int m = std::min(kHy4IdxBlock, ch.n - q0), t = ch.t0 + q0;
        k::hy4_idx_scores(iq + size_t(t) * IH * ID, iw + size_t(t) * IH, wscale, pg, b.pos + t, isc, npool, m, npool,
                          st);
        k::glm_topk(isc, npool, m, npool, kg, groups + size_t(t) * kg, s.get(kBufITopk), st);
      }
      k::hy4_idx_expand(groups + size_t(ch.t0) * kg, b.pos + ch.t0, sel + size_t(ch.t0) * width, ch.n, kg, st);
      save(sel);
      k::hy4_mla_attn(qp + size_t(ch.t0) * H * R, H, keys, sel + size_t(ch.t0) * width, width, 0, b.pos + ch.t0,
                      w.sinks.p<float>(), ol + size_t(ch.t0) * H * c.kv_lora, ch.n, scale, st, nullptr, ws, ws_bytes);
    }
  } else {
    for (const auto& ch : b.chunks) {
      bf16* ring = cache.ring.as<bf16>() + size_t(ch.slot) * k::kHy4Window * R;
      const k::Hy4Keys keys{nullptr, nullptr, ring, kv + size_t(ch.t0) * R, ch.s0};
      k::hy4_mla_attn(qp + size_t(ch.t0) * H * R, H, keys, nullptr, 0, c.window, b.pos + ch.t0, w.sinks.p<float>(),
                      ol + size_t(ch.t0) * H * c.kv_lora, ch.n, scale, st, nullptr, ws, ws_bytes);
      if (ch.verify) {  // the ring rows the chunk overwrites, for restoring them after rejected drafts
        if (ch.n > kHy4MaxVerify) throw std::runtime_error("hy4: verify chunk longer than the backups");
        k::hy4_ring_backup(ring, cache.ring_bak.as<bf16>() + size_t(ch.slot) * kHy4MaxVerify * R, ch.s0, ch.n, st);
      }
      k::hy4_ring_store(kv + size_t(ch.t0) * R, ring, ch.s0, ch.n, st);
    }
  }
  bf16* o = s.at<bf16>(kBufO);
  k::gemm_grouped_bf16_wmma(ol, w.w_uv.p<bf16>(), o, T, H, c.v_dim, c.kv_lora, st);
  k::hy4_gate_mul(o, g, size_t(T) * H * c.v_dim, st);
  linear(w.o, o, H * c.v_dim, out, D, T, st);
  norm(out, D, w.post_proj_norm, T, D, c.eps, st);
}

// ---- MLP / MoE ----------------------------------------------------------------------------------------------------

void dense_mlp(const Hy4Config& c, const Hy4LayerWeights& w, const Hy4Scratch& s, const bf16* hn, bf16* out, int T,
               hipStream_t st) {
  const int D = c.dim, I = w.gate_p.out;
  bf16* g = s.at<bf16>(kBufMlpG);
  bf16* u = s.at<bf16>(kBufMlpU);
  bf16* h = s.at<bf16>(kBufMlpH);
  linear(w.gate_p, hn, D, g, I, T, st);
  linear(w.up_p, hn, D, u, I, T, st);
  k::hy4_swiglu(g, u, I, h, T, I, st);
  linear(w.down_p, h, I, out, D, T, st);
  norm(out, D, w.post_mlp_norm, T, D, c.eps, st);
}

int cap_tiles(const Hy4Config& c, int T, int tile_rows) { return T * c.n_active / tile_rows + c.n_experts + 1; }

void moe(const Hy4Config& c, const Hy4LayerWeights& w, const Hy4Scratch& s, const bf16* hn, bf16* out, int T,
         hipStream_t st, RemoteExperts* remote) {
  const int D = c.dim, E = c.n_experts, K = c.n_active, P = T * K, Lt = c.latent, I = c.moe_inter;
  const bool any_remote = w.n_resident < E;
  if (any_remote && (!remote || !remote->can_enqueue())) throw std::runtime_error("hy4: remote experts need a backend");
  float* logits = s.at<float>(kBufLogits);
  int32_t* sel = s.at<int32_t>(kBufSel);
  float* rw = s.at<float>(kBufRw);
  linear_f32(w.router, hn, D, logits, E, T, st);
  bf16* lat = s.at<bf16>(kBufLat);
  linear(w.pre_expert, hn, D, lat, Lt, T, st);
  const bool small = T <= k::kMoeSmallMaxTokens;  // the routing and the bookkeeping in one block (moe_dispatch_small)
  if (!small) k::glm_route(logits, w.router_bias.p<float>(), sel, rw, T, E, K, c.route_scale, st);

  auto* plan = s.at<k::MoePlan>(kBufMoePlan);
  auto* lt = s.at<k::Exl3Tile>(kBufMoeLTiles);
  auto* rtl = s.at<k::Exl3Tile>(kBufMoeRTiles);
  int32_t* eidx = s.at<int32_t>(kBufEIdx);
  float* ewt = s.at<float>(kBufEWt);
  int32_t* row_expert = s.at<int32_t>(kBufRowExpert);
  int32_t* pidx = s.at<int32_t>(kBufPidx);
  int32_t* tok2u = s.at<int32_t>(kBufTok2u);
  int32_t* u2t = tok2u + T;
  uint32_t* hist = s.at<uint32_t>(kBufRouteHist) + size_t(w.layer) * E;  // the running routing histogram
  const int tile_rows = hy4_expert_tile_rows(w.mx, T);
  const int ct = cap_tiles(c, T, tile_rows);
  if (any_remote && (size_t(P) > remote->capacity_rows || ct > remote->max_tiles))
    throw std::runtime_error("hy4: remote staging too small");
  if (small) {
    k::MoeSmallArgs ma{sel, rw, T, K, hist, w.resident.as<uint8_t>(), E, P, ct, tile_rows, plan, lt, rtl, eidx, ewt,
                       row_expert, pidx, any_remote, any_remote, tok2u, u2t,
                       any_remote ? remote->p2u_host : nullptr, any_remote ? remote->u2p_host : nullptr,
                       any_remote ? remote->plan_host : nullptr, any_remote ? remote->tiles_host : nullptr,
                       any_remote ? remote->w_host : nullptr, any_remote ? remote->e_host : nullptr};
    ma.route_logits = logits;
    ma.route_bias = w.router_bias.p<float>();
    ma.route_scale = c.route_scale;
    if (any_remote) {
      ma.x = lat;
      ma.dim = Lt;
      ma.x_dst = remote->in_host;
    }
    k::moe_dispatch_small(ma, st);
    if (any_remote) remote->enqueue(w.layer, st, P);
  } else {
    k::moe_count(sel, T, K, s.at<int32_t>(kBufMoeCount), E, st, hist);
    k::moe_plan(s.at<int32_t>(kBufMoeCount), w.resident.as<uint8_t>(), E, P, plan, lt, rtl, s.at<int32_t>(kBufMoeOff),
                s.at<int32_t>(kBufMoeFill), ct, st, tile_rows);
    k::moe_scatter(sel, rw, T, K, s.at<int32_t>(kBufMoeOff), s.at<int32_t>(kBufMoeFill), eidx, ewt, row_expert, pidx,
                   st);
    if (any_remote) {  // the remote tokens' latent rows to the APU, which starts once the card stream gets here
      k::remote_tokens(pidx, eidx, T, K, plan, tok2u, u2t, remote->p2u_host, remote->u2p_host, st);
      k::stage_remote(plan, rtl, ewt, row_expert, remote->plan_host, remote->tiles_host, remote->w_host,
                      remote->e_host, P, st);
      k::copy_remote_tokens(lat, Lt, u2t, plan, remote->in_host, T, st);
      remote->enqueue(w.layer, st, P);
    }
  }
  bf16* pin = s.at<bf16>(kBufExpIn);
  bf16* pout = s.at<bf16>(kBufPout);
  k::gather_rows(lat, eidx, pin, P, Lt, st);
  if (w.n_resident > 0)
    hy4_experts_dev(c, w, pin, ewt, lt, std::min(P, ct), &plan->n_local_tiles, P, s.at<bf16>(kBufGu), s.at<bf16>(kBufEh),
                    pout, st, nullptr, &plan->remote0);
  // The shared expert meanwhile.
  bf16* sg = s.at<bf16>(kBufShG);
  bf16* su = s.at<bf16>(kBufShU);
  bf16* sh = s.at<bf16>(kBufShH);
  bf16* shared = s.at<bf16>(kBufShared);
  linear(w.sh_gate, hn, D, sg, I, T, st);
  linear(w.sh_up, hn, D, su, I, T, st);
  k::hy4_swiglu(sg, su, I, sh, T, I, st);
  linear(w.sh_down, sh, I, shared, D, T, st);
  bf16* msum = s.at<bf16>(kBufMsum);
  if (any_remote) {
    remote->join(st);
    k::hy4_moe_sum(pout, pidx, &plan->remote0, tok2u, reinterpret_cast<const float*>(remote->out_host), msum, T, K, Lt,
                   st);
  } else {
    k::hy4_moe_sum(pout, pidx, nullptr, nullptr, nullptr, msum, T, K, Lt, st);
  }
  bf16* routed = s.at<bf16>(kBufRouted);
  linear(w.post_expert, msum, Lt, routed, D, T, st);
  k::hy4_add(routed, shared, out, size_t(T) * D, st);
  norm(out, D, w.combine_norm, T, D, c.eps, st);
}

void engram(const Hy4Config& c, const Hy4LayerWeights& w, const Hy4Scratch& s, uint16_t* x, const Hy4Batch& b,
            hipStream_t st) {
  const int T = b.T, D = c.dim, m = w.engram;
  if (!b.eng_q[m] || !b.eng_s[m] || !b.eng_mask) throw std::runtime_error("hy4: Engram rows missing from the batch");
  if (b.eng_flag[m])  // the host's readers fill the rows while the layers before run
    UH_CHECK(hipStreamWaitValue32(st, b.eng_flag[m], b.eng_ticket, hipStreamWaitValueGte, 0xffffffffu));
  bf16* rows = s.at<bf16>(kBufEngRows);
  float* kv = s.at<float>(kBufEngKv);
  k::hy4_engram_rows(b.eng_q[m], b.eng_s[m], w.eng_lookup_norm.p<bf16>(), rows, T, st);
  linear_f32(w.eng_kv, rows, w.eng_kv.in, kv, w.eng_kv.out, T, st);
  k::hy4_engram_mix(kv, w.eng_bias.p<bf16>(), w.eng_nk.p<bf16>(), w.eng_nq.p<bf16>(), w.eng_pre_gamma.p<bf16>(),
                    w.eng_hcs.p<bf16>(), b.eng_mask, x, T, c.hc, D, st);
}

}  // namespace

void set_hy4_expert_histogram(std::vector<std::vector<uint64_t>>* hist) { g_hist = hist; }

void hy4_embed(const Hy4Config& c, const Hy4HeadWeights& h, const int32_t* ids, uint16_t* x, int T, hipStream_t st) {
  k::embed_hc(ids, h.embed.p<bf16>(), x, T, c.dim, c.hc, st, c.vocab);
}

void hy4_experts_dev(const Hy4Config& c, const Hy4LayerWeights& w, const uint16_t* xa, const float* rw,
                     const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev, int P, uint16_t* gu,
                     uint16_t* eh, uint16_t* out, hipStream_t st, const int* row0_dev, const int* row1_dev) {
  const int E = c.n_experts, I = c.moe_inter, Lt = c.latent;
  max_tiles = std::min(max_tiles, P);
  if (w.nv) {  // NVFP4: gate, up, the clamped SwiGLU, down (8-row tiles and the GEMVs for decode-sized steps)
    const bool gemv = P / c.n_active <= kHy4GemvTokens;
    uint16_t* g = gu;
    uint16_t* u = gu + size_t(P) * I;
    const auto lin = [&](const uint16_t* x, int K, int m3, uint16_t* y, int N) {
      if (gemv && k::hy4_wmma_gemv(1, x, K, K, tiles, max_tiles, w.mx_tab(m3, 0, E),
                                   reinterpret_cast<const void* const*>(w.mx_tab(m3, 1, E)), y, N, N, st, ntiles_dev,
                                   nullptr, P / c.n_active))
        return;  // the cards (the APU keeps the GEMV)
      if (gemv)
        k::hy4_nvfp4_grouped_gemv(x, K, K, tiles, max_tiles, w.mx_tab(m3, 0, E), w.mx_tab(m3, 1, E), y, N, N, st,
                                  ntiles_dev);
      else
        k::hy4_nvfp4_grouped_gemm(x, K, tiles, max_tiles, w.mx_tab(m3, 0, E), w.mx_tab(m3, 1, E), y, N, N, st,
                                  ntiles_dev);
    };
    lin(xa, Lt, 0, g, I);
    lin(xa, Lt, 1, u, I);
    k::hy4_swiglu_clamp2(g, u, eh, P, I, c.swiglu_limit, st);
    lin(eh, I, 2, out, Lt);
    k::hy4_scale_rows(out, rw, P, Lt, st, row0_dev, row1_dev);
    return;
  }
  if (!w.mx && P / c.n_active <= kHy4GemvTokens) {  // decode-sized: 8-row tiles, WMMA on the cards, else the GEMV
    if (!k::hy4_wmma_gemv(0, xa, Lt, Lt, tiles, max_tiles, w.gu_w(), reinterpret_cast<const void* const*>(w.gu_s(E)),
                          gu, 2 * I, 2 * I, st, ntiles_dev, nullptr, P / c.n_active))
      k::hy4_fp8_grouped_gemv(xa, Lt, Lt, tiles, max_tiles, w.gu_w(), w.gu_s(E), gu, 2 * I, 2 * I, st, ntiles_dev);
    k::hy4_swiglu_clamp(gu, eh, P, I, c.swiglu_limit, st);
    if (!k::hy4_wmma_gemv(0, eh, I, I, tiles, max_tiles, w.dn_w(E), reinterpret_cast<const void* const*>(w.dn_s(E)),
                          out, Lt, Lt, st, ntiles_dev, nullptr, P / c.n_active))
      k::hy4_fp8_grouped_gemv(eh, I, I, tiles, max_tiles, w.dn_w(E), w.dn_s(E), out, Lt, Lt, st, ntiles_dev);
    k::hy4_scale_rows(out, rw, P, Lt, st, row0_dev, row1_dev);
    return;
  }
  if (w.mx) {  // the mx_moe kernels on bf16 rows (no activation quantization), gate and up in one launch
    const bool gemv = k::mx_use_gemv(P / c.n_active);
    const int rows = std::min(P / c.n_active, k::kExl3TileRows);
    uint16_t* g = gu;
    uint16_t* u = gu + size_t(P) * I;
    if (gemv) {
      k::mx_grouped_gemv(xa, Lt, tiles, max_tiles, w.mx_tab(0, 0, E), w.mx_tab(0, 1, E), g, I, I, st, ntiles_dev,
                         w.mx_tab(1, 0, E), w.mx_tab(1, 1, E), u, rows);
      k::hy4_swiglu_clamp2(g, u, eh, P, I, c.swiglu_limit, st);
      k::mx_grouped_gemv(eh, I, tiles, max_tiles, w.mx_tab(2, 0, E), w.mx_tab(2, 1, E), out, Lt, Lt, st, ntiles_dev,
                         nullptr, nullptr, nullptr, rows);
    } else {
      k::mx_grouped_gemm(xa, Lt, tiles, max_tiles, w.mx_tab(0, 0, E), w.mx_tab(0, 1, E), g, I, I, st, ntiles_dev,
                         w.mx_tab(1, 0, E), w.mx_tab(1, 1, E), u, rows);
      k::hy4_swiglu_clamp2(g, u, eh, P, I, c.swiglu_limit, st);
      k::mx_grouped_gemm(eh, I, tiles, max_tiles, w.mx_tab(2, 0, E), w.mx_tab(2, 1, E), out, Lt, Lt, st, ntiles_dev,
                         nullptr, nullptr, nullptr, rows);
    }
    k::hy4_scale_rows(out, rw, P, Lt, st, row0_dev, row1_dev);
    return;
  }
  k::hy4_fp8_grouped_gemm(xa, Lt, tiles, max_tiles, w.gu_w(), w.gu_s(E), gu, 2 * I, 2 * I, st, ntiles_dev);
  k::hy4_swiglu_clamp(gu, eh, P, I, c.swiglu_limit, st);
  k::hy4_fp8_grouped_gemm(eh, I, tiles, max_tiles, w.dn_w(E), w.dn_s(E), out, Lt, Lt, st, ntiles_dev);
  k::hy4_scale_rows(out, rw, P, Lt, st, row0_dev, row1_dev);
}

void hy4_layer_forward(const Hy4Config& c, const Hy4LayerWeights& w, Hy4Cache& cache, const Hy4Scratch& s,
                       uint16_t* x, const Hy4Batch& b, hipStream_t st, RemoteExperts* remote) {
  const int T = b.T, D = c.dim;
  if (T > s.max_tokens) throw std::runtime_error("hy4_layer_forward: batch larger than the scratch");
  if (T <= 0) return;
  bf16* h = s.at<bf16>(kBufH);
  bf16* hn = s.at<bf16>(kBufHn);
  bf16* out = s.at<bf16>(kBufOut);
  float* post = s.at<float>(kBufPost);
  set_dense_tiles(s, T, st);
  if (w.engram >= 0) engram(c, w, s, x, b, st);
  // UNHOLY_HY4_ENGRAM_ONLY (debugging, tools/ref/hy4_oracle/engram_check.py): the layer stops after the Engram add.
  static const bool engram_only = std::getenv("UNHOLY_HY4_ENGRAM_ONLY") != nullptr;
  if (engram_only) return;
  k::hy4_ihc_pre(x, w.hc_attn_fn.p<float>(), w.hc_attn_scale.p<float>(), w.hc_attn_base.p<float>(), h, post, T, c.hc,
                 D, c.eps, c.hc_eps, c.hc_mag, st, w.input_norm.p<bf16>(), hn);
  attention(c, w, cache, s, hn, out, b, st);
  k::hy4_ihc_post(out, x, post, T, c.hc, D, st);
  k::hy4_ihc_pre(x, w.hc_ffn_fn.p<float>(), w.hc_ffn_scale.p<float>(), w.hc_ffn_base.p<float>(), h, post, T, c.hc, D,
                 c.eps, c.hc_eps, c.hc_mag, st, w.post_norm.p<bf16>(), hn);
  if (w.dense) dense_mlp(c, w, s, hn, out, T, st);
  else moe(c, w, s, hn, out, T, st, remote);
  k::hy4_ihc_post(out, x, post, T, c.hc, D, st);
}

void hy4_head_forward(const Hy4Config& c, const Hy4HeadWeights& hw, const Hy4Scratch& s, const uint16_t* x, int T,
                      float* logits, hipStream_t st, uint16_t* hn_out) {
  const int D = c.dim, V = c.vocab, G = std::min(s.max_tokens, 64);
  bf16* h = s.at<bf16>(kBufH);
  for (int r0 = 0; r0 < T; r0 += G) {
    const int n = std::min(G, T - r0);
    k::hy4_hc_head(x + size_t(r0) * c.hc * D, hw.hc_fn.p<float>(), hw.hc_scale.p<float>(), hw.hc_base.p<float>(), h, n,
                   c.hc, D, c.eps, c.hc_eps, st);
    norm(h, D, hw.norm, n, D, c.eps, st);
    if (hn_out)
      UH_CHECK(hipMemcpyAsync(hn_out + size_t(r0) * D, h, size_t(n) * D * 2, hipMemcpyDeviceToDevice, st));
    linear_f32(hw.head, h, D, logits + size_t(r0) * V, V, n, st);
  }
}

}  // namespace unholy::model

// ---- DFly2 -------------------------------------------------------------------------------------------------------

namespace unholy::model {

namespace {

enum DflyBuf : int {
  dIds, dPos, dAnchor, dHistP, dCntP, dLists, dCounts, dX, dResid, dA, dAtt, dMlp, dQr, dQ, dQn, dQa, dQp, dKv, dG,
  dOl, dO, dMg, dMu, dMh, dEin, dE, dHrow, dH, dCat, dF, dRowSel, dHIdx, dHid, dEmb, dHn, dEn, dCat2, dCg, dCu, dCh,
  dCorr, dLogits, dConf, dTiles, dBonus, dAttnWs, dArgmaxWs, dHid5, dCand, dTopkWs, dLse, dTokTmp, dConfTmp, dBufs
};
constexpr int kDflyW = 2048 + 8;  // a row's key list: the history (<= 2048), then the block's rows (<= 6)
// The head chain's shortlist: the uncorrected rows' top K tokens, among which the corrected logits pick.
constexpr int kDraftShortlist = 256;

std::vector<size_t> dfly_sizes(const Hy4Config& c, int Rm) {
  const size_t R = size_t(Rm), N = R * kHy4DraftRows, D = c.dim, H = c.heads_sparse, I = 9216;
  std::vector<size_t> z = {
      N * 4, N * 4, R * 4, 8 * R * 8, 8 * R * 8, N * kDflyW * 4, N * 4,      // ids .. counts
      N * D * 2, N * D * 2, N * D * 2, N * D * 2, N * D * 2,                 // x, resid, a, att, mlp
      N * c.q_lora * 2, N * H * 256 * 2, N * H * 192 * 2, N * H * 512 * 2, N * H * 576 * 2, N * 576 * 2,
      N * H * 256 * 2, N * H * 512 * 2, N * H * 256 * 2,                    // g, ol, o
      N * I * 2, N * I * 2, N * I * 2,                                       // MLP
      R * D * 2, R * D * 2, R * D * 2, R * D * 2, R * 2 * D * 2, R * D * 2,  // fusion
      kHy4DraftTokens * R * 4, R * 4,                                        // row selections, hidden rows
      R * D * 2, R * D * 2, R * D * 2, R * D * 2, R * 2 * D * 2, R * D * 2, R * D * 2, R * D * 2, R * D * 2,
      size_t(kHy4DraftTokens) * R * c.vocab * 4, size_t(kHy4DraftTokens) * R * 4,  // logits [5R][V], conf [5][R]
      2 * (N / k::kHy4GemvRows + N / k::kHy4TileRows + 2) * sizeof(k::Exl3Tile), R * 4,  // N rows, R rows
      k::hy4_attn_split_bytes(c.heads_sparse), kHy4DraftTokens * R * size_t(k::kHy4ArgmaxSlices) * 3 * 4,
      // the shortlisted head: the uncorrected rows, their top-K, top-k work, log-sum-exp, argmax outputs
      kHy4DraftTokens * R * D * 2, kHy4DraftTokens * R * size_t(kDraftShortlist) * 4,
      k::glm_topk_cand_bytes(kHy4DraftTokens * Rm, c.vocab, kDraftShortlist), kHy4DraftTokens * R * 4,
      kHy4DraftTokens * R * 4, kHy4DraftTokens * R * 4};
  if (z.size() != size_t(dBufs)) throw std::logic_error("dfly scratch: buffer list out of date");
  return z;
}

size_t host_words(int Rm) { return size_t(Rm) * (2 * kHy4DraftRows + 1 + kHy4DraftTokens + 2) + 4 + size_t(Rm) * 20; }

}  // namespace

Hy4DflyScratch::~Hy4DflyScratch() {
  if (host) (void)hipHostFree(host);
}
Hy4DflyScratch::Hy4DflyScratch(Hy4DflyScratch&& o) noexcept
    : max_reqs(o.max_reqs), bufs(std::move(o.bufs)), tile_rows{o.tile_rows[0], o.tile_rows[1]}, host(o.host) {
  o.host = nullptr;
}
Hy4DflyScratch& Hy4DflyScratch::operator=(Hy4DflyScratch&& o) noexcept {
  if (this != &o) {
    if (host) (void)hipHostFree(host);
    max_reqs = o.max_reqs;
    bufs = std::move(o.bufs);
    tile_rows[0] = o.tile_rows[0];
    tile_rows[1] = o.tile_rows[1];
    host = o.host;
    o.host = nullptr;
  }
  return *this;
}

size_t hy4_dfly_scratch_bytes(const Hy4Config& c, int max_reqs) {
  size_t b = 0;
  for (size_t z : dfly_sizes(c, max_reqs)) b += (std::max<size_t>(z, 16) + 255) / 256 * 256;
  return b;
}

Hy4DflyScratch make_hy4_dfly_scratch(const Hy4Config& c, int dev, int max_reqs) {
  Hy4DflyScratch s;
  s.max_reqs = max_reqs;
  rt::DeviceGuard g(dev);
  for (size_t z : dfly_sizes(c, max_reqs)) s.bufs.emplace_back(dev, std::max<size_t>(z, 16));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&s.host), host_words(max_reqs) * 4, hipHostMallocPortable));
  return s;
}

void hy4_dfly_draft(const Hy4Config& c, const Hy4DflyWeights& w, const Hy4HeadWeights& head,
                    const std::vector<Hy4Cache*>& target, const int32_t* tables, int max_pages, const uint16_t* hidden,
                    Hy4DflyScratch& s, const std::vector<Hy4DflyReq>& reqs, int32_t* tokens, hipStream_t st,
                    float* conf) {
  const int R = int(reqs.size()), nL = int(w.layers.size());
  if (R == 0) return;
  if (R > s.max_reqs) throw std::runtime_error("dfly: more requests than its scratch");
  if (int(target.size()) != nL) throw std::runtime_error("dfly: target caches");
  const int N = R * kHy4DraftRows, D = c.dim, H = c.heads_sparse, RW = k::kHy4Row, V = c.vocab, I = w.layers[0].mg.out;
  const auto buf = [&](DflyBuf b) { return s.bufs[size_t(b)].data(); };
  const auto bb = [&](DflyBuf b) { return static_cast<bf16*>(buf(b)); };
  const auto ib = [&](DflyBuf b) { return static_cast<int32_t*>(buf(b)); };
  // The call's inputs, staged in pinned memory: ids, positions, anchors, head-chain rows, hidden rows, bonus tokens; the
  // history pointers of every draft layer.
  int32_t* hp = s.host;
  int32_t *ids = hp, *pos = ids + N, *anc = pos + N, *rsel = anc + R, *hidx = rsel + kHy4DraftTokens * R,
          *bonus = hidx + R;
  uint64_t* ptrs = reinterpret_cast<uint64_t*>(hp + ((bonus + R - hp + 1) & ~1));
  uint64_t *histp = ptrs, *cntp = ptrs + size_t(nL) * R;
  for (int r = 0; r < R; ++r) {
    const auto& q = reqs[size_t(r)];
    const int a = q.anchor, row = (a - 1) % kHy4MaxVerify;
    for (int j = 0; j < kHy4DraftRows; ++j) {
      ids[r * kHy4DraftRows + j] = j == 0 ? q.bonus : w.mask_id;
      pos[r * kHy4DraftRows + j] = q.rope + j;  // rotary positions (the block's slots are the anchor's + j)
    }
    anc[r] = a;
    for (int t = 0; t < kHy4DraftTokens; ++t) rsel[t * R + r] = r * kHy4DraftRows + t + 1;
    hidx[r] = q.slot * kHy4MaxVerify + row;
    bonus[r] = q.bonus;
    for (int L = 0; L < nL; ++L) {
      histp[size_t(L) * R + r] = reinterpret_cast<uint64_t>(target[size_t(L)]->dsel.as<int32_t>() +
                                                             (size_t(q.slot) * kHy4MaxVerify + row) * 2048);
      cntp[size_t(L) * R + r] = reinterpret_cast<uint64_t>(target[size_t(L)]->dcount.as<int32_t>() +
                                                            size_t(q.slot) * kHy4MaxVerify + row);
    }
  }
  const auto up = [&](void* dst, const void* src, size_t bytes) {
    UH_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, st));
  };
  up(buf(dIds), ids, size_t(N) * 4);
  up(buf(dPos), pos, size_t(N) * 4);
  up(buf(dAnchor), anc, size_t(R) * 4);
  up(buf(dRowSel), rsel, size_t(kHy4DraftTokens) * R * 4);
  up(buf(dHIdx), hidx, size_t(R) * 4);
  up(buf(dBonus), bonus, size_t(R) * 4);
  up(buf(dHistP), histp, size_t(nL) * R * 8);
  up(buf(dCntP), cntp, size_t(nL) * R * 8);
  const int32_t* dpos = ib(dPos);

  // Embeddings; row 0 of each block fused with the target's hidden state.
  bf16* x = bb(dX);
  // fp8 linears run on tile lists of their row count: the block rows (N) and one row per request (R)
  auto* tN = static_cast<k::Exl3Tile*>(buf(dTiles));
  auto* tR = tN + (N + k::kHy4GemvRows - 1) / k::kHy4GemvRows + (N + k::kHy4TileRows - 1) / k::kHy4TileRows;
  set_dense_tiles_at(tR, R, st, &s.tile_rows[1]);
  k::gather_rows(w.embed.p<bf16>(), ib(dIds), x, N, D, st);
  k::gather_rows(w.embed.p<bf16>(), ib(dBonus), bb(dEin), R, D, st);
  k::rmsnorm_hf(bb(dEin), D, w.enorm.p<bf16>(), bb(dE), D, R, D, c.eps, st);
  k::gather_rows(hidden, ib(dHIdx), bb(dHrow), R, D, st);
  k::rmsnorm_hf(bb(dHrow), D, w.hnorm.p<bf16>(), bb(dH), D, R, D, c.eps, st);
  k::hy4_concat2(bb(dE), bb(dH), bb(dCat), R, D, st);
  linear(w.eh, bb(dCat), 2 * D, bb(dF), D, R, st);
  k::hy4_gated_add(x, kHy4DraftRows * D, w.fgate.p<bf16>(), bb(dF), R, D, st);

  set_dense_tiles_at(tN, N, st, &s.tile_rows[0]);
  bf16 *resid = bb(dResid), *a = bb(dA), *att = bb(dAtt), *mlp = bb(dMlp);
  const float scale = 1.0f / std::sqrt(float(c.nope + c.rope));
  for (int L = 0; L < nL; ++L) {
    const Hy4DflyLayer& W = w.layers[size_t(L)];
    if (L > 0) k::hy4_add(mlp, resid, x, size_t(N) * D, st);
    UH_CHECK(hipMemcpyAsync(resid, x, size_t(N) * D * 2, hipMemcpyDeviceToDevice, st));
    k::rmsnorm_hf(x, D, W.in_norm.p<bf16>(), a, D, N, D, c.eps, st);
    // attention
    bf16* qr = bb(dQr);
    linear(W.q_a, a, D, qr, c.q_lora, N, st);
    k::rmsnorm_hf(qr, c.q_lora, W.q_a_norm.p<bf16>(), qr, c.q_lora, N, c.q_lora, c.eps, st);
    bf16* q = bb(dQ);
    linear(W.q_b, qr, c.q_lora, q, H * 256, N, st);
    k::hy4_rope(q, H * 256, N, H, 256, c.nope, dpos, c.theta_sparse, st);
    bf16* kv = bb(dKv);
    linear(W.kv_a, a, D, kv, RW, N, st);
    k::rmsnorm_hf(kv, RW, W.kv_a_norm.p<bf16>(), kv, RW, N, c.kv_lora, c.eps, st);
    k::hy4_rope(kv, RW, N, 1, RW, c.kv_lora, dpos, c.theta_sparse, st);
    bf16* g = bb(dG);
    linear(W.gate, a, D, g, H * 256, N, st);
    k::hy4_split_q(q, bb(dQn), N, H, st);
    k::gemm_grouped_bf16_wmma(bb(dQn), W.w_ukT.p<bf16>(), bb(dQa), N, H, c.kv_lora, c.nope, st);
    k::hy4_pack_q(bb(dQa), q, bb(dQp), N, H, st);
    k::hy4_dfly_lists(reinterpret_cast<const int32_t* const*>(static_cast<uint64_t*>(buf(dHistP)) + size_t(L) * R),
                      reinterpret_cast<const int32_t* const*>(static_cast<uint64_t*>(buf(dCntP)) + size_t(L) * R),
                      ib(dAnchor), ib(dLists), ib(dCounts), R, kDflyW, st);
    for (int r = 0; r < R; ++r) {
      const auto& rq = reqs[size_t(r)];
      k::Hy4Keys keys{target[size_t(L)]->rows.as<bf16>(), tables + size_t(rq.slot) * max_pages, nullptr, nullptr, 0,
                      kv + size_t(r) * kHy4DraftRows * RW, rq.anchor};
      k::hy4_mla_attn(bb(dQp) + size_t(r) * kHy4DraftRows * H * RW, H, keys,
                      ib(dLists) + size_t(r) * kHy4DraftRows * kDflyW, kDflyW, 0, dpos + r * kHy4DraftRows,
                      W.sinks.p<float>(), bb(dOl) + size_t(r) * kHy4DraftRows * H * c.kv_lora, kHy4DraftRows, scale,
                      st, ib(dCounts) + r * kHy4DraftRows, static_cast<float*>(buf(dAttnWs)),
                      s.bufs[size_t(dAttnWs)].size());
    }
    k::gemm_grouped_bf16_wmma(bb(dOl), W.w_uv.p<bf16>(), bb(dO), N, H, c.v_dim, c.kv_lora, st);
    k::hy4_gate_mul(bb(dO), g, size_t(N) * H * c.v_dim, st);
    linear(W.o, bb(dO), H * c.v_dim, att, D, N, st);
    k::rmsnorm_hf(att, D, W.post_proj_norm.p<bf16>(), att, D, N, D, c.eps, st);
    k::hy4_add(att, resid, x, size_t(N) * D, st);
    UH_CHECK(hipMemcpyAsync(resid, x, size_t(N) * D * 2, hipMemcpyDeviceToDevice, st));
    // MLP (dense SwiGLU, no clamp)
    k::rmsnorm_hf(x, D, W.post_norm.p<bf16>(), a, D, N, D, c.eps, st);
    linear(W.mg, a, D, bb(dMg), I, N, st);
    linear(W.mu, a, D, bb(dMu), I, N, st);
    k::hy4_swiglu(bb(dMg), bb(dMu), I, bb(dMh), N, I, st);
    linear(W.md, bb(dMh), I, mlp, D, N, st);
  }
  k::hy4_add(mlp, resid, x, size_t(N) * D, st);
  k::rmsnorm_hf(x, D, w.final_norm.p<bf16>(), x, D, N, D, c.eps, st);

  // The head chain: row t + 1 of each block, corrected with the previous token's embedding, through the LM head.
  // By default one pass of the LM head over all 5 uncorrected rows (logits are linear in the row: lm_head(hid +
  // delta) = lm_head(hid) + lm_head(delta)), each row's top kDraftShortlist tokens, and per step only those tokens'
  // corrections (hy4_draft_pick): 2 LM-head passes a step instead of 6. Drafts only (outputs are the verify step's);
  // UNHOLY_HY4_DFLY_EXACT_HEAD=1: the full head per step, as the reference does.
  set_dense_tiles_at(tR, R, st, &s.tile_rows[1]);
  const int32_t* prev = ib(dBonus);
  static const bool exact_head = std::getenv("UNHOLY_HY4_DFLY_EXACT_HEAD") != nullptr;
  if (!exact_head) {
    const int R5 = kHy4DraftTokens * R;
    float* base = static_cast<float*>(buf(dLogits));
    bf16* hid5 = bb(dHid5);  // rows t * R + r
    k::gather_rows(x, ib(dRowSel), hid5, R5, D, st);
    linear_f32(head.head, hid5, D, base, V, R5, st);
    k::hy4_argmax_conf(base, R5, V, ib(dTokTmp), static_cast<float*>(buf(dConfTmp)), static_cast<float*>(buf(dArgmaxWs)),
                       st, static_cast<float*>(buf(dLse)));
    k::glm_topk(base, V, R5, V, kDraftShortlist, ib(dCand), buf(dTopkWs), st);
    for (int t = 0; t < kHy4DraftTokens; ++t) {
      const bf16* hid = hid5 + size_t(t) * R * D;
      k::gather_rows(w.embed.p<bf16>(), prev, bb(dEmb), R, D, st);
      k::rmsnorm_hf(hid, D, w.c_hnorm.p<bf16>(), bb(dHn), D, R, D, c.eps, st);
      k::rmsnorm_hf(bb(dEmb), D, w.c_enorm.p<bf16>(), bb(dEn), D, R, D, c.eps, st);
      k::hy4_concat2(bb(dHn), bb(dEn), bb(dCat2), R, D, st);
      linear(w.cg, bb(dCat2), 2 * D, bb(dCg), D, R, st);
      linear(w.cu, bb(dCat2), 2 * D, bb(dCu), D, R, st);
      k::hy4_swiglu(bb(dCg), bb(dCu), D, bb(dCh), R, D, st);
      linear(w.cd, bb(dCh), D, bb(dHn), D, R, st);  // the correction (delta) of each request's row
      k::hy4_draft_pick(base + size_t(t) * R * V, static_cast<float*>(buf(dLse)) + t * R,
                        ib(dCand) + size_t(t) * R * kDraftShortlist, kDraftShortlist, head.head.w.p<bf16>(), bb(dHn),
                        R, V, D, tokens + t * R, static_cast<float*>(buf(dConf)) + t * R, st);
      prev = tokens + t * R;
    }
  }
  for (int t = 0; exact_head && t < kHy4DraftTokens; ++t) {
    k::gather_rows(x, ib(dRowSel) + t * R, bb(dHid), R, D, st);
    k::gather_rows(w.embed.p<bf16>(), prev, bb(dEmb), R, D, st);
    k::rmsnorm_hf(bb(dHid), D, w.c_hnorm.p<bf16>(), bb(dHn), D, R, D, c.eps, st);
    k::rmsnorm_hf(bb(dEmb), D, w.c_enorm.p<bf16>(), bb(dEn), D, R, D, c.eps, st);
    k::hy4_concat2(bb(dHn), bb(dEn), bb(dCat2), R, D, st);
    linear(w.cg, bb(dCat2), 2 * D, bb(dCg), D, R, st);
    linear(w.cu, bb(dCat2), 2 * D, bb(dCu), D, R, st);
    k::hy4_swiglu(bb(dCg), bb(dCu), D, bb(dCh), R, D, st);
    linear(w.cd, bb(dCh), D, bb(dHn), D, R, st);
    k::hy4_add(bb(dHid), bb(dHn), bb(dCorr), size_t(R) * D, st);
    linear_f32(head.head, bb(dCorr), D, static_cast<float*>(buf(dLogits)), V, R, st);
    k::hy4_argmax_conf(static_cast<float*>(buf(dLogits)), R, V, tokens + t * R,
                       static_cast<float*>(buf(dConf)) + t * R, static_cast<float*>(buf(dArgmaxWs)), st);
    prev = tokens + t * R;
  }
  if (conf)
    UH_CHECK(hipMemcpyAsync(conf, buf(dConf), size_t(kHy4DraftTokens) * R * 4, hipMemcpyDeviceToDevice, st));

  // Debug (tools/ref/hy4_dfly_check.py): UNHOLY_HY4_DFLY_DUMP=DIR writes request 0's inputs and outputs of the
  // UNHOLY_HY4_DFLY_DUMP_AT-th call (default 20): the target hidden row, each layer's history positions and their KV
  // rows, the final-normed block, the draft tokens and probabilities.
  static const char* dump = std::getenv("UNHOLY_HY4_DFLY_DUMP");
  static int calls = 0;
  if (!dump || ++calls != (std::getenv("UNHOLY_HY4_DFLY_DUMP_AT") ? std::atoi(std::getenv("UNHOLY_HY4_DFLY_DUMP_AT")) : 20))
    return;
  UH_CHECK(hipStreamSynchronize(st));
  const auto d2h = [](void* dst, const void* src, size_t bytes) { UH_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost)); };
  const auto put = [&](const std::string& name, const void* p, size_t bytes) {
    FILE* f = std::fopen((std::string(dump) + "/" + name).c_str(), "wb");
    if (!f) throw std::runtime_error("dfly dump: cannot write " + name);
    std::fwrite(p, 1, bytes, f);
    std::fclose(f);
  };
  std::vector<uint16_t> hrow(static_cast<size_t>(D)), xb(static_cast<size_t>(kHy4DraftRows) * D);
  d2h(hrow.data(), buf(dHrow), size_t(D) * 2);
  d2h(xb.data(), buf(dX), xb.size() * 2);
  put("hidden.bin", hrow.data(), hrow.size() * 2);
  put("x.bin", xb.data(), xb.size() * 2);
  std::vector<int32_t> toks(static_cast<size_t>(kHy4DraftTokens) * R), table(static_cast<size_t>(max_pages));
  std::vector<float> cf(static_cast<size_t>(kHy4DraftTokens) * R);
  d2h(toks.data(), tokens, toks.size() * 4);
  d2h(cf.data(), buf(dConf), cf.size() * 4);
  d2h(table.data(), tables + size_t(reqs[0].slot) * max_pages, table.size() * 4);
  std::string meta = "{\"anchor\": " + std::to_string(reqs[0].anchor) + ", \"bonus\": " + std::to_string(reqs[0].bonus) +
                     ", \"tokens\": [";
  for (int t = 0; t < kHy4DraftTokens; ++t) meta += (t ? ", " : "") + std::to_string(toks[size_t(t) * R]);
  meta += "], \"conf\": [";
  for (int t = 0; t < kHy4DraftTokens; ++t) meta += (t ? ", " : "") + std::to_string(cf[size_t(t) * R]);
  meta += "], \"layers\": [";
  for (int L = 0; L < nL; ++L) {
    int32_t cnt = 0;
    d2h(&cnt, reinterpret_cast<const void*>(cntp[size_t(L) * R]), 4);
    std::vector<int32_t> hist(static_cast<size_t>(cnt));
    d2h(hist.data(), reinterpret_cast<const void*>(histp[size_t(L) * R]), size_t(cnt) * 4);
    std::vector<uint16_t> rows(static_cast<size_t>(cnt) * RW);
    for (int i = 0; i < cnt; ++i) {
      const int p = hist[size_t(i)];
      const size_t phys = size_t(table[size_t(p / k::kHy4PageRows)]) * k::kHy4PageRows + p % k::kHy4PageRows;
      d2h(rows.data() + size_t(i) * RW, target[size_t(L)]->rows.as<uint16_t>() + phys * RW, size_t(RW) * 2);
    }
    put("hist_" + std::to_string(L) + ".bin", hist.data(), hist.size() * 4);
    put("kv_" + std::to_string(L) + ".bin", rows.data(), rows.size() * 2);
    meta += (L ? ", " : "") + std::to_string(w.target[size_t(L)]);
  }
  meta += "]}\n";
  put("meta.json", meta.data(), meta.size());
  std::fprintf(stderr, "dfly: dumped call %d to %s\n", calls, dump);
}

}  // namespace unholy::model
