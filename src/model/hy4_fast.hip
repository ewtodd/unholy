// Hy4 flash layer forward (model/hy4.hpp): transformers' HYV4FlashDecoderLayer on the fast kernels.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "kernels/elem_fast.hpp"
#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/glm_fast.hpp"
#include "kernels/hy4.hpp"
#include "kernels/moe_dispatch.hpp"
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

void linear(const Hy4Linear& l, const bf16* x, int ldx, bf16* out, int ldo, int rows, hipStream_t st) {
  if (rows <= 0) return;
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
  bf16* qr = s.at<bf16>(kQr);
  linear(w.q_a, hn, D, qr, c.q_lora, T, st);
  norm(qr, c.q_lora, w.q_a_norm, T, c.q_lora, c.eps, st);
  bf16* q = s.at<bf16>(kQ);
  linear(w.q_b, qr, c.q_lora, q, H * 256, T, st);
  k::hy4_rope(q, H * 256, T, H, 256, c.nope, b.pos, theta, st);
  bf16* kv = s.at<bf16>(kKv);
  linear(w.kv_a, hn, D, kv, R, T, st);
  norm(kv, R, w.kv_a_norm, T, c.kv_lora, c.eps, st);
  k::hy4_rope(kv, R, T, 1, R, c.kv_lora, b.pos, theta, st);
  bf16* g = s.at<bf16>(kG);
  linear(w.gate, hn, D, g, H * 256, T, st);
  bf16* qn = s.at<bf16>(kQn);
  bf16* qa = s.at<bf16>(kQa);
  bf16* qp = s.at<bf16>(kQp);
  k::hy4_split_q(q, qn, T, H, st);
  k::gemm_grouped_bf16_wmma(qn, w.w_ukT.p<bf16>(), qa, T, H, c.kv_lora, c.nope, st);
  k::hy4_pack_q(qa, q, qp, T, H, st);
  bf16* ol = s.at<bf16>(kOl);
  const float scale = 1.0f / std::sqrt(float(c.nope + c.rope));

  if (w.sparse) {
    if (!b.tables) throw std::runtime_error("hy4: sparse layers need block tables");
    const int IH = c.idx_heads, ID = c.idx_dim, kg = c.idx_groups(), width = c.idx_width();
    bf16* tk = s.at<bf16>(kTk);
    linear(w.idx_k, hn, D, tk, ID, T, st);
    norm(tk, ID, w.idx_tok_norm, T, ID, c.eps, st);
    bf16* iq = s.at<bf16>(kIq);
    linear(w.idx_q, qr, c.q_lora, iq, IH * ID, T, st);
    k::hy4_rope(iq, IH * ID, T, IH, ID, ID - c.rope, b.pos, c.theta_sparse, st);
    bf16* iw = s.at<bf16>(kIw);
    linear(w.idx_w, hn, D, iw, IH, T, st);
    const float wscale = 1.0f / std::sqrt(float(IH)) / std::sqrt(float(ID));
    float* isc = s.at<float>(kIScores);
    int32_t* groups = s.at<int32_t>(kIGroups);
    int32_t* sel = s.at<int32_t>(kISel);
    for (const auto& ch : b.chunks) {
      if ((ch.s0 + ch.n + k::kHy4PageRows - 1) / k::kHy4PageRows > b.max_pages)
        throw std::runtime_error("hy4: sequence longer than the block table");
      const int32_t* table = b.tables + size_t(ch.slot) * b.max_pages;
      bf16* rows = cache.rows.as<bf16>();
      k::hy4_page_store(kv + size_t(ch.t0) * R, rows, table, ch.s0, ch.n, st);
      const k::Hy4Pooled pg{cache.pooled.as<bf16>(), table};
      k::hy4_idx_pool(tk + size_t(ch.t0) * ID, cache.tail.as<bf16>() + size_t(ch.slot) * c.idx_pool * ID,
                      w.idx_wv.p<float>(), w.idx_wg.p<float>(), w.idx_ape.p<float>(), w.idx_knorm.p<float>(), c.eps,
                      c.theta_sparse, pg, ch.s0, ch.n, st);
      const k::Hy4Keys keys{rows, table, nullptr, nullptr, 0};
      const int last = ch.s0 + ch.n - 1;
      if (last / c.idx_pool <= kg) {  // every query recalls all of its groups: dense causal attention
        k::hy4_mla_attn(qp + size_t(ch.t0) * H * R, H, keys, nullptr, 0, 0, b.pos + ch.t0, w.sinks.p<float>(),
                        ol + size_t(ch.t0) * H * c.kv_lora, ch.n, scale, st);
        continue;
      }
      const int npool = (ch.s0 + ch.n) / c.idx_pool;
      for (int q0 = 0; q0 < ch.n; q0 += kHy4IdxBlock) {
        const int m = std::min(kHy4IdxBlock, ch.n - q0), t = ch.t0 + q0;
        k::hy4_idx_scores(iq + size_t(t) * IH * ID, iw + size_t(t) * IH, wscale, pg, b.pos + t, isc, npool, m, npool,
                          st);
        k::glm_topk(isc, npool, m, npool, kg, groups + size_t(t) * kg, s.get(kITopk), st);
      }
      k::hy4_idx_expand(groups + size_t(ch.t0) * kg, b.pos + ch.t0, sel + size_t(ch.t0) * width, ch.n, kg, st);
      k::hy4_mla_attn(qp + size_t(ch.t0) * H * R, H, keys, sel + size_t(ch.t0) * width, width, 0, b.pos + ch.t0,
                      w.sinks.p<float>(), ol + size_t(ch.t0) * H * c.kv_lora, ch.n, scale, st);
    }
  } else {
    for (const auto& ch : b.chunks) {
      bf16* ring = cache.ring.as<bf16>() + size_t(ch.slot) * k::kHy4Window * R;
      const k::Hy4Keys keys{nullptr, nullptr, ring, kv + size_t(ch.t0) * R, ch.s0};
      k::hy4_mla_attn(qp + size_t(ch.t0) * H * R, H, keys, nullptr, 0, c.window, b.pos + ch.t0, w.sinks.p<float>(),
                      ol + size_t(ch.t0) * H * c.kv_lora, ch.n, scale, st);
      k::hy4_ring_store(kv + size_t(ch.t0) * R, ring, ch.s0, ch.n, st);
    }
  }
  bf16* o = s.at<bf16>(kO);
  k::gemm_grouped_bf16_wmma(ol, w.w_uv.p<bf16>(), o, T, H, c.v_dim, c.kv_lora, st);
  k::hy4_gate_mul(o, g, size_t(T) * H * c.v_dim, st);
  linear(w.o, o, H * c.v_dim, out, D, T, st);
  norm(out, D, w.post_proj_norm, T, D, c.eps, st);
}

// ---- MLP / MoE ----------------------------------------------------------------------------------------------------

void dense_mlp(const Hy4Config& c, const Hy4LayerWeights& w, const Hy4Scratch& s, const bf16* hn, bf16* out, int T,
               hipStream_t st) {
  const int D = c.dim, I = w.gate_p.out;
  bf16* g = s.at<bf16>(kMlpG);
  bf16* u = s.at<bf16>(kMlpU);
  bf16* h = s.at<bf16>(kMlpH);
  linear(w.gate_p, hn, D, g, I, T, st);
  linear(w.up_p, hn, D, u, I, T, st);
  k::hy4_swiglu(g, u, I, h, T, I, st);
  linear(w.down_p, h, I, out, D, T, st);
  norm(out, D, w.post_mlp_norm, T, D, c.eps, st);
}

int cap_tiles(const Hy4Config& c, int T) { return T * c.n_active / k::kHy4TileRows + c.n_experts + 1; }

void moe(const Hy4Config& c, const Hy4LayerWeights& w, const Hy4Scratch& s, const bf16* hn, bf16* out, int T,
         hipStream_t st, RemoteExperts* remote) {
  const int D = c.dim, E = c.n_experts, K = c.n_active, P = T * K, Lt = c.latent, I = c.moe_inter;
  const bool any_remote = w.n_resident < E;
  if (any_remote && (!remote || !remote->can_enqueue())) throw std::runtime_error("hy4: remote experts need a backend");
  float* logits = s.at<float>(kLogits);
  int32_t* sel = s.at<int32_t>(kSel);
  float* rw = s.at<float>(kRw);
  linear_f32(w.router, hn, D, logits, E, T, st);
  k::glm_route(logits, w.router_bias.p<float>(), sel, rw, T, E, K, c.route_scale, st);
  bf16* lat = s.at<bf16>(kLat);
  linear(w.pre_expert, hn, D, lat, Lt, T, st);

  auto* plan = s.at<k::MoePlan>(kMoePlan);
  auto* lt = s.at<k::Exl3Tile>(kMoeLTiles);
  auto* rtl = s.at<k::Exl3Tile>(kMoeRTiles);
  int32_t* eidx = s.at<int32_t>(kEIdx);
  float* ewt = s.at<float>(kEWt);
  int32_t* row_expert = s.at<int32_t>(kRowExpert);
  int32_t* pidx = s.at<int32_t>(kPidx);
  int32_t* tok2u = s.at<int32_t>(kTok2u);
  int32_t* u2t = tok2u + T;
  uint32_t* hist = g_hist ? s.at<uint32_t>(kRouteHist) + size_t(w.layer) * E : nullptr;
  const int ct = cap_tiles(c, T);
  if (any_remote && (size_t(P) > remote->capacity_rows || ct > remote->max_tiles))
    throw std::runtime_error("hy4: remote staging too small");
  k::moe_count(sel, T, K, s.at<int32_t>(kMoeCount), E, st, hist);
  k::moe_plan(s.at<int32_t>(kMoeCount), w.resident.as<uint8_t>(), E, P, plan, lt, rtl, s.at<int32_t>(kMoeOff),
              s.at<int32_t>(kMoeFill), ct, st, k::kHy4TileRows);
  k::moe_scatter(sel, rw, T, K, s.at<int32_t>(kMoeOff), s.at<int32_t>(kMoeFill), eidx, ewt, row_expert, pidx, st);
  if (any_remote) {  // the remote tokens' latent rows to the APU, which starts once the card stream gets here
    k::remote_tokens(pidx, eidx, T, K, plan, tok2u, u2t, remote->p2u_host, remote->u2p_host, st);
    k::stage_remote(plan, rtl, ewt, row_expert, remote->plan_host, remote->tiles_host, remote->w_host, remote->e_host,
                    P, st);
    k::copy_remote_tokens(lat, Lt, u2t, plan, remote->in_host, T, st);
    remote->enqueue(w.layer, st, P);
  }
  bf16* pin = s.at<bf16>(kExpIn);
  bf16* pout = s.at<bf16>(kPout);
  k::gather_rows(lat, eidx, pin, P, Lt, st);
  if (w.n_resident > 0)
    hy4_experts_dev(c, w, pin, ewt, lt, std::min(P, ct), &plan->n_local_tiles, P, s.at<bf16>(kGu), s.at<bf16>(kEh),
                    pout, st, nullptr, &plan->remote0);
  // The shared expert meanwhile.
  bf16* sg = s.at<bf16>(kShG);
  bf16* su = s.at<bf16>(kShU);
  bf16* sh = s.at<bf16>(kShH);
  bf16* shared = s.at<bf16>(kShared);
  linear(w.sh_gate, hn, D, sg, I, T, st);
  linear(w.sh_up, hn, D, su, I, T, st);
  k::hy4_swiglu(sg, su, I, sh, T, I, st);
  linear(w.sh_down, sh, I, shared, D, T, st);
  bf16* msum = s.at<bf16>(kMsum);
  if (any_remote) {
    remote->join(st);
    k::hy4_moe_sum(pout, pidx, &plan->remote0, tok2u, reinterpret_cast<const float*>(remote->out_host), msum, T, K, Lt,
                   st);
  } else {
    k::hy4_moe_sum(pout, pidx, nullptr, nullptr, nullptr, msum, T, K, Lt, st);
  }
  bf16* routed = s.at<bf16>(kRouted);
  linear(w.post_expert, msum, Lt, routed, D, T, st);
  k::hy4_add(routed, shared, out, size_t(T) * D, st);
  norm(out, D, w.combine_norm, T, D, c.eps, st);
}

void engram(const Hy4Config& c, const Hy4LayerWeights& w, const Hy4Scratch& s, uint16_t* x, const Hy4Batch& b,
            hipStream_t st) {
  const int T = b.T, D = c.dim, m = w.engram;
  if (!b.eng_q[m] || !b.eng_s[m] || !b.eng_mask) throw std::runtime_error("hy4: Engram rows missing from the batch");
  bf16* rows = s.at<bf16>(kEngRows);
  float* kv = s.at<float>(kEngKv);
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
  bf16* h = s.at<bf16>(kH);
  bf16* hn = s.at<bf16>(kHn);
  bf16* out = s.at<bf16>(kOut);
  float* post = s.at<float>(kPost);
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
                      float* logits, hipStream_t st) {
  const int D = c.dim, V = c.vocab, G = std::min(s.max_tokens, 64);
  bf16* h = s.at<bf16>(kH);
  for (int r0 = 0; r0 < T; r0 += G) {
    const int n = std::min(G, T - r0);
    k::hy4_hc_head(x + size_t(r0) * c.hc * D, hw.hc_fn.p<float>(), hw.hc_scale.p<float>(), hw.hc_base.p<float>(), h, n,
                   c.hc, D, c.eps, c.hc_eps, st);
    norm(h, D, hw.norm, n, D, c.eps, st);
    linear_f32(hw.head, h, D, logits + size_t(r0) * V, V, n, st);
  }
}

}  // namespace unholy::model
