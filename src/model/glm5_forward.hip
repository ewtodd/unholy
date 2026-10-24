// GLM-5.3-Flash reference forward (model/glm5.hpp): Glm5NextTextDecoderLayer over a batch of sequence chunks.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernels/ref.hpp"
#include "kernels/ref_glm.hpp"
#include "model/glm5.hpp"

namespace unholy::model {

namespace k = unholy::kernels;
using bf16 = uint16_t;

namespace {

std::vector<std::vector<uint64_t>>* g_hist = nullptr;
bool g_a8 = false;

template <typename T>
T* slot_rows(const rt::DeviceBuffer& b, int slot, size_t per_slot) {
  return b.as<T>() + size_t(slot) * per_slot;
}

void kda_forward(const Glm5Config& c, const Glm5LayerWeights& w, Glm5Cache& cache, const Glm5Scratch& s,
                 const bf16* hn, bf16* out, const Glm5Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, H = c.lin_heads, Dh = c.lin_dim, W = c.lin_width(), C3 = 3 * W;
  bf16* qkv = s.at<bf16>(kQkv);
  bf16* qkvc = s.at<bf16>(kQkvConv);
  if (w.qkv) {
    glm5_linear(w.qkv, hn, D, qkv, C3, T, s, st);
  } else {
    glm5_linear(w.q, hn, D, qkv, C3, T, s, st);
    glm5_linear(w.k, hn, D, qkv + W, C3, T, s, st);
    glm5_linear(w.v, hn, D, qkv + 2 * W, C3, T, s, st);
  }
  const size_t conv_slot = size_t(c.conv_kernel - 1) * C3, state_slot = size_t(H) * Dh * Dh;
  for (const auto& ch : b.chunks)
    k::kda_conv(qkv + size_t(ch.t0) * C3, C3, slot_rows<bf16>(cache.conv, ch.slot, conv_slot), w.conv.p<bf16>(),
                qkvc + size_t(ch.t0) * C3, ch.n, C3, st);
  // Forget gate and beta.
  bf16* low = s.at<bf16>(kLowRank);
  bf16* row = s.at<bf16>(kGateRow);
  float* g = s.at<float>(kForget);
  glm5_linear(w.f_a, hn, D, low, Dh, T, s, st);
  glm5_linear(w.f_b, low, Dh, row, W, T, s, st);
  k::kda_gate(row, w.dt_bias.p<float>(), w.a_log.p<float>(), g, T, H, Dh, c.gate_lower_bound, st);
  bf16* beta = s.at<bf16>(kBeta);
  glm5_linear(w.b, hn, D, beta, H, T, s, st);
  k::sigmoid_bf16(beta, beta, size_t(T) * H, st);
  // The delta rule, per sequence.
  bf16* core = s.at<bf16>(kCore);
  for (const auto& ch : b.chunks) {
    const bf16* base = qkvc + size_t(ch.t0) * C3;
    k::kda_recurrent(base, base + W, base + 2 * W, C3, g + size_t(ch.t0) * W, beta + size_t(ch.t0) * H,
                     slot_rows<float>(cache.state, ch.slot, state_slot), core + size_t(ch.t0) * W, ch.n, H, Dh, st);
  }
  // Gated output norm, output projection.
  glm5_linear(w.g_a, hn, D, low, Dh, T, s, st);
  glm5_linear(w.g_b, low, Dh, row, W, T, s, st);
  bf16* on = s.at<bf16>(kOutNorm);
  k::kda_out_norm(core, row, w.o_norm.p<bf16>(), on, T, H, Dh, c.norm_eps, st);
  glm5_linear(w.o, on, W, out, D, T, s, st);
}

void dsa_forward(const Glm5Config& c, const Glm5LayerWeights& w, Glm5Cache& cache, const Glm5Scratch& s,
                 const bf16* hn, bf16* out, const Glm5Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, H = c.n_heads, R = c.kv_lora;
  const int IH = c.idx_heads, ID = c.idx_dim, pool = c.idx_pool;
  const int kpools = c.idx_topk / pool, width = c.idx_topk + pool - 1;
  // Queries and the latent row.
  bf16* qr = s.at<bf16>(kQResid);
  glm5_linear(w.q_a, hn, D, qr, c.q_lora, T, s, st);
  k::rmsnorm_hf(qr, c.q_lora, w.q_a_norm.p<bf16>(), qr, c.q_lora, T, c.q_lora, c.norm_eps, st);
  bf16* q = s.at<bf16>(kQ);
  glm5_linear(w.q_b, qr, c.q_lora, q, H * c.nope_dim, T, s, st);
  bf16* lat = s.at<bf16>(kLatent);
  glm5_linear(w.kv_a, hn, D, lat, R, T, s, st);
  k::rmsnorm_hf(lat, R, w.kv_a_norm.p<bf16>(), lat, R, T, R, c.norm_eps, st);
  // Indexer rows.
  bf16* iq = s.at<bf16>(kIdxQ);
  bf16* ik = s.at<bf16>(kIdxK);
  bf16* ig = s.at<bf16>(kIdxGate);
  bf16* iw = s.at<bf16>(kIdxW);
  glm5_linear(w.idx_wq_b, qr, c.q_lora, iq, IH * ID, T, s, st);
  glm5_linear(w.idx_wk, hn, D, ik, ID, T, s, st);
  k::layernorm(ik, w.idx_knorm_w.ptr, w.idx_knorm_b.ptr, w.idx_f16, ik, T, ID, 1e-6f, st);
  glm5_linear(w.idx_gate, hn, D, ig, ID, T, s, st);
  glm5_linear(w.idx_wproj, hn, D, iw, IH, T, s, st);
  // Absorbed queries.
  float* qa = s.at<float>(kQAbs);
  float* ol = s.at<float>(kLatentOut);
  k::mla_absorb_q(q, w.kv_b.ref(), qa, T, H, c.nope_dim, c.v_dim, R, st);

  float* isc = s.at<float>(kIdxScores);
  int32_t* pools = s.at<int32_t>(kIdxPools);
  int32_t* sel = s.at<int32_t>(kIdxSel);
  const size_t kv_slot = size_t(cache.max_seq) * R, row_slot = size_t(cache.max_seq) * ID;
  const size_t pk_slot = size_t(cache.max_seq / pool + 1) * ID;
  for (const auto& ch : b.chunks) {
    if (ch.s0 + ch.n > cache.max_seq) throw std::runtime_error("glm5: sequence longer than the cache");
    bf16* kv = slot_rows<bf16>(cache.kv, ch.slot, kv_slot);
    bf16* ck = slot_rows<bf16>(cache.ik, ch.slot, row_slot);
    bf16* cg = slot_rows<bf16>(cache.ig, ch.slot, row_slot);
    bf16* pk = slot_rows<bf16>(cache.pk, ch.slot, pk_slot);
    UH_CHECK(hipMemcpyAsync(kv + size_t(ch.s0) * R, lat + size_t(ch.t0) * R, size_t(ch.n) * R * 2,
                            hipMemcpyDeviceToDevice, st));
    UH_CHECK(hipMemcpyAsync(ck + size_t(ch.s0) * ID, ik + size_t(ch.t0) * ID, size_t(ch.n) * ID * 2,
                            hipMemcpyDeviceToDevice, st));
    UH_CHECK(hipMemcpyAsync(cg + size_t(ch.s0) * ID, ig + size_t(ch.t0) * ID, size_t(ch.n) * ID * 2,
                            hipMemcpyDeviceToDevice, st));
    // Pools completed by this chunk, then the selection: the top pools among those complete by each query.
    const int p0 = ch.s0 / pool, p1 = (ch.s0 + ch.n) / pool;
    k::idx_pool(ck, cg, w.idx_ape.p<float>(), pk, p0, p1 - p0, pool, ID, st);
    int32_t* cpools = pools + size_t(ch.t0) * kpools;
    if (p1 == 0) {
      UH_CHECK(hipMemsetAsync(cpools, 0xff, size_t(ch.n) * kpools * 4, st));
    } else {
      for (int q0 = 0; q0 < ch.n; q0 += kIdxBlock) {
        const int m = std::min(kIdxBlock, ch.n - q0), t = ch.t0 + q0;
        k::idx_scores(iq + size_t(t) * IH * ID, iw + size_t(t) * IH, pk, b.pos + t, isc, m, p1, IH, ID, pool,
                      1.0f / std::sqrt(float(ID)), st);
        k::topk_rows(isc, m, p1, kpools, cpools + size_t(q0) * kpools, st);
      }
    }
    int32_t* csel = sel + size_t(ch.t0) * width;
    k::idx_expand(cpools, b.pos + ch.t0, csel, ch.n, kpools, pool, st);
    k::mla_attn(qa + size_t(ch.t0) * H * R, kv, csel, width, b.pos + ch.t0, ol + size_t(ch.t0) * H * R, ch.n, H, R,
                1.0f / std::sqrt(float(c.nope_dim)), st);
  }
  bf16* o = s.at<bf16>(kAttnOut);
  k::mla_out_v(ol, w.kv_b.ref(), o, T, H, c.nope_dim, c.v_dim, R, st);
  glm5_linear(w.o, o, H * c.v_dim, out, D, T, s, st);
}

void mlp_forward(const Glm5Config& c, const GLinear& gate, const GLinear& up, const GLinear& down, const bf16* x,
                 int rows, bf16* out, const Glm5Scratch& s, hipStream_t st) {
  const int I = gate.out;
  bf16* g = s.at<bf16>(kExpGate);
  bf16* u = s.at<bf16>(kExpUp);
  bf16* h = s.at<bf16>(kExpAct);
  glm5_linear(gate, x, c.dim, g, I, rows, s, st);
  glm5_linear(up, x, c.dim, u, I, rows, s, st);
  k::swiglu_hf(g, u, h, rows, I, c.swiglu_limit, st);
  glm5_linear(down, h, I, out, c.dim, rows, s, st);
}

void moe_forward(const Glm5Config& c, const Glm5LayerWeights& w, const Glm5Scratch& s, const bf16* hn, bf16* out,
                 int T, hipStream_t st) {
  const int D = c.dim, E = c.n_experts, K = c.n_active, I = c.moe_inter;
  float* hf = s.at<float>(kF32Hn);
  float* logits = s.at<float>(kF32Rows);
  int32_t* sel = s.at<int32_t>(kSel);
  float* rw = s.at<float>(kRouteW);
  k::cast_bf16_f32(hn, hf, size_t(T) * D, st);
  k::gemm_w_f32(hf, D, w.router.ref(), logits, E, T, E, D, st);
  k::route_sigmoid(logits, w.router_bias.p<float>(), sel, rw, T, E, K, c.route_scale, st);
  std::vector<int32_t> hsel(size_t(T) * K);
  std::vector<float> hrw(size_t(T) * K);
  UH_CHECK(hipMemcpyAsync(hsel.data(), sel, hsel.size() * 4, hipMemcpyDeviceToHost, st));
  UH_CHECK(hipMemcpyAsync(hrw.data(), rw, hrw.size() * 4, hipMemcpyDeviceToHost, st));
  UH_CHECK(hipStreamSynchronize(st));
  // Pairs expert-major (ascending expert, then token); each token's pairs in ascending expert order.
  std::vector<int> count(size_t(E), 0), first(size_t(E) + 1, 0);
  for (int32_t e : hsel) {
    if (e < 0 || e >= E) throw std::runtime_error("glm5: router selected an invalid expert");
    ++count[size_t(e)];
  }
  for (int e = 0; e < E; ++e) first[size_t(e) + 1] = first[size_t(e)] + count[size_t(e)];
  const int P = T * K;
  std::vector<int32_t> tok(static_cast<size_t>(P)), pidx(static_cast<size_t>(P));
  std::vector<float> pw(static_cast<size_t>(P));
  std::vector<int> fill(first.begin(), first.end() - 1);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < K; ++j) {
      const int e = hsel[size_t(t) * K + j], p = fill[size_t(e)]++;
      tok[size_t(p)] = t;
      pw[size_t(p)] = hrw[size_t(t) * K + j];
      pidx[size_t(t) * K + j] = p;
    }
  if (g_hist) {
    auto& h = (*g_hist)[size_t(w.layer)];
    if (h.size() < size_t(E)) h.resize(size_t(E), 0);
    for (int e = 0; e < E; ++e) h[size_t(e)] += uint64_t(count[size_t(e)]);
  }
  int32_t* dtok = s.at<int32_t>(kPairTok);
  float* dpw = s.at<float>(kPairW);
  int32_t* dpidx = s.at<int32_t>(kPairIdx);
  UH_CHECK(hipMemcpyAsync(dtok, tok.data(), size_t(P) * 4, hipMemcpyHostToDevice, st));
  UH_CHECK(hipMemcpyAsync(dpw, pw.data(), size_t(P) * 4, hipMemcpyHostToDevice, st));
  UH_CHECK(hipMemcpyAsync(dpidx, pidx.data(), size_t(P) * 4, hipMemcpyHostToDevice, st));
  bf16* xin = s.at<bf16>(kExpIn);
  bf16* y = s.at<bf16>(kExpOut);
  k::gather_rows(hn, dtok, xin, P, D, st);
  bf16* g = s.at<bf16>(kExpGate);
  bf16* u = s.at<bf16>(kExpUp);
  bf16* h = s.at<bf16>(kExpAct);
  for (int e = 0; e < E; ++e) {
    const int r0 = first[size_t(e)], n = count[size_t(e)];
    if (n == 0) continue;
    if (!w.e_gate.at(size_t(e))) throw std::runtime_error("glm5: routed expert not loaded");
    glm5_linear(w.e_gate[size_t(e)], xin + size_t(r0) * D, D, g + size_t(r0) * I, I, n, s, st);
    glm5_linear(w.e_up[size_t(e)], xin + size_t(r0) * D, D, u + size_t(r0) * I, I, n, s, st);
    k::swiglu_hf(g + size_t(r0) * I, u + size_t(r0) * I, h + size_t(r0) * I, n, I, c.swiglu_limit, st);
    glm5_linear(w.e_down[size_t(e)], h + size_t(r0) * I, I, y + size_t(r0) * D, D, n, s, st);
  }
  bf16* sh = s.at<bf16>(kShared);
  mlp_forward(c, w.sh_gate, w.sh_up, w.sh_down, hn, T, sh, s, st);
  k::moe_combine_hf(y, dpw, dpidx, sh, out, T, K, D, st);
  UH_CHECK(hipStreamSynchronize(st));  // the host vectors above back in-flight copies
}

}  // namespace

void set_glm5_expert_histogram(std::vector<std::vector<uint64_t>>* hist) { g_hist = hist; }
void set_glm5_a8(bool on) { g_a8 = on; }

void glm5_linear(const GLinear& l, const uint16_t* x, int ldx, uint16_t* out, int ldo, int rows,
                 const Glm5Scratch& s, hipStream_t st) {
  if (rows <= 0) return;
  if (!l) throw std::runtime_error("glm5_linear: weight not loaded");
  if (!l.exl3()) {
    const auto r = l.ref();
    if (g_a8 && r.fmt == k::WFmt::Fp8Block) {
      if (size_t(rows) * l.in * 2 > s.bufs[kExl3X].size())
        throw std::runtime_error("glm5_linear: a8 scratch too small");
      bf16* xq = s.at<bf16>(kExl3X);
      UH_CHECK(hipMemcpy2DAsync(xq, size_t(l.in) * 2, x, size_t(ldx) * 2, size_t(l.in) * 2, rows,
                                hipMemcpyDeviceToDevice, st));
      k::fake_quant_fp8_group(xq, l.in, rows, l.in, 128, st);
      k::gemm_w(xq, l.in, r, out, ldo, rows, l.out, l.in, st);
      return;
    }
    k::gemm_w(x, ldx, r, out, ldo, rows, l.out, l.in, st);
    return;
  }
  const size_t xcap = s.bufs[kExl3X].size() / 4, ycap = s.bufs[kExl3Y].size() / 4;
  const int step = int(std::min(xcap / size_t(l.in), ycap / size_t(l.out)));
  if (step <= 0) throw std::runtime_error("glm5_linear: EXL3 scratch too small");
  float* xh = s.at<float>(kExl3X);
  float* yh = s.at<float>(kExl3Y);
  for (int r0 = 0; r0 < rows; r0 += step) {
    const int n = std::min(step, rows - r0);
    k::exl3_in(x + size_t(r0) * ldx, ldx, l.x.suh.p<uint16_t>(), xh, n, l.in, st);
    k::gemm_exl3_inner(xh, l.x.trellis.p<uint16_t>(), l.x.bits, yh, n, l.out, l.in, st);
    k::exl3_out(yh, l.x.svh.p<uint16_t>(), out + size_t(r0) * ldo, ldo, n, l.out, st);
  }
}

void glm5_layer_forward(const Glm5Config& c, const Glm5LayerWeights& w, Glm5Cache& cache, const Glm5Scratch& s,
                        uint16_t* x, const Glm5Batch& b, hipStream_t st) {
  const int T = b.T, D = c.dim, hc = c.hc_mult;
  if (T > s.max_tokens) throw std::runtime_error("glm5_layer_forward: batch larger than the scratch");
  bf16* h = s.at<bf16>(kH);
  bf16* hn = s.at<bf16>(kHn);
  bf16* out = s.at<bf16>(kOut);
  float* post = s.at<float>(kPost);
  float* comb = s.at<float>(kComb);

  k::hc_pre(x, w.hc_attn_fn.p<float>(), w.hc_attn_scale.p<float>(), w.hc_attn_base.p<float>(), h, post, comb, T, hc,
            D, c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  k::rmsnorm_hf(h, D, w.input_norm.p<bf16>(), hn, D, T, D, c.norm_eps, st);
  if (w.kda) kda_forward(c, w, cache, s, hn, out, b, st);
  else dsa_forward(c, w, cache, s, hn, out, b, st);
  k::hc_post_hf(out, x, post, comb, x, T, hc, D, st);

  k::hc_pre(x, w.hc_ffn_fn.p<float>(), w.hc_ffn_scale.p<float>(), w.hc_ffn_base.p<float>(), h, post, comb, T, hc, D,
            c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  k::rmsnorm_hf(h, D, w.post_norm.p<bf16>(), hn, D, T, D, c.norm_eps, st);
  if (w.dense) mlp_forward(c, w.gate, w.up, w.down, hn, T, out, s, st);
  else moe_forward(c, w, s, hn, out, T, st);
  k::hc_post_hf(out, x, post, comb, x, T, hc, D, st);
}

void glm5_head_forward(const Glm5Config& c, const Glm5HeadWeights& hw, const Glm5Scratch& s, const uint16_t* x,
                       int T, float* logits, hipStream_t st) {
  const int D = c.dim, V = c.vocab, G = s.head_rows;
  bf16* h = s.at<bf16>(kH);
  bf16* lb = s.at<bf16>(kHeadOut);
  for (int r0 = 0; r0 < T; r0 += G) {
    const int n = std::min(G, T - r0);
    k::hc_mean(x + size_t(r0) * c.hc_mult * D, h, n, c.hc_mult, D, st);
    k::rmsnorm_hf(h, D, hw.norm.p<bf16>(), h, D, n, D, c.norm_eps, st);
    glm5_linear(hw.head, h, D, lb, V, n, s, st);
    k::cast_bf16_f32(lb, logits + size_t(r0) * V, size_t(n) * V, st);
  }
}

}  // namespace unholy::model
