// Reference kernels for GLM-5.3-Flash (transformers' glm5_next, the oracle in tools/ref/glm_oracle): KDA
// (Kimi delta attention), the pooled DSA indexer, absorbed NoPE MLA, the sigmoid router, and the bf16
// rounding variants of the norms / SwiGLU / Hyper-Connection post-mix that transformers' code performs.
//
// Rounding: wherever transformers holds a bf16 tensor between ops, these kernels round to bf16 at the same
// point; inside KDA (fp32 in transformers too) and inside attention (bf16 matmuls with an fp32 softmax in
// transformers' eager path) they stay in fp32, the precision the fast kernels will use. Weights shared with
// DeepSeek-V4 (embedding, Hyper-Connection pre-mix, top-k) use the kernels in kernels/ref.hpp.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

// A linear weight [N, K] as the GLM checkpoints store it: bf16, fp16 (exllamav3's small projections), or
// e4m3 with fp32 128x128 block scales (weight_scale_inv [ceil(N/128), ceil(K/128)]), which transformers
// dequantizes to bf16(w * s) before its bf16 GEMMs.
enum class WFmt : uint8_t { Bf16, F16, Fp8Block };
struct WRef {
  const void* p = nullptr;
  WFmt fmt = WFmt::Bf16;
  const float* scale = nullptr;  // Fp8Block only
};

// ---- GEMMs (fp32 accumulation, kernels/ref_gemm.hip) ------------------------------------------------
// c [M, N] bf16 = a [M, K] bf16 * W^T.
void gemm_w(const bf16* a, int lda, WRef w, bf16* c, int ldc, int M, int N, int K, hipStream_t st);
// c [M, N] f32 = a [M, K] f32 * W^T (the router's fp32 linear).
void gemm_w_f32(const float* a, int lda, WRef w, float* c, int ldc, int M, int N, int K, hipStream_t st);
// Absorbed MLA with kv_b [H * (dn + dv), r] (per head: dn key rows, then dv value rows):
//   qa [T, H, r] f32 = q [T, H, dn] . W_uk(h)     (qa[t][h][c] = sum_d q[t][h][d] kv_b[h (dn+dv) + d][c])
//   o  [T, H, dv] bf16 = ol [T, H, r] . W_uv(h)^T (o[t][h][d] = sum_c ol[t][h][c] kv_b[h (dn+dv) + dn + d][c])
void mla_absorb_q(const bf16* q, WRef kv_b, float* qa, int T, int H, int dn, int dv, int r, hipStream_t st);
void mla_out_v(const float* ol, WRef kv_b, bf16* o, int T, int H, int dn, int dv, int r, hipStream_t st);

// ---- norms / elementwise -----------------------------------------------------------------------------
// Glm5NextTextRMSNorm: y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))). y may alias x.
void rmsnorm_hf(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps,
                hipStream_t st);
// nn.LayerNorm on bf16 rows (fp32 statistics, one rounding): y = bf16((x - mean) * rstd * w + b).
// w / b are bf16, or fp16 when f16 is set (exllamav3 checkpoints).
void layernorm(const bf16* x, const void* w, const void* b, bool f16, bf16* y, int rows, int dim, float eps,
               hipStream_t st);
// Glm5NextTextDecoderLayer's post-mix in bf16 arithmetic:
//   out[j] = bf16(bf16(bf16(post[j]) * x) + bf16(sum_i bf16(comb[i][j]) * res[i])).  out may alias res.
void hc_post_hf(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc,
                int dim, hipStream_t st);
// Glm5NextTextHyperHead: y [T, dim] = bf16(mean_i x[t][i]).
void hc_mean(const bf16* x, bf16* y, int T, int hc, int dim, hipStream_t st);
// y = bf16(sigmoid(x)).
void sigmoid_bf16(const bf16* x, bf16* y, size_t n, hipStream_t st);
// y = float(bf16(x)): the logits transformers' bf16 lm_head returns.
void round_f32_bf16(float* x, size_t n, hipStream_t st);

// ---- KDA ---------------------------------------------------------------------------------------------
// Depthwise causal conv (kernel 4, no bias) + SiLU over the n rows of one chunk, x [n, C] bf16 (row stride
// ldx) -> y [n, C] bf16: y = bf16(silu(bf16(sum_i w[c][i] x[t - 3 + i][c]))), rows before the chunk from
// state [3, C] (the sequence's previous three inputs, zeros at its start), which is then updated.
void kda_conv(const bf16* x, int ldx, bf16* state, const bf16* w, bf16* y, int n, int C, hipStream_t st);
// Forget gate: g [T, H*D] f32 = lb * sigmoid(exp(A_log[h]) * (fb + dt_bias)), fb bf16 [T, H*D].
void kda_gate(const bf16* fb, const float* dt_bias, const float* a_log, float* g, int T, int H, int D, float lb,
              hipStream_t st);
// The delta rule over one chunk of n tokens, token by token, fp32 (recurrent_kimi_delta_attention):
//   q, k = l2norm(q), l2norm(k); q *= D^-0.5
//   S = S * exp(g); delta = (v - S^T k) * beta; S += k delta^T; o = S^T q
// q / k / v: bf16 rows [n] of H * D (row stride ld), g f32 [n, H * D], beta bf16 [n, H]; state f32
// [H][D][D] (k index major) updated in place; o bf16 [n, H * D].
void kda_recurrent(const bf16* q, const bf16* k, const bf16* v, int ld, const float* g, const bf16* beta,
                   float* state, bf16* o, int n, int H, int D, hipStream_t st);
// Glm5NextTextRMSNormGated per head: o = bf16(w * (x * rsqrt(mean(x^2) + eps)) * sigmoid(gate)), fp32 inside.
void kda_out_norm(const bf16* x, const bf16* gate, const bf16* w, bf16* o, int rows, int H, int D, float eps,
                  hipStream_t st);

// ---- DSA indexer (pooled keys) -----------------------------------------------------------------------
// Pooled keys of pools [p0, p0 + np) of one sequence, from its key rows k [*, D] and gate rows gs [*, D]
// (bf16): per dimension d, prob_i = bf16(softmax_i(gs[4p+i][d] + ape[i][d])) over the pool's `pool` tokens,
// out[p][d] = bf16(sum_i bf16(prob_i * k[4p+i][d])). ape bf16 [pool, D] (f16 when f16 is set).
void idx_pool(const bf16* k, const bf16* gs, const void* ape, bool f16, bf16* out, int p0, int np, int pool, int D,
              hipStream_t st);
// Index scores of the chunk's n queries against the sequence's complete pools [0, npool):
//   s[t][p] = sum_h (w[t][h] * H^-0.5) * relu(scale * q[t][h] . pk[p]), q bf16 [n, H, D], w bf16 [n, H],
// -inf where pool p is not complete by query t's position (4p + pool - 1 > pos[t]). out f32 [n, npool].
void idx_scores(const bf16* q, const bf16* w, const bf16* pk, const int32_t* pos, float* out, int n, int npool,
                int H, int D, int pool, float scale, hipStream_t st);
// Selected pools [n, k] (-1 = none) -> token indices [n, k * pool + pool - 1]: each pool's tokens, then the
// query's incomplete pool (positions [pos - (pos + 1) % pool + 1, pos]), -1 padded.
void idx_expand(const int32_t* pools, const int32_t* pos, int32_t* idx, int n, int k, int pool, hipStream_t st);

// ---- attention ---------------------------------------------------------------------------------------
// Absorbed NoPE MLA over a sequence's latent rows kv [*, r] bf16 (one row per position): for query t and
// head h, softmax over its keys of scale * qa[t][h] . kv[j] (fp32), ol[t][h] = sum_j p_j kv[j] (fp32). Keys:
// idx[t][:] (width w, -1 skipped) when idx is given, else every position <= pos[t]. qa / ol f32 [n, H, r].
void mla_attn(const float* qa, const bf16* kv, const int32_t* idx, int w, const int32_t* pos, float* ol, int n,
              int H, int r, float scale, hipStream_t st);

// ---- MoE ---------------------------------------------------------------------------------------------
// Sigmoid router (noaux_tc, one group): s = sigmoid(logits); the K largest s + bias (ties: lower index)
// are selected (ascending expert order); weights = s / (sum s + 1e-20) * scale.
void route_sigmoid(const float* logits, const float* bias, int32_t* sel, float* weights, int T, int E, int K,
                   float scale, hipStream_t st);
// h = bf16(bf16(silu(min(g, lim))) * clamp(u, -lim, lim)).
void swiglu_hf(const bf16* gate, const bf16* up, bf16* h, int rows, int dim, float limit, hipStream_t st);
// Glm5NextTextMoE's sums in bf16: acc = 0; for j (ascending expert): acc = bf16(acc + bf16(y[pidx[t][j]] *
// rw[pidx[t][j]])); out[t] = bf16(acc + shared[t]). y bf16 [pairs, dim], rw f32 [pairs].
void moe_combine_hf(const bf16* y, const float* rw, const int32_t* pidx, const bf16* shared, bf16* out, int T,
                    int K, int dim, hipStream_t st);

}  // namespace unholy::kernels
