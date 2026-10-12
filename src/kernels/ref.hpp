// Reference kernels: one per DeepSeek-V4 op, written to reproduce the reference implementation's
// arithmetic (inference/model.py + kernel.py) rather than to be fast. Every rounding to bf16 that
// torch performs between ops is performed here at the same point; accumulation is fp32. The fast
// kernels that replace these on the hot path are tested against them.
//
// Conventions: bf16 = uint16_t, fp8 = uint8_t (e4m3fn), scales = uint8_t e8m0 exponents. Matrices
// are row-major. Linear weights are [out, in] (PyTorch layout); GEMMs compute C = A * W^T.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/indexer.hpp"

namespace unholy::kernels {

using bf16 = uint16_t;

// ---- quantization -------------------------------------------------------------------------------
// act_quant(x, block, ue8m0): x bf16 [rows, cols] -> q e4m3 [rows, cols], s e8m0 [rows, cols/block].
void act_quant_fp8(const bf16* x, int ldx, uint8_t* q, uint8_t* s, int rows, int cols, int block, hipStream_t st);
// In-place fake quant (quantize + dequantize back to bf16) on cols [0, cols) of each row.
void fake_quant_fp8(bf16* x, int ldx, int rows, int cols, int block, hipStream_t st);
void fake_quant_fp4(bf16* x, int ldx, int rows, int cols, int block, hipStream_t st);

// ---- GEMMs (C[M,N] = A[M,K] * W[N,K]^T) -----------------------------------------------------------
// fp8 x fp8, per-128 K-block partial sums scaled by sa[m][kb] * sw[n/128][kb]; bf16 out.
void gemm_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc,
               int M, int N, int K, hipStream_t st);
// fp8 activations x packed e2m1 weights [N, K/2] (low nibble first), per-32 partial sums scaled by
// sa[m][k/128] * sw[n][k/32]; bf16 out.
void gemm_a8w4(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc,
               int M, int N, int K, hipStream_t st);
// bf16 x bf16 -> bf16.
void gemm_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st);
// fp32 x bf16 weights -> fp32 (the reference's fp32 linears: compressor, router, LM head).
void gemm_f32(const float* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st);
// Grouped bf16 einsum "mgk,grk->mgr" with fp8 weights dequantized by 128x128 block scales
// (exactly representable in bf16): a [M, G, K] bf16, w e4m3 [G*R, K] + sw [G*R/128, K/128]; c [M, G, R] bf16.
void gemm_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st);

// ---- elementwise / norms / rope / Hyper-Connections ---------------------------------------------
void cast_bf16_f32(const bf16* x, float* y, size_t n, hipStream_t st);
// out[t][h][:] = table[ids[t]][:] for h < hc.
void embed_hc(const int32_t* ids, const bf16* table, bf16* out, int T, int dim, int hc, hipStream_t st);
// RMSNorm with weight: y = bf16(w * (x * rsqrt(mean(x^2) + eps))). y may alias x.
void rmsnorm(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps, hipStream_t st);
// Per-head RMS normalisation without weight, in torch bf16 semantics (q path).
void head_rms_bf16(bf16* x, int rows, int heads, int dim, float eps, hipStream_t st);
// Rotary on dims [off, off + 2*npair) of each head; pairs are adjacent (x[2i], x[2i+1]).
// freqs: [max_pos][npair] (cos, sin) fp32. inverse = conjugate.
void rope(bf16* x, int ld_row, int rows, int heads, int head_stride, int off, int npair, const int32_t* pos,
          const float2* freqs, bool inverse, hipStream_t st);
// 128-point Sylvester Hadamard * 128^-0.5 on each 128-wide segment (fp32 butterflies, bf16 out).
void hadamard128(bf16* x, size_t nseg, hipStream_t st);
// hc_pre: x [T, hc, dim] bf16; fn [mix, hc*dim] f32; -> y [T, dim] bf16, post [T, hc], comb [T, hc, hc].
void hc_pre(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
            float* comb, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st);
// hc_post: out [T, hc, dim] = post[j] * x + sum_i comb[i][j] * res[i].  out may alias res.
void hc_post(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc,
             int dim, hipStream_t st);
// hc_head: y [T, dim] = sum_i (sigmoid(mix_i * scale + base_i) + eps) * x_i.
void hc_head(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, int T, int hc,
             int dim, float norm_eps, float hc_eps, hipStream_t st);

// ---- attention ------------------------------------------------------------------------------------
// Compressor rows. proj_kv/proj_score: [n, coff*d] fp32 for chunk positions [s0, s0 + n) (score
// already includes ape); ring_*: [ring, coff*d] fp32 holding positions [s0 - ring, s0) at pos % ring.
// Emits rows i in [row0, row0 + nrows) (all complete blocks ending in the chunk) as fp32 [nrows, d].
void compress_rows(const float* proj_kv, const float* proj_score, const float* ring_kv, const float* ring_score,
                   float* out, int s0, int n, int row0, int nrows, int ratio, int d, bool overlap, int ring,
                   hipStream_t st);
// Adds ape[pos % ratio] to score rows.
void add_ape(float* score, const float* ape, int s0, int n, int ratio, int width, hipStream_t st);
// Saves the chunk's trailing projections into the ring (positions >= s0 + n - ring).
void ring_update(const float* proj_kv, const float* proj_score, float* ring_kv, float* ring_score, int s0, int n,
                 int ring, int width, hipStream_t st);
void cast_f32_bf16(const float* x, bf16* y, size_t n, hipStream_t st);

// Paged row access: row r of a sequence lives at pool + (table[r / rows_per_page] * rows_per_page +
// r % rows_per_page) * row_stride (elements).
struct PagedRows {
  bf16* pool = nullptr;
  const int32_t* table = nullptr;
  int rows_per_page = 0;
  int row_stride = 0;
};
void store_rows(const bf16* rows, int n, int width, PagedRows dst, int row0, hipStream_t st);

// Indexer scores: q [T, H, D] bf16, w [T, H] bf16, keys = rows [0, nkeys) of the fp4 key pool; query t sees
// keys j < (pos[t] + 1) / ratio. out [T, nkeys] f32 (bf16-valued; -inf when not visible).
void indexer_scores(const bf16* q, const bf16* w, PagedFp4 keys, const int32_t* pos, float* out, int T, int H,
                    int D, int nkeys, int ratio, hipStream_t st);
// Top-k per row of scores (desc, ties -> lower index); idx [T, k] (-1 padded when fewer visible).
void topk_rows(const float* scores, int T, int n, int k, int32_t* idx, hipStream_t st);

// Sparse attention with sink. q [T, H, D] bf16 (row stride H*D). For query t: window positions
// [pos-127, pos] (from `chunk_kv` rows when >= s0, else `ring` slot pos % 128), then compressed rows:
// comp_idx [T, ncomp] (-1 = skip) if given, else all rows j < (pos+1)/ratio (HCA). o [T, H, D] bf16.
struct AttnKv {
  const bf16* chunk_kv = nullptr;  // [n_chunk, D] for positions [s0, s0 + n_chunk)
  const bf16* ring = nullptr;      // [128, D], slot = pos % 128
  int s0 = 0;
  PagedRows comp;  // compressed rows (comp.pool == nullptr: none)
  const int32_t* comp_idx = nullptr;
  int ncomp = 0;  // width of comp_idx, or 0 for "all visible"
  int ratio = 0;  // 0: window only
};
void sparse_attn(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, int H, int D,
                 float scale, hipStream_t st);
// Saves the chunk's last <=128 kv rows into the window ring.
void window_update(const bf16* chunk_kv, bf16* ring, int s0, int n, int D, hipStream_t st);

// ---- MoE -------------------------------------------------------------------------------------------
// Router: logits fp32 [T, E] -> sel [T, K] (int32), weights [T, K] fp32. sqrtsoftplus scores, bias for
// selection (hash layers: experts from tid2eid[ids]), weights normalised and scaled.
void route(const float* logits, const float* bias, const int32_t* tid2eid, const int32_t* ids, int32_t* sel,
           float* weights, int T, int E, int K, float route_scale, hipStream_t st);
// Same routing, one block per token (parallel scores and argmax rounds).
void route_fast(const float* logits, const float* bias, const int32_t* tid2eid, const int32_t* ids, int32_t* sel,
                float* weights, int T, int E, int K, float route_scale, hipStream_t st);
// SwiGLU with clamp: h = bf16(silu(min(g, lim)) * clamp(u, -lim, lim) * (w ? w[row] : 1)).
void swiglu(const bf16* gate, const bf16* up, const float* w, bf16* h, int rows, int dim, float limit,
            hipStream_t st);
// y[rows_idx[r]] += src[r] (fp32 += bf16), r in order.
void scatter_add(const bf16* src, const int32_t* rows_idx, float* y, int n, int dim, hipStream_t st);
void gather_rows(const bf16* src, const int32_t* rows_idx, bf16* dst, int n, int dim, hipStream_t st);
// out[t] = bf16(sum_j pair_out[pidx[t][j]] (j ascending) + shared[t]), fp32 accumulation: the
// reference's y = 0; y[idx] += expert_e(...) for e ascending; y += shared.
void moe_combine(const bf16* pair_out, const int32_t* pidx, const bf16* shared, bf16* out, int T, int K, int dim,
                 hipStream_t st);
void add_bf16_to_f32(const bf16* x, float* y, size_t n, hipStream_t st);
// out[j] = start + j * step.
void positions(int32_t* out, int n, int start, int step, hipStream_t st);

// ---- EXL3 (reference) ------------------------------------------------------------------------------
// xh [M, K] fp32 = H128(x * suh) per 128-block (suh fp16 [K]).
void exl3_in(const bf16* x, int ldx, const uint16_t* suh, float* xh, int M, int K, hipStream_t st);
// c [M, N] fp32 = xh @ W_inner, W_inner decoded from the trellis (I16 [K/16, N/16, 16*bits], mul1 codebook).
void gemm_exl3_inner(const float* a, const uint16_t* trellis, int bits, float* c, int M, int N, int K, hipStream_t st);
// out [M, N] bf16 = H128(y) * svh.
void exl3_out(const float* y, const uint16_t* svh, bf16* out, int ldo, int M, int N, hipStream_t st);
// x = bf16(x * c) (a bf16 tensor times a Python scalar in torch).
void scale_bf16(bf16* x, float c, size_t n, hipStream_t st);

}  // namespace unholy::kernels
