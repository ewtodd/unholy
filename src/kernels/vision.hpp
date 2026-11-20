// Kernels of the vision towers: DSV4's (model/vision; reference inference/vision.py) and GLM-5.3-Flash's
// (model/glm5_vision; transformers Glm5NextVisionModel). Linear layers are the bf16 WMMA GEMM with fp32 output;
// these epilogues round to bf16 where the reference's bf16 ops do.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/ref.hpp"

namespace unholy::kernels {

// out = bf16(acc + bias) (an nn.Linear with bias); bias may be null.
void vit_bias(const float* acc, int M, int N, const bf16* bias, bf16* out, hipStream_t st);
// out = bf16(gelu_erf(bf16(acc + bias))).
void vit_bias_gelu(const float* acc, int M, int N, const bf16* bias, bf16* out, hipStream_t st);
// acc [M, 2I] = (gate | up): out [M, I] = bf16(bf16(silu(bf16(gate))) * bf16(up)).
void vit_silu_mul(const float* acc, int M, int I, bf16* out, hipStream_t st);
// out = bf16(w * (x * rsqrt(mean(x^2) + eps))) in fp32 (w bf16-valued).
void vit_rmsnorm(const bf16* x, int M, int D, const bf16* w, float eps, bf16* out, hipStream_t st);
// x = bf16(x + y).
void vit_add(bf16* x, const bf16* y, size_t n, hipStream_t st);
// Rotate-half 2-D rope on the q and k parts of qkv [N, 3 * H * 64] in place: per head, halves (x1, x2) ->
// (x1 cos - x2 sin, x2 cos + x1 sin) with cos / sin [N, 32] fp32.
void vit_rope(bf16* qkv, int N, int H, const float* cos, const float* sin, hipStream_t st);
// Bidirectional softmax(q k^T / 8) v over all N positions, per head (head dim 64): qkv [N, 3 * H * 64] ->
// out [N, H * 64]. fp32 scores and online softmax.
void vit_attention(const bf16* qkv, int N, int H, bf16* out, hipStream_t st);
// vit_attention on WMMA (gfx11 + gfx12): fp32 scores and softmax state, P rounded to bf16 for P V.
void vit_attention_wmma(const bf16* qkv, int N, int H, bf16* out, hipStream_t st);
// Aligner unfold (F.pad to multiples of r, F.unfold r x r stride r): x [n_h * n_w, C] -> out
// [ceil(n_h / r) * ceil(n_w / r), C * r * r], column c * r * r + i * r + j.
void vit_unfold(const bf16* x, int n_h, int n_w, int C, int r, bf16* out, hipStream_t st);
// GLM: out = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), transformers' RMSNorm order.
void vit_rmsnorm_hf(const bf16* x, int M, int D, const bf16* w, float eps, bf16* out, hipStream_t st);
// GLM: per head of the q and k parts of qkv [N, 3 * H * 64], in place: RMSNorm over the head's 64 values (qw / kw,
// rounded as vit_rmsnorm_hf), then vit_rope's rotation in fp32 without fused multiply-adds.
void vit_qk_norm_rope(bf16* qkv, int N, int H, const bf16* qw, const bf16* kw, float eps, const float* cos,
                      const float* sin, hipStream_t st);
// GLM: acc [M, 2I] = (gate | up), bias [2I] or null: g = min(bf16(gate + bias), limit), u = bf16(up + bias)
// clamped to +-limit, out [M, I] = bf16(bf16(silu(g)) * u).
void vit_swiglu_clamp(const float* acc, int M, int I, const bf16* bias, float limit, bf16* out, hipStream_t st);
// GLM merger: out = bf16(gelu_erf(bf16((x - mean) * rstd * w + b))), nn.LayerNorm in fp32 then nn.GELU.
void vit_layernorm_gelu(const bf16* x, int M, int D, const bf16* w, const bf16* b, float eps, bf16* out,
                        hipStream_t st);
// Hy4: out = bf16((x - mean) * rstd * w + b), nn.LayerNorm in fp32.
void vit_layernorm(const bf16* x, int M, int D, const bf16* w, const bf16* b, float eps, bf16* out, hipStream_t st);
// Hy4: bidirectional attention for heads of D <= 80 dims padded to 80: qkv [N][3][H][80] (pad dims zero) ->
// out [N][H * D]; softmax(q k^T * scale) in fp32 with an online softmax, P rounded to bf16 for P V (WMMA).
void vit_attention_wmma80(const bf16* qkv, int N, int H, int D, float scale, bf16* out, hipStream_t st);
// Hy4: x [gh * gw][C] += bf16(the learned grid table [edge * edge][C] resampled to gh x gw), F.interpolate's
// bilinear (align_corners false) with scale factors (g + 0.1) / edge as the PR passes them (sampling step
// float(1 / scale)); the sum rounded to bf16.
void vit_add_pos_bilinear(bf16* x, const bf16* table, int edge, int gh, int gw, int C, hipStream_t st);
// Image block rows: src[i] >= 0 -> aligned[src[i]], else params[-1 - src[i]] (the token type's learned
// row); out [n, D].
void vit_assemble(const bf16* aligned, const bf16* params, const int32_t* src, int n, int D, bf16* out,
                  hipStream_t st);

}  // namespace unholy::kernels
