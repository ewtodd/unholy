// Small-batch variants of reference elementwise kernels (ref_elem.hip): the same per-element arithmetic
// in the same order, spread over (row, slice) workgroups, so a decode step's few rows do not run on one
// workgroup each. Bit-identical to kernels::rmsnorm / kernels::hc_post.
#pragma once

#include "kernels/ref.hpp"

namespace unholy::kernels {

// kernels::rmsnorm, out of place only (y != x): every slice repeats the row's sum of squares.
void rmsnorm_slices(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps,
                    hipStream_t st);
// kernels::hc_post (in place allowed: each column is read and written by one thread).
// The mHC post-mix done by the producer of x, in place on the streams x [T][hc][dim] (x null: not asked for).
struct HcPostArgs {
  bf16* x = nullptr;
  const float* post = nullptr;
  const float* comb = nullptr;
  int hc = 0;
};
void hc_post_slices(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc,
                    int dim, hipStream_t st);
// rmsnorm_slices, then act_quant_fp8 (block 128) of its output into q / qs, in one launch. false (nothing
// launched) when the slices are not multiples of 256 values or in place.
bool rmsnorm_quant_slices(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, uint8_t* q, uint8_t* qs, int rows,
                          int dim, float eps, hipStream_t st);
// kv rows [rows, dim]: rmsnorm in place, rope (forward) on the last 2 * npair dims, fake_quant_fp8 (block
// 64) of the first dim - 2 * npair: the three reference kernels' arithmetic in one launch.
void kv_chain(bf16* kv, const bf16* w, int rows, int dim, float eps, const int32_t* pos, const float2* freqs, int npair,
              hipStream_t st);
// head_rms_bf16 then rope (forward, the last 2 * npair dims of each head) on [rows, heads, dim].
void head_rms_rope(bf16* x, int rows, int heads, int dim, float eps, const int32_t* pos, const float2* freqs, int npair,
                   hipStream_t st);
// Slices per row so a launch of `rows` rows has ~64 workgroups, each slice >= 256 values.
int row_slices(int rows, int dim);

}  // namespace unholy::kernels
