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
void hc_post_slices(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc,
                    int dim, hipStream_t st);
// Slices per row so a launch of `rows` rows has ~64 workgroups, each slice >= 256 values.
int row_slices(int rows, int dim);

}  // namespace unholy::kernels
