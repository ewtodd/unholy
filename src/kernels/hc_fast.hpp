// Fast Hyper-Connection kernels. Same contracts as kernels::hc_pre (ref.hpp).
#pragma once

#include "kernels/ref.hpp"

namespace unholy::kernels {

void hc_pre_tiled(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st);

// Small T (decode): split-K partial sums into `part` [T, 32, 25] floats, then Sinkhorn and the mix.
void hc_pre_small(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, float* part, int T, int hc, int dim, int iters, float norm_eps, float hc_eps,
                  hipStream_t st);
constexpr int kHcSmallMaxTokens = 32;

}  // namespace unholy::kernels
