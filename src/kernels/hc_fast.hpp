// Fast Hyper-Connection kernels. Same contracts as kernels::hc_pre (ref.hpp).
#pragma once

#include "kernels/ref.hpp"

namespace unholy::kernels {

void hc_pre_tiled(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st);

}  // namespace unholy::kernels
