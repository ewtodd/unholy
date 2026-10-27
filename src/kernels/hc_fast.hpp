// Fast Hyper-Connection kernels. Same contracts as kernels::hc_pre (ref.hpp).
#pragma once

#include "kernels/ref.hpp"

namespace unholy::kernels {

void hc_pre_tiled(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st);

// Small T (decode): split-K partial sums into `part` [T, 32, 25] floats, then Sinkhorn and the mix.
// counters (T zeroed words, left zeroed; one set per stream): one launch, the last of a token's 32 partial blocks
// finishing it (the same values). norm_w (with counters): also y_norm = rmsnorm_slices(y, norm_w, norm_eps2).
void hc_pre_small(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, float* part, int T, int hc, int dim, int iters, float norm_eps, float hc_eps,
                  hipStream_t st, uint32_t* counters = nullptr, const bf16* norm_w = nullptr, bf16* y_norm = nullptr,
                  float norm_eps2 = 0.0f);
constexpr int kHcSmallMaxTokens = 32;

// Fault diagnostics (UNHOLY_CANARY): hc_pre_tiled checks the address of every weight load against one
// recomputed from a second copy of the pointer passed at the end of the argument list. A mismatch (a
// register that changed under the kernel) is recorded with the wave's hardware id (HW_ID1: SE / SA /
// WGP / SIMD) and the load uses the recomputed address instead of faulting.
struct HcCanaryLog {
  static constexpr unsigned kMax = 64;
  unsigned count, pad;
  struct Event {
    uint64_t got, want;
    unsigned hw_id1, where;  // where: k0 << 16 | i
  } ev[kMax];
};
void hc_set_canary(HcCanaryLog* log);

}  // namespace unholy::kernels
