// Skinny GEMMs (M <= 32 rows: decode, small batches): one wave per output column, K split across the
// 32 lanes in 16-element chunks, a wave reduction at the end. Bandwidth-bound on the weights, which
// is the right bound at small M (the 64-row WMMA tiles leave most of the GPU idle there).
//
// Numerics follow the WMMA/reference kernels: fp32 accumulation; the W8A8 path scales each 16-wide
// chunk's exact partial sum by its 128-block scale product (the reference scales whole 128-blocks; the
// same products, summed in a different fp32 order).
#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "kernels/dev.hip.hpp"
#include "kernels/gemv.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int WAVES = 8, NT = WAVES * 32, MAXM = 32;  // MAXM: rows supported (kernels templated on R <= MAXM)

__device__ __forceinline__ void load16_fp8(const uint8_t* p, float* v) {
  const uint4 q = *reinterpret_cast<const uint4*>(p);
  e4m3x4_to_f32(q.x, v);
  e4m3x4_to_f32(q.y, v + 4);
  e4m3x4_to_f32(q.z, v + 8);
  e4m3x4_to_f32(q.w, v + 12);
}
__device__ __forceinline__ void load16_bf16(const bf16* p, float* v) {
  const uint4 q0 = *reinterpret_cast<const uint4*>(p), q1 = *reinterpret_cast<const uint4*>(p + 8);
  const uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[2 * i] = __uint_as_float(w[i] << 16);
    v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

template <typename TOut>
__device__ __forceinline__ void store(TOut* c, size_t i, float v) {
  if constexpr (sizeof(TOut) == 2) c[i] = f32_to_bf16(v);
  else c[i] = v;
}

// A wave computes C columns (n0 .. n0 + C - 1; C = 1 by default): per column, the lanes' K partition, the
// 16-element fma chains, the scaling and the wave reduction do not depend on C. Rows are unrolled up to
// MAXM with uniform guards (m < M). fp8 decodes use gfx12's conversion instructions (dev::e4m3x4_to_f32):
// the bit decode made the W8A8 GEMV ALU-bound (~380 GB/s at one row, ~85 GB/s at eight).

// W8A8: the wave's columns n0 .. n0 + C - 1 (< N).
template <int C, int R>
__device__ __forceinline__ void w8a8_cols(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw,
                                          bf16* c, int ldc, int M, int N, int K, int n0, int lane) {
  const int nc = min(C, N - n0), nkb = K / 128;
  float acc[C][R] = {};
  for (int k = lane * 16; k < K; k += 512) {
    float wv[C][16], ws[C];
    const int kb = k / 128;
#pragma unroll
    for (int j = 0; j < C; ++j)
      if (j < nc) {
        load16_fp8(w + size_t(n0 + j) * K + k, wv[j]);
        ws[j] = e8m0_to_f32(sw[size_t((n0 + j) / 128) * nkb + kb]);
      }
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (m >= M) break;
      float av[16];
      load16_fp8(a + size_t(m) * K + k, av);
      const float sm = e8m0_to_f32(sa[size_t(m) * nkb + kb]);
#pragma unroll
      for (int j = 0; j < C; ++j) {
        if (j >= nc) break;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) part = fmaf(av[i], wv[j][i], part);
        acc[j][m] += part * (sm * ws[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (j >= nc || m >= M) break;
      const float v = warp_sum(acc[j][m]);
      if (lane == 0) c[size_t(m) * ldc + n0 + j] = f32_to_bf16(v);
    }
}
template <int C, int R>
__global__ __launch_bounds__(NT) void gemv_w8a8_kernel(const uint8_t* a, const uint8_t* sa, const uint8_t* w,
                                                       const uint8_t* sw, bf16* c, int ldc, int M, int N, int K) {
  const int n0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C;
  if (n0 < N) w8a8_cols<C, R>(a, sa, w, sw, c, ldc, M, N, K, n0, threadIdx.x & 31);
}
// Several column ranges (parts with their own weights, scales and outputs) on one input: wave -> (part,
// column), then the single GEMV's code.
template <int R>
__global__ __launch_bounds__(NT) void gemv_w8a8_parts_kernel(const uint8_t* a, const uint8_t* sa, GemvParts ps, int M,
                                                             int K) {
  int n0 = blockIdx.x * WAVES + (threadIdx.x >> 5), i = 0;
  while (i + 1 < ps.n && n0 >= ps.p[i].N) n0 -= ps.p[i++].N;
  const GemvPart& p = ps.p[i];
  if (n0 < p.N)
    w8a8_cols<1, R>(a, sa, static_cast<const uint8_t*>(p.w), p.sw, static_cast<bf16*>(p.c), p.ldc, M, p.N, K, n0,
                    threadIdx.x & 31);
}

// bf16 weights x bf16 activations: the wave's columns n0 .. n0 + C - 1 (< N).
template <int C, int R, typename TOut>
__device__ __forceinline__ void bf16_cols(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N, int K,
                                          int n0, int lane) {
  const int nc = min(C, N - n0);
  float acc[C][R] = {};
  for (int k = lane * 16; k < K; k += 512) {
    float wv[C][16];
#pragma unroll
    for (int j = 0; j < C; ++j)
      if (j < nc) load16_bf16(w + size_t(n0 + j) * K + k, wv[j]);
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (m >= M) break;
      float av[16];
      load16_bf16(a + size_t(m) * lda + k, av);
#pragma unroll
      for (int j = 0; j < C; ++j) {
        if (j >= nc) break;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[j][m] = fmaf(av[i], wv[j][i], acc[j][m]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (j >= nc || m >= M) break;
      const float v = warp_sum(acc[j][m]);
      if (lane == 0) store(c, size_t(m) * ldc + n0 + j, v);
    }
}
template <int C, int R, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_kernel(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M,
                                                       int N, int K) {
  const int n0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C;
  if (n0 < N) bf16_cols<C, R, TOut>(a, lda, w, c, ldc, M, N, K, n0, threadIdx.x & 31);
}
// K < 512 (K % 64 == 0; KDA's low-rank gate projections, K = 128): a wave per 8 columns, 4 lanes per column, each
// lane K / 4 consecutive values of it (the per-512 layout above would leave most lanes idle).
template <int R, typename TOut>
__device__ __forceinline__ void bf16_smallk_cols(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N,
                                                 int K, int block) {
  const int lane = threadIdx.x & 31, n = (block * WAVES + (threadIdx.x >> 5)) * 8 + (lane >> 2);
  const int kq = K / 4, k0 = (lane & 3) * kq;
  float acc[R] = {};
  if (n < N)
    for (int k = k0; k < k0 + kq; k += 16) {
      float wv[16];
      load16_bf16(w + size_t(n) * K + k, wv);
#pragma unroll
      for (int m = 0; m < R; ++m) {
        if (m >= M) break;
        float av[16];
        load16_bf16(a + size_t(m) * lda + k, av);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[m] = fmaf(av[i], wv[i], acc[m]);
      }
    }
#pragma unroll
  for (int m = 0; m < R; ++m) {
    if (m >= M) break;
    float v = acc[m];
    v += __shfl_xor(v, 1, 32);
    v += __shfl_xor(v, 2, 32);
    if ((lane & 3) == 0 && n < N) store(c, size_t(m) * ldc + n, v);
  }
}
template <int R, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_smallk_kernel(const bf16* a, int lda, const bf16* w, TOut* c, int ldc,
                                                              int M, int N, int K) {
  bf16_smallk_cols<R, TOut>(a, lda, w, c, ldc, M, N, K, blockIdx.x);
}
// Blocks [0, nb0): the first GEMV, the rest the second.
template <int R>
__global__ __launch_bounds__(NT) void gemv_bf16_smallk2_kernel(const bf16* a0, int lda0, const bf16* w0, bf16* c0,
                                                               int ldc0, int N0, const bf16* a1, int lda1,
                                                               const bf16* w1, bf16* c1, int ldc1, int N1, int M, int K,
                                                               int nb0) {
  if (int(blockIdx.x) < nb0) bf16_smallk_cols<R, bf16>(a0, lda0, w0, c0, ldc0, M, N0, K, blockIdx.x);
  else bf16_smallk_cols<R, bf16>(a1, lda1, w1, c1, ldc1, M, N1, K, blockIdx.x - nb0);
}
template <int R, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_parts_kernel(const bf16* a, int lda, GemvParts ps, int M, int K) {
  int n0 = blockIdx.x * WAVES + (threadIdx.x >> 5), i = 0;
  while (i + 1 < ps.n && n0 >= ps.p[i].N) n0 -= ps.p[i++].N;
  const GemvPart& p = ps.p[i];
  if (n0 < p.N)
    bf16_cols<1, R, TOut>(a, lda, static_cast<const bf16*>(p.w), static_cast<TOut*>(p.c), p.ldc, M, p.N, K, n0,
                          threadIdx.x & 31);
}

// Grouped "mgk,grk->mgr": fp8 weights x 128x128 scales (exact in bf16), bf16 activations. A wave's C
// columns lie in one group (R % C == 0).
template <int C, int RM>
__global__ __launch_bounds__(NT) void gemv_grouped_fp8w_kernel(const bf16* a, const uint8_t* w, const uint8_t* sw,
                                                               bf16* c, int M, int G, int R, int K) {
  const int col0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C, lane = threadIdx.x & 31;  // col over G*R
  if (col0 >= G * R) return;
  const int g = col0 / R, r0 = col0 % R;
  float acc[C][RM] = {};
  for (int k = lane * 16; k < K; k += 512) {
    float wv[C][16];
#pragma unroll
    for (int j = 0; j < C; ++j) {
      load16_fp8(w + size_t(col0 + j) * K + k, wv[j]);
      const float s = e8m0_to_f32(sw[size_t((col0 + j) / 128) * (K / 128) + k / 128]);
#pragma unroll
      for (int i = 0; i < 16; ++i) wv[j][i] *= s;  // exact (power of two)
    }
#pragma unroll
    for (int m = 0; m < RM; ++m) {
      if (m >= M) break;
      float av[16];
      load16_bf16(a + (size_t(m) * G + g) * K + k, av);
#pragma unroll
      for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[j][m] = fmaf(av[i], wv[j][i], acc[j][m]);
    }
  }
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int m = 0; m < RM; ++m) {
      if (m >= M) break;
      const float v = warp_sum(acc[j][m]);
      if (lane == 0) c[(size_t(m) * G + g) * R + r0 + j] = f32_to_bf16(v);
    }
}

int g_force_cols = 0;
// Columns per wave: 1. With the weights streamed from DRAM (unholy-gemv-test) more columns per wave only
// cost occupancy; the template stays for tuning (gemv_force_cols).
int cols_per_wave(int) { return g_force_cols ? g_force_cols : 1; }
int blocks_for(int cols, int C) { return (cols + WAVES * C - 1) / (WAVES * C); }

// Calls f(std::integral_constant<int, R>) with the smallest row capacity R in {8, 16, 32} >= M.
template <typename F>
void by_rows(int M, F&& f) {
  if (M <= 8) f(std::integral_constant<int, 8>{});
  else if (M <= 16) f(std::integral_constant<int, 16>{});
  else f(std::integral_constant<int, 32>{});
}

}  // namespace

int gemv_max_rows() {
  static const int v = [] {
    const char* e = std::getenv("UNHOLY_GEMV_ROWS");
    return e ? std::clamp(std::atoi(e), 1, int(kGemvRowsCap)) : 8;
  }();
  return v;
}

void gemv_force_cols(int c) { g_force_cols = c == 1 || c == 2 || c == 4 ? c : 0; }

void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8: M > 32 or K % 512");
  const auto go = [&](auto r) {
    constexpr int R = decltype(r)::value;
    switch (cols_per_wave(N)) {
      case 4: gemv_w8a8_kernel<4, R><<<blocks_for(N, 4), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K); break;
      case 2: gemv_w8a8_kernel<2, R><<<blocks_for(N, 2), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K); break;
      default: gemv_w8a8_kernel<1, R><<<blocks_for(N, 1), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
    }
  };
  by_rows(M, go);
  UH_CHECK_LAUNCH();
}
template <typename TOut>
void gemv_bf16_any(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (M <= MAXM && K < 512 && K % 64 == 0) {
    by_rows(M, [&](auto r) {
      gemv_bf16_smallk_kernel<decltype(r)::value, TOut><<<(N + WAVES * 8 - 1) / (WAVES * 8), NT, 0, st>>>(a, lda, w, c,
                                                                                                         ldc, M, N, K);
    });
    UH_CHECK_LAUNCH();
    return;
  }
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16: M > 32, or K % 512 (K >= 512) or K % 64");
  const auto go = [&](auto r) {
    constexpr int R = decltype(r)::value;
    switch (cols_per_wave(N)) {
      case 4: gemv_bf16_kernel<4, R, TOut><<<blocks_for(N, 4), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K); break;
      case 2: gemv_bf16_kernel<2, R, TOut><<<blocks_for(N, 2), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K); break;
      default: gemv_bf16_kernel<1, R, TOut><<<blocks_for(N, 1), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K);
    }
  };
  by_rows(M, go);
  UH_CHECK_LAUNCH();
}
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  gemv_bf16_any(a, lda, w, c, ldc, M, N, K, st);
}
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st) {
  gemv_bf16_any(a, lda, w, c, ldc, M, N, K, st);
}
namespace {
GemvParts make_parts(const GemvPart* parts, int nparts, int& cols) {
  if (nparts < 1 || nparts > GemvParts::kMax) throw std::runtime_error("gemv parts: 1..3");
  GemvParts ps;
  ps.n = nparts;
  cols = 0;
  for (int i = 0; i < nparts; ++i) {
    ps.p[i] = parts[i];
    cols += parts[i].N;
  }
  return ps;
}
}  // namespace

void gemv_w8a8_parts(const uint8_t* a, const uint8_t* sa, const GemvPart* parts, int nparts, int M, int K,
                     hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8_parts: M > 32 or K % 512");
  int cols = 0;
  const GemvParts ps = make_parts(parts, nparts, cols);
  by_rows(M, [&](auto r) {
    gemv_w8a8_parts_kernel<decltype(r)::value><<<blocks_for(cols, 1), NT, 0, st>>>(a, sa, ps, M, K);
  });
  UH_CHECK_LAUNCH();
}
void gemv_bf16_f32out_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16_f32out_parts: M > 32 or K % 512");
  int cols = 0;
  const GemvParts ps = make_parts(parts, nparts, cols);
  by_rows(M, [&](auto r) {
    gemv_bf16_parts_kernel<decltype(r)::value, float><<<blocks_for(cols, 1), NT, 0, st>>>(a, lda, ps, M, K);
  });
  UH_CHECK_LAUNCH();
}

void gemv_bf16_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16_parts: M > 32 or K % 512");
  int cols = 0;
  const GemvParts ps = make_parts(parts, nparts, cols);
  by_rows(M, [&](auto r) {
    gemv_bf16_parts_kernel<decltype(r)::value, bf16><<<blocks_for(cols, 1), NT, 0, st>>>(a, lda, ps, M, K);
  });
  UH_CHECK_LAUNCH();
}

void gemv_bf16_smallk2(const bf16* a0, int lda0, const bf16* w0, bf16* c0, int ldc0, int N0, const bf16* a1, int lda1,
                       const bf16* w1, bf16* c1, int ldc1, int N1, int M, int K, hipStream_t st) {
  if (M > MAXM || K >= 512 || K % 64) throw std::runtime_error("gemv_bf16_smallk2: M > 32, K >= 512 or K % 64");
  const int nb0 = (N0 + WAVES * 8 - 1) / (WAVES * 8), nb1 = (N1 + WAVES * 8 - 1) / (WAVES * 8);
  by_rows(M, [&](auto r) {
    gemv_bf16_smallk2_kernel<decltype(r)::value><<<nb0 + nb1, NT, 0, st>>>(a0, lda0, w0, c0, ldc0, N0, a1, lda1, w1, c1,
                                                                          ldc1, N1, M, K, nb0);
  });
  UH_CHECK_LAUNCH();
}

void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_grouped_fp8w: M > 32 or K % 512");
  int C = cols_per_wave(G * R);
  while (R % C) C /= 2;
  const auto go = [&](auto r) {
    constexpr int RR = decltype(r)::value;
    switch (C) {
      case 4: gemv_grouped_fp8w_kernel<4, RR><<<blocks_for(G * R, 4), NT, 0, st>>>(a, w, sw, c, M, G, R, K); break;
      case 2: gemv_grouped_fp8w_kernel<2, RR><<<blocks_for(G * R, 2), NT, 0, st>>>(a, w, sw, c, M, G, R, K); break;
      default: gemv_grouped_fp8w_kernel<1, RR><<<blocks_for(G * R, 1), NT, 0, st>>>(a, w, sw, c, M, G, R, K);
    }
  };
  by_rows(M, go);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
