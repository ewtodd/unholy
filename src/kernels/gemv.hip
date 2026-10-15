// Skinny GEMMs (M <= 8 rows: decode, small batches): one wave per output column, K split across the
// 32 lanes in 16-element chunks, a wave reduction at the end. Bandwidth-bound on the weights, which
// is the right bound at small M (the 64-row WMMA tiles leave most of the GPU idle there).
//
// Numerics follow the WMMA/reference kernels: fp32 accumulation; the W8A8 path scales each 16-wide
// chunk's exact partial sum by its 128-block scale product (the reference scales whole 128-blocks; the
// same products, summed in a different fp32 order).
#include "kernels/dev.hip.hpp"
#include "kernels/gemv.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int WAVES = 8, NT = WAVES * 32, MAXM = 8;

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

// W8A8.
template <int C>
__global__ __launch_bounds__(NT) void gemv_w8a8_kernel(const uint8_t* a, const uint8_t* sa, const uint8_t* w,
                                                       const uint8_t* sw, bf16* c, int ldc, int M, int N, int K) {
  const int n0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C, lane = threadIdx.x & 31;
  if (n0 >= N) return;
  const int nc = min(C, N - n0), nkb = K / 128;
  float acc[C][MAXM] = {};
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
    for (int m = 0; m < MAXM; ++m) {
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
    for (int m = 0; m < MAXM; ++m) {
      if (j >= nc || m >= M) break;
      const float v = warp_sum(acc[j][m]);
      if (lane == 0) c[size_t(m) * ldc + n0 + j] = f32_to_bf16(v);
    }
}

// bf16 weights x bf16 activations.
template <int C, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_kernel(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M,
                                                       int N, int K) {
  const int n0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C, lane = threadIdx.x & 31;
  if (n0 >= N) return;
  const int nc = min(C, N - n0);
  float acc[C][MAXM] = {};
  for (int k = lane * 16; k < K; k += 512) {
    float wv[C][16];
#pragma unroll
    for (int j = 0; j < C; ++j)
      if (j < nc) load16_bf16(w + size_t(n0 + j) * K + k, wv[j]);
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
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
    for (int m = 0; m < MAXM; ++m) {
      if (j >= nc || m >= M) break;
      const float v = warp_sum(acc[j][m]);
      if (lane == 0) store(c, size_t(m) * ldc + n0 + j, v);
    }
}

// Grouped "mgk,grk->mgr": fp8 weights x 128x128 scales (exact in bf16), bf16 activations. A wave's C
// columns lie in one group (R % C == 0).
template <int C>
__global__ __launch_bounds__(NT) void gemv_grouped_fp8w_kernel(const bf16* a, const uint8_t* w, const uint8_t* sw,
                                                               bf16* c, int M, int G, int R, int K) {
  const int col0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C, lane = threadIdx.x & 31;  // col over G*R
  if (col0 >= G * R) return;
  const int g = col0 / R, r0 = col0 % R;
  float acc[C][MAXM] = {};
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
    for (int m = 0; m < MAXM; ++m) {
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
    for (int m = 0; m < MAXM; ++m) {
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

}  // namespace

void gemv_force_cols(int c) { g_force_cols = c == 1 || c == 2 || c == 4 ? c : 0; }

void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8: M > 8 or K % 512");
  switch (cols_per_wave(N)) {
    case 4: gemv_w8a8_kernel<4><<<blocks_for(N, 4), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K); break;
    case 2: gemv_w8a8_kernel<2><<<blocks_for(N, 2), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K); break;
    default: gemv_w8a8_kernel<1><<<blocks_for(N, 1), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
  }
  UH_CHECK_LAUNCH();
}
template <typename TOut>
void gemv_bf16_any(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16: M > 8 or K % 512");
  switch (cols_per_wave(N)) {
    case 4: gemv_bf16_kernel<4, TOut><<<blocks_for(N, 4), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K); break;
    case 2: gemv_bf16_kernel<2, TOut><<<blocks_for(N, 2), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K); break;
    default: gemv_bf16_kernel<1, TOut><<<blocks_for(N, 1), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K);
  }
  UH_CHECK_LAUNCH();
}
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  gemv_bf16_any(a, lda, w, c, ldc, M, N, K, st);
}
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st) {
  gemv_bf16_any(a, lda, w, c, ldc, M, N, K, st);
}
void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_grouped_fp8w: M > 8 or K % 512");
  int C = cols_per_wave(G * R);
  while (R % C) C /= 2;
  switch (C) {
    case 4: gemv_grouped_fp8w_kernel<4><<<blocks_for(G * R, 4), NT, 0, st>>>(a, w, sw, c, M, G, R, K); break;
    case 2: gemv_grouped_fp8w_kernel<2><<<blocks_for(G * R, 2), NT, 0, st>>>(a, w, sw, c, M, G, R, K); break;
    default: gemv_grouped_fp8w_kernel<1><<<blocks_for(G * R, 1), NT, 0, st>>>(a, w, sw, c, M, G, R, K);
  }
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
