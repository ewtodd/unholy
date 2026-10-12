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
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = e4m3_to_f32(uint8_t(w[i >> 2] >> ((i & 3) * 8)));
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

// W8A8.
__global__ __launch_bounds__(NT) void gemv_w8a8_kernel(const uint8_t* a, const uint8_t* sa, const uint8_t* w,
                                                       const uint8_t* sw, bf16* c, int ldc, int M, int N, int K) {
  const int n = blockIdx.x * WAVES + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (n >= N) return;
  const int nkb = K / 128;
  float acc[MAXM] = {};
  const uint8_t* wr = w + size_t(n) * K;
  for (int k = lane * 16; k < K; k += 512) {
    float wv[16];
    load16_fp8(wr + k, wv);
    const int kb = k / 128;
    const float ws = e8m0_to_f32(sw[size_t(n / 128) * nkb + kb]);
    for (int m = 0; m < M; ++m) {
      float av[16];
      load16_fp8(a + size_t(m) * K + k, av);
      float part = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) part = fmaf(av[i], wv[i], part);
      acc[m] += part * (e8m0_to_f32(sa[size_t(m) * nkb + kb]) * ws);
    }
  }
  for (int m = 0; m < M; ++m) {
    const float v = warp_sum(acc[m]);
    if (lane == 0) c[size_t(m) * ldc + n] = f32_to_bf16(v);
  }
}

// bf16 weights x bf16 activations.
template <typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_kernel(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M,
                                                       int N, int K) {
  const int n = blockIdx.x * WAVES + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (n >= N) return;
  float acc[MAXM] = {};
  const bf16* wr = w + size_t(n) * K;
  for (int k = lane * 16; k < K; k += 512) {
    float wv[16];
    load16_bf16(wr + k, wv);
    for (int m = 0; m < M; ++m) {
      float av[16];
      load16_bf16(a + size_t(m) * lda + k, av);
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[m] = fmaf(av[i], wv[i], acc[m]);
    }
  }
  for (int m = 0; m < M; ++m) {
    const float v = warp_sum(acc[m]);
    if (lane == 0) store(c, size_t(m) * ldc + n, v);
  }
}

// Grouped "mgk,grk->mgr": fp8 weights x 128x128 scales (exact in bf16), bf16 activations.
__global__ __launch_bounds__(NT) void gemv_grouped_fp8w_kernel(const bf16* a, const uint8_t* w, const uint8_t* sw,
                                                               bf16* c, int M, int G, int R, int K) {
  const int col = blockIdx.x * WAVES + (threadIdx.x >> 5), lane = threadIdx.x & 31;  // col over G*R
  if (col >= G * R) return;
  const int g = col / R, r = col % R;
  float acc[MAXM] = {};
  const uint8_t* wr = w + size_t(col) * K;
  for (int k = lane * 16; k < K; k += 512) {
    float wv[16];
    load16_fp8(wr + k, wv);
    const float s = e8m0_to_f32(sw[size_t(col / 128) * (K / 128) + k / 128]);
#pragma unroll
    for (int i = 0; i < 16; ++i) wv[i] *= s;  // exact (power of two)
    for (int m = 0; m < M; ++m) {
      float av[16];
      load16_bf16(a + (size_t(m) * G + g) * K + k, av);
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[m] = fmaf(av[i], wv[i], acc[m]);
    }
  }
  for (int m = 0; m < M; ++m) {
    const float v = warp_sum(acc[m]);
    if (lane == 0) c[(size_t(m) * G + g) * R + r] = f32_to_bf16(v);
  }
}

int blocks_for(int cols) { return (cols + WAVES - 1) / WAVES; }

}  // namespace

void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8: M > 8 or K % 512");
  gemv_w8a8_kernel<<<blocks_for(N), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
  UH_CHECK_LAUNCH();
}
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16: M > 8 or K % 512");
  gemv_bf16_kernel<bf16><<<blocks_for(N), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K);
  UH_CHECK_LAUNCH();
}
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16_f32out: M > 8 or K % 512");
  gemv_bf16_kernel<float><<<blocks_for(N), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K);
  UH_CHECK_LAUNCH();
}
void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_grouped_fp8w: M > 8 or K % 512");
  gemv_grouped_fp8w_kernel<<<blocks_for(G * R), NT, 0, st>>>(a, w, sw, c, M, G, R, K);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
