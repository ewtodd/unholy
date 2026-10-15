// WMMA GEMMs for the dense projections (fast path; references in kernels/ref_gemm.hip).
//
// gemm_w8a8_wmma (gfx12): fp8 x fp8 WMMA. The reference sums each 128-wide K block in fp32 and then
// scales it by sa[m][kb] * sw[n/128][kb]; here each K block accumulates into a raw WMMA accumulator
// that is folded into the output accumulator with the same scale product at the block end.
//
// gemm_bf16_wmma (gfx11 + gfx12): bf16 WMMA with fp32 accumulators, for every GEMM whose operands are
// bf16-valued: weights bf16 or fp8 x 128x128 e8m0 block scale (exact in bf16), activations bf16. The
// reference's fp32 linears (compressor, router, head) take x.float() of a bf16 tensor, so their
// products are exact in bf16 WMMA too; only the fp32 summation order differs.
//
// Tiles: 64 x 128 per 256-thread block, wave w owns columns [16w, 16w + 16) of the tile and all 4
// row fragments. Fragment layouts: see kernels/exl3_moe.hip.
#include "kernels/dev.hip.hpp"
#include "kernels/gemm_wmma.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));

constexpr int BM = 64, BN = 128, NT = 256;

__device__ __forceinline__ int c_row(int i, int e, int lane) {
#if defined(UH_GFX12)
  return i * 16 + (lane >> 4) * 8 + e;
#else
  return i * 16 + 2 * e + (lane >> 4);
#endif
}

// ---- fp8 x fp8 ----
template <typename TOut>
__global__ __launch_bounds__(NT) void w8a8_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                                                  const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw,
                                                  TOut* __restrict__ C, int ldc, int M, int N, int K) {
#if defined(UH_GFX12)
  constexpr int KB = 128, STR = KB + 16;
  __shared__ alignas(16) uint8_t sA[BM * STR];
  __shared__ alignas(16) uint8_t sW[BN * STR];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, nkb = K / KB;
  float8_t raw[4], acc[4];
  for (int i = 0; i < 4; ++i)
    for (int e = 0; e < 8; ++e) raw[i][e] = acc[i][e] = 0.0f;
  for (int kb = 0; kb < nkb; ++kb) {
    const int k0 = kb * KB;
    for (int idx = tid * 16; idx < BM * KB; idx += NT * 16) {
      const int r = idx / KB, c = idx % KB;
      uint4 v = {0, 0, 0, 0};
      if (m0 + r < M) v = *reinterpret_cast<const uint4*>(A + size_t(m0 + r) * K + k0 + c);
      *reinterpret_cast<uint4*>(&sA[r * STR + c]) = v;
    }
    for (int idx = tid * 16; idx < BN * KB; idx += NT * 16) {
      const int r = idx / KB, c = idx % KB;
      uint4 v = {0, 0, 0, 0};
      if (n0 + r < N) v = *reinterpret_cast<const uint4*>(W + size_t(n0 + r) * K + k0 + c);
      *reinterpret_cast<uint4*>(&sW[r * STR + c]) = v;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < KB; kk += 16) {
      const int ko = kk + (lane >> 4) * 8;
      const int2_t bf = *reinterpret_cast<const int2_t*>(&sW[(wave * 16 + (lane & 15)) * STR + ko]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int2_t af = *reinterpret_cast<const int2_t*>(&sA[(i * 16 + (lane & 15)) * STR + ko]);
        raw[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, bf, raw[i]);
      }
    }
    __syncthreads();
    const int n = n0 + wave * 16 + (lane & 15);
    const float wsc = n < N ? e8m0_to_f32(sw[size_t(n / 128) * nkb + kb]) : 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + c_row(i, e, lane);
        const float asc = m < M ? e8m0_to_f32(sa[size_t(m) * nkb + kb]) : 0.0f;
        acc[i][e] += raw[i][e] * (asc * wsc);
        raw[i][e] = 0.0f;
      }
  }
  const int n = n0 + wave * 16 + (lane & 15);
  if (n < N)
    for (int i = 0; i < 4; ++i)
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + c_row(i, e, lane);
        if (m < M) {
          if constexpr (sizeof(TOut) == 2) C[size_t(m) * ldc + n] = f32_to_bf16(acc[i][e]);
          else C[size_t(m) * ldc + n] = acc[i][e];
        }
      }
#endif
}

// ---- bf16 x bf16 (weights bf16, or fp8 + 128x128 scales) ----
// Grouped (blockIdx.z = g): A row m of group g at A + (m * a_rows_stride + g * a_group_stride), W rows
// g * N + n, C at C + m * ldc + g * c_group_stride.
template <bool kFp8W, typename TOut>
__global__ __launch_bounds__(NT) void bf16_kernel(const bf16* __restrict__ A, int lda, size_t a_group,
                                                  const void* __restrict__ Wv, const uint8_t* __restrict__ sw,
                                                  TOut* __restrict__ C, int ldc, size_t c_group, int M, int N,
                                                  int K) {
#if defined(UH_GFX12) || defined(UH_GFX11)
  constexpr int KS = 32, STR = KS + 8;
  __shared__ alignas(16) uint16_t sA[BM * STR];
  __shared__ alignas(16) uint16_t sW[BN * STR];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, g = blockIdx.z;
  const bf16* Ag = A + size_t(g) * a_group;
  const size_t wrow0 = size_t(g) * N;
  float8_t acc[4];
  for (int i = 0; i < 4; ++i)
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += KS) {
    {  // A: 64 x 32, 8 per thread
      const int r = tid >> 2, c = (tid & 3) * 8;
      uint4 v = {0, 0, 0, 0};
      if (m0 + r < M) v = *reinterpret_cast<const uint4*>(Ag + size_t(m0 + r) * lda + k0 + c);
      *reinterpret_cast<uint4*>(&sA[r * STR + c]) = v;
    }
    for (int idx = tid; idx < BN * 4; idx += NT) {  // W: 128 x 32, 8 per item
      const int r = idx >> 2, c = (idx & 3) * 8;
      // Kept in registers (a local array here was spilled to scratch).
      uint4 vals = {0, 0, 0, 0};
      if (n0 + r < N) {
        const size_t row = wrow0 + n0 + r;
        if constexpr (kFp8W) {
          const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(Wv) + row * K + k0 + c);
          const float s = e8m0_to_f32(sw[(row / 128) * (K / 128) + (k0 + c) / 128]);
          float f[8];
          e4m3x4_to_f32(raw.x, f);
          e4m3x4_to_f32(raw.y, f + 4);
          const auto pair = [&](int j) {
            return uint32_t(f32_to_bf16(f[j] * s)) | (uint32_t(f32_to_bf16(f[j + 1] * s)) << 16);
          };
          vals = uint4{pair(0), pair(2), pair(4), pair(6)};
        } else {
          vals = *reinterpret_cast<const uint4*>(static_cast<const bf16*>(Wv) + row * K + k0 + c);
        }
      }
      *reinterpret_cast<uint4*>(&sW[r * STR + c]) = vals;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < KS; kk += 16) {
#if defined(UH_GFX12)
      const int ko = kk + (lane >> 4) * 8;
      const bf16x8_t bfv = *reinterpret_cast<const bf16x8_t*>(&sW[(wave * 16 + (lane & 15)) * STR + ko]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const bf16x8_t afv = *reinterpret_cast<const bf16x8_t*>(&sA[(i * 16 + (lane & 15)) * STR + ko]);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(afv, bfv, acc[i]);
      }
#else
      const uint16_t* pb = &sW[(wave * 16 + (lane & 15)) * STR + kk];
      const bf16x8_t b0 = *reinterpret_cast<const bf16x8_t*>(pb), b1 = *reinterpret_cast<const bf16x8_t*>(pb + 8);
      const bf16x16_t bfv = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint16_t* pa = &sA[(i * 16 + (lane & 15)) * STR + kk];
        const bf16x8_t a0 = *reinterpret_cast<const bf16x8_t*>(pa), a1 = *reinterpret_cast<const bf16x8_t*>(pa + 8);
        const bf16x16_t afv = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afv, bfv, acc[i]);
      }
#endif
    }
    __syncthreads();
  }
  const int n = n0 + wave * 16 + (lane & 15);
  TOut* Cg = C + size_t(g) * c_group;
  if (n < N)
    for (int i = 0; i < 4; ++i)
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + c_row(i, e, lane);
        if (m < M) {
          if constexpr (sizeof(TOut) == 2) Cg[size_t(m) * ldc + n] = f32_to_bf16(acc[i][e]);
          else Cg[size_t(m) * ldc + n] = acc[i][e];
        }
      }
#endif
}

dim3 grid_for(int M, int N, int groups = 1) { return dim3((N + BN - 1) / BN, (M + BM - 1) / BM, groups); }

}  // namespace

void gemm_w8a8_wmma(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc,
                    int M, int N, int K, hipStream_t st) {
  if (K % 128) throw std::runtime_error("gemm_w8a8_wmma: K % 128");
  w8a8_kernel<bf16><<<grid_for(M, N), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
  UH_CHECK_LAUNCH();
}

void gemm_bf16_wmma(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (K % 32) throw std::runtime_error("gemm_bf16_wmma: K % 32");
  bf16_kernel<false, bf16><<<grid_for(M, N), NT, 0, st>>>(a, lda, 0, w, nullptr, c, ldc, 0, M, N, K);
  UH_CHECK_LAUNCH();
}

void gemm_bf16_f32out_wmma(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K,
                           hipStream_t st) {
  if (K % 32) throw std::runtime_error("gemm_bf16_f32out_wmma: K % 32");
  bf16_kernel<false, float><<<grid_for(M, N), NT, 0, st>>>(a, lda, 0, w, nullptr, c, ldc, 0, M, N, K);
  UH_CHECK_LAUNCH();
}

void gemm_grouped_fp8w_wmma(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                            hipStream_t st) {
  if (K % 128) throw std::runtime_error("gemm_grouped_fp8w_wmma: K % 128");
  // a [M, G, K]: row m of group g at a + m*G*K + g*K; c [M, G, R]: row m at c + m*G*R + g*R.
  bf16_kernel<true, bf16><<<grid_for(M, R, G), NT, 0, st>>>(a, G * K, size_t(K), w, sw, c, G * R, size_t(R), M, R,
                                                            K);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
