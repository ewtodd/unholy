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
//
// Steps of a few rows (multi-stream verify steps: 9-64) take skinny kernels with every output's
// arithmetic unchanged and far more workgroups streaming the weights (the 64 x 128 tiles gave wq_a's
// 1024 columns 8 blocks): the fp8 one computes a 16-column tile's 128-wide K blocks on 8 waves at once
// (each block's raw chain starts from zero, so they are independent) and folds them in order; the bf16
// one runs a wave per (16 columns, 16 rows). unholy-gemv-test checks them bit for bit against the tiles.
#include "kernels/gemm_wmma.hpp"
#include "kernels/glm_fast.hpp"

#include <cstdlib>
#include <string>

#include "kernels/dev.hip.hpp"
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

// ---- fp8 x fp8, M <= 16 * RF (a decode step's few rows) ----
// w8a8_kernel's arithmetic per element -- each 128-wide K block's raw WMMA chain from zero, then
// acc += raw * (asc * wsc) in block order -- on a 16-column tile per block, so N / 16 blocks instead of
// N / 128 stream the weights. The raw chains of a tile's K blocks are independent: wave w runs blocks w,
// w + 8, ... (fragments loaded straight from global memory, as the 64 x 128 tile stages them: zero rows
// past M), parks each round of 8 in LDS, and the block folds them into acc in block order.
template <int RF>
__global__ __launch_bounds__(NT) void w8a8_skinny_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                                                         const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw,
                                                         bf16* __restrict__ C, int ldc, int M, int N, int K) {
#if defined(UH_GFX12)
  __shared__ float sraw[8][RF][256];  // [wave][fragment][e * 32 + lane]
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = blockIdx.x * 16, nkb = K / 128;
  // Compute: the B fragment's column (rows past N read a valid row; their outputs are never stored).
  const uint8_t* wl = W + size_t(min(n0 + (lane & 15), N - 1)) * K + (lane >> 4) * 8;
  // Fold: this thread's element e = wave of fragment i's lane `lane` layout.
  const int fe = wave, n = n0 + (lane & 15);
  float acc[RF];
#pragma unroll
  for (int i = 0; i < RF; ++i) acc[i] = 0.0f;
  int2_t bf[8];  // this wave's B fragments of block `kb`, loaded a round ahead
  if (wave < nkb)
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) bf[kk] = *reinterpret_cast<const int2_t*>(wl + wave * 128 + kk * 16);
  for (int r0 = 0; r0 < nkb; r0 += 8) {
    const int kb = r0 + wave;
    if (kb < nkb) {
      const int k0 = kb * 128;
      float8_t raw[RF];
#pragma unroll
      for (int i = 0; i < RF; ++i) {
        const int m = i * 16 + (lane & 15);
        const uint8_t* al = A + size_t(m) * K + k0 + (lane >> 4) * 8;
        for (int e = 0; e < 8; ++e) raw[i][e] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          int2_t af = {0, 0};
          if (m < M) af = *reinterpret_cast<const int2_t*>(al + kk * 16);
          raw[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, bf[kk], raw[i]);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) sraw[wave][i][e * 32 + lane] = raw[i][e];
      }
      if (kb + 8 < nkb)  // the next round's, in flight during the fold
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) bf[kk] = *reinterpret_cast<const int2_t*>(wl + (kb + 8) * 128 + kk * 16);
    }
    __syncthreads();
    for (int j = 0; j < 8 && r0 + j < nkb; ++j) {
      const int kb = r0 + j;
      const float wsc = n < N ? e8m0_to_f32(sw[size_t(n / 128) * nkb + kb]) : 0.0f;
#pragma unroll
      for (int i = 0; i < RF; ++i) {
        const int m = c_row(i, fe, lane);
        const float asc = m < M ? e8m0_to_f32(sa[size_t(m) * nkb + kb]) : 0.0f;
        acc[i] += sraw[j][i][fe * 32 + lane] * (asc * wsc);
      }
    }
    __syncthreads();
  }
  if (n < N)
#pragma unroll
    for (int i = 0; i < RF; ++i) {
      const int m = c_row(i, fe, lane);
      if (m < M) C[size_t(m) * ldc + n] = f32_to_bf16(acc[i]);
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

// ---- bf16 x bf16, M <= 64, gfx12 ----
// bf16_kernel's arithmetic per element (one WMMA chain over all of K, 16 at a time, from zero; fp8
// weights dequantized as there), a wave per (16-column tile, 16-row fragment): its fragments loaded
// straight from global memory (zero rows past M). The waves of one column tile are neighbours in a block,
// so the weights come from DRAM once; N / 16 x fragments chains run at once instead of N / 128 blocks.
template <bool kFp8W, typename TOut>
__global__ __launch_bounds__(128) void bf16_skinny_kernel(const bf16* __restrict__ A, int lda, size_t a_group,
                                                          const void* __restrict__ Wv, const uint8_t* __restrict__ sw,
                                                          TOut* __restrict__ C, int ldc, size_t c_group, int M, int N,
                                                          int K, int frags) {
#if defined(UH_GFX12)
  const int lane = threadIdx.x & 31, g = blockIdx.z;
  const int p = int(blockIdx.x) * 4 + (threadIdx.x >> 5), fi = p % frags, nw = p / frags * 16;
  if (nw >= N) return;
  const int n = nw + (lane & 15), ko = (lane >> 4) * 8, m = fi * 16 + (lane & 15);
  const size_t row = size_t(g) * N + min(n, N - 1);  // columns past N read a valid row, never stored
  const bf16* Am = A + size_t(g) * a_group + size_t(m) * lda + ko;
  float8_t acc;
  for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
#pragma unroll 8
  for (int k0 = 0; k0 < K; k0 += 16) {
    bf16x8_t bfv;
    if constexpr (kFp8W) {
      const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(Wv) + row * K + k0 + ko);
      const float s = e8m0_to_f32(sw[(row / 128) * (K / 128) + (k0 + ko) / 128]);
      float f[8];
      e4m3x4_to_f32(raw.x, f);
      e4m3x4_to_f32(raw.y, f + 4);
      const auto pair = [&](int j) {
        return uint32_t(f32_to_bf16(f[j] * s)) | (uint32_t(f32_to_bf16(f[j + 1] * s)) << 16);
      };
      const uint4 v = uint4{pair(0), pair(2), pair(4), pair(6)};
      bfv = *reinterpret_cast<const bf16x8_t*>(&v);
    } else {
      bfv = *reinterpret_cast<const bf16x8_t*>(static_cast<const bf16*>(Wv) + row * K + k0 + ko);
    }
    uint4 av = {0, 0, 0, 0};
    if (m < M) av = *reinterpret_cast<const uint4*>(Am + k0);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(*reinterpret_cast<const bf16x8_t*>(&av), bfv, acc);
  }
  if (n >= N) return;
  TOut* Cg = C + size_t(g) * c_group;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int r = c_row(fi, e, lane);
    if (r < M) {
      if constexpr (sizeof(TOut) == 2) Cg[size_t(r) * ldc + n] = f32_to_bf16(acc[e]);
      else Cg[size_t(r) * ldc + n] = acc[e];
    }
  }
#endif
}

dim3 grid_for(int M, int N, int groups = 1) { return dim3((N + BN - 1) / BN, (M + BM - 1) / BM, groups); }

// gfx12 on the current device (the skinny kernels are gfx12-only).
bool dev_gfx12() {
  static int by_dev[64] = {};
  int dev = 0;
  UH_CHECK(hipGetDevice(&dev));
  if (!by_dev[dev]) {
    hipDeviceProp_t p{};
    UH_CHECK(hipGetDeviceProperties(&p, dev));
    by_dev[dev] = std::string(p.gcnArchName).rfind("gfx12", 0) == 0 ? 2 : 1;
  }
  return by_dev[dev] == 2;
}

// bf16_kernel or, on gfx12, bf16_skinny_kernel for M <= 64 (fp8 weights: M <= 32, past which every
// fragment's wave dequantizing the same weights loses to the tiles) and N <= 16384 (not the vocabulary
// head, where the tiles are as fast).
template <bool kFp8W, typename TOut>
void launch_bf16(const bf16* a, int lda, size_t a_group, const void* w, const uint8_t* sw, TOut* c, int ldc,
                 size_t c_group, int M, int N, int K, int G, bool skinny, hipStream_t st) {
  if (skinny && M <= (kFp8W ? 32 : 64) && N <= 16384 && K % 128 == 0 && dev_gfx12()) {
    const int frags = (M + 15) / 16, pairs = (N + 15) / 16 * frags;
    bf16_skinny_kernel<kFp8W, TOut><<<dim3((pairs + 3) / 4, 1, G), 128, 0, st>>>(a, lda, a_group, w, sw, c, ldc,
                                                                               c_group, M, N, K, frags);
  } else {
    bf16_kernel<kFp8W, TOut><<<grid_for(M, N, G), NT, 0, st>>>(a, lda, a_group, w, sw, c, ldc, c_group, M, N, K);
  }
  UH_CHECK_LAUNCH();
}

}  // namespace

namespace {
int g_skinny = -1;
bool use_skinny(int M) {
  static const bool env = [] {
    const char* v = std::getenv("UNHOLY_GEMM_SKINNY");
    return !v || std::string(v) != "0";
  }();
  return M <= 64 && (g_skinny < 0 ? env : g_skinny > 0);
}
}  // namespace

void gemm_force_skinny(int on) { g_skinny = on; }

void gemm_w8a8_wmma(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc,
                    int M, int N, int K, hipStream_t st) {
  if (K % 128) throw std::runtime_error("gemm_w8a8_wmma: K % 128");
  if (use_skinny(M)) {
    const dim3 grid((N + 15) / 16);
    if (M <= 16) w8a8_skinny_kernel<1><<<grid, NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
    else if (M <= 32) w8a8_skinny_kernel<2><<<grid, NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
    else if (M <= 48) w8a8_skinny_kernel<3><<<grid, NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
    else w8a8_skinny_kernel<4><<<grid, NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
  } else {
    w8a8_kernel<bf16><<<grid_for(M, N), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K);
  }
  UH_CHECK_LAUNCH();
}

void gemm_bf16_wmma(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  if (K % 32) throw std::runtime_error("gemm_bf16_wmma: K % 32");
  launch_bf16<false, bf16>(a, lda, 0, w, nullptr, c, ldc, 0, M, N, K, 1, use_skinny(M), st);
}

void gemm_bf16_f32out_wmma(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K,
                           hipStream_t st) {
  if (K % 32) throw std::runtime_error("gemm_bf16_f32out_wmma: K % 32");
  launch_bf16<false, float>(a, lda, 0, w, nullptr, c, ldc, 0, M, N, K, 1, use_skinny(M), st);
}

void gemm_grouped_fp8w_wmma(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                            hipStream_t st) {
  if (K % 128) throw std::runtime_error("gemm_grouped_fp8w_wmma: K % 128");
  // a [M, G, K]: row m of group g at a + m*G*K + g*K; c [M, G, R]: row m at c + m*G*R + g*R.
  launch_bf16<true, bf16>(a, G * K, size_t(K), w, sw, c, G * R, size_t(R), M, R, K, G, use_skinny(M), st);
}

void gemm_grouped_bf16_wmma(const bf16* a, const bf16* w, bf16* c, int M, int G, int R, int K, hipStream_t st) {
  if (K % 32) throw std::runtime_error("gemm_grouped_bf16_wmma: K % 32");
  launch_bf16<false, bf16>(a, G * K, size_t(K), w, nullptr, c, G * R, size_t(R), M, R, K, G, use_skinny(M), st);
}

}  // namespace unholy::kernels
