// EXL3 grouped MoE kernels: every routed expert of a layer in one launch per projection.
//
// Pairs (token, expert) arrive expert-major (ascending expert, contiguous rows per expert). The input
// transform fuses the A8 fake quantization the experts were trained with, * suh and the 128-point
// Hadamard, writing fp16; the GEMM decodes the trellis tiles of the current K-slab into LDS (fp16,
// [n][k]) and accumulates with fp16 WMMA (fp32 accumulators); its epilogue applies the output
// Hadamard * svh over each 128-column segment and writes bf16. Format: src/model/exl3.hpp.
//
// Fragment layouts (wave32): gfx11 A/B lane l = row/col l%16 with all 16 k (halves replicated), C
// lane l holds C[2i + l/16][l%16]; gfx12 A/B lane l = row/col l%16 with k (l/16)*8..+7, C lane l
// holds C[(l/16)*8 + i][l%16].
#include <hip/hip_fp16.h>

#include "kernels/dev.hip.hpp"
#include "kernels/exl3_moe.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));

// 16 consecutive halves from a 16-byte-aligned LDS address (two b128 loads).
__device__ __forceinline__ half16_t load16(const _Float16* p) {
  const half8_t lo = *reinterpret_cast<const half8_t*>(p);
  const half8_t hi = *reinterpret_cast<const half8_t*>(p + 8);
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

// Trellis step -> row-major tile position (tensor-core permutation).
__device__ __forceinline__ int tc_perm(int t) {
  const int lane = t >> 3, j = t & 7;
  const int r0 = (lane & 3) * 2, c0 = lane >> 2;
  const int r = r0 + (j & 1) + ((j & 2) ? 8 : 0);
  const int c = c0 + ((j & 4) ? 8 : 0);
  return r * 16 + c;
}

__device__ __forceinline__ _Float16 decode_mul1(uint32_t state) {
  const uint32_t x = state * 0x83DCD12Du;
  const uint32_t sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24) + 0x6400u;
  const __half h = __hfma(__ushort_as_half(uint16_t(sum)), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
  return _Float16(__half2float(h));  // exact (fp16 -> fp32 -> fp16)
}

// 16-bit window ending at bit (t+1)*BITS of the circular MSB-first stream in `w` (8*BITS words).
template <int BITS>
__device__ __forceinline__ uint32_t window16(const uint32_t* w, int t) {
  constexpr int LEN = 256 * BITS, NW = 8 * BITS;
  const int start = ((t + 1) * BITS - 16 + LEN) % LEN;
  const int w0 = start >> 5, b0 = start & 31;
  const uint64_t two = (uint64_t(w[w0]) << 32) | w[(w0 + 1) % NW];
  return uint32_t(two >> (48 - b0)) & 0xffffu;
}

// One block (128 threads) per (row, 128-segment).
__device__ __forceinline__ void fwht128_lds(float* b, int i) {
  for (int h = 1; h < 128; h <<= 1) {
    const float nv = (i & h) ? b[i - h] - b[i] : b[i] + b[i + h];
    __syncthreads();
    b[i] = nv;
    __syncthreads();
  }
}

__device__ __forceinline__ float fake_quant_fp8_block(float v, float* red) {
  // amax over the 128-block (4 waves): warp max then LDS.
  float a = warp_max(fabsf(v));
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (lane == 0) red[wave] = a;
  __syncthreads();
  a = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();
  a = fmaxf(a, 1e-4f);
  const float sc = e8m0_to_f32(pow2_ceil_e8m0(a * (1.0f / 448.0f)));
  return round_bf16(e4m3_to_f32(f32_to_e4m3(fminf(fmaxf(v / sc, -448.0f), 448.0f))) * sc);
}

__global__ __launch_bounds__(128) void xform_in_kernel(const bf16* x, int ldx, const int32_t* row_expert,
                                                       const uint16_t* const* suh_tab, uint16_t* xh, int K,
                                                       bool fq) {
  __shared__ float b[128];
  __shared__ float red[4];
  const int r = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  const uint16_t* suh = suh_tab[row_expert[r]];
  if (!suh) return;  // expert not on this device: its rows are never read (uniform per block)
  float v = bf16_to_f32(x[size_t(r) * ldx + seg * 128 + i]);
  if (fq) v = fake_quant_fp8_block(v, red);
  b[i] = v * __half2float(__ushort_as_half(suh[seg * 128 + i]));
  __syncthreads();
  fwht128_lds(b, i);
  xh[size_t(r) * K + seg * 128 + i] = __half_as_ushort(__float2half(b[i] * 0.08838834764831845f));
}

__global__ __launch_bounds__(128) void swiglu_xform_kernel(const bf16* g, const bf16* u, const float* w,
                                                           const int32_t* row_expert, const uint16_t* const* suh_tab,
                                                           uint16_t* xh, int I, float limit) {
  __shared__ float b[128];
  __shared__ float red[4];
  const int r = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  const uint16_t* suh = suh_tab[row_expert[r]];
  if (!suh) return;
  const size_t idx = size_t(r) * I + seg * 128 + i;
  float gv = bf16_to_f32(g[idx]), uv = bf16_to_f32(u[idx]);
  if (limit > 0.0f) {
    uv = fminf(fmaxf(uv, -limit), limit);
    gv = fminf(gv, limit);
  }
  float hv = (gv / (1.0f + expf(-gv))) * uv;
  if (w) hv = w[r] * hv;
  hv = round_bf16(hv);  // the reference's swiglu output is a bf16 tensor
  hv = fake_quant_fp8_block(hv, red);
  b[i] = hv * __half2float(__ushort_as_half(suh[seg * 128 + i]));
  __syncthreads();
  fwht128_lds(b, i);
  xh[idx] = __half_as_ushort(__float2half(b[i] * 0.08838834764831845f));
}

constexpr int BM = 64, BN = 128, KS = 32, NT = 256;

template <int BITS>
__global__ __launch_bounds__(NT) void exl3_gemm_kernel(const uint16_t* __restrict__ xh, int K,
                                                       const Exl3Tile* __restrict__ tiles,
                                                       const uint16_t* const* __restrict__ trellis_tab,
                                                       const uint16_t* const* __restrict__ svh_tab, bf16* out,
                                                       int ldo, int N) {
  constexpr int NW = 8 * BITS;           // uint32 words per 16x16 trellis tile
  constexpr int NTILE = (BN / 16) * (KS / 16);  // trellis tiles per K-slab
  constexpr int APAD = KS + 8;
  __shared__ alignas(16) _Float16 sA[BM][APAD];
  __shared__ alignas(16) _Float16 sB[BN][APAD];  // [n][k]
  __shared__ uint32_t sT[NTILE][NW];
  __shared__ float sC[BM][BN + 4];
  const Exl3Tile tile = tiles[blockIdx.y];
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const uint32_t* trellis = reinterpret_cast<const uint32_t*>(trellis_tab[tile.expert]);
  const int tiles_n = N / 16;

  float8_t acc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += KS) {
    {  // A: 64 rows x 32 halves, 8 halves per thread
      const int r = tid >> 2, c = (tid & 3) * 8;
      uint4 v = {0, 0, 0, 0};
      if (r < tile.rows) v = *reinterpret_cast<const uint4*>(xh + size_t(tile.row0 + r) * K + k0 + c);
      *reinterpret_cast<uint4*>(&sA[r][c]) = v;
    }
    for (int i = tid; i < NTILE * NW; i += NT) {  // trellis words of the slab
      const int j = i / NW, wd = i % NW;
      const int kt = j / (BN / 16), nt = j % (BN / 16);
      const size_t ti = size_t(k0 / 16 + kt) * tiles_n + (n0 / 16 + nt);
      sT[j][wd] = trellis[ti * NW + wd];
    }
    __syncthreads();
    // Decode: NTILE x 256 weights; thread -> tile tid/16, 16 consecutive trellis steps.
    {
      const int j = tid >> 4, t0 = (tid & 15) * 16;
      const int kt = j / (BN / 16), nt = j % (BN / 16);
#pragma unroll 4
      for (int s = 0; s < 16; ++s) {
        const int t = t0 + s;
        const int pos = tc_perm(t);
        sB[nt * 16 + (pos & 15)][kt * 16 + (pos >> 4)] = decode_mul1(window16<BITS>(sT[j], t));
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < KS; kk += 16) {
#if defined(UH_GFX12)
      const half8_t bf = *reinterpret_cast<const half8_t*>(&sB[wave * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const half8_t af = *reinterpret_cast<const half8_t*>(&sA[i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, acc[i]);
      }
#elif defined(UH_GFX11)
      const half16_t bf = load16(&sB[wave * 16 + (lane & 15)][kk]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const half16_t af = load16(&sA[i * 16 + (lane & 15)][kk]);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[i]);
      }
#endif
    }
    __syncthreads();
  }
  // Accumulators -> sC [row][col].
  const int col = wave * 16 + (lane & 15);
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
#if defined(UH_GFX12)
      const int row = i * 16 + (lane >> 4) * 8 + e;
#else
      const int row = i * 16 + 2 * e + (lane >> 4);
#endif
      sC[row][col] = acc[i][e];
    }
  __syncthreads();
  // Output Hadamard over each row's 128 columns: 64 rows x 64 butterflies per stage.
  for (int h = 1; h < 128; h <<= 1) {
    for (int p = tid; p < BM * 64; p += NT) {
      const int row = p >> 6, q = p & 63;
      const int a = (q / h) * 2 * h + (q % h);
      const float x0 = sC[row][a], x1 = sC[row][a + h];
      sC[row][a] = x0 + x1;
      sC[row][a + h] = x0 - x1;
    }
    __syncthreads();
  }
  const uint16_t* svh = svh_tab[tile.expert];
  for (int p = tid; p < BM * BN; p += NT) {
    const int row = p / BN, c = p % BN;
    if (row >= tile.rows) continue;
    const float v = sC[row][c] * 0.08838834764831845f * __half2float(__ushort_as_half(svh[n0 + c]));
    out[size_t(tile.row0 + row) * ldo + n0 + c] = f32_to_bf16(v);
  }
}

}  // namespace

void exl3_xform_in(const bf16* x, int ldx, const int32_t* row_expert, const uint16_t* const* suh_tab, uint16_t* xh,
                   int P, int K, bool fake_quant, hipStream_t st) {
  if (P <= 0) return;
  xform_in_kernel<<<dim3(K / 128, P), 128, 0, st>>>(x, ldx, row_expert, suh_tab, xh, K, fake_quant);
  UH_CHECK_LAUNCH();
}

void exl3_swiglu_xform(const bf16* g, const bf16* u, const float* w, const int32_t* row_expert,
                       const uint16_t* const* suh_tab, uint16_t* xh, int P, int I, float limit, hipStream_t st) {
  if (P <= 0) return;
  swiglu_xform_kernel<<<dim3(I / 128, P), 128, 0, st>>>(g, u, w, row_expert, suh_tab, xh, I, limit);
  UH_CHECK_LAUNCH();
}

void exl3_grouped_gemm(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles,
                       const uint16_t* const* trellis_tab, const uint16_t* const* svh_tab, int bits, bf16* out,
                       int ldo, int N, hipStream_t st) {
  if (ntiles <= 0) return;
  if (K % KS || N % BN) throw std::runtime_error("exl3_grouped_gemm: K/N not multiples of 32/128");
  const dim3 grid(N / BN, ntiles);
  switch (bits) {
    case 2: exl3_gemm_kernel<2><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N); break;
    case 3: exl3_gemm_kernel<3><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N); break;
    case 4: exl3_gemm_kernel<4><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N); break;
    case 5: exl3_gemm_kernel<5><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N); break;
    case 6: exl3_gemm_kernel<6><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N); break;
    default: throw std::runtime_error("exl3_grouped_gemm: unsupported bit width");
  }
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
