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

#include <algorithm>

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

// Rows [*row0 (0 if null), *row1 (P if null)), gridDim.y of them at a time: a prefill step's few-thousand-row grids
// would mostly be blocks that only find their row's expert elsewhere.
__global__ __launch_bounds__(128) void xform_in_kernel(const bf16* x, int ldx, const int32_t* row_expert,
                                                       const uint16_t* const* suh_tab, uint16_t* xh, int K,
                                                       bool fq, int P, const int* row0, const int* row1,
                                                       const int32_t* x_rows) {
  __shared__ float b[128];
  __shared__ float red[4];
  const int seg = blockIdx.x, i = threadIdx.x, r1 = row1 ? *row1 : P;
  for (int r = (row0 ? *row0 : 0) + blockIdx.y; r < r1; r += gridDim.y) {
    const uint16_t* suh = suh_tab[row_expert[r]];
    if (!suh) continue;  // expert not on this device: its rows are never read (uniform per block)
    float v = bf16_to_f32(x[size_t(x_rows ? x_rows[r] : r) * ldx + seg * 128 + i]);
    if (fq) v = fake_quant_fp8_block(v, red);
    b[i] = v * __half2float(__ushort_as_half(suh[seg * 128 + i]));
    __syncthreads();
    fwht128_lds(b, i);
    xh[size_t(r) * K + seg * 128 + i] = __half_as_ushort(__float2half(b[i] * 0.08838834764831845f));
    __syncthreads();
  }
}

template <bool FQ>
__global__ __launch_bounds__(128) void swiglu_xform_kernel(const bf16* g, const bf16* u, const float* w,
                                                           const int32_t* row_expert, const uint16_t* const* suh_tab,
                                                           uint16_t* xh, int I, float limit, int P, const int* row0,
                                                           const int* row1) {
  __shared__ float b[128];
  __shared__ float red[4];
  const int seg = blockIdx.x, i = threadIdx.x, r1 = row1 ? *row1 : P;
  for (int r = (row0 ? *row0 : 0) + blockIdx.y; r < r1; r += gridDim.y) {
    const uint16_t* suh = suh_tab[row_expert[r]];
    if (!suh) continue;
    const size_t idx = size_t(r) * I + seg * 128 + i;
    float gv = bf16_to_f32(g[idx]), uv = bf16_to_f32(u[idx]);
    if (limit > 0.0f) {
      uv = fminf(fmaxf(uv, -limit), limit);
      gv = fminf(gv, limit);
    }
    float hv = (gv / (1.0f + expf(-gv))) * uv;
    if (w) hv = w[r] * hv;
    hv = round_bf16(hv);  // the reference's swiglu output is a bf16 tensor
    if constexpr (FQ) hv = fake_quant_fp8_block(hv, red);
    b[i] = hv * __half2float(__ushort_as_half(suh[seg * 128 + i]));
    __syncthreads();
    fwht128_lds(b, i);
    xh[idx] = __half_as_ushort(__float2half(b[i] * 0.08838834764831845f));
    __syncthreads();
  }
}

constexpr int BM = 64, BN = 128, KS = 32, NT = 256;

template <int BITS>
__global__ __launch_bounds__(NT) void exl3_gemm_kernel(const uint16_t* __restrict__ xh, int K,
                                                       const Exl3Tile* __restrict__ tiles,
                                                       const uint16_t* const* __restrict__ trellis_tab,
                                                       const uint16_t* const* __restrict__ svh_tab, bf16* out,
                                                       int ldo, int N, const int* ntiles_dev) {
  constexpr int NW = 8 * BITS;           // uint32 words per 16x16 trellis tile
  constexpr int NTILE = (BN / 16) * (KS / 16);  // trellis tiles per K-slab
  constexpr int APAD = KS + 8;
  __shared__ alignas(16) _Float16 sA[BM][APAD];
  __shared__ alignas(16) _Float16 sB[BN][APAD];  // [n][k]
  __shared__ uint32_t sT[NTILE][NW];
  __shared__ float sC[BM][BN + 4];
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;  // grid sized for the maximum
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


// ---- v2: 128-row tiles, decode once per (tile, 16-column slice), wave-private B ----------------------
//
// Block = one tile (<= 128 rows of one expert) x 128 output columns; wave w owns columns
// [n0 + 16w, n0 + 16w + 16) and all rows, so it decodes exactly the trellis tiles it multiplies
// (16 columns x 64 k per slab) into its own LDS slice: no block barrier for B. A lane decodes 8
// consecutive trellis steps from three words with funnel shifts; the mul1 byte sum is one udot4.
// The accumulation runs over the same 16-wide k-steps in the same order as v1, and the epilogue is
// the same, so the results are identical.
constexpr int V2_BM = 128, V2_BN = 128, V2_KS = 64, V2_NT = 256, V2_PAD = 8;

// fp16 bits of mul1(state) for two states: x = s * 0x83DCD12D; byte sum + 0x6400 as fp16; h * 0x1eee + 0xc931.
__device__ __forceinline__ uint32_t decode_mul1_x2(uint32_t s0, uint32_t s1) {
  const uint32_t a = __builtin_amdgcn_udot4(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u, false);
  const uint32_t b = __builtin_amdgcn_udot4(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u, false);
  const __half2 h = __halves2half2(__ushort_as_half(uint16_t(a)), __ushort_as_half(uint16_t(b)));
  const __half2 r = __hfma2(h, __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee)),
                            __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931)));
  return uint32_t(__half_as_ushort(__low2half(r))) | (uint32_t(__half_as_ushort(__high2half(r))) << 16);
}

// Lane L's 8 trellis steps (8L .. 8L + 7) of one tile (`w`: its 8*BITS words), as 4 packed fp16 pairs:
// p[0] = (k r0, r0 + 1) at column c0, p[1] = (r0 + 8, r0 + 9) at c0, p[2], p[3] the same at c0 + 8, with
// r0 = (L & 3) * 2, c0 = L >> 2 (the tensor-core permutation).
template <int BITS>
__device__ __forceinline__ void decode_lane8(const uint32_t* w, int L, uint32_t p[4]) {
  constexpr int LEN = 256 * BITS, NW = 8 * BITS;
  const int s = (8 * L * BITS + BITS - 16 + LEN) % LEN;  // first bit of step 8L's window
  const int i0 = s >> 5, o0 = s & 31;
  const uint32_t w0 = w[i0], w1 = w[(i0 + 1) % NW], w2 = w[(i0 + 2) % NW];
  const uint64_t lo = (uint64_t(w0) << 32) | w1, hi = (uint64_t(w1) << 32) | w2;
  uint32_t st[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int o = o0 + j * BITS;  // window = bits [o, o + 16) of w0 w1 w2 (MSB first)
    st[j] = o <= 48 ? uint32_t(lo >> (48 - o)) & 0xffffu : uint32_t(hi >> (80 - o)) & 0xffffu;
  }
  p[0] = decode_mul1_x2(st[0], st[1]);
  p[1] = decode_mul1_x2(st[2], st[3]);
  p[2] = decode_mul1_x2(st[4], st[5]);
  p[3] = decode_mul1_x2(st[6], st[7]);
}

template <int BITS>
__global__ __launch_bounds__(V2_NT) void exl3_gemm_v2_kernel(const uint16_t* __restrict__ xh, int K,
                                                             const Exl3Tile* __restrict__ tiles,
                                                             const uint16_t* const* __restrict__ trellis_tab,
                                                             const uint16_t* const* __restrict__ svh_tab, bf16* out,
                                                             int ldo, int N, const int* ntiles_dev) {
  constexpr int NW = 8 * BITS, KT = V2_KS / 16;
  constexpr int AROW = V2_KS + V2_PAD;
  constexpr int A_BYTES = V2_BM * AROW * 2, B_BYTES = 8 * 16 * AROW * 2;
  constexpr int C_BYTES = 64 * (V2_BN + 4) * 4;  // epilogue: 64 rows at a time
  constexpr int AB_BYTES = A_BYTES + B_BYTES > C_BYTES ? A_BYTES + B_BYTES : C_BYTES;
  __shared__ alignas(16) uint8_t smem[AB_BYTES];
  __shared__ uint32_t sW[8][KT][NW];
  _Float16(*sA)[AROW] = reinterpret_cast<_Float16(*)[AROW]>(smem);
  _Float16(*sB)[16][AROW] = reinterpret_cast<_Float16(*)[16][AROW]>(smem + A_BYTES);  // [wave][n][k]
  float(*sC)[V2_BN + 4] = reinterpret_cast<float(*)[V2_BN + 4]>(smem);

  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  const Exl3Tile tile = tiles[blockIdx.y];
  const int n0 = blockIdx.x * V2_BN;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const uint32_t* trellis = reinterpret_cast<const uint32_t*>(trellis_tab[tile.expert]);
  const int tiles_n = N / 16, nt = n0 / 16 + wave;
  const int mtiles = (tile.rows + 15) >> 4;  // row tiles holding rows (wave-uniform)

  float8_t acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;

  // Software pipeline: slab k + 1's A rows and trellis words are loaded into registers while slab k is
  // decoded and multiplied.
  constexpr int WPL = (KT * NW + 31) / 32;  // trellis words per lane
  uint4 ra[4];
  uint32_t rw[WPL];
  auto load_slab = [&](int k0) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * V2_NT, r = idx >> 3, c = (idx & 7) * 8;
      ra[q] = r < tile.rows ? *reinterpret_cast<const uint4*>(xh + size_t(tile.row0 + r) * K + k0 + c)
                            : uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (int j = 0; j < WPL; ++j) {
      const int i = lane + j * 32;
      if (i < KT * NW) {
        const int kt = i / NW, wd = i % NW;
        rw[j] = trellis[(size_t(k0 / 16 + kt) * tiles_n + nt) * NW + wd];
      }
    }
  };
  load_slab(0);
  for (int k0 = 0; k0 < K; k0 += V2_KS) {
    // A slab (128 rows x 64 halves) and this wave's trellis tiles (k-tiles k0/16 .. +3 of column tile nt).
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * V2_NT, r = idx >> 3, c = (idx & 7) * 8;
      *reinterpret_cast<uint4*>(&sA[r][c]) = ra[q];
    }
#pragma unroll
    for (int j = 0; j < WPL; ++j) {
      const int i = lane + j * 32;
      if (i < KT * NW) sW[wave][i / NW][i % NW] = rw[j];
    }
    __builtin_amdgcn_wave_barrier();
    if (k0 + V2_KS < K) load_slab(k0 + V2_KS);
    {
      const int r0 = (lane & 3) * 2, c0 = lane >> 2;
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        uint32_t p[4];
        decode_lane8<BITS>(sW[wave][kt], lane, p);
        const int kb = kt * 16;
        *reinterpret_cast<uint32_t*>(&sB[wave][c0][kb + r0]) = p[0];
        *reinterpret_cast<uint32_t*>(&sB[wave][c0][kb + r0 + 8]) = p[1];
        *reinterpret_cast<uint32_t*>(&sB[wave][c0 + 8][kb + r0]) = p[2];
        *reinterpret_cast<uint32_t*>(&sB[wave][c0 + 8][kb + r0 + 8]) = p[3];
      }
    }
    __syncthreads();  // A from every thread; B is this wave's own
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      const int kk = kt * 16;
#if defined(UH_GFX12)
      const half8_t bf = *reinterpret_cast<const half8_t*>(&sB[wave][lane & 15][kk + (lane >> 4) * 8]);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const half8_t af = *reinterpret_cast<const half8_t*>(&sA[i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
          acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, acc[i]);
        }
#elif defined(UH_GFX11)
      const half16_t bf = load16(&sB[wave][lane & 15][kk]);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const half16_t af = load16(&sA[i * 16 + (lane & 15)][kk]);
          acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[i]);
        }
#endif
    }
    __syncthreads();  // before the next slab overwrites A and B
  }
  // Epilogue in two halves of 64 rows: accumulators -> sC, output Hadamard per row, * svh, bf16.
  const uint16_t* svh = svh_tab[tile.expert];
  const int col = wave * 16 + (lane & 15);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    if (half * 64 >= tile.rows) break;  // block-uniform
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      const int i = half * 4 + ii;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
#if defined(UH_GFX12)
        const int row = ii * 16 + (lane >> 4) * 8 + e;
#else
        const int row = ii * 16 + 2 * e + (lane >> 4);
#endif
        sC[row][col] = acc[i][e];
      }
    }
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
      for (int p = tid; p < 64 * 64; p += V2_NT) {
        const int row = p >> 6, q = p & 63;
        const int a = (q / h) * 2 * h + (q % h);
        const float x0 = sC[row][a], x1 = sC[row][a + h];
        sC[row][a] = x0 + x1;
        sC[row][a + h] = x0 - x1;
      }
      __syncthreads();
    }
    for (int p = tid; p < 64 * V2_BN; p += V2_NT) {
      const int row = p / V2_BN, c = p % V2_BN, grow = half * 64 + row;
      if (grow >= tile.rows) continue;
      const float v = sC[row][c] * 0.08838834764831845f * __half2float(__ushort_as_half(svh[n0 + c]));
      out[size_t(tile.row0 + grow) * ldo + n0 + c] = f32_to_bf16(v);
    }
    __syncthreads();
  }
}

}  // namespace

constexpr int kXformGridRows = 1024;  // rows per pass of the transforms' grids

void exl3_xform_in(const bf16* x, int ldx, const int32_t* row_expert, const uint16_t* const* suh_tab, uint16_t* xh,
                   int P, int K, bool fake_quant, hipStream_t st, const int* row0_dev, const int* row1_dev,
                   const int32_t* x_rows) {
  if (P <= 0) return;
  xform_in_kernel<<<dim3(K / 128, std::min(P, kXformGridRows)), 128, 0, st>>>(x, ldx, row_expert, suh_tab, xh, K,
                                                                             fake_quant, P, row0_dev, row1_dev, x_rows);
  UH_CHECK_LAUNCH();
}

void exl3_swiglu_xform(const bf16* g, const bf16* u, const float* w, const int32_t* row_expert,
                       const uint16_t* const* suh_tab, uint16_t* xh, int P, int I, float limit, hipStream_t st,
                       bool fake_quant, const int* row0_dev, const int* row1_dev) {
  if (P <= 0) return;
  const dim3 grid(I / 128, std::min(P, kXformGridRows));
  if (fake_quant)
    swiglu_xform_kernel<true><<<grid, 128, 0, st>>>(g, u, w, row_expert, suh_tab, xh, I, limit, P, row0_dev, row1_dev);
  else
    swiglu_xform_kernel<false><<<grid, 128, 0, st>>>(g, u, w, row_expert, suh_tab, xh, I, limit, P, row0_dev, row1_dev);
  UH_CHECK_LAUNCH();
}

void exl3_grouped_gemm_v1(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles,
                          const uint16_t* const* trellis_tab, const uint16_t* const* svh_tab, int bits, bf16* out,
                          int ldo, int N, hipStream_t st, const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % KS || N % BN) throw std::runtime_error("exl3_grouped_gemm_v1: K/N not multiples of 32/128");
  const dim3 grid(N / BN, ntiles);
  switch (bits) {
    case 2: exl3_gemm_kernel<2><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 3: exl3_gemm_kernel<3><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 4: exl3_gemm_kernel<4><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 5: exl3_gemm_kernel<5><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 6: exl3_gemm_kernel<6><<<grid, NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    default: throw std::runtime_error("exl3_grouped_gemm_v1: unsupported bit width");
  }
  UH_CHECK_LAUNCH();
}

void exl3_grouped_gemm(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles,
                       const uint16_t* const* trellis_tab, const uint16_t* const* svh_tab, int bits, bf16* out,
                       int ldo, int N, hipStream_t st, const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % V2_KS || N % V2_BN) throw std::runtime_error("exl3_grouped_gemm: K/N not multiples of 64/128");
  const dim3 grid(N / V2_BN, ntiles);
  switch (bits) {
    case 2: exl3_gemm_v2_kernel<2><<<grid, V2_NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 3: exl3_gemm_v2_kernel<3><<<grid, V2_NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 4: exl3_gemm_v2_kernel<4><<<grid, V2_NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 5: exl3_gemm_v2_kernel<5><<<grid, V2_NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    case 6: exl3_gemm_v2_kernel<6><<<grid, V2_NT, 0, st>>>(xh, K, tiles, trellis_tab, svh_tab, out, ldo, N, ntiles_dev); break;
    default: throw std::runtime_error("exl3_grouped_gemm: unsupported bit width");
  }
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

// ---------------------------------------------------------------------------------------------------
// Decode path: tiles with <= 8 rows (any batch of <= 8 tokens).
namespace unholy::kernels {

namespace {

constexpr int GV_ROWS = 8, GV_KC = 512, GV_NT = 256;

// One block per (16-column block nb, tile); y'[row][n] = xh[row] @ W_inner[e][:, n] for the tile's rows.
template <int BITS>
__global__ __launch_bounds__(GV_NT) void exl3_gemv_kernel(const uint16_t* __restrict__ xh, int K,
                                                          const Exl3Tile* __restrict__ tiles,
                                                          const uint16_t* const* __restrict__ trellis_tab,
                                                          float* __restrict__ yp, int N, const int* ntiles_dev) {
  constexpr int NW = 8 * BITS;
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  const Exl3Tile tile = tiles[blockIdx.y];
  const int nb = blockIdx.x, tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int M = tile.rows;
  __shared__ _Float16 sX[GV_ROWS][GV_KC];
  __shared__ uint32_t sW[8][NW];  // one tile's words per wave
  __shared__ float sRed[8][GV_ROWS][16];
  const uint32_t* trellis = reinterpret_cast<const uint32_t*>(trellis_tab[tile.expert]);
  const int tiles_n = N / 16;
  // This lane's 8 trellis steps cover rows {r0, r0+1, r0+8, r0+9} x columns {c0, c0+8}.
  const int r0 = (lane & 3) * 2, c0 = lane >> 2;
  float acc[GV_ROWS][2];
#pragma unroll
  for (int m = 0; m < GV_ROWS; ++m) acc[m][0] = acc[m][1] = 0.0f;

  // The input in chunks of GV_ROWS * GV_KC values over the tile's rows (a single row: 4096 columns per chunk, so few
  // block barriers); wave w takes the chunk's k-tiles w, w + 8, ... (the global order of the 512-column chunks: the same
  // sums), each tile's words loaded into registers one tile ahead of its decode.
  const int chunk = max(GV_KC, GV_ROWS * GV_KC / max(M, 1) / GV_KC * GV_KC);
  _Float16* sx = &sX[0][0];
  for (int kc = 0; kc < K; kc += chunk) {
    const int kn = min(chunk, K - kc), nkt = kn / 16;
    for (int i = tid; i < M * kn; i += GV_NT) {
      const int m = i / kn, k = i % kn;
      reinterpret_cast<uint16_t*>(sx)[m * chunk + k] = xh[size_t(tile.row0 + m) * K + kc + k];
    }
    __syncthreads();
    uint32_t pf0 = 0, pf1 = 0;
    const auto fetch = [&](int kt) {
      const size_t ti = size_t((kc / 16) + kt) * tiles_n + nb;
      if (lane < NW) pf0 = trellis[ti * NW + lane];
      if (32 + lane < NW) pf1 = trellis[ti * NW + 32 + lane];
    };
    if (wave < nkt) fetch(wave);
    for (int kt = wave; kt < nkt; kt += 8) {
      if (lane < NW) sW[wave][lane] = pf0;
      if (32 + lane < NW) sW[wave][32 + lane] = pf1;
      __builtin_amdgcn_wave_barrier();
      if (kt + 8 < nkt) fetch(kt + 8);
      float wv[8];
      {
        uint32_t p[4];
        decode_lane8<BITS>(sW[wave], lane, p);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          wv[j] = __half2float(__ushort_as_half(uint16_t(p[j / 2] >> (16 * (j % 2)))));
      }
      __builtin_amdgcn_wave_barrier();
      const int kb = kt * 16;
      for (int m = 0; m < M; ++m) {
        const _Float16* xr = sx + m * chunk + kb;
        const float x0 = float(xr[r0]), x1 = float(xr[r0 + 1]);
        const float x8 = float(xr[r0 + 8]), x9 = float(xr[r0 + 9]);
        // j: 0 (r0,c0) 1 (r0+1,c0) 2 (r0+8,c0) 3 (r0+9,c0) 4..7 the same rows at c0+8
        acc[m][0] += wv[0] * x0 + wv[1] * x1 + wv[2] * x8 + wv[3] * x9;
        acc[m][1] += wv[4] * x0 + wv[5] * x1 + wv[6] * x8 + wv[7] * x9;
      }
    }
    __syncthreads();
  }
  // Lanes 4c0..4c0+3 share columns c0 and c0+8.
  for (int m = 0; m < M; ++m)
    for (int h = 0; h < 2; ++h) {
      float v = acc[m][h];
      v += __shfl_xor(v, 1, 32);
      v += __shfl_xor(v, 2, 32);
      if ((lane & 3) == 0) sRed[wave][m][c0 + 8 * h] = v;
    }
  __syncthreads();
  for (int i = tid; i < M * 16; i += GV_NT) {
    const int m = i / 16, cc = i % 16;
    float v = 0.0f;
    for (int w8 = 0; w8 < 8; ++w8) v += sRed[w8][m][cc];
    yp[size_t(tile.row0 + m) * N + nb * 16 + cc] = v;
  }
}

// out[row] = bf16(H128(y'[row]) * svh[e] / sqrt(128)) for the tiles' rows; block per (segment, tile row).
__global__ __launch_bounds__(128) void exl3_out_rows_kernel(const float* yp, const Exl3Tile* tiles,
                                                            const uint16_t* const* svh_tab, bf16* out, int ldo, int N,
                                                            const int* ntiles_dev) {
  const int ti = blockIdx.y / GV_ROWS, m = blockIdx.y % GV_ROWS;
  if (ntiles_dev && ti >= *ntiles_dev) return;
  const Exl3Tile tile = tiles[ti];
  if (m >= tile.rows) return;
  __shared__ float b[128];
  const int seg = blockIdx.x, i = threadIdx.x, row = tile.row0 + m;
  b[i] = yp[size_t(row) * N + seg * 128 + i];
  __syncthreads();
  fwht128_lds(b, i);
  const uint16_t* svh = svh_tab[tile.expert];
  out[size_t(row) * ldo + seg * 128 + i] =
      f32_to_bf16(b[i] * 0.08838834764831845f * __half2float(__ushort_as_half(svh[seg * 128 + i])));
}

}  // namespace

void exl3_grouped_gemv(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles, const uint16_t* const* trellis_tab,
                       const uint16_t* const* svh_tab, int bits, float* yp, bf16* out, int ldo, int N, hipStream_t st,
                       const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % GV_KC || N % 128) throw std::runtime_error("exl3_grouped_gemv: K % 512 or N % 128");
  const dim3 grid(N / 16, ntiles);
  switch (bits) {
    case 2: exl3_gemv_kernel<2><<<grid, GV_NT, 0, st>>>(xh, K, tiles, trellis_tab, yp, N, ntiles_dev); break;
    case 3: exl3_gemv_kernel<3><<<grid, GV_NT, 0, st>>>(xh, K, tiles, trellis_tab, yp, N, ntiles_dev); break;
    case 4: exl3_gemv_kernel<4><<<grid, GV_NT, 0, st>>>(xh, K, tiles, trellis_tab, yp, N, ntiles_dev); break;
    case 5: exl3_gemv_kernel<5><<<grid, GV_NT, 0, st>>>(xh, K, tiles, trellis_tab, yp, N, ntiles_dev); break;
    case 6: exl3_gemv_kernel<6><<<grid, GV_NT, 0, st>>>(xh, K, tiles, trellis_tab, yp, N, ntiles_dev); break;
    default: throw std::runtime_error("exl3_grouped_gemv: unsupported bit width");
  }
  UH_CHECK_LAUNCH();
  exl3_out_rows_kernel<<<dim3(N / 128, ntiles * GV_ROWS), 128, 0, st>>>(yp, tiles, svh_tab, out, ldo, N, ntiles_dev);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
