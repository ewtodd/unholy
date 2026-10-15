#include <algorithm>
#include <cstdlib>
#include <string>

#include "kernels/dev.hip.hpp"
#include "kernels/mx_moe.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));

// bf16 bits of e2m1(nibble) * 2^(s - 127): exact, so the float's top half.
__device__ __forceinline__ uint32_t mx_bf16(uint32_t nib, float scale) {
  return __float_as_uint(e2m1_to_f32(nib) * scale) >> 16;
}

// Byte -> its two e2m1 values as a bf16 pair (low nibble in the low half), and a mask of the nonzero
// halves; built in LDS by each block. A power-of-two scale 2^(s - 127) is applied by adding (s - 127)
// to the exponent of the nonzero halves.
__device__ __forceinline__ void mx_build_lut(uint32_t* lut, uint32_t* nz) {
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    const uint32_t lo = mx_bf16(uint32_t(b) & 15, 1.0f), hi = mx_bf16(uint32_t(b) >> 4, 1.0f);
    lut[b] = lo | (hi << 16);
    nz[b] = ((lo & 0x7fffu) ? 0xffffu : 0u) | ((hi & 0x7fffu) ? 0xffff0000u : 0u);
  }
}
__device__ __forceinline__ void mx_dequant32_lut(const uint4 q, uint8_t s, const uint32_t* lut, const uint32_t* nz,
                                                 uint32_t out[16]) {
  const uint32_t adj = ((uint32_t(s) - 127u) << 7) & 0xffffu;
  const uint32_t add = adj | (adj << 16);
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xffu;
    // Each half's exponent changes mod 2^16 (a negative offset borrows): add the halves separately.
    const uint32_t v = lut[byte], a = add & nz[byte];
    out[i] = ((v + a) & 0xffffu) | ((v & 0xffff0000u) + (a & 0xffff0000u));
  }
}

// Thread per (chunk, output, byte of the slab): 32 code bytes, then scales.
__global__ void mx_repack_kernel(const uint8_t* w, const uint8_t* s, uint8_t* tw, uint8_t* ts, int N, int K) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  const size_t nbytes = size_t(N) * K / 2;
  if (gid < nbytes) {
    const int n = int(gid / (K / 2)), kb = int(gid % (K / 2));  // byte kb of row n: inputs 2kb, 2kb + 1
    const size_t chunk = size_t(kb / 32) * (N / 16) + n / 16;
    tw[chunk * 512 + (n % 16) * 32 + kb % 32] = w[gid];
  }
  if (gid < size_t(N) * (K / 32)) {
    const int n = int(gid / (K / 32)), b = int(gid % (K / 32));
    const size_t chunk = size_t(b / 2) * (N / 16) + n / 16;
    ts[chunk * 32 + (n % 16) * 2 + b % 2] = s[gid];
  }
}

__global__ __launch_bounds__(128) void mx_xform_in_kernel(const bf16* x, int ldx, const int32_t* row_expert,
                                                          const uint8_t* const* s_tab, bf16* xa, int K,
                                                          const int32_t* x_rows) {
  __shared__ float red[4];
  const int r = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  if (!s_tab[row_expert[r]]) return;  // expert not on this device (uniform per block)
  const size_t src = x_rows ? size_t(x_rows[r]) : size_t(r);
  const float v = fake_quant_fp8_block(bf16_to_f32(x[src * ldx + seg * 128 + i]), red);
  xa[size_t(r) * K + seg * 128 + i] = f32_to_bf16(v);
}

__global__ __launch_bounds__(128) void mx_xform_in_q_kernel(const uint8_t* q, const uint8_t* s, const int32_t* p2u,
                                                            const MoePlan* plan, const int32_t* row_expert,
                                                            const uint8_t* const* s_tab, bf16* xa, int K) {
  const int r = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  if (r < plan->remote0 || r >= plan->P || !s_tab[row_expert[r]]) return;  // uniform per block
  const size_t u = size_t(p2u[r]);
  const float v = round_bf16(e4m3_to_f32(q[u * K + seg * 128 + i]) * e8m0_to_f32(s[u * (K / 128) + seg]));
  xa[size_t(r) * K + seg * 128 + i] = f32_to_bf16(v);
}

__global__ __launch_bounds__(128) void mx_swiglu_kernel(const bf16* g, const bf16* u, const float* w,
                                                        const int32_t* row_expert, const uint8_t* const* s_tab,
                                                        bf16* hh, int I, float limit) {
  __shared__ float red[4];
  const int r = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  if (!s_tab[row_expert[r]]) return;
  const size_t idx = size_t(r) * I + seg * 128 + i;
  float gv = bf16_to_f32(g[idx]), uv = bf16_to_f32(u[idx]);
  if (limit > 0.0f) {
    uv = fminf(fmaxf(uv, -limit), limit);
    gv = fminf(gv, limit);
  }
  float hv = (gv / (1.0f + expf(-gv))) * uv;
  if (w) hv = w[r] * hv;
  hv = round_bf16(hv);  // the reference's swiglu output is a bf16 tensor
  hh[idx] = f32_to_bf16(fake_quant_fp8_block(hv, red));
}

// Grouped GEMM: block = one tile (<= 128 rows of one expert) x 128 CW output columns; wave w owns columns
// [n0 + 16 CW w, n0 + 16 CW (w + 1)) and dequantizes that weight slice per 64-k slab into its own LDS
// (lane: column lane / 2, k half lane % 2, per 16-column tile); A slabs are shared; next-slab loads are in
// flight during the current slab. CW = 2 feeds each A fragment read from LDS into two WMMAs (every output's
// k order is the same for any CW).
constexpr int MX_BM = 128, MX_BN = 128, MX_KS = 64, MX_NT = 256, MX_PAD = 8;

template <int CW>
__global__ __launch_bounds__(MX_NT) void mx_gemm_kernel(const bf16* __restrict__ xa, int K,
                                                        const Exl3Tile* __restrict__ tiles,
                                                        const uint8_t* const* w_tab, const uint8_t* const* s_tab,
                                                        bf16* out, int ldo, int N, const int* ntiles_dev,
                                                        const uint8_t* const* w_tab2, const uint8_t* const* s_tab2,
                                                        bf16* out2) {
  if (blockIdx.z) {  // the second projection on the same inputs (w1 and w3 in one launch)
    w_tab = w_tab2;
    s_tab = s_tab2;
    out = out2;
  }
  constexpr int AROW = MX_KS + MX_PAD;
  __shared__ alignas(16) uint16_t sA[MX_BM][AROW];
  __shared__ alignas(16) uint16_t sB[8][16 * CW][AROW];
  __shared__ uint32_t sLut[256], sNz[256];
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  mx_build_lut(sLut, sNz);
  __syncthreads();
  const Exl3Tile tile = tiles[blockIdx.y];
  const int n0 = blockIdx.x * MX_BN * CW;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int col = lane >> 1, kh = (lane & 1) * 32;
  const uint8_t* wbase = w_tab[tile.expert];
  const uint8_t* sbase = s_tab[tile.expert];
  const size_t tiles_n = size_t(N / 16), nt = size_t(n0 / 16 + wave * CW);
  const int mtiles = (tile.rows + 15) >> 4, arows = mtiles * 16;  // rows the WMMAs read

  float8_t acc[CW][8];
#pragma unroll
  for (int cw = 0; cw < CW; ++cw)
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[cw][i][e] = 0.0f;

  uint4 ra[4], rw[CW];
  uint8_t rs[CW];
  auto load_slab = [&](int k0) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * MX_NT, r = idx >> 3, c = (idx & 7) * 8;
      ra[q] = r < tile.rows ? *reinterpret_cast<const uint4*>(xa + size_t(tile.row0 + r) * K + k0 + c)
                            : uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (int cw = 0; cw < CW; ++cw) {
      const size_t chunk = size_t(k0 / 64) * tiles_n + nt + cw;  // lane: output col, input half kh
      rw[cw] = reinterpret_cast<const uint4*>(wbase + chunk * 512)[lane];
      rs[cw] = sbase[chunk * 32 + lane];
    }
  };
  load_slab(0);
  for (int k0 = 0; k0 < K; k0 += MX_KS) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * MX_NT, r = idx >> 3, c = (idx & 7) * 8;
      if (r < arows) *reinterpret_cast<uint4*>(&sA[r][c]) = ra[q];
    }
#pragma unroll
    for (int cw = 0; cw < CW; ++cw) {
      uint32_t d[16];
      mx_dequant32_lut(rw[cw], rs[cw], sLut, sNz, d);
      uint4* dst = reinterpret_cast<uint4*>(&sB[wave][cw * 16 + col][kh]);
#pragma unroll
      for (int j = 0; j < 4; ++j) dst[j] = uint4{d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]};
    }
    if (k0 + MX_KS < K) load_slab(k0 + MX_KS);
    __syncthreads();  // A from every thread; B is this wave's own
#pragma unroll
    for (int kt = 0; kt < MX_KS / 16; ++kt) {
      const int kk = kt * 16;
#if defined(UH_GFX12)
      bf16x8_t bfv[CW];
#pragma unroll
      for (int cw = 0; cw < CW; ++cw)
        bfv[cw] = *reinterpret_cast<const bf16x8_t*>(&sB[wave][cw * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const bf16x8_t afv = *reinterpret_cast<const bf16x8_t*>(&sA[i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
          for (int cw = 0; cw < CW; ++cw)
            acc[cw][i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(afv, bfv[cw], acc[cw][i]);
        }
#elif defined(UH_GFX11)
      bf16x16_t bfv[CW];
#pragma unroll
      for (int cw = 0; cw < CW; ++cw) {
        const uint16_t* pb = &sB[wave][cw * 16 + (lane & 15)][kk];
        const bf16x8_t b0 = *reinterpret_cast<const bf16x8_t*>(pb), b1 = *reinterpret_cast<const bf16x8_t*>(pb + 8);
        bfv[cw] = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const uint16_t* pa = &sA[i * 16 + (lane & 15)][kk];
          const bf16x8_t a0 = *reinterpret_cast<const bf16x8_t*>(pa), a1 = *reinterpret_cast<const bf16x8_t*>(pa + 8);
          const bf16x16_t afv = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#pragma unroll
          for (int cw = 0; cw < CW; ++cw) acc[cw][i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afv, bfv[cw], acc[cw][i]);
        }
#endif
    }
    __syncthreads();  // before the next slab overwrites A and B
  }
#pragma unroll
  for (int cw = 0; cw < CW; ++cw) {
    const int c = n0 + (wave * CW + cw) * 16 + (lane & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i >= mtiles) break;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
#if defined(UH_GFX12)
        const int row = i * 16 + (lane >> 4) * 8 + e;
#else
        const int row = i * 16 + 2 * e + (lane >> 4);
#endif
        if (row < tile.rows) out[size_t(tile.row0 + row) * ldo + c] = f32_to_bf16(acc[cw][i][e]);
      }
    }
  }
}

// mx_gemm_kernel with B in registers. Each lane dequantizes half of a column's 64-input slab (lane: column
// lane % 16, input half lane / 16; the same LUT code and values as there, reading those 16 code bytes
// directly) and takes the other half from the lane 16 away (permlanex16): every lane then holds what its
// WMMA B fragments need (gfx11: all 16 inputs of a k step in both half-waves; gfx12: 8 of them per
// half-wave). No B in LDS: A alone (20 KB) leaves room for three blocks per CU instead of one. Used on gfx11
// (the APU); on gfx12 it is bit-identical too but slower than mx_gemm_kernel<2> (41 vs 54 TFLOPS at 96 rows
// per expert with two column tiles per wave), so the cards keep that.
template <int CW>
__global__ __launch_bounds__(MX_NT) void mx_gemm_rb_kernel(const bf16* __restrict__ xa, int K,
                                                           const Exl3Tile* __restrict__ tiles,
                                                           const uint8_t* const* w_tab, const uint8_t* const* s_tab,
                                                           bf16* out, int ldo, int N, const int* ntiles_dev,
                                                           const uint8_t* const* w_tab2, const uint8_t* const* s_tab2,
                                                           bf16* out2) {
  if (blockIdx.z) {
    w_tab = w_tab2;
    s_tab = s_tab2;
    out = out2;
  }
  constexpr int AROW = MX_KS + MX_PAD;
  __shared__ alignas(16) uint16_t sA[MX_BM][AROW];
  __shared__ uint32_t sLut[256], sNz[256];
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  mx_build_lut(sLut, sNz);
  __syncthreads();
  const Exl3Tile tile = tiles[blockIdx.y];
  const int n0 = blockIdx.x * MX_BN * CW;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int bc = lane & 15, bh = lane >> 4;
  const uint8_t* wbase = w_tab[tile.expert];
  const uint8_t* sbase = s_tab[tile.expert];
  const size_t tiles_n = size_t(N / 16), nt = size_t(n0 / 16 + wave * CW);
  const int mtiles = (tile.rows + 15) >> 4, arows = mtiles * 16;
  float8_t acc[CW][8];
#pragma unroll
  for (int cw = 0; cw < CW; ++cw)
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[cw][i][e] = 0.0f;
  uint4 ra[4], rw[CW];
  uint8_t rs[CW];
  auto load = [&](int k0) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * MX_NT, r = idx >> 3, c = (idx & 7) * 8;
      ra[q] = r < tile.rows ? *reinterpret_cast<const uint4*>(xa + size_t(tile.row0 + r) * K + k0 + c)
                            : uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (int cw = 0; cw < CW; ++cw) {
      const size_t chunk = size_t(k0 / 64) * tiles_n + nt + cw;
      rw[cw] = *reinterpret_cast<const uint4*>(wbase + chunk * 512 + bc * 32 + bh * 16);
      rs[cw] = sbase[chunk * 32 + bc * 2 + bh];
    }
  };
  load(0);
  for (int k0 = 0; k0 < K; k0 += MX_KS) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * MX_NT, r = idx >> 3, c = (idx & 7) * 8;
      if (r < arows) *reinterpret_cast<uint4*>(&sA[r][c]) = ra[q];
    }
    // h[cw][0]: the column's inputs 0-31 of the slab, h[cw][1]: 32-63 (bf16 pairs).
    uint32_t h[CW][2][16];
#pragma unroll
    for (int cw = 0; cw < CW; ++cw) {
      uint32_t d[16];
      mx_dequant32_lut(rw[cw], rs[cw], sLut, sNz, d);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t o = __builtin_amdgcn_permlanex16(d[i], d[i], 0x76543210u, 0xfedcba98u, false, false);
        h[cw][0][i] = bh ? o : d[i];
        h[cw][1][i] = bh ? d[i] : o;
      }
    }
    if (k0 + MX_KS < K) load(k0 + MX_KS);
    __syncthreads();  // A from every thread
#pragma unroll
    for (int kt = 0; kt < MX_KS / 16; ++kt) {
#if defined(UH_GFX12)
      bf16x8_t bfv[CW];  // inputs 16 kt + 8 (lane / 16) .. + 7
#pragma unroll
      for (int cw = 0; cw < CW; ++cw) {
        const uint32_t* hb = &h[cw][kt >> 1][(kt & 1) * 8 + bh * 4];
        const uint4 b = {hb[0], hb[1], hb[2], hb[3]};
        bfv[cw] = *reinterpret_cast<const bf16x8_t*>(&b);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const bf16x8_t afv = *reinterpret_cast<const bf16x8_t*>(&sA[i * 16 + (lane & 15)][kt * 16 + bh * 8]);
#pragma unroll
          for (int cw = 0; cw < CW; ++cw)
            acc[cw][i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(afv, bfv[cw], acc[cw][i]);
        }
#elif defined(UH_GFX11)
      bf16x16_t bfv[CW];  // inputs 16 kt .. + 15
#pragma unroll
      for (int cw = 0; cw < CW; ++cw) {
        const uint32_t* hb = &h[cw][kt >> 1][(kt & 1) * 8];
        const uint4 b0 = {hb[0], hb[1], hb[2], hb[3]}, b1 = {hb[4], hb[5], hb[6], hb[7]};
        bfv[cw] = __builtin_shufflevector(*reinterpret_cast<const bf16x8_t*>(&b0), *reinterpret_cast<const bf16x8_t*>(&b1),
                                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const uint16_t* pa = &sA[i * 16 + (lane & 15)][kt * 16];
          const bf16x8_t a0 = *reinterpret_cast<const bf16x8_t*>(pa), a1 = *reinterpret_cast<const bf16x8_t*>(pa + 8);
          const bf16x16_t afv = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#pragma unroll
          for (int cw = 0; cw < CW; ++cw) acc[cw][i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afv, bfv[cw], acc[cw][i]);
        }
#endif
    }
    __syncthreads();  // before the next slab overwrites A
  }
#pragma unroll
  for (int cw = 0; cw < CW; ++cw) {
    const int c = n0 + (wave * CW + cw) * 16 + (lane & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i >= mtiles) break;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
#if defined(UH_GFX12)
        const int row = i * 16 + (lane >> 4) * 8 + e;
#else
        const int row = i * 16 + 2 * e + (lane >> 4);
#endif
        if (row < tile.rows) out[size_t(tile.row0 + row) * ldo + c] = f32_to_bf16(acc[cw][i][e]);
      }
    }
  }
}

// mx_gemm_kernel for tiles of <= 16 RF rows (steps of <= 16 RF tokens: a decode step's tiles hold a row or
// three), with the same arithmetic per output: the 16-output tile's slab dequantized as there (lane: output
// lane / 2, input half lane % 2) into the wave's own LDS, the WMMA chain over k in the same order. Waves are
// independent -- each loads its A fragments straight from global memory (zero rows past the tile) instead
// of the block staging 128 rows per slab between two barriers -- and a block is 4 of them (64 outputs).
template <int RF>
__global__ __launch_bounds__(128) void mx_gemm_skinny_kernel(const bf16* __restrict__ xa, int K,
                                                             const Exl3Tile* __restrict__ tiles,
                                                             const uint8_t* const* w_tab, const uint8_t* const* s_tab,
                                                             bf16* out, int ldo, int N, const int* ntiles_dev,
                                                             const uint8_t* const* w_tab2,
                                                             const uint8_t* const* s_tab2, bf16* out2) {
  if (blockIdx.z) {
    w_tab = w_tab2;
    s_tab = s_tab2;
    out = out2;
  }
  __shared__ alignas(16) uint16_t sB[4][16][MX_KS + MX_PAD];
  __shared__ uint32_t sLut[256], sNz[256];
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  mx_build_lut(sLut, sNz);
  __syncthreads();
  const Exl3Tile tile = tiles[blockIdx.y];
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = lane >> 1, kh = (lane & 1) * 32;
  const size_t tiles_n = size_t(N / 16), nt = size_t(blockIdx.x) * 4 + wave;
  const uint8_t* wbase = w_tab[tile.expert];
  const uint8_t* sbase = s_tab[tile.expert];
  const int mtiles = (tile.rows + 15) >> 4;
  const int ar = lane & 15;  // A fragment row within a 16-row tile
#if defined(UH_GFX12)
  const int ak = (lane >> 4) * 8;
#else
  const int ak = 0;
#endif
  // Row fragments RF at a time (a tile of more rows than the launch planned for takes more passes).
  for (int f0 = 0; f0 < mtiles; f0 += RF) {
    float8_t acc[RF];
#pragma unroll
    for (int i = 0; i < RF; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;
    // Two slabs' codes in flight: rw[j] holds slab k0 / 64 with k0 / 64 % 2 == j.
    uint4 rw[2];
    uint8_t rs[2];
    const auto load = [&](int j, int k0) {
      const size_t chunk = size_t(k0 / 64) * tiles_n + nt;
      rw[j] = reinterpret_cast<const uint4*>(wbase + chunk * 512)[lane];
      rs[j] = sbase[chunk * 32 + lane];
    };
    const auto slab = [&](int j, int k0) {
      {
        uint32_t d[16];
        mx_dequant32_lut(rw[j], rs[j], sLut, sNz, d);
        uint4* dst = reinterpret_cast<uint4*>(&sB[wave][col][kh]);
#pragma unroll
        for (int q = 0; q < 4; ++q) dst[q] = uint4{d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3]};
      }
      if (k0 + 2 * MX_KS < K) load(j, k0 + 2 * MX_KS);
      // LDS operations of a wave complete in order: the slab written above is what the reads below see.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
      for (int kt = 0; kt < MX_KS / 16; ++kt) {
        const int kk = kt * 16;
#if defined(UH_GFX12)
        const bf16x8_t bfv = *reinterpret_cast<const bf16x8_t*>(&sB[wave][lane & 15][kk + ak]);
#pragma unroll
        for (int i = 0; i < RF; ++i)
          if (f0 + i < mtiles) {
            const int r = (f0 + i) * 16 + ar;
            uint4 av = {0, 0, 0, 0};
            if (r < tile.rows) av = *reinterpret_cast<const uint4*>(xa + size_t(tile.row0 + r) * K + k0 + kk + ak);
            acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(*reinterpret_cast<const bf16x8_t*>(&av), bfv,
                                                                        acc[i]);
          }
#elif defined(UH_GFX11)
        const uint16_t* pb = &sB[wave][lane & 15][kk];
        const bf16x8_t b0 = *reinterpret_cast<const bf16x8_t*>(pb), b1 = *reinterpret_cast<const bf16x8_t*>(pb + 8);
        const bf16x16_t bfv = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#pragma unroll
        for (int i = 0; i < RF; ++i)
          if (f0 + i < mtiles) {
            const int r = (f0 + i) * 16 + ar;
            uint4 a0 = {0, 0, 0, 0}, a1 = {0, 0, 0, 0};
            if (r < tile.rows) {
              const uint4* pa = reinterpret_cast<const uint4*>(xa + size_t(tile.row0 + r) * K + k0 + kk + ak);
              a0 = pa[0];
              a1 = pa[1];
            }
            const bf16x16_t afv = __builtin_shufflevector(*reinterpret_cast<const bf16x8_t*>(&a0),
                                                          *reinterpret_cast<const bf16x8_t*>(&a1), 0, 1, 2, 3, 4, 5, 6,
                                                          7, 8, 9, 10, 11, 12, 13, 14, 15);
            acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afv, bfv, acc[i]);
          }
#endif
      }
      // The next slab's LDS writes come after these reads (in order within the wave).
      __builtin_amdgcn_wave_barrier();
    };
    load(0, 0);
    if (MX_KS < K) load(1, MX_KS);
    for (int k0 = 0; k0 < K; k0 += 2 * MX_KS) {  // K % 128 == 0 (launch)
      slab(0, k0);
      slab(1, k0 + MX_KS);
    }
    const int c = int(nt) * 16 + (lane & 15);
#pragma unroll
    for (int i = 0; i < RF; ++i) {
      if (f0 + i >= mtiles) break;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
#if defined(UH_GFX12)
        const int row = (f0 + i) * 16 + (lane >> 4) * 8 + e;
#else
        const int row = (f0 + i) * 16 + 2 * e + (lane >> 4);
#endif
        if (row < tile.rows) out[size_t(tile.row0 + row) * ldo + c] = f32_to_bf16(acc[i][e]);
      }
    }
  }
}

// GEMV for tiles of <= 8 rows: block = 8 waves x 16-output tiles (128 outputs); the tile's rows are
// staged in LDS 512 inputs at a time; per 64-input slab a wave reads its contiguous 512-byte chunk, lane
// = (output lane / 2, input half lane % 2), and accumulates its 32 products per row; lane pairs are
// summed at the end.
constexpr int MV_NT = 256, MV_MAXM = 8, MV_KC = 512;

__global__ __launch_bounds__(MV_NT) void mx_gemv_kernel(const bf16* __restrict__ xa, int K,
                                                        const Exl3Tile* __restrict__ tiles,
                                                        const uint8_t* const* w_tab, const uint8_t* const* s_tab,
                                                        bf16* out, int ldo, int N, const int* ntiles_dev,
                                                        const uint8_t* const* w_tab2, const uint8_t* const* s_tab2,
                                                        bf16* out2) {
  if (blockIdx.z) {  // the second projection on the same inputs (w1 and w3 in one launch)
    w_tab = w_tab2;
    s_tab = s_tab2;
    out = out2;
  }
  __shared__ alignas(16) uint16_t sX[MV_MAXM][MV_KC];
  __shared__ uint32_t sLut[256], sNz[256];
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  mx_build_lut(sLut, sNz);
  const Exl3Tile tile = tiles[blockIdx.y];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int M = tile.rows;
  const size_t tiles_n = size_t(N / 16), nt = size_t(blockIdx.x) * 8 + wave;
  const uint8_t* wbase = w_tab[tile.expert];
  const uint8_t* sbase = s_tab[tile.expert];
  const int kh = (lane & 1) * 32;
  float acc[MV_MAXM];
#pragma unroll
  for (int m = 0; m < MV_MAXM; ++m) acc[m] = 0.0f;
  for (int kc = 0; kc < K; kc += MV_KC) {
    __syncthreads();
    for (int i = tid; i < M * (MV_KC / 8); i += MV_NT) {  // 16-byte pieces
      const int m = i / (MV_KC / 8), c = (i % (MV_KC / 8)) * 8;
      *reinterpret_cast<uint4*>(&sX[m][c]) = *reinterpret_cast<const uint4*>(xa + size_t(tile.row0 + m) * K + kc + c);
    }
    __syncthreads();
    if (nt * 16 >= size_t(N)) continue;
#pragma unroll 2
    for (int k0 = kc; k0 < kc + MV_KC; k0 += 64) {
      const size_t chunk = size_t(k0 / 64) * tiles_n + nt;
      // Unscaled values; the block's power-of-two scale multiplies the partial sum (exactly the same as
      // scaling every product).
      const uint4 q = reinterpret_cast<const uint4*>(wbase + chunk * 512)[lane];
      const float sc = e8m0_to_f32(sbase[chunk * 32 + lane]);
      uint32_t d[16];
      {
        const uint32_t w4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) d[i] = sLut[(w4[i / 4] >> (8 * (i % 4))) & 0xffu];
      }
      const int ko = k0 - kc + kh;
#pragma unroll
      for (int m = 0; m < MV_MAXM; ++m) {
        if (m >= M) break;
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(&sX[m][ko]);
        float part = 0.0f;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          const uint32_t xv = xr[t], wv = d[t];
          part = fmaf(__uint_as_float(xv << 16), __uint_as_float(wv << 16), part);
          part = fmaf(__uint_as_float(xv & 0xffff0000u), __uint_as_float(wv & 0xffff0000u), part);
        }
        acc[m] = fmaf(part, sc, acc[m]);
      }
    }
  }
  if (nt * 16 >= size_t(N)) return;
#pragma unroll
  for (int m = 0; m < MV_MAXM; ++m) {
    if (m >= M) break;
    const float v = acc[m] + __shfl_xor(acc[m], 1, 32);
    if ((lane & 1) == 0) out[size_t(tile.row0 + m) * ldo + nt * 16 + (lane >> 1)] = f32_to_bf16(v);
  }
}

}  // namespace

int mx_gemv_max_tokens() {
  static const int v = [] {
    const char* e = std::getenv("UNHOLY_MX_GEMV_TOKENS");
    return e ? std::max(kMxGemvTileRows, std::atoi(e)) : kMxGemvTileRows;
  }();
  return v;
}

void mx_repack(uint8_t* w, uint8_t* s, int N, int K, uint8_t* tmp, hipStream_t st) {
  if (N % 16 || K % 64) throw std::runtime_error("mx_repack: N % 16 or K % 64");
  const size_t wb = size_t(N) * K / 2, sb = size_t(N) * (K / 32);
  mx_repack_kernel<<<int((wb + 255) / 256), 256, 0, st>>>(w, s, tmp, tmp + wb, N, K);
  UH_CHECK_LAUNCH();
  UH_CHECK(hipMemcpyAsync(w, tmp, wb, hipMemcpyDeviceToDevice, st));
  UH_CHECK(hipMemcpyAsync(s, tmp + wb, sb, hipMemcpyDeviceToDevice, st));
}

void mx_xform_in(const bf16* x, int ldx, const int32_t* row_expert, const uint8_t* const* s_tab, bf16* xa, int P,
                 int K, hipStream_t st, const int32_t* x_rows) {
  if (P <= 0) return;
  mx_xform_in_kernel<<<dim3(K / 128, P), 128, 0, st>>>(x, ldx, row_expert, s_tab, xa, K, x_rows);
  UH_CHECK_LAUNCH();
}

void mx_xform_in_q(const uint8_t* q, const uint8_t* s, const int32_t* p2u, const MoePlan* plan,
                   const int32_t* row_expert, const uint8_t* const* s_tab, bf16* xa, int P, int K, hipStream_t st) {
  if (P <= 0) return;
  mx_xform_in_q_kernel<<<dim3(K / 128, P), 128, 0, st>>>(q, s, p2u, plan, row_expert, s_tab, xa, K);
  UH_CHECK_LAUNCH();
}

void mx_swiglu(const bf16* g, const bf16* u, const float* w, const int32_t* row_expert, const uint8_t* const* s_tab,
               bf16* hh, int P, int I, float limit, hipStream_t st) {
  if (P <= 0) return;
  mx_swiglu_kernel<<<dim3(I / 128, P), 128, 0, st>>>(g, u, w, row_expert, s_tab, hh, I, limit);
  UH_CHECK_LAUNCH();
}

// Column tiles per wave of the grouped GEMM on the current device: UNHOLY_MX_GEMM_CW (1 / 2) or by arch.
int mx_gemm_cols(int N) {
  static int by_dev[64] = {};
  static const int forced = [] {
    const char* v = std::getenv("UNHOLY_MX_GEMM_CW");
    return v ? std::atoi(v) : 0;
  }();
  if (N % (2 * MX_BN)) return 1;
  if (forced == 1 || forced == 2) return forced;
  int dev = 0;
  UH_CHECK(hipGetDevice(&dev));
  if (!by_dev[dev]) {
    hipDeviceProp_t p{};
    UH_CHECK(hipGetDeviceProperties(&p, dev));
    by_dev[dev] = std::string(p.gcnArchName).rfind("gfx12", 0) == 0 ? 2 : 1;  // unholy-mx-test: +10% on gfx1201, none on gfx1151
  }
  return by_dev[dev];
}

namespace {
int g_rb = -1;
// gfx11 on the current device.
bool dev_gfx11() {
  static int by_dev[64] = {};
  int dev = 0;
  UH_CHECK(hipGetDevice(&dev));
  if (!by_dev[dev]) {
    hipDeviceProp_t p{};
    UH_CHECK(hipGetDeviceProperties(&p, dev));
    by_dev[dev] = std::string(p.gcnArchName).rfind("gfx11", 0) == 0 ? 2 : 1;
  }
  return by_dev[dev] == 2;
}
}  // namespace

void mx_gemm_force_rb(int on) { g_rb = on; }
bool mx_gemm_rb() {
  static const bool env = [] {
    const char* v = std::getenv("UNHOLY_MX_GEMM_RB");
    return !v || std::string(v) != "0";
  }();
  return dev_gfx11() && (g_rb < 0 ? env : g_rb > 0);
}

bool mx_skinny_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("UNHOLY_MX_SKINNY");
    return !v || std::string(v) != "0";
  }();
  return on;
}

void mx_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                     const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st, const int* ntiles_dev,
                     const uint8_t* const* w_tab2, const uint8_t* const* s_tab2, bf16* out2, int max_rows) {
  if (ntiles <= 0) return;
  if (K % MX_KS || N % MX_BN) throw std::runtime_error("mx_grouped_gemm: K % 64 or N % 128");
  if (mx_gemm_rb()) {  // gfx11: also ahead of the skinny kernel on decode tiles (182 vs 162 GB/s)
    mx_gemm_rb_kernel<1><<<dim3(N / MX_BN, ntiles, out2 ? 2 : 1), MX_NT, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo, N,
                                                                                 ntiles_dev, w_tab2, s_tab2, out2);
  } else if (max_rows <= 64 && K % (2 * MX_KS) == 0 && mx_skinny_enabled()) {
    const dim3 grid(N / 64, ntiles, out2 ? 2 : 1);
    if (max_rows <= 16)
      mx_gemm_skinny_kernel<1><<<grid, 128, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo, N, ntiles_dev, w_tab2, s_tab2, out2);
    else if (max_rows <= 32)
      mx_gemm_skinny_kernel<2><<<grid, 128, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo, N, ntiles_dev, w_tab2, s_tab2, out2);
    else
      mx_gemm_skinny_kernel<4><<<grid, 128, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo, N, ntiles_dev, w_tab2, s_tab2, out2);
  } else if (mx_gemm_cols(N) == 2) {
    mx_gemm_kernel<2><<<dim3(N / (2 * MX_BN), ntiles, out2 ? 2 : 1), MX_NT, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo,
                                                                                    N, ntiles_dev, w_tab2, s_tab2, out2);
  } else {
    mx_gemm_kernel<1><<<dim3(N / MX_BN, ntiles, out2 ? 2 : 1), MX_NT, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo, N,
                                                                              ntiles_dev, w_tab2, s_tab2, out2);
  }
  UH_CHECK_LAUNCH();
}

void mx_grouped_gemv(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                     const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st, const int* ntiles_dev,
                     const uint8_t* const* w_tab2, const uint8_t* const* s_tab2, bf16* out2, int) {
  if (ntiles <= 0) return;
  if (K % MV_KC || N % 16) throw std::runtime_error("mx_grouped_gemv: K % 512 or N % 16");
  mx_gemv_kernel<<<dim3((N / 16 + 7) / 8, ntiles, out2 ? 2 : 1), MV_NT, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo,
                                                                                  N, ntiles_dev, w_tab2, s_tab2, out2);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
