#include <algorithm>
#include <cstdlib>

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
                                                          const uint8_t* const* s_tab, bf16* xa, int K) {
  __shared__ float red[4];
  const int r = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  if (!s_tab[row_expert[r]]) return;  // expert not on this device (uniform per block)
  const float v = fake_quant_fp8_block(bf16_to_f32(x[size_t(r) * ldx + seg * 128 + i]), red);
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

// Grouped GEMM: block = one tile (<= 128 rows of one expert) x 128 output columns; wave w owns columns
// [n0 + 16w, n0 + 16w + 16) and dequantizes that weight slice per 64-k slab into its own LDS (lane:
// column lane / 2, k half lane % 2); A slabs are shared; next-slab loads are in flight during the
// current slab.
constexpr int MX_BM = 128, MX_BN = 128, MX_KS = 64, MX_NT = 256, MX_PAD = 8;

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
  __shared__ alignas(16) uint16_t sB[8][16][AROW];
  __shared__ uint32_t sLut[256], sNz[256];
  if (ntiles_dev && int(blockIdx.y) >= *ntiles_dev) return;
  mx_build_lut(sLut, sNz);
  __syncthreads();
  const Exl3Tile tile = tiles[blockIdx.y];
  const int n0 = blockIdx.x * MX_BN;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int col = lane >> 1, kh = (lane & 1) * 32;
  const uint8_t* wbase = w_tab[tile.expert];
  const uint8_t* sbase = s_tab[tile.expert];
  const size_t tiles_n = size_t(N / 16), nt = size_t(n0 / 16 + wave);
  const int mtiles = (tile.rows + 15) >> 4;

  float8_t acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;

  uint4 ra[4], rw;
  uint8_t rs;
  auto load_slab = [&](int k0) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * MX_NT, r = idx >> 3, c = (idx & 7) * 8;
      ra[q] = r < tile.rows ? *reinterpret_cast<const uint4*>(xa + size_t(tile.row0 + r) * K + k0 + c)
                            : uint4{0, 0, 0, 0};
    }
    const size_t chunk = size_t(k0 / 64) * tiles_n + nt;  // lane: output col, input half kh
    rw = reinterpret_cast<const uint4*>(wbase + chunk * 512)[lane];
    rs = sbase[chunk * 32 + lane];
  };
  load_slab(0);
  for (int k0 = 0; k0 < K; k0 += MX_KS) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = tid + q * MX_NT, r = idx >> 3, c = (idx & 7) * 8;
      *reinterpret_cast<uint4*>(&sA[r][c]) = ra[q];
    }
    {
      uint32_t d[16];
      mx_dequant32_lut(rw, rs, sLut, sNz, d);
      uint4* dst = reinterpret_cast<uint4*>(&sB[wave][col][kh]);
#pragma unroll
      for (int j = 0; j < 4; ++j) dst[j] = uint4{d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]};
    }
    if (k0 + MX_KS < K) load_slab(k0 + MX_KS);
    __syncthreads();  // A from every thread; B is this wave's own
#pragma unroll
    for (int kt = 0; kt < MX_KS / 16; ++kt) {
      const int kk = kt * 16;
#if defined(UH_GFX12)
      const bf16x8_t bfv = *reinterpret_cast<const bf16x8_t*>(&sB[wave][lane & 15][kk + (lane >> 4) * 8]);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const bf16x8_t afv = *reinterpret_cast<const bf16x8_t*>(&sA[i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
          acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(afv, bfv, acc[i]);
        }
#elif defined(UH_GFX11)
      const uint16_t* pb = &sB[wave][lane & 15][kk];
      const bf16x8_t b0 = *reinterpret_cast<const bf16x8_t*>(pb), b1 = *reinterpret_cast<const bf16x8_t*>(pb + 8);
      const bf16x16_t bfv = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < mtiles) {
          const uint16_t* pa = &sA[i * 16 + (lane & 15)][kk];
          const bf16x8_t a0 = *reinterpret_cast<const bf16x8_t*>(pa), a1 = *reinterpret_cast<const bf16x8_t*>(pa + 8);
          const bf16x16_t afv = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
          acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afv, bfv, acc[i]);
        }
#endif
    }
    __syncthreads();  // before the next slab overwrites A and B
  }
  const int c = n0 + wave * 16 + (lane & 15);
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
      if (row < tile.rows) out[size_t(tile.row0 + row) * ldo + c] = f32_to_bf16(acc[i][e]);
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
    return e ? std::max(kMxGemvTileRows, std::atoi(e)) : 64;
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
                 int K, hipStream_t st) {
  if (P <= 0) return;
  mx_xform_in_kernel<<<dim3(K / 128, P), 128, 0, st>>>(x, ldx, row_expert, s_tab, xa, K);
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

void mx_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                     const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st, const int* ntiles_dev,
                     const uint8_t* const* w_tab2, const uint8_t* const* s_tab2, bf16* out2) {
  if (ntiles <= 0) return;
  if (K % MX_KS || N % MX_BN) throw std::runtime_error("mx_grouped_gemm: K % 64 or N % 128");
  mx_gemm_kernel<<<dim3(N / MX_BN, ntiles, out2 ? 2 : 1), MX_NT, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo, N,
                                                                         ntiles_dev, w_tab2, s_tab2, out2);
  UH_CHECK_LAUNCH();
}

void mx_grouped_gemv(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                     const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st, const int* ntiles_dev,
                     const uint8_t* const* w_tab2, const uint8_t* const* s_tab2, bf16* out2) {
  if (ntiles <= 0) return;
  if (K % MV_KC || N % 16) throw std::runtime_error("mx_grouped_gemv: K % 512 or N % 16");
  mx_gemv_kernel<<<dim3((N / 16 + 7) / 8, ntiles, out2 ? 2 : 1), MV_NT, 0, st>>>(xa, K, tiles, w_tab, s_tab, out, ldo,
                                                                                  N, ntiles_dev, w_tab2, s_tab2, out2);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
