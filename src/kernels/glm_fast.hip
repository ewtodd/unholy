// GLM-5.3-Flash DSA kernels over the paged cache (kernels/glm_fast.hpp).
//
// glm_mla_attn follows attn_wmma.hip's design (one 256-thread block per query computes all 64 heads; wave w =
// (hg, dh) owns heads [16 hg, 16 hg + 16) and dims [256 dh, 256 dh + 256); keys 16 at a time, staged key-major and
// dim-major in LDS; online softmax per head with the probabilities rounded to bf16 before P V), without DSV4's
// window and sink: a query's keys are exactly its index list, read as fp8 latent rows through the block table.
#include "kernels/glm_fast.hpp"

#include <algorithm>
#include <stdexcept>

#include "kernels/dev.hip.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));

constexpr int R = 512, NS = R / 64, ID = 128, POOL = 4, PPP = kGlmPageRows / POOL;  // pools per page

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

__device__ __forceinline__ size_t row_id(const int32_t* table, int pos) {
  return size_t(table[pos / kGlmPageRows]) * kGlmPageRows + pos % kGlmPageRows;
}
__device__ __forceinline__ bf16* pool_row(const GlmPaged& pg, int p) {
  return pg.pk + (size_t(pg.table[p / PPP]) * PPP + p % PPP) * ID;
}

// One wave per (row, 64-value block); each lane holds two values (fake_quant_fp8's arithmetic, block 64).
__global__ void store_latent_kernel(const bf16* lat, GlmPaged pg, int s0, int n) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= n * NS) return;
  const int r = wid / NS, b = wid % NS;
  const bf16* x = lat + size_t(r) * R + b * 64 + lane * 2;
  const float v0 = bf16_to_f32(x[0]), v1 = bf16_to_f32(x[1]);
  const float amax = fmaxf(warp_max(fmaxf(fabsf(v0), fabsf(v1))), 1e-4f);
  const uint8_t se = pow2_ceil_e8m0(amax * (1.0f / 448.0f));
  const float sc = e8m0_to_f32(se);
  const size_t rid = row_id(pg.table, s0 + r);
  uint8_t* q = pg.lat + rid * R + b * 64 + lane * 2;
  q[0] = f32_to_e4m3(fminf(fmaxf(v0 / sc, -448.0f), 448.0f));
  q[1] = f32_to_e4m3(fminf(fmaxf(v1 / sc, -448.0f), 448.0f));
  if (lane == 0) pg.lat_s[rid * NS + b] = se;
}

// Thread per (pool, dim): idx_pool's arithmetic over the pool's 4 positions, from the tail or the chunk.
__global__ void pool_keys_kernel(const bf16* k, const bf16* g, const bf16* tail_k, const bf16* tail_g,
                                 const float* ape, GlmPaged pg, int s0, int p0, int np) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(np) * ID) return;
  const int p = p0 + int(i / ID), d = int(i % ID);
  float l[POOL], kv[POOL], mx = -INFINITY;
  for (int j = 0; j < POOL; ++j) {
    const int q = p * POOL + j;
    const size_t at = q < s0 ? size_t(q % POOL) * ID + d : size_t(q - s0) * ID + d;
    l[j] = bf16_to_f32(q < s0 ? tail_g[at] : g[at]) + ape[j * ID + d];
    kv[j] = bf16_to_f32(q < s0 ? tail_k[at] : k[at]);
    mx = fmaxf(mx, l[j]);
  }
  float sum = 0.0f;
  for (int j = 0; j < POOL; ++j) {
    l[j] = expf(l[j] - mx);
    sum += l[j];
  }
  float acc = 0.0f;
  for (int j = 0; j < POOL; ++j) acc += round_bf16(round_bf16(l[j] / sum) * kv[j]);
  pool_row(pg, p)[d] = f32_to_bf16(acc);
}

// The tail takes positions [end - end % 4, end) that the chunk wrote (positions before s0 are already there).
__global__ void tail_update_kernel(const bf16* k, const bf16* g, bf16* tail_k, bf16* tail_g, int s0, int end) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, rem = end % POOL;
  if (i >= rem * ID) return;
  const int q = end - rem + i / ID, d = i % ID;
  if (q < s0) return;
  tail_k[(q % POOL) * ID + d] = k[size_t(q - s0) * ID + d];
  tail_g[(q % POOL) * ID + d] = g[size_t(q - s0) * ID + d];
}

#if defined(UH_GFX12)
#define UH_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c)
#endif

// Index scores: a block of 8 waves covers 4 queries x 64 pools. Wave w: query w / 2, heads [16 (w % 2), +16); its
// 16 x 64 scores in four WMMA chains over the 128 key dims, then relu, the head weights, the sum over its 16 heads
// (in-lane over 8, across the half-waves), and its partner wave's half through LDS.
__global__ __launch_bounds__(256) void idx_scores_kernel(const bf16* q, const bf16* w, GlmPaged pg, const int32_t* pos,
                                                         float* out, int ld, int n, int npool, float scale) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sK[64][ID + 8];
  __shared__ float sPart[8][4][16];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int p0 = blockIdx.x * 64, qi = wave >> 1, hh = wave & 1, t = blockIdx.y * 4 + qi;
  for (int u = tid; u < 64 * (ID / 8); u += 256) {  // 64 pools x 16 units of 8 dims
    const int pp = u / (ID / 8), d0 = (u % (ID / 8)) * 8;
    uint4 v = {0, 0, 0, 0};
    if (p0 + pp < npool) v = *reinterpret_cast<const uint4*>(pool_row(pg, p0 + pp) + d0);
    *reinterpret_cast<uint4*>(&sK[pp][d0]) = v;
  }
  bf16x8_t qf[ID / 16];
  float wh[8];
  {
    const int h = hh * 16 + (lane & 15);
    const bf16* qr = q + (size_t(min(t, n - 1)) * 32 + h) * ID + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < ID / 16; ++ks) qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + ks * 16);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      wh[e] = bf16_to_f32(w[size_t(min(t, n - 1)) * 32 + hh * 16 + (lane >> 4) * 8 + e]) * 0.17677669529663687f;
  }
  __syncthreads();
#pragma unroll
  for (int ct = 0; ct < 4; ++ct) {
    float8_t s;
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < ID / 16; ++ks) {
      const bf16x8_t kf = *reinterpret_cast<const bf16x8_t*>(&sK[ct * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
      s = UH_WMMA(qf[ks], kf, s);
    }
    float v = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) v = fmaf(wh[e], fmaxf(s[e] * scale, 0.0f), v);
    v += __shfl_xor(v, 16, 32);
    if (lane < 16) sPart[wave][ct][lane] = v;
  }
  __syncthreads();
  if (hh == 0 && t < n && lane < 16) {
    const int pt = pos[t];
#pragma unroll
    for (int ct = 0; ct < 4; ++ct) {
      const int p = p0 + ct * 16 + lane;
      if (p >= npool) continue;
      const float v = sPart[wave][ct][lane] + sPart[wave + 1][ct][lane];
      out[size_t(t) * ld + p] = p * POOL + POOL - 1 <= pt ? v : -INFINITY;
    }
  }
#endif
}

constexpr int H = 64, KB = 16, NT = 256;

__global__ __launch_bounds__(NT) void mla_attn_kernel(const bf16* __restrict__ q, const int32_t* __restrict__ idx,
                                                      int w, GlmPaged pg, bf16* __restrict__ o, float scale) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sK[KB][R + 8];   // [key][dim]
  __shared__ alignas(16) uint16_t sVt[R][KB + 8];  // [dim][key]
  __shared__ float sS[4][2][16][KB + 1];
  __shared__ alignas(16) uint16_t sP[H][KB + 8];
  __shared__ float sF[H], sM[H], sL[H];
  __shared__ int64_t rowbuf[2][KB];  // the block's row ids (-1: masked), and the next block's being looked up
  const int t = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, dh = wave & 1;
  const int32_t* tidx = idx + size_t(t) * w;

  bf16x8_t qf[16];
  {
    const bf16* qr = q + (size_t(t) * H + hg * 16 + (lane & 15)) * R + dh * 256 + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + ks * 16);
  }
  float8_t acc[16];
#pragma unroll
  for (int n = 0; n < 16; ++n)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[n][e] = 0.0f;
  if (tid < H) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.0f;
  }
  const auto row_of = [&](int e) -> int64_t {
    if (e >= w) return -1;
    const int p = tidx[e];
    return p < 0 ? -1 : int64_t(row_id(pg.table, p));
  };
  if (tid < KB) rowbuf[0][tid] = row_of(tid);
  __syncthreads();
  for (int e0 = 0, buf = 0; e0 < w; e0 += KB, buf ^= 1) {
    const int64_t* rows = rowbuf[buf];
    const bool look = tid >= 32 && tid < 32 + KB && e0 + KB < w;
    const int64_t next = look ? row_of(e0 + KB + tid - 32) : -1;
    // Stage 16 rows x 512 dims, dequantized (e4m3 x power-of-two scale: exact in bf16): unit u -> key u % 16,
    // 32 dims at (u / 16) * 32, all inside one 64-value scale block.
    {
      const int key = tid & 15, d0 = (tid >> 4) * 32;
      const int64_t rid = rows[key];
      alignas(16) uint16_t hv[32];
      if (rid >= 0) {
        const float sc = e8m0_to_f32(pg.lat_s[size_t(rid) * NS + d0 / 64]);
        const uint4* src = reinterpret_cast<const uint4*>(pg.lat + size_t(rid) * R + d0);
        const uint4 c0 = src[0], c1 = src[1];
        const uint32_t cw[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float f[4];
          e4m3x4_to_f32(cw[j], f);
#pragma unroll
          for (int i = 0; i < 4; ++i) hv[j * 4 + i] = f32_to_bf16(f[i] * sc);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) hv[j] = 0;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&sK[key][d0 + j * 8]) = reinterpret_cast<uint4*>(hv)[j];
#pragma unroll
      for (int j = 0; j < 32; ++j) sVt[d0 + j][key] = hv[j];
    }
    __syncthreads();
    {
      float8_t s;
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const bf16x8_t kf = *reinterpret_cast<const bf16x8_t*>(&sK[lane & 15][dh * 256 + ks * 16 + (lane >> 4) * 8]);
        s = UH_WMMA(qf[ks], kf, s);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) sS[hg][dh][(lane >> 4) * 8 + e][lane & 15] = s[e];
    }
    __syncthreads();
    {
      const int h = tid >> 2, k0 = (tid & 3) * 4, gq = h >> 4, hl = h & 15;
      float sv[4], bm = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int key = k0 + j;
        sv[j] = rows[key] >= 0 ? (sS[gq][0][hl][key] + sS[gq][1][hl][key]) * scale : -INFINITY;
        bm = fmaxf(bm, sv[j]);
      }
      bm = fmaxf(bm, __shfl_xor(bm, 1, 32));
      bm = fmaxf(bm, __shfl_xor(bm, 2, 32));
      const float m_old = sM[h];
      const float m_new = fmaxf(m_old, bm);
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float pe = m_new == -INFINITY ? 0.0f : expf(sv[j] - m_new);
        sum += pe;
        sP[h][k0 + j] = f32_to_bf16(pe);
      }
      sum += __shfl_xor(sum, 1, 32);
      sum += __shfl_xor(sum, 2, 32);
      __syncthreads();
      if ((tid & 3) == 0) {
        const float f = m_new == -INFINITY ? 1.0f : expf(m_old - m_new);
        sF[h] = f;
        sL[h] = sL[h] * f + sum;
        sM[h] = m_new;
      }
    }
    __syncthreads();
    {
      const bf16x8_t pf = *reinterpret_cast<const bf16x8_t*>(&sP[hg * 16 + (lane & 15)][(lane >> 4) * 8]);
      float fr[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) fr[e] = sF[hg * 16 + (lane >> 4) * 8 + e];
#pragma unroll
      for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[n][e] *= fr[e];
        const bf16x8_t vf = *reinterpret_cast<const bf16x8_t*>(&sVt[dh * 256 + n * 16 + (lane & 15)][(lane >> 4) * 8]);
        acc[n] = UH_WMMA(pf, vf, acc[n]);
      }
    }
    if (look) rowbuf[buf ^ 1][tid - 32] = next;
    __syncthreads();
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = hg * 16 + (lane >> 4) * 8 + e;
    const float denom = sL[h];
    bf16* orow = o + (size_t(t) * H + h) * R + dh * 256 + (lane & 15);
#pragma unroll
    for (int n = 0; n < 16; ++n) orow[n * 16] = f32_to_bf16(denom > 0.0f ? acc[n][e] / denom : 0.0f);
  }
#endif
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// One block of 128 threads per token: K rounds of a block-wide argmax of sigmoid(logit) + bias (ties: lower index).
__global__ __launch_bounds__(128) void route_kernel(const float* logits, const float* bias, int32_t* sel,
                                                    float* weights, int E, int K, float scale) {
  __shared__ float sv[512];
  __shared__ float rv[4];
  __shared__ int ri[4];
  __shared__ int chosen[16];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* lr = logits + size_t(t) * E;
  for (int e = tid; e < E; e += 128) sv[e] = sigmoidf(lr[e]) + bias[e];
  __syncthreads();
  for (int k = 0; k < K; ++k) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int e = tid; e < E; e += 128)
      if (sv[e] > bv) bv = sv[e], bi = e;  // ascending e: the lowest index among equals
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) rv[warp] = bv, ri[warp] = bi;
    __syncthreads();
    if (tid == 0) {
      for (int w = 1; w < 4; ++w)
        if (rv[w] > rv[0] || (rv[w] == rv[0] && ri[w] < ri[0])) rv[0] = rv[w], ri[0] = ri[w];
      chosen[k] = ri[0];
      sv[ri[0]] = -INFINITY;
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int a = 1; a < K; ++a)
      for (int b = a; b > 0 && chosen[b - 1] > chosen[b]; --b) {
        const int x = chosen[b];
        chosen[b] = chosen[b - 1];
        chosen[b - 1] = x;
      }
    float s[16], sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      s[k] = sigmoidf(lr[chosen[k]]);
      sum += s[k];
    }
    for (int k = 0; k < K; ++k) {
      sel[size_t(t) * K + k] = chosen[k];
      weights[size_t(t) * K + k] = s[k] / (sum + 1e-20f) * scale;
    }
  }
}

// One block of D threads per (token, head): kernels::kda_recurrent's normalization (block_sum, / sqrt(+ 1e-6)).
__global__ void kda_prep_kernel(const bf16* q, const bf16* k, int ld, const float* g, float* qn, float* kn, float* eg,
                                int H, int D) {
  __shared__ float red[32];
  const int t = blockIdx.x / H, h = blockIdx.x % H, j = threadIdx.x;
  const float qv = bf16_to_f32(q[size_t(t) * ld + size_t(h) * D + j]);
  const float kv = bf16_to_f32(k[size_t(t) * ld + size_t(h) * D + j]);
  const float qs = block_sum(qv * qv, red);
  const float ks = block_sum(kv * kv, red);
  const size_t o = (size_t(t) * H + h) * D + j;
  qn[o] = qv / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(D)));
  kn[o] = kv / sqrtf(ks + 1e-6f);
  eg[o] = expf(g[o]);
}

// One block per (head, chunk), thread j owns state column j in registers; tokens 16 at a time through LDS. The
// arithmetic per token is kernels::kda_recurrent's.
template <int D>
__global__ __launch_bounds__(D) void kda_recurrent_kernel(const float* qn, const float* kn, const float* eg,
                                                          const bf16* v, int ldv, const bf16* beta, float* state,
                                                          bf16* o, GlmKdaBatch b, int H) {
  constexpr int TT = 16;
  __shared__ float sq[TT][D], sk[TT][D], se[TT][D];
  __shared__ float sb[TT];
  const int h = blockIdx.x, j = threadIdx.x;
  const GlmKdaChunk ch = b.c[blockIdx.y];
  float* sh = state + (size_t(ch.slot) * H + h) * D * D;
  float s[D];
#pragma unroll
  for (int i = 0; i < D; ++i) s[i] = sh[size_t(i) * D + j];
  for (int t0 = 0; t0 < ch.n; t0 += TT) {
    const int m = min(TT, ch.n - t0);
    __syncthreads();
    for (int i = 0; i < m; ++i) {
      const size_t at = (size_t(ch.t0 + t0 + i) * H + h) * D + j;
      sq[i][j] = qn[at];
      sk[i][j] = kn[at];
      se[i][j] = eg[at];
    }
    if (j < m) sb[j] = bf16_to_f32(beta[size_t(ch.t0 + t0 + j) * H + h]);
    __syncthreads();
    for (int i = 0; i < m; ++i) {
      const size_t t = size_t(ch.t0 + t0 + i);
      float mem = 0.0f;
#pragma unroll
      for (int d = 0; d < D; ++d) {
        s[d] *= se[i][d];
        mem = fmaf(s[d], sk[i][d], mem);
      }
      const float delta = (bf16_to_f32(v[t * ldv + size_t(h) * D + j]) - mem) * sb[i];
      float out = 0.0f;
#pragma unroll
      for (int d = 0; d < D; ++d) {
        s[d] = fmaf(sk[i][d], delta, s[d]);
        out = fmaf(s[d], sq[i][d], out);
      }
      o[(t * H + h) * D + j] = f32_to_bf16(out);
    }
  }
#pragma unroll
  for (int i = 0; i < D; ++i) sh[size_t(i) * D + j] = s[i];
}

}  // namespace

void glm_store_latent(const bf16* lat, GlmPaged pg, int s0, int n, hipStream_t st) {
  if (n <= 0) return;
  store_latent_kernel<<<cdiv(size_t(n) * NS * 32, 256), 256, 0, st>>>(lat, pg, s0, n);
  UH_CHECK_LAUNCH();
}

void glm_pool_keys(const bf16* k, const bf16* g, bf16* tail_k, bf16* tail_g, const float* ape, GlmPaged pg, int s0,
                   int n, hipStream_t st) {
  if (n <= 0) return;
  const int p0 = s0 / POOL, p1 = (s0 + n) / POOL;
  if (p1 > p0) {
    pool_keys_kernel<<<cdiv(size_t(p1 - p0) * ID, 256), 256, 0, st>>>(k, g, tail_k, tail_g, ape, pg, s0, p0, p1 - p0);
    UH_CHECK_LAUNCH();
  }
  if ((s0 + n) % POOL) {
    tail_update_kernel<<<cdiv(size_t(POOL) * ID, 256), 256, 0, st>>>(k, g, tail_k, tail_g, s0, s0 + n);
    UH_CHECK_LAUNCH();
  }
}

void glm_idx_scores(const bf16* q, const bf16* w, GlmPaged pg, const int32_t* pos, float* out, int ld, int n,
                    int npool, float scale, hipStream_t st) {
  if (n <= 0 || npool <= 0) return;
  idx_scores_kernel<<<dim3(cdiv(size_t(npool), 64), cdiv(size_t(n), 4)), 256, 0, st>>>(q, w, pg, pos, out, ld, n,
                                                                                         npool, scale);
  UH_CHECK_LAUNCH();
}

void glm_mla_attn(const bf16* qa, const int32_t* idx, int w, GlmPaged pg, bf16* o, int n, float scale, hipStream_t st) {
  if (n <= 0) return;
  mla_attn_kernel<<<n, NT, 0, st>>>(qa, idx, w, pg, o, scale);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

namespace unholy::kernels {

void glm_route(const float* logits, const float* bias, int32_t* sel, float* weights, int T, int E, int K, float scale,
               hipStream_t st) {
  if (E > 512 || K > 16) throw std::runtime_error("glm_route: E > 512 or K > 16");
  if (T <= 0) return;
  route_kernel<<<T, 128, 0, st>>>(logits, bias, sel, weights, E, K, scale);
  UH_CHECK_LAUNCH();
}

void glm_kda_prep(const bf16* q, const bf16* k, int ld, const float* g, float* qn, float* kn, float* eg, int T, int H,
                  int D, hipStream_t st) {
  if (D != 128) throw std::runtime_error("glm_kda_prep: head dim must be 128");
  if (T <= 0) return;
  kda_prep_kernel<<<T * H, D, 0, st>>>(q, k, ld, g, qn, kn, eg, H, D);
  UH_CHECK_LAUNCH();
}

void glm_kda_recurrent(const float* qn, const float* kn, const float* eg, const bf16* v, int ldv, const bf16* beta,
                       float* state, bf16* o, const GlmKdaBatch& b, int H, int D, hipStream_t st) {
  if (D != 128) throw std::runtime_error("glm_kda_recurrent: head dim must be 128");
  if (b.nchunks <= 0) return;
  kda_recurrent_kernel<128><<<dim3(H, b.nchunks), 128, 0, st>>>(qn, kn, eg, v, ldv, beta, state, o, b, H);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
