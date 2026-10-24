// GLM-5.3-Flash DSA kernels over the paged cache (kernels/glm_fast.hpp).
//
// glm_mla_attn follows attn_wmma.hip's design (one 256-thread block per query computes all 64 heads; wave w =
// (hg, dh) owns heads [16 hg, 16 hg + 16) and dims [256 dh, 256 dh + 256); keys 16 at a time, staged key-major and
// dim-major in LDS; online softmax per head with the probabilities rounded to bf16 before P V), without DSV4's
// window and sink: a query's keys are exactly its index list, read as fp8 latent rows through the block table.
#include "kernels/glm_fast.hpp"

#include <algorithm>
#include <cstdlib>
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

// The tail (a ring of 4 slots, position % 4) takes the chunk's last <= 4 positions, also when the chunk completes its
// pool: if its last positions are rejected drafts, the pool is incomplete again and the next chunk reads the accepted
// ones from here.
__global__ void tail_update_kernel(const bf16* k, const bf16* g, bf16* tail_k, bf16* tail_g, int s0, int end) {
  const int first = max(s0, end - POOL);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (end - first) * ID) return;
  const int q = first + i / ID, d = i % ID;
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

// kPart: block (t, s) takes the s-th of gridDim.y equal spans of the query's entries and writes its unnormalized state
// (acc [64][512], then m and l per head) to part instead of the output.
template <bool kPart>
__global__ __launch_bounds__(NT) void mla_attn_kernel(const bf16* __restrict__ q, const int32_t* __restrict__ idx,
                                                      int w, const int32_t* __restrict__ pos, GlmPaged pg,
                                                      bf16* __restrict__ o, float scale, float* __restrict__ part,
                                                      int span) {
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
  const int cnt = min(w, glm_sel_count(pos[t], (w + 1) / POOL, POOL));  // compact lists: the valid entries come first
  if constexpr (kPart) span = ((cnt + int(gridDim.y) - 1) / int(gridDim.y) + KB - 1) / KB * KB;  // this query's share
  const int lo = kPart ? min(cnt, int(blockIdx.y) * span) : 0, hi = kPart ? min(cnt, lo + span) : cnt;

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
    if (e >= hi) return -1;
    const int p = tidx[e];
    return p < 0 ? -1 : int64_t(row_id(pg.table, p));
  };
  if (tid < KB) rowbuf[0][tid] = row_of(lo + tid);
  __syncthreads();
  for (int e0 = lo, buf = 0; e0 < hi; e0 += KB, buf ^= 1) {
    const int64_t* rows = rowbuf[buf];
    const bool look = tid >= 32 && tid < 32 + KB && e0 + KB < hi;
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
  if constexpr (kPart) {
    float* pb = part + (size_t(t) * gridDim.y + blockIdx.y) * H * (R + 2);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = hg * 16 + (lane >> 4) * 8 + e;
      float* prow = pb + size_t(h) * R + dh * 256 + (lane & 15);
#pragma unroll
      for (int n = 0; n < 16; ++n) prow[n * 16] = acc[n][e];
    }
    if (tid < H) {
      pb[size_t(H) * R + 2 * tid] = sM[tid];
      pb[size_t(H) * R + 2 * tid + 1] = sL[tid];
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = hg * 16 + (lane >> 4) * 8 + e;
      const float denom = sL[h];
      bf16* orow = o + (size_t(t) * H + h) * R + dh * 256 + (lane & 15);
#pragma unroll
      for (int n = 0; n < 16; ++n) orow[n * 16] = f32_to_bf16(denom > 0.0f ? acc[n][e] / denom : 0.0f);
    }
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
                                                          bf16* o, GlmKdaBatch b, int H, float* bak) {
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
      if (bak && t0 + i == ch.save_after) {  // the state after this token (a verify chunk's rollback point)
        float* bh = bak + (size_t(ch.slot) * H + h) * D * D;
#pragma unroll
        for (int d = 0; d < D; ++d) bh[size_t(d) * D + j] = s[d];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < D; ++i) sh[size_t(i) * D + j] = s[i];
}

// kernels::kda_conv's arithmetic, one thread per (channel, chunk).
__global__ void kda_conv_kernel(const bf16* x, int ldx, bf16* state, const bf16* w, bf16* y, GlmKdaBatch b, int C,
                                bf16* bak) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const GlmKdaChunk ch = b.c[blockIdx.y];
  bf16* sc = state + size_t(ch.slot) * 3 * C;
  float w4[4], h[3];
  for (int i = 0; i < 4; ++i) w4[i] = bf16_to_f32(w[size_t(c) * 4 + i]);
  for (int i = 0; i < 3; ++i) h[i] = bf16_to_f32(sc[size_t(i) * C + c]);
  for (int t = 0; t < ch.n; ++t) {
    const float xt = bf16_to_f32(x[size_t(ch.t0 + t) * ldx + c]);
    float acc = w4[0] * h[0];
    acc = fmaf(w4[1], h[1], acc);
    acc = fmaf(w4[2], h[2], acc);
    acc = fmaf(w4[3], xt, acc);
    const float a = round_bf16(acc);
    y[size_t(ch.t0 + t) * C + c] = f32_to_bf16(a * sigmoidf(a));
    h[0] = h[1];
    h[1] = h[2];
    h[2] = xt;
    if (bak && t == ch.save_after)
      for (int i = 0; i < 3; ++i) bak[(size_t(ch.slot) * 3 + i) * C + c] = f32_to_bf16(h[i]);
  }
  for (int i = 0; i < 3; ++i) sc[size_t(i) * C + c] = f32_to_bf16(h[i]);
}

__global__ void add_bf16_kernel(bf16* x, const bf16* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = f32_to_bf16(bf16_to_f32(x[i]) + bf16_to_f32(y[i]));
}

// One block per row: both halves' RMS norms with their weights.
__global__ void mtp_cat_kernel(const bf16* emb, const bf16* h, const bf16* enorm, const bf16* hnorm, bf16* cat, int dim,
                               float eps) {
  __shared__ float red[32];
  const int r = blockIdx.x;
  const bf16* src[2] = {emb + size_t(r) * dim, h + size_t(r) * dim};
  const bf16* wt[2] = {enorm, hnorm};
  for (int half = 0; half < 2; ++half) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
      const float v = bf16_to_f32(src[half][i]);
      ss += v * v;
    }
    const float rs = rsqrtf(block_sum(ss, red) / float(dim) + eps);
    bf16* dst = cat + (size_t(r) * 2 + half) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
      dst[i] = f32_to_bf16(bf16_to_f32(src[half][i]) * rs * bf16_to_f32(wt[half][i]));
  }
}

// One block of 256 threads per row: argmax (lowest id among equals), then the softmax probability of it.
__global__ __launch_bounds__(256) void argmax_conf_kernel(const float* logits, int V, int32_t* tok, float* conf) {
  __shared__ float rv[8];
  __shared__ int ri[8];
  __shared__ float red[32];
  const float* row = logits + size_t(blockIdx.x) * V;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int v = tid; v < V; v += 256)
    if (row[v] > bv) bv = row[v], bi = v;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(bv, o, 32);
    const int oi = __shfl_xor(bi, o, 32);
    if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
  }
  if (lane == 0) rv[warp] = bv, ri[warp] = bi;
  __syncthreads();
  if (tid == 0)
    for (int w = 1; w < 8; ++w)
      if (rv[w] > rv[0] || (rv[w] == rv[0] && ri[w] < ri[0])) rv[0] = rv[w], ri[0] = ri[w];
  __syncthreads();
  const float mx = rv[0];
  float s = 0.0f;
  for (int v = tid; v < V; v += 256) s += expf(row[v] - mx);
  s = block_sum(s, red);
  if (tid == 0) {
    tok[blockIdx.x] = ri[0];
    conf[blockIdx.x] = 1.0f / s;
  }
}

// The split attention's combine: one block per (query, head), each thread two of the 512 dims.
__global__ __launch_bounds__(256) void mla_combine_kernel(const float* part, int splits, bf16* o) {
  const int t = blockIdx.x / H, h = blockIdx.x % H;
  const float* base = part + size_t(t) * splits * H * (R + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, base[size_t(s) * H * (R + 2) + size_t(H) * R + 2 * h]);
  float L = 0.0f, f[64];
  for (int s = 0; s < splits && s < 64; ++s) {
    const float* ml = base + size_t(s) * H * (R + 2) + size_t(H) * R + 2 * h;
    f[s] = ml[0] == -INFINITY ? 0.0f : expf(ml[0] - M);
    L += ml[1] * f[s];
  }
  for (int d = threadIdx.x; d < R; d += blockDim.x) {
    float a = 0.0f;
    for (int s = 0; s < splits && s < 64; ++s) a += base[size_t(s) * H * (R + 2) + size_t(h) * R + d] * f[s];
    o[(size_t(t) * H + h) * R + d] = f32_to_bf16(L > 0.0f ? a / L : 0.0f);
  }
}

// Order-preserving map of fp32 -> uint32 (larger float -> larger key).
__device__ __forceinline__ uint32_t sort_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block (256 threads) per segment: kernels::topk_rows' radix select (4 passes of 8 bits; keys above the
// threshold, then ties by lowest position; -inf never taken), sorted by (value desc, position asc). Segment `seg`
// of row r: vals + r * ld + seg * span, n entries; out_val / out_idx [k] at out_off; an entry's index is
// map[position] when map is given, else seg * span + position.
__global__ __launch_bounds__(256) void topk_seg_kernel(const float* vals, size_t ld, int span, int ncols, int k,
                                                       const int32_t* map, size_t map_ld, float* out_val,
                                                       int32_t* out_idx, size_t out_ld) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t prefix_sel, need_sel, nvalid, count_above;
  __shared__ uint32_t sel_keys[1024];
  __shared__ int32_t sel_pos[1024];
  const int seg = blockIdx.x, r = blockIdx.y;
  const int c0 = seg * span, n = min(span, ncols - c0);
  const float* row = vals + size_t(r) * ld + c0;
  const uint32_t neg_inf_key = sort_key(-INFINITY);
  if (threadIdx.x == 0) nvalid = 0;
  __syncthreads();
  uint32_t local = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += sort_key(row[i]) > neg_inf_key;
  atomicAdd(&nvalid, local);
  __syncthreads();
  const uint32_t want = min(uint32_t(k), nvalid);
  uint32_t prefix = 0, need = want;
  if (want > 0) {
    for (int pass = 3; pass >= 0; --pass) {
      for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
      __syncthreads();
      const uint32_t mask_hi = pass == 3 ? 0u : (0xffffffffu << ((pass + 1) * 8));
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t key = sort_key(row[i]);
        if (key <= neg_inf_key || (key & mask_hi) != (prefix & mask_hi)) continue;
        atomicAdd(&hist[(key >> (pass * 8)) & 0xff], 1u);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t cum = 0;
        int b = 255;
        for (; b > 0; --b) {
          if (cum + hist[b] >= need) break;
          cum += hist[b];
        }
        prefix_sel = prefix | (uint32_t(b) << (pass * 8));
        need_sel = need - cum;
      }
      __syncthreads();
      prefix = prefix_sel;
      need = need_sel;
    }
  }
  if (threadIdx.x == 0) count_above = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n && want > 0; i += blockDim.x) {
    const uint32_t key = sort_key(row[i]);
    if (key > prefix && key > neg_inf_key) {
      const uint32_t slot = atomicAdd(&count_above, 1u);
      sel_keys[slot] = key;
      sel_pos[slot] = i;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32 && want > 0) {
    const uint32_t base = count_above;
    uint32_t taken = 0;
    for (int i0 = 0; i0 < n && taken < need; i0 += 32) {
      const int i = i0 + threadIdx.x;
      const bool tie = i < n && sort_key(row[i]) == prefix;
      const uint64_t ballot = __ballot(tie);
      const uint32_t before = __popcll(ballot & ((1ull << threadIdx.x) - 1));
      if (tie && taken + before < need) {
        sel_keys[base + taken + before] = prefix;
        sel_pos[base + taken + before] = i;
      }
      taken += __popcll(ballot);
    }
  }
  __syncthreads();
  int size = 1;
  while (size < k) size <<= 1;
  for (int i = threadIdx.x; i < size; i += blockDim.x)
    if (uint32_t(i) >= want) {
      sel_keys[i] = 0;
      sel_pos[i] = 0x7fffffff;
    }
  __syncthreads();
  for (int len = 2; len <= size; len <<= 1)
    for (int stride = len >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & len) == 0;
          const bool i_first = sel_keys[i] > sel_keys[j] || (sel_keys[i] == sel_keys[j] && sel_pos[i] < sel_pos[j]);
          if (desc ? !i_first : i_first) {
            const uint32_t tk = sel_keys[i];
            sel_keys[i] = sel_keys[j];
            sel_keys[j] = tk;
            const int32_t tp = sel_pos[i];
            sel_pos[i] = sel_pos[j];
            sel_pos[j] = tp;
          }
        }
      }
      __syncthreads();
    }
  const size_t ob = size_t(r) * out_ld + size_t(seg) * k;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const bool ok = uint32_t(i) < want;
    const int p = sel_pos[i];
    if (out_val) out_val[ob + i] = ok ? row[p] : -INFINITY;
    out_idx[ob + i] = ok ? (map ? map[size_t(r) * map_ld + c0 + p] : c0 + p) : -1;
  }
}

__global__ void moe_combine_slices_kernel(const bf16* pair_out, const int32_t* pidx, const bf16* shared, bf16* out,
                                          int K, int dim) {
  const int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  float acc = 0.0f;
  for (int j = 0; j < K; ++j) acc += bf16_to_f32(pair_out[size_t(pidx[size_t(t) * K + j]) * dim + d]);
  out[size_t(t) * dim + d] = f32_to_bf16(acc + bf16_to_f32(shared[size_t(t) * dim + d]));
}

// Block-stride over the compact tokens; each thread sums 8 columns (16-byte loads, two 16-byte stores).
__global__ __launch_bounds__(256) void remote_partial_kernel(const bf16* pair_out, const int32_t* u2p,
                                                             const int* n_tokens, float* dst, int K, int dim) {
  const int U = *n_tokens, cols = dim / 8;
  for (int u = blockIdx.x; u < U; u += gridDim.x)
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      float acc[8] = {};
      for (int j = 0; j < K; ++j) {
        const int p = u2p[size_t(u) * K + j];
        if (p < 0) continue;
        const uint4 r = reinterpret_cast<const uint4*>(pair_out + size_t(p) * dim)[c];
        const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          acc[2 * k] += bf16_to_f32(uint16_t(w[k]));
          acc[2 * k + 1] += bf16_to_f32(uint16_t(w[k] >> 16));
        }
      }
      float4* d = reinterpret_cast<float4*>(dst + size_t(u) * dim + size_t(c) * 8);
      d[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      d[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
  __threadfence_system();
}

__global__ void copy_partial_kernel(const float* src, float* dst, const int* n_tokens, int dim) {
  const size_t n = size_t(*n_tokens) * dim / 4;  // uint4 units
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) d[i] = s[i];
}

__global__ void moe_combine_partial_slices_kernel(const bf16* pair_out, const int32_t* pidx, const int* remote0,
                                                  const int32_t* tok2u, const float* partial, const bf16* shared,
                                                  bf16* out, int K, int dim) {
  const int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  const int r0 = *remote0, u = tok2u[t];
  float acc = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int p = pidx[size_t(t) * K + j];
    if (p < r0) acc += bf16_to_f32(pair_out[size_t(p) * dim + d]);
  }
  if (u >= 0) acc += partial[size_t(u) * dim + d];
  out[size_t(t) * dim + d] = f32_to_bf16(acc + bf16_to_f32(shared[size_t(t) * dim + d]));
}

// The compact selection list: entry j < 4 v (v valid pools) -> pool j / 4's token j % 4, then the tail, then -1.
__global__ void idx_expand_compact_kernel(const int32_t* pools, const int32_t* pos, int32_t* idx, int n, int k,
                                          int pool) {
  const int w = k * pool + pool - 1;
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(n) * w) return;
  const int t = int(i / w), j = int(i % w), p = pos[t];
  const int full = (p + 1) / pool, valid = full < k ? full : k, tail = (p + 1) % pool;
  int v = -1;
  if (j < valid * pool) v = pools[size_t(t) * k + j / pool] * pool + j % pool;
  else if (j < valid * pool + tail) v = p + 1 - tail + (j - valid * pool);
  idx[i] = v;
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
  tail_update_kernel<<<cdiv(size_t(POOL) * ID, 256), 256, 0, st>>>(k, g, tail_k, tail_g, s0, s0 + n);
  UH_CHECK_LAUNCH();
}

void glm_idx_scores(const bf16* q, const bf16* w, GlmPaged pg, const int32_t* pos, float* out, int ld, int n,
                    int npool, float scale, hipStream_t st) {
  if (n <= 0 || npool <= 0) return;
  idx_scores_kernel<<<dim3(cdiv(size_t(npool), 64), cdiv(size_t(n), 4)), 256, 0, st>>>(q, w, pg, pos, out, ld, n,
                                                                                         npool, scale);
  UH_CHECK_LAUNCH();
}

void glm_mla_attn(const bf16* qa, const int32_t* idx, int w, const int32_t* pos, GlmPaged pg, bf16* o, int n,
                  float scale, hipStream_t st) {
  if (n <= 0) return;
  mla_attn_kernel<false><<<n, NT, 0, st>>>(qa, idx, w, pos, pg, o, scale, nullptr, 0);
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
                       float* state, bf16* o, const GlmKdaBatch& b, int H, int D, hipStream_t st, float* backup) {
  if (D != 128) throw std::runtime_error("glm_kda_recurrent: head dim must be 128");
  if (b.nchunks <= 0) return;
  kda_recurrent_kernel<128><<<dim3(H, b.nchunks), 128, 0, st>>>(qn, kn, eg, v, ldv, beta, state, o, b, H, backup);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

namespace unholy::kernels {

void glm_kda_conv(const bf16* x, int ldx, bf16* state, const bf16* w, bf16* y, const GlmKdaBatch& b, int C,
                  hipStream_t st, bf16* backup) {
  if (b.nchunks <= 0) return;
  kda_conv_kernel<<<dim3(cdiv(size_t(C), 256), b.nchunks), 256, 0, st>>>(x, ldx, state, w, y, b, C, backup);
  UH_CHECK_LAUNCH();
}

void glm_add_bf16(bf16* x, const bf16* y, size_t n, hipStream_t st) {
  if (n == 0) return;
  add_bf16_kernel<<<std::min(cdiv(n, 256), 4096), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}

void glm_mtp_cat(const bf16* emb, const bf16* h, const bf16* enorm, const bf16* hnorm, bf16* cat, int rows, int dim,
                 float eps, hipStream_t st) {
  if (rows <= 0) return;
  mtp_cat_kernel<<<rows, 256, 0, st>>>(emb, h, enorm, hnorm, cat, dim, eps);
  UH_CHECK_LAUNCH();
}

void glm_argmax_conf(const float* logits, int rows, int V, int32_t* tok, float* conf, hipStream_t st) {
  if (rows <= 0) return;
  argmax_conf_kernel<<<rows, 256, 0, st>>>(logits, V, tok, conf);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

namespace unholy::kernels {

size_t glm_topk_cand_bytes(int rows, int cols, int k) {
  const size_t segs = size_t((cols + kGlmTopkSeg - 1) / kGlmTopkSeg);
  return size_t(rows) * segs * k * 8;
}

void glm_topk(const float* scores, int ld, int n, int cols, int k, int32_t* idx, void* cand, hipStream_t st) {
  if (n <= 0 || cols <= 0) return;
  if (k > 1024) throw std::runtime_error("glm_topk: k > 1024");
  // UNHOLY_GLM_TOPK_SEG (tests): a smaller segment, so short sequences take the two-stage path (cand must hold the
  // extra segments: the GLM scratch has room for 8).
  static const int seg_len = [] {
    const char* v = std::getenv("UNHOLY_GLM_TOPK_SEG");
    return v ? std::clamp(std::atoi(v), 1, kGlmTopkSeg) : kGlmTopkSeg;
  }();
  if (cols <= seg_len) {
    topk_seg_kernel<<<dim3(1, n), 256, 0, st>>>(scores, size_t(ld), cols, cols, k, nullptr, 0, nullptr, idx, size_t(k));
    UH_CHECK_LAUNCH();
    return;
  }
  const int segs = (cols + seg_len - 1) / seg_len;
  float* cv = static_cast<float*>(cand);
  int32_t* ci = reinterpret_cast<int32_t*>(cv + size_t(n) * segs * k);
  const size_t cl = size_t(segs) * k;
  topk_seg_kernel<<<dim3(segs, n), 256, 0, st>>>(scores, size_t(ld), seg_len, cols, k, nullptr, 0, cv, ci, cl);
  UH_CHECK_LAUNCH();
  topk_seg_kernel<<<dim3(1, n), 256, 0, st>>>(cv, cl, int(cl), int(cl), k, ci, cl, nullptr, idx, size_t(k));
  UH_CHECK_LAUNCH();
}

size_t glm_mla_part_bytes(int n, int splits) { return size_t(n) * splits * H * (R + 2) * 4; }

void glm_mla_attn_split(const bf16* qa, const int32_t* idx, int w, const int32_t* pos, GlmPaged pg, bf16* o, int n,
                        float scale, int splits, float* part, hipStream_t st) {
  if (n <= 0) return;
  if (splits < 1 || splits > 64) throw std::runtime_error("glm_mla_attn_split: 1..64 splits");
  mla_attn_kernel<true><<<dim3(n, splits), NT, 0, st>>>(qa, idx, w, pos, pg, o, scale, part, 0);
  UH_CHECK_LAUNCH();
  mla_combine_kernel<<<n * H, 256, 0, st>>>(part, splits, o);
  UH_CHECK_LAUNCH();
}

void glm_remote_partial(const bf16* pair_out, const int32_t* u2p, const int* n_tokens, float* dst, int K, int dim,
                        int max_tokens, hipStream_t st) {
  if (max_tokens <= 0) return;
  remote_partial_kernel<<<std::min(max_tokens, 1024), 256, 0, st>>>(pair_out, u2p, n_tokens, dst, K, dim);
  UH_CHECK_LAUNCH();
}

void glm_copy_partial(const float* src, float* dst, const int* n_tokens, int dim, int max_tokens, hipStream_t st) {
  const size_t units = size_t(max_tokens) * dim / 4;
  if (units == 0) return;
  copy_partial_kernel<<<int(std::min<size_t>((units + 255) / 256, 1024)), 256, 0, st>>>(src, dst, n_tokens, dim);
  UH_CHECK_LAUNCH();
}

void glm_moe_combine_partial(const bf16* pair_out, const int32_t* pidx, const int* remote0, const int32_t* tok2u,
                             const float* partial, const bf16* shared, bf16* out, int T, int K, int dim,
                             hipStream_t st) {
  if (T <= 0) return;
  moe_combine_partial_slices_kernel<<<dim3(T, (dim + 255) / 256), 256, 0, st>>>(pair_out, pidx, remote0, tok2u,
                                                                                partial, shared, out, K, dim);
  UH_CHECK_LAUNCH();
}

void glm_moe_combine(const bf16* pair_out, const int32_t* pidx, const bf16* shared, bf16* out, int T, int K, int dim,
                     hipStream_t st) {
  if (T <= 0) return;
  moe_combine_slices_kernel<<<dim3(T, (dim + 255) / 256), 256, 0, st>>>(pair_out, pidx, shared, out, K, dim);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

namespace unholy::kernels {

void glm_idx_expand(const int32_t* pools, const int32_t* pos, int32_t* idx, int n, int k, int pool, hipStream_t st) {
  const size_t total = size_t(n) * (k * pool + pool - 1);
  if (total == 0) return;
  idx_expand_compact_kernel<<<cdiv(total, 256), 256, 0, st>>>(pools, pos, idx, n, k, pool);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
