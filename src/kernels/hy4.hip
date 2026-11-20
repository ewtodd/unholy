// Hy4 flash kernels (kernels/hy4.hpp).
//
// hy4_mla_attn follows glm_mla_attn's design (kernels/glm_fast.hip): one block per query computes all heads; wave
// w = (hg, dh) owns heads [16 hg, 16 hg + 16) and half the score dimensions (dh 0: latent dims 0-255; dh 1: latent
// dims 256-511 and the 64 roped dims); keys 16 at a time staged key-major and (latent only) dim-major in LDS; online
// softmax per head starting from the head's sink (m = sink, l = 1, no value), probabilities rounded to bf16 before
// P V. The keys are bf16 rows (no fp8 latent), read through the block table (sparse layers) or from the window ring
// and the chunk's own rows (window layers).
#include "kernels/hy4.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernels/dev.hip.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf(float x) { return x / (1.0f + expf(-x)); }

// ---- hyper-connections ----------------------------------------------------------------------------------------

constexpr int kHcThreads = 256, kMaxMix = 8;

// One block per token: the flat streams' fp32 sum of squares and the NM mixes, then the weighted stream sum.
template <int NM>
__device__ void hc_mix(const bf16* xt, const float* fn, int width, float eps, float* mix, float* red) {
  float ss = 0.0f, d[NM];
#pragma unroll
  for (int j = 0; j < NM; ++j) d[j] = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = bf16_to_f32(xt[i]);
    ss = fmaf(v, v, ss);
#pragma unroll
    for (int j = 0; j < NM; ++j) d[j] = fmaf(fn[size_t(j) * width + i], v, d[j]);
  }
  const float rs = rsqrtf(block_sum(ss, red) / float(width) + eps);
#pragma unroll
  for (int j = 0; j < NM; ++j) mix[j] = block_sum(d[j], red) * rs;
}

__global__ __launch_bounds__(kHcThreads) void ihc_pre_kernel(const bf16* x, const float* fn, const float* scale,
                                                             const float* base, bf16* y, float* post, int hc, int dim,
                                                             float eps, float hc_eps, float mag, const bf16* norm_w,
                                                             bf16* hn) {
  __shared__ float red[32];
  __shared__ float sy[4096];
  const int t = blockIdx.x;
  const bf16* xt = x + size_t(t) * hc * dim;
  float mix[kMaxMix];
  hc_mix<kMaxMix>(xt, fn, hc * dim, eps, mix, red);  // hc == 4: 2 hc == 8 mixes
  float pre[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pre[i] = sigmoidf(mix[i] * scale[0] + base[i]) + hc_eps;
    if (threadIdx.x == 0) post[size_t(t) * hc + i] = mag * sigmoidf(mix[4 + i] * scale[1] + base[4 + i]) + hc_eps;
  }
  float ss = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += pre[i] * bf16_to_f32(xt[size_t(i) * dim + d]);
    const bf16 v = f32_to_bf16(acc);
    y[size_t(t) * dim + d] = v;
    const float fv = bf16_to_f32(v);
    sy[d] = fv;
    ss = fmaf(fv, fv, ss);
  }
  if (!norm_w) return;
  const float rs = rsqrtf(block_sum(ss, red) / float(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    hn[size_t(t) * dim + d] = f32_to_bf16(bf16_to_f32(norm_w[d]) * round_bf16(sy[d] * rs));
}

__global__ void ihc_post_kernel(const bf16* out, bf16* x, const float* post, int T, int hc, int dim) {
  const size_t n = size_t(T) * hc * dim;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int d = int(i % dim);
    const size_t th = i / dim;  // t * hc + stream
    const size_t t = th / hc;
    x[i] = f32_to_bf16(post[th] * bf16_to_f32(out[t * dim + d]) + bf16_to_f32(x[i]));
  }
}

__global__ __launch_bounds__(kHcThreads) void hc_head_kernel(const bf16* x, const float* fn, const float* scale,
                                                             const float* base, bf16* y, int hc, int dim, float eps,
                                                             float hc_eps) {
  __shared__ float red[32];
  const int t = blockIdx.x;
  const bf16* xt = x + size_t(t) * hc * dim;
  float mix[4];
  hc_mix<4>(xt, fn, hc * dim, eps, mix, red);
  float pre[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) pre[i] = sigmoidf(mix[i] * scale[0] + base[i]) + hc_eps;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += pre[i] * bf16_to_f32(xt[size_t(i) * dim + d]);
    y[size_t(t) * dim + d] = f32_to_bf16(acc);
  }
}

// ---- RoPE ------------------------------------------------------------------------------------------------------

// inv_freq[i] = 1 / theta^(2i / 64), as transformers computes it (fp32), from the host.
struct InvFreq {
  float f[32];
};
InvFreq inv_freq(float theta) {
  InvFreq r;
  for (int i = 0; i < 32; ++i) r.f[i] = 1.0f / std::pow(theta, float(2 * i) / 64.0f);
  return r;
}

__global__ void rope_kernel(bf16* x, int ld, int rows, int heads, int hstride, int off, const int32_t* pos,
                            InvFreq fr) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(rows) * heads * 32) return;
  const int pr = int(i % 32), h = int((i / 32) % heads), r = int(i / (32 * size_t(heads)));
  bf16* p = x + size_t(r) * ld + size_t(h) * hstride + off + 2 * pr;
  const float a = __fmul_rn(float(pos[r]), fr.f[pr]);
  const float c = round_bf16(cosf(a)), s = round_bf16(sinf(a));
  const float x0 = bf16_to_f32(p[0]), x1 = bf16_to_f32(p[1]);
  p[0] = f32_to_bf16(__fadd_rn(__fmul_rn(x0, c), __fmul_rn(-x1, s)));
  p[1] = f32_to_bf16(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
}

// ---- attention -------------------------------------------------------------------------------------------------

#if defined(UH_GFX12)
#define UH_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c)
#endif

constexpr int R = kHy4Row, L = kHy4Lat, KB = 16;

__device__ __forceinline__ const bf16* key_row(const Hy4Keys& k, int p) {
  if (k.table)
    return k.pages + (size_t(k.table[p / kHy4PageRows]) * kHy4PageRows + p % kHy4PageRows) * R;
  return p >= k.s0 ? k.chunk + size_t(p - k.s0) * R : k.ring + size_t(p % kHy4Window) * R;
}

template <int H>
__global__ __launch_bounds__(H * 4) void mla_attn_kernel(const bf16* __restrict__ q, Hy4Keys keys,
                                                         const int32_t* __restrict__ idx, int w, int window,
                                                         const int32_t* __restrict__ pos, const float* __restrict__ sinks,
                                                         bf16* __restrict__ o, float scale) {
#if defined(UH_GFX12)
  constexpr int NT = H * 4, HG = H / 16;
  __shared__ alignas(16) uint16_t sK[KB][R + 8];   // [key][dim]
  __shared__ alignas(16) uint16_t sVt[L][KB + 8];  // [dim][key], latent only
  __shared__ float sS[HG][2][16][KB + 1];
  __shared__ alignas(16) uint16_t sP[H][KB + 8];
  __shared__ float sF[H], sM[H], sL[H];
  __shared__ int sOk[KB];
  const int t = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, dh = wave & 1;
  const int pt = pos[t];
  int cnt, lo = 0;
  if (idx) {  // the compact selection: min(groups before the query's, k) groups, then the query's own group
    const int k = (w - kHy4Pool) / kHy4Pool;
    cnt = min(pt / kHy4Pool, k) * kHy4Pool + pt % kHy4Pool + 1;
  } else {
    lo = window > 0 ? max(0, pt - window + 1) : 0;
    cnt = pt - lo + 1;
  }
  const int32_t* tidx = idx ? idx + size_t(t) * w : nullptr;

  // Query fragments: dh 0 holds dims 0-255 (16 steps), dh 1 dims 256-511 and 512-575 (20 steps).
  constexpr int QS = 20;
  bf16x8_t qf[QS];
  {
    const bf16* qr = q + (size_t(t) * H + hg * 16 + (lane & 15)) * R + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < QS; ++ks) {
      const int d0 = dh == 0 ? ks * 16 : 256 + ks * 16;
      if (dh == 0 && ks >= 16) qf[ks] = qf[0];
      else qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + d0);
    }
  }
  float8_t acc[16];
#pragma unroll
  for (int n = 0; n < 16; ++n)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[n][e] = 0.0f;
  if (tid < H) {
    sM[tid] = sinks[tid];
    sL[tid] = 1.0f;
  }
  for (int e0 = 0; e0 < cnt; e0 += KB) {
    __syncthreads();
    if (tid < KB) {
      const int e = e0 + tid;
      int p = -1;
      if (e < cnt) p = tidx ? tidx[e] : lo + e;
      sOk[tid] = p;
    }
    __syncthreads();
    // Stage 16 rows x 576 dims: unit u -> key u / 72, 8 dims at (u % 72) * 8.
    for (int u = tid; u < KB * (R / 8); u += NT) {
      const int key = u / (R / 8), d0 = (u % (R / 8)) * 8;
      const int p = sOk[key];
      uint4 v = {0, 0, 0, 0};
      if (p >= 0) v = *reinterpret_cast<const uint4*>(key_row(keys, p) + d0);
      *reinterpret_cast<uint4*>(&sK[key][d0]) = v;
      if (d0 < L) {
        const uint16_t* hv = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) sVt[d0 + j][key] = hv[j];
      }
    }
    __syncthreads();
    {
      float8_t s;
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < QS; ++ks) {
        if (dh == 0 && ks >= 16) break;
        const int d0 = dh == 0 ? ks * 16 : 256 + ks * 16;
        const bf16x8_t kf = *reinterpret_cast<const bf16x8_t*>(&sK[lane & 15][d0 + (lane >> 4) * 8]);
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
        sv[j] = sOk[key] >= 0 ? (sS[gq][0][hl][key] + sS[gq][1][hl][key]) * scale : -INFINITY;
        bm = fmaxf(bm, sv[j]);
      }
      bm = fmaxf(bm, __shfl_xor(bm, 1, 32));
      bm = fmaxf(bm, __shfl_xor(bm, 2, 32));
      const float m_old = sM[h];
      const float m_new = fmaxf(m_old, bm);
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float pe = sv[j] == -INFINITY ? 0.0f : expf(sv[j] - m_new);
        sum += pe;
        sP[h][k0 + j] = f32_to_bf16(pe);
      }
      sum += __shfl_xor(sum, 1, 32);
      sum += __shfl_xor(sum, 2, 32);
      __syncthreads();
      if ((tid & 3) == 0) {
        const float f = expf(m_old - m_new);
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
  }
  __syncthreads();
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = hg * 16 + (lane >> 4) * 8 + e;
    const float denom = sL[h];
    bf16* orow = o + (size_t(t) * H + h) * L + dh * 256 + (lane & 15);
#pragma unroll
    for (int n = 0; n < 16; ++n) orow[n * 16] = f32_to_bf16(acc[n][e] / denom);
  }
#endif
}

__global__ void split_q_kernel(const bf16* q, bf16* qn, int T, int H) {
  const size_t n = size_t(T) * H * 192;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    qn[i] = q[(i / 192) * 256 + i % 192];
}

__global__ void pack_q_kernel(const bf16* qa, const bf16* q, bf16* out, int T, int H) {
  const size_t n = size_t(T) * H * R;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t th = i / R;
    const int d = int(i % R);
    out[i] = d < L ? qa[th * L + d] : q[th * 256 + 192 + (d - L)];
  }
}

__global__ void gate_mul_kernel(bf16* o, const bf16* g, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    o[i] = f32_to_bf16(bf16_to_f32(o[i]) * round_bf16(sigmoidf(bf16_to_f32(g[i]))));
}

__global__ void ring_store_kernel(const bf16* rows, bf16* ring, int s0, int n) {
  const int first = max(0, n - kHy4Window);
  const size_t total = size_t(n - first) * R;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int r = first + int(i / R), d = int(i % R);
    ring[size_t((s0 + r) % kHy4Window) * R + d] = rows[size_t(r) * R + d];
  }
}

__global__ void page_store_kernel(const bf16* rows, bf16* pages, const int32_t* table, int s0, int n) {
  const size_t total = size_t(n) * (R / 8);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int r = int(i / (R / 8)), d0 = int(i % (R / 8)) * 8, p = s0 + r;
    bf16* dst = pages + (size_t(table[p / kHy4PageRows]) * kHy4PageRows + p % kHy4PageRows) * R + d0;
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(rows + size_t(r) * R + d0);
  }
}

// ---- indexer ---------------------------------------------------------------------------------------------------

constexpr int ID = kHy4IdxDim, POOL = kHy4Pool, PPP = kHy4PageRows / POOL;

__device__ __forceinline__ bf16* pool_row(const Hy4Pooled& pg, int p) {
  return pg.pk + (size_t(pg.table[p / PPP]) * PPP + p % PPP) * ID;
}

// One block of 128 threads (thread = output dim) per completed group.
__global__ __launch_bounds__(ID) void idx_pool_kernel(const bf16* tk, const bf16* tail, const float* wv,
                                                      const float* wg, const float* ape, const float* knorm, float eps,
                                                      InvFreq fr, Hy4Pooled pg, int s0, int p0) {
  __shared__ float sk[POOL][ID];
  __shared__ float red[32];
  __shared__ float sp[ID];
  const int p = p0 + blockIdx.x, d = threadIdx.x;
  for (int j = 0; j < POOL; ++j) {
    const int q = p * POOL + j;
    sk[j][d] = bf16_to_f32(q < s0 ? tail[(q % POOL) * ID + d] : tk[size_t(q - s0) * ID + d]);
  }
  __syncthreads();
  float v[POOL], g[POOL], mx = -INFINITY;
#pragma unroll
  for (int j = 0; j < POOL; ++j) {
    float av = 0.0f, ag = 0.0f;
    for (int c = 0; c < ID; ++c) {
      av = fmaf(wv[d * ID + c], sk[j][c], av);
      ag = fmaf(wg[d * ID + c], sk[j][c], ag);
    }
    v[j] = av;
    g[j] = ag + ape[j * ID + d];
    mx = fmaxf(mx, g[j]);
  }
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < POOL; ++j) {
    g[j] = expf(g[j] - mx);
    sum += g[j];
  }
  float pooled = 0.0f;
#pragma unroll
  for (int j = 0; j < POOL; ++j) pooled += v[j] * (g[j] / sum);
  const float rs = rsqrtf(block_sum(pooled * pooled, red) / float(ID) + eps);
  sp[d] = round_bf16(knorm[d] * (pooled * rs));
  __syncthreads();
  float out = sp[d];
  if (d >= ID - 64) {  // RoPE on the last 64 dims at the group's first position
    const int k = d - (ID - 64), pr = k / 2;
    const float a = __fmul_rn(float(p * POOL), fr.f[pr]);
    const float c = round_bf16(cosf(a)), s = round_bf16(sinf(a));
    const float x0 = sp[d & ~1], x1 = sp[d | 1];
    out = (k & 1) == 0 ? __fadd_rn(__fmul_rn(x0, c), __fmul_rn(-x1, s)) : __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
  }
  pool_row(pg, p)[d] = f32_to_bf16(out);
}

__global__ void idx_tail_kernel(const bf16* tk, bf16* tail, int s0, int end) {
  const int first = max(s0, end - POOL);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (end - first) * ID) return;
  const int q = first + i / ID, d = i % ID;
  tail[(q % POOL) * ID + d] = tk[size_t(q - s0) * ID + d];
}

// A block of 8 waves covers 8 queries x 64 groups; wave w: query w, its 16 heads x 64 groups in four WMMA chains.
__global__ __launch_bounds__(256) void idx_scores_kernel(const bf16* q, const bf16* hw, float wscale, Hy4Pooled pg,
                                                         const int32_t* pos, float* out, int ld, int n, int npool) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sK[64][ID + 8];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int p0 = blockIdx.x * 64, t = blockIdx.y * 8 + wave;
  for (int u = tid; u < 64 * (ID / 8); u += 256) {
    const int pp = u / (ID / 8), d0 = (u % (ID / 8)) * 8;
    uint4 v = {0, 0, 0, 0};
    if (p0 + pp < npool) v = *reinterpret_cast<const uint4*>(pool_row(pg, p0 + pp) + d0);
    *reinterpret_cast<uint4*>(&sK[pp][d0]) = v;
  }
  const int tq = min(t, n - 1);
  bf16x8_t qf[ID / 16];
  float wh[8];
  {
    const bf16* qr = q + (size_t(tq) * kHy4IdxHeads + (lane & 15)) * ID + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < ID / 16; ++ks) qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + ks * 16);
#pragma unroll
    for (int e = 0; e < 8; ++e) wh[e] = bf16_to_f32(hw[size_t(tq) * kHy4IdxHeads + (lane >> 4) * 8 + e]) * wscale;
  }
  __syncthreads();
  const int own = t < n ? pos[t] / POOL : 0;
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
    for (int e = 0; e < 8; ++e) v = fmaf(wh[e], fmaxf(s[e], 0.0f), v);
    v += __shfl_xor(v, 16, 32);
    const int p = p0 + ct * 16 + (lane & 15);
    if (t < n && lane < 16 && p < npool) out[size_t(t) * ld + p] = p < own ? v : -INFINITY;
  }
#endif
}

__global__ void idx_expand_kernel(const int32_t* groups, const int32_t* pos, int32_t* idx, int n, int k) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const int pt = pos[t], own = pt / POOL, w = k * POOL + POOL;
  int32_t* row = idx + size_t(t) * w;
  int c = 0;
  for (int j = 0; j < k; ++j) {
    const int g = groups[size_t(t) * k + j];
    if (g < 0 || g >= own) continue;
    for (int m = 0; m < POOL; ++m) row[c++] = g * POOL + m;
  }
  for (int q = own * POOL; q <= pt; ++q) row[c++] = q;
  while (c < w) row[c++] = -1;
}

// ---- Engram ----------------------------------------------------------------------------------------------------

// nn.RMSNorm(eps=None) on bf16 inputs: ATen takes the epsilon of the compute type (fp32), not bf16's 2^-7 as
// the docstring says (checked against torch 2.13).
constexpr float kRmsEps = 1.1920928955078125e-07f;

// One block of 16 x 96 threads... in 6 waves per head: block per token, thread per (head, dim) over 1536.
__global__ __launch_bounds__(256) void engram_rows_kernel(const uint8_t* q, const bf16* s, const bf16* w, bf16* out) {
  __shared__ float sv[16 * 96];
  __shared__ float sr[16];
  const int t = blockIdx.x;
  for (int i = threadIdx.x; i < 16 * 96; i += blockDim.x) {
    const int h = i / 96;
    const float sc = bf16_to_f32(s[size_t(t) * 16 + h]);
    sv[i] = round_bf16(e4m3_to_f32(q[size_t(t) * 1536 + i]) * sc);
  }
  __syncthreads();
  if (threadIdx.x < 16 * 8) {  // 8 lanes per head
    const int h = threadIdx.x / 8, l = threadIdx.x % 8;
    float ss = 0.0f;
    for (int d = l; d < 96; d += 8) ss = fmaf(sv[h * 96 + d], sv[h * 96 + d], ss);
    ss += __shfl_xor(ss, 1, 8);
    ss += __shfl_xor(ss, 2, 8);
    ss += __shfl_xor(ss, 4, 8);
    if (l == 0) sr[h] = rsqrtf(ss / 96.0f + kRmsEps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 16 * 96; i += blockDim.x)
    out[size_t(t) * 1536 + i] = f32_to_bf16(sv[i] * sr[i / 96] * bf16_to_f32(w[i]));
}

// nn.RMSNorm over dim values held in sbuf (fp32 of bf16 values): returns the scale; one block.
__device__ float rms_scale(const float* v, int dim, float* red) {
  float ss = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) ss = fmaf(v[d], v[d], ss);
  return rsqrtf(block_sum(ss, red) / float(dim) + kRmsEps);
}

constexpr int kEngDim = 3072;

__global__ __launch_bounds__(256) void engram_mix_kernel(const float* kv, const bf16* bias, const bf16* nk,
                                                         const bf16* nq, const bf16* pre_gamma, const bf16* hcs,
                                                         const uint8_t* mask, bf16* x, int hc) {
  __shared__ float red[32];
  __shared__ float sa[kEngDim], sb[kEngDim], sm[kEngDim];
  const int t = blockIdx.x;
  if (!mask[t]) return;
  constexpr int D = kEngDim;
  const float* kvt = kv + size_t(t) * 5 * D;
  // main = bf16(pre_gamma * bf16(rms(value)))
  for (int d = threadIdx.x; d < D; d += blockDim.x) sa[d] = round_bf16(kvt[4 * D + d] + bf16_to_f32(bias[4 * D + d]));
  __syncthreads();
  float rs = rms_scale(sa, D, red);
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    sm[d] = round_bf16(bf16_to_f32(pre_gamma[d]) * round_bf16(sa[d] * rs));
  bf16* xt = x + size_t(t) * hc * D;
  for (int i = 0; i < hc; ++i) {
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      sa[d] = round_bf16(kvt[size_t(i) * D + d] + bf16_to_f32(bias[size_t(i) * D + d]));  // key projection
      sb[d] = bf16_to_f32(xt[size_t(i) * D + d]);                                        // the stream (query)
    }
    __syncthreads();
    const float rk = rms_scale(sa, D, red), rq = rms_scale(sb, D, red);
    float dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      const float key = round_bf16(sa[d] * rk * bf16_to_f32(nk[size_t(i) * D + d]));
      const float qry = round_bf16(sb[d] * rq * bf16_to_f32(nq[size_t(i) * D + d]));
      dot += round_bf16(key * qry);
    }
    float g = round_bf16(block_sum(dot, red));
    g = round_bf16(g * (1.0f / sqrtf(float(D))));
    const float mag = round_bf16(sqrtf(fmaxf(fabsf(g), 1e-6f)));
    g = round_bf16(mag * (g > 0.0f ? 1.0f : g < 0.0f ? -1.0f : 0.0f));
    g = round_bf16(sigmoidf(g));
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      const float br = round_bf16(bf16_to_f32(hcs[size_t(i) * D + d]) * sm[d]);
      const float add = round_bf16(round_bf16(g * sm[d]) + br);
      xt[size_t(i) * D + d] = f32_to_bf16(sb[d] + add);
    }
  }
}

// ---- MLP / MoE -------------------------------------------------------------------------------------------------

__global__ void swiglu_kernel(const bf16* g, const bf16* u, int ld, bf16* h, int rows, int I) {
  const size_t n = size_t(rows) * I;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / I, c = i % I;
    const float a = round_bf16(siluf(bf16_to_f32(g[r * ld + c])));
    h[i] = f32_to_bf16(a * bf16_to_f32(u[r * ld + c]));
  }
}

__global__ void swiglu_clamp_kernel(const bf16* gu, bf16* h, int rows, int I, float lim, const int* rows_dev) {
  const int nr = rows_dev ? *rows_dev : rows;
  const size_t n = size_t(nr) * I;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / I, c = i % I;
    const float g = fminf(bf16_to_f32(gu[r * 2 * I + c]), lim);
    const float u = fminf(fmaxf(bf16_to_f32(gu[r * 2 * I + I + c]), -lim), lim);
    h[i] = f32_to_bf16(round_bf16(siluf(g)) * u);
  }
}

__global__ void scale_rows_kernel(bf16* y, const float* w, int rows, int dim, const int* r0_dev, const int* r1_dev) {
  const int r0 = r0_dev ? *r0_dev : 0, r1 = r1_dev ? *r1_dev : rows;
  const size_t n = size_t(max(0, r1 - r0)) * dim;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = r0 + i / dim;
    y[r * dim + i % dim] = f32_to_bf16(bf16_to_f32(y[r * dim + i % dim]) * w[r]);
  }
}

__global__ void moe_sum_kernel(const bf16* pair_out, const int32_t* pidx, const int* remote0, const int32_t* tok2u,
                               const float* remote, bf16* out, int T, int K, int dim) {
  const size_t n = size_t(T) * dim;
  const int r0 = remote0 ? *remote0 : 1 << 30;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t t = i / dim, d = i % dim;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
      const int p = pidx[t * K + j];
      if (p < r0) acc += bf16_to_f32(pair_out[size_t(p) * dim + d]);
    }
    if (remote && tok2u) {
      const int u = tok2u[t];
      if (u >= 0) acc += remote[size_t(u) * dim + d];
    }
    out[i] = f32_to_bf16(acc);
  }
}

__global__ void gather_remote_kernel(const bf16* in, const int32_t* p2u, const MoePlan* plan, bf16* xa, int dim) {
  const int r0 = plan->remote0, P = plan->P, v = dim / 8;
  const size_t n = size_t(max(0, P - r0)) * v;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t p = r0 + i / v, c = i % v;
    reinterpret_cast<uint4*>(xa + p * dim)[c] = reinterpret_cast<const uint4*>(in + size_t(p2u[p]) * dim)[c];
  }
}

__global__ void add_kernel(const bf16* a, const bf16* b, bf16* out, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    out[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

__global__ void bias_kernel(const float* x, const bf16* bias, bf16* out, int rows, int cols) {
  const size_t n = size_t(rows) * cols;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    out[i] = f32_to_bf16(x[i] + bf16_to_f32(bias[i % cols]));
}

// ---- fp8 experts -----------------------------------------------------------------------------------------------

__device__ __forceinline__ int c_row(int i, int e, int lane) {
#if defined(UH_GFX12)
  return i * 16 + (lane >> 4) * 8 + e;
#else
  return i * 16 + 2 * e + (lane >> 4);
#endif
}

constexpr int BM = kHy4TileRows, BN = 128, GNT = 256;

// gemm_wmma.hip's bf16_kernel with the weights e4m3 x bf16 block scales (dequantized to bf16 as the oracle does) and
// the rows of one expert tile per blockIdx.y.
// The NVFP4 scale byte (unsigned e4m3; 0 and 0x7f mean zero, as ggml_ue4m3_to_fp32).
__device__ __forceinline__ float ue4m3_to_f32(uint8_t x) { return x == 0 || x == 0x7f ? 0.0f : e4m3_to_f32(x & 0x7f); }

// Eight consecutive weights k .. k + 7 (k % 8 == 0) of NVFP4 row `row` (codes [N][K / 2], scales [N][K / 16]; within
// each 16-value group byte j holds values j (low nibble) and j + 8): their bf16 values, exact, packed in pairs.
__device__ __forceinline__ uint4 nvfp4_eight(const uint8_t* W, const uint8_t* S, size_t row, int K, int k) {
  const uint2 raw = *reinterpret_cast<const uint2*>(W + row * (K / 2) + (k / 16) * 8);
  const float s = ue4m3_to_f32(S[row * (K / 16) + k / 16]);
  const int sh = (k & 8) ? 4 : 0;
  float f[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) f[j] = e2m1_to_f32(((j < 4 ? raw.x >> (8 * j) : raw.y >> (8 * (j - 4))) >> sh) & 15u) * s;
  const auto pair = [&](int j) { return uint32_t(f32_to_bf16(f[j])) | (uint32_t(f32_to_bf16(f[j + 1])) << 16); };
  return uint4{pair(0), pair(2), pair(4), pair(6)};
}

// kFmt 0: e4m3 [N][K] with bf16 128x128 scales (s_tab -> bf16); 1: NVFP4 (s_tab -> uint8 [N][K / 16]).
template <int kFmt>
__global__ __launch_bounds__(GNT) void fp8_grouped_kernel(const bf16* __restrict__ A, int K, const Exl3Tile* tiles,
                                                          int ntiles, const int* ntiles_dev,
                                                          const uint8_t* const* w_tab, const void* const* s_tab,
                                                          bf16* __restrict__ C, int ldc, int N) {
#if defined(UH_GFX12) || defined(UH_GFX11)
  const int nt = ntiles_dev ? *ntiles_dev : ntiles;
  if (int(blockIdx.y) >= nt) return;
  const Exl3Tile tile = tiles[blockIdx.y];
  const uint8_t* W = w_tab[tile.expert];
  const void* Sv = s_tab[tile.expert];
  if (!W || tile.rows <= 0) return;
  constexpr int KS = 32, STR = KS + 8;
  __shared__ alignas(16) uint16_t sA[BM * STR];
  __shared__ alignas(16) uint16_t sW[BN * STR];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = blockIdx.x * BN, M = tile.rows;
  const bf16* Ag = A + size_t(tile.row0) * K;
  const int kb = K / 128;
  float8_t acc[4];
  for (int i = 0; i < 4; ++i)
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += KS) {
    {
      const int r = tid >> 2, c = (tid & 3) * 8;
      uint4 v = {0, 0, 0, 0};
      if (r < M) v = *reinterpret_cast<const uint4*>(Ag + size_t(r) * K + k0 + c);
      *reinterpret_cast<uint4*>(&sA[r * STR + c]) = v;
    }
    for (int it = tid; it < BN * 4; it += GNT) {
      const int r = it >> 2, c = (it & 3) * 8;
      uint4 vals = {0, 0, 0, 0};
      if (n0 + r < N) {
        const size_t row = n0 + r;
        if constexpr (kFmt == 1) {
          vals = nvfp4_eight(W, static_cast<const uint8_t*>(Sv), row, K, k0 + c);
        } else {
          const bf16* S = static_cast<const bf16*>(Sv);
          const uint2 raw = *reinterpret_cast<const uint2*>(W + row * K + k0 + c);
          const float s = bf16_to_f32(S[(row / 128) * kb + (k0 + c) / 128]);
          float f[8];
          e4m3x4_to_f32(raw.x, f);
          e4m3x4_to_f32(raw.y, f + 4);
          const auto pair = [&](int j) {
            return uint32_t(f32_to_bf16(f[j] * s)) | (uint32_t(f32_to_bf16(f[j + 1] * s)) << 16);
          };
          vals = uint4{pair(0), pair(2), pair(4), pair(6)};
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
  if (n < N)
    for (int i = 0; i < 4; ++i)
      for (int e = 0; e < 8; ++e) {
        const int m = c_row(i, e, lane);
        if (m < M) C[size_t(tile.row0 + m) * ldc + n] = f32_to_bf16(acc[i][e]);
      }
#endif
}

__global__ void fp8_to_mx_kernel(const uint8_t* w, const bf16* s, int kb, int row0, int N, int K, uint8_t* codes,
                                 uint8_t* scales) {
  const int groups = K / 32;
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(N) * groups) return;
  const int r = int(i / groups), g = int(i % groups), sr = row0 + r;
  const float sc = bf16_to_f32(s[size_t(sr / 128) * kb + (g * 32) / 128]);
  float v[32], amax = 0.0f;
  for (int j = 0; j < 32; ++j) {
    v[j] = round_bf16(e4m3_to_f32(w[size_t(sr) * K + g * 32 + j]) * sc);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  int best = -127;
  if (amax > 0.0f) {
    const int e_hi = int(ceilf(log2f(amax / 6.0f)));
    float best_err = INFINITY;
    for (int e = e_hi; e >= e_hi - 1; --e) {
      const float inv = exp2f(float(-e)), sv = exp2f(float(e));
      float err = 0.0f;
      for (int j = 0; j < 32; ++j) {
        const float q = round_e2m1(fminf(fmaxf(v[j] * inv, -6.0f), 6.0f)) * sv;
        err += (q - v[j]) * (q - v[j]);
      }
      if (err < best_err) best_err = err, best = e;
    }
  }
  best = max(-127, min(127, best));
  scales[size_t(r) * groups + g] = uint8_t(best + 127);
  const float inv = exp2f(float(-best));
  uint8_t* out = codes + size_t(r) * (K / 2) + g * 16;
  for (int j = 0; j < 32; j += 2) {
    uint32_t nib[2];
    for (int k2 = 0; k2 < 2; ++k2) {
      const float q = round_e2m1(fminf(fmaxf(v[j + k2] * inv, -6.0f), 6.0f));
      const float a = fabsf(q);
      const uint32_t m = a < 2.0f ? uint32_t(a * 2.0f) : (a == 6.0f ? 7u : uint32_t(a) + 2u);
      nib[k2] = m | (q < 0.0f ? 8u : 0u);
    }
    out[j / 2] = uint8_t(nib[0] | (nib[1] << 4));
  }
}

__global__ void swiglu_clamp2_kernel(const bf16* g, const bf16* u, bf16* h, int rows, int I, float lim) {
  const size_t n = size_t(rows) * I;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const float gv = fminf(bf16_to_f32(g[i]), lim);
    const float uv = fminf(fmaxf(bf16_to_f32(u[i]), -lim), lim);
    h[i] = f32_to_bf16(round_bf16(siluf(gv)) * uv);
  }
}

constexpr int kGemvCols = 4;  // output columns per wave

template <int kFmt>
__global__ __launch_bounds__(256) void fp8_gemv_kernel(const bf16* __restrict__ A, int ldi, int K, const Exl3Tile* tiles,
                                                       int ntiles, const int* ntiles_dev, const uint8_t* const* w_tab,
                                                       const void* const* s_tab, bf16* __restrict__ C, int ldc, int N) {
  const int nt = ntiles_dev ? *ntiles_dev : ntiles;
  if (int(blockIdx.y) >= nt) return;
  const Exl3Tile tile = tiles[blockIdx.y];
  const uint8_t* W = w_tab[tile.expert];
  const void* Sv = s_tab[tile.expert];
  if (!W || tile.rows <= 0) return;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int c0 = (blockIdx.x * 8 + wave) * kGemvCols;
  if (c0 >= N) return;
  const int rows = min(tile.rows, kHy4GemvRows), kb = K / 128;
  const bf16* Ar = A + size_t(tile.row0) * ldi;
  float acc[kGemvCols][kHy4GemvRows];
#pragma unroll
  for (int j = 0; j < kGemvCols; ++j)
#pragma unroll
    for (int r = 0; r < kHy4GemvRows; ++r) acc[j][r] = 0.0f;
  for (int k0 = lane * 16; k0 < K; k0 += 32 * 16) {
    float x[kHy4GemvRows][16];
#pragma unroll
    for (int r = 0; r < kHy4GemvRows; ++r) {
      if (r < rows) {
        const uint4 a0 = *reinterpret_cast<const uint4*>(Ar + size_t(r) * ldi + k0);
        const uint4 a1 = *reinterpret_cast<const uint4*>(Ar + size_t(r) * ldi + k0 + 8);
        const uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          x[r][2 * q] = bf16_to_f32(uint16_t(aw[q]));
          x[r][2 * q + 1] = bf16_to_f32(uint16_t(aw[q] >> 16));
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kGemvCols; ++j) {
      const int col = c0 + j;
      if (col >= N) break;
      float wf[16];
      if constexpr (kFmt == 1) {
        const uint2 raw = *reinterpret_cast<const uint2*>(W + size_t(col) * (K / 2) + k0 / 2);
        const float sc = ue4m3_to_f32(static_cast<const uint8_t*>(Sv)[size_t(col) * (K / 16) + k0 / 16]);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t byte = (q < 4 ? raw.x >> (8 * q) : raw.y >> (8 * (q - 4))) & 0xffu;
          wf[q] = e2m1_to_f32(byte & 15u) * sc;
          wf[q + 8] = e2m1_to_f32(byte >> 4) * sc;
        }
      } else {
        const bf16* S = static_cast<const bf16*>(Sv);
        const uint4 wv = *reinterpret_cast<const uint4*>(W + size_t(col) * K + k0);
        const float sc = bf16_to_f32(S[size_t(col / 128) * kb + k0 / 128]);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          e4m3x4_to_f32(ww[q], wf + 4 * q);
#pragma unroll
          for (int i = 0; i < 4; ++i) wf[4 * q + i] = round_bf16(wf[4 * q + i] * sc);
        }
      }
#pragma unroll
      for (int r = 0; r < kHy4GemvRows; ++r) {
        if (r < rows) {
          float a = acc[j][r];
#pragma unroll
          for (int i = 0; i < 16; ++i) a = fmaf(wf[i], x[r][i], a);
          acc[j][r] = a;
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < kGemvCols; ++j) {
    const int col = c0 + j;
#pragma unroll
    for (int r = 0; r < kHy4GemvRows; ++r) {
      if (r >= rows) continue;
      const float v = warp_sum(acc[j][r]);
      if (lane == 0 && col < N) C[size_t(tile.row0 + r) * ldc + col] = f32_to_bf16(v);
    }
  }
}

// GGUF NVFP4 blocks (64 values: 4 scale bytes, then 32 code bytes) of an [N][K] matrix -> codes [N][K / 2] and scales
// [N][K / 16] (a block's codes are already in group order: group g's 8 bytes at 8 g).
__global__ void nvfp4_split_kernel(const uint8_t* blocks, int N, int K, uint8_t* codes, uint8_t* scales) {
  const size_t nb = size_t(N) * (K / 64);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nb; i += size_t(gridDim.x) * blockDim.x) {
    const uint8_t* b = blocks + i * 36;
    const size_t row = i / (K / 64), blk = i % (K / 64);
    for (int s = 0; s < 4; ++s) scales[row * (K / 16) + blk * 4 + s] = b[s];
    uint8_t* dst = codes + row * (K / 2) + blk * 32;
    for (int j = 0; j < 32; ++j) dst[j] = b[4 + j];
  }
}

__global__ void dense_tiles_kernel(Exl3Tile* t, int rows, int tile_rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r0 = i * tile_rows;
  if (r0 < rows) t[i] = Exl3Tile{0, r0, min(tile_rows, rows - r0)};
}

__global__ void dequant_fp8_kernel(const uint8_t* w, const bf16* s, bf16* out, int N, int K) {
  const size_t n = size_t(N) * K;
  const int kb = (K + 127) / 128;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / K, c = i % K;
    out[i] = f32_to_bf16(e4m3_to_f32(w[i]) * bf16_to_f32(s[(r / 128) * kb + c / 128]));
  }
}

__global__ void bf16_f32_kernel(const bf16* x, float* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = bf16_to_f32(x[i]);
}

int grid1d(size_t n) { return int(std::min<size_t>((n + 255) / 256, 8192)); }

}  // namespace

void hy4_ihc_pre(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post, int T,
                 int hc, int dim, float eps, float hc_eps, float mag, hipStream_t st, const bf16* norm_w, bf16* hn) {
  if (T <= 0) return;
  if (hc != 4 || dim > 4096) throw std::runtime_error("hy4_ihc_pre: hc 4, dim <= 4096");
  ihc_pre_kernel<<<T, kHcThreads, 0, st>>>(x, fn, scale, base, y, post, hc, dim, eps, hc_eps, mag, norm_w, hn);
  UH_CHECK_LAUNCH();
}

void hy4_ihc_post(const bf16* out, bf16* x, const float* post, int T, int hc, int dim, hipStream_t st) {
  if (T <= 0) return;
  ihc_post_kernel<<<grid1d(size_t(T) * hc * dim), 256, 0, st>>>(out, x, post, T, hc, dim);
  UH_CHECK_LAUNCH();
}

void hy4_hc_head(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, int T, int hc,
                 int dim, float eps, float hc_eps, hipStream_t st) {
  if (T <= 0) return;
  if (hc != 4) throw std::runtime_error("hy4_hc_head: hc 4");
  hc_head_kernel<<<T, kHcThreads, 0, st>>>(x, fn, scale, base, y, hc, dim, eps, hc_eps);
  UH_CHECK_LAUNCH();
}

void hy4_rope(bf16* x, int ld, int rows, int heads, int hstride, int off, const int32_t* pos, float theta,
              hipStream_t st) {
  if (rows <= 0) return;
  const size_t n = size_t(rows) * heads * 32;
  rope_kernel<<<cdiv(n, 256), 256, 0, st>>>(x, ld, rows, heads, hstride, off, pos, inv_freq(theta));
  UH_CHECK_LAUNCH();
}

void hy4_mla_attn(const bf16* q, int H, const Hy4Keys& keys, const int32_t* idx, int w, int window,
                  const int32_t* pos, const float* sinks, bf16* o, int n, float scale, hipStream_t st) {
  if (n <= 0) return;
  if (H == 32) mla_attn_kernel<32><<<n, 128, 0, st>>>(q, keys, idx, w, window, pos, sinks, o, scale);
  else if (H == 64) mla_attn_kernel<64><<<n, 256, 0, st>>>(q, keys, idx, w, window, pos, sinks, o, scale);
  else throw std::runtime_error("hy4_mla_attn: 32 or 64 heads");
  UH_CHECK_LAUNCH();
}

void hy4_split_q(const bf16* q, bf16* qn, int T, int H, hipStream_t st) {
  if (T <= 0) return;
  split_q_kernel<<<grid1d(size_t(T) * H * 192), 256, 0, st>>>(q, qn, T, H);
  UH_CHECK_LAUNCH();
}

void hy4_pack_q(const bf16* qa, const bf16* q, bf16* out, int T, int H, hipStream_t st) {
  if (T <= 0) return;
  pack_q_kernel<<<grid1d(size_t(T) * H * R), 256, 0, st>>>(qa, q, out, T, H);
  UH_CHECK_LAUNCH();
}

void hy4_gate_mul(bf16* o, const bf16* g, size_t n, hipStream_t st) {
  if (!n) return;
  gate_mul_kernel<<<grid1d(n), 256, 0, st>>>(o, g, n);
  UH_CHECK_LAUNCH();
}

void hy4_ring_store(const bf16* rows, bf16* ring, int s0, int n, hipStream_t st) {
  if (n <= 0) return;
  ring_store_kernel<<<grid1d(size_t(std::min(n, kHy4Window)) * R), 256, 0, st>>>(rows, ring, s0, n);
  UH_CHECK_LAUNCH();
}

void hy4_page_store(const bf16* rows, bf16* pages, const int32_t* table, int s0, int n, hipStream_t st) {
  if (n <= 0) return;
  page_store_kernel<<<grid1d(size_t(n) * (R / 8)), 256, 0, st>>>(rows, pages, table, s0, n);
  UH_CHECK_LAUNCH();
}

void hy4_idx_pool(const bf16* tk, bf16* tail, const float* wv, const float* wg, const float* ape, const float* knorm,
                  float eps, float theta, Hy4Pooled pg, int s0, int n, hipStream_t st) {
  if (n <= 0) return;
  const int end = s0 + n, p0 = s0 / POOL, p1 = end / POOL;  // groups whose last member is in [s0, end)
  if (p1 > p0)
    idx_pool_kernel<<<p1 - p0, ID, 0, st>>>(tk, tail, wv, wg, ape, knorm, eps, inv_freq(theta), pg, s0, p0);
  const int first = std::max(s0, end - POOL);
  idx_tail_kernel<<<cdiv(size_t(end - first) * ID, 256), 256, 0, st>>>(tk, tail, s0, end);
  UH_CHECK_LAUNCH();
}

void hy4_idx_scores(const bf16* q, const bf16* hw, float wscale, Hy4Pooled pg, const int32_t* pos, float* out, int ld,
                    int n, int npool, hipStream_t st) {
  if (n <= 0 || npool <= 0) return;
  idx_scores_kernel<<<dim3(cdiv(size_t(npool), 64), cdiv(size_t(n), 8)), 256, 0, st>>>(q, hw, wscale, pg, pos, out,
                                                                                      ld, n, npool);
  UH_CHECK_LAUNCH();
}

void hy4_idx_expand(const int32_t* groups, const int32_t* pos, int32_t* idx, int n, int k, hipStream_t st) {
  if (n <= 0) return;
  idx_expand_kernel<<<cdiv(size_t(n), 64), 64, 0, st>>>(groups, pos, idx, n, k);
  UH_CHECK_LAUNCH();
}

void hy4_engram_rows(const uint8_t* q, const bf16* s, const bf16* w, bf16* out, int T, hipStream_t st) {
  if (T <= 0) return;
  engram_rows_kernel<<<T, 256, 0, st>>>(q, s, w, out);
  UH_CHECK_LAUNCH();
}

void hy4_engram_mix(const float* kv, const bf16* bias, const bf16* nk, const bf16* nq, const bf16* pre_gamma,
                    const bf16* hcs, const uint8_t* mask, bf16* x, int T, int hc, int dim, hipStream_t st) {
  if (T <= 0) return;
  if (dim != kEngDim) throw std::runtime_error("hy4_engram_mix: dim 3072");
  engram_mix_kernel<<<T, 256, 0, st>>>(kv, bias, nk, nq, pre_gamma, hcs, mask, x, hc);
  UH_CHECK_LAUNCH();
}

void hy4_swiglu(const bf16* g, const bf16* u, int ld, bf16* h, int rows, int I, hipStream_t st) {
  if (rows <= 0) return;
  swiglu_kernel<<<grid1d(size_t(rows) * I), 256, 0, st>>>(g, u, ld, h, rows, I);
  UH_CHECK_LAUNCH();
}

void hy4_swiglu_clamp(const bf16* gu, bf16* h, int rows, int I, float limit, hipStream_t st, const int* rows_dev) {
  if (rows <= 0) return;
  swiglu_clamp_kernel<<<grid1d(size_t(rows) * I), 256, 0, st>>>(gu, h, rows, I, limit, rows_dev);
  UH_CHECK_LAUNCH();
}

void hy4_scale_rows(bf16* y, const float* w, int rows, int dim, hipStream_t st, const int* r0_dev, const int* r1_dev) {
  if (rows <= 0) return;
  scale_rows_kernel<<<grid1d(size_t(rows) * dim), 256, 0, st>>>(y, w, rows, dim, r0_dev, r1_dev);
  UH_CHECK_LAUNCH();
}

void hy4_moe_sum(const bf16* pair_out, const int32_t* pidx, const int* remote0, const int32_t* tok2u,
                 const float* remote, bf16* out, int T, int K, int dim, hipStream_t st) {
  if (T <= 0) return;
  moe_sum_kernel<<<grid1d(size_t(T) * dim), 256, 0, st>>>(pair_out, pidx, remote0, tok2u, remote, out, T, K, dim);
  UH_CHECK_LAUNCH();
}

void hy4_gather_remote(const bf16* in, const int32_t* p2u, const MoePlan* plan, bf16* xa, int dim, int max_rows,
                       hipStream_t st) {
  if (max_rows <= 0) return;
  if (dim % 8) throw std::runtime_error("hy4_gather_remote: dim % 8");
  gather_remote_kernel<<<grid1d(size_t(max_rows) * (dim / 8)), 256, 0, st>>>(in, p2u, plan, xa, dim);
  UH_CHECK_LAUNCH();
}

void hy4_add(const bf16* a, const bf16* b, bf16* out, size_t n, hipStream_t st) {
  if (!n) return;
  add_kernel<<<grid1d(n), 256, 0, st>>>(a, b, out, n);
  UH_CHECK_LAUNCH();
}

void hy4_bias_bf16(const float* x, const bf16* bias, bf16* out, int rows, int cols, hipStream_t st) {
  if (rows <= 0) return;
  bias_kernel<<<grid1d(size_t(rows) * cols), 256, 0, st>>>(x, bias, out, rows, cols);
  UH_CHECK_LAUNCH();
}

void hy4_fp8_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                          const bf16* const* s_tab, bf16* out, int ldo, int N, hipStream_t st, const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % 128 || N % 128) throw std::runtime_error("hy4_fp8_grouped_gemm: K, N % 128");
  fp8_grouped_kernel<0><<<dim3(N / BN, ntiles), GNT, 0, st>>>(xa, K, tiles, ntiles, ntiles_dev, w_tab,
                                                              reinterpret_cast<const void* const*>(s_tab), out, ldo, N);
  UH_CHECK_LAUNCH();
}

void hy4_fp8_to_mxfp4(const uint8_t* w, const bf16* s, int kb, int row0, int N, int K, uint8_t* codes, uint8_t* scales,
                      hipStream_t st) {
  if (K % 32) throw std::runtime_error("hy4_fp8_to_mxfp4: K % 32");
  fp8_to_mx_kernel<<<cdiv(size_t(N) * (K / 32), 128), 128, 0, st>>>(w, s, kb, row0, N, K, codes, scales);
  UH_CHECK_LAUNCH();
}

void hy4_swiglu_clamp2(const bf16* g, const bf16* u, bf16* h, int rows, int I, float limit, hipStream_t st) {
  if (rows <= 0) return;
  swiglu_clamp2_kernel<<<grid1d(size_t(rows) * I), 256, 0, st>>>(g, u, h, rows, I, limit);
  UH_CHECK_LAUNCH();
}

void hy4_fp8_grouped_gemv(const bf16* xa, int ldi, int K, const Exl3Tile* tiles, int ntiles,
                          const uint8_t* const* w_tab, const bf16* const* s_tab, bf16* out, int ldo, int N,
                          hipStream_t st, const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % 512 || ldi % 8) throw std::runtime_error("hy4_fp8_grouped_gemv: K % 512, ldi % 8");
  const int cols_per_block = 8 * kGemvCols;
  fp8_gemv_kernel<0><<<dim3((N + cols_per_block - 1) / cols_per_block, ntiles), 256, 0, st>>>(
      xa, ldi, K, tiles, ntiles, ntiles_dev, w_tab, reinterpret_cast<const void* const*>(s_tab), out, ldo, N);
  UH_CHECK_LAUNCH();
}

void hy4_nvfp4_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                            const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st,
                            const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % 128 || N % 128) throw std::runtime_error("hy4_nvfp4_grouped_gemm: K, N % 128");
  fp8_grouped_kernel<1><<<dim3(N / BN, ntiles), GNT, 0, st>>>(xa, K, tiles, ntiles, ntiles_dev, w_tab,
                                                              reinterpret_cast<const void* const*>(s_tab), out, ldo, N);
  UH_CHECK_LAUNCH();
}

void hy4_nvfp4_grouped_gemv(const bf16* xa, int ldi, int K, const Exl3Tile* tiles, int ntiles,
                            const uint8_t* const* w_tab, const uint8_t* const* s_tab, bf16* out, int ldo, int N,
                            hipStream_t st, const int* ntiles_dev) {
  if (ntiles <= 0) return;
  if (K % 512 || ldi % 8) throw std::runtime_error("hy4_nvfp4_grouped_gemv: K % 512, ldi % 8");
  const int cols_per_block = 8 * kGemvCols;
  fp8_gemv_kernel<1><<<dim3((N + cols_per_block - 1) / cols_per_block, ntiles), 256, 0, st>>>(
      xa, ldi, K, tiles, ntiles, ntiles_dev, w_tab, reinterpret_cast<const void* const*>(s_tab), out, ldo, N);
  UH_CHECK_LAUNCH();
}

void hy4_nvfp4_split(const uint8_t* blocks, int N, int K, uint8_t* codes, uint8_t* scales, hipStream_t st) {
  if (K % 64) throw std::runtime_error("hy4_nvfp4_split: K % 64");
  nvfp4_split_kernel<<<grid1d(size_t(N) * (K / 64)), 256, 0, st>>>(blocks, N, K, codes, scales);
  UH_CHECK_LAUNCH();
}

void hy4_dense_tiles(Exl3Tile* tiles, int rows, int tile_rows, hipStream_t st) {
  if (rows <= 0) return;
  const int n = (rows + tile_rows - 1) / tile_rows;
  dense_tiles_kernel<<<(n + 63) / 64, 64, 0, st>>>(tiles, rows, tile_rows);
  UH_CHECK_LAUNCH();
}

void hy4_dequant_fp8(const uint8_t* w, const bf16* s, bf16* out, int N, int K, hipStream_t st) {
  dequant_fp8_kernel<<<grid1d(size_t(N) * K), 256, 0, st>>>(w, s, out, N, K);
  UH_CHECK_LAUNCH();
}

void hy4_bf16_to_f32(const bf16* x, float* y, size_t n, hipStream_t st) {
  if (!n) return;
  bf16_f32_kernel<<<grid1d(n), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
