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
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
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

// hy4_ihc_fused: grid (S = hc * dim / kHcSlice, T), 256 threads. Block (s, t) first, with `out`, adds the previous
// sublayer's output to its slice of token t's streams (rmsnorm_hf of the out row, its scale recomputed by every
// block in rmsnorm_hf's order, then ihc_post), then writes its slice's sum of squares and 2 hc mix dots to ws; the
// token's last block to arrive (counter, reset by it) adds the slices in order and finishes as ihc_pre does.
constexpr int kHcSlice = 512;

__global__ __launch_bounds__(kHcThreads) void ihc_fused_kernel(const bf16* out, const bf16* out_norm,
                                                               const float* post_in, bf16* x, const float* fn,
                                                               const float* scale, const float* base, bf16* y,
                                                               float* post_out, int hc, int dim, float eps,
                                                               float hc_eps, float mag, const bf16* norm_w, bf16* hn,
                                                               float* ws, uint32_t* count) {
  constexpr int NV = 1 + kMaxMix;  // sum of squares, 2 hc == 8 dots
  __shared__ float red[32];
  __shared__ float wred[kHcThreads / 32][NV];
  __shared__ float tot[NV];
  __shared__ float sy[4096];
  __shared__ bool last;
  const int t = blockIdx.y, sl = blockIdx.x, S = gridDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int width = hc * dim, i0 = sl * kHcSlice, stream = i0 / dim, d0 = i0 % dim;
  bf16* xt = x + size_t(t) * width;
  float rs_out = 0.0f, pin = 0.0f;
  if (out) {
    const bf16* orow = out + size_t(t) * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += kHcThreads) {
      const float v = bf16_to_f32(orow[i]);
      ss += v * v;
    }
    ss = block_sum(ss, red);
    rs_out = rsqrtf(ss / float(dim) + eps);
    pin = post_in[size_t(t) * hc + stream];
  }
  float v9[NV];
#pragma unroll
  for (int j = 0; j < NV; ++j) v9[j] = 0.0f;
  for (int k = tid; k < kHcSlice; k += kHcThreads) {
    const int fi = i0 + k;
    float v;
    if (out) {
      const int d = d0 + k;
      const float on = bf16_to_f32(f32_to_bf16(bf16_to_f32(out_norm[d]) * round_bf16(bf16_to_f32(out[size_t(t) * dim + d]) * rs_out)));
      const bf16 nx = f32_to_bf16(pin * on + bf16_to_f32(xt[fi]));
      xt[fi] = nx;
      v = bf16_to_f32(nx);
    } else {
      v = bf16_to_f32(xt[fi]);
    }
    v9[0] = fmaf(v, v, v9[0]);
#pragma unroll
    for (int j = 0; j < kMaxMix; ++j) v9[1 + j] = fmaf(fn[size_t(j) * width + fi], v, v9[1 + j]);
  }
#pragma unroll
  for (int j = 0; j < NV; ++j) v9[j] = warp_sum(v9[j]);
  if (lane == 0)
#pragma unroll
    for (int j = 0; j < NV; ++j) wred[warp][j] = v9[j];
  __syncthreads();
  if (tid < NV) {
    float a = 0.0f;
    for (int w = 0; w < kHcThreads / 32; ++w) a += wred[w][tid];
    ws[(size_t(t) * S + sl) * NV + tid] = a;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&count[t], 1u) == unsigned(S - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (tid < NV) {
    float a = 0.0f;
    for (int s2 = 0; s2 < S; ++s2) a += __builtin_nontemporal_load(&ws[(size_t(t) * S + s2) * NV + tid]);
    tot[tid] = a;
  }
  if (tid == 0) count[t] = 0;
  __syncthreads();
  const float rs = rsqrtf(tot[0] / float(width) + eps);
  float pre[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pre[i] = sigmoidf(tot[1 + i] * rs * scale[0] + base[i]) + hc_eps;
    if (tid == 0) post_out[size_t(t) * hc + i] = mag * sigmoidf(tot[5 + i] * rs * scale[1] + base[4 + i]) + hc_eps;
  }
  float ss = 0.0f;
  for (int d = tid; d < dim; d += kHcThreads) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += pre[i] * bf16_to_f32(xt[size_t(i) * dim + d]);
    const bf16 vb = f32_to_bf16(acc);
    y[size_t(t) * dim + d] = vb;
    const float fv = bf16_to_f32(vb);
    sy[d] = fv;
    ss = fmaf(fv, fv, ss);
  }
  if (!norm_w) return;
  const float rs2 = rsqrtf(block_sum(ss, red) / float(dim) + eps);
  for (int d = tid; d < dim; d += kHcThreads)
    hn[size_t(t) * dim + d] = f32_to_bf16(bf16_to_f32(norm_w[d]) * round_bf16(sy[d] * rs2));
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

// M-RoPE's plane of rotary pair i (interleaved sections [12, 10, 10]: t, h, w).
__device__ __forceinline__ int mrope_plane(int i) { return i < 30 && i % 3 != 0 ? i % 3 : 0; }

// Rotary pair pr (of 32) of row r: src -> dst (may be the same).
__device__ __forceinline__ void rope_pair(const bf16* src, bf16* dst, int r, int pr, const int32_t* pos,
                                          const int32_t* pos3, const InvFreq& fr) {
  const int ps = pos3 ? pos3[r * 3 + mrope_plane(pr)] : pos[r];
  const float a = __fmul_rn(float(ps), fr.f[pr]);
  const float c = round_bf16(cosf(a)), s = round_bf16(sinf(a));
  const float x0 = bf16_to_f32(src[0]), x1 = bf16_to_f32(src[1]);
  dst[0] = f32_to_bf16(__fadd_rn(__fmul_rn(x0, c), __fmul_rn(-x1, s)));
  dst[1] = f32_to_bf16(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
}

__global__ void rope_kernel(bf16* x, int ld, int rows, int heads, int hstride, int off, const int32_t* pos,
                            InvFreq fr, const int32_t* pos3) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(rows) * heads * 32) return;
  const int pr = int(i % 32), h = int((i / 32) % heads), r = int(i / (32 * size_t(heads)));
  bf16* p = x + size_t(r) * ld + size_t(h) * hstride + off + 2 * pr;
  rope_pair(p, p, r, pr, pos, pos3, fr);
}

// hy4_qkv_post: grid (T, 2), 256 threads: block y 0 the q_a part's rmsnorm (rmsnorm_hf's order) into qr, block y 1
// the key row's latent rmsnorm and its rope pairs into kv.
__global__ __launch_bounds__(256) void qkv_post_kernel(const bf16* qkv, int ld, int nq, const bf16* qn_w, bf16* qr,
                                                       int nl, const bf16* kn_w, bf16* kv, int nkv, float eps,
                                                       const int32_t* pos, const int32_t* pos3, InvFreq fr) {
  __shared__ float red[32];
  const int t = blockIdx.x;
  const bf16* src = qkv + size_t(t) * ld + (blockIdx.y ? nq : 0);
  const int n = blockIdx.y ? nl : nq;
  const bf16* w = blockIdx.y ? kn_w : qn_w;
  bf16* dst = blockIdx.y ? kv + size_t(t) * nkv : qr + size_t(t) * nq;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = bf16_to_f32(src[i]);
    ss += v * v;
  }
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / float(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    dst[i] = f32_to_bf16(bf16_to_f32(w[i]) * round_bf16(bf16_to_f32(src[i]) * r));
  if (blockIdx.y && int(threadIdx.x) < (nkv - nl) / 2)
    rope_pair(src + nl + 2 * threadIdx.x, dst + nl + 2 * threadIdx.x, t, threadIdx.x, pos, pos3, fr);
}

// ---- attention -------------------------------------------------------------------------------------------------

#if defined(UH_GFX12)
#define UH_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c)
#endif

constexpr int R = kHy4Row, L = kHy4Lat, KB = 16;

__device__ __forceinline__ const bf16* key_row(const Hy4Keys& k, int p) {
  if (k.block && p >= k.anchor) return k.block + size_t(p - k.anchor) * R;
  if (k.table)
    return k.pages + (size_t(k.table[p / kHy4PageRows]) * kHy4PageRows + p % kHy4PageRows) * R;
  return p >= k.s0 ? k.chunk + size_t(p - k.s0) * R : k.ring + size_t(p % kHy4Window) * R;
}

template <int H>
__global__ __launch_bounds__(H * 4) void mla_attn_kernel(const bf16* __restrict__ q, Hy4Keys keys,
                                                         const int32_t* __restrict__ idx, int w, int window,
                                                         const int32_t* __restrict__ pos, const float* __restrict__ sinks,
                                                         bf16* __restrict__ o, float scale,
                                                         const int32_t* __restrict__ counts, int S,
                                                         float* __restrict__ po, float* __restrict__ pml) {
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
  if (idx && counts) {
    cnt = counts[t];
  } else if (idx) {  // the compact selection: min(groups before the query's, k) groups, then the query's own group
    const int k = (w - kHy4Pool) / kHy4Pool;
    cnt = min(pt / kHy4Pool, k) * kHy4Pool + pt % kHy4Pool + 1;
  } else {
    lo = window > 0 ? max(0, pt - window + 1) : 0;
    cnt = pt - lo + 1;
  }
  const int32_t* tidx = idx ? idx + size_t(t) * w : nullptr;
  // Split si of S takes a contiguous run of whole key tiles; split 0 carries the sink.
  const int si = blockIdx.y;
  int e_lo = 0, e_hi = cnt;
  if (S > 1) {
    const int per = ((cnt + KB - 1) / KB + S - 1) / S * KB;
    e_lo = min(cnt, si * per);
    e_hi = min(cnt, e_lo + per);
  }

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
    sM[tid] = si == 0 ? sinks[tid] : -INFINITY;
    sL[tid] = si == 0 ? 1.0f : 0.0f;
  }
  for (int e0 = e_lo; e0 < e_hi; e0 += KB) {
    __syncthreads();
    if (tid < KB) {
      const int e = e0 + tid;
      int p = -1;
      if (e < e_hi) p = tidx ? tidx[e] : lo + e;
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
        const float f = m_new == -INFINITY ? 1.0f : expf(m_old - m_new);  // (a split's first tiles may be all empty)
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
  if (S > 1) {  // the split's unnormalized output and its softmax state, for mla_combine_kernel
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = hg * 16 + (lane >> 4) * 8 + e;
      float* prow = po + ((size_t(t) * S + si) * H + h) * L + dh * 256 + (lane & 15);
#pragma unroll
      for (int n = 0; n < 16; ++n) prow[n * 16] = acc[n][e];
    }
    if (tid < H) {
      pml[((size_t(t) * S + si) * H + tid) * 2] = sM[tid];
      pml[((size_t(t) * S + si) * H + tid) * 2 + 1] = sL[tid];
    }
    return;
  }
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

// One block per (query, head): the S splits' partial outputs merged by their softmax states.
__global__ void mla_combine_kernel(const float* __restrict__ po, const float* __restrict__ pml, bf16* __restrict__ o,
                                   int H, int S) {
  const int t = blockIdx.x / H, h = blockIdx.x % H;
  __shared__ float sw[kHy4AttnMaxSplit];
  __shared__ float sinv;
  if (threadIdx.x == 0) {
    float M = -INFINITY;
    for (int s = 0; s < S; ++s) M = fmaxf(M, pml[((size_t(t) * S + s) * H + h) * 2]);
    float sum = 0.0f;
    for (int s = 0; s < S; ++s) {
      const float m = pml[((size_t(t) * S + s) * H + h) * 2], l = pml[((size_t(t) * S + s) * H + h) * 2 + 1];
      const float wgt = m == -INFINITY ? 0.0f : expf(m - M);
      sw[s] = wgt;
      sum += l * wgt;
    }
    sinv = 1.0f / sum;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < L; d += blockDim.x) {
    float a = 0.0f;
    for (int s = 0; s < S; ++s) a += po[((size_t(t) * S + s) * H + h) * L + d] * sw[s];
    o[(size_t(t) * H + h) * L + d] = f32_to_bf16(a * sinv);
  }
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

__global__ void ring_store_kernel(const bf16* rows, bf16* ring, int s0, int n, bf16* bak) {
  const int first = max(0, n - kHy4Window);
  const size_t total = size_t(n - first) * R;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int r = first + int(i / R), d = int(i % R);
    bf16* slot = ring + size_t((s0 + r) % kHy4Window) * R + d;
    if (bak) bak[size_t(r) * R + d] = *slot;  // n <= kHy4Window with a backup: each slot once
    *slot = rows[size_t(r) * R + d];
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
                                                      InvFreq fr, Hy4Pooled pg, int s0, int p0, Hy4GroupRope gr) {
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
    const int k = d - (ID - 64), pr = k / 2, q0 = p * POOL;
    const int ps = !gr.rows ? q0 : q0 >= s0 ? gr.rows[(q0 - s0) * 3 + mrope_plane(pr)] : gr.g0[mrope_plane(pr)];
    const float a = __fmul_rn(float(ps), fr.f[pr]);
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

__global__ void ring_backup_kernel(const bf16* ring, bf16* bak, int s0, int n) {
  const size_t total = size_t(n) * (R / 8);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int r = int(i / (R / 8)), d0 = int(i % (R / 8)) * 8;
    *reinterpret_cast<uint4*>(bak + size_t(r) * R + d0) =
        *reinterpret_cast<const uint4*>(ring + size_t((s0 + r) % kHy4Window) * R + d0);
  }
}

__global__ void ring_restore_kernel(bf16* ring, const bf16* bak, int s0, int from, int to) {
  const size_t total = size_t(to - from) * (R / 8);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int p = from + int(i / (R / 8)), d0 = int(i % (R / 8)) * 8;
    *reinterpret_cast<uint4*>(ring + size_t(p % kHy4Window) * R + d0) =
        *reinterpret_cast<const uint4*>(bak + size_t(p - s0) * R + d0);
  }
}

__global__ void tail_restore_kernel(bf16* tail, const bf16* bak, const bf16* vk, int s0, int from) {
  const int d = threadIdx.x;  // 128 threads
  for (int j = 0; j < kHy4Pool; ++j) tail[j * kHy4IdxDim + d] = bak[j * kHy4IdxDim + d];
  for (int q = max(s0, from - kHy4Pool); q < from; ++q)
    tail[(q % kHy4Pool) * kHy4IdxDim + d] = vk[size_t(q - s0) * kHy4IdxDim + d];
}

// One block per saved row.
__global__ void dfly_save_sel_kernel(const int32_t* sel, int w, const int32_t* pos, int t0, int n, int32_t* dst,
                                     int32_t* counts, int groups) {
  const int t = t0 + blockIdx.x;
  const int p = pos[t], r = p % 6;
  int32_t* out = dst + size_t(r) * 2048;
  int cnt;
  if (sel) {
    cnt = min(p / kHy4Pool, groups) * kHy4Pool + p % kHy4Pool + 1;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) out[i] = i < cnt ? sel[size_t(t) * w + i] : -1;
  } else {
    cnt = p + 1;  // dense: every position <= p (p < 2048 here)
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) out[i] = i < cnt ? i : -1;
  }
  if (threadIdx.x == 0) counts[r] = cnt;
}

__global__ void dfly_lists_kernel(const int32_t* const* hist, const int32_t* const* hcount, const int32_t* anchor,
                                  int32_t* lists, int32_t* counts, int w) {
  const int r = blockIdx.x / 6, j = blockIdx.x % 6;
  const int hc = *hcount[r], a = anchor[r];
  int32_t* out = lists + (size_t(r) * 6 + j) * w;
  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    int v = -1;
    if (i < hc) v = hist[r][i];
    else if (i - hc <= j) v = a + (i - hc);
    out[i] = v;
  }
  if (threadIdx.x == 0) counts[r * 6 + j] = hc + j + 1;
}

// Slice s of row r: its max, the max's first index and sum(exp(x - max)) -> ws[(r * S + s) * 3 ..].
__global__ __launch_bounds__(256) void argmax_slice_kernel(const float* logits, int V, float* ws) {
  __shared__ float sv[8], ss[8];
  __shared__ int si[8];
  const int r = blockIdx.y, sl = blockIdx.x, S = gridDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = (V + S - 1) / S, v0 = sl * per, v1 = min(V, v0 + per);
  const float* row = logits + size_t(r) * V;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int v = v0 + tid; v < v1; v += 256)
    if (row[v] > bv) bv = row[v], bi = v;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(bv, o, 32);
    const int oi = __shfl_xor(bi, o, 32);
    if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
  }
  if (lane == 0) sv[warp] = bv, si[warp] = bi;
  __syncthreads();
  float mx = sv[0];
  int mi = si[0];
  for (int w = 1; w < 8; ++w)
    if (sv[w] > mx || (sv[w] == mx && si[w] < mi)) mx = sv[w], mi = si[w];
  float s = 0.0f;
  if (mx > -INFINITY)
    for (int v = v0 + tid; v < v1; v += 256) s += expf(row[v] - mx);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
  if (lane == 0) ss[warp] = s;
  __syncthreads();
  if (tid == 0) {
    float tot = 0.0f;
    for (int w = 0; w < 8; ++w) tot += ss[w];
    float* out = ws + (size_t(r) * S + sl) * 3;
    out[0] = mx;
    out[1] = __int_as_float(mi);
    out[2] = tot;
  }
}

__global__ void argmax_combine_kernel(const float* ws, int S, int32_t* tok, float* conf, float* lse) {
  const int r = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* p = ws + size_t(r) * S * 3;
  float mx = -INFINITY;
  int mi = 0x7fffffff;
  for (int s = 0; s < S; ++s) {
    const float v = p[s * 3];
    const int i = __float_as_int(p[s * 3 + 1]);
    if (v > mx || (v == mx && i < mi)) mx = v, mi = i;
  }
  float tot = 0.0f;
  for (int s = 0; s < S; ++s)
    if (p[s * 3] > -INFINITY) tot += p[s * 3 + 2] * expf(p[s * 3] - mx);
  tok[r] = mi;
  conf[r] = 1.0f / tot;
  if (lse) lse[r] = mx + logf(tot);
}

// Grid (K / 8, R), 256 threads: each warp scores one candidate (uint4 loads of its head row and the delta) into
// ws [R][K]; the row's last block to finish (counter ws_count[r], reset by it) picks.
__global__ __launch_bounds__(256) void draft_pick_kernel(const float* base, const float* lse, const int32_t* cand,
                                                         int K, const bf16* head, const bf16* delta, int V, int D,
                                                         int32_t* tok, float* conf, float* ws, uint32_t* ws_count) {
  __shared__ float rv[8], rs0[8], rs1[8];
  __shared__ int ri[8];
  __shared__ bool last;
  const int r = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int32_t* cr = cand + size_t(r) * K;
  const float* br = base + size_t(r) * V;
  float* sl = ws + size_t(r) * K;
  {
    const int c = blockIdx.x * 8 + warp;
    const int id = c < K ? cr[c] : -1;
    float s = 0.0f;
    if (id >= 0) {
      const uint4* hr = reinterpret_cast<const uint4*>(head + size_t(id) * D);
      const uint4* dr = reinterpret_cast<const uint4*>(delta + size_t(r) * D);
      for (int i = lane; i < D / 8; i += 32) {
        const uint4 h = hr[i], d = dr[i];
        const uint32_t hw[4] = {h.x, h.y, h.z, h.w}, dw[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          s = fmaf(__uint_as_float(hw[j] << 16), __uint_as_float(dw[j] << 16), s);
          s = fmaf(__uint_as_float(hw[j] & 0xffff0000u), __uint_as_float(dw[j] & 0xffff0000u), s);
        }
      }
    }
    s = warp_sum(s);
    if (lane == 0 && c < K) sl[c] = id >= 0 ? br[id] + s : -INFINITY;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&ws_count[r], 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  // best candidate, and the mass: 1 - sum exp(base_c - lse) + sum exp(l_c - lse)
  const float L = lse[r];
  float bv = -INFINITY, s0 = 0.0f, s1 = 0.0f;
  int bi = 0x7fffffff;
  for (int c = tid; c < K; c += 256) {
    const int id = cr[c];
    if (id < 0) continue;
    const float l = __builtin_nontemporal_load(&sl[c]);
    if (l > bv || (l == bv && id < bi)) bv = l, bi = id;
    s0 += expf(br[id] - L);
    s1 += expf(l - L);
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(bv, o, 32);
    const int oi = __shfl_xor(bi, o, 32);
    if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    s0 += __shfl_xor(s0, o, 32);
    s1 += __shfl_xor(s1, o, 32);
  }
  if (lane == 0) rv[warp] = bv, ri[warp] = bi, rs0[warp] = s0, rs1[warp] = s1;
  __syncthreads();
  if (tid == 0) {
    float v = rv[0], a = rs0[0], b = rs1[0];
    int i = ri[0];
    for (int w = 1; w < 8; ++w) {
      if (rv[w] > v || (rv[w] == v && ri[w] < i)) v = rv[w], i = ri[w];
      a += rs0[w];
      b += rs1[w];
    }
    tok[r] = i;
    conf[r] = expf(v - L) / fmaxf(1.0f - a + b, 1e-30f);
    ws_count[r] = 0;
  }
}

__global__ void gated_add_kernel(bf16* x, int ldx, const bf16* g, const bf16* f, int rows, int dim) {
  const size_t n = size_t(rows) * dim;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / dim, d = i % dim;
    bf16* xp = x + r * ldx + d;
    *xp = f32_to_bf16(bf16_to_f32(*xp) + round_bf16(bf16_to_f32(g[d]) * bf16_to_f32(f[i])));
  }
}

__global__ void concat2_kernel(const bf16* a, const bf16* b, bf16* out, int rows, int dim) {
  const size_t n = size_t(rows) * 2 * dim;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / (2 * dim), d = i % (2 * dim);
    out[i] = d < size_t(dim) ? a[r * dim + d] : b[r * dim + d - dim];
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
      if constexpr (kFmt == 1) {  // wf[q] / wf[q + 8]: byte q's low / high nibble
        const uint2 raw = *reinterpret_cast<const uint2*>(W + size_t(col) * (K / 2) + k0 / 2);
        const float sc = ue4m3_to_f32(static_cast<const uint8_t*>(Sv)[size_t(col) * (K / 16) + k0 / 16]);
        e2m1x8_scaled(raw.x, sc, wf, wf + 8);
        e2m1x8_scaled(raw.y, sc, wf + 4, wf + 12);
      } else {  // bf16(w * s), the reference's dequantized weight
        const bf16* S = static_cast<const bf16*>(Sv);
        const uint4 wv = *reinterpret_cast<const uint4*>(W + size_t(col) * K + k0);
        const float sc = bf16_to_f32(S[size_t(col / 128) * kb + k0 / 128]);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          e4m3x4_scaled(ww[q], sc, wf + 4 * q);
#pragma unroll
          for (int i = 0; i < 4; ++i) wf[4 * q + i] = round_bf16_finite(wf[4 * q + i]);
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

// Lane l of a wave: B fragments for columns c0 + (l & 15) and c0 + 16 + (l & 15), A for row (l & 15) of the tile.
// K goes in chunks of 128 (one fp8 scale block), dealt to the 8 waves in turn; in a chunk at k0 the lane's half
// h = l >> 4 takes inputs k0 + 64 h .. + 63, contiguous (a column's two halves read a whole 128-byte line of fp8),
// fed to 8 WMMAs 8 inputs at a time (A and B permute K alike). C element e of lane l: row 8 h + e, column l & 15.
template <int kFmt>
__global__ __launch_bounds__(256) void wmma_gemv_kernel(const bf16* __restrict__ A, int ldi, int K,
                                                        const Exl3Tile* tiles, int ntiles, const int* ntiles_dev,
                                                        const uint8_t* const* w_tab, const void* const* s_tab,
                                                        const uint8_t* w_direct, bf16* __restrict__ C, int ldc, int N,
                                                        const bf16* __restrict__ addend) {
#if defined(UH_GFX12)
  __shared__ float red[8][2][256];
  const int nt = ntiles_dev ? *ntiles_dev : ntiles;
  if (int(blockIdx.y) >= nt) return;
  const Exl3Tile tile = tiles[blockIdx.y];
  const uint8_t* W = w_tab ? w_tab[tile.expert] : w_direct;
  const void* Sv = s_tab ? s_tab[tile.expert] : nullptr;
  if (!W || tile.rows <= 0) return;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, h = lane >> 4, l16 = lane & 15;
  const int c0 = blockIdx.x * 32;
  const int rows = min(tile.rows, 16);
  const bool aok = l16 < rows;
  const bf16* Ar = A + size_t(tile.row0 + (aok ? l16 : 0)) * ldi;
  const auto to_bf16 = [](float f) {  // finite values (the weights): round to nearest even
    const uint32_t u = __float_as_uint(f);
    return __builtin_bit_cast(__bf16, uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
  };
  float8_t acc[2];
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[g][e] = 0.0f;
  for (int ch = wave; ch < K / 128; ch += 8) {
    const int kk = ch * 128 + h * 64;
    bf16x8_t a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (aok) a[j] = *reinterpret_cast<const bf16x8_t*>(Ar + kk + 8 * j);
      else
#pragma unroll
        for (int i = 0; i < 8; ++i) a[j][i] = __bf16(0.0f);
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const int col = c0 + g * 16 + l16;
      if constexpr (kFmt == 2) {
        const uint16_t* Wr = reinterpret_cast<const uint16_t*>(W) + size_t(col) * K + kk;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[g] = UH_WMMA(a[j], *reinterpret_cast<const bf16x8_t*>(Wr + 8 * j), acc[g]);
      } else if constexpr (kFmt == 0) {
        const uint8_t* Wr = W + size_t(col) * K + kk;
        const float sc = bf16_to_f32(static_cast<const bf16*>(Sv)[size_t(col / 128) * (K / 128) + ch]);
        uint4 wv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) wv[q] = *reinterpret_cast<const uint4*>(Wr + 16 * q);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t ww[4] = {wv[q].x, wv[q].y, wv[q].z, wv[q].w};
          float f[16];
#pragma unroll
          for (int i = 0; i < 4; ++i) e4m3x4_scaled(ww[i], sc, f + 4 * i);
          bf16x8_t b0, b1;
#pragma unroll
          for (int i = 0; i < 8; ++i) b0[i] = to_bf16(f[i]), b1[i] = to_bf16(f[i + 8]);
          acc[g] = UH_WMMA(a[2 * q], b0, acc[g]);
          acc[g] = UH_WMMA(a[2 * q + 1], b1, acc[g]);
        }
      } else {  // NVFP4: per 16 inputs 8 bytes (byte j's low nibble input j, its high one input j + 8) and a scale
        const uint8_t* Wr = W + size_t(col) * (K / 2) + kk / 2;
        const uint32_t sc4 = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(Sv) + size_t(col) * (K / 16) + kk / 16);
        uint4 wv[2];
        wv[0] = *reinterpret_cast<const uint4*>(Wr);
        wv[1] = *reinterpret_cast<const uint4*>(Wr + 16);
#pragma unroll
        for (int q = 0; q < 4; ++q) {  // group q: 16 inputs
          const uint4& v = wv[q >> 1];
          const uint32_t lo = (q & 1) ? v.z : v.x, hi = (q & 1) ? v.w : v.y;
          const float sc = ue4m3_to_f32(uint8_t(sc4 >> (8 * q)));
          float f[16];
          e2m1x8_scaled(lo, sc, f, f + 8);
          e2m1x8_scaled(hi, sc, f + 4, f + 12);
          bf16x8_t b0, b1;
#pragma unroll
          for (int i = 0; i < 8; ++i) b0[i] = to_bf16(f[i]), b1[i] = to_bf16(f[i + 8]);
          acc[g] = UH_WMMA(a[2 * q], b0, acc[g]);
          acc[g] = UH_WMMA(a[2 * q + 1], b1, acc[g]);
        }
      }
    }
  }
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) red[wave][g][(h * 8 + e) * 16 + l16] = acc[g][e];
  __syncthreads();
  const int t = threadIdx.x, r = t >> 4, cc = t & 15;
  if (r < rows)
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) sum += red[w][g][t];
      const size_t at = size_t(tile.row0 + r) * ldc + c0 + g * 16 + cc;
      bf16 v = f32_to_bf16(sum);
      if (addend) v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(addend[at]));  // hy4_add's rounding
      C[at] = v;
    }
#elif defined(UH_GFX11)
  // gfx11 (the APU): a WMMA operand is all 16 inputs of a k step in both half-waves. Each lane decodes only its own
  // 64 inputs of a chunk (k0 + 64 h ..); WMMA step j takes inputs k0 + 8 j .. + 7 (half 0's) and k0 + 64 + 8 j ..
  // (half 1's), each lane getting the other half's 8 from the lane 16 away (permlanex16), for A and B alike.
  // C element e of lane l: row 2 e + h, column l & 15.
  typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
  __shared__ float red[8][2][256];
  const int nt = ntiles_dev ? *ntiles_dev : ntiles;
  if (int(blockIdx.y) >= nt) return;
  const Exl3Tile tile = tiles[blockIdx.y];
  const uint8_t* W = w_tab ? w_tab[tile.expert] : w_direct;
  const void* Sv = s_tab ? s_tab[tile.expert] : nullptr;
  if (!W || tile.rows <= 0) return;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, h = lane >> 4, l16 = lane & 15;
  const int c0 = blockIdx.x * 32;
  const int rows = min(tile.rows, 16);
  const bool aok = l16 < rows;
  const bf16* Ar = A + size_t(tile.row0 + (aok ? l16 : 0)) * ldi;
  const auto pack = [](float a, float b) {  // two finite values as bf16 (nearest even), a in the low half
    const uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
    return ((ua + 0x7fffu + ((ua >> 16) & 1u)) >> 16) | ((ub + 0x7fffu + ((ub >> 16) & 1u)) & 0xffff0000u);
  };
  const auto xchg = [](uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); };
  const auto frag = [&](const uint32_t* own) {  // 4 dwords (8 bf16) of this half -> the 16-input operand
    uint32_t other[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) other[i] = xchg(own[i]);
    const uint32_t* lo = h == 0 ? own : other;
    const uint32_t* hi = h == 0 ? other : own;
    const u32x8 v = {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
    return __builtin_bit_cast(bf16x16_t, v);
  };
  float8_t acc[2];
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[g][e] = 0.0f;
  for (int ch = wave; ch < K / 128; ch += 8) {
    const int kk = ch * 128 + h * 64;
    bf16x16_t afr[8];  // the A operands of the chunk's 8 steps (shared by both column groups)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint4 v = {0, 0, 0, 0};
      if (aok) v = *reinterpret_cast<const uint4*>(Ar + kk + 8 * j);
      const uint32_t own[4] = {v.x, v.y, v.z, v.w};
      afr[j] = frag(own);
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const int col = c0 + g * 16 + l16;
      uint32_t b[8][4];  // this lane's 64 inputs as bf16 pairs, step j's 8 in b[j]
      if constexpr (kFmt == 2) {
        const uint16_t* Wr = reinterpret_cast<const uint16_t*>(W) + size_t(col) * K + kk;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint4 v = *reinterpret_cast<const uint4*>(Wr + 8 * j);
          b[j][0] = v.x, b[j][1] = v.y, b[j][2] = v.z, b[j][3] = v.w;
        }
      } else if constexpr (kFmt == 0) {
        const uint8_t* Wr = W + size_t(col) * K + kk;
        const float sc = bf16_to_f32(static_cast<const bf16*>(Sv)[size_t(col / 128) * (K / 128) + ch]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 wv = *reinterpret_cast<const uint4*>(Wr + 16 * q);
          const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
          float f[16];
#pragma unroll
          for (int i = 0; i < 4; ++i) e4m3x4_scaled(ww[i], sc, f + 4 * i);
#pragma unroll
          for (int i = 0; i < 4; ++i) b[2 * q][i] = pack(f[2 * i], f[2 * i + 1]), b[2 * q + 1][i] = pack(f[8 + 2 * i], f[9 + 2 * i]);
        }
      } else {
        const uint8_t* Wr = W + size_t(col) * (K / 2) + kk / 2;
        const uint32_t sc4 = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(Sv) + size_t(col) * (K / 16) + kk / 16);
        const uint4 w0 = *reinterpret_cast<const uint4*>(Wr), w1 = *reinterpret_cast<const uint4*>(Wr + 16);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4& v = q < 2 ? w0 : w1;
          const uint32_t lo = (q & 1) ? v.z : v.x, hi = (q & 1) ? v.w : v.y;
          const float sc = ue4m3_to_f32(uint8_t(sc4 >> (8 * q)));
          float f[16];
          e2m1x8_scaled(lo, sc, f, f + 8);
          e2m1x8_scaled(hi, sc, f + 4, f + 12);
#pragma unroll
          for (int i = 0; i < 4; ++i) b[2 * q][i] = pack(f[2 * i], f[2 * i + 1]), b[2 * q + 1][i] = pack(f[8 + 2 * i], f[9 + 2 * i]);
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[g] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afr[j], frag(b[j]), acc[g]);
    }
  }
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) red[wave][g][(2 * e + h) * 16 + l16] = acc[g][e];
  __syncthreads();
  const int t = threadIdx.x, r = t >> 4, cc = t & 15;
  if (r < rows)
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) sum += red[w][g][t];
      const size_t at = size_t(tile.row0 + r) * ldc + c0 + g * 16 + cc;
      bf16 v = f32_to_bf16(sum);
      if (addend) v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(addend[at]));  // hy4_add's rounding
      C[at] = v;
    }
#endif
}

bool device_gfx12() {
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

size_t hy4_ihc_ws_bytes(int max_rows, int hc, int dim) {
  return size_t(max_rows) * (size_t(hc) * dim / kHcSlice * (1 + kMaxMix) + 1) * 4;
}

void hy4_ihc_fused(const bf16* out, const bf16* out_norm, const float* post_in, bf16* x, const float* fn,
                   const float* scale, const float* base, bf16* y, float* post_out, int T, int hc, int dim, float eps,
                   float hc_eps, float mag, const bf16* norm_w, bf16* hn, void* ws, int ws_rows, hipStream_t st) {
  if (T <= 0) return;
  if (hc != 4 || dim > 4096 || dim % kHcSlice) throw std::runtime_error("hy4_ihc_fused: hc 4, dim <= 4096, dim % 512");
  if (T > ws_rows) throw std::runtime_error("hy4_ihc_fused: more rows than the workspace");
  const int S = hc * dim / kHcSlice;
  auto* count = static_cast<uint32_t*>(ws);  // at a fixed place: every call's counters start at 0
  auto* part = reinterpret_cast<float*>(count + ws_rows);
  ihc_fused_kernel<<<dim3(unsigned(S), unsigned(T)), kHcThreads, 0, st>>>(
      out, out_norm, post_in, x, fn, scale, base, y, post_out, hc, dim, eps, hc_eps, mag, norm_w, hn, part, count);
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

void hy4_qkv_post(const bf16* qkv, int ld, int nq, const bf16* q_norm, bf16* qr, int nl, const bf16* kv_norm,
                  bf16* kv, int nkv, int T, float eps, const int32_t* pos, const int32_t* pos3, float theta,
                  hipStream_t st) {
  if (T <= 0) return;
  if (nkv - nl != 64) throw std::runtime_error("hy4_qkv_post: 32 rope pairs");
  qkv_post_kernel<<<dim3(unsigned(T), 2), 256, 0, st>>>(qkv, ld, nq, q_norm, qr, nl, kv_norm, kv, nkv, eps, pos, pos3,
                                                        inv_freq(theta));
  UH_CHECK_LAUNCH();
}

void hy4_rope(bf16* x, int ld, int rows, int heads, int hstride, int off, const int32_t* pos, float theta,
              hipStream_t st, const int32_t* pos3) {
  if (rows <= 0) return;
  const size_t n = size_t(rows) * heads * 32;
  rope_kernel<<<cdiv(n, 256), 256, 0, st>>>(x, ld, rows, heads, hstride, off, pos, inv_freq(theta), pos3);
  UH_CHECK_LAUNCH();
}

void hy4_mla_attn(const bf16* q, int H, const Hy4Keys& keys, const int32_t* idx, int w, int window,
                  const int32_t* pos, const float* sinks, bf16* o, int n, float scale, hipStream_t st,
                  const int32_t* counts, float* ws, size_t ws_bytes) {
  if (n <= 0) return;
  int S = 1;
  static const bool no_split = std::getenv("UNHOLY_HY4_NO_ATTN_SPLIT") != nullptr;  // A/B checks
  if (ws && n <= kHy4AttnSplitRows && !no_split) {
    S = std::min(kHy4AttnMaxSplit, std::max(1, kHy4AttnSplitBlocks / n));
    while (S > 1 && size_t(n) * S * H * (L + 2) * 4 > ws_bytes) --S;
  }
  float* po = ws;
  float* pml = S > 1 ? ws + size_t(n) * S * H * L : nullptr;
  const dim3 grid(static_cast<unsigned>(n), static_cast<unsigned>(S));
  if (H == 32)
    mla_attn_kernel<32><<<grid, 128, 0, st>>>(q, keys, idx, w, window, pos, sinks, o, scale, counts, S, po, pml);
  else if (H == 64)
    mla_attn_kernel<64><<<grid, 256, 0, st>>>(q, keys, idx, w, window, pos, sinks, o, scale, counts, S, po, pml);
  else throw std::runtime_error("hy4_mla_attn: 32 or 64 heads");
  UH_CHECK_LAUNCH();
  if (S > 1) {
    mla_combine_kernel<<<n * H, 128, 0, st>>>(po, pml, o, H, S);
    UH_CHECK_LAUNCH();
  }
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

void hy4_ring_store(const bf16* rows, bf16* ring, int s0, int n, hipStream_t st, bf16* bak) {
  if (n <= 0) return;
  if (bak && n > kHy4Window) throw std::runtime_error("hy4_ring_store: a backup of more rows than the ring");
  ring_store_kernel<<<grid1d(size_t(std::min(n, kHy4Window)) * R), 256, 0, st>>>(rows, ring, s0, n, bak);
  UH_CHECK_LAUNCH();
}

void hy4_page_store(const bf16* rows, bf16* pages, const int32_t* table, int s0, int n, hipStream_t st) {
  if (n <= 0) return;
  page_store_kernel<<<grid1d(size_t(n) * (R / 8)), 256, 0, st>>>(rows, pages, table, s0, n);
  UH_CHECK_LAUNCH();
}

void hy4_idx_pool(const bf16* tk, bf16* tail, const float* wv, const float* wg, const float* ape, const float* knorm,
                  float eps, float theta, Hy4Pooled pg, int s0, int n, hipStream_t st, Hy4GroupRope gr) {
  if (n <= 0) return;
  const int end = s0 + n, p0 = s0 / POOL, p1 = end / POOL;  // groups whose last member is in [s0, end)
  if (p1 > p0)
    idx_pool_kernel<<<p1 - p0, ID, 0, st>>>(tk, tail, wv, wg, ape, knorm, eps, inv_freq(theta), pg, s0, p0, gr);
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

void hy4_ring_backup(const bf16* ring, bf16* bak, int s0, int n, hipStream_t st) {
  if (n <= 0) return;
  ring_backup_kernel<<<grid1d(size_t(n) * (R / 8)), 256, 0, st>>>(ring, bak, s0, n);
  UH_CHECK_LAUNCH();
}

void hy4_ring_restore(bf16* ring, const bf16* bak, int s0, int from, int to, hipStream_t st) {
  if (to <= from) return;
  ring_restore_kernel<<<grid1d(size_t(to - from) * (R / 8)), 256, 0, st>>>(ring, bak, s0, from, to);
  UH_CHECK_LAUNCH();
}

void hy4_tail_restore(bf16* tail, const bf16* bak, const bf16* vk, int s0, int from, hipStream_t st) {
  tail_restore_kernel<<<1, kHy4IdxDim, 0, st>>>(tail, bak, vk, s0, from);
  UH_CHECK_LAUNCH();
}

void hy4_dfly_save_sel(const int32_t* sel, int w, const int32_t* pos, int t0, int n, int32_t* dst, int32_t* counts,
                       int groups, hipStream_t st) {
  if (n <= 0) return;
  dfly_save_sel_kernel<<<n, 256, 0, st>>>(sel, w, pos, t0, n, dst, counts, groups);
  UH_CHECK_LAUNCH();
}

void hy4_dfly_lists(const int32_t* const* hist, const int32_t* const* hcount, const int32_t* anchor, int32_t* lists,
                    int32_t* counts, int R, int w, hipStream_t st) {
  if (R <= 0) return;
  dfly_lists_kernel<<<R * 6, 256, 0, st>>>(hist, hcount, anchor, lists, counts, w);
  UH_CHECK_LAUNCH();
}

void hy4_argmax_conf(const float* logits, int rows, int V, int32_t* tok, float* conf, float* ws, hipStream_t st,
                     float* lse) {
  if (rows <= 0) return;
  argmax_slice_kernel<<<dim3(kHy4ArgmaxSlices, unsigned(rows)), 256, 0, st>>>(logits, V, ws);
  UH_CHECK_LAUNCH();
  argmax_combine_kernel<<<rows, 32, 0, st>>>(ws, kHy4ArgmaxSlices, tok, conf, lse);
  UH_CHECK_LAUNCH();
}

void hy4_draft_pick(const float* base, const float* lse, const int32_t* cand, int K, const bf16* head,
                    const bf16* delta, int R, int V, int D, int32_t* tok, float* conf, void* ws, int ws_rows,
                    hipStream_t st) {
  if (R <= 0) return;
  if (D % 8) throw std::runtime_error("hy4_draft_pick: dim % 8");
  if (R > ws_rows) throw std::runtime_error("hy4_draft_pick: more rows than the workspace");
  auto* count = static_cast<uint32_t*>(ws);  // at a fixed place: every call's counters start at 0
  auto* scores = reinterpret_cast<float*>(count + ws_rows);
  draft_pick_kernel<<<dim3(unsigned((K + 7) / 8), unsigned(R)), 256, 0, st>>>(base, lse, cand, K, head, delta, V, D,
                                                                               tok, conf, scores, count);
  UH_CHECK_LAUNCH();
}

void hy4_gated_add(bf16* x, int ldx, const bf16* g, const bf16* f, int rows, int dim, hipStream_t st) {
  if (rows <= 0) return;
  gated_add_kernel<<<grid1d(size_t(rows) * dim), 256, 0, st>>>(x, ldx, g, f, rows, dim);
  UH_CHECK_LAUNCH();
}

void hy4_concat2(const bf16* a, const bf16* b, bf16* out, int rows, int dim, hipStream_t st) {
  if (rows <= 0) return;
  concat2_kernel<<<grid1d(size_t(rows) * 2 * dim), 256, 0, st>>>(a, b, out, rows, dim);
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

bool hy4_wmma_gemv(int fmt, const bf16* xa, int ldi, int K, const Exl3Tile* tiles, int ntiles,
                   const uint8_t* const* w_tab, const void* const* s_tab, bf16* out, int ldo, int N, hipStream_t st,
                   const int* ntiles_dev, const void* w_direct, int step_rows, const bf16* addend) {
  static const bool off = std::getenv("UNHOLY_HY4_NO_WMMA_GEMV") != nullptr;  // A/B checks
  static const bool apu_off = std::getenv("UNHOLY_HY4_NO_WMMA_GEMV_APU") != nullptr;
  if (off || ntiles <= 0 || K % 128 || N % 32) return false;
  // gfx11 (the APU, the half-exchange variant) at any row count too, though the GEMV is ~5% faster on 1-row steps: a
  // row's result then never depends on the step's size, so decoding with drafts stays bit-identical to without.
  (void)step_rows;
  if (!device_gfx12() && apu_off) return false;
  const dim3 grid(unsigned(N / 32), unsigned(ntiles));
  const auto* wd = static_cast<const uint8_t*>(w_direct);
  if (fmt == 0)
    wmma_gemv_kernel<0><<<grid, 256, 0, st>>>(xa, ldi, K, tiles, ntiles, ntiles_dev, w_tab, s_tab, wd, out, ldo, N, addend);
  else if (fmt == 1)
    wmma_gemv_kernel<1><<<grid, 256, 0, st>>>(xa, ldi, K, tiles, ntiles, ntiles_dev, w_tab, s_tab, wd, out, ldo, N, addend);
  else
    wmma_gemv_kernel<2><<<grid, 256, 0, st>>>(xa, ldi, K, tiles, ntiles, ntiles_dev, w_tab, s_tab, wd, out, ldo, N, addend);
  UH_CHECK_LAUNCH();
  return true;
}

namespace {

// Per-head few-row bf16 GEMV (gfx12): grid (N / 32, heads, row groups of 16), 256 threads. A block takes 32 output
// columns of head blockIdx.y; its 8 waves take K in chunks of CK, in turn (lane half h: inputs k0 + h CK / 2 ..,
// contiguous, CK / 16 WMMA steps, A and B permuted alike), reduced in LDS. Epilogue: bf16, then optionally times
// bf16(sigmoid(gate)) as hy4_gate_mul; the block of column 0 also copies inputs [K, K + tail) to columns [N, N + tail).
template <int CK>
__global__ __launch_bounds__(256) void heads_gemv_kernel(const bf16* __restrict__ A, int lda, int a_head, int K,
                                                         const bf16* __restrict__ W, bf16* __restrict__ C, int ldc,
                                                         int c_head, int N, int rows, const bf16* __restrict__ gate,
                                                         int ldg, int g_head, int tail, const int32_t* pos,
                                                         const int32_t* pos3, InvFreq fr) {
#if defined(UH_GFX12)
  __shared__ float red[8][2][256];
  const int hd = blockIdx.y, r0 = blockIdx.z * 16, nr = min(16, rows - r0);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, h = lane >> 4, l16 = lane & 15;
  const int c0 = blockIdx.x * 32;
  const bool aok = l16 < nr;
  const bf16* Ar = A + size_t(r0 + (aok ? l16 : 0)) * lda + size_t(hd) * a_head;
  const uint16_t* Wh = reinterpret_cast<const uint16_t*>(W) + size_t(hd) * N * K;
  constexpr int S = CK / 16;
  float8_t acc[2];
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[g][e] = 0.0f;
  for (int ch = wave; ch < K / CK; ch += 8) {
    const int kk = ch * CK + h * (CK / 2);
    bf16x8_t a[S];
#pragma unroll
    for (int j = 0; j < S; ++j) {
      if (aok) a[j] = *reinterpret_cast<const bf16x8_t*>(Ar + kk + 8 * j);
      else
#pragma unroll
        for (int i = 0; i < 8; ++i) a[j][i] = __bf16(0.0f);
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const uint16_t* Wr = Wh + size_t(c0 + g * 16 + l16) * K + kk;
      bf16x8_t b[S];
#pragma unroll
      for (int j = 0; j < S; ++j) b[j] = *reinterpret_cast<const bf16x8_t*>(Wr + 8 * j);
#pragma unroll
      for (int j = 0; j < S; ++j) acc[g] = UH_WMMA(a[j], b[j], acc[g]);
    }
  }
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) red[wave][g][(h * 8 + e) * 16 + l16] = acc[g][e];
  __syncthreads();
  const int t = threadIdx.x, r = t >> 4, cc = t & 15;
  if (r < nr) {
    const size_t row = size_t(r0 + r);
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) sum += red[w][g][t];
      const int col = c0 + g * 16 + cc;
      bf16 v = f32_to_bf16(sum);
      if (gate)
        v = f32_to_bf16(bf16_to_f32(v) * round_bf16(sigmoidf(bf16_to_f32(gate[row * ldg + size_t(hd) * g_head + col]))));
      C[row * ldc + size_t(hd) * c_head + col] = v;
    }
  }
  if (tail > 0 && blockIdx.x == 0) {
    if (pos)  // the tail as rope pairs (tail == 64)
      for (int i = t; i < nr * 32; i += 256) {
        const int row = r0 + i / 32, pr = i % 32;
        rope_pair(A + size_t(row) * lda + size_t(hd) * a_head + K + 2 * pr,
                  C + size_t(row) * ldc + size_t(hd) * c_head + N + 2 * pr, row, pr, pos, pos3, fr);
      }
    else
      for (int i = t; i < nr * tail; i += 256) {
        const size_t row = size_t(r0 + i / tail);
        const int d = i % tail;
        C[row * ldc + size_t(hd) * c_head + N + d] = A[row * lda + size_t(hd) * a_head + K + d];
      }
  }
#endif
}

}  // namespace

bool hy4_heads_gemv(const bf16* a, int lda, int a_head, int K, const bf16* w, int heads, bf16* c, int ldc, int c_head,
                    int N, int rows, hipStream_t st, const bf16* gate, int ldg, int g_head, int tail,
                    const int32_t* rope_pos, const int32_t* rope_pos3, float rope_theta) {
  static const bool off = std::getenv("UNHOLY_HY4_NO_HEADS_GEMV") != nullptr;  // A/B checks
  if (off || rows <= 0 || heads <= 0 || N % 32 || !device_gfx12()) return false;
  if (rope_pos && tail != 64) throw std::runtime_error("hy4_heads_gemv: a rope tail is 64 wide");
  const dim3 grid(unsigned(N / 32), unsigned(heads), unsigned((rows + 15) / 16));
  const InvFreq fr = rope_pos ? inv_freq(rope_theta) : InvFreq{};
  if (K % 64 == 0 && K / 64 >= 8)
    heads_gemv_kernel<64><<<grid, 256, 0, st>>>(a, lda, a_head, K, w, c, ldc, c_head, N, rows, gate, ldg, g_head, tail,
                                                rope_pos, rope_pos3, fr);
  else if (K % 32 == 0)
    heads_gemv_kernel<32><<<grid, 256, 0, st>>>(a, lda, a_head, K, w, c, ldc, c_head, N, rows, gate, ldg, g_head, tail,
                                                rope_pos, rope_pos3, fr);
  else
    return false;
  UH_CHECK_LAUNCH();
  return true;
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
