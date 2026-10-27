// Vision tower kernels (kernels/vision.hpp).
#include "kernels/dev.hip.hpp"
#include "kernels/vision.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

__global__ void vit_bias_kernel(const float* acc, int N, size_t total, const bf16* bias, bf16* out) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float b = bias ? bf16_to_f32(bias[i % size_t(N)]) : 0.0f;
  out[i] = f32_to_bf16(acc[i] + b);
}

__global__ void vit_bias_gelu_kernel(const float* acc, int N, size_t total, const bf16* bias, bf16* out) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float y = round_bf16(acc[i] + bf16_to_f32(bias[i % size_t(N)]));
  out[i] = f32_to_bf16(0.5f * y * (1.0f + erff(y * 0.70710678118654752f)));
}

__global__ void vit_silu_mul_kernel(const float* acc, int I, size_t total, bf16* out) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const size_t m = i / size_t(I), j = i % size_t(I);
  const float g = round_bf16(acc[m * 2 * I + j]), u = round_bf16(acc[m * 2 * I + I + j]);
  const float s = round_bf16(g / (1.0f + expf(-g)));
  out[i] = f32_to_bf16(s * u);
}

// kHf: transformers' order, bf16(x * r) before the weight.
template <bool kHf>
__global__ void vit_rmsnorm_kernel(const bf16* x, int D, const bf16* w, float eps, bf16* out) {
  __shared__ float red[8];
  const bf16* xr = x + size_t(blockIdx.x) * D;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = bf16_to_f32(xr[i]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < int(blockDim.x / 32); ++i) t += red[i];
    red[0] = t;
  }
  __syncthreads();
  const float r = rsqrtf(red[0] / float(D) + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float y = bf16_to_f32(xr[i]) * r;
    out[size_t(blockIdx.x) * D + i] = f32_to_bf16(bf16_to_f32(w[i]) * (kHf ? round_bf16(y) : y));
  }
}

// One wave per (row, head of q or k); lane j holds the rope pair (j, 32 + j). The q and k heads are adjacent in
// the row, so head hp < 2H sits at hp * 64.
__global__ void vit_qk_norm_rope_kernel(bf16* qkv, int N, int H, const bf16* qw, const bf16* kw, float eps,
                                        const float* cos, const float* sin) {
  const size_t wave = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (wave >= size_t(N) * 2 * H) return;
  const int j = threadIdx.x & 31, n = int(wave / (2 * H)), hp = int(wave % (2 * H));
  bf16* x = qkv + size_t(n) * 3 * H * 64 + size_t(hp) * 64;
  const bf16* w = hp < H ? qw : kw;
  const float x1 = bf16_to_f32(x[j]), x2 = bf16_to_f32(x[32 + j]);
  const float r = rsqrtf(warp_sum(x1 * x1 + x2 * x2) / 64.0f + eps);
  const float y1 = round_bf16(bf16_to_f32(w[j]) * round_bf16(x1 * r));
  const float y2 = round_bf16(bf16_to_f32(w[32 + j]) * round_bf16(x2 * r));
  const float c = cos[size_t(n) * 32 + j], s = sin[size_t(n) * 32 + j];
  x[j] = f32_to_bf16(__fsub_rn(__fmul_rn(y1, c), __fmul_rn(y2, s)));
  x[32 + j] = f32_to_bf16(__fadd_rn(__fmul_rn(y2, c), __fmul_rn(y1, s)));
}

__global__ void vit_swiglu_clamp_kernel(const float* acc, int I, size_t total, const bf16* bias, float limit,
                                        bf16* out) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const size_t m = i / size_t(I), j = i % size_t(I);
  float g = acc[m * 2 * I + j], u = acc[m * 2 * I + I + j];
  if (bias) {
    g += bf16_to_f32(bias[j]);
    u += bf16_to_f32(bias[I + j]);
  }
  g = fminf(round_bf16(g), limit);
  u = fminf(fmaxf(round_bf16(u), -limit), limit);
  const float s = round_bf16(g / (1.0f + expf(-g)));
  out[i] = f32_to_bf16(s * u);
}

__global__ void vit_layernorm_gelu_kernel(const bf16* x, int D, const bf16* w, const bf16* b, float eps, bf16* out) {
  __shared__ float red[32];
  const bf16* xr = x + size_t(blockIdx.x) * D;
  float s = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += bf16_to_f32(xr[i]);
  const float mean = block_sum(s, red) / float(D);
  float v = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float d = bf16_to_f32(xr[i]) - mean;
    v += d * d;
  }
  const float rstd = rsqrtf(block_sum(v, red) / float(D) + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float y = round_bf16((bf16_to_f32(xr[i]) - mean) * rstd * bf16_to_f32(w[i]) + bf16_to_f32(b[i]));
    out[size_t(blockIdx.x) * D + i] = f32_to_bf16(0.5f * y * (1.0f + erff(y * 0.70710678118654752f)));
  }
}

__global__ void vit_add_kernel(bf16* x, const bf16* y, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = f32_to_bf16(bf16_to_f32(x[i]) + bf16_to_f32(y[i]));
}

// One block per (row, q-or-k): H heads x 32 pairs.
__global__ void vit_rope_kernel(bf16* qkv, int H, const float* cos, const float* sin) {
  const int n = blockIdx.x, part = blockIdx.y;  // 0: q, 1: k
  bf16* row = qkv + size_t(n) * 3 * H * 64 + size_t(part) * H * 64;
  for (int idx = threadIdx.x; idx < H * 32; idx += blockDim.x) {
    const int h = idx / 32, j = idx % 32;
    bf16* x = row + h * 64;
    const float c = cos[size_t(n) * 32 + j], s = sin[size_t(n) * 32 + j];
    const float x1 = bf16_to_f32(x[j]), x2 = bf16_to_f32(x[32 + j]);
    x[j] = f32_to_bf16(x1 * c - x2 * s);
    x[32 + j] = f32_to_bf16(x2 * c + x1 * s);
  }
}

// Flash-style: block = 32 queries of one head, 256 threads; thread t owns query t / 8 and, per 64-key
// tile, keys (t % 8) * 8 .. +7 for the scores and output dims (t % 8) * 8 .. +7 for P V.
constexpr int kQ = 32, kK = 64, kD = 64;
__global__ __launch_bounds__(256) void vit_attention_kernel(const bf16* qkv, int N, int H, bf16* out) {
  __shared__ float sq[kQ][kD + 1];
  __shared__ float sk[kK][kD + 1];
  __shared__ float sv[kK][kD + 1];
  __shared__ float sp[kQ][kK + 1];
  const int h = blockIdx.y, q0 = blockIdx.x * kQ, t = threadIdx.x;
  const int qi = t / 8, part = t % 8;
  const size_t ld = size_t(3) * H * kD;
  for (int i = t; i < kQ * kD; i += 256) {
    const int r = i / kD, d = i % kD;
    sq[r][d] = q0 + r < N ? bf16_to_f32(qkv[size_t(q0 + r) * ld + size_t(h) * kD + d]) : 0.0f;
  }
  float m = -INFINITY, l = 0.0f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float scale = 0.125f;  // 1 / sqrt(64)
  for (int k0 = 0; k0 < N; k0 += kK) {
    __syncthreads();
    for (int i = t; i < kK * kD; i += 256) {
      const int r = i / kD, d = i % kD;
      const bool ok = k0 + r < N;
      const size_t base = size_t(k0 + r) * ld + size_t(h) * kD + d;
      sk[r][d] = ok ? bf16_to_f32(qkv[base + size_t(H) * kD]) : 0.0f;
      sv[r][d] = ok ? bf16_to_f32(qkv[base + size_t(2) * H * kD]) : 0.0f;
    }
    __syncthreads();
    float s[8], bm = -INFINITY;
    for (int j = 0; j < 8; ++j) {
      const int key = part * 8 + j;
      float dot = 0.0f;
      for (int d = 0; d < kD; ++d) dot = fmaf(sq[qi][d], sk[key][d], dot);
      s[j] = k0 + key < N ? dot * scale : -INFINITY;
      bm = fmaxf(bm, s[j]);
    }
    for (int o = 1; o < 8; o <<= 1) bm = fmaxf(bm, __shfl_xor(bm, o, 32));
    const float mn = fmaxf(m, bm), f = expf(m - mn);
    float ps = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const float p = s[j] == -INFINITY ? 0.0f : expf(s[j] - mn);
      sp[qi][part * 8 + j] = p;
      ps += p;
    }
    for (int o = 1; o < 8; o <<= 1) ps += __shfl_xor(ps, o, 32);
    l = l * f + ps;
    m = mn;
    __syncthreads();
    for (int j = 0; j < 8; ++j) {
      float a = acc[j] * f;
      const int d = part * 8 + j;
      for (int key = 0; key < kK; ++key) a = fmaf(sp[qi][key], sv[key][d], a);
      acc[j] = a;
    }
  }
  if (q0 + qi < N)
    for (int j = 0; j < 8; ++j)
      out[size_t(q0 + qi) * H * kD + size_t(h) * kD + part * 8 + j] = f32_to_bf16(acc[j] / l);
}

// WMMA flash attention: a block of 8 waves takes 128 queries of one head, a wave 16 of them. Per 64-key tile (K
// as is and V transposed in LDS, so keys and V's columns are B fragments): S = Q K^T in fp32 (16 WMMAs), the online
// softmax per query row across the 16 lanes holding its keys, P to LDS as bf16 and back as A fragments, O += P V.
// Fragment layouts: see kernels/exl3_moe.hip.
constexpr int kWQ = 128;  // queries per block
#if defined(UH_GFX12) || defined(UH_GFX11)
constexpr int kWK = 64, kWStr = 64 + 8;  // keys per tile, LDS row stride
typedef float float8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));
#if defined(UH_GFX12)
using frag_t = bf16x8_t;
#else
using frag_t = bf16x16_t;
#endif
// The A / B fragment of this lane, whose row (or column) starts at `row`, for k = k0 .. k0 + 15.
__device__ __forceinline__ frag_t frag_at(const uint16_t* row, int k0, int lane) {
#if defined(UH_GFX12)
  return *reinterpret_cast<const bf16x8_t*>(row + k0 + (lane >> 4) * 8);
#else
  const bf16x8_t a = *reinterpret_cast<const bf16x8_t*>(row + k0), b = *reinterpret_cast<const bf16x8_t*>(row + k0 + 8);
  return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#endif
}
__device__ __forceinline__ float8_t wmma_bf16(frag_t a, frag_t b, float8_t c) {
#if defined(UH_GFX12)
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
}
// Row of the 16 x 16 accumulator that element e of this lane holds (its column is lane % 16).
__device__ __forceinline__ int acc_row(int e, int lane) {
#if defined(UH_GFX12)
  return (lane >> 4) * 8 + e;
#else
  return 2 * e + (lane >> 4);
#endif
}
#endif

__global__ __launch_bounds__(256) void vit_attention_wmma_kernel(const bf16* qkv, int N, int H, bf16* out) {
#if defined(UH_GFX12) || defined(UH_GFX11)
  __shared__ alignas(16) uint16_t sK[kWK * kWStr];   // [key][dim]
  __shared__ alignas(16) uint16_t sVt[64 * kWStr];   // [dim][key]
  __shared__ alignas(16) uint16_t sP[8][16 * kWStr];  // per wave [query][key]
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, h = blockIdx.y;
  const int q0 = blockIdx.x * kWQ + wave * 16;
  const size_t ld = size_t(3) * H * 64;
  frag_t qf[4];
  {
    // Rows past N repeat the last query; they are never stored.
    const bf16* row = qkv + size_t(min(q0 + (lane & 15), N - 1)) * ld + size_t(h) * 64;
#pragma unroll
    for (int d = 0; d < 4; ++d) qf[d] = frag_at(row, d * 16, lane);
  }
  float8_t o[4];
  float m[8], l[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    m[e] = -INFINITY;
    l[e] = 0.0f;
#pragma unroll
    for (int f = 0; f < 4; ++f) o[f][e] = 0.0f;
  }
  constexpr float kScale = 0.125f * 1.4426950408889634f;  // 1 / sqrt(64), in log2 units
  uint16_t* P = sP[wave];
  for (int k0 = 0; k0 < N; k0 += kWK) {
    __syncthreads();
    for (int i = tid; i < kWK * 8; i += 256) {
      const int r = i >> 3, c = (i & 7) * 8;
      uint4 kv = {0, 0, 0, 0}, vv = {0, 0, 0, 0};
      if (k0 + r < N) {
        const bf16* row = qkv + size_t(k0 + r) * ld + size_t(h) * 64 + c;
        kv = *reinterpret_cast<const uint4*>(row + size_t(H) * 64);
        vv = *reinterpret_cast<const uint4*>(row + size_t(2) * H * 64);
      }
      *reinterpret_cast<uint4*>(&sK[r * kWStr + c]) = kv;
      const uint32_t w[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) sVt[(c + j) * kWStr + r] = uint16_t(w[j >> 1] >> ((j & 1) * 16));
    }
    __syncthreads();
    float8_t s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 8; ++e) s[j][e] = 0.0f;
#pragma unroll
      for (int d = 0; d < 4; ++d) s[j] = wmma_bf16(qf[d], frag_at(&sK[(j * 16 + (lane & 15)) * kWStr], d * 16, lane), s[j]);
    }
    // Every tile has a visible key (k0 < N) in lane 0's column, so each row's max is finite.
    const int key = k0 + (lane & 15);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][e] = key + j * 16 < N ? s[j][e] * kScale : -INFINITY;
        mx = fmaxf(mx, s[j][e]);
      }
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) mx = fmaxf(mx, __shfl_xor(mx, x, 32));
      const float mn = fmaxf(m[e], mx), alpha = exp2f(m[e] - mn);
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][e] = exp2f(s[j][e] - mn);
        sum += s[j][e];
        P[acc_row(e, lane) * kWStr + j * 16 + (lane & 15)] = f32_to_bf16(s[j][e]);
      }
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) sum += __shfl_xor(sum, x, 32);
      l[e] = l[e] * alpha + sum;
      m[e] = mn;
#pragma unroll
      for (int f = 0; f < 4; ++f) o[f][e] *= alpha;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const frag_t pf = frag_at(&P[(lane & 15) * kWStr], kk * 16, lane);
#pragma unroll
      for (int f = 0; f < 4; ++f) o[f] = wmma_bf16(pf, frag_at(&sVt[(f * 16 + (lane & 15)) * kWStr], kk * 16, lane), o[f]);
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int q = q0 + acc_row(e, lane);
    if (q < N)
#pragma unroll
      for (int f = 0; f < 4; ++f)
        out[size_t(q) * H * 64 + size_t(h) * 64 + f * 16 + (lane & 15)] = f32_to_bf16(o[f][e] / l[e]);
  }
#endif
}

__global__ void vit_unfold_kernel(const bf16* x, int n_h, int n_w, int C, int r, int lw, bf16* out) {
  const int l = blockIdx.x;
  const int R = l / lw, S = l % lw, K = C * r * r;
  for (int col = threadIdx.x; col < K; col += blockDim.x) {
    const int c = col / (r * r), ij = col % (r * r), i = ij / r, j = ij % r;
    const int y = R * r + i, xx = S * r + j;
    out[size_t(l) * K + col] = y < n_h && xx < n_w ? x[(size_t(y) * n_w + xx) * C + c] : bf16(0);
  }
}

__global__ void vit_assemble_kernel(const bf16* aligned, const bf16* params, const int32_t* src, int D, bf16* out) {
  const int i = blockIdx.x, s = src[i];
  const bf16* row = s >= 0 ? aligned + size_t(s) * D : params + size_t(-1 - s) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) out[size_t(i) * D + d] = row[d];
}

unsigned blocks_for(size_t n) { return unsigned((n + 255) / 256); }

}  // namespace

void vit_bias(const float* acc, int M, int N, const bf16* bias, bf16* out, hipStream_t st) {
  const size_t n = size_t(M) * N;
  vit_bias_kernel<<<blocks_for(n), 256, 0, st>>>(acc, N, n, bias, out);
  UH_CHECK_LAUNCH();
}
void vit_bias_gelu(const float* acc, int M, int N, const bf16* bias, bf16* out, hipStream_t st) {
  const size_t n = size_t(M) * N;
  vit_bias_gelu_kernel<<<blocks_for(n), 256, 0, st>>>(acc, N, n, bias, out);
  UH_CHECK_LAUNCH();
}
void vit_silu_mul(const float* acc, int M, int I, bf16* out, hipStream_t st) {
  const size_t n = size_t(M) * I;
  vit_silu_mul_kernel<<<blocks_for(n), 256, 0, st>>>(acc, I, n, out);
  UH_CHECK_LAUNCH();
}
void vit_rmsnorm(const bf16* x, int M, int D, const bf16* w, float eps, bf16* out, hipStream_t st) {
  if (M <= 0) return;
  vit_rmsnorm_kernel<false><<<M, 256, 0, st>>>(x, D, w, eps, out);
  UH_CHECK_LAUNCH();
}
void vit_rmsnorm_hf(const bf16* x, int M, int D, const bf16* w, float eps, bf16* out, hipStream_t st) {
  if (M <= 0) return;
  vit_rmsnorm_kernel<true><<<M, 256, 0, st>>>(x, D, w, eps, out);
  UH_CHECK_LAUNCH();
}
void vit_qk_norm_rope(bf16* qkv, int N, int H, const bf16* qw, const bf16* kw, float eps, const float* cos,
                      const float* sin, hipStream_t st) {
  if (N <= 0) return;
  vit_qk_norm_rope_kernel<<<blocks_for(size_t(N) * 2 * H * 32), 256, 0, st>>>(qkv, N, H, qw, kw, eps, cos, sin);
  UH_CHECK_LAUNCH();
}
void vit_swiglu_clamp(const float* acc, int M, int I, const bf16* bias, float limit, bf16* out, hipStream_t st) {
  const size_t n = size_t(M) * I;
  if (n == 0) return;
  vit_swiglu_clamp_kernel<<<blocks_for(n), 256, 0, st>>>(acc, I, n, bias, limit, out);
  UH_CHECK_LAUNCH();
}
void vit_layernorm_gelu(const bf16* x, int M, int D, const bf16* w, const bf16* b, float eps, bf16* out,
                        hipStream_t st) {
  if (M <= 0) return;
  vit_layernorm_gelu_kernel<<<M, 256, 0, st>>>(x, D, w, b, eps, out);
  UH_CHECK_LAUNCH();
}
void vit_add(bf16* x, const bf16* y, size_t n, hipStream_t st) {
  vit_add_kernel<<<blocks_for(n), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}
void vit_rope(bf16* qkv, int N, int H, const float* cos, const float* sin, hipStream_t st) {
  if (N <= 0) return;
  vit_rope_kernel<<<dim3(unsigned(N), 2), 256, 0, st>>>(qkv, H, cos, sin);
  UH_CHECK_LAUNCH();
}
void vit_attention(const bf16* qkv, int N, int H, bf16* out, hipStream_t st) {
  if (N <= 0) return;
  vit_attention_kernel<<<dim3(unsigned((N + kQ - 1) / kQ), unsigned(H)), 256, 0, st>>>(qkv, N, H, out);
  UH_CHECK_LAUNCH();
}
void vit_attention_wmma(const bf16* qkv, int N, int H, bf16* out, hipStream_t st) {
  if (N <= 0) return;
  vit_attention_wmma_kernel<<<dim3(unsigned((N + kWQ - 1) / kWQ), unsigned(H)), 256, 0, st>>>(qkv, N, H, out);
  UH_CHECK_LAUNCH();
}
void vit_unfold(const bf16* x, int n_h, int n_w, int C, int r, bf16* out, hipStream_t st) {
  const int lh = (n_h + r - 1) / r, lw = (n_w + r - 1) / r;
  vit_unfold_kernel<<<lh * lw, 256, 0, st>>>(x, n_h, n_w, C, r, lw, out);
  UH_CHECK_LAUNCH();
}
void vit_assemble(const bf16* aligned, const bf16* params, const int32_t* src, int n, int D, bf16* out,
                  hipStream_t st) {
  if (n <= 0) return;
  vit_assemble_kernel<<<n, 256, 0, st>>>(aligned, params, src, D, out);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
