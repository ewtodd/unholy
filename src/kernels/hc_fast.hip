// Hyper-Connection pre-mixing, tiled over tokens (fast path; reference: hc_mix_kernel in ref_elem.hip).
//
// The reference kernel reads the [24, 4*4096] fp32 mixing matrix once per token (1.5 MB): at 1024
// tokens that is 1.5 GB of L2 traffic per call. Here a block takes 16 tokens; each K-chunk of the
// matrix is staged in LDS once and used by all 16. Thread layout: token tid/16, K lanes tid%16.
#include "kernels/dev.hip.hpp"
#include "kernels/hc_fast.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int TB = 16, KC = 256, MIX = 24, NT = 256;

__global__ __launch_bounds__(NT) void hc_pre_tiled_kernel(const bf16* __restrict__ x, const float* __restrict__ fn,
                                                          const float* __restrict__ scale,
                                                          const float* __restrict__ base, bf16* __restrict__ y,
                                                          float* __restrict__ post_out, float* __restrict__ comb_out,
                                                          int T, int hc, int dim, int iters, float norm_eps,
                                                          float hc_eps) {
  __shared__ float sFn[MIX][KC];
  __shared__ float sMix[TB][MIX];
  __shared__ float sPre[TB][4];
  const int width = hc * dim;
  const int tid = threadIdx.x, tt = tid >> 4, kl = tid & 15;
  const int t = blockIdx.x * TB + tt;
  const bool valid = t < T;
  const bf16* xr = x + size_t(valid ? t : 0) * width;
  float acc[MIX];
#pragma unroll
  for (int j = 0; j < MIX; ++j) acc[j] = 0.0f;
  float ss = 0.0f;
  for (int k0 = 0; k0 < width; k0 += KC) {
    for (int i = tid; i < MIX * KC; i += NT) sFn[i / KC][i % KC] = fn[size_t(i / KC) * width + k0 + i % KC];
    __syncthreads();
    if (valid)
      for (int kk = kl; kk < KC; kk += 16) {
        const float v = bf16_to_f32(xr[k0 + kk]);
        ss += v * v;
#pragma unroll
        for (int j = 0; j < MIX; ++j) acc[j] = fmaf(v, sFn[j][kk], acc[j]);
      }
    __syncthreads();
  }
  // Reduce over the 16 K lanes of each token (contiguous lanes within the wave).
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    ss += __shfl_xor(ss, o, 32);
#pragma unroll
    for (int j = 0; j < MIX; ++j) acc[j] += __shfl_xor(acc[j], o, 32);
  }
  if (kl == 0) {
    const float rs = rsqrtf(ss / float(width) + norm_eps);
#pragma unroll
    for (int j = 0; j < MIX; ++j) sMix[tt][j] = acc[j] * rs;
  }
  __syncthreads();
  // Sinkhorn per token (one thread each), same op order as the reference kernel. hc is 4 (checked by
  // the launcher) and a compile-time constant here, so the arrays stay in registers (runtime-indexed
  // they were spilled to scratch).
  if (tid < TB && blockIdx.x * TB + tid < T) {
    constexpr int HC = 4;
    const float* m = sMix[tid];
    float pre[HC], post[HC], comb[HC][HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) pre[j] = 1.0f / (1.0f + expf(-(m[j] * scale[0] + base[j]))) + hc_eps;
#pragma unroll
    for (int j = 0; j < HC; ++j) post[j] = 2.0f / (1.0f + expf(-(m[HC + j] * scale[1] + base[HC + j])));
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        comb[j][k] = m[2 * HC + j * HC + k] * scale[2] + base[2 * HC + j * HC + k];
        mx = fmaxf(mx, comb[j][k]);
      }
      float sum = 0.0f;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        comb[j][k] = expf(comb[j][k] - mx);
        sum += comb[j][k];
      }
#pragma unroll
      for (int k = 0; k < HC; ++k) comb[j][k] = comb[j][k] / sum + hc_eps;
    }
    auto col_norm = [&]() {
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        float cs = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) cs += comb[j][k];
#pragma unroll
        for (int j = 0; j < HC; ++j) comb[j][k] = comb[j][k] / (cs + hc_eps);
      }
    };
    col_norm();
    for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float rsum = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) rsum += comb[j][k];
#pragma unroll
        for (int k = 0; k < HC; ++k) comb[j][k] = comb[j][k] / (rsum + hc_eps);
      }
      col_norm();
    }
    const int tok = blockIdx.x * TB + tid;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      sPre[tid][j] = pre[j];
      post_out[size_t(tok) * HC + j] = post[j];
#pragma unroll
      for (int k = 0; k < HC; ++k) comb_out[(size_t(tok) * HC + j) * HC + k] = comb[j][k];
    }
  }
  __syncthreads();
  // y = sum_i pre_i x_i: 16 tokens x dim, 256 threads.
  for (int i = tid; i < TB * dim; i += NT) {
    const int tk = i / dim, d = i % dim;
    const int tok = blockIdx.x * TB + tk;
    if (tok >= T) continue;
    const bf16* xt = x + size_t(tok) * width;
    float v = 0.0f;
    for (int h = 0; h < hc; ++h) v += sPre[tk][h] * bf16_to_f32(xt[size_t(h) * dim + d]);
    y[size_t(tok) * dim + d] = f32_to_bf16(v);
  }
}


// Small T: grid (T, 32 K-splits) partial sums, then one block per token for Sinkhorn and the mix.
constexpr int SPLITS = 32;

__global__ __launch_bounds__(NT) void hc_partial_kernel(const bf16* __restrict__ x, const float* __restrict__ fn,
                                                        float* __restrict__ part, int width) {
  __shared__ float red[MIX + 1][8];
  const int t = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int ks = width / SPLITS, k0 = sp * ks;
  const bf16* xr = x + size_t(t) * width;
  float acc[MIX + 1];
#pragma unroll
  for (int j = 0; j <= MIX; ++j) acc[j] = 0.0f;
  for (int k = k0 + tid; k < k0 + ks; k += NT) {
    const float v = bf16_to_f32(xr[k]);
    acc[MIX] += v * v;
#pragma unroll
    for (int j = 0; j < MIX; ++j) acc[j] = fmaf(v, fn[size_t(j) * width + k], acc[j]);
  }
#pragma unroll
  for (int j = 0; j <= MIX; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) red[j][wave] = v;
  }
  __syncthreads();
  if (tid <= MIX) {
    float v = 0.0f;
    for (int w = 0; w < 8; ++w) v += red[tid][w];
    part[(size_t(t) * SPLITS + sp) * (MIX + 1) + tid] = v;
  }
}

__global__ __launch_bounds__(NT) void hc_finish_kernel(const bf16* __restrict__ x, const float* __restrict__ part,
                                                       const float* __restrict__ scale, const float* __restrict__ base,
                                                       bf16* __restrict__ y, float* __restrict__ post_out,
                                                       float* __restrict__ comb_out, int hc, int dim, int iters,
                                                       float norm_eps, float hc_eps) {
  __shared__ float m[MIX + 1];
  __shared__ float pre[4];
  const int t = blockIdx.x, tid = threadIdx.x, width = hc * dim;
  if (tid <= MIX) {
    float v = 0.0f;
    for (int sp = 0; sp < SPLITS; ++sp) v += part[(size_t(t) * SPLITS + sp) * (MIX + 1) + tid];
    m[tid] = v;
  }
  __syncthreads();
  if (tid == 0) {
    const float rs = rsqrtf(m[MIX] / float(width) + norm_eps);
    constexpr int HC = 4;  // checked by the launcher; constant so the arrays below stay in registers
    float mx[MIX];
    _Pragma("unroll") for (int j = 0; j < MIX; ++j) mx[j] = m[j] * rs;
    float post[HC], comb[HC][HC];
    _Pragma("unroll") for (int j = 0; j < HC; ++j) pre[j] = 1.0f / (1.0f + expf(-(mx[j] * scale[0] + base[j]))) + hc_eps;
    _Pragma("unroll") for (int j = 0; j < HC; ++j) post[j] = 2.0f / (1.0f + expf(-(mx[HC + j] * scale[1] + base[HC + j])));
    _Pragma("unroll") for (int j = 0; j < HC; ++j) {
      float mm = -INFINITY;
      _Pragma("unroll") for (int k = 0; k < HC; ++k) {
        comb[j][k] = mx[2 * HC + j * HC + k] * scale[2] + base[2 * HC + j * HC + k];
        mm = fmaxf(mm, comb[j][k]);
      }
      float sum = 0.0f;
      _Pragma("unroll") for (int k = 0; k < HC; ++k) {
        comb[j][k] = expf(comb[j][k] - mm);
        sum += comb[j][k];
      }
      _Pragma("unroll") for (int k = 0; k < HC; ++k) comb[j][k] = comb[j][k] / sum + hc_eps;
    }
    auto col_norm = [&]() {
      _Pragma("unroll") for (int k = 0; k < HC; ++k) {
        float cs = 0.0f;
        _Pragma("unroll") for (int j = 0; j < HC; ++j) cs += comb[j][k];
        _Pragma("unroll") for (int j = 0; j < HC; ++j) comb[j][k] = comb[j][k] / (cs + hc_eps);
      }
    };
    col_norm();
    for (int it = 0; it < iters - 1; ++it) {
      _Pragma("unroll") for (int j = 0; j < HC; ++j) {
        float rsum = 0.0f;
        _Pragma("unroll") for (int k = 0; k < HC; ++k) rsum += comb[j][k];
        _Pragma("unroll") for (int k = 0; k < HC; ++k) comb[j][k] = comb[j][k] / (rsum + hc_eps);
      }
      col_norm();
    }
    _Pragma("unroll") for (int j = 0; j < HC; ++j) {
      post_out[size_t(t) * HC + j] = post[j];
      _Pragma("unroll") for (int k = 0; k < HC; ++k) comb_out[(size_t(t) * HC + j) * HC + k] = comb[j][k];
    }
  }
  __syncthreads();
  const bf16* xt = x + size_t(t) * width;
  for (int d = tid; d < dim; d += NT) {
    float v = 0.0f;
    for (int h = 0; h < hc; ++h) v += pre[h] * bf16_to_f32(xt[size_t(h) * dim + d]);
    y[size_t(t) * dim + d] = f32_to_bf16(v);
  }
}

}  // namespace

void hc_pre_tiled(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st) {
  if (hc != 4 || (hc * dim) % KC) throw std::runtime_error("hc_pre_tiled: unsupported shape");
  hc_pre_tiled_kernel<<<(T + TB - 1) / TB, NT, 0, st>>>(x, fn, scale, base, y, post, comb, T, hc, dim, iters,
                                                         norm_eps, hc_eps);
  UH_CHECK_LAUNCH();
}

void hc_pre_small(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, float* part, int T, int hc, int dim, int iters, float norm_eps, float hc_eps,
                  hipStream_t st) {
  if (hc != 4 || (hc * dim) % (SPLITS * NT)) throw std::runtime_error("hc_pre_small: unsupported shape");
  hc_partial_kernel<<<dim3(T, SPLITS), NT, 0, st>>>(x, fn, part, hc * dim);
  UH_CHECK_LAUNCH();
  hc_finish_kernel<<<T, NT, 0, st>>>(x, part, scale, base, y, post, comb, hc, dim, iters, norm_eps, hc_eps);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
