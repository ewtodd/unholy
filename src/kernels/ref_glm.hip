// GLM-5.3-Flash reference kernels (kernels/ref_glm.hpp). The GEMMs are in ref_gemm.hip.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <stdexcept>

#include "kernels/dev.hip.hpp"
#include "kernels/ref_glm.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

__device__ __forceinline__ float f16_to_f32(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float load_h(const void* p, bool f16, size_t i) {
  const uint16_t v = static_cast<const uint16_t*>(p)[i];
  return f16 ? f16_to_f32(v) : bf16_to_f32(v);
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- norms / elementwise ----
__global__ void rmsnorm_hf_kernel(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int dim, float eps) {
  __shared__ float red[32];
  const bf16* xr = x + size_t(blockIdx.x) * ldx;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf16_to_f32(xr[i]);
    ss += v * v;
  }
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / float(dim) + eps);
  bf16* yr = y + size_t(blockIdx.x) * ldy;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    yr[i] = f32_to_bf16(bf16_to_f32(w[i]) * round_bf16(bf16_to_f32(xr[i]) * r));
}

__global__ void layernorm_kernel(const bf16* x, const void* w, const void* b, bool f16, bf16* y, int dim, float eps) {
  __shared__ float red[32];
  const bf16* xr = x + size_t(blockIdx.x) * dim;
  float s = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) s += bf16_to_f32(xr[i]);
  const float mean = block_sum(s, red) / float(dim);
  float v = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = bf16_to_f32(xr[i]) - mean;
    v += d * d;
  }
  const float rstd = rsqrtf(block_sum(v, red) / float(dim) + eps);
  bf16* yr = y + size_t(blockIdx.x) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    yr[i] = f32_to_bf16((bf16_to_f32(xr[i]) - mean) * rstd * load_h(w, f16, i) + load_h(b, f16, i));
}

__global__ void hc_post_hf_kernel(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out,
                                  int hc, int dim) {
  const int t = blockIdx.x;
  float pt[4], ct[4][4];
  for (int j = 0; j < hc; ++j) pt[j] = round_bf16(post[size_t(t) * hc + j]);
  for (int i = 0; i < hc; ++i)
    for (int j = 0; j < hc; ++j) ct[i][j] = round_bf16(comb[(size_t(t) * hc + i) * hc + j]);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float xv = bf16_to_f32(x[size_t(t) * dim + d]);
    float r[4];
    for (int i = 0; i < hc; ++i) r[i] = bf16_to_f32(res[(size_t(t) * hc + i) * dim + d]);
    for (int j = 0; j < hc; ++j) {
      float s = 0.0f;
      for (int i = 0; i < hc; ++i) s = fmaf(ct[i][j], r[i], s);
      out[(size_t(t) * hc + j) * dim + d] = f32_to_bf16(round_bf16(pt[j] * xv) + round_bf16(s));
    }
  }
}

__global__ void hc_mean_kernel(const bf16* x, bf16* y, int hc, int dim) {
  const int t = blockIdx.x;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float s = 0.0f;
    for (int i = 0; i < hc; ++i) s += bf16_to_f32(x[(size_t(t) * hc + i) * dim + d]);
    y[size_t(t) * dim + d] = f32_to_bf16(s / float(hc));
  }
}

__global__ void sigmoid_bf16_kernel(const bf16* x, bf16* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = f32_to_bf16(sigmoidf(bf16_to_f32(x[i])));
}

__global__ void round_f32_bf16_kernel(float* x, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = round_bf16(x[i]);
}

// ---- KDA ----
// One thread per channel, the chunk's rows in order.
__global__ void kda_conv_kernel(const bf16* x, int ldx, bf16* state, const bf16* w, bf16* y, int n, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float w4[4], h[3];
  for (int i = 0; i < 4; ++i) w4[i] = bf16_to_f32(w[size_t(c) * 4 + i]);
  for (int i = 0; i < 3; ++i) h[i] = bf16_to_f32(state[size_t(i) * C + c]);
  for (int t = 0; t < n; ++t) {
    const float xt = bf16_to_f32(x[size_t(t) * ldx + c]);
    float acc = w4[0] * h[0];
    acc = fmaf(w4[1], h[1], acc);
    acc = fmaf(w4[2], h[2], acc);
    acc = fmaf(w4[3], xt, acc);
    const float a = round_bf16(acc);
    y[size_t(t) * C + c] = f32_to_bf16(a * sigmoidf(a));
    h[0] = h[1];
    h[1] = h[2];
    h[2] = xt;
  }
  for (int i = 0; i < 3; ++i) state[size_t(i) * C + c] = f32_to_bf16(h[i]);
}

__global__ void kda_gate_kernel(const bf16* fb, const float* dt_bias, const float* a_log, float* g, size_t n, int HD,
                                int D, float lb) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int c = int(i % HD);
    g[i] = lb * sigmoidf(expf(a_log[c / D]) * (bf16_to_f32(fb[i]) + dt_bias[c]));
  }
}

// One block per head, one thread per value column v holding S[:, v] in registers.
template <int D>
__global__ __launch_bounds__(D) void kda_recurrent_kernel(const bf16* q, const bf16* k, const bf16* v, int ld,
                                                          const float* g, const bf16* beta, float* state, bf16* o,
                                                          int n, int H) {
  __shared__ float qs[D], ks[D], es[D];
  __shared__ float red[32];
  const int h = blockIdx.x, j = threadIdx.x;
  float* sh = state + size_t(h) * D * D;
  float s[D];
#pragma unroll
  for (int i = 0; i < D; ++i) s[i] = sh[size_t(i) * D + j];
  for (int t = 0; t < n; ++t) {
    const size_t row = size_t(t) * ld + size_t(h) * D;
    const float qv = bf16_to_f32(q[row + j]), kv = bf16_to_f32(k[row + j]);
    const float qn = block_sum(qv * qv, red);
    const float kn = block_sum(kv * kv, red);
    qs[j] = qv / sqrtf(qn + 1e-6f) * (1.0f / sqrtf(float(D)));
    ks[j] = kv / sqrtf(kn + 1e-6f);
    es[j] = expf(g[(size_t(t) * H + h) * D + j]);
    __syncthreads();
    const float b = bf16_to_f32(beta[size_t(t) * H + h]);
    float mem = 0.0f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      s[i] *= es[i];
      mem = fmaf(s[i], ks[i], mem);
    }
    const float delta = (bf16_to_f32(v[row + j]) - mem) * b;
    float out = 0.0f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      s[i] = fmaf(ks[i], delta, s[i]);
      out = fmaf(s[i], qs[i], out);
    }
    o[(size_t(t) * H + h) * D + j] = f32_to_bf16(out);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < D; ++i) sh[size_t(i) * D + j] = s[i];
}

__global__ void kda_out_norm_kernel(const bf16* x, const bf16* gate, const bf16* w, bf16* o, int D, float eps) {
  __shared__ float red[32];
  const size_t base = size_t(blockIdx.x) * D;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = bf16_to_f32(x[base + i]);
    ss += v * v;
  }
  const float r = rsqrtf(block_sum(ss, red) / float(D) + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    o[base + i] =
        f32_to_bf16(bf16_to_f32(w[i]) * (bf16_to_f32(x[base + i]) * r) * sigmoidf(bf16_to_f32(gate[base + i])));
}

// ---- indexer ----
__global__ void idx_pool_kernel(const bf16* k, const bf16* gs, const float* ape, bf16* out, int p0, int np, int pool,
                                int D) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(np) * D) return;
  const int p = p0 + int(i / D), d = int(i % D);
  float l[8], mx = -INFINITY;
  for (int j = 0; j < pool; ++j) {
    l[j] = bf16_to_f32(gs[(size_t(p) * pool + j) * D + d]) + ape[size_t(j) * D + d];
    mx = fmaxf(mx, l[j]);
  }
  float sum = 0.0f;
  for (int j = 0; j < pool; ++j) {
    l[j] = expf(l[j] - mx);
    sum += l[j];
  }
  float acc = 0.0f;
  for (int j = 0; j < pool; ++j)
    acc += round_bf16(round_bf16(l[j] / sum) * bf16_to_f32(k[(size_t(p) * pool + j) * D + d]));
  out[size_t(p) * D + d] = f32_to_bf16(acc);
}

// One block of 128 threads per (query, 128 pools); the query's heads in LDS.
__global__ void idx_scores_kernel(const bf16* q, const bf16* w, const bf16* pk, const int32_t* pos, float* out,
                                  int npool, int H, int D, int pool, float scale) {
  extern __shared__ float qsh[];  // [H * D + H]
  const int t = blockIdx.y;
  for (int i = threadIdx.x; i < H * D; i += blockDim.x) qsh[i] = bf16_to_f32(q[size_t(t) * H * D + i]);
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    qsh[H * D + i] = bf16_to_f32(w[size_t(t) * H + i]) * rsqrtf(float(H));
  __syncthreads();
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npool) return;
  float r = -INFINITY;
  if (p * pool + pool - 1 <= pos[t]) {
    r = 0.0f;
    const bf16* kr = pk + size_t(p) * D;
    for (int h = 0; h < H; ++h) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot = fmaf(qsh[h * D + d], bf16_to_f32(kr[d]), dot);
      r = fmaf(qsh[H * D + h], fmaxf(dot * scale, 0.0f), r);
    }
  }
  out[size_t(t) * npool + p] = r;
}

__global__ void idx_expand_kernel(const int32_t* pools, const int32_t* pos, int32_t* idx, int n, int k, int pool) {
  const int width = k * pool + pool - 1;
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(n) * width) return;
  const int t = int(i / width), j = int(i % width);
  int v;
  if (j < k * pool) {
    const int p = pools[size_t(t) * k + j / pool];
    v = p < 0 ? -1 : p * pool + j % pool;
  } else {
    const int jt = j - k * pool, tail = (pos[t] + 1) % pool;
    v = jt < tail ? pos[t] + 1 - tail + jt : -1;
  }
  idx[i] = v;
}

// ---- attention: one block per (query, head), online softmax; each thread owns r / 256 latent dims ----
template <int R>
__global__ __launch_bounds__(256) void mla_attn_kernel(const float* qa, const bf16* kv, const int32_t* idx, int w,
                                                       const int32_t* pos, float* ol, int H, float scale) {
  constexpr int P = R / 256;
  __shared__ float red[32];
  const int t = blockIdx.x / H, h = blockIdx.x % H;
  const float* qr = qa + (size_t(t) * H + h) * R;
  float qv[P], acc[P];
  for (int i = 0; i < P; ++i) {
    qv[i] = qr[threadIdx.x + 256 * i];
    acc[i] = 0.0f;
  }
  const int nk = idx ? w : pos[t] + 1;
  float m = -INFINITY, l = 0.0f;
  for (int jj = 0; jj < nk; ++jj) {
    const int j = idx ? idx[size_t(t) * w + jj] : jj;
    if (j < 0) continue;
    const bf16* kr = kv + size_t(j) * R;
    float kvv[P], dot = 0.0f;
    for (int i = 0; i < P; ++i) {
      kvv[i] = bf16_to_f32(kr[threadIdx.x + 256 * i]);
      dot = fmaf(qv[i], kvv[i], dot);
    }
    const float s = block_sum(dot, red) * scale;
    const float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
    l = l * corr + p;
    for (int i = 0; i < P; ++i) acc[i] = fmaf(p, kvv[i], acc[i] * corr);
    m = mn;
  }
  float* orow = ol + (size_t(t) * H + h) * R;
  for (int i = 0; i < P; ++i) orow[threadIdx.x + 256 * i] = l > 0.0f ? acc[i] / l : 0.0f;
}

// ---- weights ----
// One block of 256 threads per 128x128 tile.
__global__ void quant_fp8_block_kernel(const bf16* w, uint8_t* q, float* s, int N, int K) {
  __shared__ float red[32];
  const int nb = blockIdx.y, kb = blockIdx.x, kbs = (K + 127) / 128;
  const int r0 = nb * 128, c0 = kb * 128, rows = min(128, N - r0), cols = min(128, K - c0);
  float amax = 0.0f;
  for (int i = threadIdx.x; i < rows * cols; i += blockDim.x)
    amax = fmaxf(amax, fabsf(bf16_to_f32(w[size_t(r0 + i / cols) * K + c0 + i % cols])));
  amax = warp_max(amax);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = 0.0f;
  for (int i = 0; i < int(blockDim.x / 32); ++i) amax = fmaxf(amax, red[i]);
  const float sc = fmaxf(amax, 1e-12f) / 448.0f;
  if (threadIdx.x == 0) s[size_t(nb) * kbs + kb] = sc;
  for (int i = threadIdx.x; i < rows * cols; i += blockDim.x) {
    const size_t at = size_t(r0 + i / cols) * K + c0 + i % cols;
    q[at] = f32_to_e4m3(fminf(fmaxf(bf16_to_f32(w[at]) / sc, -448.0f), 448.0f));
  }
}

// One wave per (row, 128-column group); each lane holds four values.
__global__ void fake_quant_fp8_group_kernel(bf16* x, int ldx, int rows, int cols) {
  const int ng = cols / 128;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= rows * ng) return;
  bf16* p = x + size_t(wid / ng) * ldx + size_t(wid % ng) * 128 + lane * 4;
  float v[4], amax = 0.0f;
  for (int i = 0; i < 4; ++i) {
    v[i] = bf16_to_f32(p[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float sc = fmaxf(warp_max(amax), 1e-10f) / 448.0f;
  for (int i = 0; i < 4; ++i)
    p[i] = f32_to_bf16(e4m3_to_f32(f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.0f), 448.0f))) * sc);
}

// ---- MoE ----
// One thread per token (E <= 512, K <= 16).
__global__ void route_sigmoid_kernel(const float* logits, const float* bias, int32_t* sel, float* weights, int T,
                                     int E, int K, float scale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* lr = logits + size_t(t) * E;
  int chosen[16];
  for (int k = 0; k < K; ++k) {
    int best = -1;
    float bv = -INFINITY;
    for (int e = 0; e < E; ++e) {
      bool taken = false;
      for (int i = 0; i < k; ++i) taken |= chosen[i] == e;
      if (taken) continue;
      const float v = sigmoidf(lr[e]) + bias[e];
      if (best < 0 || v > bv) best = e, bv = v;
    }
    chosen[k] = best;
  }
  for (int a = 1; a < K; ++a)  // ascending expert order
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

__global__ void swiglu_hf_kernel(const bf16* gate, const bf16* up, bf16* h, size_t n, float lim) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const float g = fminf(bf16_to_f32(gate[i]), lim);
    const float u = fminf(fmaxf(bf16_to_f32(up[i]), -lim), lim);
    h[i] = f32_to_bf16(round_bf16(g * sigmoidf(g)) * u);
  }
}

__global__ void moe_combine_hf_kernel(const bf16* y, const float* rw, const int32_t* pidx, const bf16* shared,
                                      bf16* out, int K, int dim) {
  const int t = blockIdx.x;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
      const int p = pidx[size_t(t) * K + j];
      acc = round_bf16(acc + round_bf16(bf16_to_f32(y[size_t(p) * dim + d]) * rw[p]));
    }
    out[size_t(t) * dim + d] = f32_to_bf16(acc + bf16_to_f32(shared[size_t(t) * dim + d]));
  }
}

}  // namespace

void rmsnorm_hf(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps,
                hipStream_t st) {
  if (rows <= 0) return;
  rmsnorm_hf_kernel<<<rows, 256, 0, st>>>(x, ldx, w, y, ldy, dim, eps);
  UH_CHECK_LAUNCH();
}

void layernorm(const bf16* x, const void* w, const void* b, bool f16, bf16* y, int rows, int dim, float eps,
               hipStream_t st) {
  if (rows <= 0) return;
  layernorm_kernel<<<rows, 128, 0, st>>>(x, w, b, f16, y, dim, eps);
  UH_CHECK_LAUNCH();
}

void hc_post_hf(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc,
                int dim, hipStream_t st) {
  if (hc != 4) throw std::runtime_error("hc_post_hf: hc must be 4");
  if (T <= 0) return;
  hc_post_hf_kernel<<<T, 256, 0, st>>>(x, res, post, comb, out, hc, dim);
  UH_CHECK_LAUNCH();
}

void hc_mean(const bf16* x, bf16* y, int T, int hc, int dim, hipStream_t st) {
  if (T <= 0) return;
  hc_mean_kernel<<<T, 256, 0, st>>>(x, y, hc, dim);
  UH_CHECK_LAUNCH();
}

void sigmoid_bf16(const bf16* x, bf16* y, size_t n, hipStream_t st) {
  if (n == 0) return;
  sigmoid_bf16_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}

void round_f32_bf16(float* x, size_t n, hipStream_t st) {
  if (n == 0) return;
  round_f32_bf16_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(x, n);
  UH_CHECK_LAUNCH();
}

void kda_conv(const bf16* x, int ldx, bf16* state, const bf16* w, bf16* y, int n, int C, hipStream_t st) {
  if (n <= 0) return;
  kda_conv_kernel<<<cdiv(size_t(C), 256), 256, 0, st>>>(x, ldx, state, w, y, n, C);
  UH_CHECK_LAUNCH();
}

void kda_gate(const bf16* fb, const float* dt_bias, const float* a_log, float* g, int T, int H, int D, float lb,
              hipStream_t st) {
  const size_t n = size_t(T) * H * D;
  if (n == 0) return;
  kda_gate_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(fb, dt_bias, a_log, g, n, H * D, D, lb);
  UH_CHECK_LAUNCH();
}

void kda_recurrent(const bf16* q, const bf16* k, const bf16* v, int ld, const float* g, const bf16* beta,
                   float* state, bf16* o, int n, int H, int D, hipStream_t st) {
  if (D != 128) throw std::runtime_error("kda_recurrent: head dim must be 128");
  if (n <= 0) return;
  kda_recurrent_kernel<128><<<H, 128, 0, st>>>(q, k, v, ld, g, beta, state, o, n, H);
  UH_CHECK_LAUNCH();
}

void kda_out_norm(const bf16* x, const bf16* gate, const bf16* w, bf16* o, int rows, int H, int D, float eps,
                  hipStream_t st) {
  if (rows <= 0) return;
  kda_out_norm_kernel<<<rows * H, 128, 0, st>>>(x, gate, w, o, D, eps);
  UH_CHECK_LAUNCH();
}

void idx_pool(const bf16* k, const bf16* gs, const float* ape, bf16* out, int p0, int np, int pool, int D,
              hipStream_t st) {
  if (pool > 8) throw std::runtime_error("idx_pool: pool > 8");
  if (np <= 0) return;
  idx_pool_kernel<<<cdiv(size_t(np) * D, 256), 256, 0, st>>>(k, gs, ape, out, p0, np, pool, D);
  UH_CHECK_LAUNCH();
}

void idx_scores(const bf16* q, const bf16* w, const bf16* pk, const int32_t* pos, float* out, int n, int npool,
                int H, int D, int pool, float scale, hipStream_t st) {
  if (n <= 0 || npool <= 0) return;
  const size_t lds = size_t(H * D + H) * 4;
  idx_scores_kernel<<<dim3(cdiv(size_t(npool), 128), n), 128, lds, st>>>(q, w, pk, pos, out, npool, H, D, pool,
                                                                         scale);
  UH_CHECK_LAUNCH();
}

void idx_expand(const int32_t* pools, const int32_t* pos, int32_t* idx, int n, int k, int pool, hipStream_t st) {
  const size_t total = size_t(n) * (k * pool + pool - 1);
  if (total == 0) return;
  idx_expand_kernel<<<cdiv(total, 256), 256, 0, st>>>(pools, pos, idx, n, k, pool);
  UH_CHECK_LAUNCH();
}

void mla_attn(const float* qa, const bf16* kv, const int32_t* idx, int w, const int32_t* pos, float* ol, int n,
              int H, int r, float scale, hipStream_t st) {
  if (r != 512) throw std::runtime_error("mla_attn: latent rank must be 512");
  if (n <= 0) return;
  mla_attn_kernel<512><<<n * H, 256, 0, st>>>(qa, kv, idx, w, pos, ol, H, scale);
  UH_CHECK_LAUNCH();
}

void quant_fp8_block(const bf16* w, uint8_t* q, float* s, int N, int K, hipStream_t st) {
  quant_fp8_block_kernel<<<dim3((K + 127) / 128, (N + 127) / 128), 256, 0, st>>>(w, q, s, N, K);
  UH_CHECK_LAUNCH();
}

void fake_quant_fp8_group(bf16* x, int ldx, int rows, int cols, int group, hipStream_t st) {
  if (group != 128 || cols % 128 != 0) throw std::runtime_error("fake_quant_fp8_group: groups of 128 only");
  const size_t waves = size_t(rows) * (cols / 128);
  if (waves == 0) return;
  fake_quant_fp8_group_kernel<<<cdiv(waves * 32, 256), 256, 0, st>>>(x, ldx, rows, cols);
  UH_CHECK_LAUNCH();
}

void route_sigmoid(const float* logits, const float* bias, int32_t* sel, float* weights, int T, int E, int K,
                   float scale, hipStream_t st) {
  if (K > 16) throw std::runtime_error("route_sigmoid: K > 16");
  if (T <= 0) return;
  route_sigmoid_kernel<<<cdiv(size_t(T), 64), 64, 0, st>>>(logits, bias, sel, weights, T, E, K, scale);
  UH_CHECK_LAUNCH();
}

void swiglu_hf(const bf16* gate, const bf16* up, bf16* h, int rows, int dim, float limit, hipStream_t st) {
  const size_t n = size_t(rows) * dim;
  if (n == 0) return;
  swiglu_hf_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(gate, up, h, n, limit);
  UH_CHECK_LAUNCH();
}

void moe_combine_hf(const bf16* y, const float* rw, const int32_t* pidx, const bf16* shared, bf16* out, int T,
                    int K, int dim, hipStream_t st) {
  if (T <= 0) return;
  moe_combine_hf_kernel<<<T, 256, 0, st>>>(y, rw, pidx, shared, out, K, dim);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
