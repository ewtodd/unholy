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
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    out[size_t(blockIdx.x) * D + i] = f32_to_bf16(bf16_to_f32(w[i]) * (bf16_to_f32(xr[i]) * r));
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
  vit_rmsnorm_kernel<<<M, 256, 0, st>>>(x, D, w, eps, out);
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
