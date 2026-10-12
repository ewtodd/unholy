// Reference elementwise kernels: quantization, norms, rotary, Hadamard, Hyper-Connections.
#include <algorithm>

#include "kernels/dev.hip.hpp"
#include "kernels/ref.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

// ---- quantization ----
// One wave per (row, block). block in {32, 64, 128}: each lane holds block/32 values.
__global__ void act_quant_fp8_kernel(const bf16* x, int ldx, uint8_t* q, uint8_t* s, int rows, int cols, int block,
                                     bool fake) {
  const int nb = cols / block;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= rows * nb) return;
  const int r = wid / nb, b = wid % nb;
  const int per = block / 32;
  bf16* xr = const_cast<bf16*>(x) + size_t(r) * ldx + size_t(b) * block;
  float v[4];
  float amax = 0.0f;
  for (int i = 0; i < per; ++i) {
    v[i] = bf16_to_f32(xr[lane * per + i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = fmaxf(warp_max(amax), 1e-4f);
  const uint8_t se = pow2_ceil_e8m0(amax * (1.0f / 448.0f));
  const float sc = e8m0_to_f32(se);
  for (int i = 0; i < per; ++i) {
    const uint8_t qv = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.0f), 448.0f));
    if (fake) xr[lane * per + i] = f32_to_bf16(e4m3_to_f32(qv) * sc);
    else q[size_t(r) * cols + size_t(b) * block + lane * per + i] = qv;
  }
  if (!fake && lane == 0) s[size_t(r) * nb + b] = se;
}

__global__ void fake_quant_fp4_kernel(bf16* x, int ldx, int rows, int cols, int block) {
  const int nb = cols / block;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= rows * nb) return;
  const int r = wid / nb, b = wid % nb;
  const int per = block / 32;
  bf16* xr = x + size_t(r) * ldx + size_t(b) * block;
  float v[4];
  float amax = 0.0f;
  for (int i = 0; i < per; ++i) {
    v[i] = bf16_to_f32(xr[lane * per + i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = fmaxf(warp_max(amax), 6.0f * 1.1754943508222875e-38f);  // 6 * 2^-126
  const float sc = e8m0_to_f32(pow2_ceil_e8m0(amax * (1.0f / 6.0f)));
  for (int i = 0; i < per; ++i) xr[lane * per + i] = f32_to_bf16(round_e2m1(fminf(fmaxf(v[i] / sc, -6.0f), 6.0f)) * sc);
}

// ---- norms ----
__global__ void rmsnorm_kernel(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int dim, float eps) {
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
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float t = bf16_to_f32(xr[i]) * r;
    yr[i] = f32_to_bf16(bf16_to_f32(w[i]) * t);
  }
}

// q *= rsqrt(q.square().mean(-1) + eps) on a bf16 tensor: every intermediate is a bf16 tensor.
__global__ void head_rms_bf16_kernel(bf16* x, int dim, float eps) {
  __shared__ float red[32];
  bf16* xr = x + size_t(blockIdx.x) * dim;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf16_to_f32(xr[i]);
    ss += round_bf16(v * v);
  }
  ss = block_sum(ss, red);
  const float mean = round_bf16(ss / float(dim));
  const float r = round_bf16(rsqrtf(round_bf16(mean + eps)));
  for (int i = threadIdx.x; i < dim; i += blockDim.x) xr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * r);
}

// ---- rotary ----
__global__ void rope_kernel(bf16* x, int ld_row, int heads, int head_stride, int off, int npair, const int32_t* pos,
                            const float2* freqs, bool inverse) {
  const int row = blockIdx.x;
  const int p = pos[row];
  for (int i = threadIdx.x; i < heads * npair; i += blockDim.x) {
    const int h = i / npair, j = i % npair;
    bf16* e = x + size_t(row) * ld_row + size_t(h) * head_stride + off + 2 * j;
    const float a = bf16_to_f32(e[0]), b = bf16_to_f32(e[1]);
    const float2 f = freqs[size_t(p) * npair + j];
    const float c = f.x, s = inverse ? -f.y : f.y;
    e[0] = f32_to_bf16(a * c - b * s);
    e[1] = f32_to_bf16(a * s + b * c);
  }
}

// ---- Hadamard ----
__global__ void hadamard128_kernel(bf16* x, size_t nseg) {
  __shared__ float buf[4][128];
  const int local = threadIdx.x / 32, lane = threadIdx.x & 31;
  const size_t seg = size_t(blockIdx.x) * 4 + local;
  const bool valid = seg < nseg;  // no early return: every wave takes the block barriers
  bf16* p = x + seg * 128;
  float* b = buf[local];
  if (valid)
    for (int i = lane; i < 128; i += 32) b[i] = bf16_to_f32(p[i]);
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {
    if (valid)
      for (int t = lane; t < 64; t += 32) {
        const int i = (t / h) * 2 * h + (t % h);
        const float a = b[i], c = b[i + h];
        b[i] = a + c;
        b[i + h] = a - c;
      }
    __syncthreads();
  }
  const float sc = 0.08838834764831845f;  // 128^-0.5
  if (valid)
    for (int i = lane; i < 128; i += 32) p[i] = f32_to_bf16(b[i] * sc);
}

// ---- Hyper-Connections ----
// One block (256 threads) per token. mix = (2 + hc) * hc <= 24 for hc = 4.
template <bool kHead>
__global__ void hc_mix_kernel(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y,
                              float* post_out, float* comb_out, int hc, int dim, int iters, float norm_eps,
                              float hc_eps) {
  constexpr int kMaxMix = 24;
  __shared__ float red[32];
  __shared__ float part[kMaxMix][8];
  __shared__ float mixes[kMaxMix];
  __shared__ float pre[4], post[4], comb[4][4];
  const int mix = kHead ? hc : (2 + hc) * hc;
  const int width = hc * dim;
  const bf16* xr = x + size_t(blockIdx.x) * width;
  float ss = 0.0f;
  float acc[kMaxMix];
  for (int j = 0; j < kMaxMix; ++j) acc[j] = 0.0f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    const float v = bf16_to_f32(xr[k]);
    ss += v * v;
    for (int j = 0; j < mix; ++j) acc[j] = fmaf(v, fn[size_t(j) * width + k], acc[j]);
  }
  ss = block_sum(ss, red);
  const float rs = rsqrtf(ss / float(width) + norm_eps);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int j = 0; j < mix; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) part[j][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x < mix) {
    float v = 0.0f;
    for (int w = 0; w < int(blockDim.x / 32); ++w) v += part[threadIdx.x][w];
    mixes[threadIdx.x] = v * rs;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (kHead) {
      for (int j = 0; j < hc; ++j) pre[j] = 1.0f / (1.0f + expf(-(mixes[j] * scale[0] + base[j]))) + hc_eps;
    } else {
      for (int j = 0; j < hc; ++j) pre[j] = 1.0f / (1.0f + expf(-(mixes[j] * scale[0] + base[j]))) + hc_eps;
      for (int j = 0; j < hc; ++j) post[j] = 2.0f / (1.0f + expf(-(mixes[hc + j] * scale[1] + base[hc + j])));
      for (int j = 0; j < hc; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < hc; ++k) {
          comb[j][k] = mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
          mx = fmaxf(mx, comb[j][k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < hc; ++k) {
          comb[j][k] = expf(comb[j][k] - mx);
          sum += comb[j][k];
        }
        for (int k = 0; k < hc; ++k) comb[j][k] = comb[j][k] / sum + hc_eps;
      }
      auto col_norm = [&]() {
        for (int k = 0; k < hc; ++k) {
          float cs = 0.0f;
          for (int j = 0; j < hc; ++j) cs += comb[j][k];
          for (int j = 0; j < hc; ++j) comb[j][k] = comb[j][k] / (cs + hc_eps);
        }
      };
      col_norm();
      for (int it = 0; it < iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) {
          float rsum = 0.0f;
          for (int k = 0; k < hc; ++k) rsum += comb[j][k];
          for (int k = 0; k < hc; ++k) comb[j][k] = comb[j][k] / (rsum + hc_eps);
        }
        col_norm();
      }
      for (int j = 0; j < hc; ++j) post_out[size_t(blockIdx.x) * hc + j] = post[j];
      for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb_out[(size_t(blockIdx.x) * hc + j) * hc + k] = comb[j][k];
    }
  }
  __syncthreads();
  bf16* yr = y + size_t(blockIdx.x) * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float v = 0.0f;
    for (int i = 0; i < hc; ++i) v += pre[i] * bf16_to_f32(xr[size_t(i) * dim + d]);
    yr[d] = f32_to_bf16(v);
  }
}

__global__ void hc_post_kernel(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out,
                               int hc, int dim) {
  const int t = blockIdx.x;
  const float* pt = post + size_t(t) * hc;
  const float* ct = comb + size_t(t) * hc * hc;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float xv = bf16_to_f32(x[size_t(t) * dim + d]);
    float r[4];
    for (int i = 0; i < hc; ++i) r[i] = bf16_to_f32(res[(size_t(t) * hc + i) * dim + d]);
    for (int j = 0; j < hc; ++j) {
      float s = 0.0f;
      for (int i = 0; i < hc; ++i) s += ct[i * hc + j] * r[i];
      out[(size_t(t) * hc + j) * dim + d] = f32_to_bf16(pt[j] * xv + s);
    }
  }
}

__global__ void embed_hc_kernel(const int32_t* ids, const bf16* table, bf16* out, int dim, int hc) {
  const int t = blockIdx.x;
  const bf16* src = table + size_t(ids[t]) * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const bf16 v = src[d];
    for (int h = 0; h < hc; ++h) out[(size_t(t) * hc + h) * dim + d] = v;
  }
}

__global__ void bf16_to_f32_kernel(const bf16* x, float* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = bf16_to_f32(x[i]);
}
__global__ void f32_to_bf16_kernel(const float* x, bf16* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = f32_to_bf16(x[i]);
}
__global__ void scale_bf16_kernel(bf16* x, float c, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = f32_to_bf16(bf16_to_f32(x[i]) * c);
}
__global__ void positions_kernel(int32_t* out, int n, int start, int step) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = start + i * step;
}
__global__ void add_bf16_to_f32_kernel(const bf16* x, float* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] += bf16_to_f32(x[i]);
}

__global__ void copy16_kernel(uint4* dst, const uint4* src, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
  __threadfence_system();
}
__global__ void copy1_kernel(uint8_t* dst, const uint8_t* src, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
  __threadfence_system();
}

}  // namespace

void act_quant_fp8(const bf16* x, int ldx, uint8_t* q, uint8_t* s, int rows, int cols, int block, hipStream_t st) {
  const size_t waves = size_t(rows) * (cols / block);
  act_quant_fp8_kernel<<<cdiv(waves * 32, 256), 256, 0, st>>>(x, ldx, q, s, rows, cols, block, false);
  UH_CHECK_LAUNCH();
}
void fake_quant_fp8(bf16* x, int ldx, int rows, int cols, int block, hipStream_t st) {
  const size_t waves = size_t(rows) * (cols / block);
  act_quant_fp8_kernel<<<cdiv(waves * 32, 256), 256, 0, st>>>(x, ldx, nullptr, nullptr, rows, cols, block, true);
  UH_CHECK_LAUNCH();
}
void fake_quant_fp4(bf16* x, int ldx, int rows, int cols, int block, hipStream_t st) {
  const size_t waves = size_t(rows) * (cols / block);
  fake_quant_fp4_kernel<<<cdiv(waves * 32, 256), 256, 0, st>>>(x, ldx, rows, cols, block);
  UH_CHECK_LAUNCH();
}
void rmsnorm(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps, hipStream_t st) {
  rmsnorm_kernel<<<rows, 256, 0, st>>>(x, ldx, w, y, ldy, dim, eps);
  UH_CHECK_LAUNCH();
}
void head_rms_bf16(bf16* x, int rows, int heads, int dim, float eps, hipStream_t st) {
  head_rms_bf16_kernel<<<rows * heads, 128, 0, st>>>(x, dim, eps);
  UH_CHECK_LAUNCH();
}
void rope(bf16* x, int ld_row, int rows, int heads, int head_stride, int off, int npair, const int32_t* pos,
          const float2* freqs, bool inverse, hipStream_t st) {
  rope_kernel<<<rows, 256, 0, st>>>(x, ld_row, heads, head_stride, off, npair, pos, freqs, inverse);
  UH_CHECK_LAUNCH();
}
void hadamard128(bf16* x, size_t nseg, hipStream_t st) {
  hadamard128_kernel<<<cdiv(nseg, 4), 128, 0, st>>>(x, nseg);
  UH_CHECK_LAUNCH();
}
void hc_pre(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post, float* comb,
            int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st) {
  if (hc != 4) throw std::runtime_error("hc_pre: hc must be 4");
  hc_mix_kernel<false><<<T, 256, 0, st>>>(x, fn, scale, base, y, post, comb, hc, dim, iters, norm_eps, hc_eps);
  UH_CHECK_LAUNCH();
}
void hc_post(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc, int dim,
             hipStream_t st) {
  hc_post_kernel<<<T, 256, 0, st>>>(x, res, post, comb, out, hc, dim);
  UH_CHECK_LAUNCH();
}
void hc_head(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, int T, int hc, int dim,
             float norm_eps, float hc_eps, hipStream_t st) {
  hc_mix_kernel<true><<<T, 256, 0, st>>>(x, fn, scale, base, y, nullptr, nullptr, hc, dim, 0, norm_eps, hc_eps);
  UH_CHECK_LAUNCH();
}
void embed_hc(const int32_t* ids, const bf16* table, bf16* out, int T, int dim, int hc, hipStream_t st) {
  embed_hc_kernel<<<T, 256, 0, st>>>(ids, table, out, dim, hc);
  UH_CHECK_LAUNCH();
}
void cast_bf16_f32(const bf16* x, float* y, size_t n, hipStream_t st) {
  bf16_to_f32_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}
void cast_f32_bf16(const float* x, bf16* y, size_t n, hipStream_t st) {
  f32_to_bf16_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}
void add_bf16_to_f32(const bf16* x, float* y, size_t n, hipStream_t st) {
  add_bf16_to_f32_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(x, y, n);
  UH_CHECK_LAUNCH();
}

void scale_bf16(bf16* x, float c, size_t n, hipStream_t st) {
  scale_bf16_kernel<<<std::min(cdiv(n, 256), 65535), 256, 0, st>>>(x, c, n);
  UH_CHECK_LAUNCH();
}

void positions(int32_t* out, int n, int start, int step, hipStream_t st) {
  if (n <= 0) return;
  positions_kernel<<<(n + 255) / 256, 256, 0, st>>>(out, n, start, step);
  UH_CHECK_LAUNCH();
}

void copy_kernel(void* dst, const void* src, size_t bytes, hipStream_t st) {
  if (bytes == 0) return;
  const bool v16 = (reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) | bytes) % 16 == 0;
  const size_t n = v16 ? bytes / 16 : bytes;
  const int blocks = int(std::min<size_t>((n + 255) / 256, 1024));
  if (v16) copy16_kernel<<<blocks, 256, 0, st>>>(static_cast<uint4*>(dst), static_cast<const uint4*>(src), n);
  else copy1_kernel<<<blocks, 256, 0, st>>>(static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src), n);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
