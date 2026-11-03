#include "kernels/elem_fast.hpp"

#include <algorithm>
#include <stdexcept>

#include "kernels/dev.hip.hpp"
#include "kernels/elem_dev.hip.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int NT = 256;

// rmsnorm_kernel (ref_elem.hip) with blockIdx.y choosing the slice of the row written; the sum of
// squares uses the same thread partition and block_sum in every slice.
__global__ __launch_bounds__(NT) void rmsnorm_slices_kernel(const bf16* __restrict__ x, int ldx,
                                                            const bf16* __restrict__ w, bf16* __restrict__ y, int ldy,
                                                            int dim, float eps, int per) {
  __shared__ float red[32];
  const bf16* xr = x + size_t(blockIdx.x) * ldx;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float v = bf16_to_f32(xr[i]);
    ss += v * v;
  }
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / float(dim) + eps);
  bf16* yr = y + size_t(blockIdx.x) * ldy;
  const int d1 = min(dim, int(blockIdx.y + 1) * per);
  for (int i = blockIdx.y * per + threadIdx.x; i < d1; i += NT) {
    const float t = bf16_to_f32(xr[i]) * r;
    yr[i] = f32_to_bf16(bf16_to_f32(w[i]) * t);
  }
}

// hc_post_kernel (ref_elem.hip) over the columns of slice blockIdx.y.
__global__ __launch_bounds__(NT) void hc_post_slices_kernel(const bf16* x, const bf16* res, const float* post,
                                                            const float* comb, bf16* out, int hc, int dim, int per) {
  const int t = blockIdx.x;
  const int d1 = min(dim, int(blockIdx.y + 1) * per);
  for (int d = blockIdx.y * per + threadIdx.x; d < d1; d += NT)
    hc_post_column(bf16_to_f32(x[size_t(t) * dim + d]), res, post, comb, out, hc, dim, t, d);
}

// rmsnorm_slices_kernel, then act_quant_fp8 (128-blocks) of the slice's output values: per 128-block the
// same amax (exact, any order), scale exponent and codes. The slice is a multiple of 256 values: each
// thread takes one value per 256, waves 0-3 / 4-7 of a 256-chunk hold its two 128-blocks.
__global__ __launch_bounds__(NT) void rmsnorm_quant_slices_kernel(const bf16* __restrict__ x, int ldx,
                                                                  const bf16* __restrict__ w, bf16* __restrict__ y,
                                                                  int ldy, uint8_t* __restrict__ q,
                                                                  uint8_t* __restrict__ qs, int dim, float eps, int per) {
  __shared__ float red[32];
  __shared__ float bmax[8];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const bf16* xr = x + size_t(row) * ldx;
  float ss = 0.0f;
  for (int i = tid; i < dim; i += NT) {
    const float v = bf16_to_f32(xr[i]);
    ss += v * v;
  }
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / float(dim) + eps);
  bf16* yr = y + size_t(row) * ldy;
  for (int c0 = blockIdx.y * per; c0 < (blockIdx.y + 1) * per; c0 += NT) {
    const int d = c0 + tid;
    const float t = bf16_to_f32(xr[d]) * r;
    const bf16 yv = f32_to_bf16(bf16_to_f32(w[d]) * t);
    yr[d] = yv;
    const float v = bf16_to_f32(yv);
    const float a = warp_max(fabsf(v));
    if (lane == 0) bmax[wave] = a;
    __syncthreads();
    const int g = wave & ~3;
    const float m = fmaxf(fmaxf(fmaxf(bmax[g], bmax[g + 1]), fmaxf(bmax[g + 2], bmax[g + 3])), 1e-4f);
    __syncthreads();
    const uint8_t se = pow2_ceil_e8m0(m * (1.0f / 448.0f));
    const float sc = e8m0_to_f32(se);
    q[size_t(row) * dim + d] = f32_to_e4m3(fminf(fmaxf(v / sc, -448.0f), 448.0f));
    if ((tid & 127) == 0) qs[size_t(row) * (dim / 128) + d / 128] = se;
  }
}

// kv_chain_row (elem_dev.hip.hpp): a block per row.
__global__ __launch_bounds__(NT) void kv_chain_kernel(bf16* __restrict__ x, const bf16* __restrict__ w, int dim,
                                                      float eps, const int32_t* __restrict__ pos,
                                                      const float2* __restrict__ freqs, int npair) {
  __shared__ float red[32];
  kv_chain_row(x + size_t(blockIdx.x) * dim, w, dim, eps, pos[blockIdx.x], freqs, npair, red);
}

// head_rms_rope_head (elem_dev.hip.hpp) of one (row, head): block of 128 threads.
__global__ __launch_bounds__(128) void head_rms_rope_kernel(bf16* __restrict__ x, int heads, int dim, float eps,
                                                            const int32_t* __restrict__ pos,
                                                            const float2* __restrict__ freqs, int npair) {
  __shared__ float red[8];
  head_rms_rope_head(x + size_t(blockIdx.x) * dim, dim, eps, pos[blockIdx.x / heads], freqs, npair, red);
}

}  // namespace

int row_slices(int rows, int dim) { return std::clamp(64 / std::max(rows, 1), 1, std::max(1, dim / NT)); }

void rmsnorm_slices(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps,
                    hipStream_t st) {
  if (x == y) throw std::runtime_error("rmsnorm_slices: in place");
  const int s = row_slices(rows, dim);
  rmsnorm_slices_kernel<<<dim3(rows, s), NT, 0, st>>>(x, ldx, w, y, ldy, dim, eps, (dim + s - 1) / s);
  UH_CHECK_LAUNCH();
}

bool rmsnorm_quant_slices(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, uint8_t* q, uint8_t* qs, int rows,
                          int dim, float eps, hipStream_t st) {
  const int sl = row_slices(rows, dim);
  if (x == y || dim % sl || (dim / sl) % NT) return false;
  rmsnorm_quant_slices_kernel<<<dim3(rows, sl), NT, 0, st>>>(x, ldx, w, y, ldy, q, qs, dim, eps, dim / sl);
  UH_CHECK_LAUNCH();
  return true;
}

void kv_chain(bf16* kv, const bf16* w, int rows, int dim, float eps, const int32_t* pos, const float2* freqs, int npair,
              hipStream_t st) {
  if (dim > 2 * NT || (dim - 2 * npair) % 64 || (dim - 2 * npair) / 64 > NT / 32 || npair > NT)
    throw std::runtime_error("kv_chain: shape");
  kv_chain_kernel<<<rows, NT, 0, st>>>(kv, w, dim, eps, pos, freqs, npair);
  UH_CHECK_LAUNCH();
}

void head_rms_rope(bf16* x, int rows, int heads, int dim, float eps, const int32_t* pos, const float2* freqs, int npair,
                   hipStream_t st) {
  if (npair > 128) throw std::runtime_error("head_rms_rope: npair > 128");
  head_rms_rope_kernel<<<rows * heads, 128, 0, st>>>(x, heads, dim, eps, pos, freqs, npair);
  UH_CHECK_LAUNCH();
}

void hc_post_slices(const bf16* x, const bf16* res, const float* post, const float* comb, bf16* out, int T, int hc,
                    int dim, hipStream_t st) {
  if (hc > 4) throw std::runtime_error("hc_post_slices: hc > 4");
  const int s = row_slices(T, dim);
  hc_post_slices_kernel<<<dim3(T, s), NT, 0, st>>>(x, res, post, comb, out, hc, dim, (dim + s - 1) / s);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
