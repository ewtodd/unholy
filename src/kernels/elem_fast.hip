#include "kernels/elem_fast.hpp"

#include <algorithm>
#include <stdexcept>

#include "kernels/dev.hip.hpp"
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
  const float* pt = post + size_t(t) * hc;
  const float* ct = comb + size_t(t) * hc * hc;
  const int d1 = min(dim, int(blockIdx.y + 1) * per);
  for (int d = blockIdx.y * per + threadIdx.x; d < d1; d += NT) {
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

}  // namespace

int row_slices(int rows, int dim) { return std::clamp(64 / std::max(rows, 1), 1, std::max(1, dim / NT)); }

void rmsnorm_slices(const bf16* x, int ldx, const bf16* w, bf16* y, int ldy, int rows, int dim, float eps,
                    hipStream_t st) {
  if (x == y) throw std::runtime_error("rmsnorm_slices: in place");
  const int s = row_slices(rows, dim);
  rmsnorm_slices_kernel<<<dim3(rows, s), NT, 0, st>>>(x, ldx, w, y, ldy, dim, eps, (dim + s - 1) / s);
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
