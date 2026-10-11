// Reference MoE kernels: router, SwiGLU, row gather / scatter-add.
#include "kernels/dev.hip.hpp"
#include "kernels/ref.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

// One thread per token (E <= 256, K <= 8).
__global__ void route_kernel(const float* logits, const float* bias, const int32_t* tid2eid, const int32_t* ids,
                             int32_t* sel, float* weights, int T, int E, int K, float route_scale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* lg = logits + size_t(t) * E;
  auto score = [&](int e) {
    const float x = lg[e];
    const float sp = x > 20.0f ? x : log1pf(expf(x));  // F.softplus(beta=1, threshold=20)
    return sqrtf(sp);
  };
  int chosen[8];
  if (tid2eid) {
    for (int k = 0; k < K; ++k) chosen[k] = tid2eid[size_t(ids[t]) * K + k];
  } else {
    // Top-K of score + bias, descending, ties -> lower expert id.
    for (int k = 0; k < K; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        bool used = false;
        for (int j = 0; j < k; ++j) used |= chosen[j] == e;
        if (used) continue;
        const float v = score(e) + bias[e];
        if (best < 0 || v > bv) {
          best = e;
          bv = v;
        }
      }
      chosen[k] = best;
    }
  }
  float w[8], sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    w[k] = score(chosen[k]);
    sum += w[k];
  }
  for (int k = 0; k < K; ++k) {
    sel[size_t(t) * K + k] = chosen[k];
    weights[size_t(t) * K + k] = (w[k] / sum) * route_scale;
  }
}

__global__ void swiglu_kernel(const bf16* gate, const bf16* up, const float* w, bf16* h, int rows, int dim,
                              float limit) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(rows) * dim) return;
  float g = bf16_to_f32(gate[gid]), u = bf16_to_f32(up[gid]);
  if (limit > 0.0f) {
    u = fminf(fmaxf(u, -limit), limit);
    g = fminf(g, limit);
  }
  float x = (g / (1.0f + expf(-g))) * u;
  if (w) x = w[gid / dim] * x;
  h[gid] = f32_to_bf16(x);
}

__global__ void scatter_add_kernel(const bf16* src, const int32_t* rows_idx, float* y, int n, int dim) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n) * dim) return;
  const int r = int(gid / dim), c = int(gid % dim);
  y[size_t(rows_idx[r]) * dim + c] += bf16_to_f32(src[gid]);
}

__global__ void gather_rows_kernel(const bf16* src, const int32_t* rows_idx, bf16* dst, int n, int dim) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n) * dim) return;
  const int r = int(gid / dim), c = int(gid % dim);
  dst[gid] = src[size_t(rows_idx[r]) * dim + c];
}

__global__ void moe_combine_kernel(const bf16* pair_out, const int32_t* pidx, const bf16* shared, bf16* out, int K,
                                   int dim) {
  const int t = blockIdx.x;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) acc += bf16_to_f32(pair_out[size_t(pidx[size_t(t) * K + j]) * dim + d]);
    acc += bf16_to_f32(shared[size_t(t) * dim + d]);
    out[size_t(t) * dim + d] = f32_to_bf16(acc);
  }
}

}  // namespace

void route(const float* logits, const float* bias, const int32_t* tid2eid, const int32_t* ids, int32_t* sel,
           float* weights, int T, int E, int K, float route_scale, hipStream_t st) {
  if (K > 8) throw std::runtime_error("route: K > 8");
  route_kernel<<<cdiv(T, 64), 64, 0, st>>>(logits, bias, tid2eid, ids, sel, weights, T, E, K, route_scale);
  UH_CHECK_LAUNCH();
}
void swiglu(const bf16* gate, const bf16* up, const float* w, bf16* h, int rows, int dim, float limit, hipStream_t st) {
  swiglu_kernel<<<cdiv(size_t(rows) * dim, 256), 256, 0, st>>>(gate, up, w, h, rows, dim, limit);
  UH_CHECK_LAUNCH();
}
void scatter_add(const bf16* src, const int32_t* rows_idx, float* y, int n, int dim, hipStream_t st) {
  if (n <= 0) return;
  scatter_add_kernel<<<cdiv(size_t(n) * dim, 256), 256, 0, st>>>(src, rows_idx, y, n, dim);
  UH_CHECK_LAUNCH();
}
void gather_rows(const bf16* src, const int32_t* rows_idx, bf16* dst, int n, int dim, hipStream_t st) {
  if (n <= 0) return;
  gather_rows_kernel<<<cdiv(size_t(n) * dim, 256), 256, 0, st>>>(src, rows_idx, dst, n, dim);
  UH_CHECK_LAUNCH();
}

void moe_combine(const bf16* pair_out, const int32_t* pidx, const bf16* shared, bf16* out, int T, int K, int dim,
                 hipStream_t st) {
  moe_combine_kernel<<<T, 256, 0, st>>>(pair_out, pidx, shared, out, K, dim);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels
