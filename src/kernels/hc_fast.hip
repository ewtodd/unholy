// Hyper-Connection pre-mixing, tiled over tokens (fast path; reference: hc_mix_kernel in ref_elem.hip).
//
// The reference kernel reads the [24, 4*4096] fp32 mixing matrix once per token (1.5 MB): at 1024
// tokens that is 1.5 GB of L2 traffic per call. Here a block takes 16 tokens; each K-chunk of the
// matrix is staged in LDS once and used by all 16. Thread layout: token tid/16, K lanes tid%16.
#include "kernels/dev.hip.hpp"
#include "kernels/hc_fast.hpp"
#include "kernels/elem_fast.hpp"
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
                                                          float hc_eps,
                                                          uint64_t want_fn, HcCanaryLog* log) {
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
    for (int i = tid; i < MIX * KC; i += NT) {
      const size_t off = size_t(i / KC) * width + k0 + i % KC;
      const float* p = fn + off;
      if (log != nullptr) {
        uint64_t got = reinterpret_cast<uint64_t>(p);
        asm volatile("" : "+v"(got));  // the compiler may not fold the check into one on the arguments
        const uint64_t want = want_fn + off * 4;
        if (got != want) {
          const unsigned n = __hip_atomic_fetch_add(&log->count, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          if (n < HcCanaryLog::kMax) {
            log->ev[n].got = got;
            log->ev[n].want = want;
            log->ev[n].hw_id1 = __builtin_amdgcn_s_getreg((31 << 11) | 23);  // HW_REG_HW_ID1
            log->ev[n].where = (unsigned(k0) << 16) | unsigned(i);
          }
          got = want;
        }
        p = reinterpret_cast<const float*>(got);
      }
      sFn[i / KC][i % KC] = *p;
    }
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

__device__ __forceinline__ void hc_partial_body(const bf16* __restrict__ x, const float* __restrict__ fn,
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

__global__ __launch_bounds__(NT) void hc_partial_kernel(const bf16* __restrict__ x, const float* __restrict__ fn,
                                                        float* __restrict__ part, int width) {
  hc_partial_body(x, fn, part, width);
}

// One block per (token, slice of dim): the partial sums, then Sinkhorn on wave 0 (lane l mod 16 owns
// comb[l / 4][l % 4]; lanes 0-3 compute pre, 4-7 post), then the mix of the slice's columns. Every value
// goes through the same operations in the same order as one thread doing it all (sums in index order),
// so the slices and the lanes change nothing; every slice repeats the (tiny) Sinkhorn.
// kCoherent: `part` was written by other blocks of the same launch (read past the caches).
// kNorm (hc_finish_norm_kernel): the whole row is mixed for the sum of squares (rmsnorm_slices' thread partition),
// the slice is written, normalized with nw and, with q, quantized as rmsnorm_quant_slices does.
template <bool kCoherent, bool kNorm = false>
__device__ __forceinline__ void hc_finish_token(const bf16* __restrict__ x, const float* __restrict__ part,
                                                const float* __restrict__ scale, const float* __restrict__ base,
                                                bf16* __restrict__ y, float* __restrict__ post_out,
                                                float* __restrict__ comb_out, int hc, int dim, int iters, float norm_eps,
                                                float hc_eps, int per, int t, int slice, const bf16* nw = nullptr,
                                                bf16* yn = nullptr, float eps2 = 0.0f, uint8_t* q = nullptr,
                                                uint8_t* qs = nullptr) {
  __shared__ float m[MIX + 1];
  __shared__ float pre[4];
  __shared__ float sp_part[SPLITS * (MIX + 1)];  // the token's partial sums, loaded at once (summed in split order)
  const int tid = threadIdx.x, width = hc * dim;
  for (int i = tid; i < SPLITS * (MIX + 1); i += NT) {
    const float* p = &part[size_t(t) * SPLITS * (MIX + 1) + i];
    if constexpr (kCoherent) sp_part[i] = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    else sp_part[i] = *p;
  }
  __syncthreads();
  if (tid <= MIX) {
    float v = 0.0f;
    for (int sp = 0; sp < SPLITS; ++sp) v += sp_part[sp * (MIX + 1) + tid];
    m[tid] = v;
  }
  __syncthreads();
  if (tid < 32) {
    constexpr int HC = 4;  // checked by the launcher
    const float rs = rsqrtf(m[MIX] / float(width) + norm_eps);
    const int l = tid & 15, row0 = tid & ~3, col0 = (tid & 16) | (l & 3);
    if (tid < HC) {
      const float mx = m[tid] * rs;
      pre[tid] = 1.0f / (1.0f + expf(-(mx * scale[0] + base[tid]))) + hc_eps;
    }
    float post = 0.0f;
    if (tid >= HC && tid < 2 * HC) {
      const float mx = m[tid] * rs;
      post = 2.0f / (1.0f + expf(-(mx * scale[1] + base[tid])));
    }
    const float mx = m[2 * HC + l] * rs;
    float c = mx * scale[2] + base[2 * HC + l];
    const auto row = [&](float v, int k) { return __shfl(v, row0 + k, 32); };
    const auto col = [&](float v, int j) { return __shfl(v, col0 + 4 * j, 32); };
    {  // softmax over the row
      float mm = -INFINITY;
      _Pragma("unroll") for (int k = 0; k < HC; ++k) mm = fmaxf(mm, row(c, k));
      const float e = expf(c - mm);
      float sum = 0.0f;
      _Pragma("unroll") for (int k = 0; k < HC; ++k) sum += row(e, k);
      c = e / sum + hc_eps;
    }
    const auto col_norm = [&]() {
      float cs = 0.0f;
      _Pragma("unroll") for (int j = 0; j < HC; ++j) cs += col(c, j);
      c = c / (cs + hc_eps);
    };
    col_norm();
    for (int it = 0; it < iters - 1; ++it) {
      float rsum = 0.0f;
      _Pragma("unroll") for (int k = 0; k < HC; ++k) rsum += row(c, k);
      c = c / (rsum + hc_eps);
      col_norm();
    }
    if (slice == 0) {
      if (tid < 16) comb_out[size_t(t) * HC * HC + l] = c;
      if (tid >= HC && tid < 2 * HC) post_out[size_t(t) * HC + tid - HC] = post;
    }
  }
  __syncthreads();
  const bf16* xt = x + size_t(t) * width;
  const int d0 = slice * per, d1 = min(dim, int(slice + 1) * per);
  if constexpr (!kNorm) {
    for (int d = d0 + tid; d < d1; d += NT) {
      float v = 0.0f;
      for (int h = 0; h < hc; ++h) v += pre[h] * bf16_to_f32(xt[size_t(h) * dim + d]);
      y[size_t(t) * dim + d] = f32_to_bf16(v);
    }
  } else {
    __shared__ float red[32];
    float ss = 0.0f;
    for (int d = tid; d < dim; d += NT) {
      float v = 0.0f;
      for (int h = 0; h < hc; ++h) v += pre[h] * bf16_to_f32(xt[size_t(h) * dim + d]);
      const bf16 yb = f32_to_bf16(v);
      if (d >= d0 && d < d1) y[size_t(t) * dim + d] = yb;
      const float yv = bf16_to_f32(yb);
      ss += yv * yv;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / float(dim) + eps2);
    const bf16* yr = y + size_t(t) * dim;  // this thread's own slice values
    if (q) {  // rmsnorm_quant_slices_kernel's chunks of the slice
      __shared__ float bmax[8];
      const int lane = tid & 31, wave = tid >> 5;
      for (int c0 = d0; c0 < d1; c0 += NT) {
        const int d = c0 + tid;
        const float tv = bf16_to_f32(yr[d]) * r;
        const bf16 yv = f32_to_bf16(bf16_to_f32(nw[d]) * tv);
        yn[size_t(t) * dim + d] = yv;
        const float v = bf16_to_f32(yv);
        const float a = warp_max(fabsf(v));
        if (lane == 0) bmax[wave] = a;
        __syncthreads();
        const int g = wave & ~3;
        const float m = fmaxf(fmaxf(fmaxf(bmax[g], bmax[g + 1]), fmaxf(bmax[g + 2], bmax[g + 3])), 1e-4f);
        __syncthreads();
        const uint8_t se = pow2_ceil_e8m0(m * (1.0f / 448.0f));
        const float sc = e8m0_to_f32(se);
        q[size_t(t) * dim + d] = f32_to_e4m3(fminf(fmaxf(v / sc, -448.0f), 448.0f));
        if ((tid & 127) == 0) qs[size_t(t) * (dim / 128) + d / 128] = se;
      }
    } else {
      for (int d = d0 + tid; d < d1; d += NT)
        yn[size_t(t) * dim + d] = f32_to_bf16(bf16_to_f32(nw[d]) * (bf16_to_f32(yr[d]) * r));
    }
  }
}

__global__ __launch_bounds__(NT) void hc_finish_norm_kernel(const bf16* __restrict__ x, const float* __restrict__ part,
                                                            const float* __restrict__ scale,
                                                            const float* __restrict__ base, bf16* __restrict__ y,
                                                            float* __restrict__ post_out, float* __restrict__ comb_out,
                                                            int hc, int dim, int iters, float norm_eps, float hc_eps,
                                                            int per, const bf16* __restrict__ nw, bf16* __restrict__ yn,
                                                            float eps2, uint8_t* __restrict__ q,
                                                            uint8_t* __restrict__ qs) {
  hc_finish_token<false, true>(x, part, scale, base, y, post_out, comb_out, hc, dim, iters, norm_eps, hc_eps, per,
                               blockIdx.x, blockIdx.y, nw, yn, eps2, q, qs);
}

__global__ __launch_bounds__(NT) void hc_finish_kernel(const bf16* __restrict__ x, const float* __restrict__ part,
                                                       const float* __restrict__ scale, const float* __restrict__ base,
                                                       bf16* __restrict__ y, float* __restrict__ post_out,
                                                       float* __restrict__ comb_out, int hc, int dim, int iters,
                                                       float norm_eps, float hc_eps, int per) {
  hc_finish_token<false>(x, part, scale, base, y, post_out, comb_out, hc, dim, iters, norm_eps, hc_eps, per, blockIdx.x,
                         blockIdx.y);
}

// hc_partial_kernel, then the last of the token's SPLITS blocks to finish runs hc_finish for the whole row (one
// slice) and, with norm_w, rmsnorm_slices_kernel's arithmetic on that row (same thread partition, so the same values).
__global__ __launch_bounds__(NT) void hc_pre_fused_kernel(const bf16* __restrict__ x, const float* __restrict__ fn,
                                                          float* __restrict__ part, const float* __restrict__ scale,
                                                          const float* __restrict__ base, bf16* __restrict__ y,
                                                          float* __restrict__ post_out, float* __restrict__ comb_out,
                                                          int hc, int dim, int iters, float norm_eps, float hc_eps,
                                                          uint32_t* __restrict__ counters, const bf16* __restrict__ nw,
                                                          bf16* __restrict__ yn, float eps2, uint8_t* __restrict__ q,
                                                          uint8_t* __restrict__ qs) {
  hc_partial_body(x, fn, part, hc * dim);
  __shared__ bool last;
  const int t = blockIdx.x, tid = threadIdx.x;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&counters[t], 1u) == SPLITS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  hc_finish_token<true>(x, part, scale, base, y, post_out, comb_out, hc, dim, iters, norm_eps, hc_eps, dim, t, 0);
  if (nw) {
    __shared__ float red[32];
    __syncthreads();  // this block's y
    const bf16* yr = y + size_t(t) * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += NT) {
      const float v = bf16_to_f32(yr[i]);
      ss += v * v;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / float(dim) + eps2);
    if (q) {  // rmsnorm_quant_slices_kernel's chunks
      __shared__ float bmax[8];
      const int lane = tid & 31, wave = tid >> 5;
      for (int c0 = 0; c0 < dim; c0 += NT) {
        const int d = c0 + tid;
        const float tv = bf16_to_f32(yr[d]) * r;
        const bf16 yv = f32_to_bf16(bf16_to_f32(nw[d]) * tv);
        yn[size_t(t) * dim + d] = yv;
        const float v = bf16_to_f32(yv);
        const float a = warp_max(fabsf(v));
        if (lane == 0) bmax[wave] = a;
        __syncthreads();
        const int g = wave & ~3;
        const float m = fmaxf(fmaxf(fmaxf(bmax[g], bmax[g + 1]), fmaxf(bmax[g + 2], bmax[g + 3])), 1e-4f);
        __syncthreads();
        const uint8_t se = pow2_ceil_e8m0(m * (1.0f / 448.0f));
        const float sc = e8m0_to_f32(se);
        q[size_t(t) * dim + d] = f32_to_e4m3(fminf(fmaxf(v / sc, -448.0f), 448.0f));
        if ((tid & 127) == 0) qs[size_t(t) * (dim / 128) + d / 128] = se;
      }
    } else {
      for (int i = tid; i < dim; i += NT)
        yn[size_t(t) * dim + i] = f32_to_bf16(bf16_to_f32(nw[i]) * (bf16_to_f32(yr[i]) * r));
    }
  }
  if (tid == 0) counters[t] = 0;
}

}  // namespace

HcCanaryLog* g_canary = nullptr;

void hc_set_canary(HcCanaryLog* log) { g_canary = log; }

void hc_pre_tiled(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, int T, int hc, int dim, int iters, float norm_eps, float hc_eps, hipStream_t st) {
  if (hc != 4 || (hc * dim) % KC) throw std::runtime_error("hc_pre_tiled: unsupported shape");
  hc_pre_tiled_kernel<<<(T + TB - 1) / TB, NT, 0, st>>>(x, fn, scale, base, y, post, comb, T, hc, dim, iters,
                                                         norm_eps, hc_eps, reinterpret_cast<uint64_t>(fn), g_canary);
  UH_CHECK_LAUNCH();
}

bool hc_pre_small(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post,
                  float* comb, float* part, int T, int hc, int dim, int iters, float norm_eps, float hc_eps,
                  hipStream_t st, uint32_t* counters, const bf16* norm_w, bf16* y_norm, float norm_eps2, uint8_t* q,
                  uint8_t* qs) {
  if (hc != 4 || (hc * dim) % (SPLITS * NT)) throw std::runtime_error("hc_pre_small: unsupported shape");
  if (q && dim % NT) throw std::runtime_error("hc_pre_small: quantized output needs dim % 256 == 0");
  if (counters) {
    hc_pre_fused_kernel<<<dim3(T, SPLITS), NT, 0, st>>>(x, fn, part, scale, base, y, post, comb, hc, dim, iters,
                                                         norm_eps, hc_eps, counters, norm_w, y_norm, norm_eps2, q, qs);
    UH_CHECK_LAUNCH();
    return true;
  }
  hc_partial_kernel<<<dim3(T, SPLITS), NT, 0, st>>>(x, fn, part, hc * dim);
  UH_CHECK_LAUNCH();
  const int sl = row_slices(T, dim);
  if (norm_w && !(q && (dim % sl || (dim / sl) % NT))) {  // the norm in the finish's slices (each mixes the row)
    hc_finish_norm_kernel<<<dim3(T, sl), NT, 0, st>>>(x, part, scale, base, y, post, comb, hc, dim, iters, norm_eps,
                                                      hc_eps, (dim + sl - 1) / sl, norm_w, y_norm, norm_eps2, q, qs);
    UH_CHECK_LAUNCH();
    return true;
  }
  hc_finish_kernel<<<dim3(T, sl), NT, 0, st>>>(x, part, scale, base, y, post, comb, hc, dim, iters, norm_eps, hc_eps,
                                               (dim + sl - 1) / sl);
  UH_CHECK_LAUNCH();
  return norm_w == nullptr;
}

}  // namespace unholy::kernels
