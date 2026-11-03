// Device bodies of elem_fast's row kernels, shared with fused launches (indexer.hip's attn_row_chains) so that both
// run the same code.
#pragma once

#include "kernels/dev.hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

// block_sum's operations for the 128 threads of group threadIdx.x / 128 (in a 128-thread block: block_sum). red: 8
// floats.
__device__ __forceinline__ float group128_sum(float v, float* red) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < 4 ? red[(warp & ~3) + lane] : 0.0f;
  return warp_sum(v);
}

// head_rms_bf16_kernel then rope_kernel on one head's row xr (position p) by the 128 threads of its group.
__device__ __forceinline__ void head_rms_rope_head(bf16* __restrict__ xr, int dim, float eps, int p,
                                                   const float2* __restrict__ freqs, int npair, float* red) {
  const int t = threadIdx.x & 127;
  float ss = 0.0f;
  for (int i = t; i < dim; i += 128) {
    const float v = bf16_to_f32(xr[i]);
    ss += round_bf16(v * v);
  }
  ss = group128_sum(ss, red);
  const float mean = round_bf16(ss / float(dim));
  const float r = round_bf16(rsqrtf(round_bf16(mean + eps)));
  for (int i = t; i < dim; i += 128) xr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * r);
  __syncthreads();
  if (t < npair) {
    bf16* e = xr + dim - 2 * npair + 2 * t;
    const float a = bf16_to_f32(e[0]), b = bf16_to_f32(e[1]);
    const float2 f = freqs[size_t(p) * npair + t];
    const float c = f.x, s = f.y;
    e[0] = f32_to_bf16(a * c - b * s);
    e[1] = f32_to_bf16(a * s + b * c);
  }
}

// The kv row chain (kv_norm in place, rope on the last 2 * npair dims, fp8 fake quant of the others in 64-blocks):
// rmsnorm_kernel, rope_kernel and act_quant_fp8_kernel (fake, block 64) of row xr (position p) in a block of 256
// threads, each with its own thread partition and arithmetic. red: 32 floats.
__device__ __forceinline__ void kv_chain_row(bf16* __restrict__ xr, const bf16* __restrict__ w, int dim, float eps,
                                             int p, const float2* __restrict__ freqs, int npair, float* red) {
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  float ss = 0.0f;
  for (int i = tid; i < dim; i += 256) {
    const float v = bf16_to_f32(xr[i]);
    ss += v * v;
  }
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / float(dim) + eps);
  for (int i = tid; i < dim; i += 256) {
    const float t = bf16_to_f32(xr[i]) * r;
    xr[i] = f32_to_bf16(bf16_to_f32(w[i]) * t);
  }
  __syncthreads();
  const int off = dim - 2 * npair;
  if (tid < npair) {  // rope_kernel, one head
    bf16* e = xr + off + 2 * tid;
    const float a = bf16_to_f32(e[0]), b = bf16_to_f32(e[1]);
    const float2 f = freqs[size_t(p) * npair + tid];
    const float c = f.x, s = f.y;
    e[0] = f32_to_bf16(a * c - b * s);
    e[1] = f32_to_bf16(a * s + b * c);
  }
  if (wave < off / 64) {  // act_quant_fp8_kernel, fake, block 64: wave = block, lane = 2 values
    bf16* xb = xr + wave * 64;
    float v[2];
    float amax = 0.0f;
    for (int i = 0; i < 2; ++i) {
      v[i] = bf16_to_f32(xb[lane * 2 + i]);
      amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(warp_max(amax), 1e-4f);
    const float sc = e8m0_to_f32(pow2_ceil_e8m0(amax * (1.0f / 448.0f)));
    for (int i = 0; i < 2; ++i)
      xb[lane * 2 + i] = f32_to_bf16(e4m3_to_f32(f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.0f), 448.0f))) * sc);
  }
}

}  // namespace unholy::kernels
