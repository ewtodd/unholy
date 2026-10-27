// Device-side numeric helpers. Storage types are raw integers (bf16 = uint16_t, fp8 = uint8_t) so the
// same code compiles for gfx1201 and gfx1151 and every conversion is explicit and exactly specified.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::dev {

using bf16 = uint16_t;

// Arch families for the WMMA paths (each fat-binary pass defines its own target macro).
#if defined(__gfx1200__) || defined(__gfx1201__)
#define UH_GFX12 1
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define UH_GFX11 1
#endif

__device__ __forceinline__ float bf16_to_f32(bf16 v) { return __uint_as_float(uint32_t(v) << 16); }

// Round-to-nearest-even (NaN preserved as quiet NaN).
__device__ __forceinline__ bf16 f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return bf16((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return bf16(u >> 16);
}

// bf16(x) as float: the value torch would hold after rounding an fp32 intermediate to bf16.
__device__ __forceinline__ float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }
// The same for finite inputs only (no NaN branch): three integer ops.
__device__ __forceinline__ float round_bf16_finite(float f) {
  const uint32_t u = __float_as_uint(f);
  return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}

// OCP fp8 e4m3fn.
__device__ __forceinline__ float e4m3_to_f32(uint8_t v) {
  const uint32_t e = (v >> 3) & 0xf, m = v & 7;
  float a = e == 0 ? float(m) * (1.0f / 512.0f) : __uint_as_float((e + 120u) << 23 | m << 20);
  return (v & 0x80) ? -a : a;
}

// Four e4m3fn codes (the bytes of w, lowest first): gfx12's conversion instructions (exact: the values
// of e4m3_to_f32 for every finite code; unholy-gemv-test checks all of them), the bit decode elsewhere.
__device__ __forceinline__ void e4m3x4_to_f32(uint32_t w, float* v) {
#if defined(UH_GFX12) && __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
  const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(int(w), false);
  const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(int(w), true);
  v[0] = lo[0];
  v[1] = lo[1];
  v[2] = hi[0];
  v[3] = hi[1];
#else
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = e4m3_to_f32(uint8_t(w >> (8 * i)));
#endif
}

// Round-to-nearest-even into e4m3fn; caller guarantees |x| <= 448 (the reference clamps first).
__device__ __forceinline__ uint8_t f32_to_e4m3(float x) {
  uint32_t u = __float_as_uint(x);
  const uint8_t sign = uint8_t((u >> 24) & 0x80);
  u &= 0x7fffffffu;
  const int e = int(u >> 23) - 127;
  if (e < -6) {  // e4m3 subnormal range: value = m * 2^-9
    const float r = rintf(__uint_as_float(u) * 512.0f);  // exact scaling, RNE (8 -> min normal 0x08)
    return sign | uint8_t(r);
  }
  u += 0x7ffffu + ((u >> 20) & 1u);  // RNE at 3 mantissa bits
  const uint32_t ee = (u >> 23) - 127 + 7, man = (u >> 20) & 7;
  return sign | uint8_t(ee << 3 | man);
}

__device__ __forceinline__ float e8m0_to_f32(uint8_t v) { return __uint_as_float(uint32_t(v) << 23); }

// e2m1 grid value of a nibble.
__device__ __forceinline__ float e2m1_to_f32(uint32_t nib) {
  const uint32_t m = nib & 7;
  // 0, 0.5, 1, 1.5, 2, 3, 4, 6
  const float mag = m < 4 ? 0.5f * float(m) : (m == 7 ? 6.0f : float(m - 2));
  return (nib & 8) ? -mag : mag;
}

// Round-to-nearest-even onto the e2m1 grid (x already clamped to [-6, 6]).
__device__ __forceinline__ float round_e2m1(float x) {
  const float a = fabsf(x);
  float r;
  if (a <= 0.25f) r = 0.0f;
  else if (a < 0.75f) r = 0.5f;
  else if (a <= 1.25f) r = 1.0f;
  else if (a < 1.75f) r = 1.5f;
  else if (a <= 2.5f) r = 2.0f;
  else if (a < 3.5f) r = 3.0f;
  else if (a <= 5.0f) r = 4.0f;
  else r = 6.0f;
  return x < 0 ? -r : r;
}

// Power-of-two scale 2^ceil(log2(v)) for positive normal v (DeepSeek kernel.py fast_round_scale),
// returned as the e8m0 exponent byte.
__device__ __forceinline__ uint8_t pow2_ceil_e8m0(float v) {
  const uint32_t b = __float_as_uint(v);
  const int e = int((b >> 23) & 0xff) - 127 + ((b & 0x7fffff) != 0 ? 1 : 0);
  return uint8_t(e + 127);
}

// Wave32 reductions (RDNA HIP default wave size).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}

// Block-wide sum for blockDim.x <= 1024 (multiple of 32); `red` holds >= 32 floats.
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < nw ? red[lane] : 0.0f;
  return warp_sum(v);
}

// The mHC post-mix of column d of token t (hc_post_kernel, ref_elem.hip): stream j = bf16(post[j] * x + sum over i of
// comb[i][j] * res[i]), with x = xv and res the token's hc stream rows [hc][dim] (out may be res: each column alone).
__device__ __forceinline__ void hc_post_column(float xv, const bf16* res, const float* post, const float* comb,
                                               bf16* out, int hc, int dim, int t, int d) {
  const float* pt = post + size_t(t) * hc;
  const float* ct = comb + size_t(t) * hc * hc;
  float r[4];
  for (int i = 0; i < hc; ++i) r[i] = bf16_to_f32(res[(size_t(t) * hc + i) * dim + d]);
  for (int j = 0; j < hc; ++j) {
    float s = 0.0f;
    for (int i = 0; i < hc; ++i) s += ct[i * hc + j] * r[i];
    out[(size_t(t) * hc + j) * dim + d] = f32_to_bf16(pt[j] * xv + s);
  }
}

// A8 fake quantization of one 128-value block held by a 128-thread block (one value per thread):
// the reference's act_quant (amax floor 1e-4, power-of-two e8m0 scale, e4m3, back to bf16). `red`: 4
// floats of LDS.
__device__ __forceinline__ float fake_quant_fp8_block(float v, float* red) {
  float a = warp_max(fabsf(v));
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (lane == 0) red[wave] = a;
  __syncthreads();
  a = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();
  a = fmaxf(a, 1e-4f);
  const float sc = e8m0_to_f32(pow2_ceil_e8m0(a * (1.0f / 448.0f)));
  return round_bf16(e4m3_to_f32(f32_to_e4m3(fminf(fmaxf(v / sc, -448.0f), 448.0f))) * sc);
}

}  // namespace unholy::dev
