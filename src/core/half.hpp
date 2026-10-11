// Host-side IEEE binary16 / bfloat16 / fp8 conversions for reference code and tests.
// Device code uses the HIP intrinsics; these exist so CPU references round exactly like the GPU.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace unholy {

inline float half_to_float(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t man = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {  // subnormal: renormalize
      int e = -1;
      uint32_t m = man;
      do {
        m <<= 1;
        ++e;
      } while (!(m & 0x400));
      f = sign | uint32_t(127 - 15 - e) << 23 | (m & 0x3ff) << 13;
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | man << 13;
  } else {
    f = sign | (exp + 127 - 15) << 23 | man << 13;
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// Round-to-nearest-even from double. Exact for any double, so a computation done exactly in
// double and rounded once here matches a fused fp16 operation (e.g. v_fma_f16).
inline uint16_t double_to_half(double x) {
  const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
  x = std::fabs(x);
  if (std::isnan(x)) return sign | 0x7e00;
  if (x >= 65520.0) return sign | 0x7c00;  // rounds to inf
  if (x < std::ldexp(1.0, -25)) return sign;  // below half the smallest subnormal
  int e;
  std::frexp(x, &e);  // x = f * 2^e, f in [0.5, 1)
  // Quantum: normals have 10 fraction bits below the leading one (2^(e-11)); subnormals 2^-24.
  const int q_exp = std::max(e - 11, -24);
  const double q = std::ldexp(1.0, q_exp);
  double n = x / q;  // exact (power-of-two division)
  double fl = std::floor(n);
  const double rem = n - fl;
  if (rem > 0.5 || (rem == 0.5 && std::fmod(fl, 2.0) != 0.0)) fl += 1.0;
  const double r = fl * q;
  if (r >= 65520.0) return sign | 0x7c00;
  // Encode r exactly.
  if (r < std::ldexp(1.0, -14)) return sign | uint16_t(r / std::ldexp(1.0, -24));
  int re;
  const double rf = std::frexp(r, &re);  // r = rf * 2^re
  const uint16_t exp = uint16_t(re - 1 + 15);
  const uint16_t man = uint16_t((rf * 2.0 - 1.0) * 1024.0);
  return sign | uint16_t(exp << 10) | man;
}

inline uint16_t float_to_half(float x) { return double_to_half(double(x)); }

inline float bf16_to_float(uint16_t b) {
  const uint32_t f = uint32_t(b) << 16;
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// OCP fp8 e4m3fn (bias 7, no infinities, 0x7f/0xff = NaN).
inline float e4m3_to_float(uint8_t v) {
  const float sign = (v & 0x80) ? -1.0f : 1.0f;
  const int exp = (v >> 3) & 0xf;
  const int man = v & 0x7;
  if (exp == 0xf && man == 0x7) return NAN;
  if (exp == 0) return sign * std::ldexp(float(man), -9);
  return sign * std::ldexp(1.0f + man / 8.0f, exp - 7);
}

// OCP e8m0 scale: 2^(v - 127), 0xff = NaN.
inline float e8m0_to_float(uint8_t v) {
  if (v == 0xff) return NAN;
  return std::ldexp(1.0f, int(v) - 127);
}

// OCP fp4 e2m1 (MXFP4 element), low 4 bits of `v`.
inline float e2m1_to_float(uint8_t v) {
  static constexpr float kLut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = kLut[v & 7];
  return (v & 8) ? -m : m;
}

}  // namespace unholy
