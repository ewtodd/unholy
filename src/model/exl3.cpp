#include "model/exl3.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/half.hpp"

namespace unholy::model {

namespace {

// Codebook constants, from exllamav3_ext/quant/codebook.cuh.
constexpr uint32_t kMcgMult = 0xCBAC1FEDu;
constexpr uint32_t kMul1Mult = 0x83DCD12Du;

uint16_t sum_halves(uint32_t x) {
  // fp16 add of the two halves, rounded once (exact in double first).
  const double lo = half_to_float(uint16_t(x & 0xffff));
  const double hi = half_to_float(uint16_t(x >> 16));
  return double_to_half(lo + hi);
}

}  // namespace

const std::array<uint16_t, 256>& exl3_tensor_core_perm() {
  static const std::array<uint16_t, 256> perm = [] {
    std::array<uint16_t, 256> p{};
    for (int t = 0; t < 32; ++t) {
      const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
      const int c0 = t / 4, c1 = c0 + 8;
      p[t * 8 + 0] = uint16_t(r0 * 16 + c0);
      p[t * 8 + 1] = uint16_t(r1 * 16 + c0);
      p[t * 8 + 2] = uint16_t(r2 * 16 + c0);
      p[t * 8 + 3] = uint16_t(r3 * 16 + c0);
      p[t * 8 + 4] = uint16_t(r0 * 16 + c1);
      p[t * 8 + 5] = uint16_t(r1 * 16 + c1);
      p[t * 8 + 6] = uint16_t(r2 * 16 + c1);
      p[t * 8 + 7] = uint16_t(r3 * 16 + c1);
    }
    return p;
  }();
  return perm;
}

uint16_t exl3_decode_state(Exl3Codebook cb, uint32_t state) {
  switch (cb) {
    case Exl3Codebook::ThreeInst: {
      uint32_t x = state * 89226354u + 64248484u;
      x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
      return sum_halves(x);
    }
    case Exl3Codebook::Mcg: {
      uint32_t x = state * kMcgMult;
      x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
      return sum_halves(x);
    }
    case Exl3Codebook::Mul1: {
      const uint32_t x = state * kMul1Mult;
      const uint32_t sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24) + 0x6400u;
      // fma(h, 1/147.7, -10.39) in fp16, one rounding.
      const double h = half_to_float(uint16_t(sum));
      const double k_inv = half_to_float(0x1eee);
      const double k_bias = half_to_float(0xc931);
      return double_to_half(h * k_inv + k_bias);
    }
  }
  return 0;
}

uint32_t exl3_tile_state(const uint32_t* words, int K, int t) {
  const int len = 256 * K;  // bits
  const int end = (t + 1) * K;
  uint32_t s = 0;
  for (int i = 0; i < 16; ++i) {
    const int p = ((end - 16 + i) % len + len) % len;
    s = (s << 1) | ((words[p / 32] >> (31 - p % 32)) & 1u);
  }
  return s;
}

std::optional<Exl3Linear> find_exl3_linear(const Checkpoint& ck, const std::string& prefix) {
  Exl3Linear l;
  l.prefix = prefix;
  l.trellis = ck.find(prefix + ".trellis");
  if (!l.trellis) return std::nullopt;
  l.suh = ck.find(prefix + ".suh");
  l.svh = ck.find(prefix + ".svh");
  if (!l.suh || !l.svh)
    throw std::runtime_error(prefix + ": trellis without suh/svh (packed su/sv not supported)");
  const auto& sh = l.trellis->shape;
  if (sh.size() != 3 || l.trellis->dtype != DType::I16) throw std::runtime_error(prefix + ": bad trellis");
  if (sh[2] % 16 != 0) throw std::runtime_error(prefix + ": half-integer bitrate not supported");
  l.K = int(sh[2] / 16);
  l.in_features = int(sh[0] * 16);
  l.out_features = int(sh[1] * 16);
  if (l.suh->numel() != l.in_features || l.svh->numel() != l.out_features)
    throw std::runtime_error(prefix + ": suh/svh shape mismatch");

  auto marker = [&](const char* name, uint32_t expect) -> bool {
    const TensorInfo* t = ck.find(prefix + "." + name);
    if (!t) return false;
    const uint32_t v = ck.as<uint32_t>(*t)[0];
    if (v != expect) throw std::runtime_error(prefix + ": unexpected " + name + " multiplier");
    return true;
  };
  if (marker("mul1", kMul1Mult)) l.cb = Exl3Codebook::Mul1;
  else if (marker("mcg", kMcgMult)) l.cb = Exl3Codebook::Mcg;
  else l.cb = Exl3Codebook::ThreeInst;
  return l;
}

void exl3_reconstruct_inner_cpu(const Checkpoint& ck, const Exl3Linear& l, uint16_t* w) {
  const auto& perm = exl3_tensor_core_perm();
  const auto bytes = ck.bytes(*l.trellis);
  const int tiles_k = l.in_features / 16, tiles_n = l.out_features / 16;
  const size_t tile_words = size_t(8 * l.K);
  std::vector<uint32_t> words(tile_words);
  for (int kb = 0; kb < tiles_k; ++kb)
    for (int nb = 0; nb < tiles_n; ++nb) {
      const size_t tile = size_t(kb) * tiles_n + nb;
      std::memcpy(words.data(), bytes.data() + tile * tile_words * 4, tile_words * 4);
      for (int t = 0; t < 256; ++t) {
        const uint16_t v = exl3_decode_state(l.cb, exl3_tile_state(words.data(), l.K, t));
        const int pos = perm[t];
        w[size_t(kb * 16 + pos / 16) * l.out_features + nb * 16 + pos % 16] = v;
      }
    }
}

void fwht128(double* x, int stride) {
  for (int h = 1; h < 128; h *= 2)
    for (int i = 0; i < 128; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const double a = x[size_t(j) * stride], b = x[size_t(j + h) * stride];
        x[size_t(j) * stride] = a + b;
        x[size_t(j + h) * stride] = a - b;
      }
  const double s = 1.0 / std::sqrt(128.0);
  for (int i = 0; i < 128; ++i) x[size_t(i) * stride] *= s;
}

void exl3_dense_cpu(const Checkpoint& ck, const Exl3Linear& l, float* out) {
  const int K = l.in_features, N = l.out_features;
  if (K % 128 || N % 128) throw std::runtime_error(l.prefix + ": dims not multiples of 128");
  std::vector<uint16_t> inner(size_t(K) * N);
  exl3_reconstruct_inner_cpu(ck, l, inner.data());
  std::vector<double> w(size_t(K) * N);
  for (size_t i = 0; i < w.size(); ++i) w[i] = half_to_float(inner[i]);
  const auto suh = ck.as<uint16_t>(*l.suh);
  const auto svh = ck.as<uint16_t>(*l.svh);
  // Hk along the input dim (column-wise blocks of 128 rows), then row scale suh.
  for (int n = 0; n < N; ++n)
    for (int k0 = 0; k0 < K; k0 += 128) fwht128(&w[size_t(k0) * N + n], N);
  for (int k = 0; k < K; ++k) {
    const double s = half_to_float(suh[k]);
    for (int n = 0; n < N; ++n) w[size_t(k) * N + n] *= s;
  }
  // Hn along the output dim, then column scale svh.
  for (int k = 0; k < K; ++k)
    for (int n0 = 0; n0 < N; n0 += 128) fwht128(&w[size_t(k) * N + n0], 1);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) out[size_t(k) * N + n] = float(w[size_t(k) * N + n] * half_to_float(svh[n]));
}

}  // namespace unholy::model
