// unholy-hy4-gemv-bench: Hy4 flash's few-row expert GEMV (kernels::hy4_fp8_grouped_gemv / hy4_nvfp4_grouped_gemv) on
// one device with synthetic experts of the model's shape (N = K = 1536: gate, up and down are all 1536 x 1536).
//
//   unholy-hy4-gemv-bench [--device D] [--fmt fp8|nvfp4] [--experts E] [--rows R] [--iters N]
//
// E experts (distinct weights, as a layer's cold experts on the APU: D = 2), R rows each (1: decode, up to 6: a
// DFly2 verify step), timed over N calls: the weight bytes streamed per second. The first tile's outputs are checked
// against a host reference (the kernel's dequantization: bf16(e4m3 * scale) for FP8, e2m1 * scale for NVFP4).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "kernels/exl3_moe.hpp"
#include "kernels/hy4.hpp"
#include "rt/hip.hpp"

using namespace unholy;
namespace k = unholy::kernels;

namespace {

float bf16_f(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
float e4m3(uint8_t v) {
  const uint32_t e = (v >> 3) & 0xf, m = v & 7;
  float a = e == 0 ? float(m) / 512.0f : std::ldexp(1.0f + float(m) / 8.0f, int(e) - 7);
  return (v & 0x80) ? -a : a;
}
float e2m1(uint32_t n) {
  const uint32_t m = n & 7;
  const float mag = m < 4 ? 0.5f * float(m) : (m == 7 ? 6.0f : float(m - 2));
  return (n & 8) ? -mag : mag;
}
float ue4m3(uint8_t x) { return x == 0 || x == 0x7f ? 0.0f : e4m3(x & 0x7f); }

}  // namespace

int main(int argc, char** argv) {
  int dev = 2, E = 6, R = 1, iters = 200;
  std::string fmt = "fp8";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--device") dev = std::stoi(v);
    else if (a == "--fmt") fmt = v;
    else if (a == "--experts") E = std::stoi(v);
    else if (a == "--rows") R = std::stoi(v);
    else if (a == "--iters") iters = std::stoi(v);
    else {
      std::fprintf(stderr, "usage: %s [--device D] [--fmt fp8|nvfp4] [--experts E] [--rows R] [--iters N]\n", argv[0]);
      return 2;
    }
  }
  const int N = 1536, K = 1536;
  const bool nv = fmt == "nvfp4";
  UH_CHECK(hipSetDevice(dev));
  std::mt19937 rng(1);
  const size_t wbytes = nv ? size_t(N) * K / 2 : size_t(N) * K;
  const size_t sbytes = nv ? size_t(N) * (K / 16) : size_t(N / 128) * (K / 128) * 2;
  std::vector<uint8_t> w(size_t(E) * wbytes), sc(size_t(E) * sbytes);
  for (auto& b : w) {
    b = uint8_t(rng());
    if (!nv && (b & 0x7f) == 0x7f) b ^= 1;  // no NaN codes
  }
  if (nv) {
    for (auto& b : sc) b = uint8_t(0x20 + rng() % 0x30);  // ue4m3 around 2^-3 .. 2^2
  } else {
    auto* s16 = reinterpret_cast<uint16_t*>(sc.data());
    for (size_t i = 0; i < sc.size() / 2; ++i) s16[i] = f_bf16(0.002f + 0.01f * float(rng() % 1000) / 1000.0f);
  }
  std::vector<uint16_t> x(size_t(E) * R * K);
  for (auto& v : x) v = f_bf16(float(int(rng() % 2001) - 1000) / 500.0f);
  void *dw, *ds, *dx, *dy, *dwt, *dst, *dtiles;
  UH_CHECK(hipMalloc(&dw, w.size()));
  UH_CHECK(hipMalloc(&ds, sc.size()));
  UH_CHECK(hipMalloc(&dx, x.size() * 2));
  UH_CHECK(hipMalloc(&dy, size_t(E) * R * N * 2));
  UH_CHECK(hipMalloc(&dwt, size_t(E) * 8));
  UH_CHECK(hipMalloc(&dst, size_t(E) * 8));
  UH_CHECK(hipMalloc(&dtiles, size_t(E) * sizeof(k::Exl3Tile)));
  UH_CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(ds, sc.data(), sc.size(), hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
  std::vector<const void*> wt(static_cast<size_t>(E)), stab(static_cast<size_t>(E));
  std::vector<k::Exl3Tile> tiles(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) {
    wt[size_t(e)] = static_cast<uint8_t*>(dw) + size_t(e) * wbytes;
    stab[size_t(e)] = static_cast<uint8_t*>(ds) + size_t(e) * sbytes;
    tiles[size_t(e)] = {e, e * R, R};
  }
  UH_CHECK(hipMemcpy(dwt, wt.data(), size_t(E) * 8, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dst, stab.data(), size_t(E) * 8, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dtiles, tiles.data(), tiles.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
  hipStream_t st;
  UH_CHECK(hipStreamCreate(&st));
  std::function<void()> call = [&] {
    if (nv)
      k::hy4_nvfp4_grouped_gemv(static_cast<uint16_t*>(dx), K, K, static_cast<k::Exl3Tile*>(dtiles), E,
                                static_cast<const uint8_t* const*>(dwt), static_cast<const uint8_t* const*>(dst),
                                static_cast<uint16_t*>(dy), N, N, st);
    else
      k::hy4_fp8_grouped_gemv(static_cast<uint16_t*>(dx), K, K, static_cast<k::Exl3Tile*>(dtiles), E,
                              static_cast<const uint8_t* const*>(dwt), static_cast<const uint16_t* const*>(dst),
                              static_cast<uint16_t*>(dy), N, N, st);
  };
  static const bool gemm = std::getenv("BENCH_GEMM") != nullptr;  // the 64-row tile GEMM (prefill) instead
  if (gemm) {
    call = [&] {
      if (nv)
        k::hy4_nvfp4_grouped_gemm(static_cast<uint16_t*>(dx), K, static_cast<k::Exl3Tile*>(dtiles), E,
                                  static_cast<const uint8_t* const*>(dwt), static_cast<const uint8_t* const*>(dst),
                                  static_cast<uint16_t*>(dy), N, N, st);
      else
        k::hy4_fp8_grouped_gemm(static_cast<uint16_t*>(dx), K, static_cast<k::Exl3Tile*>(dtiles), E,
                                static_cast<const uint8_t* const*>(dwt), static_cast<const uint16_t* const*>(dst),
                                static_cast<uint16_t*>(dy), N, N, st);
    };
  }
  static const bool wmma = std::getenv("BENCH_WMMA") != nullptr;  // the few-row WMMA kernel instead
  if (wmma) {
    const auto gemv_call = call;
    call = [&, gemv_call] {
      if (!k::hy4_wmma_gemv(nv ? 1 : 0, static_cast<uint16_t*>(dx), K, K, static_cast<k::Exl3Tile*>(dtiles), E,
                            static_cast<const uint8_t* const*>(dwt), static_cast<const void* const*>(dst),
                            static_cast<uint16_t*>(dy), N, N, st))
        gemv_call();
    };
  }
  for (int i = 0; i < 5; ++i) call();
  hipEvent_t a, b;
  UH_CHECK(hipEventCreate(&a));
  UH_CHECK(hipEventCreate(&b));
  UH_CHECK(hipEventRecord(a, st));
  for (int i = 0; i < iters; ++i) call();
  UH_CHECK(hipEventRecord(b, st));
  UH_CHECK(hipEventSynchronize(b));
  float ms = 0;
  UH_CHECK(hipEventElapsedTime(&ms, a, b));
  const double us = double(ms) * 1e3 / iters, gbs = double(E) * double(wbytes + sbytes) / (us * 1e-6) / 1e9;
  // Check the first tile against the host.
  std::vector<uint16_t> y(size_t(R) * N);
  UH_CHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
  double worst = 0;
  for (int r = 0; r < R; ++r)
    for (int n = 0; n < N; ++n) {
      double acc = 0, mag = 0;
      for (int kk = 0; kk < K; ++kk) {
        float wv;
        if (nv) {
          const uint8_t byte = w[size_t(n) * (K / 2) + kk / 16 * 8 + kk % 8];
          const uint32_t nib = kk % 16 < 8 ? (byte & 15u) : (byte >> 4);
          wv = e2m1(nib) * ue4m3(sc[size_t(n) * (K / 16) + kk / 16]);
        } else {
          const float s = bf16_f(reinterpret_cast<const uint16_t*>(sc.data())[size_t(n / 128) * (K / 128) + kk / 128]);
          wv = bf16_f(f_bf16(e4m3(w[size_t(n) * K + kk]) * s));
        }
        const double xv = bf16_f(x[size_t(r) * K + kk]);
        acc += double(wv) * xv;
        mag += std::fabs(double(wv) * xv);
      }
      worst = std::max(worst, std::fabs(double(bf16_f(y[size_t(r) * N + n])) - acc) / std::max(mag, 1e-30));
    }
  const double tflops = 2.0 * N * K * double(E) * R / (us * 1e-6) / 1e12;
  std::printf("device %d %s, %d experts x %d rows: %.1f us per call, %.1f GB/s of weights, %.1f TFLOPS; first tile vs "
              "host: max error %.2e of the sum of |products|\n", dev, fmt.c_str(), E, R, us, gbs, tflops, worst);
  return worst < 1e-2 ? 0 : 1;
}
