// unholy-exl3-gemm-test: the grouped EXL3 GEMM (v2, 128-row tiles) against v1 (64-row tiles) on
// random trellis words (any bit pattern is a valid code), random inputs and output scales. Outputs
// must be identical; prints both kernels' times and the effective weight-decode rate.
//
//   unholy-exl3-gemm-test [--experts 64] [--rows-mean 96] [--K 4096] [--N 2048] [--bits 3] [--dev 0]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/exl3_moe.hpp"
#include "rt/hip.hpp"

using namespace unholy;
namespace k = unholy::kernels;

namespace {
uint16_t f2h(float f) { return uint16_t(__half_as_ushort(__float2half(f))); }
}  // namespace

int main(int argc, char** argv) {
  int E = 64, rows_mean = 96, K = 4096, N = 2048, bits = 3, dev = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = std::atoi(argv[i + 1]);
    if (a == "--experts") E = v;
    else if (a == "--rows-mean") rows_mean = v;
    else if (a == "--K") K = v;
    else if (a == "--N") N = v;
    else if (a == "--bits") bits = v;
    else if (a == "--dev") dev = v;
  }
  const char* dump = std::getenv("UH_DUMP_OUT");  // write v2's output (bf16) there
  UH_CHECK(hipSetDevice(dev));
  std::mt19937_64 rng(1);
  std::vector<int> counts(static_cast<size_t>(E));
  int P = 0;
  for (auto& c : counts) {
    c = int(rng() % uint64_t(2 * rows_mean + 1));
    P += c;
  }
  // Tile lists: v1 runs of <= 64 rows, v2 of <= 128.
  std::vector<k::Exl3Tile> t1, t2;
  for (int e = 0, row = 0; e < E; row += counts[size_t(e)], ++e) {
    for (int i = 0; i < counts[size_t(e)]; i += 64) t1.push_back({e, row + i, std::min(64, counts[size_t(e)] - i)});
    for (int i = 0; i < counts[size_t(e)]; i += 128) t2.push_back({e, row + i, std::min(128, counts[size_t(e)] - i)});
  }
  const size_t words = size_t(K / 16) * (N / 16) * 8 * bits;  // uint32 per expert
  std::vector<uint32_t> htr(words * E);
  for (auto& w : htr) w = uint32_t(rng());
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> hx(size_t(P) * K), hsvh(size_t(N) * E);
  for (auto& v : hx) v = f2h(nd(rng) * 0.3f);
  for (auto& v : hsvh) v = f2h((rng() & 1 ? 1.0f : -1.0f) * (0.5f + 0.01f * float(rng() % 100)));

  rt::DeviceBuffer dtr(dev, htr.size() * 4), dx(dev, hx.size() * 2), dsvh(dev, hsvh.size() * 2),
      dtab(dev, size_t(E) * 8), dstab(dev, size_t(E) * 8), dt1(dev, t1.size() * sizeof(k::Exl3Tile)),
      dt2(dev, t2.size() * sizeof(k::Exl3Tile)), o1(dev, size_t(P) * N * 2), o2(dev, size_t(P) * N * 2);
  UH_CHECK(hipMemcpy(dtr.data(), htr.data(), htr.size() * 4, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dx.data(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dsvh.data(), hsvh.data(), hsvh.size() * 2, hipMemcpyHostToDevice));
  std::vector<const uint16_t*> tab(static_cast<size_t>(E)), stab(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) {
    tab[size_t(e)] = reinterpret_cast<const uint16_t*>(dtr.as<uint32_t>() + size_t(e) * words);
    stab[size_t(e)] = dsvh.as<uint16_t>() + size_t(e) * N;
  }
  UH_CHECK(hipMemcpy(dtab.data(), tab.data(), tab.size() * 8, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dstab.data(), stab.data(), stab.size() * 8, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dt1.data(), t1.data(), t1.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dt2.data(), t2.data(), t2.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
  UH_CHECK(hipMemset(o1.data(), 0, o1.size()));
  UH_CHECK(hipMemset(o2.data(), 0, o2.size()));
  const auto* trt = dtab.as<const uint16_t*>();
  const auto* svt = dstab.as<const uint16_t*>();

  auto time = [&](auto fn) {
    fn();
    UH_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 3; ++r) fn();
    UH_CHECK(hipDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3;
  };
  const double ms1 = time([&] {
    k::exl3_grouped_gemm_v1(dx.as<uint16_t>(), K, dt1.as<k::Exl3Tile>(), int(t1.size()), trt, svt, bits,
                            o1.as<uint16_t>(), N, N, nullptr);
  });
  const double ms2 = time([&] {
    k::exl3_grouped_gemm(dx.as<uint16_t>(), K, dt2.as<k::Exl3Tile>(), int(t2.size()), trt, svt, bits,
                         o2.as<uint16_t>(), N, N, nullptr);
  });
  std::vector<uint16_t> a(size_t(P) * N), b(size_t(P) * N);
  UH_CHECK(hipMemcpy(a.data(), o1.data(), a.size() * 2, hipMemcpyDeviceToHost));
  UH_CHECK(hipMemcpy(b.data(), o2.data(), b.size() * 2, hipMemcpyDeviceToHost));
  size_t diff = 0;
  for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
  if (dump)
    if (FILE* fd = std::fopen(dump, "wb")) {
      std::fwrite(b.data(), 2, b.size(), fd);
      std::fclose(fd);
    }
  // Decode path: GEMV on tiles of <= 8 rows vs the GEMM on the same rows (different summation order).
  double gemv_ms = 0, maxrel = 0;
  {
    std::vector<k::Exl3Tile> tv;
    int rowsv = 0;
    for (int e = 0; e < E; ++e) {
      const int r = 1 + int(rng() % 8);
      tv.push_back({e, rowsv, r});
      rowsv += r;
    }
    rt::DeviceBuffer dtv(dev, tv.size() * sizeof(k::Exl3Tile)), yp(dev, size_t(rowsv) * N * 4),
        og(dev, size_t(rowsv) * N * 2), ov(dev, size_t(rowsv) * N * 2);
    UH_CHECK(hipMemcpy(dtv.data(), tv.data(), tv.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
    k::exl3_grouped_gemm(dx.as<uint16_t>(), K, dtv.as<k::Exl3Tile>(), int(tv.size()), trt, svt, bits,
                         og.as<uint16_t>(), N, N, nullptr);
    gemv_ms = time([&] {
      k::exl3_grouped_gemv(dx.as<uint16_t>(), K, dtv.as<k::Exl3Tile>(), int(tv.size()), trt, svt, bits,
                           yp.as<float>(), ov.as<uint16_t>(), N, N, nullptr);
    });
    std::vector<uint16_t> g(size_t(rowsv) * N), v(size_t(rowsv) * N);
    UH_CHECK(hipMemcpy(g.data(), og.data(), g.size() * 2, hipMemcpyDeviceToHost));
    UH_CHECK(hipMemcpy(v.data(), ov.data(), v.size() * 2, hipMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < g.size(); ++i) {
      uint32_t ug = uint32_t(g[i]) << 16, uv = uint32_t(v[i]) << 16;
      float fg, fv;
      std::memcpy(&fg, &ug, 4);
      std::memcpy(&fv, &uv, 4);
      num += double(fg - fv) * (fg - fv);
      den += double(fg) * fg;
    }
    maxrel = std::sqrt(num / std::max(den, 1e-30));
  }
  std::printf("GEMV (E=%d tiles of 1-8 rows): %.3f ms, %.0f G weights/s; rel L2 vs GEMM %.2e\n", E, gemv_ms,
              double(E) * K * N / 1e9 / (gemv_ms / 1e3), maxrel);
  const double gw = double(E) * K * N / 1e9;  // weights per expert pass
  std::printf("E=%d P=%d K=%d N=%d bits=%d: v1 %.2f ms (%zu tiles), v2 %.2f ms (%zu tiles) -> %.2fx; "
              "v2 %.0f G weights/s, %.1f TFLOPS; outputs %zu / %zu differ\n",
              E, P, K, N, bits, ms1, t1.size(), ms2, t2.size(), ms1 / ms2, gw / (ms2 / 1e3),
              2.0 * P * double(K) * N / (ms2 / 1e3) / 1e12, diff, a.size());
  return diff ? 1 : 0;
}
