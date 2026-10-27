// unholy-exl3-gemm-test: the grouped EXL3 GEMM (v2, 128-row tiles) against v1 (64-row tiles) on
// random trellis words (any bit pattern is a valid code), random inputs and output scales. Outputs
// must be identical; prints both kernels' times and the effective weight-decode rate.
//
//   unholy-exl3-gemm-test [--experts 64] [--rows-mean 96] [--K 4096] [--N 2048] [--bits 3] [--dev 0] [--gemv-rows R]
//                         [--gemv-tiles T]
//
// The decode GEMV runs on one tile per expert of R rows (0: 1-8 at random); UH_DUMP_GEMV=FILE writes its output (bf16)
// for comparisons between builds. With T, also back-to-back GEMVs of T tiles each (output transform in the launch)
// that rotate over the experts, so their weights come from memory rather than the last-level cache, as in decode.
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
  int E = 64, rows_mean = 96, K = 4096, N = 2048, bits = 3, dev = 0, gemv_rows = 0, gemv_tiles = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = std::atoi(argv[i + 1]);
    if (a == "--experts") E = v;
    else if (a == "--rows-mean") rows_mean = v;
    else if (a == "--K") K = v;
    else if (a == "--N") N = v;
    else if (a == "--bits") bits = v;
    else if (a == "--dev") dev = v;
    else if (a == "--gemv-rows") gemv_rows = v;
    else if (a == "--gemv-tiles") gemv_tiles = v;
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
  double gemv_ms = 0, maxrel = 0, gemv_fused_ms = 0;
  size_t gemv_fused_diff = 0;
  {
    std::vector<k::Exl3Tile> tv;
    int rowsv = 0;
    for (int e = 0; e < E; ++e) {
      const int r = gemv_rows > 0 ? gemv_rows : 1 + int(rng() % 8);
      tv.push_back({e, rowsv, r});
      rowsv += r;
    }
    rt::DeviceBuffer dtv(dev, tv.size() * sizeof(k::Exl3Tile)), yp(dev, size_t(rowsv) * N * 4),
        og(dev, size_t(rowsv) * N * 2), ov(dev, size_t(rowsv) * N * 2), dxv(dev, size_t(rowsv) * K * 2);
    {  // its own input rows (the GEMM's P rows may be fewer)
      std::vector<uint16_t> hxv(size_t(rowsv) * K);
      for (auto& v : hxv) v = f2h(nd(rng) * 0.3f);
      UH_CHECK(hipMemcpy(dxv.data(), hxv.data(), hxv.size() * 2, hipMemcpyHostToDevice));
    }
    UH_CHECK(hipMemcpy(dtv.data(), tv.data(), tv.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
    k::exl3_grouped_gemm(dxv.as<uint16_t>(), K, dtv.as<k::Exl3Tile>(), int(tv.size()), trt, svt, bits,
                         og.as<uint16_t>(), N, N, nullptr);
    gemv_ms = time([&] {
      k::exl3_grouped_gemv(dxv.as<uint16_t>(), K, dtv.as<k::Exl3Tile>(), int(tv.size()), trt, svt, bits,
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
    {  // the output transforms in the same launch (counters): bit for bit the separate launch
      rt::DeviceBuffer of(dev, size_t(rowsv) * N * 2), cnt(dev, tv.size() * (N / 128) * 4);
      UH_CHECK(hipMemset(cnt.data(), 0, cnt.size()));
      gemv_fused_ms = time([&] {
        k::exl3_grouped_gemv(dxv.as<uint16_t>(), K, dtv.as<k::Exl3Tile>(), int(tv.size()), trt, svt, bits, yp.as<float>(),
                             of.as<uint16_t>(), N, N, nullptr, nullptr, cnt.as<uint32_t>());
      });
      std::vector<uint16_t> f(size_t(rowsv) * N);
      UH_CHECK(hipMemcpy(f.data(), of.data(), f.size() * 2, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < f.size(); ++i) gemv_fused_diff += f[i] != v[i];
    }
    if (gemv_tiles > 0 && E / gemv_tiles > 0) {  // rotating launches of gemv_tiles experts each
      const int groups = E / gemv_tiles, reps = 4;
      rt::DeviceBuffer cnt(dev, size_t(gemv_tiles) * (N / 128) * 4), of(dev, size_t(rowsv) * N * 2);
      UH_CHECK(hipMemset(cnt.data(), 0, cnt.size()));
      hipEvent_t e0, e1;
      UH_CHECK(hipEventCreate(&e0));
      UH_CHECK(hipEventCreate(&e1));
      auto run = [&] {
        for (int g = 0; g < groups; ++g)
          k::exl3_grouped_gemv(dxv.as<uint16_t>(), K, dtv.as<k::Exl3Tile>() + size_t(g) * gemv_tiles, gemv_tiles, trt,
                               svt, bits, yp.as<float>(), of.as<uint16_t>(), N, N, nullptr, nullptr,
                               cnt.as<uint32_t>());
      };
      run();
      UH_CHECK(hipEventRecord(e0, nullptr));
      for (int r = 0; r < reps; ++r) run();
      UH_CHECK(hipEventRecord(e1, nullptr));
      UH_CHECK(hipEventSynchronize(e1));
      float ms = 0;
      UH_CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double us = 1e3 * ms / (reps * groups);
      std::printf("GEMV rotating (%d tiles per launch over %d experts): %.1f us per launch, %.0f GB/s of weights\n",
                  gemv_tiles, E, us, double(gemv_tiles) * K * N * bits / 8 / 1e3 / us);
      UH_CHECK(hipEventDestroy(e0));
      UH_CHECK(hipEventDestroy(e1));
    }
    if (const char* gd = std::getenv("UH_DUMP_GEMV"))
      if (FILE* fd = std::fopen(gd, "wb")) {
        std::fwrite(v.data(), 2, v.size(), fd);
        std::fclose(fd);
      }
  }
  std::printf("GEMV (E=%d tiles of %s rows): %.3f ms, %.0f G weights/s, %.0f GB/s of weights; rel L2 vs GEMM %.2e\n",
              E, gemv_rows > 0 ? std::to_string(gemv_rows).c_str() : "1-8", gemv_ms,
              double(E) * K * N / 1e9 / (gemv_ms / 1e3), double(E) * K * N * bits / 8 / 1e9 / (gemv_ms / 1e3), maxrel);
  std::printf("GEMV with the output transform in the launch: %.3f ms; %zu outputs differ\n", gemv_fused_ms,
              gemv_fused_diff);
  // The fused dense linear (exl3_linear_gemv) against exl3_xform_in + exl3_grouped_gemv on expert 0's weights.
  size_t fused_diff = 0;
  double sep_ms = 0, fused_ms = 0;
  {
    const int rows = gemv_rows > 0 ? gemv_rows : 1;
    std::vector<uint16_t> hxb(size_t(rows) * K), hsuh(K);
    for (auto& v : hxb) {
      const float f = nd(rng) * 0.3f;
      uint32_t u;
      std::memcpy(&u, &f, 4);
      v = uint16_t(u >> 16);
    }
    for (auto& v : hsuh) v = f2h((rng() & 1 ? 1.0f : -1.0f) * (0.5f + 0.01f * float(rng() % 100)));
    rt::DeviceBuffer dxb(dev, hxb.size() * 2), dsuh(dev, hsuh.size() * 2), dtab3(dev, 3 * 8), dzero(dev, size_t(rows) * 4),
        dxh(dev, size_t(rows) * K * 2), dyp(dev, size_t(rows) * N * 4), dcnt(dev, size_t(N / 128) * 4),
        oa(dev, size_t(rows) * N * 2), ob(dev, size_t(rows) * N * 2), dt(dev, sizeof(k::Exl3Tile));
    UH_CHECK(hipMemcpy(dxb.data(), hxb.data(), hxb.size() * 2, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dsuh.data(), hsuh.data(), hsuh.size() * 2, hipMemcpyHostToDevice));
    const void* t3[3] = {tab[0], dsuh.data(), stab[0]};
    UH_CHECK(hipMemcpy(dtab3.data(), t3, sizeof t3, hipMemcpyHostToDevice));
    UH_CHECK(hipMemset(dzero.data(), 0, dzero.size()));
    UH_CHECK(hipMemset(dcnt.data(), 0, dcnt.size()));
    const k::Exl3Tile t{0, 0, rows};
    UH_CHECK(hipMemcpy(dt.data(), &t, sizeof t, hipMemcpyHostToDevice));
    const auto* t3d = dtab3.as<const uint16_t*>();
    sep_ms = time([&] {
      k::exl3_xform_in(dxb.as<uint16_t>(), K, dzero.as<int32_t>(), t3d + 1, dxh.as<uint16_t>(), rows, K, false, nullptr);
      k::exl3_grouped_gemv(dxh.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), 1, t3d, t3d + 2, bits, dyp.as<float>(),
                           oa.as<uint16_t>(), N, N, nullptr);
    });
    fused_ms = time([&] {
      k::exl3_linear_gemv(dxb.as<uint16_t>(), K, K, dt.as<k::Exl3Tile>(), 1, t3d, bits, dyp.as<float>(),
                          dcnt.as<uint32_t>(), ob.as<uint16_t>(), N, N, nullptr);
    });
    std::vector<uint16_t> a2(size_t(rows) * N), b2(size_t(rows) * N);
    UH_CHECK(hipMemcpy(a2.data(), oa.data(), a2.size() * 2, hipMemcpyDeviceToHost));
    UH_CHECK(hipMemcpy(b2.data(), ob.data(), b2.size() * 2, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < a2.size(); ++i) fused_diff += a2[i] != b2[i];
    std::vector<uint32_t> cnt(N / 128);
    UH_CHECK(hipMemcpy(cnt.data(), dcnt.data(), cnt.size() * 4, hipMemcpyDeviceToHost));
    for (uint32_t c : cnt) fused_diff += c != 0;  // counters left zeroed
    std::printf("dense linear (%d rows): xform_in + GEMV %.3f ms, fused %.3f ms; %zu outputs differ\n", rows, sep_ms,
                fused_ms, fused_diff);
  }
  const double gw = double(E) * K * N / 1e9;  // weights per expert pass
  std::printf("E=%d P=%d K=%d N=%d bits=%d: v1 %.2f ms (%zu tiles), v2 %.2f ms (%zu tiles) -> %.2fx; "
              "v2 %.0f G weights/s, %.1f TFLOPS; outputs %zu / %zu differ\n",
              E, P, K, N, bits, ms1, t1.size(), ms2, t2.size(), ms1 / ms2, gw / (ms2 / 1e3),
              2.0 * P * double(K) * N / (ms2 / 1e3) / 1e12, diff, a.size());
  return diff || fused_diff || gemv_fused_diff ? 1 : 0;
}
