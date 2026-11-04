// unholy-mx-test: native MXFP4 grouped expert kernels (mx_xform_in + mx_grouped_gemm / gemv) against the
// reference (act_quant_fp8 per 128 + gemm_a8w4, one expert at a time) on random weights and inputs.
// Same products, different fp32 summation order: expect rel L2 ~1e-3 (bf16 outputs), not identity.
// Then decode-sized tiles (1-3 rows, one of 37): the 128-row GEMM against its skinny kernel (max_rows 16,
// which takes three passes over the 37-row tile, and 64), bit for bit, and their times.
//
//   unholy-mx-test [--experts 16] [--rows-mean 96] [--K 4096] [--N 2048] [--dev 0]
//     [--counts FILE] [--reps 3] [--gemm-only 1] [--no-ref 1] [--check-dequant 1] [--glu 1]
// (mode 0 only: --counts, rows per expert from FILE (E = their number); --reps; --no-ref skips the reference;
// --check-dequant: mx_dequant32_perm against mx_dequant32_lut, all scales x all values, on the device;
// --glu 1: gfx11, w1 / w3 GEMM + mx_swiglu against mx_grouped_gemm_glu, bit for bit and timed)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "kernels/mx_moe.hpp"
#include "kernels/ref.hpp"
#include "rt/hip.hpp"

using namespace unholy;
namespace k = unholy::kernels;

namespace {
uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float from_bf16(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace

int main(int argc, char** argv) {
  int E_opt = 16, rows_mean = 96, K = 4096, N = 2048, dev = 0, reps = 3, gemm_only = 0, no_ref = 0, check_dequant = 0, glu = 0;
  std::vector<int> file_counts;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = std::atoi(argv[i + 1]);
    if (a == "--counts") {
      std::ifstream f(argv[i + 1]);
      for (int c; f >> c;) file_counts.push_back(c);
      if (file_counts.empty()) {
        std::fprintf(stderr, "--counts: no integers in %s\n", argv[i + 1]);
        return 2;
      }
    } else if (a == "--glu") {
      glu = v;
    } else if (a == "--check-dequant") {
      check_dequant = v;
    } else if (a == "--reps") reps = std::max(v, 1);
    else if (a == "--gemm-only") gemm_only = v;
    else if (a == "--no-ref") no_ref = v;
    else if (a == "--experts") E_opt = v;
    else if (a == "--rows-mean") rows_mean = v;
    else if (a == "--K") K = v;
    else if (a == "--N") N = v;
    else if (a == "--dev") dev = v;
  }
  UH_CHECK(hipSetDevice(dev));
  if (check_dequant) {
    unsigned long long total = 0;
    const unsigned long long bad = k::mx_check_dequant(&total);
    std::printf("dequant perm vs lut: %llu mismatches of %llu\n", bad, total);
    if (bad) return 1;
  }
  std::mt19937_64 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  bool all_same = true;
  for (int mode = 0; mode < (gemm_only ? 1 : 3); ++mode) {  // 0: GEMM (prefill tiles), 1: GEMV (<= 8 rows), 2: GEMM, decode tiles
    const int E = mode == 0 && !file_counts.empty() ? int(file_counts.size()) : E_opt;
    std::vector<int> counts(static_cast<size_t>(E));
    int P = 0;
    for (size_t ci = 0; ci < counts.size(); ++ci) {
      int& c = counts[ci];
      c = mode == 0 && !file_counts.empty() ? file_counts[ci] : mode == 2 ? 1 + int(rng() % 3) : mode ? 1 + int(rng() % 8) : int(rng() % uint64_t(2 * rows_mean + 1));
      P += c;
    }
    if (mode == 2) {
      P += 37 - counts[0];
      counts[0] = 37;
    }
    std::vector<k::Exl3Tile> tiles;
    std::vector<int> row_expert;
    for (int e = 0, row = 0; e < E; row += counts[size_t(e)], ++e) {
      for (int i = 0; i < counts[size_t(e)]; i += k::kExl3TileRows)
        tiles.push_back({e, row + i, std::min(k::kExl3TileRows, counts[size_t(e)] - i)});
      for (int i = 0; i < counts[size_t(e)]; ++i) row_expert.push_back(e);
    }
    std::vector<uint8_t> hw(size_t(E) * N * K / 2), hs(size_t(E) * N * (K / 32));
    for (auto& b : hw) b = uint8_t(rng());
    for (auto& s : hs) s = uint8_t(127 - 4 + int(rng() % 5));  // 2^-4 .. 2^0
    std::vector<uint16_t> hx(size_t(P) * K);
    for (auto& v : hx) v = to_bf16(nd(rng) * 0.5f);

    rt::DeviceBuffer dw(dev, hw.size()), ds(dev, hs.size()), tw(dev, hw.size()), ts(dev, hs.size()),
        tmp(dev, size_t(N) * K / 2 + size_t(N) * (K / 32)), dx(dev, hx.size() * 2), xa(dev, hx.size() * 2),
        re(dev, row_expert.size() * 4), dt(dev, tiles.size() * sizeof(k::Exl3Tile)), wt(dev, size_t(E) * 8),
        st(dev, size_t(E) * 8), out(dev, size_t(P) * N * 2), ref(dev, size_t(P) * N * 2), qb(dev, size_t(P) * K),
        qs(dev, size_t(P) * (K / 128));
    UH_CHECK(hipMemcpy(dw.data(), hw.data(), hw.size(), hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(ds.data(), hs.data(), hs.size(), hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dx.data(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(tw.data(), hw.data(), hw.size(), hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(ts.data(), hs.data(), hs.size(), hipMemcpyHostToDevice));
    for (int e = 0; e < E; ++e)  // the fast kernels read the tile-major copy
      k::mx_repack(tw.as<uint8_t>() + size_t(e) * N * K / 2, ts.as<uint8_t>() + size_t(e) * N * (K / 32), N, K,
                   tmp.as<uint8_t>(), nullptr);
    UH_CHECK(hipMemcpy(re.data(), row_expert.data(), row_expert.size() * 4, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dt.data(), tiles.data(), tiles.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
    std::vector<const uint8_t*> wtab(static_cast<size_t>(E)), stab(static_cast<size_t>(E));
    std::vector<const uint8_t*> wref(static_cast<size_t>(E)), sref(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) {
      wtab[size_t(e)] = tw.as<uint8_t>() + size_t(e) * N * K / 2;
      stab[size_t(e)] = ts.as<uint8_t>() + size_t(e) * N * (K / 32);
      wref[size_t(e)] = dw.as<uint8_t>() + size_t(e) * N * K / 2;
      sref[size_t(e)] = ds.as<uint8_t>() + size_t(e) * N * (K / 32);
    }
    UH_CHECK(hipMemcpy(wt.data(), wtab.data(), size_t(E) * 8, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(st.data(), stab.data(), size_t(E) * 8, hipMemcpyHostToDevice));
    const auto* wtp = wt.as<const uint8_t*>();
    const auto* stp = st.as<const uint8_t*>();

    // Reference, expert by expert.
    const bool ref_on = !(mode == 0 && no_ref);
    if (ref_on) k::act_quant_fp8(dx.as<uint16_t>(), K, qb.as<uint8_t>(), qs.as<uint8_t>(), P, K, 128, nullptr);
    for (int e = 0, row = 0; ref_on && e < E; row += counts[size_t(e)], ++e)
      if (counts[size_t(e)] > 0)
        k::gemm_a8w4(qb.as<uint8_t>() + size_t(row) * K, qs.as<uint8_t>() + size_t(row) * (K / 128), wref[size_t(e)],
                     sref[size_t(e)], ref.as<uint16_t>() + size_t(row) * N, N, counts[size_t(e)], N, K, nullptr);
    if (mode == 0 && ref_on) {  // the reference on the tiled copy must give the same bits as on the original
      rt::DeviceBuffer ref2(dev, size_t(P) * N * 2);
      for (int e = 0, row = 0; e < E; row += counts[size_t(e)], ++e)
        if (counts[size_t(e)] > 0)
          k::gemm_a8w4_tiled(qb.as<uint8_t>() + size_t(row) * K, qs.as<uint8_t>() + size_t(row) * (K / 128),
                             wtab[size_t(e)], stab[size_t(e)], ref2.as<uint16_t>() + size_t(row) * N, N,
                             counts[size_t(e)], N, K, nullptr);
      std::vector<uint16_t> r1(size_t(P) * N), r2(size_t(P) * N);
      UH_CHECK(hipMemcpy(r1.data(), ref.data(), r1.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(r2.data(), ref2.data(), r2.size() * 2, hipMemcpyDeviceToHost));
      std::printf("reference on tiled vs original layout: %s\n", r1 == r2 ? "identical" : "DIFFERENT");
    }
    // Fast path.
    k::mx_xform_in(dx.as<uint16_t>(), K, re.as<int32_t>(), stp, xa.as<uint16_t>(), P, K, nullptr);
    if (mode == 2) {
      std::vector<uint16_t> o[3];
      double ms[3];
      const int max_rows[3] = {k::kExl3TileRows, 16, 64};
      for (int v = 0; v < 3; ++v) {
        k::mx_gemm_force_rb(v ? 0 : -1);  // gfx11: the skinny variants with the register-B tiles off
        const auto go = [&] {
          k::mx_grouped_gemm(xa.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), int(tiles.size()), wtp, stp,
                             out.as<uint16_t>(), N, N, nullptr, nullptr, nullptr, nullptr, nullptr, max_rows[v]);
        };
        UH_CHECK(hipMemset(out.data(), 0, out.size()));
        go();
        UH_CHECK(hipDeviceSynchronize());
        o[v].resize(size_t(P) * N);
        UH_CHECK(hipMemcpy(o[v].data(), out.data(), o[v].size() * 2, hipMemcpyDeviceToHost));
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 5; ++r) go();
        UH_CHECK(hipDeviceSynchronize());
        ms[v] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
      }
      k::mx_gemm_force_rb(-1);
      const bool same = o[0] == o[1] && o[0] == o[2];
      all_same &= same;
      const double bytes = double(E) * N * K * (0.5 + 1.0 / 32);
      std::printf("GEMM decode tiles: E=%d P=%d K=%d N=%d: 128-row tiles %.3f ms (%.0f GB/s), skinny max_rows 16 "
                  "%.3f ms (%.0f GB/s), 64 %.3f ms (%.0f GB/s): %s\n",
                  E, P, K, N, ms[0], bytes / (ms[0] / 1e3) / 1e9, ms[1], bytes / (ms[1] / 1e3) / 1e9, ms[2],
                  bytes / (ms[2] / 1e3) / 1e9, same ? "bit-identical" : "OUTPUTS DIFFER");
      continue;
    }
    auto run = [&] {
      if (mode == 0)
        k::mx_grouped_gemm(xa.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), int(tiles.size()), wtp, stp,
                           out.as<uint16_t>(), N, N, nullptr);
      else
        k::mx_grouped_gemv(xa.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), int(tiles.size()), wtp, stp,
                           out.as<uint16_t>(), N, N, nullptr);
    };
    run();
    UH_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    const int nrep = mode == 0 ? reps : 3;
    for (int r = 0; r < nrep; ++r) run();
    UH_CHECK(hipDeviceSynchronize());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / nrep;
    std::vector<uint16_t> a(size_t(P) * N), b(size_t(P) * N);
    UH_CHECK(hipMemcpy(b.data(), out.data(), b.size() * 2, hipMemcpyDeviceToHost));
    if (!ref_on) {
      const double bytes = double(E) * N * K * (0.5 + 1.0 / 32);
      std::printf("GEMM: E=%d P=%d K=%d N=%d: %.3f ms (%.0f GB/s of weights, %.1f TFLOPS); no reference\n", E, P, K, N,
                  ms, bytes / (ms / 1e3) / 1e9, 2.0 * P * double(K) * N / (ms / 1e3) / 1e12);
    }
    if (ref_on) UH_CHECK(hipMemcpy(a.data(), ref.data(), a.size() * 2, hipMemcpyDeviceToHost));
    double num = 0, den = 0;
    size_t same = 0;
    for (size_t i = 0; ref_on && i < a.size(); ++i) {
      const double x = from_bf16(a[i]), y = from_bf16(b[i]);
      num += (x - y) * (x - y);
      den += x * x;
      same += a[i] == b[i];
    }
    const double bytes = double(E) * N * K * (0.5 + 1.0 / 32);
    if (ref_on)
    std::printf("%s: E=%d P=%d K=%d N=%d: %.3f ms (%.0f GB/s of weights, %.1f TFLOPS); rel L2 vs reference %.2e, "
                "%.1f%% bit-identical\n",
                mode ? "GEMV" : "GEMM", E, P, K, N, ms, bytes / (ms / 1e3) / 1e9, 2.0 * P * double(K) * N / (ms / 1e3) / 1e12,
                std::sqrt(num / std::max(den, 1e-30)), 100.0 * same / a.size());
    if (mode == 0) {  // gfx11: the register-B tiles (default) against the LDS ones
      std::vector<uint16_t> o[2];
      double t[2];
      for (int v = 0; v < 2; ++v) {
        k::mx_gemm_force_rb(v);
        UH_CHECK(hipMemset(out.data(), 0, out.size()));
        run();
        UH_CHECK(hipDeviceSynchronize());
        o[v].resize(size_t(P) * N);
        UH_CHECK(hipMemcpy(o[v].data(), out.data(), o[v].size() * 2, hipMemcpyDeviceToHost));
        const auto t1 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) run();
        UH_CHECK(hipDeviceSynchronize());
        t[v] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / reps;
      }
      k::mx_gemm_force_rb(-1);
      all_same &= o[0] == o[1];
      std::printf("  tiles, B in LDS / in registers (gfx11): %.3f / %.3f ms (%.1f / %.1f TFLOPS), %s\n", t[0], t[1],
                  2.0 * P * double(K) * N / (t[0] / 1e3) / 1e12, 2.0 * P * double(K) * N / (t[1] / 1e3) / 1e12,
                  o[0] == o[1] ? "bit-identical" : "DIFFERENT");
      if (k::mx_gemm_rb()) {  // A in one buffer, two barriers per slab, against the default (double-buffered) output
        k::mx_gemm_rb_double_buffer(0);
        UH_CHECK(hipMemset(out.data(), 0, out.size()));
        run();
        UH_CHECK(hipDeviceSynchronize());
        std::vector<uint16_t> od(size_t(P) * N);
        UH_CHECK(hipMemcpy(od.data(), out.data(), od.size() * 2, hipMemcpyDeviceToHost));
        const auto t1 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) run();
        UH_CHECK(hipDeviceSynchronize());
        const double td = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / reps;
        k::mx_gemm_rb_double_buffer(-1);
        all_same &= od == o[1];
        std::printf("  rb with A in one buffer: %.3f ms (%.1f TFLOPS), %s\n", td,
                    2.0 * P * double(K) * N / (td / 1e3) / 1e12, od == o[1] ? "bit-identical" : "DIFFERENT");
      }
    }
    if (mode == 0 && glu && !k::mx_gemm_rb()) std::printf("GLU: skipped (gfx11 register-B tiles only)\n");
    if (mode == 0 && glu && k::mx_gemm_rb()) {  // w1 / w3 + mx_swiglu against the fused kernel (gfx11)
      // w3 of expert e is the tile-major copy of expert e + 1's weights; w2's scale table (which rows are resident) is
      // null for every third expert; random route weights, limit 10.
      std::vector<const uint8_t*> w3tab(static_cast<size_t>(E)), s3tab(static_cast<size_t>(E)), s2tab(static_cast<size_t>(E));
      for (int e = 0; e < E; ++e) {
        w3tab[size_t(e)] = wtab[size_t((e + 1) % E)];
        s3tab[size_t(e)] = stab[size_t((e + 1) % E)];
        s2tab[size_t(e)] = e % 3 == 2 ? nullptr : stab[size_t(e)];
      }
      std::vector<float> hwt(static_cast<size_t>(P));
      for (auto& v : hwt) v = 0.5f + float(rng() % 1000) / 1000.0f;
      rt::DeviceBuffer w3t(dev, size_t(E) * 8), s3t(dev, size_t(E) * 8), s2t(dev, size_t(E) * 8), dwt(dev, size_t(P) * 4),
          gb(dev, size_t(P) * N * 2), ub(dev, size_t(P) * N * 2), hold(dev, size_t(P) * N * 2), hnew(dev, size_t(P) * N * 2);
      UH_CHECK(hipMemcpy(w3t.data(), w3tab.data(), size_t(E) * 8, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(s3t.data(), s3tab.data(), size_t(E) * 8, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(s2t.data(), s2tab.data(), size_t(E) * 8, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(dwt.data(), hwt.data(), size_t(P) * 4, hipMemcpyHostToDevice));
      const auto* w3p = w3t.as<const uint8_t*>();
      const auto* s3p = s3t.as<const uint8_t*>();
      const auto* s2p = s2t.as<const uint8_t*>();
      const float limit = 10.0f;
      const auto tm = [&](auto&& f) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) f();
        UH_CHECK(hipDeviceSynchronize());
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
      };
      const auto gemm_old = [&] {
        k::mx_grouped_gemm(xa.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), int(tiles.size()), wtp, stp, gb.as<uint16_t>(), N, N,
                           nullptr, nullptr, w3p, s3p, ub.as<uint16_t>());
      };
      const auto swiglu_old = [&] {
        k::mx_swiglu(gb.as<uint16_t>(), ub.as<uint16_t>(), dwt.as<float>(), re.as<int32_t>(), s2p, hold.as<uint16_t>(), P, N,
                     limit, nullptr);
      };
      const auto fused = [&] {
        k::mx_grouped_gemm_glu(xa.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), int(tiles.size()), wtp, stp, w3p, s3p, N,
                               k::MxGlu{dwt.as<float>(), re.as<int32_t>(), s2p, hnew.as<uint16_t>(), limit, nullptr}, nullptr);
      };
      k::mx_gemm_force_rb(1);
      UH_CHECK(hipMemset(hold.data(), 0, hold.size()));
      gemm_old();
      swiglu_old();
      UH_CHECK(hipDeviceSynchronize());
      const double t_gemm = tm(gemm_old), t_swi = tm(swiglu_old);
      std::vector<uint16_t> ho(size_t(P) * N), hn(size_t(P) * N);
      UH_CHECK(hipMemcpy(ho.data(), hold.data(), ho.size() * 2, hipMemcpyDeviceToHost));
      double t_f[2];
      bool same = true;
      for (int v = 0; v < 2; ++v) {  // A double-buffered / one buffer
        k::mx_gemm_rb_double_buffer(1 - v);
        UH_CHECK(hipMemset(hnew.data(), 0, hnew.size()));
        fused();
        UH_CHECK(hipDeviceSynchronize());
        UH_CHECK(hipMemcpy(hn.data(), hnew.data(), hn.size() * 2, hipMemcpyDeviceToHost));
        same &= hn == ho;
        t_f[v] = tm(fused);
      }
      k::mx_gemm_rb_double_buffer(-1);
      k::mx_gemm_force_rb(-1);
      all_same &= same;
      std::printf("GLU: old %.3f ms (GEMM %.3f + swiglu %.3f), fused %.3f ms (DB on), %.3f ms (DB off): %s\n",
                  t_gemm + t_swi, t_gemm, t_swi, t_f[0], t_f[1], same ? "bit-identical" : "DIFFERENT");
    }
    if (mode == 0 && std::getenv("MX_TEST_SKINNY")) {  // experiment: the skinny kernel on prefill tiles
      const auto go = [&] {
        k::mx_grouped_gemm(xa.as<uint16_t>(), K, dt.as<k::Exl3Tile>(), int(tiles.size()), wtp, stp,
                           out.as<uint16_t>(), N, N, nullptr, nullptr, nullptr, nullptr, nullptr, 64);
      };
      UH_CHECK(hipMemset(out.data(), 0, out.size()));
      go();
      UH_CHECK(hipDeviceSynchronize());
      std::vector<uint16_t> c(size_t(P) * N);
      UH_CHECK(hipMemcpy(c.data(), out.data(), c.size() * 2, hipMemcpyDeviceToHost));
      const auto t1 = std::chrono::steady_clock::now();
      for (int r = 0; r < 3; ++r) go();
      UH_CHECK(hipDeviceSynchronize());
      const double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 3;
      std::printf("  skinny on the same tiles: %.3f ms (%.1f TFLOPS), %s\n", ms2,
                  2.0 * P * double(K) * N / (ms2 / 1e3) / 1e12, c == b ? "bit-identical" : "DIFFERENT");
    }
  }
  return all_same ? 0 : 1;
}
