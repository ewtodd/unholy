// unholy-hy4-dense-bench: the few-row linears of Hy4 flash's decode and verify steps on one card, the GEMVs against
// the few-row WMMA kernel (kernels::hy4_wmma_gemv). Each call runs C copies of a matrix (C tiles naming C "experts"),
// more bytes than the card's 64 MB cache holds, so every weight is read from memory once per call: the time per
// matrix is the call's / C.
//
//   unholy-hy4-dense-bench [--device D] [--iters N]
//
// Shapes: the bf16 q_a + kv_a, the MoE's pre-expert and routed-up projections (bf16 output); the fp8 gate, o_proj and
// q_b of a window layer; an fp8 and an NVFP4 routed expert's gate (1536 x 1536). Rows 1, 2, 4, 6, 8. Then the
// attention's per-head products (kernels::hy4_heads_gemv against the grouped GEMM path), rows 1 .. 32.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/exl3_moe.hpp"
#include "kernels/gemv.hpp"
#include "kernels/glm_fast.hpp"
#include "kernels/ref_glm.hpp"
#include "kernels/hy4.hpp"
#include "rt/hip.hpp"

using namespace unholy;
namespace k = unholy::kernels;

int main(int argc, char** argv) {
  int dev = 1, iters = 20;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--device") dev = std::stoi(v);
    else if (a == "--iters") iters = std::stoi(v);
  }
  UH_CHECK(hipSetDevice(dev));
  hipStream_t st;
  UH_CHECK(hipStreamCreate(&st));
  std::mt19937 rng(3);
  const auto timed = [&](const std::function<void()>& f) {
    for (int i = 0; i < 2; ++i) f();
    hipEvent_t a, b;
    UH_CHECK(hipEventCreate(&a));
    UH_CHECK(hipEventCreate(&b));
    UH_CHECK(hipEventRecord(a, st));
    for (int i = 0; i < iters; ++i) f();
    UH_CHECK(hipEventRecord(b, st));
    UH_CHECK(hipEventSynchronize(b));
    float ms = 0;
    UH_CHECK(hipEventElapsedTime(&ms, a, b));
    return double(ms) * 1e3 / iters;
  };
  struct Shape {
    const char* name;
    int N, K, fmt;  // fmt: 0 fp8, 1 NVFP4, 2 bf16
  };
  const Shape shapes[] = {{"q_a bf16", 1024, 3072, 2},        {"kv_a bf16", 576, 3072, 2},
                          {"q_a+kv_a bf16", 1600, 3072, 2},   {"pre_expert bf16", 1536, 3072, 2},
                          {"routed_up bf16", 3072, 1536, 2},  {"gate fp8", 16384, 3072, 0},
                          {"o_proj fp8", 3072, 16384, 0},     {"q_b fp8", 16384, 1024, 0},
                          {"expert fp8", 3072, 1536, 0},      {"expert nvfp4", 3072, 1536, 1}};
  for (const Shape& sh : shapes) {
    const size_t wbytes = sh.fmt == 2 ? size_t(sh.N) * sh.K * 2 : sh.fmt == 0 ? size_t(sh.N) * sh.K : size_t(sh.N) * sh.K / 2;
    const size_t sbytes = sh.fmt == 0 ? size_t(sh.N / 128) * (sh.K / 128) * 2 : sh.fmt == 1 ? size_t(sh.N) * (sh.K / 16) : 0;
    const int C = int(std::min<size_t>(64, std::max<size_t>(2, (size_t(512) << 20) / wbytes)));
    std::vector<uint8_t> hw(wbytes);
    for (auto& b : hw) b = uint8_t(rng());
    if (sh.fmt == 0)
      for (auto& b : hw)
        if ((b & 0x7f) == 0x7f) b ^= 1;
    if (sh.fmt == 2)
      for (size_t i = 1; i < wbytes; i += 2) hw[i] = uint8_t(0x3c + (hw[i] & 0x81));
    std::vector<uint8_t> hs(sbytes);
    if (sh.fmt == 0)
      for (size_t i = 0; i < sbytes; i += 2) hs[i] = 0x00, hs[i + 1] = 0x3c;
    if (sh.fmt == 1)
      for (auto& b : hs) b = uint8_t(0x20 + rng() % 0x30);
    void *w, *s = nullptr, *x, *y, *wtab, *stab, *tiles;
    UH_CHECK(hipMalloc(&w, wbytes * C));
    for (int c = 0; c < C; ++c) UH_CHECK(hipMemcpy(static_cast<uint8_t*>(w) + wbytes * c, hw.data(), wbytes, hipMemcpyHostToDevice));
    if (sbytes) {
      UH_CHECK(hipMalloc(&s, sbytes * C));
      for (int c = 0; c < C; ++c) UH_CHECK(hipMemcpy(static_cast<uint8_t*>(s) + sbytes * c, hs.data(), sbytes, hipMemcpyHostToDevice));
    }
    std::vector<uint16_t> hx(size_t(8) * sh.K);
    for (auto& v : hx) v = uint16_t(0x3c00 + rng() % 0x200);
    UH_CHECK(hipMalloc(&x, hx.size() * 2));
    UH_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    UH_CHECK(hipMalloc(&y, size_t(8) * sh.N * 2));
    std::vector<const void*> pw(static_cast<size_t>(C)), ps(static_cast<size_t>(C));
    for (int c = 0; c < C; ++c) {
      pw[size_t(c)] = static_cast<uint8_t*>(w) + wbytes * c;
      ps[size_t(c)] = s ? static_cast<uint8_t*>(s) + sbytes * c : nullptr;
    }
    UH_CHECK(hipMalloc(&wtab, size_t(C) * 8));
    UH_CHECK(hipMalloc(&stab, size_t(C) * 8));
    UH_CHECK(hipMemcpy(wtab, pw.data(), size_t(C) * 8, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(stab, ps.data(), size_t(C) * 8, hipMemcpyHostToDevice));
    UH_CHECK(hipMalloc(&tiles, size_t(C) * sizeof(k::Exl3Tile)));
    for (int M : {1, 2, 4, 6, 8}) {
      std::vector<k::Exl3Tile> ht(static_cast<size_t>(C));
      for (int c = 0; c < C; ++c) ht[size_t(c)] = {c, 0, M};
      UH_CHECK(hipMemcpy(tiles, ht.data(), ht.size() * sizeof(k::Exl3Tile), hipMemcpyHostToDevice));
      const auto* bx = static_cast<const uint16_t*>(x);
      auto* by = static_cast<uint16_t*>(y);
      const auto* tl = static_cast<const k::Exl3Tile*>(tiles);
      const auto* tw = static_cast<const uint8_t* const*>(wtab);
      double t_gemv = 0;
      if (sh.fmt == 2)
        t_gemv = timed([&] {
          for (int c = 0; c < C; ++c)
            k::gemv_bf16(bx, sh.K, static_cast<const uint16_t*>(pw[size_t(c)]), by, sh.N, M, sh.N, sh.K, st);
        });
      else if (sh.fmt == 0)
        t_gemv = timed([&] {
          k::hy4_fp8_grouped_gemv(bx, sh.K, sh.K, tl, C, tw, static_cast<const uint16_t* const*>(stab), by, sh.N, sh.N, st);
        });
      else
        t_gemv = timed([&] {
          k::hy4_nvfp4_grouped_gemv(bx, sh.K, sh.K, tl, C, tw, static_cast<const uint8_t* const*>(stab), by, sh.N, sh.N, st);
        });
      const double t_wmma = timed([&] {
        k::hy4_wmma_gemv(sh.fmt, bx, sh.K, sh.K, tl, C, tw, sh.fmt == 2 ? nullptr : static_cast<const void* const*>(stab),
                         by, sh.N, sh.N, st);
      });
      // One launch per matrix, as a step runs them (a launch's few column blocks may leave CUs idle).
      const double t_single = timed([&] {
        for (int c = 0; c < C; ++c)
          k::hy4_wmma_gemv(sh.fmt, bx, sh.K, sh.K, tl + c, 1, tw,
                           sh.fmt == 2 ? nullptr : static_cast<const void* const*>(stab), by, sh.N, sh.N, st);
      });
      std::printf("%-16s N %6d K %5d rows %d (%2d copies): gemv %7.1f us/matrix (%4.0f GB/s)   wmma %7.1f us/matrix "
                  "(%4.0f GB/s), one launch each %7.1f us (%4.0f GB/s)\n", sh.name, sh.N, sh.K, M, C, t_gemv / C,
                  double(wbytes + sbytes) * C / t_gemv / 1e3, t_wmma / C, double(wbytes + sbytes) * C / t_wmma / 1e3,
                  t_single / C, double(wbytes + sbytes) * C / t_single / 1e3);
    }
    for (void* p : {w, s, x, y, wtab, stab, tiles})
      if (p) UH_CHECK(hipFree(p));
  }
  // The attention's per-head bf16 products: the absorbed query (w_ukT [64][512][192], the rope tail copied) and the
  // gated value-up (w_uv [64][256][512]), the grouped GEMM (+ split / pack / gate kernels) against the per-head GEMV.
  struct Heads {
    const char* name;
    int H, N, K, a_head, c_head, tail;
    bool gate;
  };
  const Heads heads[] = {{"w_ukT", 64, 512, 192, 256, 576, 64, false}, {"w_uv+gate", 64, 256, 512, 512, 256, 0, true}};
  for (const Heads& hs : heads) {
    const size_t wbytes = size_t(hs.H) * hs.N * hs.K * 2;
    const int C = int(std::max<size_t>(2, (size_t(512) << 20) / wbytes));
    std::vector<uint16_t> hw(wbytes / 2);
    for (auto& v : hw) v = uint16_t(0x3c00 + (rng() & 0x81ff) - 0x100);
    void *w, *x, *g, *y0, *y1, *tmp0, *tmp1;
    UH_CHECK(hipMalloc(&w, wbytes * C));
    for (int c = 0; c < C; ++c) UH_CHECK(hipMemcpy(static_cast<uint8_t*>(w) + wbytes * c, hw.data(), wbytes, hipMemcpyHostToDevice));
    const int maxr = 32, lda = hs.H * hs.a_head, ldc = hs.H * hs.c_head;
    std::vector<uint16_t> hx(size_t(maxr) * lda), hg(size_t(maxr) * hs.H * hs.N);
    for (auto& v : hx) v = uint16_t(0x3c00 + rng() % 0x200 - 0x100 + ((rng() & 1) << 15));
    for (auto& v : hg) v = uint16_t(0x3f00 + rng() % 0x100 + ((rng() & 1) << 15));
    UH_CHECK(hipMalloc(&x, hx.size() * 2));
    UH_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    UH_CHECK(hipMalloc(&g, hg.size() * 2));
    UH_CHECK(hipMemcpy(g, hg.data(), hg.size() * 2, hipMemcpyHostToDevice));
    UH_CHECK(hipMalloc(&y0, size_t(maxr) * ldc * 2));
    UH_CHECK(hipMalloc(&y1, size_t(maxr) * ldc * 2));
    UH_CHECK(hipMalloc(&tmp0, size_t(maxr) * hs.H * std::max(hs.K, hs.N) * 2));
    UH_CHECK(hipMalloc(&tmp1, size_t(maxr) * hs.H * std::max(hs.K, hs.N) * 2));
    for (int M : {1, 2, 4, 6, 8, 16, 32}) {
      const auto* bx = static_cast<const uint16_t*>(x);
      const auto* bg = static_cast<const uint16_t*>(g);
      auto* o0 = static_cast<uint16_t*>(y0);
      auto* o1 = static_cast<uint16_t*>(y1);
      auto* t0 = static_cast<uint16_t*>(tmp0);
      auto* t1 = static_cast<uint16_t*>(tmp1);
      const auto wc = [&](int c) { return reinterpret_cast<const uint16_t*>(static_cast<uint8_t*>(w) + wbytes * size_t(c)); };
      const auto ref = [&](int c) {
        if (hs.tail) {
          k::hy4_split_q(bx, t0, M, hs.H, st);
          k::gemm_grouped_bf16_wmma(t0, wc(c), t1, M, hs.H, hs.N, hs.K, st);
          k::hy4_pack_q(t1, bx, o0, M, hs.H, st);
        } else {
          k::gemm_grouped_bf16_wmma(bx, wc(c), o0, M, hs.H, hs.N, hs.K, st);
          k::hy4_gate_mul(o0, bg, size_t(M) * hs.H * hs.N, st);
        }
      };
      const auto fused = [&](int c) {
        if (!k::hy4_heads_gemv(bx, lda, hs.a_head, hs.K, wc(c), hs.H, o1, ldc, hs.c_head, hs.N, M, st,
                               hs.gate ? bg : nullptr, hs.H * hs.N, hs.N, hs.tail))
          throw std::runtime_error("hy4_heads_gemv refused the shape");
      };
      const double t_ref = timed([&] { for (int c = 0; c < C; ++c) ref(c); });
      const double t_new = timed([&] { for (int c = 0; c < C; ++c) fused(c); });
      ref(0);
      fused(0);
      UH_CHECK(hipStreamSynchronize(st));
      std::vector<uint16_t> a(size_t(M) * ldc), b(size_t(M) * ldc);
      UH_CHECK(hipMemcpy(a.data(), o0, a.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(b.data(), o1, b.size() * 2, hipMemcpyDeviceToHost));
      const auto f = [](uint16_t v) { uint32_t u = uint32_t(v) << 16; float r; std::memcpy(&r, &u, 4); return r; };
      double worst = 0, mag = 0;
      size_t same = 0;
      for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, double(std::fabs(f(a[i]) - f(b[i]))));
        mag = std::max(mag, double(std::fabs(f(a[i]))));
        same += a[i] == b[i];
      }
      std::printf("%-10s H %d N %4d K %4d rows %2d (%2d copies): GEMM path %6.1f us   per-head GEMV %6.1f us   "
                  "(%4.0f GB/s); max |diff| %.3g of max %.3g, %.1f%% bit-equal\n", hs.name, hs.H, hs.N, hs.K, M, C,
                  t_ref / C, t_new / C, double(wbytes) * C / t_new / 1e3, worst, mag, 100.0 * double(same) / double(a.size()));
    }
    for (void* p : {w, x, g, y0, y1, tmp0, tmp1}) UH_CHECK(hipFree(p));
  }
  // A sublayer boundary: rmsnorm of the output, the stream add (ihc_post) and the next mix with its norm (ihc_pre),
  // three kernels, against hy4_ihc_fused; random data of the model's shape (hc 4, dim 3072).
  {
    const int hc = 4, D = 3072, W = hc * D;
    const float eps = 1e-6f, hc_eps = 1e-6f, mag = 2.0f;
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const auto bf = [](float f) {
      uint32_t v;
      std::memcpy(&v, &f, 4);
      return uint16_t((v + 0x7fffu + ((v >> 16) & 1u)) >> 16);
    };
    const auto f = [](uint16_t v) { uint32_t w = uint32_t(v) << 16; float r; std::memcpy(&r, &w, 4); return r; };
    for (int T : {1, 6, 16}) {
      std::vector<uint16_t> hx(size_t(T) * W), hout(size_t(T) * D), hnw(D), honw(D);
      std::vector<float> hfn(size_t(2 * hc) * W), hbase(2 * hc), hscale{0.8f, 1.3f}, hpost(size_t(T) * hc);
      for (auto& v : hx) v = bf(u(rng) * 2.0f);
      for (auto& v : hout) v = bf(u(rng) * 0.5f);
      for (auto& v : hnw) v = bf(1.0f + 0.2f * u(rng));
      for (auto& v : honw) v = bf(1.0f + 0.2f * u(rng));
      for (auto& v : hfn) v = 0.02f * u(rng);
      for (auto& v : hbase) v = 0.5f * u(rng);
      for (auto& v : hpost) v = 1.0f + u(rng);
      void *dx0, *dx1, *dout, *dout0, *dnw, *donw, *dfn, *dbase, *dscale, *dp0, *dp1, *dy0, *dy1, *dh0, *dh1, *dws;
      const size_t wsb = k::hy4_ihc_ws_bytes(T, hc, D);
      for (auto* pp : {&dx0, &dx1}) UH_CHECK(hipMalloc(pp, hx.size() * 2));
      for (auto* pp : {&dout, &dout0, &dy0, &dy1, &dh0, &dh1}) UH_CHECK(hipMalloc(pp, hout.size() * 2));
      for (auto* pp : {&dnw, &donw}) UH_CHECK(hipMalloc(pp, D * 2));
      UH_CHECK(hipMalloc(&dfn, hfn.size() * 4));
      UH_CHECK(hipMalloc(&dbase, hbase.size() * 4));
      UH_CHECK(hipMalloc(&dscale, 8));
      for (auto* pp : {&dp0, &dp1}) UH_CHECK(hipMalloc(pp, hpost.size() * 4));
      UH_CHECK(hipMalloc(&dws, wsb));
      UH_CHECK(hipMemset(dws, 0, wsb));
      UH_CHECK(hipMemcpy(dout, hout.data(), hout.size() * 2, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(dnw, hnw.data(), D * 2, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(donw, honw.data(), D * 2, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(dfn, hfn.data(), hfn.size() * 4, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(dbase, hbase.data(), hbase.size() * 4, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(dscale, hscale.data(), 8, hipMemcpyHostToDevice));
      const auto reset = [&] {
        UH_CHECK(hipMemcpy(dx0, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        UH_CHECK(hipMemcpy(dx1, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        UH_CHECK(hipMemcpy(dp0, hpost.data(), hpost.size() * 4, hipMemcpyHostToDevice));
        UH_CHECK(hipMemcpy(dp1, hpost.data(), hpost.size() * 4, hipMemcpyHostToDevice));
        UH_CHECK(hipMemcpy(dout0, hout.data(), hout.size() * 2, hipMemcpyHostToDevice));
      };
      const auto b16 = [](void* p) { return static_cast<uint16_t*>(p); };
      const auto f32 = [](void* p) { return static_cast<float*>(p); };
      const auto three = [&] {
        k::rmsnorm_hf(b16(dout0), D, b16(donw), b16(dout0), D, T, D, eps, st);
        k::hy4_ihc_post(b16(dout0), b16(dx0), f32(dp0), T, hc, D, st);
        k::hy4_ihc_pre(b16(dx0), f32(dfn), f32(dscale), f32(dbase), b16(dy0), f32(dp0), T, hc, D, eps, hc_eps, mag, st,
                       b16(dnw), b16(dh0));
      };
      const auto fused = [&] {
        k::hy4_ihc_fused(b16(dout), b16(donw), f32(dp1), b16(dx1), f32(dfn), f32(dscale), f32(dbase), b16(dy1), f32(dp1),
                         T, hc, D, eps, hc_eps, mag, b16(dnw), b16(dh1), dws, T, st);
      };
      reset();
      three();
      fused();
      UH_CHECK(hipStreamSynchronize(st));
      std::vector<uint16_t> x0(hx.size()), x1(hx.size()), y0(hout.size()), y1(hout.size()), h0(hout.size()), h1(hout.size());
      std::vector<float> p0(hpost.size()), p1(hpost.size());
      UH_CHECK(hipMemcpy(x0.data(), dx0, x0.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(x1.data(), dx1, x1.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(y0.data(), dy0, y0.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(y1.data(), dy1, y1.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(h0.data(), dh0, h0.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(h1.data(), dh1, h1.size() * 2, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(p0.data(), dp0, p0.size() * 4, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(p1.data(), dp1, p1.size() * 4, hipMemcpyDeviceToHost));
      const auto cmp = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* name) {
        size_t same = 0;
        double worst = 0;
        for (size_t i = 0; i < a.size(); ++i) {
          same += a[i] == b[i];
          worst = std::max(worst, double(std::fabs(f(a[i]) - f(b[i]))));
        }
        std::printf("  %s %.2f%% bit-equal, max |diff| %.3g", name, 100.0 * double(same) / double(a.size()), worst);
      };
      double pw = 0;
      for (size_t i = 0; i < p0.size(); ++i) pw = std::max(pw, double(std::fabs(p0[i] - p1[i]) / std::fabs(p0[i])));
      // timing (the inputs drift over the repeats; only the time matters)
      const double t3 = timed(three), t1 = timed(fused);
      std::printf("hc boundary T %2d: three kernels %6.1f us, fused %6.1f us;", T, t3, t1);
      cmp(x0, x1, "streams");
      cmp(y0, y1, "mix");
      cmp(h0, h1, "normed");
      std::printf("  post max rel diff %.3g\n", pw);
      for (void* pp : {dx0, dx1, dout, dout0, dnw, donw, dfn, dbase, dscale, dp0, dp1, dy0, dy1, dh0, dh1, dws})
        UH_CHECK(hipFree(pp));
    }
  }
  return 0;
}
