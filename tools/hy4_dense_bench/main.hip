// unholy-hy4-dense-bench: the few-row linears of Hy4 flash's decode and verify steps on one card, the GEMVs against
// the few-row WMMA kernel (kernels::hy4_wmma_gemv). Each call runs C copies of a matrix (C tiles naming C "experts"),
// more bytes than the card's 64 MB cache holds, so every weight is read from memory once per call: the time per
// matrix is the call's / C.
//
//   unholy-hy4-dense-bench [--device D] [--iters N]
//
// Shapes: the bf16 q_a + kv_a, the MoE's pre-expert and routed-up projections (bf16 output); the fp8 gate, o_proj and
// q_b of a window layer; an fp8 and an NVFP4 routed expert's gate (1536 x 1536). Rows 1, 2, 4, 6, 8.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "kernels/exl3_moe.hpp"
#include "kernels/gemv.hpp"
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
  const Shape shapes[] = {{"q_a+kv_a bf16", 1600, 3072, 2},   {"pre_expert bf16", 1536, 3072, 2},
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
      std::printf("%-16s N %6d K %5d rows %d (%2d copies): gemv %7.1f us/matrix (%4.0f GB/s)   wmma %7.1f us/matrix "
                  "(%4.0f GB/s)\n", sh.name, sh.N, sh.K, M, C, t_gemv / C, double(wbytes + sbytes) * C / t_gemv / 1e3,
                  t_wmma / C, double(wbytes + sbytes) * C / t_wmma / 1e3);
    }
    for (void* p : {w, s, x, y, wtab, stab, tiles})
      if (p) UH_CHECK(hipFree(p));
  }
  return 0;
}
