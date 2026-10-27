// unholy-gemv-test: the skinny decode GEMVs (gemv.hpp) on the model's decode shapes: outputs must not
// depend on the columns per wave (bit for bit), and the time / weight bandwidth of each variant. Likewise
// the WMMA GEMMs' 64 x 128 tiles vs their skinny kernels for M <= 64 (gemm_wmma.hpp): bit for bit.
//
//   unholy-gemv-test [--dev 0] [--reps 50] [--only-smallk]
//
// Also GLM's KDA low-rank gate projections (gemv_bf16_smallk2: two N = 8192, K = 128 bf16 GEMVs in one launch).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/dev.hip.hpp"
#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "rt/hip.hpp"

using namespace unholy;
namespace k = unholy::kernels;

namespace {
// Every e4m3 code through e4m3x4_to_f32 (hardware on gfx12) and e4m3_to_f32 (bit decode).
__global__ void fp8_decode_kernel(uint32_t* bad) {
  const uint32_t c = threadIdx.x;
  float v[4];
  unholy::dev::e4m3x4_to_f32(c | (c << 8) | (c << 16) | (c << 24), v);
  const float r = unholy::dev::e4m3_to_f32(uint8_t(c));
  if ((c & 0x7f) != 0x7f)
    for (int i = 0; i < 4; ++i)
      if (__float_as_uint(v[i]) != __float_as_uint(r)) atomicAdd(bad, 1u);
}
uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
template <typename T>
rt::DeviceBuffer upload(int dev, const std::vector<T>& v) {
  rt::DeviceBuffer b(dev, v.size() * sizeof(T));
  UH_CHECK(hipMemcpy(b.data(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return b;
}
std::vector<uint8_t> download(const rt::DeviceBuffer& b) {
  std::vector<uint8_t> v(b.size());
  UH_CHECK(hipMemcpy(v.data(), b.data(), b.size(), hipMemcpyDeviceToHost));
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  int dev = 0, reps = 50;
  bool only_smallk = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--only-smallk") only_smallk = true;
    else if (i + 1 < argc && a == "--dev") dev = std::stoi(argv[++i]);
    else if (i + 1 < argc && a == "--reps") reps = std::stoi(argv[++i]);
  }
  rt::DeviceGuard g(dev);
  hipStream_t st;
  UH_CHECK(hipStreamCreate(&st));
  std::mt19937 rng(7);
  const auto fp8 = [&](size_t n) {  // finite e4m3 codes
    std::vector<uint8_t> v(n);
    for (auto& x : v) {
      x = uint8_t(rng());
      if ((x & 0x7f) == 0x7f) x &= 0xfe;
    }
    return v;
  };
  const auto e8m0 = [&](size_t n) {
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = uint8_t(124 + rng() % 7);
    return v;
  };
  const auto bf16v = [&](size_t n) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = to_bf16(d(rng));
    return v;
  };
  struct Shape {
    const char* name;
    char kind;  // 'w' w8a8, 'b' bf16, 'f' bf16 -> f32, 'g' grouped fp8w
    int N, K, G;
  };
  const Shape shapes[] = {
      {"wq_a", 'w', 1024, 4096, 0},    {"wkv", 'w', 512, 4096, 0},     {"wq_b", 'w', 32768, 1024, 0},
      {"wo_b", 'w', 4096, 8192, 0},    {"sw1", 'w', 2048, 4096, 0},    {"sw2", 'w', 4096, 2048, 0},
      {"idx_wq_b", 'w', 8192, 1024, 0}, {"comp_wkv", 'f', 1024, 4096, 0}, {"gate", 'f', 256, 4096, 0},
      {"head", 'f', 129280, 4096, 0},  {"wo_a", 'g', 1024, 4096, 8},  {"idx_wproj", 'b', 64, 4096, 0},
  };
  {
    rt::DeviceBuffer bad(dev, 4);
    UH_CHECK(hipMemset(bad.data(), 0, 4));
    fp8_decode_kernel<<<1, 256, 0, st>>>(bad.as<uint32_t>());
    uint32_t nbad = 0;
    UH_CHECK(hipMemcpy(&nbad, bad.data(), 4, hipMemcpyDeviceToHost));
    std::printf("e4m3 decode: %u of 254 finite codes differ from the bit decode\n", nbad / 4);
    if (nbad) return 1;
  }
  {  // KDA's f_b / g_b: weights streamed from DRAM (copies past the last-level cache)
    const int N = 8192, K = 128, copies = 40;
    const size_t wsize = size_t(N) * K * 2;
    rt::DeviceBuffer w(dev, 2 * wsize * copies), a = upload(dev, bf16v(size_t(8) * K)), c(dev, size_t(8) * N * 4);
    {
      const auto hw = bf16v(size_t(N) * K);
      for (int i = 0; i < 2 * copies; ++i)
        UH_CHECK(hipMemcpy(w.as<uint16_t>() + size_t(i) * N * K, hw.data(), wsize, hipMemcpyHostToDevice));
    }
    for (const int M : {1, 2}) {
      int cur = 0;
      const auto run = [&] {
        const uint16_t* w0 = w.as<uint16_t>() + size_t(2 * cur) * N * K;
        cur = (cur + 1) % copies;
        k::gemv_bf16_smallk2(a.as<uint16_t>(), K, w0, c.as<uint16_t>(), N, N, a.as<uint16_t>(), K, w0 + size_t(N) * K,
                             c.as<uint16_t>() + size_t(4) * N, N, N, M, K, st);
      };
      hipEvent_t e0, e1;
      UH_CHECK(hipEventCreate(&e0));
      UH_CHECK(hipEventCreate(&e1));
      for (int i = 0; i < 3; ++i) run();
      UH_CHECK(hipEventRecord(e0, st));
      for (int i = 0; i < reps; ++i) run();
      UH_CHECK(hipEventRecord(e1, st));
      UH_CHECK(hipEventSynchronize(e1));
      float ms = 0;
      UH_CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double us = 1e3 * ms / reps;
      std::printf("kda f_b+g_b M=%d N=2x%d K=%d: %6.1f us %4.0f GB/s\n", M, N, K, us, 2.0 * wsize / us * 1e-3);
      UH_CHECK(hipEventDestroy(e0));
      UH_CHECK(hipEventDestroy(e1));
    }
  }
  if (only_smallk) return 0;
  bool all_same = true;
  for (const Shape& s : shapes) {
    const int G = s.kind == 'g' ? s.G : 1, N = s.N, K = s.K;
    const size_t cols = size_t(G) * N;
    const bool f8w = s.kind == 'w' || s.kind == 'g';
    // Enough weight copies (>= 256 MB) that the timed loop streams them from DRAM, as a decode step does,
    // instead of from the 64 MB last-level cache.
    const size_t wsize = cols * K * (f8w ? 1 : 2);
    const int copies = int(std::max<size_t>(1, ((size_t(256) << 20) + wsize - 1) / wsize));
    rt::DeviceBuffer w(dev, wsize * copies);
    {
      const auto host = f8w ? fp8(cols * K) : std::vector<uint8_t>();
      const auto hostb = f8w ? std::vector<uint16_t>() : bf16v(cols * K);
      for (int i = 0; i < copies; ++i)
        UH_CHECK(hipMemcpy(w.as<uint8_t>() + size_t(i) * wsize, f8w ? static_cast<const void*>(host.data()) : hostb.data(),
                           wsize, hipMemcpyHostToDevice));
    }
    int cur = 0;
    rt::DeviceBuffer sw = upload(dev, e8m0((cols + 127) / 128 * (K / 128)));
    rt::DeviceBuffer a8 = upload(dev, fp8(size_t(64) * K)), sa = upload(dev, e8m0(size_t(64) * (K / 128)));
    rt::DeviceBuffer ab = upload(dev, bf16v(size_t(64) * G * K));
    rt::DeviceBuffer out(dev, size_t(64) * cols * 4);
    const double wbytes = double(cols) * K * (f8w ? 1 : 2);
    for (const int M : {1, 2, 4, 8, 9, 12, 16, 24, 32, 40, 48, 64}) {
      const auto run = [&]() {
        const uint8_t* wp = w.as<uint8_t>() + size_t(cur) * wsize;
        cur = (cur + 1) % copies;
        switch (s.kind) {
          case 'w': k::gemv_w8a8(a8.as<uint8_t>(), sa.as<uint8_t>(), wp, sw.as<uint8_t>(), out.as<uint16_t>(), N, M, N, K, st); break;
          case 'b': k::gemv_bf16(ab.as<uint16_t>(), K, reinterpret_cast<const uint16_t*>(wp), out.as<uint16_t>(), N, M, N, K, st); break;
          case 'f': k::gemv_bf16_f32out(ab.as<uint16_t>(), K, reinterpret_cast<const uint16_t*>(wp), out.as<float>(), N, M, N, K, st); break;
          case 'g': k::gemv_grouped_fp8w(ab.as<uint16_t>(), wp, sw.as<uint8_t>(), out.as<uint16_t>(), M, G, N, K, st); break;
        }
      };
      const auto time = [&](const auto& f) {
        hipEvent_t e0, e1;
        UH_CHECK(hipEventCreate(&e0));
        UH_CHECK(hipEventCreate(&e1));
        for (int i = 0; i < 3; ++i) f();
        UH_CHECK(hipEventRecord(e0, st));
        for (int i = 0; i < reps; ++i) f();
        UH_CHECK(hipEventRecord(e1, st));
        UH_CHECK(hipEventSynchronize(e1));
        float ms = 0;
        UH_CHECK(hipEventElapsedTime(&ms, e0, e1));
        UH_CHECK(hipEventDestroy(e0));
        UH_CHECK(hipEventDestroy(e1));
        return 1e3 * ms / reps;
      };
      const auto wmma = [&]() {
        const uint8_t* wp = w.as<uint8_t>() + size_t(cur) * wsize;
        cur = (cur + 1) % copies;
        switch (s.kind) {
          case 'w': k::gemm_w8a8_wmma(a8.as<uint8_t>(), sa.as<uint8_t>(), wp, sw.as<uint8_t>(), out.as<uint16_t>(), N, M, N, K, st); break;
          case 'b': k::gemm_bf16_wmma(ab.as<uint16_t>(), K, reinterpret_cast<const uint16_t*>(wp), out.as<uint16_t>(), N, M, N, K, st); break;
          case 'f': k::gemm_bf16_f32out_wmma(ab.as<uint16_t>(), K, reinterpret_cast<const uint16_t*>(wp), out.as<float>(), N, M, N, K, st); break;
          case 'g': k::gemm_grouped_fp8w_wmma(ab.as<uint16_t>(), wp, sw.as<uint8_t>(), out.as<uint16_t>(), M, G, N, K, st); break;
        }
      };
      {  // the 64 x 128 tiles vs the skinny kernels (M <= 64): bit for bit, and time
        std::vector<uint8_t> o[2];
        double us[2];
        for (int sk = 0; sk < 2; ++sk) {
          k::gemm_force_skinny(sk);
          UH_CHECK(hipMemsetAsync(out.data(), 0, out.size(), st));
          wmma();
          UH_CHECK(hipStreamSynchronize(st));
          o[sk] = download(out);
          us[sk] = time(wmma);
        }
        k::gemm_force_skinny(-1);
        const bool same = o[0] == o[1];
        all_same &= same;
        std::printf("%-9s M=%2d N=%6zu K=%5d: WMMA tiled %7.1f us %4.0f GB/s, skinny %7.1f us %4.0f GB/s%s |", s.name,
                    M, cols, K, us[0], wbytes / us[0] * 1e-3, us[1], wbytes / us[1] * 1e-3,
                    same ? "" : " DIFFERS");
      }
      if (M > 32) {
        std::printf("\n");
        continue;
      }
      std::vector<uint8_t> ref;
      for (const int C : M > 8 ? std::vector<int>{1} : std::vector<int>{1, 2, 4}) {
        k::gemv_force_cols(C);
        UH_CHECK(hipMemsetAsync(out.data(), 0, out.size(), st));
        run();
        UH_CHECK(hipStreamSynchronize(st));
        const auto o = download(out);
        if (ref.empty()) ref = o;
        const bool same = o == ref;
        all_same &= same;
        hipEvent_t e0, e1;
        UH_CHECK(hipEventCreate(&e0));
        UH_CHECK(hipEventCreate(&e1));
        for (int i = 0; i < 3; ++i) run();
        UH_CHECK(hipEventRecord(e0, st));
        for (int i = 0; i < reps; ++i) run();
        UH_CHECK(hipEventRecord(e1, st));
        UH_CHECK(hipEventSynchronize(e1));
        float ms = 0;
        UH_CHECK(hipEventElapsedTime(&ms, e0, e1));
        const double us = 1e3 * ms / reps;
        std::printf("  C=%d %6.1f us %4.0f GB/s%s", C, us, wbytes / us * 1e-3, same ? "" : " DIFFERS");
        UH_CHECK(hipEventDestroy(e0));
        UH_CHECK(hipEventDestroy(e1));
      }
      k::gemv_force_cols(0);
      std::printf("\n");
    }
  }
  std::printf(all_same ? "outputs identical for every column count and GEMM kernel\n" : "OUTPUTS DIFFER\n");
  return all_same ? 0 : 1;
}
