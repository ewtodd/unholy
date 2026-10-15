// unholy-attn-test: the WMMA sparse attention (attn_wmma.hpp) against the reference kernel on decode-sized
// calls over a paged pool: window-only, CSA (top-512 of the compressed rows) and HCA (every compressed row)
// at several depths. Prints rel L2 vs the reference (the WMMA kernel blocks its softmax by 16 keys, the
// reference by 64: noise-level) and the time per call.
//
//   unholy-attn-test [--dev 0] [--reps 20] [--T 1]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/attn_wmma.hpp"
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
template <typename T>
rt::DeviceBuffer upload(int dev, const std::vector<T>& v) {
  rt::DeviceBuffer b(dev, std::max<size_t>(v.size() * sizeof(T), 4));
  UH_CHECK(hipMemcpy(b.data(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  int dev = 0, reps = 20, T = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--dev") dev = std::stoi(argv[i + 1]);
    else if (a == "--reps") reps = std::stoi(argv[i + 1]);
    else if (a == "--T") T = std::stoi(argv[i + 1]);
  }
  rt::DeviceGuard g(dev);
  hipStream_t st;
  UH_CHECK(hipStreamCreate(&st));
  constexpr int H = 64, D = 512, PT = 256;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const auto randv = [&](size_t n, float s) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = to_bf16(nd(rng) * s);
    return v;
  };
  struct Case {
    const char* name;
    int ratio, depth;
    bool topk;
  };
  const Case cases[] = {{"window", 0, 4096, false}, {"csa 4k", 4, 4096, true},       {"csa 512k", 4, 524288, true},
                        {"hca 4k", 128, 4096, false}, {"hca 128k", 128, 131072, false}, {"hca 512k", 128, 524288, false}};
  const float scale = 1.0f / std::sqrt(float(D));
  bool all_same = true;
  for (const Case& c : cases) {
    const int depth = c.depth, s0 = depth - T;  // the call feeds positions [s0, depth)
    const int rpp = c.ratio ? PT / c.ratio : 1, nrows = c.ratio ? depth / c.ratio : 0;
    const int pages = std::max(1, (nrows + rpp - 1) / rpp);
    std::vector<int32_t> table(static_cast<size_t>(pages));
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    auto pool = upload(dev, randv(size_t(pages) * rpp * D, 1.0f));
    auto ring = upload(dev, randv(size_t(128) * D, 1.0f));
    auto ckv = upload(dev, randv(size_t(T) * D, 1.0f));
    auto q = upload(dev, randv(size_t(T) * H * D, 0.3f));
    std::vector<float> sinkv(H);
    for (auto& x : sinkv) x = nd(rng);
    auto sink = upload(dev, sinkv);
    std::vector<int32_t> posv(static_cast<size_t>(T));
    std::iota(posv.begin(), posv.end(), s0);
    auto pos = upload(dev, posv);
    auto tab = upload(dev, table);
    const int ncomp = c.topk ? std::min(512, nrows) : 0;
    std::vector<int32_t> idxv(size_t(T) * std::max(ncomp, 1));
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < ncomp; ++i) idxv[size_t(t) * ncomp + i] = int32_t(rng() % uint32_t((s0 + t + 1) / 4));
    auto idx = upload(dev, idxv);
    k::AttnKv kv;
    kv.chunk_kv = ckv.as<uint16_t>();
    kv.ring = ring.as<uint16_t>();
    kv.s0 = s0;
    kv.ratio = c.ratio;
    if (c.ratio) kv.comp = k::PagedRows{pool.as<uint16_t>(), tab.as<int32_t>(), rpp, D};
    if (c.topk) {
      kv.comp_idx = idx.as<int32_t>();
      kv.ncomp = ncomp;
    }
    rt::DeviceBuffer o_ref(dev, size_t(T) * H * D * 2), o_fast(dev, size_t(T) * H * D * 2);
    k::sparse_attn(q.as<uint16_t>(), pos.as<int32_t>(), kv, sink.as<float>(), o_ref.as<uint16_t>(), T, H, D, scale, st);
    k::sparse_attn_wmma(q.as<uint16_t>(), pos.as<int32_t>(), kv, sink.as<float>(), o_fast.as<uint16_t>(), T, H, D, scale, st);
    UH_CHECK(hipStreamSynchronize(st));
    std::vector<uint16_t> a(size_t(T) * H * D), b(a.size());
    UH_CHECK(hipMemcpy(a.data(), o_ref.data(), a.size() * 2, hipMemcpyDeviceToHost));
    UH_CHECK(hipMemcpy(b.data(), o_fast.data(), b.size() * 2, hipMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double x = from_bf16(a[i]), y = from_bf16(b[i]);
      num += (x - y) * (x - y);
      den += x * x;
    }
    const int keys = 128 + (c.topk ? ncomp : nrows);
    std::printf("%-9s T=%d keys %6d: rel L2 vs reference %.2e;", c.name, T, keys, std::sqrt(num / std::max(den, 1e-30)));
    {
      hipEvent_t e0, e1;
      UH_CHECK(hipEventCreate(&e0));
      UH_CHECK(hipEventCreate(&e1));
      UH_CHECK(hipEventRecord(e0, st));
      for (int i = 0; i < reps; ++i)
        k::sparse_attn_wmma(q.as<uint16_t>(), pos.as<int32_t>(), kv, sink.as<float>(), o_fast.as<uint16_t>(), T, H, D,
                            scale, st);
      UH_CHECK(hipEventRecord(e1, st));
      UH_CHECK(hipEventSynchronize(e1));
      float ms = 0;
      UH_CHECK(hipEventElapsedTime(&ms, e0, e1));
      std::printf(" fused %7.1f us", 1e3 * ms / reps);
    }
    {  // the split path: bit for bit the fused kernel's output, and its time
      const int max_total = keys;
      const size_t s_floats = size_t(T) * H * ((max_total + 15) / 16 * 16);
      rt::DeviceBuffer sbuf(dev, s_floats * 4), o_split(dev, size_t(T) * H * D * 2);
      const auto run = [&]() {
        if (!k::sparse_attn_wmma_split(q.as<uint16_t>(), pos.as<int32_t>(), kv, sink.as<float>(), o_split.as<uint16_t>(), T,
                                       scale, max_total, sbuf.as<float>(), s_floats, st))
          throw std::runtime_error("split refused");
      };
      run();
      UH_CHECK(hipStreamSynchronize(st));
      std::vector<uint16_t> c2(b.size());
      UH_CHECK(hipMemcpy(c2.data(), o_split.data(), c2.size() * 2, hipMemcpyDeviceToHost));
      size_t same = 0;
      for (size_t i = 0; i < c2.size(); ++i) same += c2[i] == b[i];
      hipEvent_t e0, e1;
      UH_CHECK(hipEventCreate(&e0));
      UH_CHECK(hipEventCreate(&e1));
      UH_CHECK(hipEventRecord(e0, st));
      for (int i = 0; i < reps; ++i) run();
      UH_CHECK(hipEventRecord(e1, st));
      UH_CHECK(hipEventSynchronize(e1));
      float ms = 0;
      UH_CHECK(hipEventElapsedTime(&ms, e0, e1));
      std::printf(", split %7.1f us (%zu / %zu outputs identical to fused)", 1e3 * ms / reps, same, c2.size());
      all_same &= same == c2.size();
    }
    std::printf("\n");
  }
  std::printf(all_same ? "split path bit-identical to the fused kernel\n" : "SPLIT PATH DIFFERS\n");
  return all_same ? 0 : 1;
}
