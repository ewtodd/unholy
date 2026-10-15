// unholy-indexer-test: the fp4 indexer fast path (quant_q_fp4 + indexer_scores_fp4 + topk_bf16) against
// the reference kernels (fake_quant_fp4 + indexer_scores + topk_rows) on random data, through the same
// paged fp4 key pool (store_rows_fp4). Scores and selected indices must match exactly.
//
//   unholy-indexer-test [--T 37] [--keys 5000] [--topk 512] [--seed 1] [--dev 0]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kernels/indexer.hpp"
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
  int T = 37, nkeys = 5000, topk = 512, dev = 0;
  uint64_t seed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = std::atoi(argv[i + 1]);
    if (a == "--T") T = v;
    else if (a == "--keys") nkeys = v;
    else if (a == "--topk") topk = v;
    else if (a == "--seed") seed = uint64_t(v);
    else if (a == "--dev") dev = v;
  }
  const int H = 64, D = 128;
  UH_CHECK(hipSetDevice(dev));
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  // Keys with per-row magnitudes spread over a few octaves (exercises the scales).
  std::vector<uint16_t> hk(size_t(nkeys) * D), hq(size_t(T) * H * D), hw(size_t(T) * H);
  for (int r = 0; r < nkeys; ++r) {
    const float m = std::ldexp(1.0f, int(rng() % 7) - 3);
    for (int c = 0; c < D; ++c) hk[size_t(r) * D + c] = to_bf16(nd(rng) * m);
  }
  for (auto& v : hq) v = to_bf16(nd(rng) * 0.5f);
  for (auto& v : hw) v = to_bf16(nd(rng) * 0.1f);
  // Positions: the chunk ends at the last key, earlier queries see fewer keys.
  std::vector<int32_t> hpos(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) hpos[size_t(t)] = std::max(0, nkeys * 4 - 1 - (T - 1 - t) * 37);
  const int pages = (nkeys + k::kFp4RowsPerPage - 1) / k::kFp4RowsPerPage;
  std::vector<int32_t> table(static_cast<size_t>(pages));
  std::iota(table.begin(), table.end(), 0);
  std::shuffle(table.begin(), table.end(), rng);  // non-identity block table

  rt::DeviceBuffer dk(dev, hk.size() * 2), dq(dev, hq.size() * 2), dq2(dev, hq.size() * 2), dw(dev, hw.size() * 2),
      dpos(dev, size_t(T) * 4), dtab(dev, size_t(pages) * 4), pool(dev, size_t(pages) * k::kFp4PageBytes),
      qc(dev, size_t(T) * H * D), qs(dev, size_t(T) * H), sref(dev, size_t(T) * nkeys * 4),
      sfast(dev, size_t(T) * nkeys * 2), iref(dev, size_t(T) * topk * 4), ifast(dev, size_t(T) * topk * 4);
  UH_CHECK(hipMemcpy(dk.data(), hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dq.data(), hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dq2.data(), hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dw.data(), hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dpos.data(), hpos.data(), hpos.size() * 4, hipMemcpyHostToDevice));
  UH_CHECK(hipMemcpy(dtab.data(), table.data(), table.size() * 4, hipMemcpyHostToDevice));
  const k::PagedFp4 keys{pool.as<uint8_t>(), dtab.as<int32_t>()};
  k::store_rows_fp4(dk.as<uint16_t>(), nkeys, keys, 0, nullptr);

  // Round trip: stored codes decode to exactly fake_quant_fp4's values.
  {
    rt::DeviceBuffer back(dev, hk.size() * 2);
    k::load_rows_fp4(keys, 0, nkeys, back.as<uint16_t>(), nullptr);
    k::fake_quant_fp4(dk.as<uint16_t>(), D, nkeys, D, 32, nullptr);
    std::vector<uint16_t> a(hk.size()), b(hk.size());
    UH_CHECK(hipMemcpy(a.data(), back.data(), a.size() * 2, hipMemcpyDeviceToHost));
    UH_CHECK(hipMemcpy(b.data(), dk.data(), b.size() * 2, hipMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += (a[i] != b[i]) && !((a[i] | b[i]) == 0x8000);
    std::printf("fp4 store/load vs fake_quant_fp4: %zu / %zu differ\n", diff, a.size());
  }

  // Reference: fake-quantized bf16 q, fp32 scores, float top-k.
  k::fake_quant_fp4(dq.as<uint16_t>(), H * D, T, H * D, 32, nullptr);
  k::indexer_scores(dq.as<uint16_t>(), dw.as<uint16_t>(), keys, dpos.as<int32_t>(), sref.as<float>(), T, H, D, nkeys,
                    4, nullptr);
  k::topk_rows(sref.as<float>(), T, nkeys, topk, iref.as<int32_t>(), nullptr);
  // Fast path.
  k::quant_q_fp4(dq2.as<uint16_t>(), T, H, qc.as<uint8_t>(), qs.as<uint8_t>(), nullptr);
  UH_CHECK(hipDeviceSynchronize());
  const auto t0 = std::chrono::steady_clock::now();
  k::indexer_scores_fp4(qc.as<uint8_t>(), qs.as<uint8_t>(), dw.as<uint16_t>(), keys, dpos.as<int32_t>(),
                        sfast.as<uint16_t>(), T, H, nkeys, 4, nullptr);
  UH_CHECK(hipDeviceSynchronize());
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  // Top-k as the model runs it: one group of T rows, with scratch (several workgroups per row for a few
  // long rows, k::kTopkSplitMinKeys).
  k::TopkBatch tb;
  tb.ngroups = 1;
  tb.g[0] = {0, T, nkeys, topk, 0};
  const size_t tk_words = k::topk_split_scratch_words(std::min(T, k::kTopkSplitMaxRows));
  rt::DeviceBuffer tk_scr(dev, tk_words * 4);
  const auto run_topk = [&]() {
    k::topk_bf16_batch(sfast.as<uint16_t>(), tb, ifast.as<int32_t>(), nullptr, tk_scr.as<uint32_t>(), tk_words);
  };
  run_topk();
  UH_CHECK(hipDeviceSynchronize());
  double topk_us = 0;
  {
    const int reps = 20;
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) run_topk();
    UH_CHECK(hipDeviceSynchronize());
    topk_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / reps;
  }

  std::vector<float> a(size_t(T) * nkeys);
  std::vector<uint16_t> b(size_t(T) * nkeys);
  std::vector<int32_t> ia(size_t(T) * topk), ib(size_t(T) * topk);
  UH_CHECK(hipMemcpy(a.data(), sref.data(), a.size() * 4, hipMemcpyDeviceToHost));
  UH_CHECK(hipMemcpy(b.data(), sfast.data(), b.size() * 2, hipMemcpyDeviceToHost));
  UH_CHECK(hipMemcpy(ia.data(), iref.data(), ia.size() * 4, hipMemcpyDeviceToHost));
  UH_CHECK(hipMemcpy(ib.data(), ifast.data(), ib.size() * 4, hipMemcpyDeviceToHost));
  size_t sdiff = 0, idiff = 0, finite = 0;
  double maxrel = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const float fb = from_bf16(b[i]);
    if (std::isfinite(a[i])) ++finite;
    if (!(fb == a[i] || (std::isinf(fb) && std::isinf(a[i]) && (fb < 0) == (a[i] < 0)))) {
      ++sdiff;
      if (std::isfinite(a[i]) && a[i] != 0) maxrel = std::max(maxrel, double(std::fabs(fb - a[i]) / std::fabs(a[i])));
    }
  }
  for (size_t i = 0; i < ia.size(); ++i) idiff += ia[i] != ib[i];
  const double gop = 2.0 * T * H * D * double(nkeys) / 1e9;
  std::printf("T=%d keys=%d: scores %zu / %zu differ (max rel %.3g; %zu finite), top-%d indices %zu / %zu differ; "
              "fast scores %.3f ms (%.1f TOPS), top-k %.1f us\n",
              T, nkeys, sdiff, a.size(), maxrel, finite, topk, idiff, ia.size(), ms, gop / ms, topk_us);
  return sdiff || idiff ? 1 : 0;
}
