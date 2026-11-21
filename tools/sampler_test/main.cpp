// unholy-sampler-test: the host sampler (engine/sampler.hpp). CPU only.
//
//   unholy-sampler-test
//
// 1. Sampler::exp_nonpos against exp in double over [-87, 0] (relative error).
// 2. weights_pass on a 120,832-token row shaped like a model's (a few peaks over a spread) at temperature 0.9: the
//    total against double precision.
// 3. Statistics on a small distribution: sample() frequencies and verify()'s acceptance rate and output frequencies
//    against the exact softmax (deviations in standard errors).
// 4. Time per sample() on the 120,832-token row: the whole-vocabulary path and the general (filtered) one.
// Exits 1 when a check fails.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/sampler.hpp"

using namespace unholy::engine;

int main() {
  int failed = 0;
  {  // 1
    double worst = 0;
    for (int i = 0; i <= 2000000; ++i) {
      const float x = -87.0f * float(i) / 2000000.0f;
      const double ref = std::exp(double(x)), got = double(Sampler::exp_nonpos(x));
      worst = std::max(worst, std::fabs(got - ref) / ref);
    }
    std::printf("exp_nonpos: max relative error %.3e over [-87, 0]\n", worst);
    if (worst > 1e-6) ++failed;
  }
  const int V = 120832;
  std::vector<float> row(V);
  {
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (auto& v : row) v = nd(rng);
    row[1234] = 24.0f, row[777] = 22.5f, row[99999] = 21.0f, row[5] = 18.0f;
  }
  {  // 2
    const float m = *std::max_element(row.begin(), row.end());
    std::vector<float> w(V);
    std::vector<double> blocks((V + Sampler::kWeightBlock - 1) / Sampler::kWeightBlock);
    const double got = Sampler::weights_pass(row.data(), V, m, float(1.0 / 0.9), w.data(), blocks.data());
    double ref = 0;
    for (float v : row) ref += std::exp((double(v) - m) / 0.9);
    std::printf("weights_pass: total %.9g vs %.9g in double, relative error %.3e\n", got, ref, std::fabs(got - ref) / ref);
    if (std::fabs(got - ref) / ref > 1e-6) ++failed;
  }
  {  // 3
    const std::vector<float> l = {2.0f, 1.0f, 0.5f, -1.0f, 3.0f, 0.0f};
    const int n = int(l.size());
    const double T = 0.9;
    std::vector<double> p(static_cast<size_t>(n));
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[size_t(i)] = std::exp(double(l[size_t(i)]) / T);
    for (auto& v : p) v /= z;
    SamplingParams sp;
    sp.temperature = float(T);
    Sampler s(sp, 42);
    const int N = 2000000;
    std::vector<int> cnt(size_t(n), 0), cv(size_t(n), 0);
    int acc = 0;
    const int draft = 0;
    for (int k = 0; k < N; ++k) ++cnt[size_t(s.sample(l.data(), n).token)];
    for (int k = 0; k < N; ++k) {
      bool a = false;
      const auto r = s.verify(l.data(), n, draft, a);
      acc += a;
      ++cv[size_t(r.token)];
    }
    double worst = 0, worst_v = 0;
    for (int i = 0; i < n; ++i) {
      const double se = std::sqrt(p[size_t(i)] * (1 - p[size_t(i)]) / N);
      worst = std::max(worst, std::fabs(cnt[size_t(i)] / double(N) - p[size_t(i)]) / se);
      worst_v = std::max(worst_v, std::fabs(cv[size_t(i)] / double(N) - p[size_t(i)]) / se);
    }
    const double pa = p[size_t(draft)], sea = std::sqrt(pa * (1 - pa) / N);
    const double dev_a = std::fabs(acc / double(N) - pa) / sea;
    std::printf("sample(): worst deviation %.2f standard errors; verify(): output %.2f, acceptance %.4f vs %.4f "
                "(%.2f)\n", worst, worst_v, acc / double(N), pa, dev_a);
    if (worst > 5 || worst_v > 5 || dev_a > 5) ++failed;
  }
  {  // 4
    for (int filtered = 0; filtered < 2; ++filtered) {
      SamplingParams sp;
      sp.temperature = 0.9f;
      if (filtered) sp.min_p = 1e-30f;  // the general path, every token a candidate
      Sampler s(sp, 1);
      const int N = 300;
      long sink = 0;
      const auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < N; ++k) sink += s.sample(row.data(), V).token;
      const double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6 / N;
      std::printf("sample() on %d tokens, %s: %.0f us (token sum %ld)\n", V,
                  filtered ? "general path (min_p)" : "whole vocabulary", us, sink);
    }
  }
  std::printf(failed ? "FAILED (%d)\n" : "ok\n", failed);
  return failed ? 1 : 0;
}
