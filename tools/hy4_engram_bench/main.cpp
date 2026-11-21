// unholy-hy4-engram-bench: the host side of Hy4 flash's Engram (model::Hy4Engram::lookup_all), the random reads of the
// SSD pack a step does before its first kernel. CPU and disk only (no GPU).
//
//   unholy-hy4-engram-bench --model DIR --engram DIR --ids FILE [--tokens T] [--steps N] [--offset K]
//
// The n-grams come from consecutive positions of --ids (uint32 token ids), starting at --offset; each step looks up
// T positions (1: a decode step, 6: a DFly2 verify step). --random SEED: uniformly random token ids instead (rows the
// page cache has not seen: the reads go to the SSD). Reports the time per step (mean, median, p90, max).
// UNHOLY_HY4_ENGRAM_THREADS sets the reader count.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "model/hy4.hpp"
#include "model/safetensors.hpp"

using namespace unholy;

int main(int argc, char** argv) {
  std::string model_dir = "/scratch/models/hy4-flash-fp8", engram = "/scratch/models/hy4-flash-engram", ids_path;
  int T = 1, steps = 200;
  long offset = 0;
  long seed = -1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model_dir = v;
    else if (a == "--engram") engram = v;
    else if (a == "--ids") ids_path = v;
    else if (a == "--tokens") T = std::stoi(v);
    else if (a == "--steps") steps = std::stoi(v);
    else if (a == "--offset") offset = std::stol(v);
    else if (a == "--random") seed = std::stol(v);
    else {
      std::fprintf(stderr, "usage: %s --model DIR --engram DIR (--ids FILE | --random SEED) [--tokens T] [--steps N] [--offset K]\n",
                   argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(model_dir);
    const auto c = model::Hy4Config::from_hf(ck->config());
    model::Hy4Engram eng(engram, c);
    std::vector<uint32_t> ids;
    if (seed >= 0) {
      std::mt19937_64 rng(static_cast<uint64_t>(seed));
      ids.resize(size_t(offset) + size_t(steps) * T + 2);
      for (auto& v : ids) v = uint32_t(rng() % uint64_t(c.vocab));
    } else {
      std::ifstream f(ids_path, std::ios::binary);
      f.seekg(0, std::ios::end);
      ids.resize(size_t(f.tellg()) / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(ids.data()), std::streamsize(ids.size() * 4));
    }
    if (ids.size() < size_t(offset) + size_t(steps) * T + 2) throw std::runtime_error("ids file too short");
    const int M = eng.memories();
    std::vector<std::vector<uint8_t>> q(size_t(M), std::vector<uint8_t>(size_t(T) * 16 * 96));
    std::vector<std::vector<uint16_t>> s(size_t(M), std::vector<uint16_t>(size_t(T) * 16));
    std::vector<uint8_t*> qp;
    std::vector<uint16_t*> sp;
    for (int m = 0; m < M; ++m) qp.push_back(q[size_t(m)].data()), sp.push_back(s[size_t(m)].data());
    std::vector<uint8_t> mask(size_t(T), 1);
    std::vector<int32_t> ngram(size_t(T) * 3);
    std::vector<double> ms;
    for (int k = 0; k < steps; ++k) {
      const size_t p0 = size_t(offset) + 2 + size_t(k) * T;
      for (int t = 0; t < T; ++t)
        for (int j = 0; j < 3; ++j) ngram[size_t(t) * 3 + j] = int32_t(ids[p0 + t - j]);
      const auto t0 = std::chrono::steady_clock::now();
      eng.lookup_all(ngram.data(), mask.data(), T, qp.data(), sp.data());
      ms.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3);
    }
    double sum = 0;
    for (double v : ms) sum += v;
    std::sort(ms.begin(), ms.end());
    std::printf("%d tokens per step, %d steps: mean %.3f ms  median %.3f  p90 %.3f  max %.3f\n", T, steps,
                sum / steps, ms[ms.size() / 2], ms[ms.size() * 9 / 10], ms.back());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
