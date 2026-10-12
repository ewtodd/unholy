// unholy-bench: step timings of the resident model without the server.
//
// For each batch size B: prefill B sequences of --ctx tokens (chunks of --chunk tokens, one sequence
// per step), then time --steps decode steps at batch B. Prints the prefill rate and the decode step
// time / aggregate rate. Token ids are drawn from --ids (a raw int32 file) or are random.
//
//   unholy-bench --model DIR [--exl3 DIR] --profile FILE [--batch 1,2,4,8] [--ctx 1024] [--chunk 1024]
//                [--steps 16] [--ids FILE] [--verify N]
//
// --verify N: every decode step gives each sequence N tokens with N logit rows (a speculative-decoding
// verification step of N - 1 drafts), so the step time is what verifying N - 1 drafts costs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "engine/plan.hpp"
#include "engine/resident.hpp"
#include "model/dsv4.hpp"
#include "model/safetensors.hpp"

using namespace unholy;

int main(int argc, char** argv) {
  std::string model_dir = "/scratch/dsv4-ve", exl3_dir, profile, ids_path, hist_path;
  std::vector<int> batches = {1, 2, 4, 8};
  int ctx = 1024, chunk = 1024, steps = 16, verify = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model_dir = v;
    else if (a == "--exl3") exl3_dir = v;
    else if (a == "--profile") profile = v;
    else if (a == "--ids") ids_path = v;
    else if (a == "--hist") hist_path = v;  // routing histogram of the prefills -> FILE (profile format)
    else if (a == "--ctx") ctx = std::stoi(v);
    else if (a == "--chunk") chunk = std::stoi(v);
    else if (a == "--steps") steps = std::stoi(v);
    else if (a == "--verify") verify = std::stoi(v);
    else if (a == "--batch") {
      batches.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) batches.push_back(std::stoi(t));
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(model_dir);
    std::unique_ptr<model::Checkpoint> xck;
    if (!exl3_dir.empty()) xck = model::Checkpoint::open(exl3_dir);
    const auto c = model::Dsv4Config::from_hf(ck->config());
    int max_b = 1;
    for (int b : batches) max_b = std::max(max_b, b);
    const int max_seq = (ctx + steps * verify + 1 + model::kPageTokens - 1) / model::kPageTokens * model::kPageTokens;
    model::CacheGeometry geo{max_b, max_seq / model::kPageTokens, max_b * (max_seq / model::kPageTokens)};
    const int max_tokens = std::max(chunk, max_b * verify);
    engine::PlanInput pin;
    pin.cfg = &c;
    pin.dense = ck.get();
    pin.exl3 = xck.get();
    pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
    pin.geo = geo;
    pin.max_tokens = max_tokens;
    pin.max_seq = max_seq;
    pin.max_logit_rows = max_b * verify;
    const auto plan = engine::make_plan(pin);
    std::printf("%s", plan.summary().c_str());
    engine::ResidentModel m(c, *ck, xck.get(), plan, geo, max_tokens, max_seq, max_b * verify);
    std::vector<int32_t> tables(size_t(geo.max_slots) * geo.max_pages_per_seq);
    for (size_t i = 0; i < tables.size(); ++i) tables[i] = int32_t(i);
    m.set_block_tables(tables);

    std::vector<int32_t> pool;
    if (!ids_path.empty()) {
      std::ifstream f(ids_path, std::ios::binary);
      int32_t x;
      while (f.read(reinterpret_cast<char*>(&x), 4)) pool.push_back(x);
    }
    std::mt19937 rng(1);
    const auto tok_at = [&](int seq, int p) {
      if (!pool.empty()) return pool[size_t(seq * 7919 + p) % pool.size()];
      return int32_t(1000 + rng() % 100000);
    };
    std::vector<float> logits;
    std::vector<std::vector<uint64_t>> hist;
    if (!hist_path.empty()) model::set_expert_histogram(&hist);
    for (const int B : batches) {
      // Prefill: chunks of a sequence are submitted ahead (up to the model's in-flight depth); stream
      // order keeps each chunk after the one before it on every card.
      const auto tp = std::chrono::steady_clock::now();
      std::deque<uint64_t> flying;
      for (int s = 0; s < B; ++s)
        for (int p0 = 0; p0 < ctx; p0 += chunk) {
          const int n = std::min(chunk, ctx - p0);
          engine::StepInput in;
          in.chunks = {{s, p0, n, 0}};
          for (int k = 0; k < n; ++k) {
            in.ids.push_back(tok_at(s, p0 + k));
            in.pos.push_back(p0 + k);
          }
          if (p0 + n == ctx) in.logit_rows = {n - 1};
          if (int(flying.size()) >= m.max_inflight()) {
            m.wait(flying.front(), logits);
            flying.pop_front();
          }
          flying.push_back(m.submit(in));
        }
      for (; !flying.empty(); flying.pop_front()) m.wait(flying.front(), logits);
      const double pre_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
      if (!hist_path.empty()) {
        model::set_expert_histogram(nullptr);
        std::ofstream f(hist_path);
        for (size_t L = 0; L < hist.size(); ++L) {
          f << "L" << L;
          for (const uint64_t v : hist[L]) f << " " << v;
          f << "\n";
        }
        hist_path.clear();
      }
      // Decode.
      std::vector<double> ts;
      for (int st = 0; st < steps; ++st) {
        engine::StepInput in;
        for (int s = 0; s < B; ++s) {
          const int p0 = ctx + st * verify;
          in.chunks.push_back({s, p0, verify, s * verify});
          for (int k = 0; k < verify; ++k) {
            in.ids.push_back(tok_at(s, p0 + k));
            in.pos.push_back(p0 + k);
            in.logit_rows.push_back(s * verify + k);
          }
        }
        const auto t0 = std::chrono::steady_clock::now();
        m.step(in, logits);
        ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(ts.begin(), ts.end());
      double mean = 0;
      for (size_t i = 1; i < ts.size(); ++i) mean += ts[i];  // drop the fastest as warm-up noise? keep median too
      mean /= double(ts.size() - 1);
      const double med = ts[ts.size() / 2];
      std::printf("B=%d ctx=%d verify=%d: prefill %.0f tok/s; decode step median %.1f ms (mean %.1f) -> %.1f "
                  "steps/s/stream, %.1f steps/s aggregate\n",
                  B, ctx, verify, B * ctx / pre_s, med * 1e3, mean * 1e3, 1.0 / med, B / med);
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
