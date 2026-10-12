// unholy-curve: prefill and decode rates as a function of context depth, one sequence.
//
// Prefills --ctx tokens in --chunk-token steps (up to the model's in-flight depth submitted ahead) and
// prints each chunk's marginal rate: chunk tokens / (its completion time - the previous chunk's). At
// each --probe depth it drains the pipeline and times --decode-steps teacher-forced decode steps that
// continue the sequence, then resumes the prefill. Token ids cycle through --ids (raw int32 file).
//
//   unholy-curve --model DIR [--exl3 DIR] --profile FILE --ids FILE [--ctx 524288] [--chunk 4096]
//                [--probes 4096,32768,131072,524288] [--decode-steps 8] [--every 8]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "engine/plan.hpp"
#include "engine/resident.hpp"
#include "model/dsv4.hpp"
#include "model/safetensors.hpp"

using namespace unholy;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  std::string model_dir = "/scratch/dsv4-ve", exl3_dir, profile, ids_path;
  std::vector<int> probes = {4096, 32768, 131072, 524288};
  int ctx = 524288, chunk = 4096, dsteps = 8, every = 8;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model_dir = v;
    else if (a == "--exl3") exl3_dir = v;
    else if (a == "--profile") profile = v;
    else if (a == "--ids") ids_path = v;
    else if (a == "--ctx") ctx = std::stoi(v);
    else if (a == "--chunk") chunk = std::stoi(v);
    else if (a == "--decode-steps") dsteps = std::stoi(v);
    else if (a == "--every") every = std::stoi(v);  // print every N-th chunk (and all probes)
    else if (a == "--probes") {
      probes.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) probes.push_back(std::stoi(t));
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
    const int extra = int(probes.size()) * dsteps + 16;
    const int max_seq = (ctx + extra + model::kPageTokens - 1) / model::kPageTokens * model::kPageTokens;
    model::CacheGeometry geo{1, max_seq / model::kPageTokens, max_seq / model::kPageTokens};
    engine::PlanInput pin;
    pin.cfg = &c;
    pin.dense = ck.get();
    pin.exl3 = xck.get();
    pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
    pin.geo = geo;
    pin.max_tokens = chunk;
    pin.max_seq = max_seq;
    pin.max_logit_rows = 8;
    const auto plan = engine::make_plan(pin);
    std::printf("%s", plan.summary().c_str());
    const auto t_load = clk::now();
    engine::ResidentModel m(c, *ck, xck.get(), plan, geo, chunk, max_seq, 8);
    std::printf("experts %s; loaded in %.1f s; in-flight depth %d\n", xck ? "EXL3" : "native MXFP4",
                std::chrono::duration<double>(clk::now() - t_load).count(), m.max_inflight());
    std::fflush(stdout);
    std::vector<int32_t> tables(static_cast<size_t>(geo.max_pages_per_seq));
    for (size_t i = 0; i < tables.size(); ++i) tables[i] = int32_t(i);
    m.set_block_tables(tables);

    std::vector<int32_t> pool;
    {
      std::ifstream f(ids_path, std::ios::binary);
      int32_t x;
      while (f.read(reinterpret_cast<char*>(&x), 4)) pool.push_back(x);
      if (pool.empty()) throw std::runtime_error("no token ids in " + ids_path);
    }
    std::vector<float> logits;
    std::deque<std::pair<uint64_t, int>> flying;  // (ticket, chunk end depth)
    int pos = 0, chunks_done = 0, depth_done = 0;
    clk::time_point last = clk::now();
    size_t next_probe = 0;
    const auto t_start = last;
    double prefill_s = 0;
    int prefill_tokens = 0;
    std::printf("%10s %10s %12s\n", "depth", "chunk_ms", "prefill_tok/s");
    const auto complete = [&]() {
      m.wait(flying.front().first, logits);
      const auto now = clk::now();
      const double s = std::chrono::duration<double>(now - last).count();
      const int end = flying.front().second, n = end - depth_done;
      prefill_s += s;
      prefill_tokens += n;
      last = now;
      depth_done = end;
      flying.pop_front();
      if (++chunks_done % every == 0 || end >= ctx)
        std::printf("%10d %10.0f %12.0f\n", end, s * 1e3, n / s), std::fflush(stdout);
    };
    while (pos < ctx) {
      const int n = std::min(chunk, ctx - pos);
      engine::StepInput in;
      in.chunks = {{0, pos, n, 0}};
      for (int k = 0; k < n; ++k) {
        in.ids.push_back(pool[size_t(pos + k) % pool.size()]);
        in.pos.push_back(pos + k);
      }
      in.logit_rows = {n - 1};
      if (int(flying.size()) >= m.max_inflight()) complete();
      flying.push_back({m.submit(in), pos + n});
      pos += n;
      if (next_probe < probes.size() && pos >= probes[next_probe]) {
        while (!flying.empty()) complete();
        std::vector<double> ts;
        for (int st = 0; st < dsteps; ++st) {
          engine::StepInput d;
          d.chunks = {{0, pos, 1, 0}};
          d.ids = {pool[size_t(pos) % pool.size()]};
          d.pos = {pos};
          d.logit_rows = {0};
          const auto t0 = clk::now();
          m.step(d, logits);
          ts.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
          ++pos;
        }
        std::sort(ts.begin(), ts.end());
        std::printf("decode at depth %d: median %.1f ms/token (%.1f tok/s), min %.1f, max %.1f\n", pos,
                    ts[ts.size() / 2], 1e3 / ts[ts.size() / 2], ts.front(), ts.back());
        std::fflush(stdout);
        depth_done = pos;
        last = clk::now();
        ++next_probe;
      }
    }
    while (!flying.empty()) complete();
    std::printf("prefill %d tokens in %.1f s: %.0f tok/s average (wall %.1f s incl. decode probes)\n", prefill_tokens,
                prefill_s, prefill_tokens / prefill_s, std::chrono::duration<double>(clk::now() - t_start).count());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
