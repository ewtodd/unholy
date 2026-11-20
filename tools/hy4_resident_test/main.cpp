// unholy-hy4-resident-test: Hy4 flash on the resident path (engine/hy4_resident.hpp: both cards, the APU's cold
// experts, the step interface the scheduler uses) against the oracle (tools/ref/hy4_oracle/run.py): the oracle's tokens
// as prefill steps of --chunk tokens and then one-token decode steps, every position's logits compared by KL and top-1.
//
//   unholy-hy4-resident-test --model DIR --oracle DIR [--engram DIR] [--profile FILE] [--chunk C] [--slots N]
//                            [--card-margin-gb G] [--out logits.bin]
//
// --slots N runs the sequence in slot N - 1 of N (block tables offset by the earlier slots' pages): a slot other than 0.
//
// --spec G: DFly2 instead. The oracle's prompt, then G tokens of greedy decoding twice: one token per step, and with
// drafts (5 per step, verified in one chunk, the rejected ones rolled back); the two continuations should agree (up
// to near-ties flipped by the different step shapes). Reports the accepted drafts per step and both speeds.
// --spec-only plain|draft runs one of the two (for profiles); --spec-only draft-first runs both, drafted first.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine/hy4_resident.hpp"
#include "gufo/json.hpp"
#include "model/hy4.hpp"
#include "model/safetensors.hpp"

using namespace unholy;

namespace {

std::string read_text(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = "/scratch/models/hy4-flash-fp8", engram = "/scratch/models/hy4-flash-engram", oracle,
              profile = "data/expert_profile_hy4_code4k.txt", out_path, experts_gguf;
  int chunk = 512, slots = 1, spec = 0;
  std::string spec_only;
  bool mxfp4 = false;
  double margin = 2.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--model") model_dir = next();
    else if (a == "--oracle") oracle = next();
    else if (a == "--engram") engram = next();
    else if (a == "--profile") profile = next();
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--slots") slots = std::stoi(next());
    else if (a == "--card-margin-gb") margin = std::stod(next());
    else if (a == "--out") out_path = next();
    else if (a == "--mxfp4") mxfp4 = true;
    else if (a == "--experts-gguf") experts_gguf = next();
    else if (a == "--spec") spec = std::stoi(next());
    else if (a == "--spec-only") spec_only = next();
    else {
      std::fprintf(stderr, "usage: %s --model DIR --oracle DIR [--engram DIR] [--profile F] [--chunk C] [--slots N] "
                   "[--card-margin-gb G] [--out F]\n", argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(model_dir);
    std::unique_ptr<model::Checkpoint> gk;
    if (!experts_gguf.empty()) gk = model::Checkpoint::open_gguf({experts_gguf});
    const auto c = model::Hy4Config::from_hf(ck->config());
    const auto meta = gufo::json::parse(read_text(oracle + "/meta.json"));
    const int S = int(meta.member_size("S")), P = int(meta.member_size("prompt"));
    std::vector<int32_t> ids;
    for (const auto& v : meta.find("ids")->items()) ids.push_back(int32_t(v.as_double()));
    const int max_seq = (std::max(S, P + spec + 16) + 255) / 256 * 256, pps = max_seq / 256;
    model::CacheGeometry geo;
    geo.max_slots = slots;
    geo.max_pages_per_seq = pps;
    geo.num_pages = pps * slots;
    engine::Hy4PlanInput pin;
    pin.cfg = &c;
    pin.ck = ck.get();
    pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
    pin.geo = geo;
    pin.max_tokens = chunk;
    pin.max_seq = max_seq;
    pin.max_logit_rows = chunk;
    pin.card_margin = size_t(margin * double(1ull << 30));
    pin.mxfp4 = mxfp4;
    pin.nvfp4 = bool(gk);
    pin.dfly = spec > 0;
    const auto plan = engine::make_hy4_plan(pin);
    std::printf("%s", plan.summary().c_str());
    engine::Hy4ResidentModel m(c, *ck, engram, plan, geo, chunk, max_seq, chunk, 0, mxfp4, gk.get(), spec > 0);
    const int slot = slots - 1;
    std::vector<int32_t> tables(size_t(slots) * pps);
    for (size_t i = 0; i < tables.size(); ++i) tables[i] = int32_t(i);
    m.set_block_tables(tables);

    const int V = c.vocab;
    const auto argmax = [&](const float* l) { return int32_t(std::max_element(l, l + V) - l); };
    if (spec > 0) {
      std::vector<float> out;
      const auto prefill = [&]() {
        for (int s0 = 0; s0 < P; s0 += chunk) {
          const int n = std::min(chunk, P - s0);
          engine::StepInput in;
          in.chunks.push_back({slot, s0, n, 0});
          for (int i = 0; i < n; ++i) in.ids.push_back(ids[size_t(s0 + i)]), in.pos.push_back(s0 + i);
          in.logit_rows.push_back(n - 1);
          m.step(in, out);
        }
        return argmax(out.data());
      };
      // plain greedy decoding
      std::vector<int32_t> plain, gen;
      double t_plain = 0, t_spec = 0, t_draft = 0;
      int steps = 0, accepted = 0;
      std::vector<int> hist(model::kHy4DraftTokens + 1, 0);
      const auto run_plain = [&]() {
        plain = {prefill()};
        const auto ts = std::chrono::steady_clock::now();
        while (int(plain.size()) < spec) {
          engine::StepInput in;
          in.chunks.push_back({slot, P + int(plain.size()) - 1, 1, 0});
          in.ids.push_back(plain.back());
          in.pos.push_back(P + int(plain.size()) - 1);
          in.logit_rows.push_back(0);
          m.step(in, out);
          plain.push_back(argmax(out.data()));
        }
        t_plain = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
      };
      // with drafts
      const auto run_draft = [&]() {
        gen = {prefill()};
        int computed = P;
        std::vector<engine::RingRestore> restores;
        std::vector<int32_t> drafts;
        std::vector<float> conf;
        const auto ts = std::chrono::steady_clock::now();
        while (int(gen.size()) < spec) {
          const auto td = std::chrono::steady_clock::now();
          m.draft(restores, {{slot, computed - 1, gen.back()}}, drafts, conf);
          t_draft += std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
          restores.clear();
          engine::StepInput in;
          const int n = 1 + model::kHy4DraftTokens;
          in.chunks.push_back({slot, computed, n, 0});
          in.ids.push_back(gen.back());
          for (int j = 0; j < model::kHy4DraftTokens; ++j) in.ids.push_back(drafts[size_t(j)]);
          for (int j = 0; j < n; ++j) in.pos.push_back(computed + j), in.logit_rows.push_back(j);
          in.backup.push_back({slot, computed, n});
          m.step(in, out);
          int a = 0;
          while (a < model::kHy4DraftTokens && drafts[size_t(a)] == argmax(out.data() + size_t(a) * V)) ++a;
          for (int j = 0; j < a; ++j) gen.push_back(drafts[size_t(j)]);
          gen.push_back(argmax(out.data() + size_t(a) * V));
          if (a < model::kHy4DraftTokens) restores.push_back({slot, computed, computed + 1 + a, computed + n});
          computed += 1 + a;
          ++steps;
          accepted += a;
          ++hist[size_t(a)];
        }
        t_spec = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        gen.resize(size_t(spec));
      };
      if (spec_only == "draft-first") run_draft(), run_plain();
      else {
        if (spec_only != "draft") run_plain();
        if (spec_only != "plain") run_draft();
      }
      if (!plain.empty()) std::printf("plain greedy: %d tokens, %.1f tok/s\n", spec - 1, (spec - 1) / t_plain);
      if (!gen.empty()) {
        std::printf("DFly2: %d steps, %.2f drafts accepted per step (%.1f%%), %.2f tokens per step, %.1f tok/s "
                    "(draft %.1f ms per step)\n", steps, double(accepted) / steps,
                    100.0 * accepted / (steps * model::kHy4DraftTokens), double(spec - 1) / steps,
                    (spec - 1) / t_spec, t_draft * 1e3 / steps);
        std::printf("accepted per step:");
        for (size_t i = 0; i < hist.size(); ++i) std::printf(" %zu:%d", i, hist[i]);
        std::printf("\n");
      }
      if (!gen.empty() && !plain.empty()) {
        int same = 0;
        while (same < spec && gen[size_t(same)] == plain[size_t(same)]) ++same;
        std::printf("the two continuations agree on the first %d of %d tokens\n", same, spec);
      }
      return 0;
    }
    std::vector<float> logits(size_t(S) * V), step;
    const auto t0 = std::chrono::steady_clock::now();
    double t_pre = 0, t_dec = 0;
    const auto run = [&](int s0, int n) {
      engine::StepInput in;
      in.chunks.push_back({slot, s0, n, 0});
      for (int i = 0; i < n; ++i) {
        in.ids.push_back(ids[size_t(s0 + i)]);
        in.pos.push_back(s0 + i);
        in.logit_rows.push_back(i);
      }
      const auto ts = std::chrono::steady_clock::now();
      m.step(in, step);
      (n > 1 ? t_pre : t_dec) += std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
      std::copy(step.begin(), step.end(), logits.begin() + size_t(s0) * V);
    };
    for (int s0 = 0; s0 < P; s0 += chunk) run(s0, std::min(chunk, P - s0));
    for (int t = P; t < S; ++t) run(t, 1);
    std::printf("steps: prefill %.2f s for %d tokens (%.0f tok/s), decode %.1f ms/token (%d tokens), %.1f s in all\n",
                t_pre, P, P / std::max(t_pre, 1e-9), S > P ? t_dec * 1e3 / (S - P) : 0.0, S - P,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::vector<float> lr(size_t(S) * V);
    {
      std::ifstream f(oracle + "/logits.bin", std::ios::binary);
      if (!f.read(reinterpret_cast<char*>(lr.data()), std::streamsize(lr.size() * 4)))
        throw std::runtime_error("no oracle logits.bin");
    }
    if (!out_path.empty())
      std::ofstream(out_path, std::ios::binary).write(reinterpret_cast<const char*>(logits.data()),
                                                       std::streamsize(logits.size() * 4));
    double kl_sum = 0;
    int agree = 0;
    std::vector<double> kls;
    for (int t = 0; t < S; ++t) {
      const float* a = lr.data() + size_t(t) * V;
      const float* b = logits.data() + size_t(t) * V;
      double ma = -1e300, mb = -1e300;
      int ia = 0, ib = 0;
      for (int v = 0; v < V; ++v) {
        if (a[v] > ma) ma = a[v], ia = v;
        if (b[v] > mb) mb = b[v], ib = v;
      }
      double za = 0, zb = 0;
      for (int v = 0; v < V; ++v) za += std::exp(a[v] - ma), zb += std::exp(b[v] - mb);
      const double la = std::log(za) + ma, lb = std::log(zb) + mb;
      double kl = 0;
      for (int v = 0; v < V; ++v) {
        const double lpa = a[v] - la, lpb = b[v] - lb;
        kl += std::exp(lpa) * (lpa - lpb);
      }
      kls.push_back(kl);
      kl_sum += kl;
      agree += ia == ib;
    }
    std::sort(kls.begin(), kls.end());
    std::printf("resident, slot %d of %d, %d positions: mean KL %.3e  median %.3e  p99 %.3e  top-1 agreement %.2f%%\n",
                slot, slots, S, kl_sum / S, kls[kls.size() / 2], kls[kls.size() * 99 / 100], 100.0 * agree / S);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
