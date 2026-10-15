// unholy-resident-test: the whole model resident across both cards and the APU (per the planner),
// run on an oracle prompt: one prefill step + teacher-forced decode steps. Compares logits with the
// oracle and, bit for bit, with a single-card streaming run (`unholy-ref-run --out`) of the same
// weights: same kernels in the same order, so the placement must not change a single bit.
//
//   unholy-resident-test --oracle DIR --profile FILE [--exl3 DIR] [--ref logits.bin] [--ckpt DIR] [--out FILE]
//
// --out writes this run's logits (fp32 [steps, vocab]) for a later --ref: e.g. the same placement with
// UNHOLY_REMOTE_Q=0 (bf16 pair rows to the APU) as the reference for the compact fp8 transport.
//
// --card-margin-bytes N: the planner's card margin (default 2 GiB). Lowering it by exactly what a change
// adds to the scratch keeps the expert placement, so logits stay comparable bit for bit with --ref.
//
// --batch N adds a batch-invariance check: N sequences (prompts of P - 61 s tokens) decode the oracle's
// continuation alone, then in fresh slots all together (one token each per step), then up to four of
// them as verify chunks of 4 tokens with every row's logits (<= 16 rows: the GEMV regime,
// kGemvMaxRows); every logit must equal the solo run's. (Past that the GEMM kernels sum in another order.)
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "engine/plan.hpp"
#include "engine/resident.hpp"
#include "gufo/json.hpp"
#include "model/dsv4.hpp"
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
std::vector<float> read_f32(const std::string& p, size_t n) {
  std::vector<float> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 4));
  return v;
}
double kl_row(const float* a, const float* b, int V) {
  double ma = -1e300, mb = -1e300;
  for (int v = 0; v < V; ++v) ma = std::max(ma, double(a[v])), mb = std::max(mb, double(b[v]));
  double za = 0, zb = 0;
  for (int v = 0; v < V; ++v) za += std::exp(a[v] - ma), zb += std::exp(b[v] - mb);
  const double la = std::log(za) + ma, lb = std::log(zb) + mb;
  double kl = 0;
  for (int v = 0; v < V; ++v) kl += std::exp(a[v] - la) * ((a[v] - la) - (b[v] - lb));
  return kl;
}
}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = "/scratch/dsv4-ve", oracle, exl3_dir, profile, ref_path, out_path;
  int batch = 0;
  long long card_margin = -1;  // bytes; -1: the planner's default
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--oracle") oracle = next();
    else if (a == "--exl3") exl3_dir = next();
    else if (a == "--profile") profile = next();
    else if (a == "--ref") ref_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--batch") batch = std::stoi(next());
    else if (a == "--card-margin-bytes") card_margin = std::stoll(next());
    else {
      std::fprintf(stderr, "usage: %s --oracle DIR --profile FILE [--exl3 DIR] [--ref logits.bin] [--out FILE] [--batch N]\n", argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(ckpt);
    std::unique_ptr<model::Checkpoint> xck;
    if (!exl3_dir.empty()) xck = model::Checkpoint::open(exl3_dir);
    const auto c = model::Dsv4Config::from_hf(ck->config());
    const auto meta = gufo::json::parse(read_text(oracle + "/meta.json"));
    const int S = int(meta.member_size("S")), P = int(meta.member_size("prompt"));
    std::vector<int32_t> ids;
    for (const auto& v : meta.find("ids")->items()) ids.push_back(int32_t(v.as_double()));

    const int max_tokens = 1024, max_seq = 2048;
    const int slots = 1 + 3 * batch, pps = max_seq / model::kPageTokens;
    model::CacheGeometry geo{slots, pps, slots * pps};
    engine::PlanInput pin;
    pin.cfg = &c;
    pin.dense = ck.get();
    pin.exl3 = xck.get();
    pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
    pin.geo = geo;
    pin.max_tokens = max_tokens;
    pin.max_seq = max_seq;
    if (card_margin >= 0) pin.card_margin = size_t(card_margin);
    const auto plan = engine::make_plan(pin);
    std::printf("%s", plan.summary().c_str());
    std::fflush(stdout);

    const auto t0 = std::chrono::steady_clock::now();
    engine::ResidentModel m(c, *ck, xck.get(), plan, geo, max_tokens, max_seq);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("loaded in %.1f s\n", std::chrono::duration<double>(t1 - t0).count());
    std::vector<int32_t> table(size_t(geo.max_slots) * geo.max_pages_per_seq);
    std::iota(table.begin(), table.end(), 0);
    m.set_block_tables(table);

    std::vector<float> logits;  // [S][V]
    logits.reserve(size_t(S) * c.vocab);
    {
      engine::StepInput in;
      in.chunks = {{0, 0, P, 0}};
      in.ids.assign(ids.begin(), ids.begin() + P);
      in.pos.resize(size_t(P));
      std::iota(in.pos.begin(), in.pos.end(), 0);
      in.logit_rows.resize(size_t(P));
      std::iota(in.logit_rows.begin(), in.logit_rows.end(), 0);
      const auto ts = std::chrono::steady_clock::now();
      std::vector<float> out;
      m.step(in, out);
      std::printf("prefill %d tokens: %.2f s\n", P,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
      logits.insert(logits.end(), out.begin(), out.end());
    }
    for (int t = P; t < S; ++t) {
      engine::StepInput in;
      in.chunks = {{0, t, 1, 0}};
      in.ids = {ids[size_t(t)]};
      in.pos = {t};
      in.logit_rows = {0};
      const auto ts = std::chrono::steady_clock::now();
      std::vector<float> out;
      m.step(in, out);
      std::printf("decode pos %d: %.0f ms\n", t,
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      logits.insert(logits.end(), out.begin(), out.end());
    }
    const auto ref = read_f32(oracle + "/logits.bin", logits.size());
    double kl = 0;
    int agree = 0;
    for (int t = 0; t < S; ++t) {
      const float* a = ref.data() + size_t(t) * c.vocab;
      const float* b = logits.data() + size_t(t) * c.vocab;
      kl += kl_row(a, b, c.vocab);
      agree += std::max_element(a, a + c.vocab) - a == std::max_element(b, b + c.vocab) - b;
    }
    std::printf("vs oracle: mean KL %.3e, top-1 agreement %.2f%%\n", kl / S, 100.0 * agree / S);
    if (!out_path.empty()) {
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
    }
    if (!ref_path.empty()) {
      const auto single = read_f32(ref_path, logits.size());
      size_t same = 0;
      for (size_t i = 0; i < logits.size(); ++i) same += std::memcmp(&single[i], &logits[i], 4) == 0;
      std::printf("vs single-card streaming run: %zu / %zu logits bit-identical\n", same, logits.size());
    }
    if (batch > 0) {
      const int N = batch, D = S - P, V = c.vocab;
      const auto len = [&](int s) { return P - 61 * s; };
      const auto prefill = [&](int slot, int L) {
        engine::StepInput in;
        in.chunks = {{slot, 0, L, 0}};
        in.ids.assign(ids.begin(), ids.begin() + L);
        in.pos.resize(size_t(L));
        std::iota(in.pos.begin(), in.pos.end(), 0);
        in.logit_rows = {L - 1};
        std::vector<float> out;
        m.step(in, out);
      };
      // Steps of n tokens per sequence (every row's logits) over slots base + s (s < ns), compared with `solo`.
      const auto together = [&](int base, int n, int ns, const std::vector<std::vector<float>>& solo) {
        size_t same = 0, total = 0;
        for (int d = 0; d + n <= D; d += n) {
          engine::StepInput in;
          for (int s = 0; s < ns; ++s) {
            in.chunks.push_back({base + s, len(s) + d, n, s * n});
            for (int i = 0; i < n; ++i) {
              in.ids.push_back(ids[size_t(len(s) + d + i)]);
              in.pos.push_back(len(s) + d + i);
              in.logit_rows.push_back(s * n + i);
            }
          }
          std::vector<float> out;
          m.step(in, out);
          for (int s = 0; s < ns; ++s)
            for (int i = 0; i < n; ++i) {
              const float* a = out.data() + size_t(s * n + i) * V;
              const float* b = solo[size_t(s)].data() + size_t(d + i) * V;
              for (int v = 0; v < V; ++v) same += std::memcmp(a + v, b + v, 4) == 0;
              total += size_t(V);
            }
        }
        return std::pair<size_t, size_t>{same, total};
      };
      std::vector<std::vector<float>> solo(static_cast<size_t>(N));
      for (int s = 0; s < N; ++s) {
        prefill(1 + s, len(s));
        for (int d = 0; d < D; ++d) {
          engine::StepInput in;
          in.chunks = {{1 + s, len(s) + d, 1, 0}};
          in.ids = {ids[size_t(len(s) + d)]};
          in.pos = {len(s) + d};
          in.logit_rows = {0};
          std::vector<float> out;
          m.step(in, out);
          solo[size_t(s)].insert(solo[size_t(s)].end(), out.begin(), out.end());
        }
      }
      for (int s = 0; s < N; ++s) prefill(1 + N + s, len(s));
      const auto [s1, t1] = together(1 + N, 1, N, solo);
      std::printf("batch of %d, one token each: %zu / %zu logits bit-identical to the solo runs\n", N, s1, t1);
      const int nv = std::min(N, 4);
      for (int s = 0; s < nv; ++s) prefill(1 + 2 * N + s, len(s));
      const auto [s4, t4] = together(1 + 2 * N, 4, nv, solo);
      std::printf("batch of %d, 4-token verify chunks: %zu / %zu logits bit-identical to the solo runs\n", nv, s4, t4);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
