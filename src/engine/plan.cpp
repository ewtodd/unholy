#include "engine/plan.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "rt/hip.hpp"

namespace unholy::engine {

std::vector<DeviceInfo> query_devices() {
  int n = 0;
  UH_CHECK(hipGetDeviceCount(&n));
  std::vector<DeviceInfo> out;
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p{};
    UH_CHECK(hipGetDeviceProperties(&p, i));
    DeviceInfo d;
    d.id = i;
    d.arch = p.gcnArchName;
    if (auto c = d.arch.find(':'); c != std::string::npos) d.arch.resize(c);
    d.integrated = p.integrated != 0;
    rt::DeviceGuard g(i);
    UH_CHECK(hipMemGetInfo(&d.free_bytes, &d.total_bytes));
    out.push_back(d);
  }
  return out;
}

UsageProfile load_usage_profile(const std::string& path, int n_layers, int n_experts) {
  UsageProfile u(size_t(n_layers), std::vector<double>(size_t(n_experts), 1.0));
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open usage profile " + path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.size() < 2 || line[0] != 'L') continue;
    std::istringstream ss(line.substr(1));
    int layer = -1;
    ss >> layer;
    if (layer < 0 || layer >= n_layers) continue;
    for (int e = 0; e < n_experts; ++e) {
      double v = 0;
      if (!(ss >> v)) break;
      u[size_t(layer)][size_t(e)] = v + 1.0;  // +1: never-seen experts keep a tiny non-zero weight
    }
  }
  return u;
}

size_t expert_bytes(const model::Checkpoint& ck, int layer, int expert, bool exl3) {
  const std::string p = "layers." + std::to_string(layer) + ".ffn.experts." + std::to_string(expert) + ".";
  size_t b = 0;
  for (const char* m : {"w1", "w2", "w3"}) {
    if (exl3) {
      for (const char* s : {".trellis", ".suh", ".svh"}) b += ck.at(p + m + s).nbytes;
    } else {
      for (const char* s : {".weight", ".scale"}) b += ck.at(p + m + s).nbytes;
    }
  }
  return b;
}

size_t dense_layer_bytes(const model::Checkpoint& ck, int layer) {
  const std::string p = "layers." + std::to_string(layer) + ".";
  size_t b = 0;
  for (const auto& t : ck.tensors()) {
    if (t.name.compare(0, p.size(), p) != 0) continue;
    if (t.name.find(".ffn.experts.") != std::string::npos) continue;
    b += t.name.find("tid2eid") != std::string::npos ? t.nbytes / 2 : t.nbytes;  // I64 -> I32
  }
  return b;
}

Plan make_plan(const PlanInput& in) {
  const auto& c = *in.cfg;
  const auto devs = query_devices();
  Plan pl;
  int ncard = 0;
  for (const auto& d : devs) {
    if (d.integrated) pl.apu = d.id;
    else if (ncard < 2) pl.card[ncard++] = d.id;
  }
  if (ncard < 2 || pl.apu < 0) throw std::runtime_error("plan: need two discrete cards and the APU");
  auto info = [&](int id) -> const DeviceInfo& { return devs[size_t(id)]; };

  // Layer split: balance dense + cache bytes; card 0 also holds the embedding, card 1 the head.
  std::vector<size_t> layer_bytes(size_t(c.n_layers));
  size_t total = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    const int r = c.ratio(L);
    layer_bytes[size_t(L)] = dense_layer_bytes(*in.dense, L) + model::layer_cache_bytes(c, r, r == 4, in.geo);
    total += layer_bytes[size_t(L)];
  }
  pl.layer_card.assign(size_t(c.n_layers), 1);
  size_t acc = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    if (acc + layer_bytes[size_t(L)] / 2 > total / 2) break;
    pl.layer_card[size_t(L)] = 0;
    acc += layer_bytes[size_t(L)];
  }
  const size_t embed = in.dense->at("embed.weight").nbytes;
  const size_t head = in.dense->at("head.weight").nbytes + in.dense->at("norm.weight").nbytes +
                      in.dense->at("hc_head_fn").nbytes;
  for (int L = 0; L < c.n_layers; ++L) {
    const int k = pl.layer_card[size_t(L)];
    const int r = c.ratio(L);
    pl.dense[k] += dense_layer_bytes(*in.dense, L);
    pl.cache[k] += model::layer_cache_bytes(c, r, r == 4, in.geo);
  }
  pl.dense[0] += embed;
  pl.dense[1] += head;
  for (int k = 0; k < 2; ++k) {
    // Scratch, plus a logits buffer on the head card.
    pl.scratch[k] = model::scratch_bytes(c, in.max_tokens, in.max_seq) +
                    (k == 1 ? size_t(in.max_tokens) * c.vocab * 4 + size_t(in.max_tokens) * c.dim * 6 : 0) +
                    size_t(in.max_tokens) * c.hc_mult * c.dim * 2;
    const size_t used = pl.dense[k] + pl.cache[k] + pl.scratch[k] + in.card_margin;
    const size_t free = info(pl.card[k]).free_bytes;
    if (used > free) throw std::runtime_error("plan: card " + std::to_string(k) + " cannot hold its dense layers");
    pl.budget[k] = free - used;
  }

  // Hot experts: greedy by per-layer routing share, per card budget.
  struct Cand {
    double share;
    int layer, expert;
    size_t bytes;
  };
  std::vector<Cand> cands[2];
  double mass_total = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    double sum = 0;
    for (double v : in.usage[size_t(L)]) sum += v;
    for (int e = 0; e < c.n_experts; ++e) {
      const double share = in.usage[size_t(L)][size_t(e)] / sum;
      cands[pl.layer_card[size_t(L)]].push_back({share, L, e, expert_bytes(in.exl3 ? *in.exl3 : *in.dense, L, e,
                                                                            in.exl3 != nullptr)});
    }
    mass_total += 1.0;
  }
  pl.on_card.assign(size_t(c.n_layers), std::vector<uint8_t>(size_t(c.n_experts), 0));
  double mass_cards = 0;
  for (int k = 0; k < 2; ++k) {
    std::sort(cands[k].begin(), cands[k].end(), [](const Cand& a, const Cand& b) { return a.share > b.share; });
    for (const auto& cd : cands[k]) {
      if (pl.experts[k] + cd.bytes > pl.budget[k]) continue;
      pl.experts[k] += cd.bytes;
      pl.on_card[size_t(cd.layer)][size_t(cd.expert)] = 1;
      mass_cards += cd.share;
    }
  }
  for (int k = 0; k < 2; ++k)
    for (const auto& cd : cands[k])
      if (!pl.on_card[size_t(cd.layer)][size_t(cd.expert)]) pl.apu_experts += cd.bytes;
  pl.routed_on_cards = mass_cards / mass_total;
  pl.apu_free = info(pl.apu).free_bytes;
  if (pl.apu_experts + in.apu_margin > pl.apu_free) throw std::runtime_error("plan: the APU cannot hold the cold experts");
  return pl;
}

std::string Plan::summary() const {
  auto gb = [](size_t b) { return double(b) / 1e9; };
  std::ostringstream s;
  int first[2] = {-1, -1}, last[2] = {-1, -1};
  for (size_t L = 0; L < layer_card.size(); ++L) {
    const int k = layer_card[L];
    if (first[k] < 0) first[k] = int(L);
    last[k] = int(L);
  }
  char buf[512];
  for (int k = 0; k < 2; ++k) {
    size_t hot = 0;
    for (size_t L = 0; L < on_card.size(); ++L)
      if (layer_card[L] == k)
        for (auto v : on_card[L]) hot += v;
    std::snprintf(buf, sizeof buf,
                  "card %d (dev %d): layers %d-%d  dense %.2f GB  cache %.2f GB  scratch %.2f GB  hot experts %zu "
                  "(%.2f GB of %.2f GB budget)\n",
                  k, card[k], first[k], last[k], gb(dense[k]), gb(cache[k]), gb(scratch[k]), hot, gb(experts[k]),
                  gb(budget[k]));
    s << buf;
  }
  std::snprintf(buf, sizeof buf, "apu (dev %d): cold experts %.2f GB of %.2f GB free; predicted routing on cards %.1f%%\n",
                apu, gb(apu_experts), gb(apu_free), 100.0 * routed_on_cards);
  s << buf;
  return s.str();
}

}  // namespace unholy::engine
