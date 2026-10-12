// unholy-dspark-test: our DSpark drafter (model/forward_ref dspark_*) against the reference
// (tools/ref/oracle/dspark.py), fed the oracle's own target hidden states so only the drafter differs.
//
//   unholy-dspark-test --model DIR --ids FILE --prompt P --decode D --hidden ORACLE_RUN --spec ORACLE_SPEC
//                      [--dev 0]
//
// --hidden: run.py output with h_040..h_042 (bf16 [S, hc, dim]); --spec: dspark.py output for the same
// ids/prompt/decode. Replays the reference loop: main-KV of positions [0, P), then for each decode
// position t in [P, S - 1) the main-KV of t and a draft block [ids[t + 1], noise x 4] at t + 1 .. t + 5.
// Prints, per position, the drafts of both, the logits' relative L2 and top-1 agreement, the confidences.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernels/dspark.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"
#include "model/loader.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

using namespace unholy;

namespace {
template <typename T>
std::vector<T> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const size_t n = size_t(f.tellg());
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = "/scratch/dsv4-ve", ids_path, hidden_dir, spec_dir;
  int P = 0, Dn = 0, dev = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model_dir = v;
    else if (a == "--ids") ids_path = v;
    else if (a == "--prompt") P = std::stoi(v);
    else if (a == "--decode") Dn = std::stoi(v);
    else if (a == "--hidden") hidden_dir = v;
    else if (a == "--spec") spec_dir = v;
    else if (a == "--dev") dev = std::stoi(v);
  }
  try {
    auto ck = model::Checkpoint::open(model_dir);
    const auto c = model::Dsv4Config::from_hf(ck->config());
    const int S = P + Dn, D = c.dim, HC = c.hc_mult, V = c.vocab, Bk = c.dspark_block;
    const int NT = int(c.dspark_target_layers.size());
    const auto ids_all = read_file<uint32_t>(ids_path);
    if (int(ids_all.size()) < S) throw std::runtime_error("not enough ids");
    UH_CHECK(hipSetDevice(dev));
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));

    // Drafter with every routed expert on this card; the target's LM head.
    model::LoadOptions lo;
    const auto dw = model::load_dspark(*ck, c, dev, st, lo);
    model::Uploader up(*ck);
    const auto& ht = ck->at("head.weight");
    rt::DeviceBuffer head(dev, ht.nbytes);
    up.upload(ht, head.data(), st);
    up.sync();
    const auto emb_tab = ck->as<uint16_t>(ck->at("embed.weight"));

    model::CacheGeometry geo{1, 64, 64};
    std::vector<model::LayerCache> caches;
    for (const auto& w : dw.stages) caches.push_back(model::make_layer_cache(c, w, geo, dev, st));
    auto scr = model::make_scratch(c, dev, 256, 4096);
    auto rope = model::make_rope_tables(c, dev, S + 16);

    // Main hidden rows [S, NT * D]: the HC means of the oracle's dumps.
    std::vector<uint16_t> mh(size_t(S) * NT * D);
    for (int j = 0; j < NT; ++j) {
      char name[64];
      std::snprintf(name, sizeof name, "/h_%03d.bin", c.dspark_target_layers[size_t(j)]);
      const auto h = read_file<uint16_t>(hidden_dir + name);
      if (h.size() != size_t(S) * HC * D) throw std::runtime_error(std::string("unexpected size of ") + name);
      rt::DeviceBuffer dh(dev, h.size() * 2), dout(dev, size_t(S) * D * 2), drows(dev, size_t(S) * 4);
      std::vector<int32_t> rows(static_cast<size_t>(S));
      for (int s = 0; s < S; ++s) rows[size_t(s)] = s;
      UH_CHECK(hipMemcpy(dh.data(), h.data(), h.size() * 2, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(drows.data(), rows.data(), rows.size() * 4, hipMemcpyHostToDevice));
      kernels::hc_mean_rows(dh.as<uint16_t>(), HC, D, drows.as<int32_t>(), S, dout.as<uint16_t>(), D, st);
      std::vector<uint16_t> m(size_t(S) * D);
      UH_CHECK(hipMemcpy(m.data(), dout.data(), m.size() * 2, hipMemcpyDeviceToHost));
      for (int s = 0; s < S; ++s)
        std::memcpy(&mh[(size_t(s) * NT + j) * D], &m[size_t(s) * D], size_t(D) * 2);
    }
    rt::DeviceBuffer dmh(dev, mh.size() * 2), dslot(dev, size_t(S) * 4), dpos(dev, size_t(S) * 4);
    UH_CHECK(hipMemcpy(dmh.data(), mh.data(), mh.size() * 2, hipMemcpyHostToDevice));
    {
      std::vector<int32_t> zeros(static_cast<size_t>(S), 0), pos(static_cast<size_t>(S));
      for (int s = 0; s < S; ++s) pos[size_t(s)] = s;
      UH_CHECK(hipMemcpy(dslot.data(), zeros.data(), zeros.size() * 4, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(dpos.data(), pos.data(), pos.size() * 4, hipMemcpyHostToDevice));
    }
    // Prefill: the main-KV of [0, P); the window keeps the last 128, so only those are computed (as the
    // server does for each chunk).
    const int p0 = std::max(0, P - 128);
    model::dspark_main_kv(c, dw, caches, scr, rope, dmh.as<uint16_t>() + size_t(p0) * NT * D, dslot.as<int32_t>(),
                          dpos.as<int32_t>() + p0, P - p0, st);

    const auto ref_logits = read_file<float>(spec_dir + "/spec_logits.bin");
    const auto ref_ids = read_file<int32_t>(spec_dir + "/spec_ids.bin");
    const auto ref_conf = read_file<float>(spec_dir + "/spec_conf.bin");
    rt::DeviceBuffer x(dev, size_t(Bk) * HC * D * 2), bids(dev, size_t(Bk) * 4), bpos(dev, size_t(Bk) * 4),
        logits(dev, size_t(Bk) * V * 4), tok(dev, size_t(Bk + 1) * 4), conf(dev, size_t(Bk) * 4);
    int same_drafts = 0, n = 0;
    double worst = 0;
    for (int t = P; t < S - 1; ++t, ++n) {
      model::dspark_main_kv(c, dw, caches, scr, rope, dmh.as<uint16_t>() + size_t(t) * NT * D,
                            dslot.as<int32_t>(), dpos.as<int32_t>() + t, 1, st);
      std::vector<int32_t> block(static_cast<size_t>(Bk), c.dspark_noise_token), bp(static_cast<size_t>(Bk));
      block[0] = int32_t(ids_all[size_t(t + 1)]);
      for (int i = 0; i < Bk; ++i) bp[size_t(i)] = t + 1 + i;
      std::vector<uint16_t> hx(size_t(Bk) * HC * D);
      for (int i = 0; i < Bk; ++i)
        for (int h = 0; h < HC; ++h)
          std::memcpy(&hx[(size_t(i) * HC + h) * D], &emb_tab[size_t(block[size_t(i)]) * D], size_t(D) * 2);
      UH_CHECK(hipMemcpy(x.data(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(bids.data(), block.data(), block.size() * 4, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(bpos.data(), bp.data(), bp.size() * 4, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(tok.data(), block.data(), 4, hipMemcpyHostToDevice));
      model::DraftDesc dd;
      dd.slots = {0};
      dd.start = {t};
      dd.T = Bk;
      dd.ids = bids.as<int32_t>();
      dd.pos = bpos.as<int32_t>();
      for (size_t s = 0; s < dw.stages.size(); ++s)
        model::dspark_stage(c, dw.stages[s], caches[s], scr, rope, x.as<uint16_t>(), dd, st);
      model::dspark_head(c, dw, scr, x.as<uint16_t>(), head.as<uint16_t>(), 1, logits.as<float>(), tok.as<int32_t>(),
                         conf.as<float>(), st);
      std::vector<float> lg(size_t(Bk) * V), cf(static_cast<size_t>(Bk));
      std::vector<int32_t> tk(static_cast<size_t>(Bk + 1));
      UH_CHECK(hipMemcpy(lg.data(), logits.data(), lg.size() * 4, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(cf.data(), conf.data(), cf.size() * 4, hipMemcpyDeviceToHost));
      UH_CHECK(hipMemcpy(tk.data(), tok.data(), tk.size() * 4, hipMemcpyDeviceToHost));
      const float* rl = &ref_logits[size_t(n) * Bk * V];
      const int32_t* ri = &ref_ids[size_t(n) * (Bk + 1)];
      const float* rc = &ref_conf[size_t(n) * Bk];
      std::printf("t=%d drafts", t);
      bool same = true;
      for (int i = 1; i <= Bk; ++i) {
        std::printf(" %d%s", tk[size_t(i)], tk[size_t(i)] == ri[i] ? "" : "*");
        same &= tk[size_t(i)] == ri[i];
      }
      same_drafts += same;
      std::printf(" | ref");
      for (int i = 1; i <= Bk; ++i) std::printf(" %d", ri[i]);
      std::printf(" | rel L2");
      for (int i = 0; i < Bk; ++i) {
        double num = 0, den = 0;
        for (int v = 0; v < V; ++v) {
          const double a = lg[size_t(i) * V + size_t(v)], b = rl[size_t(i) * V + size_t(v)];
          num += (a - b) * (a - b);
          den += b * b;
        }
        const double r = std::sqrt(num / std::max(den, 1e-30));
        worst = std::max(worst, r);
        std::printf(" %.1e", r);
      }
      std::printf(" | conf");
      for (int i = 0; i < Bk; ++i) std::printf(" %.3f/%.3f", cf[size_t(i)], rc[i]);
      std::printf("\n");
    }
    std::printf("%d/%d positions with identical drafts; worst logits rel L2 %.2e\n", same_drafts, n, worst);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
