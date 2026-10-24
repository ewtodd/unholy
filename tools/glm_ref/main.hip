// unholy-glm-ref: GLM-5.3-Flash on the reference path (model/glm5.hpp), layer-streamed on one GPU like the oracle
// (tools/ref/glm_oracle/run.py): each layer is loaded, run over every position (the prompt in chunks, then the
// teacher-forced continuation one token at a time), compared with the oracle's dump of its output, and freed;
// then the head, compared with the oracle's logits by KL divergence and top-1 agreement.
//
//   unholy-glm-ref --oracle DIR [--ckpt FP8_DIR] [--dense DIR] [--experts DIR] [--chunk C] [--device D]
//                  [--isolate] [--kda-fp8] [--a8] [--slots N] [--layers N] [--out logits.bin]
//
// --dense / --experts take the dense weights / the routed experts from another checkpoint (an EXL3 quant: builds
// A-D of docs/glm53-flash-quant.md); --ckpt (the FP8 release) still supplies what EXL3 checkpoints leave out.
// --isolate starts every layer from the oracle's input to it, so each layer's own error shows.
// --kda-fp8 re-quantizes the KDA q / k / v / o projections to fp8 (builds C / D: the release's fp8 dense with KDA in
// fp8 too). --a8 quantizes the activations of every fp8 linear dynamically, per 128 columns (as fp8 serving
// stacks run the release). --slots N runs N copies of the sequence in different slots of the same batches;
// the reference's per-row arithmetic does not depend on the other rows, so every copy must match slot 0 bit for bit.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gufo/json.hpp"
#include "kernels/ref.hpp"
#include "model/glm5.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

using namespace unholy;

namespace {

std::string read_text(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

bool read_file(const std::string& p, void* dst, size_t bytes) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  f.read(static_cast<char*>(dst), std::streamsize(bytes));
  if (size_t(f.gcount()) != bytes) throw std::runtime_error(p + ": short read");
  return true;
}

float bf(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Relative error of ours against the oracle's residual [S, row], overall and the worst position.
void compare_layer(int L, const std::vector<uint16_t>& ours, const std::vector<uint16_t>& ref, int S, size_t row,
                   double secs) {
  double num = 0, den = 0, worst = 0, maxd = 0;
  int worst_pos = 0;
  for (int t = 0; t < S; ++t) {
    double n = 0, d = 0;
    for (size_t i = 0; i < row; ++i) {
      const double a = bf(ours[size_t(t) * row + i]), b = bf(ref[size_t(t) * row + i]);
      n += (a - b) * (a - b);
      d += b * b;
      maxd = std::max(maxd, std::fabs(a - b));
    }
    num += n;
    den += d;
    const double rel = std::sqrt(n / std::max(d, 1e-30));
    if (rel > worst) worst = rel, worst_pos = t;
  }
  std::printf("layer %2d: rel err %.3e  worst %.3e (pos %d)  max |d| %.3e  %.1f s\n", L, std::sqrt(num / den), worst,
              worst_pos, maxd, secs);
  std::fflush(stdout);
}

// The DSA layer's token selection per query against the oracle's (sets: the order is the top-k's business).
void compare_selection(int L, const std::vector<int32_t>& ours, const std::vector<int32_t>& ref, int S, int width) {
  int same = 0, selective = 0, selective_same = 0, worst_pos = -1;
  double jsum = 0, worst = 2;
  std::vector<int32_t> a, b;
  for (int t = 0; t < S; ++t) {
    a.clear();
    b.clear();
    for (int j = 0; j < width; ++j) {
      if (ref[size_t(t) * width + j] >= 0) a.push_back(ref[size_t(t) * width + j]);
      if (ours[size_t(t) * width + j] >= 0) b.push_back(ours[size_t(t) * width + j]);
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    size_t i = 0, k = 0, both = 0;
    while (i < a.size() && k < b.size()) {
      if (a[i] == b[k]) ++both, ++i, ++k;
      else if (a[i] < b[k]) ++i;
      else ++k;
    }
    const size_t uni = a.size() + b.size() - both;
    const double jac = uni ? double(both) / double(uni) : 1.0;
    jsum += jac;
    same += a == b;
    if (int(a.size()) < t + 1) ++selective, selective_same += a == b;  // the query did not see every position
    if (jac < worst) worst = jac, worst_pos = t;
  }
  std::printf("layer %2d selection: %d / %d queries identical (%d / %d of the selective ones), mean Jaccard %.5f, "
              "worst %.4f (pos %d)\n",
              L, same, S, selective_same, selective, jsum / S, worst, worst_pos);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = "/scratch/models/glm53-flash-fp8", oracle, dense_dir, experts_dir, out_path;
  int chunk = 0, dev = 0, n_layers = -1, slots = 1;
  bool isolate = false, kda_fp8 = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--oracle") oracle = next();
    else if (a == "--dense") dense_dir = next();
    else if (a == "--experts") experts_dir = next();
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--device") dev = std::stoi(next());
    else if (a == "--layers") n_layers = std::stoi(next());
    else if (a == "--isolate") isolate = true;
    else if (a == "--kda-fp8") kda_fp8 = true;
    else if (a == "--slots") slots = std::stoi(next());
    else if (a == "--a8") model::set_glm5_a8(true);
    else if (a == "--out") out_path = next();
    else {
      std::fprintf(stderr,
                   "usage: %s --oracle DIR [--ckpt DIR] [--dense DIR] [--experts DIR] [--chunk C] [--device D] "
                   "[--isolate] [--kda-fp8] [--a8] [--slots N] [--layers N] [--out F]\n",
                   argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(ckpt);
    std::unique_ptr<model::Checkpoint> dck, eck;
    if (!dense_dir.empty()) dck = model::Checkpoint::open(dense_dir);
    if (!experts_dir.empty()) eck = model::Checkpoint::open(experts_dir);
    const model::Checkpoint& dense = dck ? *dck : *ck;
    const model::Checkpoint& experts = eck ? *eck : dense;
    const auto c = model::Glm5Config::from_hf(ck->config());
    const auto meta = gufo::json::parse(read_text(oracle + "/meta.json"));
    const int S = int(meta.member_size("S")), P = int(meta.member_size("prompt"));
    std::vector<int32_t> ids;
    for (const auto& v : meta.find("ids")->items()) ids.push_back(int32_t(v.as_double()));
    if (chunk <= 0) chunk = P;
    if (n_layers < 0) n_layers = c.n_layers;
    const size_t row = size_t(c.hc_mult) * c.dim;

    rt::DeviceGuard g(dev);
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));
    const int max_seq = (S + 255) / 256 * 256;
    auto scr = model::make_glm5_scratch(c, dev, slots * std::max(chunk, 1), max_seq);
    // x: the residual of each slot's copy [slots][S][row]; xb / bpos: a multi-slot batch's rows and positions.
    const size_t xslot = size_t(S) * row;
    rt::DeviceBuffer x(dev, slots * xslot * 2), dpos(dev, size_t(S) * 4), dids(dev, size_t(S) * 4);
    rt::DeviceBuffer xb(dev, slots > 1 ? size_t(slots) * chunk * row * 2 : 16), bpos(dev, size_t(slots) * chunk * 4);
    std::vector<int32_t> pos(static_cast<size_t>(S));
    for (int t = 0; t < S; ++t) pos[size_t(t)] = t;
    UH_CHECK(hipMemcpy(dpos.data(), pos.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dids.data(), ids.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    {
      auto hw = model::load_glm5_head(dense, dev, st, true, false);
      kernels::embed_hc(dids.as<int32_t>(), hw.embed.p<uint16_t>(), x.as<uint16_t>(), S, c.dim, c.hc_mult, st);
      for (int i = 1; i < slots; ++i)
        UH_CHECK(hipMemcpyAsync(x.as<uint16_t>() + i * xslot, x.data(), xslot * 2, hipMemcpyDeviceToDevice, st));
      UH_CHECK(hipStreamSynchronize(st));
    }

    std::vector<uint16_t> ours(size_t(S) * row), ref(size_t(S) * row);
    const int width = c.idx_topk + c.idx_pool - 1;  // indexer selection per query
    std::vector<int32_t> sel(size_t(S) * width), sel_ref(size_t(S) * width);
    const auto t0 = std::chrono::steady_clock::now();
    for (int L = 0; L < n_layers; ++L) {
      char name[64];
      if (isolate && L > 0) {
        std::snprintf(name, sizeof name, "/h_%03d.bin", L - 1);
        if (!read_file(oracle + name, ref.data(), ref.size() * 2))
          throw std::runtime_error("--isolate needs every layer's dump");
        for (int i = 0; i < slots; ++i)
          UH_CHECK(hipMemcpy(x.as<uint16_t>() + i * xslot, ref.data(), ref.size() * 2, hipMemcpyHostToDevice));
      }
      const auto tl = std::chrono::steady_clock::now();
      model::Glm5LoadOptions lo;
      lo.experts = &experts;
      lo.aux = ck.get();
      lo.kda_fp8 = kda_fp8;
      auto w = model::load_glm5_layer(dense, c, L, dev, st, lo);
      auto cache = model::make_glm5_cache(c, L, slots, max_seq, dev, st);
      auto run = [&](int s0, int n) {
        model::Glm5Batch b;
        b.T = n * slots;
        if (slots == 1) {
          b.chunks = {{0, s0, n, 0}};
          b.pos = dpos.as<int32_t>() + s0;
          model::glm5_layer_forward(c, w, cache, scr, x.as<uint16_t>() + size_t(s0) * row, b, st);
        } else {  // slot i's rows at batch rows [i n, i n + n)
          for (int i = 0; i < slots; ++i) {
            b.chunks.push_back({i, s0, n, i * n});
            UH_CHECK(hipMemcpyAsync(xb.as<uint16_t>() + size_t(i) * n * row,
                                    x.as<uint16_t>() + i * xslot + size_t(s0) * row, size_t(n) * row * 2,
                                    hipMemcpyDeviceToDevice, st));
            UH_CHECK(hipMemcpyAsync(bpos.as<int32_t>() + i * n, dpos.as<int32_t>() + s0, size_t(n) * 4,
                                    hipMemcpyDeviceToDevice, st));
          }
          b.pos = bpos.as<int32_t>();
          model::glm5_layer_forward(c, w, cache, scr, xb.as<uint16_t>(), b, st);
          for (int i = 0; i < slots; ++i)
            UH_CHECK(hipMemcpyAsync(x.as<uint16_t>() + i * xslot + size_t(s0) * row,
                                    xb.as<uint16_t>() + size_t(i) * n * row, size_t(n) * row * 2,
                                    hipMemcpyDeviceToDevice, st));
        }
        if (!w.kda)
          UH_CHECK(hipMemcpyAsync(sel.data() + size_t(s0) * width, scr.at<int32_t>(model::kIdxSel),
                                  size_t(n) * width * 4, hipMemcpyDeviceToHost, st));
      };
      for (int s0 = 0; s0 < P; s0 += chunk) run(s0, std::min(chunk, P - s0));
      for (int t = P; t < S; ++t) run(t, 1);
      UH_CHECK(hipStreamSynchronize(st));
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
      std::snprintf(name, sizeof name, "/h_%03d.bin", L);
      if (read_file(oracle + name, ref.data(), ref.size() * 2)) {
        UH_CHECK(hipMemcpy(ours.data(), x.data(), ours.size() * 2, hipMemcpyDeviceToHost));
        compare_layer(L, ours, ref, S, row, secs);
      } else {
        std::printf("layer %2d: %.1f s (no oracle dump)\n", L, secs);
        std::fflush(stdout);
      }
      std::snprintf(name, sizeof name, "/topk_%03d.bin", L);
      if (!w.kda && read_file(oracle + name, sel_ref.data(), sel_ref.size() * 4))
        compare_selection(L, sel, sel_ref, S, width);
      if (slots > 1) {
        std::vector<uint16_t> a(xslot), bb(xslot);
        UH_CHECK(hipMemcpy(a.data(), x.data(), xslot * 2, hipMemcpyDeviceToHost));
        int differ = 0;
        for (int i = 1; i < slots; ++i) {
          UH_CHECK(hipMemcpy(bb.data(), x.as<uint16_t>() + i * xslot, xslot * 2, hipMemcpyDeviceToHost));
          differ += a != bb;
        }
        std::printf("layer %2d: %d of %d slot copies differ from slot 0\n", L, differ, slots - 1);
      }
    }
    if (n_layers < c.n_layers) return 0;

    const int V = c.vocab;
    rt::DeviceBuffer logits(dev, size_t(S) * V * 4);
    {
      auto hw = model::load_glm5_head(dense, dev, st, false, true);
      model::glm5_head_forward(c, hw, scr, x.as<uint16_t>(), S, logits.as<float>(), st);
      UH_CHECK(hipStreamSynchronize(st));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> lo(size_t(S) * V), lr(size_t(S) * V);
    UH_CHECK(hipMemcpy(lo.data(), logits.data(), lo.size() * 4, hipMemcpyDeviceToHost));
    if (!read_file(oracle + "/logits.bin", lr.data(), lr.size() * 4)) throw std::runtime_error("no oracle logits.bin");
    if (!out_path.empty()) {
      std::ofstream f(out_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(lo.data()), std::streamsize(lo.size() * 4));
    }
    // KL(oracle || ours) per position, top-1 agreement.
    double kl_sum = 0, kl_max = 0;
    int agree = 0, kl_argmax = -1;
    std::vector<double> kls;
    for (int t = 0; t < S; ++t) {
      const float* a = lr.data() + size_t(t) * V;
      const float* b = lo.data() + size_t(t) * V;
      double ma = -1e300, mb = -1e300;
      int ia = 0, ib = 0;
      for (int v = 0; v < V; ++v) {
        if (a[v] > ma) ma = a[v], ia = v;
        if (b[v] > mb) mb = b[v], ib = v;
      }
      double za = 0, zb = 0;
      for (int v = 0; v < V; ++v) {
        za += std::exp(a[v] - ma);
        zb += std::exp(b[v] - mb);
      }
      const double la = std::log(za) + ma, lb = std::log(zb) + mb;
      double kl = 0;
      for (int v = 0; v < V; ++v) {
        const double lpa = a[v] - la, lpb = b[v] - lb;
        kl += std::exp(lpa) * (lpa - lpb);
      }
      kls.push_back(kl);
      kl_sum += kl;
      if (kl > kl_max) kl_max = kl, kl_argmax = t;
      agree += ia == ib;
      if (t >= P - 1) std::printf("  pos %4d: oracle argmax %6d ours %6d  KL %.3e\n", t, ia, ib, kl);
    }
    std::sort(kls.begin(), kls.end());
    std::printf("full model, %d positions (%d prompt + %d decode), %.1f s: mean KL %.3e  median %.3e  p99 %.3e  "
                "max %.3e (pos %d)  top-1 agreement %.2f%%\n",
                S, P, S - P, secs, kl_sum / S, kls[kls.size() / 2], kls[kls.size() * 99 / 100], kl_max, kl_argmax,
                100.0 * agree / S);
    UH_CHECK(hipStreamDestroy(st));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
