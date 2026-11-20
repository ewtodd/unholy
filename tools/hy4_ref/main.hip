// unholy-hy4-ref: Hy4 flash (model/hy4.hpp) layer-streamed on one GPU like the oracle (tools/ref/hy4_oracle/run.py):
// each layer is loaded with all its experts, run over every position (the prompt in chunks, then the teacher-forced
// continuation one token at a time), compared with the oracle's dump of its output, and freed; then the head,
// compared with the oracle's logits by KL divergence and top-1 agreement.
//
//   unholy-hy4-ref --oracle DIR [--ckpt FP8_DIR] [--engram DIR] [--chunk C] [--device D] [--isolate] [--layers N]
//                  [--dump DIR] [--out logits.bin] [--mxfp4]
//
// --mxfp4 converts the routed experts to MXFP4 at load (model::Hy4LoadOptions::mxfp4); --experts-gguf FILE takes them
// in NVFP4 from a GGUF build (AngelSlim's SPARK-4Bit).
//
// --isolate starts every layer from the oracle's input to it, so each layer's own error shows.
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
#include "model/hy4.hpp"
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

template <typename T>
T* host_mapped(size_t n) {
  void* p = nullptr;
  UH_CHECK(hipHostMalloc(&p, std::max<size_t>(n * sizeof(T), 64), hipHostMallocPortable | hipHostMallocMapped));
  return static_cast<T*>(p);
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = "/scratch/models/hy4-flash-fp8", engram_dir = "/scratch/models/hy4-flash-engram", oracle, experts_gguf,
              out_path, dump_dir;
  int chunk = 0, dev = 0, n_layers = -1;
  bool isolate = false, mxfp4 = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--oracle") oracle = next();
    else if (a == "--engram") engram_dir = next();
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--device") dev = std::stoi(next());
    else if (a == "--layers") n_layers = std::stoi(next());
    else if (a == "--isolate") isolate = true;
    else if (a == "--mxfp4") mxfp4 = true;
    else if (a == "--experts-gguf") experts_gguf = next();
    else if (a == "--dump") dump_dir = next();
    else if (a == "--out") out_path = next();
    else {
      std::fprintf(stderr,
                   "usage: %s --oracle DIR [--ckpt DIR] [--engram DIR] [--chunk C] [--device D] [--isolate] "
                   "[--layers N] [--dump DIR] [--out F]\n",
                   argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(ckpt);
    std::unique_ptr<model::Checkpoint> gk;
    if (!experts_gguf.empty()) gk = model::Checkpoint::open_gguf({experts_gguf});
    const auto c = model::Hy4Config::from_hf(ck->config());
    model::Hy4Engram eng(engram_dir, c);
    const auto meta = gufo::json::parse(read_text(oracle + "/meta.json"));
    const int S = int(meta.member_size("S")), P = int(meta.member_size("prompt"));
    std::vector<int32_t> ids;
    for (const auto& v : meta.find("ids")->items()) ids.push_back(int32_t(v.as_double()));
    if (chunk <= 0) chunk = P;
    if (n_layers < 0) n_layers = c.n_layers;
    const size_t row = size_t(c.hc) * c.dim;

    // Engram rows of every position, per memory (mapped pinned host memory the kernels read directly).
    std::vector<int32_t> ngram(size_t(S) * 3);
    std::vector<uint8_t> mask_h(static_cast<size_t>(S));
    for (int t = 0; t < S; ++t) {
      for (int j = 0; j < 3; ++j) ngram[size_t(t) * 3 + j] = t - j >= 0 ? ids[size_t(t - j)] : c.engram_pad;
      const int id = ids[size_t(t)];
      mask_h[size_t(t)] = id != c.image_token && id != c.video_token && id != c.engram_pad;
    }
    uint8_t* emask = host_mapped<uint8_t>(size_t(S));
    std::memcpy(emask, mask_h.data(), size_t(S));
    std::vector<uint8_t*> eq(size_t(eng.memories()));
    std::vector<uint16_t*> es(size_t(eng.memories()));
    const auto te = std::chrono::steady_clock::now();
    for (int m = 0; m < eng.memories(); ++m) {
      eq[size_t(m)] = host_mapped<uint8_t>(size_t(S) * 16 * 96);
      es[size_t(m)] = host_mapped<uint16_t>(size_t(S) * 16);
      eng.lookup(m, ngram.data(), mask_h.data(), S, eq[size_t(m)], es[size_t(m)]);
    }
    std::printf("engram rows: %d memories x %d positions in %.2f s\n", eng.memories(), S,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count());

    rt::DeviceGuard g(dev);
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));
    const int max_seq = (S + 255) / 256 * 256;
    auto scr = model::make_hy4_scratch(c, dev, std::max(chunk, 1), max_seq);
    rt::DeviceBuffer x(dev, size_t(S) * row * 2), dpos(dev, size_t(S + 1) * 4), dids(dev, size_t(S) * 4);
    const int pps = max_seq / kernels::kHy4PageRows;
    rt::DeviceBuffer dtables(dev, size_t(pps) * 4);
    {
      std::vector<int32_t> tab(static_cast<size_t>(pps));
      for (size_t i = 0; i < tab.size(); ++i) tab[i] = int32_t(i);
      UH_CHECK(hipMemcpy(dtables.data(), tab.data(), tab.size() * 4, hipMemcpyHostToDevice));
    }
    std::vector<int32_t> pos(static_cast<size_t>(S + 1));
    for (int t = 0; t <= S; ++t) pos[size_t(t)] = t;
    UH_CHECK(hipMemcpy(dpos.data(), pos.data(), size_t(S + 1) * 4, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dids.data(), ids.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    {
      auto hw = model::load_hy4_head(*ck, c, dev, st, true, false);
      model::hy4_embed(c, hw, dids.as<int32_t>(), x.as<uint16_t>(), S, st);
      UH_CHECK(hipStreamSynchronize(st));
    }

    std::vector<uint16_t> ours(size_t(S) * row), ref(size_t(S) * row);
    const auto t0 = std::chrono::steady_clock::now();
    double t_prefill = 0, t_decode = 0;
    for (int L = 0; L < n_layers; ++L) {
      char name[64];
      if (isolate && L > 0) {
        std::snprintf(name, sizeof name, "/h_%03d.bin", L - 1);
        if (!read_file(oracle + name, ref.data(), ref.size() * 2))
          throw std::runtime_error("--isolate needs every layer's dump");
        UH_CHECK(hipMemcpy(x.data(), ref.data(), ref.size() * 2, hipMemcpyHostToDevice));
      }
      const auto tl = std::chrono::steady_clock::now();
      model::Hy4LoadOptions lo;
      lo.mxfp4 = mxfp4;
      lo.experts_gguf = gk.get();
      auto w = model::load_hy4_layer(*ck, c, L, dev, st, lo);
      auto cache = model::make_hy4_cache(c, L, 1, pps, dev, st);
      auto run = [&](int s0, int n) {
        model::Hy4Batch b;
        b.T = n;
        b.chunks = {{0, s0, n, 0}};
        b.pos = dpos.as<int32_t>() + s0;
        b.pos_host = pos.data() + s0;
        b.tables = dtables.as<int32_t>();
        b.max_pages = pps;
        for (int m = 0; m < eng.memories(); ++m) {
          b.eng_q[m] = eq[size_t(m)] + size_t(s0) * 16 * 96;
          b.eng_s[m] = es[size_t(m)] + size_t(s0) * 16;
        }
        b.eng_mask = emask + s0;
        model::hy4_layer_forward(c, w, cache, scr, x.as<uint16_t>() + size_t(s0) * row, b, st);
      };
      UH_CHECK(hipStreamSynchronize(st));
      const auto tp = std::chrono::steady_clock::now();
      for (int s0 = 0; s0 < P; s0 += chunk) run(s0, std::min(chunk, P - s0));
      UH_CHECK(hipStreamSynchronize(st));
      const auto td = std::chrono::steady_clock::now();
      for (int t = P; t < S; ++t) run(t, 1);
      UH_CHECK(hipStreamSynchronize(st));
      const double pre_s = std::chrono::duration<double>(td - tp).count();
      const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
      t_prefill += pre_s;
      t_decode += dec_s;
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
      std::printf("layer %2d (%s%s): prefill %.1f ms (%d tokens), decode %.2f ms/token\n", L,
                  w.sparse ? "sparse" : "window", w.engram >= 0 ? " + engram" : "", pre_s * 1e3, P,
                  S > P ? dec_s * 1e3 / (S - P) : 0.0);
      if (!dump_dir.empty()) {
        UH_CHECK(hipMemcpy(ours.data(), x.data(), ours.size() * 2, hipMemcpyDeviceToHost));
        std::snprintf(name, sizeof name, "/h_%03d.bin", L);
        std::ofstream(dump_dir + name, std::ios::binary).write(reinterpret_cast<const char*>(ours.data()),
                                                                std::streamsize(ours.size() * 2));
      }
      std::snprintf(name, sizeof name, "/h_%03d.bin", L);
      if (read_file(oracle + name, ref.data(), ref.size() * 2)) {
        UH_CHECK(hipMemcpy(ours.data(), x.data(), ours.size() * 2, hipMemcpyDeviceToHost));
        compare_layer(L, ours, ref, S, row, secs);
      } else {
        std::printf("layer %2d: %.1f s (no oracle dump)\n", L, secs);
        std::fflush(stdout);
      }
    }
    std::printf("layers: prefill %.2f s for %d tokens (%.0f tok/s), decode %.1f ms/token\n", t_prefill, P,
                P / std::max(t_prefill, 1e-9), S > P ? t_decode * 1e3 / (S - P) : 0.0);
    if (n_layers < c.n_layers) return 0;

    const int V = c.vocab;
    rt::DeviceBuffer logits(dev, size_t(S) * V * 4);
    {
      auto hw = model::load_hy4_head(*ck, c, dev, st, false, true);
      model::hy4_head_forward(c, hw, scr, x.as<uint16_t>(), S, logits.as<float>(), st);
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
