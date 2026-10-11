// unholy-ref-run: the whole model on the reference path, layer-streamed on one GPU (each layer is
// loaded, run over every position, and freed, like the oracle), then the head; compares the logits
// with the oracle's (tools/ref/oracle/run.py) by KL divergence and top-1 agreement.
//
//   unholy-ref-run --oracle DIR [--ckpt DIR] [--chunk C] [--device D] [--out logits.bin]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gufo/json.hpp"
#include "kernels/ref.hpp"
#include "model/dsv4.hpp"
#include "model/loader.hpp"
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

rt::DeviceBuffer upload(model::Uploader& up, const model::Checkpoint& ck, const std::string& name, int dev,
                        hipStream_t st) {
  const auto& t = ck.at(name);
  rt::DeviceBuffer b(dev, t.nbytes);
  up.upload(t, b.data(), st);
  up.sync();
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = "/scratch/dsv4-ve", oracle, exl3_dir, out_path;
  int chunk = 0, dev = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--oracle") oracle = next();
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--device") dev = std::stoi(next());
    else if (a == "--exl3") exl3_dir = next();
    else if (a == "--out") out_path = next();
    else {
      std::fprintf(stderr, "usage: %s --oracle DIR [--ckpt DIR] [--chunk C] [--device D] [--out F]\n", argv[0]);
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
    if (chunk <= 0) chunk = P;
    const size_t row = size_t(c.hc_mult) * c.dim;

    rt::DeviceGuard g(dev);
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));
    model::Uploader up(*ck);
    std::unique_ptr<model::Uploader> xup;
    if (xck) xup = std::make_unique<model::Uploader>(*xck);
    const int max_seq = (S + 255) / 256 * 256;
    auto scr = model::make_scratch(c, dev, std::max(chunk, 1), max_seq);
    auto rope = model::make_rope_tables(c, dev, max_seq + 128);
    rt::DeviceBuffer x(dev, size_t(S) * row * 2);
    rt::DeviceBuffer dids(dev, size_t(S) * 4), dpos(dev, size_t(S) * 4);
    std::vector<int32_t> pos(static_cast<size_t>(S));
    for (int t = 0; t < S; ++t) pos[size_t(t)] = t;
    UH_CHECK(hipMemcpy(dids.data(), ids.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dpos.data(), pos.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    {
      auto table = upload(up, *ck, "embed.weight", dev, st);
      kernels::embed_hc(dids.as<int32_t>(), table.as<uint16_t>(), x.as<uint16_t>(), S, c.dim, c.hc_mult, st);
      UH_CHECK(hipStreamSynchronize(st));
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int L = 0; L < c.n_layers; ++L) {
      const auto tl = std::chrono::steady_clock::now();
      auto w = model::load_layer(*ck, c, L, dev, st, &up, xck.get(), xup.get());
      auto state = model::make_layer_state(c, w, dev, max_seq, st);
      auto run = [&](int s0, int n) {
        model::layer_forward(c, w, state, scr, rope, x.as<uint16_t>() + size_t(s0) * row, dids.as<int32_t>() + s0,
                             dpos.as<int32_t>() + s0, s0, n, st);
      };
      for (int s0 = 0; s0 < P; s0 += chunk) run(s0, std::min(chunk, P - s0));
      for (int t = P; t < S; ++t) run(t, 1);
      UH_CHECK(hipStreamSynchronize(st));
      std::printf("layer %2d: %.1f s\n", L,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
      std::fflush(stdout);
    }

    // Head: hc_head -> norm -> fp32 linear.
    const int V = c.vocab;
    rt::DeviceBuffer hb(dev, size_t(S) * c.dim * 2), hf(dev, size_t(S) * c.dim * 4);
    rt::DeviceBuffer logits(dev, size_t(S) * V * 4);
    {
      auto fn = upload(up, *ck, "hc_head_fn", dev, st), sc = upload(up, *ck, "hc_head_scale", dev, st),
           base = upload(up, *ck, "hc_head_base", dev, st), nw = upload(up, *ck, "norm.weight", dev, st),
           head = upload(up, *ck, "head.weight", dev, st);
      kernels::hc_head(x.as<uint16_t>(), fn.as<float>(), sc.as<float>(), base.as<float>(), hb.as<uint16_t>(), S,
                       c.hc_mult, c.dim, c.norm_eps, c.hc_eps, st);
      kernels::rmsnorm(hb.as<uint16_t>(), c.dim, nw.as<uint16_t>(), hb.as<uint16_t>(), c.dim, S, c.dim, c.norm_eps,
                       st);
      kernels::cast_bf16_f32(hb.as<uint16_t>(), hf.as<float>(), size_t(S) * c.dim, st);
      kernels::gemm_f32(hf.as<float>(), c.dim, head.as<uint16_t>(), logits.as<float>(), V, S, V, c.dim, st);
      UH_CHECK(hipStreamSynchronize(st));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<float> ours(size_t(S) * V), ref(size_t(S) * V);
    UH_CHECK(hipMemcpy(ours.data(), logits.data(), ours.size() * 4, hipMemcpyDeviceToHost));
    {
      std::ifstream f(oracle + "/logits.bin", std::ios::binary);
      if (!f) throw std::runtime_error("no oracle logits.bin");
      f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(ref.size() * 4));
    }
    if (!out_path.empty()) {
      std::ofstream f(out_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(ours.data()), std::streamsize(ours.size() * 4));
    }
    // KL(oracle || ours) per position, top-1 agreement.
    double kl_sum = 0, kl_max = 0;
    int agree = 0, kl_argmax = -1;
    std::vector<double> kls;
    for (int t = 0; t < S; ++t) {
      const float* a = ref.data() + size_t(t) * V;
      const float* b = ours.data() + size_t(t) * V;
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
      if (t == P - 1 || t >= P)
        std::printf("  pos %4d: oracle argmax %6d ours %6d  KL %.3e\n", t, ia, ib, kl);
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
