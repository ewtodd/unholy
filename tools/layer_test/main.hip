// unholy-layer-test: run one DeepSeek-V4 layer of the reference path on one GPU and compare it with
// the oracle's dump (tools/ref/oracle/run.py).
//
//   unholy-layer-test --ckpt /scratch/dsv4-ve --oracle DIR --layer L [--chunk C] [--device D]
//
// Input: the oracle's residual after layer L-1 (or the embedding for L = 0). The prompt is pushed
// through in chunks of C tokens (default: one chunk), then the continuation one token at a time, as
// the oracle did; the output is compared with the oracle's residual after layer L.
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

#include "core/half.hpp"
#include "gufo/json.hpp"
#include "kernels/ref.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

using namespace unholy;
using model::Dsv4Config;

namespace {

std::string read_text(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

std::vector<uint16_t> read_bf16(const std::string& p, size_t n) {
  std::vector<uint16_t> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 2));
  if (size_t(f.gcount()) != n * 2) throw std::runtime_error(p + ": short read");
  return v;
}

struct Stats {
  double num = 0, den = 0, maxd = 0, worst_rel = 0;
  int worst_pos = -1;
};

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt = "/scratch/dsv4-ve", oracle, exl3_dir;
  int layer = 0, chunk = 0, dev = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--oracle") oracle = next();
    else if (a == "--layer") layer = std::stoi(next());
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--device") dev = std::stoi(next());
    else if (a == "--exl3") exl3_dir = next();
    else {
      std::fprintf(stderr, "usage: %s --oracle DIR --layer L [--ckpt DIR] [--chunk C] [--device D]\n", argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(ckpt);
    std::unique_ptr<model::Checkpoint> xck;
    if (!exl3_dir.empty()) xck = model::Checkpoint::open(exl3_dir);
    const Dsv4Config c = Dsv4Config::from_hf(ck->config());
    const auto meta = gufo::json::parse(read_text(oracle + "/meta.json"));
    const int S = int(meta.member_size("S")), P = int(meta.member_size("prompt"));
    std::vector<int32_t> ids;
    for (const auto& v : meta.find("ids")->items()) ids.push_back(int32_t(v.as_double()));
    if (chunk <= 0) chunk = P;
    const size_t row = size_t(c.hc_mult) * c.dim;  // elements per position

    rt::DeviceGuard g(dev);
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));
    const auto t0 = std::chrono::steady_clock::now();
    model::LoadOptions lo;
    lo.exl3 = xck.get();
    const auto w = model::load_layer(*ck, c, layer, dev, st, lo);
    const auto t1 = std::chrono::steady_clock::now();
    const int max_seq = (S + 255) / 256 * 256;
    const model::CacheGeometry geo{1, max_seq / model::kPageTokens, max_seq / model::kPageTokens};
    auto cache = model::make_layer_cache(c, w, geo, dev, st);
    std::vector<int32_t> table(static_cast<size_t>(geo.max_pages_per_seq));
    for (size_t i = 0; i < table.size(); ++i) table[i] = int32_t(i);
    rt::DeviceBuffer bt(dev, table.size() * 4);
    UH_CHECK(hipMemcpy(bt.data(), table.data(), table.size() * 4, hipMemcpyHostToDevice));
    auto scr = model::make_scratch(c, dev, std::max(chunk, 1), max_seq);
    auto rope = model::make_rope_tables(c, dev, max_seq + 128);

    rt::DeviceBuffer x(dev, size_t(S) * row * 2);
    rt::DeviceBuffer dids(dev, size_t(S) * 4), dpos(dev, size_t(S) * 4);
    std::vector<int32_t> pos(static_cast<size_t>(S));
    for (int t = 0; t < S; ++t) pos[size_t(t)] = t;
    UH_CHECK(hipMemcpy(dids.data(), ids.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(dpos.data(), pos.data(), size_t(S) * 4, hipMemcpyHostToDevice));
    if (layer == 0) {
      const auto& et = ck->at("embed.weight");
      rt::DeviceBuffer table(dev, et.nbytes);
      UH_CHECK(hipMemcpy(table.data(), ck->bytes(et).data(), et.nbytes, hipMemcpyHostToDevice));
      kernels::embed_hc(dids.as<int32_t>(), table.as<uint16_t>(), x.as<uint16_t>(), S, c.dim, c.hc_mult, st);
    } else {
      char name[64];
      std::snprintf(name, sizeof name, "/h_%03d.bin", layer - 1);
      const auto in = read_bf16(oracle + name, size_t(S) * row);
      UH_CHECK(hipMemcpy(x.data(), in.data(), in.size() * 2, hipMemcpyHostToDevice));
    }

    auto run = [&](int s0, int n) {
      model::BatchDesc b;
      b.chunks = {{0, s0, n, 0}};
      b.T = n;
      b.ids = dids.as<int32_t>() + s0;
      b.pos = dpos.as<int32_t>() + s0;
      b.block_tables = bt.as<int32_t>();
      b.max_pages_per_seq = geo.max_pages_per_seq;
      model::layer_forward(c, w, cache, scr, rope, x.as<uint16_t>() + size_t(s0) * row, b, st);
    };
    for (int s0 = 0; s0 < P; s0 += chunk) run(s0, std::min(chunk, P - s0));
    UH_CHECK(hipStreamSynchronize(st));
    const auto t2 = std::chrono::steady_clock::now();
    for (int t = P; t < S; ++t) run(t, 1);
    UH_CHECK(hipStreamSynchronize(st));
    const auto t3 = std::chrono::steady_clock::now();

    std::vector<uint16_t> out(size_t(S) * row);
    UH_CHECK(hipMemcpy(out.data(), x.data(), out.size() * 2, hipMemcpyDeviceToHost));
    char name[64];
    std::snprintf(name, sizeof name, "/h_%03d.bin", layer);
    const auto ref = read_bf16(oracle + name, size_t(S) * row);

    Stats pre, dec;
    size_t exact = 0;
    for (int t = 0; t < S; ++t) {
      double num = 0, den = 0;
      Stats& s = t < P ? pre : dec;
      for (size_t i = 0; i < row; ++i) {
        const double a = bf16_to_float(out[size_t(t) * row + i]);
        const double b = bf16_to_float(ref[size_t(t) * row + i]);
        num += (a - b) * (a - b);
        den += b * b;
        s.maxd = std::max(s.maxd, std::fabs(a - b));
        exact += out[size_t(t) * row + i] == ref[size_t(t) * row + i];
      }
      s.num += num;
      s.den += den;
      const double rel = std::sqrt(num / std::max(den, 1e-30));
      if (rel > s.worst_rel) {
        s.worst_rel = rel;
        s.worst_pos = t;
      }
    }
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("layer %d (ratio %d%s): load %.0f ms, prefill %d tok in chunks of %d: %.0f ms, decode %d tok: %.0f ms\n",
                layer, w.ratio, w.hash ? ", hash" : "", ms(t0, t1), P, chunk, ms(t1, t2), S - P, ms(t2, t3));
    std::printf("  prefill: rel L2 %.3e  max|d| %.3e  worst pos %d (rel %.3e)\n", std::sqrt(pre.num / pre.den),
                pre.maxd, pre.worst_pos, pre.worst_rel);
    if (S > P)
      std::printf("  decode : rel L2 %.3e  max|d| %.3e  worst pos %d (rel %.3e)\n", std::sqrt(dec.num / dec.den),
                  dec.maxd, dec.worst_pos, dec.worst_rel);
    std::printf("  bit-identical elements: %.2f%%\n", 100.0 * double(exact) / double(out.size()));
    UH_CHECK(hipStreamDestroy(st));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
