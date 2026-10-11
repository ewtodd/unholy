// unholy-ckpt: inspect checkpoints and dump reference decodes for cross-checking.
//
//   unholy-ckpt info <dir>
//   unholy-ckpt exl3 <dir> <prefix> <out.bin> [--inner]
//       --inner: W_inner as fp16 [in][out]; default: dense W as f32 [in][out]
//   unholy-ckpt compare <native_dir> <exl3_dir> <prefix>...
//       weight-space error of the EXL3 linear against the native (MXFP4 / fp8) one
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "model/exl3.hpp"
#include "model/native.hpp"
#include "model/safetensors.hpp"

using namespace unholy::model;

static int cmd_info(const std::string& dir) {
  auto ck = Checkpoint::open(dir);
  std::map<std::string, std::pair<size_t, uint64_t>> by_dtype;
  for (const auto& t : ck->tensors()) {
    auto& e = by_dtype[dtype_name(t.dtype)];
    e.first++;
    e.second += t.nbytes;
  }
  std::printf("%s: %zu shards, %zu tensors, %.3f GB\n", dir.c_str(), ck->shard_paths().size(),
              ck->tensors().size(), ck->total_bytes() / 1e9);
  for (const auto& [name, e] : by_dtype)
    std::printf("  %-8s %8zu tensors %10.3f GB\n", name.c_str(), e.first, e.second / 1e9);
  return 0;
}

static int cmd_exl3(const std::string& dir, const std::string& prefix, const std::string& out, bool inner) {
  auto ck = Checkpoint::open(dir);
  auto l = find_exl3_linear(*ck, prefix);
  if (!l) {
    std::fprintf(stderr, "%s: no EXL3 linear %s\n", dir.c_str(), prefix.c_str());
    return 1;
  }
  std::printf("%s: in %d out %d K %d codebook %d\n", prefix.c_str(), l->in_features, l->out_features, l->K,
              int(l->cb));
  const size_t n = size_t(l->in_features) * l->out_features;
  FILE* f = std::fopen(out.c_str(), "wb");
  if (!f) {
    std::perror(out.c_str());
    return 1;
  }
  if (inner) {
    std::vector<uint16_t> w(n);
    exl3_reconstruct_inner_cpu(*ck, *l, w.data());
    std::fwrite(w.data(), 2, n, f);
  } else {
    std::vector<float> w(n);
    exl3_dense_cpu(*ck, *l, w.data());
    std::fwrite(w.data(), 4, n, f);
  }
  std::fclose(f);
  return 0;
}

static int cmd_compare(const std::string& native_dir, const std::string& exl3_dir,
                       const std::vector<std::string>& prefixes) {
  auto nat = Checkpoint::open(native_dir);
  auto ex = Checkpoint::open(exl3_dir);
  for (const auto& prefix : prefixes) {
    auto l = find_exl3_linear(*ex, prefix);
    if (!l) {
      std::fprintf(stderr, "%s: not EXL3 in %s\n", prefix.c_str(), exl3_dir.c_str());
      return 1;
    }
    const size_t n = size_t(l->in_features) * l->out_features;
    std::vector<float> we(n), wn(n);
    exl3_dense_cpu(*ex, *l, we.data());  // [in][out]
    int out = 0, in = 0;
    native_dense_cpu(*nat, prefix, wn.data(), &out, &in);  // [out][in]
    if (out != l->out_features || in != l->in_features) {
      std::fprintf(stderr, "%s: shape mismatch native [%d,%d] exl3 [in %d, out %d]\n", prefix.c_str(), out, in,
                   l->in_features, l->out_features);
      return 1;
    }
    double num = 0, den = 0, dot = 0, ne = 0, maxd = 0;
    for (int o = 0; o < out; ++o)
      for (int i = 0; i < in; ++i) {
        const double a = wn[size_t(o) * in + i], b = we[size_t(i) * out + o];
        num += (a - b) * (a - b);
        den += a * a;
        ne += b * b;
        dot += a * b;
        maxd = std::max(maxd, std::fabs(a - b));
      }
    std::printf("%-36s [%5d x %5d] K=%d  rel L2 %.4f  cos %.5f  max|d| %.3e\n", prefix.c_str(), out, in, l->K,
                std::sqrt(num / den), dot / std::sqrt(den * ne), maxd);
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc >= 3 && !std::strcmp(argv[1], "info")) return cmd_info(argv[2]);
    if (argc >= 5 && !std::strcmp(argv[1], "exl3"))
      return cmd_exl3(argv[2], argv[3], argv[4], argc >= 6 && !std::strcmp(argv[5], "--inner"));
    if (argc >= 5 && !std::strcmp(argv[1], "compare"))
      return cmd_compare(argv[2], argv[3], std::vector<std::string>(argv + 4, argv + argc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "usage: %s info <dir> | exl3 <dir> <prefix> <out.bin> [--inner]\n", argv[0]);
  return 2;
}
