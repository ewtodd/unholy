// unholy-expert-pack-check: the pack's slots against what the loader leaves on the device.
//
//   unholy-expert-pack-check --model DIR --pack FILE [--layers 0,21,42,m0] [--device 0]
//
// Loads each listed layer's routed experts with model::load_layer (which repacks them on the device with
// kernels::mx_repack) and compares every expert's device bytes with its pack slot, byte for byte. Layers as in
// unholy-expert-pack: plain numbers are main layers, m<i> the DSpark stages.
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/dsv4.hpp"
#include "model/expert_pack.hpp"
#include "rt/hip.hpp"

using namespace unholy;
using namespace unholy::model;
using namespace unholy::model::pack;

int main(int argc, char** argv) {
  std::string model, packf, layers = "0,21,42,m0";
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model") model = val();
    else if (a == "--pack") packf = val();
    else if (a == "--layers") layers = val();
    else if (a == "--device") device = std::stoi(val());
    else {
      std::fprintf(stderr, "usage: unholy-expert-pack-check --model DIR --pack FILE [--layers 0,21,42,m0] [--device D]\n");
      return 2;
    }
  }
  try {
    auto ck = Checkpoint::open(model);
    const Dsv4Config cfg = Dsv4Config::from_hf(ck->config());
    int fd = ::open(packf.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("open " + packf);
    const Header h = read_header(fd);
    const Geometry g = geometry_from(h);
    if (h.source_hash != source_hash(*ck, g)) throw std::runtime_error("the pack was built from another checkpoint");
    UH_CHECK(hipSetDevice(device));
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));

    uint8_t* slot = alloc_aligned(aligned_buffer_bytes(h.slot_bytes));
    std::vector<uint8_t> host(h.slot_bytes);
    size_t total_bad = 0, total = 0;
    // Layer list: "0,21,42,m0".
    size_t pos = 0;
    while (pos < layers.size()) {
      size_t end = layers.find(',', pos);
      if (end == std::string::npos) end = layers.size();
      std::string tok = layers.substr(pos, end - pos);
      pos = end + 1;
      const bool mtp = !tok.empty() && tok[0] == 'm';
      const int idx = std::stoi(mtp ? tok.substr(1) : tok);
      const int group = mtp ? g.n_layers + idx : idx;
      LoadOptions opt;
      opt.dense = false;
      if (mtp) opt.prefix = "mtp." + std::to_string(idx) + ".";
      LayerWeights lw = load_layer(*ck, cfg, mtp ? cfg.n_layers + idx : idx, device, st, opt);
      UH_CHECK(hipStreamSynchronize(st));
      size_t bad = 0;
      for (int e = 0; e < g.n_experts; ++e) {
        const size_t si = g.slot_index(group, e);
        const uint8_t* want = read_aligned(fd, h.slots_offset + si * h.slot_bytes, h.slot_bytes, slot);
        const Fp4Linear* m[3] = {&lw.w1[size_t(e)], &lw.w3[size_t(e)], &lw.w2[size_t(e)]};
        for (int k = 0; k < 3; ++k) {
          const size_t nc = size_t(m[k]->out) * m[k]->in / 2, ns = size_t(m[k]->out) * m[k]->in / 32;
          if (nc != g.part_bytes[2 * k] || ns != g.part_bytes[2 * k + 1] || !m[k]->tiled)
            throw std::runtime_error("device matrix shape / layout differs from the pack's geometry");
          UH_CHECK(hipMemcpy(host.data(), m[k]->w.ptr, nc, hipMemcpyDeviceToHost));
          UH_CHECK(hipMemcpy(host.data() + nc, m[k]->s.ptr, ns, hipMemcpyDeviceToHost));
          if (std::memcmp(host.data(), want + g.part_offset[2 * k], nc) ||
              std::memcmp(host.data() + nc, want + g.part_offset[2 * k + 1], ns)) {
            ++bad;
            std::fprintf(stderr, "%s%d: expert %d matrix %d differs\n", mtp ? "mtp." : "layer ", idx, e, k);
          }
        }
      }
      std::printf("%s%d: %d experts compared, %zu differ\n", mtp ? "mtp " : "layer ", idx, g.n_experts, bad);
      total_bad += bad;
      total += size_t(g.n_experts);
    }
    std::printf("%s: %zu experts, %zu mismatching matrices\n", total_bad ? "FAIL" : "PASS", total, total_bad);
    return total_bad ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unholy-expert-pack-check: %s\n", e.what());
    return 1;
  }
}
