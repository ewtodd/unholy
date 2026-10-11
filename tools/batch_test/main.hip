// unholy-batch-test: batching and paging self-consistency for one layer of the reference path.
//
// Three sequences (different slices of an id file, each starting at position 0) are run (a) alone,
// one chunk, identity block table; (b) interleaved in mixed batches with random chunk sizes (some 1:
// decode-shaped) and shuffled physical pages. Every op is row-independent or per-chunk, so (b) must
// reproduce (a) bit for bit.
//
//   unholy-batch-test --layer L --ids FILE [--ckpt DIR] [--device D] [--seed S]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"
#include "model/loader.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

using namespace unholy;

int main(int argc, char** argv) {
  std::string ckpt = "/scratch/dsv4-ve", ids_path;
  int layer = 2, dev = 0;
  unsigned seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--ids") ids_path = next();
    else if (a == "--layer") layer = std::stoi(next());
    else if (a == "--device") dev = std::stoi(next());
    else if (a == "--seed") seed = unsigned(std::stoul(next()));
    else {
      std::fprintf(stderr, "usage: %s --layer L --ids FILE [--ckpt DIR] [--device D] [--seed S]\n", argv[0]);
      return 2;
    }
  }
  try {
    auto ck = model::Checkpoint::open(ckpt);
    const auto c = model::Dsv4Config::from_hf(ck->config());
    std::vector<uint32_t> all;
    {
      std::ifstream f(ids_path, std::ios::binary);
      f.seekg(0, std::ios::end);
      all.resize(size_t(f.tellg()) / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(all.data()), std::streamsize(all.size() * 4));
    }
    struct Seq {
      int begin, len;
    };
    const std::vector<Seq> seqs = {{0, 600}, {1000, 523}, {2000, 333}};
    const int nseq = int(seqs.size());
    const size_t row = size_t(c.hc_mult) * c.dim;
    const int max_seq = 1024;
    const int pages_per_seq = max_seq / model::kPageTokens;

    rt::DeviceGuard g(dev);
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));
    const auto w = model::load_layer(*ck, c, layer, dev, st);
    auto rope = model::make_rope_tables(c, dev, max_seq + 128);
    auto scr = model::make_scratch(c, dev, 1024, max_seq);
    rt::DeviceBuffer table_buf;
    {
      const auto& et = ck->at("embed.weight");
      table_buf = rt::DeviceBuffer(dev, et.nbytes);
      model::Uploader up(*ck);
      up.upload(et, table_buf.data(), st);
      up.sync();
    }

    // Per sequence: ids, positions, input residual (embedding), solo output.
    std::vector<rt::DeviceBuffer> ids(nseq), pos(nseq), xin(nseq);
    std::vector<std::vector<uint16_t>> solo(nseq);
    for (int s = 0; s < nseq; ++s) {
      std::vector<int32_t> hid(size_t(seqs[s].len)), hpos(size_t(seqs[s].len));
      for (int t = 0; t < seqs[s].len; ++t) {
        hid[size_t(t)] = int32_t(all.at(size_t(seqs[s].begin + t)));
        hpos[size_t(t)] = t;
      }
      ids[s] = rt::DeviceBuffer(dev, hid.size() * 4);
      pos[s] = rt::DeviceBuffer(dev, hpos.size() * 4);
      UH_CHECK(hipMemcpy(ids[s].data(), hid.data(), hid.size() * 4, hipMemcpyHostToDevice));
      UH_CHECK(hipMemcpy(pos[s].data(), hpos.data(), hpos.size() * 4, hipMemcpyHostToDevice));
      xin[s] = rt::DeviceBuffer(dev, size_t(seqs[s].len) * row * 2);
      kernels::embed_hc(ids[s].as<int32_t>(), table_buf.as<uint16_t>(), xin[s].as<uint16_t>(), seqs[s].len, c.dim,
                        c.hc_mult, st);
    }
    UH_CHECK(hipStreamSynchronize(st));

    // (a) solo: slot 0, identity table, one chunk.
    {
      const model::CacheGeometry geo{1, pages_per_seq, pages_per_seq};
      std::vector<int32_t> table(static_cast<size_t>(pages_per_seq));
      std::iota(table.begin(), table.end(), 0);
      rt::DeviceBuffer bt(dev, table.size() * 4);
      UH_CHECK(hipMemcpy(bt.data(), table.data(), table.size() * 4, hipMemcpyHostToDevice));
      for (int s = 0; s < nseq; ++s) {
        auto cache = model::make_layer_cache(c, w, geo, dev, st);
        rt::DeviceBuffer x(dev, xin[s].size());
        UH_CHECK(hipMemcpy(x.data(), xin[s].data(), x.size(), hipMemcpyDeviceToDevice));
        model::BatchDesc b;
        b.chunks = {{0, 0, seqs[s].len, 0}};
        b.T = seqs[s].len;
        b.ids = ids[s].as<int32_t>();
        b.pos = pos[s].as<int32_t>();
        b.block_tables = bt.as<int32_t>();
        b.max_pages_per_seq = pages_per_seq;
        model::layer_forward(c, w, cache, scr, rope, x.as<uint16_t>(), b, st);
        solo[s].resize(x.size() / 2);
        UH_CHECK(hipMemcpy(solo[s].data(), x.data(), x.size(), hipMemcpyDeviceToHost));
      }
    }

    // (b) interleaved: slots 0..2 (shuffled: seq s -> slot perm[s]), shuffled physical pages.
    std::mt19937 rng(seed);
    const model::CacheGeometry geo{nseq, pages_per_seq, nseq * pages_per_seq};
    std::vector<int32_t> phys(size_t(geo.num_pages));
    std::iota(phys.begin(), phys.end(), 0);
    std::shuffle(phys.begin(), phys.end(), rng);
    std::vector<int> slot_of(static_cast<size_t>(nseq));
    std::iota(slot_of.begin(), slot_of.end(), 0);
    std::shuffle(slot_of.begin(), slot_of.end(), rng);
    std::vector<int32_t> tables(size_t(nseq) * pages_per_seq);
    for (int s = 0; s < nseq; ++s)
      for (int p = 0; p < pages_per_seq; ++p)
        tables[size_t(slot_of[s]) * pages_per_seq + p] = phys[size_t(s) * pages_per_seq + p];
    rt::DeviceBuffer bt(dev, tables.size() * 4);
    UH_CHECK(hipMemcpy(bt.data(), tables.data(), tables.size() * 4, hipMemcpyHostToDevice));
    auto cache = model::make_layer_cache(c, w, geo, dev, st);
    std::vector<rt::DeviceBuffer> xout(nseq);
    for (int s = 0; s < nseq; ++s) {
      xout[s] = rt::DeviceBuffer(dev, xin[s].size());
      UH_CHECK(hipMemcpy(xout[s].data(), xin[s].data(), xin[s].size(), hipMemcpyDeviceToDevice));
    }
    rt::DeviceBuffer bx(dev, size_t(1024) * row * 2), bids(dev, 1024 * 4), bpos(dev, 1024 * 4);
    std::vector<int> done(size_t(nseq), 0);
    int steps = 0;
    std::uniform_int_distribution<int> pick(0, 3);
    while (true) {
      model::BatchDesc b;
      b.block_tables = bt.as<int32_t>();
      b.max_pages_per_seq = pages_per_seq;
      for (int s = 0; s < nseq; ++s) {
        if (done[s] >= seqs[s].len) continue;
        const int r = pick(rng);
        const int want = r == 0 ? 1 : (r == 1 ? 37 : (r == 2 ? 128 : 211));  // decode-shaped and odd sizes
        const int n = std::min(want, seqs[s].len - done[s]);
        if (b.T + n > 1024) continue;
        b.chunks.push_back({slot_of[s], done[s], n, b.T});
        // Gather this chunk's rows into the batch buffers.
        UH_CHECK(hipMemcpyAsync(bx.as<uint16_t>() + size_t(b.T) * row, xout[s].as<uint16_t>() + size_t(done[s]) * row,
                                size_t(n) * row * 2, hipMemcpyDeviceToDevice, st));
        UH_CHECK(hipMemcpyAsync(bids.as<int32_t>() + b.T, ids[s].as<int32_t>() + done[s], size_t(n) * 4,
                                hipMemcpyDeviceToDevice, st));
        UH_CHECK(hipMemcpyAsync(bpos.as<int32_t>() + b.T, pos[s].as<int32_t>() + done[s], size_t(n) * 4,
                                hipMemcpyDeviceToDevice, st));
        b.T += n;
      }
      if (b.chunks.empty()) break;
      b.ids = bids.as<int32_t>();
      b.pos = bpos.as<int32_t>();
      model::layer_forward(c, w, cache, scr, rope, bx.as<uint16_t>(), b, st);
      for (const auto& ch : b.chunks) {
        const int s = int(std::find(slot_of.begin(), slot_of.end(), ch.slot) - slot_of.begin());
        UH_CHECK(hipMemcpyAsync(xout[s].as<uint16_t>() + size_t(ch.s0) * row, bx.as<uint16_t>() + size_t(ch.t0) * row,
                                size_t(ch.n) * row * 2, hipMemcpyDeviceToDevice, st));
        done[s] += ch.n;
      }
      ++steps;
    }
    UH_CHECK(hipStreamSynchronize(st));
    bool ok = true;
    for (int s = 0; s < nseq; ++s) {
      std::vector<uint16_t> got(solo[s].size());
      UH_CHECK(hipMemcpy(got.data(), xout[s].data(), got.size() * 2, hipMemcpyDeviceToHost));
      size_t diff = 0;
      for (size_t i = 0; i < got.size(); ++i) diff += got[i] != solo[s][i];
      std::printf("layer %d seq %d (%d tokens, slot %d): %zu / %zu elements differ from the solo run\n", layer, s,
                  seqs[s].len, slot_of[s], diff, got.size());
      ok &= diff == 0;
    }
    std::printf("%d interleaved steps: %s\n", steps, ok ? "PASS (bit-identical)" : "FAIL");
    UH_CHECK(hipStreamDestroy(st));
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
