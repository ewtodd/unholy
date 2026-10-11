// The reference forward over a batch of sequence chunks, with paged compressed-row pools.
//
// A batch is a list of chunks: tokens [t0, t0 + n) of the batch are positions [s0, s0 + n) of the
// sequence in slot `slot` (a prefill chunk, or one decode token). Everything token-parallel (HC
// mixing, norms, projections, wo_a/wo_b, the MoE) runs over all T tokens at once; what depends on a
// sequence's history (compressors, indexer, attention, the window ring) runs per chunk.
//
// Paging: a page covers kPageTokens positions of a sequence in every layer — 64 CSA rows, 2 HCA rows,
// 64 indexer rows. One block table per slot maps logical pages to physical pages of every pool on the
// device (each layer has its own pools; physical page ids are shared across layers), so the cache
// manager allocates pages once per sequence position range, not per layer.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "model/dsv4.hpp"

namespace unholy::model {

constexpr int kPageTokens = 256;

struct CacheGeometry {
  int max_slots = 0;          // concurrent sequences
  int max_pages_per_seq = 0;  // block-table width
  int num_pages = 0;          // physical pages per pool
};

struct LayerCache {
  int layer = 0, ratio = 0;
  bool indexer = false;
  int slots = 0, window = 0, ring = 0, width = 0, iring = 0, iwidth = 0;
  rt::DeviceBuffer comp, idx;                            // bf16 [pages][rows_per_page][d]
  rt::DeviceBuffer window_rows;                          // bf16 [slots][window][head_dim]
  rt::DeviceBuffer ring_kv, ring_sc, iring_kv, iring_sc;  // f32 [slots][ring][width]
};
LayerCache make_layer_cache(const Dsv4Config& c, const LayerWeights& w, const CacheGeometry& g, int dev,
                            hipStream_t st);

struct ChunkDesc {
  int slot = 0, s0 = 0, n = 0, t0 = 0;
};
struct BatchDesc {
  std::vector<ChunkDesc> chunks;
  int T = 0;
  const int32_t* ids = nullptr;           // device [T]
  const int32_t* pos = nullptr;           // device [T]
  const int32_t* block_tables = nullptr;  // device [max_slots][max_pages_per_seq]
  int max_pages_per_seq = 0;
};

// Scratch for batches of up to max_tokens tokens whose sequences are at most max_seq long.
struct Scratch {
  int max_tokens = 0, max_keys = 0;
  std::vector<rt::DeviceBuffer> bufs;
  void* get(size_t i) const { return bufs.at(i).data(); }
};
Scratch make_scratch(const Dsv4Config& c, int dev, int max_tokens, int max_seq);

// One layer for the whole batch; x: residual [T, hc, dim] bf16, updated in place.
void layer_forward(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                   const RopeTables& rope, uint16_t* x, const BatchDesc& b, hipStream_t st);

}  // namespace unholy::model
