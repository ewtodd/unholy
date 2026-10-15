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

#include <array>
#include <cstdint>
#include <vector>

#include <stdexcept>

#include "kernels/exl3_moe.hpp"
#include "kernels/moe_dispatch.hpp"
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
  // Image tokens: device [T][2] window position range (lo, hi) per row, or null (every row's window is
  // its last 128 positions). See k::AttnKv::vis.
  const int32_t* vis = nullptr;
};

// Scratch for computing one routed expert over up to `rows` tokens (any device).
struct ExpertScratch {
  rt::DeviceBuffer qb, qs, h1, h3, hh, xf, y2;
  int rows = 0;
};
ExpertScratch make_expert_scratch(const Dsv4Config& c, int dev, int rows);

// Scratch for the grouped EXL3 path over up to `rows` pair rows.
struct Exl3MoeScratch {
  rt::DeviceBuffer xh1, xh3, h1, h3, xh2, tiles, row_expert, yp;
  int rows = 0, max_tiles = 0;
};
// exl3: the buffers only the EXL3 experts use (xh3, yp).
Exl3MoeScratch make_exl3_moe_scratch(const Dsv4Config& c, int dev, int rows, bool exl3 = true);

// The routed experts of `w` that are resident here, for the given tiles (runs of <= 64 pair rows of one
// expert), over pair rows in[P, dim] with routing weights rw[P] and row_expert[P] (device-visible).
// out rows of the tiles are written; other rows are untouched. Disabled by UNHOLY_REF_EXL3=1.
bool exl3_fast_enabled();
void exl3_moe_forward(const Dsv4Config& c, const LayerWeights& w, const uint16_t* in, const float* rw,
                      const int32_t* row_expert, const std::vector<kernels::Exl3Tile>& tiles, int P,
                      const Exl3MoeScratch& s, uint16_t* out, hipStream_t st);
// Same with the tile list on the device: the grid is sized for max_tiles, *ntiles_dev tiles run.
// in_rows (native experts only): `in` is the token rows, pair row r takes row in_rows[r].
void exl3_moe_forward_dev(const Dsv4Config& c, const LayerWeights& w, const uint16_t* in, const float* rw,
                          const int32_t* row_expert, const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev,
                          int P, const Exl3MoeScratch& s, uint16_t* out, hipStream_t st,
                          const int32_t* in_rows = nullptr);
// Native MXFP4 experts on compact fp8 inputs (kernels::quant_remote_tokens): rows [plan->remote0,
// plan->P) rebuilt from their tokens' codes / scale exponents, then the same GEMMs as the pair-row path.
// With u2p: the pair outputs stay in scratch and `out` gets one combined row per token
// (kernels::remote_combine).
void mx_moe_forward_dev_q(const Dsv4Config& c, const LayerWeights& w, const uint8_t* q, const uint8_t* s,
                          const int32_t* p2u, const kernels::MoePlan* plan, const float* rw, const int32_t* row_expert,
                          const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev, int P,
                          const Exl3MoeScratch& sc, uint16_t* out, hipStream_t st, const int32_t* u2p = nullptr);
// Tiles of <= 64 rows for each (expert, row0, cnt) run.
std::vector<kernels::Exl3Tile> exl3_tiles(const std::vector<std::array<int, 3>>& runs);

// Scratch for batches of up to max_tokens tokens whose sequences are at most max_seq long.
// Device-side MoE dispatch state (kernels/moe_dispatch.hpp).
struct MoeDispatchScratch {
  rt::DeviceBuffer count, off, fill, plan, ltiles, rtiles;
  int max_tiles = 0;
};

struct Scratch {
  int max_tokens = 0, max_keys = 0;
  rt::DeviceBuffer route_hist;  // uint32 [n_layers][n_experts]: pairs routed per expert, running total
  std::vector<rt::DeviceBuffer> bufs;
  ExpertScratch expert;
  Exl3MoeScratch exl3;
  MoeDispatchScratch dispatch;
  void* get(size_t i) const { return bufs.at(i).data(); }
};
// exl3: room for EXL3 routed experts (native MXFP4 needs less); the reference expert path's scratch only
// when it can run (ref_experts_needed()).
Scratch make_scratch(const Dsv4Config& c, int dev, int max_tokens, int max_seq, bool exl3 = true);
// Device bytes those objects take (for the planner).
size_t scratch_bytes(const Dsv4Config& c, int max_tokens, int max_seq, bool exl3 = true);
size_t exl3_moe_scratch_bytes(const Dsv4Config& c, int rows, bool exl3 = true);
bool ref_experts_needed();
size_t expert_scratch_bytes(const Dsv4Config& c, int rows);
size_t layer_cache_bytes(const Dsv4Config& c, int ratio, bool indexer, const CacheGeometry& g);

// Routed expert `e` of `w` on `cnt` rows: out = w2(swiglu(w1(in), w3(in)) * rw). `in` may be modified
// (the EXL3 path fake-quantizes it in place). Pointers may be pinned host memory mapped into `dev`.
void expert_forward(const Dsv4Config& c, const LayerWeights& w, int e, uint16_t* in, const float* rw, int cnt,
                    uint16_t* out, const ExpertScratch& s, hipStream_t st);

// Routed experts of a layer that live on another device. The MoE half writes the inputs of all
// (token, expert) pairs, expert-major in ascending expert order, to `in_host` (row = pair index) and
// their routing weights to `w_host`; run() computes the listed jobs into `out_host` at the same rows.
// The buffers are pinned host memory owned by the implementation (src/engine).
class RemoteExperts {
 public:
  struct Job {
    int expert, row0, cnt;
  };
  virtual ~RemoteExperts() = default;
  virtual void run(int layer, const std::vector<Job>& jobs) = 0;
  // Device-dispatch mode. Before enqueue() the card's stream has written plan_host, tiles_host (remote
  // tiles), in_host / w_host / e_host. enqueue() orders the remote computation after the card stream's
  // preceding work; join() orders the card stream's following work after it. No host waits.
  virtual bool can_enqueue() const { return false; }
  virtual void enqueue(int layer, hipStream_t card_st, int P) { throw std::runtime_error("enqueue unsupported"); }
  virtual void join(hipStream_t card_st) { throw std::runtime_error("join unsupported"); }
  kernels::MoePlan* plan_host = nullptr;
  kernels::Exl3Tile* tiles_host = nullptr;
  int max_tiles = 0;
  uint16_t* in_host = nullptr;
  float* w_host = nullptr;
  uint16_t* out_host = nullptr;
  int32_t* e_host = nullptr;  // expert of each pair row
  size_t capacity_rows = 0;
  // Compact fp8 inputs instead of in_host (native MXFP4 experts, when quant_in()): codes [tokens, dim]
  // and e8m0 scale exponents [tokens, dim / 128] of the tokens with remote pairs, and each remote pair
  // row's index into them (p2u_host [capacity_rows]).
  virtual bool quant_in() const { return false; }
  // Combined outputs (with quant_in()): one bf16 row per token with remote pairs in out_host (the
  // remote pairs' sum, kernels::remote_combine), u2p_host [tokens, K] says which pair rows to sum.
  virtual bool combine_out() const { return false; }
  int32_t* u2p_host = nullptr;
  uint8_t* inq_host = nullptr;
  uint8_t* ins_host = nullptr;
  int32_t* p2u_host = nullptr;
};

// Optional per-layer routing histogram ([layer][expert] counts), filled when set.
void set_expert_histogram(std::vector<std::vector<uint64_t>>* hist);

// One layer for the whole batch; x: residual [T, hc, dim] bf16, updated in place. Experts that `w`
// does not hold are computed by `remote` (required if any are missing).
void layer_forward(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                   const RopeTables& rope, uint16_t* x, const BatchDesc& b, hipStream_t st,
                   RemoteExperts* remote = nullptr);

// ---- DSpark (model.py DSparkBlock / Transformer.forward_spec) ----------------------------------------
// A draft block per request: rows [B r, B r + B) of the batch (B = dspark_block: the request's next
// token, then noise tokens) at positions start + 1 .. start + B in `slot`. Each stage's window ring holds
// the main-KV of the slot's positions; a request sees its last min(128, start + 1) of them in slot order.
struct DraftDesc {
  std::vector<int> slots, start;  // host, per request
  int T = 0;                      // B * requests
  const int32_t* ids = nullptr;   // device [T]
  const int32_t* pos = nullptr;   // device [T]
};
// main_x = main_norm(main_proj(main_hidden)) for R rows (main_hidden [R, n_target_layers * dim], the
// target's HC means) and each stage's main-KV of it, stored at pos[r] % 128 of slot slots[r] (device).
void dspark_main_kv(const Dsv4Config& c, const DSparkWeights& d, std::vector<LayerCache>& stages,
                    const Scratch& scr, const RopeTables& rope, const uint16_t* main_hidden, const int32_t* slots,
                    const int32_t* pos, int R, hipStream_t st);
// One stage over the draft blocks; x [T, hc, dim] in place.
void dspark_stage(const Dsv4Config& c, const LayerWeights& w, LayerCache& cache, const Scratch& scr,
                  const RopeTables& rope, uint16_t* x, const DraftDesc& d, hipStream_t st,
                  RemoteExperts* remote = nullptr);
// The head path after the last stage: hc_head, norm, the LM head (head_w bf16 [vocab, dim]) into logits
// [T, vocab] f32, then per position i the Markov bias of token i and its argmax as token i + 1 (tok
// [R, B + 1], column 0 set by the caller), and the confidences conf [T].
void dspark_head(const Dsv4Config& c, const DSparkWeights& d, const Scratch& scr, const uint16_t* x,
                 const uint16_t* head_w, int R, float* logits, int32_t* tok, float* conf, hipStream_t st);

}  // namespace unholy::model
