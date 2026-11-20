// Hy4 flash (hy_v4_flash, tencent/Hy4-flash-FP8): configuration, device weights, per-sequence state, the Engram
// store and the layer forward (docs/PLAN-hy4-flash.md).
//
// The architecture, as transformers' modeling_hy_v4_flash.py has it (PR #50903, the oracle in tools/ref/hy4_oracle):
// 42 layers of hc_mult = 4 identity hyper-connection streams (DeepSeek-V4's pre-mix without the Sinkhorn comb; post
// = 2 sigmoid; plain residual adds); gated MLA everywhere (q_lora 1024, kv_lora 512, nope 192 + rope 64, value 256,
// learned sinks, a sigmoid output gate, a norm after o_proj): layers 0, 1, 5, 9, .., 41 are "indexed" (32 heads,
// RoPE theta 1e7, a compressed indexer recalling 511 of the 4-token groups before the query's plus its own once the
// context passes 2048 tokens), the others slide over 512 keys (64 heads, theta 1e4). Layer 0 has a dense SwiGLU
// MLP; the rest a latent MoE: router (sigmoid + bias, top 16 of 512, weights normalized x 3.998) on the full width,
// the experts on a 1536-wide projection of it (e4m3 weights with 128x128 bf16 scales, gate / up clamped to 10), one
// shared expert on the full width, a norm over the sum. Layers 1, 11, 21, 31 first add an Engram memory: hashed
// bi- and trigram rows of 96 values (8 tables per order) gated into each stream.
//
// This implementation: dense weights dequantized to bf16(w * s) at load (the oracle's arithmetic), the routed
// experts kept fp8 (kernels::hy4_fp8_grouped_gemm), the attention rows in bf16 (sparse layers paged, window
// layers in a 512-row ring per slot), the Engram tables read from the SSD (tools/ref/hy4_engram_pack.py's pack).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "gufo/json.hpp"
#include "kernels/exl3_moe.hpp"
#include "kernels/hy4.hpp"
#include "kernels/moe_dispatch.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"
#include "model/safetensors.hpp"
#include "rt/arena.hpp"
#include "rt/hip.hpp"

namespace unholy::model {

class Uploader;

struct Hy4Config {
  int vocab = 0, dim = 0, n_layers = 0;
  int q_lora = 0, kv_lora = 0, nope = 0, rope = 0, v_dim = 0;
  int heads_sparse = 0, heads_window = 0, window = 0;
  float theta_sparse = 0, theta_window = 0;
  int idx_heads = 0, idx_dim = 0, idx_topk = 0, idx_pool = 0;
  int inter = 0, moe_inter = 0, latent = 0, n_experts = 0, n_active = 0;
  float route_scale = 0, swiglu_limit = 0;
  int hc = 0;
  float hc_eps = 0, hc_mag = 0, eps = 0;
  int bos = 0, eos = 0, pad = 0, image_token = -1, video_token = -1;
  int engram_pad = 0;
  std::vector<uint8_t> sparse;  // [n_layers] 1 = indexed attention, 0 = sliding window
  std::vector<uint8_t> dense;   // [n_layers] 1 = dense MLP
  std::vector<int> engram;      // [n_layers] index of the layer's Engram memory (0..3), -1 = none

  // From config.json (the text_config of an HYV4FlashForConditionalGeneration checkpoint, ids from the top level).
  static Hy4Config from_hf(const gufo::json::Value& cfg);
  bool is_sparse(int l) const { return sparse.at(size_t(l)) != 0; }
  bool is_dense(int l) const { return dense.at(size_t(l)) != 0; }
  int heads(int l) const { return is_sparse(l) ? heads_sparse : heads_window; }
  float theta(int l) const { return is_sparse(l) ? theta_sparse : theta_window; }
  int idx_groups() const { return idx_topk / idx_pool - 1; }       // complete groups a query recalls (511)
  int idx_width() const { return idx_groups() * idx_pool + idx_pool; }  // its selection list (2048)
  int n_engram() const;
};

// A linear [out, in]: bf16, or the release's e4m3 with bf16 128x128 scales kept as they are (w8 / s8; the kernels use
// bf16(w * s), the oracle's dequantization). tab: {w8, s8} as a one-matrix table for the grouped fp8 kernels.
struct Hy4Linear {
  DevTensor w, w8, s8;
  rt::DeviceBuffer tab;
  int out = 0, in = 0;
  bool fp8() const { return bool(w8); }
  explicit operator bool() const { return bool(w) || bool(w8); }
};

struct Hy4LayerWeights {
  int layer = 0;
  bool sparse = false, dense = false;
  int engram = -1;
  DevTensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;  // f32
  DevTensor input_norm, post_norm;
  // attention
  Hy4Linear q_a, q_b, kv_a, gate, o;
  DevTensor q_a_norm, kv_a_norm, post_proj_norm, sinks;
  DevTensor w_ukT, w_uv;  // kv_b per head: bf16 [H][512][192] (key rows, transposed), [H][256][512]
  // indexer (sparse layers)
  Hy4Linear idx_q, idx_k, idx_w;
  DevTensor idx_tok_norm, idx_wv, idx_wg, idx_ape, idx_knorm;  // bf16 [128]; f32 [128][128] x 2, [4][128], [128]
  // dense MLP (layer 0)
  Hy4Linear gate_p, up_p, down_p;
  DevTensor post_mlp_norm;
  // MoE
  Hy4Linear router, pre_expert, post_expert, sh_gate, sh_up, sh_down;
  DevTensor router_bias, combine_norm;
  rt::DeviceBuffer experts;   // this device's routed experts: fp8 [gate_up codes, scales, down codes, scales], or
                              // MXFP4 (mx) [w1 codes, scales, w3 codes, scales, w2 codes, scales], tile-major repacked
  rt::DeviceBuffer tab;       // fp8: [gu_w E][gu_s E][dn_w E][dn_s E]; mx: [w1 w, w1 s, w3 w, w3 s, w2 w, w2 s] x E
  bool mx = false;            // routed experts in MXFP4 (converted from the release's e4m3 at load)
  bool nv = false;            // routed experts in NVFP4 from a GGUF build (tables as mx: [w1, w3, w2] x [codes, scales])
  rt::DeviceBuffer resident;  // uint8 [E]
  int n_resident = 0;
  const uint8_t* const* gu_w() const { return tab.as<const uint8_t* const>(); }
  const uint16_t* const* gu_s(int E) const { return tab.as<const uint16_t* const>() + E; }
  const uint8_t* const* dn_w(int E) const { return tab.as<const uint8_t* const>() + 2 * E; }
  const uint16_t* const* dn_s(int E) const { return tab.as<const uint16_t* const>() + 3 * E; }
  const uint8_t* const* mx_tab(int m, int f, int E) const { return tab.as<const uint8_t* const>() + size_t(m * 2 + f) * E; }
  // Engram
  Hy4Linear eng_kv;  // [5 dim][1536]: key projections 0-3, value projection
  DevTensor eng_bias, eng_lookup_norm, eng_nk, eng_nq, eng_pre_gamma, eng_hcs;  // bf16
};

struct Hy4LoadOptions {
  bool load_dense = true;
  bool load_experts = true;
  const std::vector<uint8_t>* expert_mask = nullptr;  // [E] 1 = load (null: all)
  bool mxfp4 = false;  // routed experts converted to MXFP4 (kernels::hy4_fp8_to_mxfp4) for the mx_moe kernels
  bool dense_fp8 = true;  // keep the release's fp8 dense linears fp8 (false: dequantized to bf16 at load)
  const Checkpoint* experts_gguf = nullptr;  // NVFP4 routed experts from a GGUF build (AngelSlim's SPARK-4Bit)
  Uploader* up = nullptr;
  Uploader* experts_up = nullptr;  // for experts_gguf (null: one per call)
};
Hy4LayerWeights load_hy4_layer(const Checkpoint& ck, const Hy4Config& c, int layer, int dev, hipStream_t st,
                               const Hy4LoadOptions& opt = {});

struct Hy4HeadWeights {
  DevTensor embed;                        // bf16 [vocab][dim]
  DevTensor hc_fn, hc_base, hc_scale;     // f32
  DevTensor norm;
  Hy4Linear head;
};
Hy4HeadWeights load_hy4_head(const Checkpoint& ck, const Hy4Config& c, int dev, hipStream_t st, bool embed = true,
                             bool head = true);

// Device bytes of a layer's dense part as loaded (fp8 dequantized; kv_b also split per head), and of one expert.
size_t hy4_dense_layer_bytes(const Checkpoint& ck, const Hy4Config& c, int layer, bool dense_fp8 = true);
enum class Hy4ExpertFormat : uint8_t { Fp8, Mxfp4, Nvfp4 };
size_t hy4_expert_bytes(const Hy4Config& c, Hy4ExpertFormat f = Hy4ExpertFormat::Fp8);

// ---- Engram store -----------------------------------------------------------------------------------------------
// The pack (tools/ref/hy4_engram_pack.py: per memory, e4m3 rows [rows][96] and bf16 scales [rows]) read with pread;
// the hash as transformers' HYV4FlashEngramLayer.hash_ngrams.
class Hy4Engram {
 public:
  Hy4Engram(const std::string& dir, const Hy4Config& c);
  ~Hy4Engram();
  Hy4Engram(const Hy4Engram&) = delete;
  Hy4Engram& operator=(const Hy4Engram&) = delete;
  // Rows of memory m for T positions whose n-gram tokens are ngram [T][3] (the token, the one before, the one before
  // that; the pad id before a sequence's start): q [T][16][96], s [T][16]. Positions with mask 0 get zeros.
  void lookup(int m, const int32_t* ngram, const uint8_t* mask, int T, uint8_t* q, uint16_t* s) const;
  // Every memory's rows at once (q[m], s[m] as lookup's), the reads spread over threads.
  void lookup_all(const int32_t* ngram, const uint8_t* mask, int T, uint8_t* const* q, uint16_t* const* s) const;
  int memories() const { return int(mem_.size()); }

 private:
  void fill(int m, const int32_t* ngram, const uint8_t* mask, int t, uint8_t* q, uint16_t* s) const;
  struct Mem {
    int layer = 0, qfd = -1, sfd = -1;
    std::vector<int32_t> lookup;
    std::vector<int64_t> mult, moduli, offsets;
  };
  std::vector<Mem> mem_;
  int pad_ = 0;
};

// ---- state -------------------------------------------------------------------------------------------------------
// Sparse layers: rows bf16 [pages][256][576], pooled keys bf16 [pages][64][128], tails bf16 [slots][4][128].
// Window layers: rings bf16 [slots][512][576].
// Speculative decoding (kHy4MaxVerify rows per verify chunk): window layers back up the ring rows a verify chunk
// overwrites (ring_bak [slots][6][576]); sparse layers the indexer tail (tail_bak [slots][4][128]) and the chunk's token
// keys (vkeys [slots][6][128]); DFly2's index layers (dfly) keep, per slot, the recalled positions of the last 6
// positions (dsel [slots][6][2048], row pos % 6; dcount [slots][6]).
constexpr int kHy4MaxVerify = 6;
struct Hy4Cache {
  int layer = 0, slots = 0, num_pages = 0;
  bool sparse = false;
  rt::DeviceBuffer rows, pooled, tail, ring;
  rt::DeviceBuffer ring_bak, tail_bak, vkeys, dsel, dcount;
};
Hy4Cache make_hy4_cache(const Hy4Config& c, int layer, int slots, int num_pages, int dev, hipStream_t st,
                        bool dfly = false);
size_t hy4_cache_bytes(const Hy4Config& c, bool sparse, int slots, int num_pages, bool dfly = false);
// A slot's state back to how it was after position from - 1 of a verify chunk at s0 (positions [from, to) rejected).
void hy4_restore_slot(Hy4Cache& cache, int slot, int s0, int from, int to, hipStream_t st);

// Tokens [t0, t0 + n) of the batch are positions [s0, s0 + n) of the sequence in `slot` (one chunk per slot).
struct Hy4Chunk {
  int slot = 0, s0 = 0, n = 0, t0 = 0;
  bool verify = false;  // a verify chunk (its token and drafts): back up the state it overwrites
  int g0[3] = {0, 0, 0};  // with pos3: the M-RoPE planes of the first slot of the indexer group the chunk starts in
};
struct Hy4Batch {
  std::vector<Hy4Chunk> chunks;
  int T = 0;
  const int32_t* pos = nullptr;     // device [T]: the rows' slots (cache positions; RoPE too unless pos3)
  const int32_t* pos3 = nullptr;    // device [T][3] or null: M-RoPE's planes (t, h, w) for RoPE (images)
  const int32_t* tables = nullptr;  // device block tables [slots][max_pages]
  int max_pages = 0;
  const int32_t* pos_host = nullptr;  // host copy of pos (the indexer's launch sizes)
  // Engram rows of each memory for the batch's rows (device-readable, e.g. mapped pinned host memory).
  const uint8_t* eng_q[4] = {};
  const uint16_t* eng_s[4] = {};
  const uint8_t* eng_mask = nullptr;
};

enum Hy4Buf : int {
  kBufH, kBufHn, kBufPost, kBufOut,                                                          // layer
  kBufQr, kBufQ, kBufQn, kBufQa, kBufQp, kBufKv, kBufOl, kBufO, kBufG,                       // attention
  kBufTk, kBufIq, kBufIw, kBufIScores, kBufIGroups, kBufISel, kBufITopk,                     // indexer
  kBufLogits, kBufSel, kBufRw, kBufLat, kBufEIdx, kBufEWt, kBufRowExpert, kBufPidx, kBufTok2u,  // MoE
  kBufExpIn, kBufGu, kBufEh, kBufPout, kBufMsum, kBufRouted, kBufShared, kBufShG, kBufShU, kBufShH,
  kBufMoeCount, kBufMoeOff, kBufMoeFill, kBufMoePlan, kBufMoeLTiles, kBufMoeRTiles, kBufRouteHist,
  kBufMlpG, kBufMlpU, kBufMlpH,                                                              // dense MLP
  kBufEngRows, kBufEngKv,                                                                    // Engram
  kBufDenseTiles,                                                                            // fp8 linears
  kBufAttnSplit,                                                                             // few-query attention
  kHy4ScratchBufs
};
constexpr int kHy4IdxBlock = 64;  // queries per index-score pass
struct Hy4Scratch {
  int max_tokens = 0, max_seq = 0;
  std::vector<rt::DeviceBuffer> bufs;
  void* get(size_t i) const { return bufs.at(i).data(); }
  template <typename T>
  T* at(Hy4Buf b) const {
    return static_cast<T*>(get(size_t(b)));
  }
};
Hy4Scratch make_hy4_scratch(const Hy4Config& c, int dev, int max_tokens, int max_seq);
size_t hy4_scratch_bytes(const Hy4Config& c, int max_tokens, int max_seq);

// Embedding into the hc streams: x [T][hc][dim] bf16.
void hy4_embed(const Hy4Config& c, const Hy4HeadWeights& h, const int32_t* ids, uint16_t* x, int T, hipStream_t st);
// One layer for the whole batch; x [T][hc][dim] updated in place. Routed experts this device does not hold run
// through `remote` (required if any are missing).
void hy4_layer_forward(const Hy4Config& c, const Hy4LayerWeights& w, Hy4Cache& cache, const Hy4Scratch& s,
                       uint16_t* x, const Hy4Batch& b, hipStream_t st, RemoteExperts* remote = nullptr);
// x [T][hc][dim] -> logits f32 [T][vocab] (hyper-head, norm, the fp32 LM head).
// hn_out (optional): the normed rows the LM head reads [T][dim] (DFly2's target hidden states).
void hy4_head_forward(const Hy4Config& c, const Hy4HeadWeights& h, const Hy4Scratch& s, const uint16_t* x, int T,
                      float* logits, hipStream_t st, uint16_t* hn_out = nullptr);
// Routed experts of `w` on pair rows xa [P][latent] (tiles over rows this device holds): gate / up, clamped SwiGLU,
// down, times the routing weight rw [P] -> out [P][latent]. eh / gu: [P][moe_inter] / [P][2 moe_inter] scratch.
// Tiles of hy4_expert_tile_rows(w, tokens) rows.
void hy4_experts_dev(const Hy4Config& c, const Hy4LayerWeights& w, const uint16_t* xa, const float* rw,
                     const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev, int P, uint16_t* gu,
                     uint16_t* eh, uint16_t* out, hipStream_t st, const int* row0_dev = nullptr,
                     const int* row1_dev = nullptr);

// Rows per expert tile for a step of `tokens` tokens: fp8 kHy4GemvRows up to kHy4GemvTokens tokens, else kHy4TileRows; MXFP4 the mx GEMV's when mx_use_gemv(tokens),
// else kExl3TileRows. hy4_max_tiles: the tile list's room for a step of up to max_tokens tokens.
constexpr int kHy4GemvTokens = 32;  // steps of up to this many tokens take the fp8 GEMVs
int hy4_expert_tile_rows(bool mx, int tokens);
int hy4_max_tiles(const Hy4Config& c, int max_tokens);

// ---- DFly2 (the checkpoint's block drafter; vLLM's `dfly`) ---------------------------------------------------------
// Per draft step and request: a block of 6 rows at positions anchor .. anchor + 5 (the sampled "bonus" token fused with
// the target's final hidden state of the last accepted row, then 5 mask tokens) through 5 MLA + dense SwiGLU layers
// (one residual stream, 32 heads, theta 1e7), each attending to the positions target layer 25 + 4 i recalled for that
// row (its KV rows) and the block's own rows (causal); then the final norm and, row by row, a hidden-correction MLP
// that also sees the previous token's embedding, the target's LM head and argmax: 5 draft tokens.
constexpr int kHy4DraftTokens = 5, kHy4DraftRows = kHy4DraftTokens + 1;
struct Hy4DflyLayer {
  Hy4Linear q_a, q_b, kv_a, gate, o, mg, mu, md;
  DevTensor in_norm, post_norm, q_a_norm, kv_a_norm, post_proj_norm, sinks, w_ukT, w_uv;
};
struct Hy4DflyWeights {
  std::vector<Hy4DflyLayer> layers;
  std::vector<int> target;  // target layer of each draft layer (its KV rows and its indexer's picks)
  int mask_id = 0;
  DevTensor embed, enorm, hnorm, fgate, c_hnorm, c_enorm, final_norm;
  Hy4Linear eh, cg, cu, cd;
};
Hy4DflyWeights load_hy4_dfly(const Checkpoint& ck, const Hy4Config& c, int dev, hipStream_t st);
size_t hy4_dfly_bytes(const Checkpoint& ck);  // its weights and the embedding table it reads
bool hy4_has_dfly(const Checkpoint& ck);
struct Hy4DflyScratch {
  int max_reqs = 0;
  std::vector<rt::DeviceBuffer> bufs;
  int32_t* host = nullptr;  // pinned staging of the per-call inputs
  ~Hy4DflyScratch();
  Hy4DflyScratch() = default;
  Hy4DflyScratch(Hy4DflyScratch&&) noexcept;
  Hy4DflyScratch& operator=(Hy4DflyScratch&&) noexcept;
};
Hy4DflyScratch make_hy4_dfly_scratch(const Hy4Config& c, int dev, int max_reqs);
size_t hy4_dfly_scratch_bytes(const Hy4Config& c, int max_reqs);
struct Hy4DflyReq {
  int slot = 0, anchor = 0;  // anchor: the bonus token's slot (the last accepted row's + 1)
  int32_t bonus = 0;
  int rope = 0;              // the anchor's rotary position (its slot, or past images M-RoPE's shifted one)
};
// Draft tokens (device int32 [5][R]) on `st`, and (conf, optional) the drafter's probability of each (fp32 [5][R]). target: the 5 target layers' caches (their pages, the saved picks);
// tables: device block tables [slots][max_pages]; hidden: device [slots][6][dim], the final hidden state of each
// slot's last 6 positions (row pos % 6).
void hy4_dfly_draft(const Hy4Config& c, const Hy4DflyWeights& w, const Hy4HeadWeights& head,
                    const std::vector<Hy4Cache*>& target, const int32_t* tables, int max_pages, const uint16_t* hidden,
                    Hy4DflyScratch& s, const std::vector<Hy4DflyReq>& reqs, int32_t* tokens, hipStream_t st,
                    float* conf = nullptr);

// Optional per-layer routing histogram ([layer][expert] counts), filled when set.
void set_hy4_expert_histogram(std::vector<std::vector<uint64_t>>* hist);

}  // namespace unholy::model
