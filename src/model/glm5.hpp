// GLM-5.3-Flash (glm5_next): configuration, device weights from the FP8 release or an exllamav3 EXL3 quant,
// per-sequence state, and the reference layer forward (docs/PLAN-glm-dsv41.md, G4).
//
// The architecture, as transformers' modeling_glm5_next.py has it: 45 layers of hc_mult = 4 Hyper-Connection
// streams (mHC: the DeepSeek-V4 pre-mix, a bf16 post-mix, an unweighted mean at the head); every fourth layer
// (3, 7, ..., 43) is DeepSeek sparse attention (MLA without RoPE, kv_lora 512, 64 heads of 256, plus an
// indexer that scores 4-token pools of keys and keeps the top 512 pools and the query's incomplete pool), the
// others are KDA (Kimi delta attention: 64 heads of 128, a short conv, a per-channel forget gate bounded below
// by -5, a gated output norm); layers 0-2 have a dense SwiGLU MLP, the rest 288 routed experts (top 8, sigmoid
// router with a selection bias, weights x 2.5) plus one shared expert; SwiGLU inputs are clamped to 10.
//
// The forward here is the reference path: kernels/ref_glm.hpp in transformers' order and rounding points, one
// device, any batch of sequence chunks (prefill chunks or decode tokens of different slots). Its state per
// sequence: the KDA conv tail and recurrent state, the MLA latent rows, the indexer's key and gate rows and
// the pooled keys of complete pools, kept contiguous per slot (paging is the fast path's business).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gufo/json.hpp"
#include "kernels/ref_glm.hpp"
#include "kernels/exl3_moe.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"
#include "rt/arena.hpp"

namespace unholy::model {

class Uploader;

struct Glm5Config {
  int vocab = 0, dim = 0, n_layers = 0;
  int n_heads = 0, q_lora = 0, kv_lora = 0, nope_dim = 0, v_dim = 0;      // DSA attention
  int idx_heads = 0, idx_dim = 0, idx_topk = 0, idx_pool = 0;             // its indexer
  bool idx_tail = false;
  int lin_heads = 0, lin_dim = 0, conv_kernel = 0;                         // KDA
  float gate_lower_bound = 0;
  int inter = 0, moe_inter = 0, n_experts = 0, n_active = 0, n_shared = 0;  // MLPs
  float route_scale = 0, swiglu_limit = 0;
  int hc_mult = 0, sinkhorn_iters = 0;
  float norm_eps = 0, hc_eps = 0;
  int n_mtp = 0;
  std::vector<uint8_t> kda;    // [n_layers] 1 = linear attention (KDA), 0 = DSA
  std::vector<uint8_t> dense;  // [n_layers] 1 = dense MLP, 0 = MoE

  // From config.json (the text_config of a Glm5NextForConditionalGeneration checkpoint).
  static Glm5Config from_hf(const gufo::json::Value& cfg);
  // Layer n_layers (the MTP layer) is DSA + MoE.
  bool is_kda(int layer) const { return layer < n_layers && kda.at(size_t(layer)) != 0; }
  bool is_dense(int layer) const { return layer < n_layers && dense.at(size_t(layer)) != 0; }
  int lin_width() const { return lin_heads * lin_dim; }  // 8192
};

// One linear layer from either checkpoint: bf16 / fp16 / fp8 with block scales, or EXL3.
struct GLinear {
  DevTensor w, s;
  Exl3Weights x;
  rt::DeviceBuffer tab;  // fast path, EXL3: device [trellis, suh, svh] (one-expert tables for the grouped kernels)
  int out = 0, in = 0;
  bool exl3() const { return bool(x.trellis); }
  explicit operator bool() const { return bool(w) || exl3(); }
  kernels::WRef ref() const;
};

struct Glm5LayerWeights {
  int layer = 0;
  bool kda = false, dense = false;
  DevTensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;  // fn as f32
  DevTensor input_norm, post_norm;
  GLinear o;  // attention output projection (both kinds)
  // KDA: q / k / v (fused into qkv in EXL3 checkpoints), conv [3 * width, 4], gates, output norm.
  GLinear q, k, v, qkv, f_a, f_b, b, g_a, g_b;
  DevTensor conv, dt_bias, a_log, o_norm;
  // DSA
  GLinear q_a, q_b, kv_a, kv_b;
  DevTensor q_a_norm, kv_a_norm;
  GLinear idx_wq_b, idx_wk, idx_wproj, idx_gate;
  DevTensor idx_knorm_w, idx_knorm_b, idx_ape;  // ape as f32
  bool idx_f16 = false;  // k_norm stored as fp16
  // MLP: dense, or router + shared expert + routed experts.
  GLinear gate, up, down;
  GLinear router;
  DevTensor router_bias;
  GLinear sh_gate, sh_up, sh_down;
  std::vector<GLinear> e_gate, e_up, e_down;
  DevTensor w_ukT, w_uv;  // fast path: kv_b split per head, bf16 [H][kv_lora][nope] (transposed) and [H][v][kv_lora]
  // Fast path: the routed experts' tables [gate, up, down] x [trellis, suh, svh] x E (null where the expert is not
  // on this device), their bit widths, and which experts this device holds.
  rt::DeviceBuffer exl3_tab, resident;
  int exl3_bits[3] = {0, 0, 0};
  int n_resident = 0;
  const uint16_t* const* tab(int m, int f, int n_experts) const {
    return exl3_tab.as<const uint16_t* const>() + size_t(m * 3 + f) * n_experts;
  }
};

struct Glm5LoadOptions {
  const Checkpoint* experts = nullptr;  // routed experts from here (null: the dense checkpoint)
  const Checkpoint* aux = nullptr;      // tensors the dense checkpoint lacks (exllamav3 leaves out the
                                        // indexer's pooling weights): the FP8 release
  bool load_experts = true;
  bool kda_fp8 = false;  // re-quantize the KDA q / k / v / o projections (bf16 in the release) to 128x128 block fp8
  bool load_dense = true;                         // attention, norms, router, shared expert (false: experts only)
  const std::vector<uint8_t>* expert_mask = nullptr;  // [n_experts] 1 = load this routed expert (null: all)
  rt::Arena* arena = nullptr;                     // allocate the weights from here (null: one buffer each)
  Uploader* up = nullptr;                         // reuse pinned staging across layers (per checkpoint)
  Uploader* experts_up = nullptr;
};
// Loads one layer to `dev` (synchronously).
Glm5LayerWeights load_glm5_layer(const Checkpoint& ck, const Glm5Config& c, int layer, int dev, hipStream_t st,
                                 const Glm5LoadOptions& opt = {});

// The MTP (nextn) layer, index n_layers: eh_proj([enorm(emb(t + 1)), hnorm(h_t)]) -> a DSA + MoE layer with plain
// pre-norm residuals (no Hyper-Connections) -> the shared head's norm -> the trunk's LM head (DeepSeek-V3's scheme;
// h_t is the trunk's final-normed hidden state at t, the LM head's input). exllamav3 keeps it in mtp.safetensors.
struct Glm5MtpWeights {
  Glm5LayerWeights layer;
  GLinear eh;  // [dim, 2 dim]
  DevTensor enorm, hnorm, head_norm;
};
Glm5MtpWeights load_glm5_mtp(const Checkpoint& ck, const Glm5Config& c, int dev, hipStream_t st,
                             const Glm5LoadOptions& opt = {});

// Embedding (bf16 [vocab, dim]), final norm, LM head.
struct Glm5HeadWeights {
  DevTensor embed, norm;
  GLinear head;
};
Glm5HeadWeights load_glm5_head(const Checkpoint& ck, int dev, hipStream_t st, bool embed = true, bool head = true);

// Per-layer state for `slots` sequences of up to max_seq positions (zeroed).
struct Glm5Cache {
  int layer = 0, slots = 0, max_seq = 0;
  bool kda = false;
  rt::DeviceBuffer conv;   // bf16 [slots][conv_kernel - 1][3 * width]
  rt::DeviceBuffer state;  // f32 [slots][lin_heads][lin_dim][lin_dim]
  rt::DeviceBuffer kv;     // bf16 [slots][max_seq][kv_lora]
  rt::DeviceBuffer ik, ig;  // bf16 [slots][max_seq][idx_dim]: indexer keys, pool gate scores
  rt::DeviceBuffer pk;      // bf16 [slots][max_seq / idx_pool][idx_dim]: pooled keys
};
Glm5Cache make_glm5_cache(const Glm5Config& c, int layer, int slots, int max_seq, int dev, hipStream_t st);
// Resets one slot's state (a new sequence in it).
void clear_glm5_slot(const Glm5Config& c, Glm5Cache& cache, int slot, hipStream_t st);

// Tokens [t0, t0 + n) of the batch are positions [s0, s0 + n) of the sequence in `slot` (one chunk per slot).
struct Glm5Chunk {
  int slot = 0, s0 = 0, n = 0, t0 = 0;
  int save_after = -1;  // fast path, >= 0: back up the KDA state after this token of the chunk (a verify chunk)
};
struct Glm5Batch {
  std::vector<Glm5Chunk> chunks;
  int T = 0;
  const int32_t* pos = nullptr;  // device [T]
  const int32_t* tables = nullptr;  // fast path: device block tables [slots][max_pages] (kernels/glm_fast.hpp)
  int max_pages = 0;
};

// Working memory for batches of up to max_tokens tokens of sequences up to max_seq long.
enum Glm5Buf : int {
  kH, kHn, kPost, kComb, kOut,                                            // layer
  kQkv, kQkvConv, kLowRank, kGateRow, kForget, kBeta, kCore, kOutNorm,    // KDA
  kQResid, kQ, kLatent, kQAbs, kLatentOut, kAttnOut,                      // DSA
  kIdxQ, kIdxK, kIdxGate, kIdxW, kIdxScores, kIdxPools, kIdxSel,          // its indexer
  kF32Rows, kF32Hn, kSel, kRouteW, kPairTok, kPairW, kPairIdx,            // MoE
  kExpIn, kExpGate, kExpUp, kExpAct, kExpOut, kShared,
  kExl3X, kExl3Y, kHeadOut,                                               // EXL3 sides, head
  kDenseTiles, kZeroRows, kHcPart, kKdaQ, kKdaK, kKdaE,                   // fast path
  kMoeCount, kMoeOff, kMoeFill, kMoePlan, kMoeLTiles, kMoeRTiles, kEIdx, kEWt, kRowExpert, kXh1, kXh3, kXh2, kYp,
  kRouteHist, kTopkCand, kAttnPart,
  kGlm5ScratchBufs
};
constexpr int kIdxBlock = 64;  // queries per index-score pass
constexpr int kAttnSplitRows = 4, kAttnSplits = 32;  // chunks of <= 4 queries: attention over 32 key splits
struct Glm5Scratch {
  int max_tokens = 0, max_seq = 0, head_rows = 0;
  std::vector<rt::DeviceBuffer> bufs;
  void* get(size_t i) const { return bufs.at(i).data(); }
  template <typename T>
  T* at(Glm5Buf b) const {
    return static_cast<T*>(get(size_t(b)));
  }
};
// fast_only: for the fast path alone (smaller: its absorbed-attention buffers are bf16).
Glm5Scratch make_glm5_scratch(const Glm5Config& c, int dev, int max_tokens, int max_seq, bool fast_only = false);
size_t glm5_scratch_bytes(const Glm5Config& c, int max_tokens, int max_seq, bool fast_only = false);
// Device bytes of a layer's dense part (everything but routed experts) as the fast path holds it (fp16 tensors also
// converted to bf16, kv_b also split per head), and of one routed expert.
size_t glm5_dense_layer_bytes(const Checkpoint& ck, int layer);
size_t glm5_expert_bytes(const Checkpoint& ck, int layer, int expert);

// Optional per-layer routing histogram ([layer][expert] counts), filled when set.
void set_glm5_expert_histogram(std::vector<std::vector<uint64_t>>* hist);

// One layer for the whole batch; x: residual [T, hc_mult, dim] bf16, updated in place.
void glm5_layer_forward(const Glm5Config& c, const Glm5LayerWeights& w, Glm5Cache& cache, const Glm5Scratch& scr,
                        uint16_t* x, const Glm5Batch& b, hipStream_t st);

// x [T, hc_mult, dim] -> logits [T, vocab] f32 (the bf16 values transformers' head returns). Rows are processed
// in groups of the scratch's max_tokens.
void glm5_head_forward(const Glm5Config& c, const Glm5HeadWeights& h, const Glm5Scratch& scr, const uint16_t* x,
                       int T, float* logits, hipStream_t st);

// Dynamic fp8 activations (kernels::fake_quant_fp8_group) ahead of every block-fp8 linear, as fp8 serving stacks run
// the release; off by default (the oracle keeps activations in bf16).
void set_glm5_a8(bool on);
// fp8 storage of the cache (kernels::fake_quant_fp8, power-of-two scales as DSV4's KV): the MLA latent rows (per 64
// values) and / or the indexer's pooled keys (per row of 128), quantized as they are written. Off by default.
void set_glm5_cache_fp8(bool latent, bool pooled);
bool glm5_cache_fp8_latent();
bool glm5_cache_fp8_pooled();

// out [rows, l.out] bf16 = x [rows, l.in] (row stride ldx) through `l`; EXL3 uses scratch for its fp32 sides.
void glm5_linear(const GLinear& l, const uint16_t* x, int ldx, uint16_t* out, int ldo, int rows,
                 const Glm5Scratch& scr, hipStream_t st);

// ---- fast path (docs/PLAN-glm-dsv41.md, G5) --------------------------------------------------------------------
// The same layer on the fast kernels: EXL3 linears on the grouped EXL3 GEMM / GEMV kernels (no activation fake
// quantization), small projections on the bf16 WMMA GEMM / GEMV, the routed experts dispatched on the device
// (kernels/moe_dispatch), the Hyper-Connection and norm kernels of the DSV4 fast path. Arithmetic differs from the
// reference at the bf16-rounding level; the check is KL against the oracle (tools/glm_ref --fast).

// Fast path state of one layer: KDA's per-slot conv tail and recurrent state (as Glm5Cache), or the DSA layer's
// paged rows (kernels/glm_fast.hpp: fp8 latent, bf16 pooled keys) and per-slot pool tails.
struct Glm5PagedCache {
  int layer = 0, slots = 0, num_pages = 0;
  bool kda = false;
  rt::DeviceBuffer conv, state;
  rt::DeviceBuffer conv_bak, state_bak;  // KDA, with backups: the state after a verify chunk's first token
  rt::DeviceBuffer lat, lat_s, pk, tail_k, tail_g;
};
// backups: room for the KDA state backups of speculative decoding (Glm5Chunk::save_after, glm5_restore_kda).
Glm5PagedCache make_glm5_paged_cache(const Glm5Config& c, int layer, int slots, int num_pages, int dev,
                                     hipStream_t st, bool backups = false);
size_t glm5_paged_cache_bytes(const Glm5Config& c, bool kda, int slots, int num_pages, bool backups = false);
// A slot's KDA state back to its backup (a rejected draft), on st.
void glm5_restore_kda(Glm5PagedCache& cache, int slot, hipStream_t st);

// After load_glm5_layer: EXL3 pointer tables, fp16 small weights converted to bf16 (exllamav3 stored the release's
// bf16 values as fp16; converting back is exact for them), the routed experts' grouped tables.
void prepare_glm5_fast(const Glm5Config& c, Glm5LayerWeights& w, int dev, hipStream_t st);
void prepare_glm5_fast_head(Glm5HeadWeights& h, int dev, hipStream_t st);
void prepare_glm5_fast_mtp(const Glm5Config& c, Glm5MtpWeights& m, int dev, hipStream_t st);
// The layer on the fast kernels; routed experts this device does not hold run through `remote` (the APU).
void glm5_layer_forward_fast(const Glm5Config& c, const Glm5LayerWeights& w, Glm5PagedCache& cache,
                             const Glm5Scratch& scr, uint16_t* x, const Glm5Batch& b, hipStream_t st,
                             RemoteExperts* remote = nullptr);
// Routed experts of `w` that this device holds, over the pair rows of the tiles (*ntiles_dev of max_tiles run): in
// bf16 [P, dim] with routing weights rw [P] and row_expert [P] -> out bf16 [P, dim] for the tiles' rows.
struct Glm5ExpertBufs {
  uint16_t *xh1, *xh3, *xh2;  // fp16 [P, dim], [P, dim], [P, moe_inter]
  uint16_t *h1, *h3;          // bf16 [P, moe_inter]
  float* yp;                  // fp32 [min(P, 64), max(dim, moe_inter)] (GEMV steps)
};
void glm5_experts_dev(const Glm5Config& c, const Glm5LayerWeights& w, const uint16_t* in, const float* rw,
                      const int32_t* row_expert, const kernels::Exl3Tile* tiles, int max_tiles, const int* ntiles_dev,
                      int P, const Glm5ExpertBufs& b, uint16_t* out, hipStream_t st);
void glm5_head_forward_fast(const Glm5Config& c, const Glm5HeadWeights& h, const Glm5Scratch& scr, const uint16_t* x,
                            int T, float* logits, hipStream_t st);
// The LM head on rows that are already normed (hn [T, dim]) -> logits [T, vocab] f32.
void glm5_head_logits(const Glm5Config& c, const Glm5HeadWeights& h, const Glm5Scratch& scr, const uint16_t* hn, int T,
                      float* logits, hipStream_t st);

// MTP on the fast path. x0 [rows, dim] = eh_proj([rms(emb) * enorm, rms(h) * hnorm]) (emb, h bf16 [rows, dim]).
void glm5_mtp_in(const Glm5Config& c, const Glm5MtpWeights& m, const uint16_t* emb, const uint16_t* h, uint16_t* x0,
                 int rows, const Glm5Scratch& scr, hipStream_t st);
// The MTP layer's cache rows for the batch's positions, from x0 (its attention input norm, then the DSA rows only).
void glm5_mtp_cache(const Glm5Config& c, const Glm5MtpWeights& m, Glm5PagedCache& cache, const Glm5Scratch& scr,
                    const uint16_t* x0, const Glm5Batch& b, hipStream_t st);
// The whole MTP layer on x [rows, dim] in place (plain residuals; it writes the batch's cache rows), then the shared
// head's norm into hn [rows, dim].
void glm5_mtp_forward(const Glm5Config& c, const Glm5MtpWeights& m, Glm5PagedCache& cache, const Glm5Scratch& scr,
                      uint16_t* x, uint16_t* hn, const Glm5Batch& b, hipStream_t st, RemoteExperts* remote = nullptr);
}  // namespace unholy::model
