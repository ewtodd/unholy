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
#include "model/dsv4.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

namespace unholy::model {

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
  bool is_kda(int layer) const { return kda.at(size_t(layer)) != 0; }
  bool is_dense(int layer) const { return dense.at(size_t(layer)) != 0; }
  int lin_width() const { return lin_heads * lin_dim; }  // 8192
};

// One linear layer from either checkpoint: bf16 / fp16 / fp8 with block scales, or EXL3.
struct GLinear {
  DevTensor w, s;
  Exl3Weights x;
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
  DevTensor idx_knorm_w, idx_knorm_b, idx_ape;
  bool idx_f16 = false;  // k_norm / ape stored as fp16
  // MLP: dense, or router + shared expert + routed experts.
  GLinear gate, up, down;
  GLinear router;
  DevTensor router_bias;
  GLinear sh_gate, sh_up, sh_down;
  std::vector<GLinear> e_gate, e_up, e_down;
};

struct Glm5LoadOptions {
  const Checkpoint* experts = nullptr;  // routed experts from here (null: the dense checkpoint)
  const Checkpoint* aux = nullptr;      // tensors the dense checkpoint lacks (exllamav3 leaves out the
                                        // indexer's pooling weights): the FP8 release
  bool load_experts = true;
};
// Loads one layer to `dev` (synchronously).
Glm5LayerWeights load_glm5_layer(const Checkpoint& ck, const Glm5Config& c, int layer, int dev, hipStream_t st,
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
};
struct Glm5Batch {
  std::vector<Glm5Chunk> chunks;
  int T = 0;
  const int32_t* pos = nullptr;  // device [T]
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
  kGlm5ScratchBufs
};
constexpr int kIdxBlock = 64;  // queries per index-score pass
struct Glm5Scratch {
  int max_tokens = 0, max_seq = 0, head_rows = 0;
  std::vector<rt::DeviceBuffer> bufs;
  void* get(size_t i) const { return bufs.at(i).data(); }
  template <typename T>
  T* at(Glm5Buf b) const {
    return static_cast<T*>(get(size_t(b)));
  }
};
Glm5Scratch make_glm5_scratch(const Glm5Config& c, int dev, int max_tokens, int max_seq);

// Optional per-layer routing histogram ([layer][expert] counts), filled when set.
void set_glm5_expert_histogram(std::vector<std::vector<uint64_t>>* hist);

// One layer for the whole batch; x: residual [T, hc_mult, dim] bf16, updated in place.
void glm5_layer_forward(const Glm5Config& c, const Glm5LayerWeights& w, Glm5Cache& cache, const Glm5Scratch& scr,
                        uint16_t* x, const Glm5Batch& b, hipStream_t st);

// x [T, hc_mult, dim] -> logits [T, vocab] f32 (the bf16 values transformers' head returns). Rows are processed
// in groups of the scratch's max_tokens.
void glm5_head_forward(const Glm5Config& c, const Glm5HeadWeights& h, const Glm5Scratch& scr, const uint16_t* x,
                       int T, float* logits, hipStream_t st);

// out [rows, l.out] bf16 = x [rows, l.in] (row stride ldx) through `l`; EXL3 uses scratch for its fp32 sides.
void glm5_linear(const GLinear& l, const uint16_t* x, int ldx, uint16_t* out, int ldo, int rows,
                 const Glm5Scratch& scr, hipStream_t st);

}  // namespace unholy::model
