// DeepSeek-V4-Flash: configuration, device weights, per-sequence state, and the reference layer forward.
//
// The forward here is the *reference path*: it calls the kernels in kernels/ref.hpp in the order and
// with the rounding points of DeepSeek's inference/model.py, on one device, for one sequence chunk at
// a time. It is what the fast path (batched, multi-device, fused) is validated against, and what the
// oracle comparison (tools/layer_test) checks.
//
// Generalisation over the reference: chunked prefill. The reference handles a prefill at position 0
// or one decode token; here any chunk [s0, s0 + n) works, with the window ring, the compressor
// projection rings and the paged compressed rows carrying state between chunks. The math is the
// reference's token-sequential definition (its decode path), which its prefill path also computes.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gufo/json.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

namespace unholy::model {

struct Dsv4Config {
  int vocab = 0, dim = 0, n_layers = 0, n_heads = 0, head_dim = 0, rope_dim = 0;
  int q_lora = 0, o_lora = 0, o_groups = 0, window = 0;
  int idx_heads = 0, idx_dim = 0, idx_topk = 0;
  int n_experts = 0, n_active = 0, moe_inter = 0, n_hash_layers = 0;
  int hc_mult = 0, sinkhorn_iters = 0;
  float norm_eps = 0, hc_eps = 0, route_scale = 0, swiglu_limit = 0;
  double rope_theta = 0, compress_rope_theta = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0;
  int original_seq_len = 0;
  std::vector<int> compress_ratios;  // n_layers + n_mtp entries
  int n_mtp = 0, dspark_block = 0, dspark_noise_token = 0, dspark_markov_rank = 0;
  std::vector<int> dspark_target_layers;

  static Dsv4Config from_hf(const gufo::json::Value& cfg);
  int ratio(int layer) const { return compress_ratios.at(size_t(layer)); }
};

// Rotary (cos, sin) tables [max_pos][rope_dim / 2], computed like precompute_freqs_cis in fp32.
// `yarn` = the compressed-layer family (compress_rope_theta + YaRN); otherwise rope_theta, no YaRN.
std::vector<float> rope_table(const Dsv4Config& c, int max_pos, bool yarn);

// A device copy of one checkpoint tensor (raw bytes, optional dtype conversion).
struct DevTensor {
  rt::DeviceBuffer buf;
  DType dtype = DType::Unknown;
  std::vector<int64_t> shape;
  template <typename T>
  T* p() const {
    return buf.as<T>();
  }
  explicit operator bool() const { return buf.data() != nullptr; }
};

struct Fp8Linear {  // e4m3 [out, in] + e8m0 [ceil(out/128), in/128]
  DevTensor w, s;
  int out = 0, in = 0;
};
struct Fp4Linear {  // packed e2m1 [out, in/2] + e8m0 [out, in/32]
  DevTensor w, s;
  int out = 0, in = 0;
};
struct Exl3Weights {  // EXL3 linear (trellis I16, suh/svh F16), mul1 codebook
  DevTensor trellis, suh, svh;
  int in = 0, out = 0, bits = 0;
};
struct CompressorWeights {
  DevTensor wkv, wgate, ape, norm;  // bf16 [coff*d, dim], bf16, f32 [ratio, coff*d], bf16 [d]
  int ratio = 0, d = 0;
  bool overlap = false, rotate = false;
  int width() const { return overlap ? 2 * d : d; }
  int ring() const { return overlap ? 2 * ratio : ratio; }
};

struct LayerWeights {
  int layer = 0, ratio = 0;
  bool hash = false;
  DevTensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  DevTensor attn_norm, ffn_norm;
  DevTensor sink, q_norm, kv_norm;
  Fp8Linear wq_a, wq_b, wkv, wo_a, wo_b;
  std::optional<CompressorWeights> comp;
  bool has_indexer = false;
  Fp8Linear idx_wq_b;
  DevTensor idx_wproj;  // bf16 [idx_heads, dim]
  std::optional<CompressorWeights> idx_comp;
  DevTensor gate_w, gate_bias, tid2eid;  // bf16 [E, dim], f32 [E], int32 [vocab, K]
  std::vector<Fp4Linear> w1, w2, w3;     // routed experts
  bool exl3_experts = false;              // routed experts in EXL3 instead (x1, x2, x3)
  std::vector<Exl3Weights> x1, x2, x3;
  Fp8Linear sw1, sw2, sw3;               // shared expert
};

class Uploader;
// Loads one layer's weights to `dev` (coalesced O_DIRECT reads; pass an Uploader to reuse its
// pinned staging across layers).
// `exl3`: take the routed experts from this EXL3 checkpoint instead of the native MXFP4 ones.
LayerWeights load_layer(const Checkpoint& ck, const Dsv4Config& c, int layer, int dev, hipStream_t st,
                        Uploader* up = nullptr, const Checkpoint* exl3 = nullptr, Uploader* exl3_up = nullptr);

// Per-sequence, per-layer state. Compressed rows are addressed through page tables (identity for
// now) so the kernels already work on paged pools.
struct LayerState {
  int layer = 0;
  rt::DeviceBuffer window;                    // bf16 [128][head_dim]
  rt::DeviceBuffer comp_ring_kv, comp_ring_sc;  // f32 [ring][width]
  rt::DeviceBuffer idx_ring_kv, idx_ring_sc;
  rt::DeviceBuffer comp_pool, idx_pool;  // bf16 rows
  rt::DeviceBuffer page_table;           // int32, identity
  int rows_per_page = 64;
  int max_rows = 0;
};
LayerState make_layer_state(const Dsv4Config& c, const LayerWeights& w, int dev, int max_seq, hipStream_t st);

// Reusable scratch for chunks of up to max_tokens tokens.
struct Scratch {
  int max_tokens = 0, max_keys = 0;
  std::vector<rt::DeviceBuffer> bufs;
  void* get(size_t i) const { return bufs.at(i).data(); }
};
Scratch make_scratch(const Dsv4Config& c, int dev, int max_tokens, int max_seq);

struct RopeTables {
  rt::DeviceBuffer plain, yarn;  // float2 [max_pos][rope_dim/2]
  int max_pos = 0;
};
RopeTables make_rope_tables(const Dsv4Config& c, int dev, int max_pos);

// One layer over chunk positions [s0, s0 + n) of a sequence. x: residual [n, hc, dim] bf16, updated
// in place. ids/pos: device int32 [n].
void layer_forward(const Dsv4Config& c, const LayerWeights& w, LayerState& s, const Scratch& scr,
                   const RopeTables& rope, uint16_t* x, const int32_t* ids, const int32_t* pos, int s0, int n,
                   hipStream_t st);

}  // namespace unholy::model
