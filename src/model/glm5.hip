// GLM-5.3-Flash: configuration, weight loading, per-sequence state, scratch (model/glm5.hpp).
#include "model/glm5.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "core/half.hpp"
#include "kernels/exl3_moe.hpp"
#include "kernels/hc_fast.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/ref.hpp"
#include "model/exl3.hpp"
#include "model/loader.hpp"

namespace unholy::model {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// Config

Glm5Config Glm5Config::from_hf(const gufo::json::Value& top) {
  const gufo::json::Value& j = top.find("text_config") ? *top.find("text_config") : top;
  auto num = [](const gufo::json::Value& o, const char* key) {
    const auto* v = o.find(key);
    if (!v || !v->is_number()) throw std::runtime_error(std::string("config: missing ") + key);
    return v->as_double();
  };
  auto strs = [&](const char* key) {
    std::vector<std::string> out;
    const auto* v = j.find(key);
    if (!v) throw std::runtime_error(std::string("config: missing ") + key);
    for (const auto& e : v->items()) out.push_back(e.get_str());
    return out;
  };
  Glm5Config c;
  c.vocab = int(num(j, "vocab_size"));
  c.dim = int(num(j, "hidden_size"));
  c.n_layers = int(num(j, "num_hidden_layers"));
  c.n_heads = int(num(j, "num_attention_heads"));
  c.q_lora = int(num(j, "q_lora_rank"));
  c.kv_lora = int(num(j, "kv_lora_rank"));
  c.nope_dim = int(num(j, "qk_nope_head_dim"));
  c.v_dim = int(num(j, "v_head_dim"));
  c.idx_heads = int(num(j, "index_n_heads"));
  c.idx_dim = int(num(j, "index_head_dim"));
  c.idx_topk = int(num(j, "index_topk"));
  c.idx_pool = int(num(j, "index_kpool"));
  c.idx_tail = j.find("index_kpool_always_select_tail") && j.find("index_kpool_always_select_tail")->as_bool();
  const auto* lin = j.find("linear_attn_config");
  if (!lin) throw std::runtime_error("config: missing linear_attn_config");
  c.lin_heads = int(num(*lin, "num_heads"));
  c.lin_dim = int(num(*lin, "head_dim"));
  c.conv_kernel = int(num(*lin, "short_conv_kernel_size"));
  c.gate_lower_bound = float(num(*lin, "gate_lower_bound"));
  c.inter = int(num(j, "intermediate_size"));
  c.moe_inter = int(num(j, "moe_intermediate_size"));
  c.n_experts = int(num(j, "n_routed_experts"));
  c.n_active = int(num(j, "num_experts_per_tok"));
  c.n_shared = int(num(j, "n_shared_experts"));
  c.route_scale = float(num(j, "routed_scaling_factor"));
  c.swiglu_limit = float(num(j, "swiglu_limit"));
  c.hc_mult = int(num(j, "hc_mult"));
  c.sinkhorn_iters = int(num(j, "hc_sinkhorn_iters"));
  c.norm_eps = float(num(j, "rms_norm_eps"));
  c.hc_eps = float(num(j, "hc_eps"));
  c.n_mtp = int(j.member_double("num_nextn_predict_layers"));
  for (const auto& t : strs("layer_types")) {
    if (t != "linear_attention" && t != "deepseek_sparse_attention")
      throw std::runtime_error("config: layer type " + t);
    c.kda.push_back(t == "linear_attention");
  }
  for (const auto& t : strs("mlp_layer_types")) {
    if (t != "dense" && t != "sparse") throw std::runtime_error("config: mlp layer type " + t);
    c.dense.push_back(t == "dense");
  }
  for (const auto& t : strs("indexer_types"))
    if (t != "full") throw std::runtime_error("config: shared indexer layers are not supported yet");
  // What this implementation assumes about the release.
  if (int(num(j, "qk_rope_head_dim")) != 0) throw std::runtime_error("config: GLM attention with RoPE");
  if (j.member_str("scoring_func") != "sigmoid" || int(num(j, "n_group")) != 1 || int(num(j, "topk_group")) != 1)
    throw std::runtime_error("config: router other than sigmoid / one group");
  if (!j.find("norm_topk_prob") || !j.find("norm_topk_prob")->as_bool())
    throw std::runtime_error("config: unnormalised top-k weights");
  if (!c.idx_tail || c.idx_topk % c.idx_pool != 0) throw std::runtime_error("config: indexer without the tail pool");
  if (c.n_shared != 1 || c.conv_kernel != 4 || c.hc_mult != 4) throw std::runtime_error("config: unexpected shape");
  if (int(c.kda.size()) != c.n_layers || int(c.dense.size()) != c.n_layers)
    throw std::runtime_error("config: layer type lists do not match num_hidden_layers");
  return c;
}

// ---------------------------------------------------------------------------------------------------
// Weights

kernels::WRef GLinear::ref() const {
  if (exl3()) throw std::runtime_error("GLinear::ref on an EXL3 linear");
  kernels::WRef r;
  r.p = w.ptr;
  switch (w.dtype) {
    case DType::BF16: r.fmt = kernels::WFmt::Bf16; break;
    case DType::F16: r.fmt = kernels::WFmt::F16; break;
    case DType::F8_E4M3:
      r.fmt = kernels::WFmt::Fp8Block;
      r.scale = s.p<float>();
      break;
    default: throw std::runtime_error("GLinear: unsupported weight dtype");
  }
  return r;
}

namespace {

const std::string kPrefix = "model.language_model.";

// Collects (tensor, device destination) pairs of one checkpoint; the bytes move in one coalesced pass.
struct LoadCtx {
  const Checkpoint& ck;
  int dev;
  std::vector<Uploader::Item> items;
  std::vector<std::pair<DevTensor, DevTensor*>> to_f32;  // bf16 staging -> f32 destination, after the pass
  std::vector<rt::DeviceBuffer> keep;

  void* alloc(DevTensor& d, size_t bytes) {
    d.buf = rt::DeviceBuffer(dev, bytes);
    d.ptr = d.buf.data();
    return d.ptr;
  }
  DevTensor upload(const std::string& name) {
    const TensorInfo& t = ck.at(name);
    DevTensor d;
    d.dtype = t.dtype;
    d.shape = t.shape;
    items.push_back({&t, alloc(d, t.nbytes)});
    return d;
  }
  // An fp32 tensor that a checkpoint may store as bf16 (converted after the pass) or fp16 (converted here).
  void upload_f32(const std::string& name, DevTensor& dst) {
    const TensorInfo& t = ck.at(name);
    if (t.dtype == DType::F32) {
      dst = upload(name);
      return;
    }
    if (t.dtype == DType::F16) {
      const auto src = ck.as<uint16_t>(t);
      std::vector<float> host(src.size());
      for (size_t i = 0; i < src.size(); ++i) host[i] = half_to_float(src[i]);
      dst.dtype = DType::F32;
      dst.shape = t.shape;
      rt::DeviceGuard g(dev);
      UH_CHECK(hipMemcpy(alloc(dst, host.size() * 4), host.data(), host.size() * 4, hipMemcpyHostToDevice));
      return;
    }
    if (t.dtype != DType::BF16) throw std::runtime_error(name + ": expected F32 or BF16");
    to_f32.push_back({upload(name), &dst});
  }
  void run(hipStream_t st) {
    rt::DeviceGuard g(dev);
    if (!items.empty()) {
      Uploader up(ck);
      up.upload_many(std::move(items), st);
      up.sync();
      items.clear();
    }
    for (auto& [src, dst] : to_f32) {
      const size_t n = size_t(src.buf.size() / 2);
      DevTensor f;
      f.dtype = DType::F32;
      f.shape = src.shape;
      alloc(f, n * 4);
      k::cast_bf16_f32(src.p<uint16_t>(), f.p<float>(), n, st);
      UH_CHECK(hipStreamSynchronize(st));
      *dst = std::move(f);
    }
    to_f32.clear();
  }
};

GLinear glinear(LoadCtx& lc, const std::string& prefix) {
  GLinear l;
  if (lc.ck.find(prefix + ".trellis")) {
    const auto info = find_exl3_linear(lc.ck, prefix);
    if (!info || info->cb != Exl3Codebook::Mul1) throw std::runtime_error(prefix + ": not an EXL3 mul1 linear");
    l.x.trellis = lc.upload(prefix + ".trellis");
    l.x.suh = lc.upload(prefix + ".suh");
    l.x.svh = lc.upload(prefix + ".svh");
    l.x.in = l.in = info->in_features;
    l.x.out = l.out = info->out_features;
    l.x.bits = info->K;
    return l;
  }
  l.w = lc.upload(prefix + ".weight");
  if (l.w.shape.size() != 2) throw std::runtime_error(prefix + ": not a matrix");
  l.out = int(l.w.shape[0]);
  l.in = int(l.w.shape[1]);
  if (l.w.dtype == DType::F8_E4M3) {
    l.s = lc.upload(prefix + ".weight_scale_inv");
    if (l.s.dtype != DType::F32) throw std::runtime_error(prefix + ": fp8 without fp32 block scales");
  } else if (l.w.dtype != DType::BF16 && l.w.dtype != DType::F16) {
    throw std::runtime_error(prefix + ": unsupported dtype " + dtype_name(l.w.dtype));
  }
  return l;
}

// A bf16 linear -> 128x128 block fp8 with fp32 scales (kernels::quant_fp8_block).
void requant_fp8(GLinear& l, int dev, hipStream_t st) {
  if (l.exl3() || l.w.dtype != DType::BF16) return;
  DevTensor q, s;
  q.buf = rt::DeviceBuffer(dev, size_t(l.out) * l.in);
  q.ptr = q.buf.data();
  q.dtype = DType::F8_E4M3;
  q.shape = l.w.shape;
  const size_t nb = size_t((l.out + 127) / 128) * ((l.in + 127) / 128);
  s.buf = rt::DeviceBuffer(dev, nb * 4);
  s.ptr = s.buf.data();
  s.dtype = DType::F32;
  s.shape = {(l.out + 127) / 128, (l.in + 127) / 128};
  rt::DeviceGuard g(dev);
  k::quant_fp8_block(l.w.p<uint16_t>(), q.p<uint8_t>(), s.p<float>(), l.out, l.in, st);
  UH_CHECK(hipStreamSynchronize(st));
  l.w = std::move(q);
  l.s = std::move(s);
}

}  // namespace

Glm5LayerWeights load_glm5_layer(const Checkpoint& ck, const Glm5Config& c, int layer, int dev, hipStream_t st,
                                 const Glm5LoadOptions& opt) {
  const std::string p = kPrefix + "layers." + std::to_string(layer) + ".";
  LoadCtx lc{ck, dev, {}, {}, {}};
  Glm5LayerWeights w;
  w.layer = layer;
  w.kda = c.is_kda(layer);
  w.dense = c.is_dense(layer);
  lc.upload_f32(p + "hc_attn_fn", w.hc_attn_fn);
  w.hc_attn_base = lc.upload(p + "hc_attn_base");
  w.hc_attn_scale = lc.upload(p + "hc_attn_scale");
  lc.upload_f32(p + "hc_ffn_fn", w.hc_ffn_fn);
  w.hc_ffn_base = lc.upload(p + "hc_ffn_base");
  w.hc_ffn_scale = lc.upload(p + "hc_ffn_scale");
  w.input_norm = lc.upload(p + "input_layernorm.weight");
  w.post_norm = lc.upload(p + "post_attention_layernorm.weight");

  const std::string a = p + "self_attn.";
  w.o = glinear(lc, a + "o_proj");
  std::unique_ptr<LoadCtx> ax;  // tensors from opt.aux
  if (w.kda) {
    const int W = c.lin_width();
    if (ck.find(a + "qkv_proj.trellis")) {
      w.qkv = glinear(lc, a + "qkv_proj");
    } else {
      w.q = glinear(lc, a + "q_proj");
      w.k = glinear(lc, a + "k_proj");
      w.v = glinear(lc, a + "v_proj");
    }
    // conv [3 W, 4] bf16: one tensor (EXL3) or q / k / v ones (the release), concatenated.
    if (ck.find(a + "conv1d.weight")) {
      w.conv = lc.upload(a + "conv1d.weight");
    } else {
      lc.alloc(w.conv, size_t(3) * W * c.conv_kernel * 2);
      w.conv.dtype = DType::BF16;
      const char* parts[3] = {"q_conv1d", "k_conv1d", "v_conv1d"};
      for (int i = 0; i < 3; ++i) {
        const TensorInfo& t = ck.at(a + parts[i] + ".weight");
        if (t.nbytes != size_t(W) * c.conv_kernel * 2) throw std::runtime_error(t.name + ": unexpected size");
        lc.items.push_back({&t, static_cast<uint8_t*>(w.conv.ptr) + size_t(i) * t.nbytes});
      }
    }
    if (w.conv.dtype != DType::BF16) throw std::runtime_error(a + "conv: expected bf16");
    w.f_a = glinear(lc, a + "f_a_proj");
    w.f_b = glinear(lc, a + "f_b_proj");
    w.b = glinear(lc, a + "b_proj");
    w.g_a = glinear(lc, a + "g_a_proj");
    w.g_b = glinear(lc, a + "g_b_proj");
    w.dt_bias = lc.upload(a + "dt_bias");
    w.a_log = lc.upload(a + "A_log");
    w.o_norm = lc.upload(a + "o_norm.weight");
    if (w.dt_bias.dtype != DType::F32 || w.a_log.dtype != DType::F32)
      throw std::runtime_error(a + ": gate params not f32");
  } else {
    w.q_a = glinear(lc, a + "q_a_proj");
    w.q_b = glinear(lc, a + "q_b_proj");
    w.kv_a = glinear(lc, a + "kv_a_proj_with_mqa");
    w.kv_b = glinear(lc, a + "kv_b_proj");
    if (w.kv_b.exl3()) throw std::runtime_error(a + "kv_b_proj: EXL3 is not supported (absorbed attention)");
    w.q_a_norm = lc.upload(a + "q_a_layernorm.weight");
    w.kv_a_norm = lc.upload(a + "kv_a_layernorm.weight");
    const std::string ip = a + "indexer.";
    w.idx_wq_b = glinear(lc, ip + "wq_b");
    w.idx_wk = glinear(lc, ip + "wk");
    w.idx_wproj = glinear(lc, ip + "weights_proj");
    w.idx_knorm_w = lc.upload(ip + "k_norm.weight");
    w.idx_knorm_b = lc.upload(ip + "k_norm.bias");
    w.idx_f16 = w.idx_knorm_w.dtype == DType::F16;
    // The pooling weights: bf16 in the release, fp16 / fp32 in exllamav3 4.05, left out of 3.05's index.
    LoadCtx* pc = &lc;
    if (!ck.find(ip + "index_kpool_compress_gate.weight") && !ck.find(ip + "index_kpool_compress_gate")) {
      if (!opt.aux)
        throw std::runtime_error(ip + "index_kpool_compress_*: not in the checkpoint (pass the FP8 release as aux)");
      ax = std::make_unique<LoadCtx>(LoadCtx{*opt.aux, dev, {}, {}, {}});
      pc = ax.get();
    }
    w.idx_gate.w = pc->upload(ip + "index_kpool_compress_gate");
    w.idx_gate.out = int(w.idx_gate.w.shape[0]);
    w.idx_gate.in = int(w.idx_gate.w.shape[1]);
    pc->upload_f32(ip + "index_kpool_compress_ape", w.idx_ape);
    if (w.idx_gate.w.dtype != DType::BF16 && w.idx_gate.w.dtype != DType::F16)
      throw std::runtime_error(ip + "index_kpool_compress_gate: expected bf16 or fp16");
  }

  const std::string m = p + "mlp.";
  if (w.dense) {
    w.gate = glinear(lc, m + "gate_proj");
    w.up = glinear(lc, m + "up_proj");
    w.down = glinear(lc, m + "down_proj");
  } else {
    w.router = glinear(lc, m + "gate");
    w.router_bias = lc.upload(m + "gate.e_score_correction_bias");
    if (w.router.exl3() || w.router_bias.dtype != DType::F32) throw std::runtime_error(m + "gate: unexpected format");
    w.sh_gate = glinear(lc, m + "shared_experts.gate_proj");
    w.sh_up = glinear(lc, m + "shared_experts.up_proj");
    w.sh_down = glinear(lc, m + "shared_experts.down_proj");
  }
  lc.run(st);
  if (ax) ax->run(st);
  if (w.kda && opt.kda_fp8)
    for (GLinear* l : {&w.q, &w.k, &w.v, &w.o}) requant_fp8(*l, dev, st);

  if (!w.dense && opt.load_experts) {
    LoadCtx ex{opt.experts ? *opt.experts : ck, dev, {}, {}, {}};
    const size_t E = size_t(c.n_experts);
    w.e_gate.resize(E);
    w.e_up.resize(E);
    w.e_down.resize(E);
    for (size_t e = 0; e < E; ++e) {
      const std::string ep = m + "experts." + std::to_string(e) + ".";
      w.e_gate[e] = glinear(ex, ep + "gate_proj");
      w.e_up[e] = glinear(ex, ep + "up_proj");
      w.e_down[e] = glinear(ex, ep + "down_proj");
    }
    ex.run(st);
  }
  return w;
}

Glm5HeadWeights load_glm5_head(const Checkpoint& ck, int dev, hipStream_t st, bool embed, bool head) {
  LoadCtx lc{ck, dev, {}, {}, {}};
  Glm5HeadWeights h;
  if (embed) h.embed = lc.upload(kPrefix + "embed_tokens.weight");
  if (head) {
    h.norm = lc.upload(kPrefix + "norm.weight");
    h.head = glinear(lc, "lm_head");
  }
  lc.run(st);
  return h;
}

// ---------------------------------------------------------------------------------------------------
// State and scratch

Glm5Cache make_glm5_cache(const Glm5Config& c, int layer, int slots, int max_seq, int dev, hipStream_t st) {
  Glm5Cache k;
  k.layer = layer;
  k.slots = slots;
  k.max_seq = max_seq;
  k.kda = c.is_kda(layer);
  rt::DeviceGuard g(dev);
  auto zeroed = [&](size_t bytes) {
    rt::DeviceBuffer b(dev, bytes);
    UH_CHECK(hipMemsetAsync(b.data(), 0, bytes, st));
    return b;
  };
  if (k.kda) {
    k.conv = zeroed(size_t(slots) * (c.conv_kernel - 1) * 3 * c.lin_width() * 2);
    k.state = zeroed(size_t(slots) * c.lin_heads * c.lin_dim * c.lin_dim * 4);
  } else {
    k.kv = zeroed(size_t(slots) * max_seq * c.kv_lora * 2);
    k.ik = zeroed(size_t(slots) * max_seq * c.idx_dim * 2);
    k.ig = zeroed(size_t(slots) * max_seq * c.idx_dim * 2);
    k.pk = zeroed(size_t(slots) * (max_seq / c.idx_pool + 1) * c.idx_dim * 2);
  }
  UH_CHECK(hipStreamSynchronize(st));
  return k;
}

void clear_glm5_slot(const Glm5Config& c, Glm5Cache& k, int slot, hipStream_t st) {
  if (k.kda) {
    const size_t cb = k.conv.size() / k.slots, sb = k.state.size() / k.slots;
    UH_CHECK(hipMemsetAsync(k.conv.as<uint8_t>() + slot * cb, 0, cb, st));
    UH_CHECK(hipMemsetAsync(k.state.as<uint8_t>() + slot * sb, 0, sb, st));
  }
  // DSA rows are written before they are read (positions < the sequence's length only).
  (void)c;
}

Glm5Scratch make_glm5_scratch(const Glm5Config& c, int dev, int max_tokens, int max_seq) {
  Glm5Scratch s;
  s.max_tokens = max_tokens;
  s.max_seq = max_seq;
  const size_t T = size_t(max_tokens), D = c.dim, W = c.lin_width(), H = c.n_heads, R = c.kv_lora;
  const size_t P = T * c.n_active;
  const size_t wide_in = std::max({D, size_t(c.q_lora), H * c.v_dim, W, size_t(c.inter), size_t(c.moe_inter)});
  const size_t wide_out = std::max({3 * W, H * c.nope_dim, size_t(c.inter), D});
  s.head_rows = int(std::min<size_t>(T, 64));
  const size_t ffn = std::max(P * c.moe_inter, T * c.inter);
  const size_t E = c.n_experts, max_tiles = P / k::kExl3TileRows + E + 1;
  const size_t sizes[kGlm5ScratchBufs] = {
      T * D * 2,                                  // h
      T * D * 2,                                  // hn
      T * c.hc_mult * 4,                          // post
      T * c.hc_mult * c.hc_mult * 4,              // comb
      T * D * 2,                                  // out
      T * 3 * W * 2,                              // qkv
      T * 3 * W * 2,                              // qkv after conv
      T * c.lin_dim * 2,                          // f_a / g_a out
      T * W * 2,                                  // f_b / g_b out
      T * W * 4,                                  // g
      T * c.lin_heads * 2,                        // beta
      T * W * 2,                                  // core
      T * W * 2,                                  // out norm
      T * c.q_lora * 2,                           // q_resid
      T * H * c.nope_dim * 2,                     // q
      T * R * 2,                                  // latent
      T * H * R * 4,                              // absorbed q
      T * H * R * 4,                              // latent out
      T * H * c.v_dim * 2,                        // attention out
      T * c.idx_heads * c.idx_dim * 2,            // indexer q
      T * c.idx_dim * 2,                          // indexer k
      T * c.idx_dim * 2,                          // indexer gate scores
      T * c.idx_heads * 2,                        // indexer head weights
      size_t(kIdxBlock) * (max_seq / c.idx_pool + 1) * 4,  // index scores
      T * (c.idx_topk / c.idx_pool) * 4,          // selected pools
      T * (c.idx_topk + c.idx_pool - 1) * 4,      // selected tokens
      T * std::max(D, size_t(c.n_experts)) * 4,   // f32 rows / router logits
      T * D * 4,                                  // f32 hn
      P * 4,                                      // sel
      P * 4,                                      // routing weights
      P * 4,                                      // pair -> token
      P * 4,                                      // pair weights
      P * 4,                                      // token -> pairs
      P * D * 2,                                  // gathered expert inputs
      ffn * 2,                                    // gate
      ffn * 2,                                    // up
      ffn * 2,                                    // swiglu
      P * D * 2,                                  // expert outputs
      T * D * 2,                                  // shared expert out
      T * wide_in * 4,                            // EXL3 x side
      std::max(T * wide_out, size_t(s.head_rows) * c.vocab) * 4,  // EXL3 y side
      size_t(s.head_rows) * c.vocab * 2,          // head bf16 logits
      (T / k::kExl3TileRows + 2) * sizeof(k::Exl3Tile),  // dense tiles
      std::max(P, T) * 4,                         // zero rows (row_expert of a dense linear)
      std::min<size_t>(T, k::kHcSmallMaxTokens) * 32 * 25 * 4,  // hc_pre_small partial sums
      E * 4, E * 4, E * 4,                        // MoE count, offsets, fill
      sizeof(k::MoePlan),                         // MoE plan
      max_tiles * sizeof(k::Exl3Tile),            // local tiles
      max_tiles * sizeof(k::Exl3Tile),            // remote tiles
      P * 4, P * 4, P * 4,                        // pair -> token, weight, expert
      P * D * 2, P * D * 2,                       // fp16 expert inputs (gate, up)
      std::max(P * c.moe_inter, T * c.inter) * 2,  // fp16 down inputs
      std::max(std::min<size_t>(P, 64) * std::max(D, size_t(c.moe_inter)),
               8 * std::max(3 * W, size_t(c.vocab))) * 4,  // GEMV fp32 sides
  };
  rt::DeviceGuard g(dev);
  for (size_t b : sizes) {
    s.bufs.emplace_back(dev, std::max<size_t>(b, 16));
    UH_CHECK(hipMemset(s.bufs.back().data(), 0, s.bufs.back().size()));  // MoE counts and zero rows start at 0
  }
  return s;
}

}  // namespace unholy::model
