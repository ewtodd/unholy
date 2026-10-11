#include "model/dsv4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "kernels/ref.hpp"
#include "model/exl3.hpp"
#include "model/loader.hpp"

namespace unholy::model {

namespace k = unholy::kernels;
using bf16 = uint16_t;

// ---------------------------------------------------------------------------------------------------
// Config

Dsv4Config Dsv4Config::from_hf(const gufo::json::Value& j) {
  auto num = [&](const char* key) {
    const auto* v = j.find(key);
    if (!v || !v->is_number()) throw std::runtime_error(std::string("config: missing ") + key);
    return v->as_double();
  };
  auto ints = [&](const char* key) {
    std::vector<int> out;
    if (const auto* v = j.find(key))
      for (const auto& e : v->items()) out.push_back(int(e.as_double()));
    return out;
  };
  Dsv4Config c;
  c.vocab = int(num("vocab_size"));
  c.dim = int(num("hidden_size"));
  c.n_layers = int(num("num_hidden_layers"));
  c.n_heads = int(num("num_attention_heads"));
  c.head_dim = int(num("head_dim"));
  c.rope_dim = int(num("qk_rope_head_dim"));
  c.q_lora = int(num("q_lora_rank"));
  c.o_lora = int(num("o_lora_rank"));
  c.o_groups = int(num("o_groups"));
  c.window = int(num("sliding_window"));
  c.idx_heads = int(num("index_n_heads"));
  c.idx_dim = int(num("index_head_dim"));
  c.idx_topk = int(num("index_topk"));
  c.n_experts = int(num("n_routed_experts"));
  c.n_active = int(num("num_experts_per_tok"));
  c.moe_inter = int(num("moe_intermediate_size"));
  c.n_hash_layers = int(num("num_hash_layers"));
  c.hc_mult = int(num("hc_mult"));
  c.sinkhorn_iters = int(num("hc_sinkhorn_iters"));
  c.norm_eps = float(num("rms_norm_eps"));
  c.hc_eps = float(num("hc_eps"));
  c.route_scale = float(num("routed_scaling_factor"));
  c.swiglu_limit = float(num("swiglu_limit"));
  c.rope_theta = num("rope_theta");
  c.compress_rope_theta = num("compress_rope_theta");
  const auto* rs = j.find("rope_scaling");
  if (!rs) throw std::runtime_error("config: missing rope_scaling");
  c.rope_factor = rs->member_double("factor");
  c.original_seq_len = int(rs->member_double("original_max_position_embeddings"));
  c.beta_fast = rs->member_double("beta_fast");
  c.beta_slow = rs->member_double("beta_slow");
  c.compress_ratios = ints("compress_ratios");
  c.n_mtp = int(num("num_nextn_predict_layers"));
  c.dspark_block = int(j.member_double("dspark_block_size"));
  c.dspark_noise_token = int(j.member_double("dspark_noise_token_id"));
  c.dspark_markov_rank = int(j.member_double("dspark_markov_rank"));
  c.dspark_target_layers = ints("dspark_target_layer_ids");
  if (int(c.compress_ratios.size()) < c.n_layers) throw std::runtime_error("config: compress_ratios too short");
  return c;
}

// precompute_freqs_cis (inference/model.py), fp32 like torch: freqs = 1 / base^(2i/dim); YaRN blend;
// angle = float(t) * freq; (cos, sin).
std::vector<float> rope_table(const Dsv4Config& c, int max_pos, bool yarn) {
  const int dim = c.rope_dim, half = dim / 2;
  const double base = yarn ? c.compress_rope_theta : c.rope_theta;
  std::vector<float> freqs(static_cast<size_t>(half));
  for (int i = 0; i < half; ++i) {
    const float e = float(2 * i) / float(dim);
    freqs[size_t(i)] = 1.0f / std::pow(float(base), e);
  }
  if (yarn && c.original_seq_len > 0) {
    auto corr_dim = [&](double rot) {
      return dim * std::log(c.original_seq_len / (rot * 2 * M_PI)) / (2 * std::log(base));
    };
    double lo = std::max(std::floor(corr_dim(c.beta_fast)), 0.0);
    double hi = std::min(std::ceil(corr_dim(c.beta_slow)), double(dim - 1));
    if (lo == hi) hi += 0.001;
    for (int i = 0; i < half; ++i) {
      float ramp = (float(i) - float(lo)) / float(hi - lo);
      ramp = std::min(std::max(ramp, 0.0f), 1.0f);
      const float smooth = 1.0f - ramp;
      const float f = freqs[size_t(i)];
      freqs[size_t(i)] = f / float(c.rope_factor) * (1.0f - smooth) + f * smooth;
    }
  }
  std::vector<float> out(size_t(max_pos) * half * 2);
  for (int t = 0; t < max_pos; ++t)
    for (int i = 0; i < half; ++i) {
      const float ang = float(t) * freqs[size_t(i)];
      out[(size_t(t) * half + i) * 2 + 0] = std::cos(ang);
      out[(size_t(t) * half + i) * 2 + 1] = std::sin(ang);
    }
  return out;
}

RopeTables make_rope_tables(const Dsv4Config& c, int dev, int max_pos) {
  RopeTables r;
  r.max_pos = max_pos;
  for (int yarn = 0; yarn < 2; ++yarn) {
    const auto host = rope_table(c, max_pos, yarn);
    rt::DeviceBuffer b(dev, host.size() * sizeof(float));
    rt::DeviceGuard g(dev);
    UH_CHECK(hipMemcpy(b.data(), host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
    (yarn ? r.yarn : r.plain) = std::move(b);
  }
  return r;
}

// ---------------------------------------------------------------------------------------------------
// Weights

namespace {

// Collects (tensor, device destination) pairs; the bytes move in one coalesced pass at the end.
struct LoadCtx {
  const Checkpoint& ck;
  int dev;
  std::vector<Uploader::Item> items;
  rt::Arena* arena = nullptr;
};

void allocate(LoadCtx& lc, DevTensor& d, size_t bytes) {
  if (lc.arena) {
    d.ptr = lc.arena->alloc(bytes);
  } else {
    d.buf = rt::DeviceBuffer(lc.dev, bytes);
    d.ptr = d.buf.data();
  }
}

DevTensor upload(LoadCtx& lc, const std::string& name) {
  const TensorInfo& t = lc.ck.at(name);
  DevTensor d;
  d.dtype = t.dtype;
  d.shape = t.shape;
  allocate(lc, d, t.nbytes);
  lc.items.push_back({&t, d.ptr});
  return d;
}

DevTensor upload_i64_as_i32(LoadCtx& lc, const std::string& name) {
  const TensorInfo& t = lc.ck.at(name);
  if (t.dtype != DType::I64) throw std::runtime_error(name + ": expected I64");
  const auto src = lc.ck.as<int64_t>(t);
  std::vector<int32_t> host(src.size());
  for (size_t i = 0; i < src.size(); ++i) host[i] = int32_t(src[i]);
  DevTensor d;
  d.dtype = DType::I32;
  d.shape = t.shape;
  rt::DeviceGuard g(lc.dev);
  allocate(lc, d, host.size() * 4);
  UH_CHECK(hipMemcpy(d.ptr, host.data(), host.size() * 4, hipMemcpyHostToDevice));
  return d;
}

Fp8Linear fp8(LoadCtx& lc, const std::string& prefix) {
  Fp8Linear l;
  l.w = upload(lc, prefix + ".weight");
  l.s = upload(lc, prefix + ".scale");
  if (l.w.dtype != DType::F8_E4M3 || l.s.dtype != DType::F8_E8M0) throw std::runtime_error(prefix + ": not fp8");
  l.out = int(l.w.shape[0]);
  l.in = int(l.w.shape[1]);
  return l;
}

Fp4Linear fp4(LoadCtx& lc, const std::string& prefix) {
  Fp4Linear l;
  l.w = upload(lc, prefix + ".weight");
  l.s = upload(lc, prefix + ".scale");
  if (l.w.dtype != DType::I8 || l.s.dtype != DType::F8_E8M0) throw std::runtime_error(prefix + ": not MXFP4");
  l.out = int(l.w.shape[0]);
  l.in = int(l.w.shape[1] * 2);
  return l;
}

Exl3Weights exl3_linear(LoadCtx& lc, const std::string& prefix) {
  const auto info = find_exl3_linear(lc.ck, prefix);  // validates shapes and the mul1 marker
  if (!info || info->cb != Exl3Codebook::Mul1) throw std::runtime_error(prefix + ": not an EXL3 mul1 linear");
  Exl3Weights x;
  x.trellis = upload(lc, prefix + ".trellis");
  x.suh = upload(lc, prefix + ".suh");
  x.svh = upload(lc, prefix + ".svh");
  x.in = info->in_features;
  x.out = info->out_features;
  x.bits = info->K;
  return x;
}

CompressorWeights compressor(LoadCtx& lc, const std::string& prefix, int ratio, int d, bool rotate) {
  CompressorWeights cw;
  cw.wkv = upload(lc, prefix + ".wkv.weight");
  cw.wgate = upload(lc, prefix + ".wgate.weight");
  cw.ape = upload(lc, prefix + ".ape");
  cw.norm = upload(lc, prefix + ".norm.weight");
  cw.ratio = ratio;
  cw.d = d;
  cw.overlap = ratio == 4;
  cw.rotate = rotate;
  if (cw.wkv.shape[0] != cw.width()) throw std::runtime_error(prefix + ": unexpected compressor width");
  return cw;
}

}  // namespace

LayerWeights load_layer(const Checkpoint& ck, const Dsv4Config& c, int layer, int dev, hipStream_t st,
                        const LoadOptions& opt) {
  const std::string p = "layers." + std::to_string(layer) + ".";
  LoadCtx lc{ck, dev, {}, opt.arena};
  LayerWeights w;
  w.layer = layer;
  w.ratio = c.ratio(layer);
  w.hash = layer < c.n_hash_layers;
  w.has_indexer = w.ratio == 4;
  if (opt.dense) {
    w.hc_attn_fn = upload(lc, p + "hc_attn_fn");
    w.hc_attn_base = upload(lc, p + "hc_attn_base");
    w.hc_attn_scale = upload(lc, p + "hc_attn_scale");
    w.hc_ffn_fn = upload(lc, p + "hc_ffn_fn");
    w.hc_ffn_base = upload(lc, p + "hc_ffn_base");
    w.hc_ffn_scale = upload(lc, p + "hc_ffn_scale");
    w.attn_norm = upload(lc, p + "attn_norm.weight");
    w.ffn_norm = upload(lc, p + "ffn_norm.weight");
    w.sink = upload(lc, p + "attn.attn_sink");
    w.q_norm = upload(lc, p + "attn.q_norm.weight");
    w.kv_norm = upload(lc, p + "attn.kv_norm.weight");
    w.wq_a = fp8(lc, p + "attn.wq_a");
    w.wq_b = fp8(lc, p + "attn.wq_b");
    w.wkv = fp8(lc, p + "attn.wkv");
    w.wo_a = fp8(lc, p + "attn.wo_a");
    w.wo_b = fp8(lc, p + "attn.wo_b");
    if (w.ratio) w.comp = compressor(lc, p + "attn.compressor", w.ratio, c.head_dim, false);
    if (w.has_indexer) {
      w.idx_wq_b = fp8(lc, p + "attn.indexer.wq_b");
      w.idx_wproj = upload(lc, p + "attn.indexer.weights_proj.weight");
      w.idx_comp = compressor(lc, p + "attn.indexer.compressor", 4, c.idx_dim, true);
    }
    w.gate_w = upload(lc, p + "ffn.gate.weight");
    w.gate_bias = upload(lc, p + "ffn.gate.bias");
    if (w.hash) w.tid2eid = upload_i64_as_i32(lc, p + "ffn.gate.tid2eid");
    w.sw1 = fp8(lc, p + "ffn.shared_experts.w1");
    w.sw2 = fp8(lc, p + "ffn.shared_experts.w2");
    w.sw3 = fp8(lc, p + "ffn.shared_experts.w3");
  }
  LoadCtx lx{opt.exl3 ? *opt.exl3 : ck, dev, {}, opt.arena};
  w.exl3_experts = opt.exl3 != nullptr;
  const size_t E = size_t(c.n_experts);
  if (w.exl3_experts) {
    w.x1.resize(E);
    w.x2.resize(E);
    w.x3.resize(E);
  } else {
    w.w1.resize(E);
    w.w2.resize(E);
    w.w3.resize(E);
  }
  for (int e = 0; e < c.n_experts; ++e) {
    if (opt.experts && !(*opt.experts)[size_t(e)]) continue;
    const std::string ep = p + "ffn.experts." + std::to_string(e) + ".";
    if (w.exl3_experts) {
      w.x1[size_t(e)] = exl3_linear(lx, ep + "w1");
      w.x2[size_t(e)] = exl3_linear(lx, ep + "w2");
      w.x3[size_t(e)] = exl3_linear(lx, ep + "w3");
    } else {
      w.w1[size_t(e)] = fp4(lc, ep + "w1");
      w.w2[size_t(e)] = fp4(lc, ep + "w2");
      w.w3[size_t(e)] = fp4(lc, ep + "w3");
    }
  }

  rt::DeviceGuard g(dev);
  auto run = [&](const Checkpoint& src, Uploader* up, std::vector<Uploader::Item>& items) {
    if (items.empty()) return;
    std::unique_ptr<Uploader> own;
    if (!up) {
      own = std::make_unique<Uploader>(src);
      up = own.get();
    }
    up->upload_many(std::move(items), st);
    up->sync();
  };
  run(ck, opt.up, lc.items);
  if (opt.exl3) run(*opt.exl3, opt.exl3_up, lx.items);
  if (w.exl3_experts) {
    // Pointer tables for the grouped kernels.
    std::vector<const void*> tab(9 * E, nullptr);
    for (size_t e = 0; e < E; ++e) {
      const Exl3Weights* m3[3] = {&w.x1[e], &w.x3[e], &w.x2[e]};
      for (int m = 0; m < 3; ++m) {
        if (!m3[m]->trellis) continue;
        if (w.exl3_bits[m] && w.exl3_bits[m] != m3[m]->bits)
          throw std::runtime_error(p + ": mixed EXL3 bit widths in one projection");
        w.exl3_bits[m] = m3[m]->bits;
        tab[size_t(m * 3 + 0) * E + e] = m3[m]->trellis.ptr;
        tab[size_t(m * 3 + 1) * E + e] = m3[m]->suh.ptr;
        tab[size_t(m * 3 + 2) * E + e] = m3[m]->svh.ptr;
      }
    }
    w.exl3_tab = rt::DeviceBuffer(dev, tab.size() * sizeof(void*));
    UH_CHECK(hipMemcpy(w.exl3_tab.data(), tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice));
  }
  return w;
}

}  // namespace unholy::model
