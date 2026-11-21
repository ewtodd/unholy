// Hy4 flash: configuration, weight loading, the Engram store, per-sequence state, scratch (model/hy4.hpp).
#include "model/hy4.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <mutex>
#include <thread>

#include "kernels/glm_fast.hpp"
#include "kernels/hy4.hpp"
#include "kernels/mx_moe.hpp"
#include "kernels/ref.hpp"
#include "model/loader.hpp"

namespace unholy::model {
constexpr int kHy4Pool = kernels::kHy4Pool, kHy4IdxDim = kernels::kHy4IdxDim;
}

namespace unholy::model {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// Config

int Hy4Config::n_engram() const {
  int n = 0;
  for (int e : engram) n = std::max(n, e + 1);
  return n;
}

Hy4Config Hy4Config::from_hf(const gufo::json::Value& top) {
  const gufo::json::Value& j = top.find("text_config") ? *top.find("text_config") : top;
  auto num = [](const gufo::json::Value& o, const char* key) {
    const auto* v = o.find(key);
    if (!v || !v->is_number()) throw std::runtime_error(std::string("config: missing ") + key);
    return v->as_double();
  };
  auto num_or = [](const gufo::json::Value& o, const char* key, double d) {
    const auto* v = o.find(key);
    return v && v->is_number() ? v->as_double() : d;
  };
  auto flag = [&](const char* key) {
    const auto* v = j.find(key);
    return v && v->as_bool();
  };
  auto strs = [&](const char* key) {
    std::vector<std::string> out;
    const auto* v = j.find(key);
    if (!v) throw std::runtime_error(std::string("config: missing ") + key);
    for (const auto& e : v->items()) out.push_back(e.get_str());
    return out;
  };
  Hy4Config c;
  c.vocab = int(num(j, "vocab_size"));
  c.dim = int(num(j, "hidden_size"));
  c.n_layers = int(num(j, "num_hidden_layers"));
  c.q_lora = int(num(j, "q_lora_rank"));
  c.kv_lora = int(num(j, "kv_lora_rank"));
  c.nope = int(num(j, "qk_nope_head_dim"));
  c.rope = int(num(j, "qk_rope_head_dim"));
  c.v_dim = int(num(j, "v_head_dim"));
  c.heads_sparse = int(num(j, "num_attention_heads"));
  c.heads_window = int(num_or(j, "swa_num_attention_heads", 64));
  c.window = int(num(j, "sliding_window"));
  c.theta_sparse = 1e7f;
  c.theta_window = 1e4f;
  if (const auto* rp = j.find("rope_parameters")) {
    if (const auto* s = rp->find("deepseek_sparse_attention")) c.theta_sparse = float(num_or(*s, "rope_theta", 1e7));
    if (const auto* s = rp->find("indexed_attention")) c.theta_sparse = float(num_or(*s, "rope_theta", 1e7));
    if (const auto* s = rp->find("sliding_attention")) c.theta_window = float(num_or(*s, "rope_theta", 1e4));
  }
  c.idx_heads = int(num(j, "index_n_heads"));
  c.idx_dim = int(num(j, "index_head_dim"));
  c.idx_topk = int(num(j, "index_topk"));
  c.idx_pool = int(num_or(j, "index_kpool", num_or(j, "compressed_indexer_compress_ratio", 4)));
  c.inter = int(num(j, "intermediate_size"));
  c.moe_inter = int(num(j, "moe_intermediate_size"));
  c.latent = int(num(j, "routed_expert_hidden_size"));
  c.n_experts = int(num(j, "n_routed_experts"));
  c.n_active = int(num(j, "num_experts_per_tok"));
  c.route_scale = float(num(j, "routed_scaling_factor"));
  c.swiglu_limit = float(num(j, "swiglu_limit"));
  c.hc = int(num(j, "hc_mult"));
  c.hc_eps = float(num(j, "hc_eps"));
  c.hc_mag = float(num_or(j, "hc_magnitude", 2.0));
  c.eps = float(num(j, "rms_norm_eps"));
  c.bos = int(num_or(j, "bos_token_id", 120000));
  c.eos = int(num_or(j, "eos_token_id", 120025));
  c.pad = int(num_or(j, "pad_token_id", 120002));
  c.engram_pad = int(num_or(j, "engram_pad_id", c.pad));
  c.image_token = int(num_or(top, "image_token_id", -1));
  c.video_token = int(num_or(top, "video_token_id", -1));
  for (const auto& t : strs("layer_types")) {
    const bool sp = t == "deepseek_sparse_attention" || t == "indexed_attention";
    if (!sp && t != "sliding_attention") throw std::runtime_error("config: layer type " + t);
    c.sparse.push_back(sp);
  }
  for (const auto& t : strs("mlp_layer_types")) {
    if (t != "dense" && t != "sparse") throw std::runtime_error("config: mlp layer type " + t);
    c.dense.push_back(t == "dense");
  }
  c.engram.assign(size_t(c.n_layers), -1);
  if (const auto* ids = j.find("engram_layer_ids")) {
    int m = 0;
    for (const auto& e : ids->items()) {
      const int l = int(e.as_double()) - 1;  // HYV4FlashDecoderLayer(i) holds the memory with id i + 1
      if (l < 0 || l >= c.n_layers) throw std::runtime_error("config: engram layer id out of range");
      c.engram[size_t(l)] = m++;
    }
  }
  // What this implementation assumes about the release.
  if (int(num(j, "n_group")) != 1 || int(num(j, "topk_group")) != 1 || !flag("norm_topk_prob"))
    throw std::runtime_error("config: router other than one group with normalized weights");
  if (c.hc != 4 || c.kv_lora != k::kHy4Lat || c.rope != 64 || c.kv_lora + c.rope != k::kHy4Row || c.nope != 192 ||
      c.v_dim != 256 || c.window != k::kHy4Window || c.idx_pool != k::kHy4Pool || c.idx_dim != k::kHy4IdxDim ||
      c.idx_heads != k::kHy4IdxHeads || c.dim != 3072 || c.latent % 128 || c.moe_inter % 128)
    throw std::runtime_error("config: unexpected shape (the kernels are built for Hy4 flash's)");
  if (int(c.sparse.size()) != c.n_layers || int(c.dense.size()) != c.n_layers)
    throw std::runtime_error("config: layer type lists do not match num_hidden_layers");
  if (c.n_engram() > 4) throw std::runtime_error("config: more than 4 Engram memories");
  return c;
}

// ---------------------------------------------------------------------------------------------------
// Weights

namespace {

const std::string kPrefix = "model.";

struct LoadCtx {
  const Checkpoint& ck;
  int dev;
  Uploader* up = nullptr;
  std::vector<Uploader::Item> items;
  std::deque<TensorInfo> slices;  // synthetic tensors (one expert of a stacked tensor), alive until run()
  struct Deq {
    DevTensor q, s;
    Hy4Linear* dst;
  };
  std::vector<Deq> deq;

  DevTensor alloc(size_t bytes, DType dt, std::vector<int64_t> shape) {
    DevTensor d;
    d.buf = rt::DeviceBuffer(dev, bytes);
    d.ptr = d.buf.data();
    d.dtype = dt;
    d.shape = std::move(shape);
    return d;
  }
  DevTensor upload(const std::string& name) {
    const TensorInfo& t = ck.at(name);
    DevTensor d = alloc(t.nbytes, t.dtype, t.shape);
    items.push_back({&t, d.ptr});
    return d;
  }
  bool keep_fp8 = false;  // fp8 linears stay fp8 (Hy4LoadOptions::dense_fp8)
  // A linear: bf16 as stored; fp8 (+ weight_scale_inv, bf16 128x128) kept, or dequantized to bf16 after the pass.
  Hy4Linear linear(const std::string& prefix) {
    Hy4Linear l;
    const TensorInfo& t = ck.at(prefix + ".weight");
    if (t.shape.size() != 2) throw std::runtime_error(prefix + ": not a matrix");
    l.out = int(t.shape[0]);
    l.in = int(t.shape[1]);
    if (t.dtype == DType::BF16) {
      l.w = upload(prefix + ".weight");
    } else if (t.dtype == DType::F8_E4M3 && keep_fp8) {
      l.w8 = upload(prefix + ".weight");
      l.s8 = upload(prefix + ".weight_scale_inv");
      if (l.s8.dtype != DType::BF16) throw std::runtime_error(prefix + ": fp8 without bf16 block scales");
      const void* p[2] = {l.w8.ptr, l.s8.ptr};
      l.tab = rt::DeviceBuffer(dev, sizeof p);
      rt::DeviceGuard g(dev);
      UH_CHECK(hipMemcpy(l.tab.data(), p, sizeof p, hipMemcpyHostToDevice));
    } else if (t.dtype == DType::F8_E4M3) {
      deq.push_back({upload(prefix + ".weight"), upload(prefix + ".weight_scale_inv"), nullptr});
      if (deq.back().s.dtype != DType::BF16) throw std::runtime_error(prefix + ": fp8 without bf16 block scales");
      l.w = alloc(size_t(l.out) * l.in * 2, DType::BF16, t.shape);
    } else {
      throw std::runtime_error(prefix + ": unsupported dtype " + dtype_name(t.dtype));
    }
    return l;
  }
  void bf16(const std::string& name, DevTensor& d) {
    d = upload(name);
    if (d.dtype != DType::BF16) throw std::runtime_error(name + ": expected bf16");
  }
  void f32(const std::string& name, DevTensor& d) {
    d = upload(name);
    if (d.dtype != DType::F32) throw std::runtime_error(name + ": expected f32");
  }
  // Rows [r0, r0 + n) of `name` viewed as [rows][row_bytes] into dst.
  void slice(const std::string& name, size_t row_bytes, size_t r0, size_t n, void* dst) {
    const TensorInfo& t = ck.at(name);
    TensorInfo s = t;
    s.offset = t.offset + r0 * row_bytes;
    s.nbytes = n * row_bytes;
    slices.push_back(std::move(s));
    items.push_back({&slices.back(), dst});
  }
  void run(hipStream_t st, std::vector<Hy4Linear*> deq_dst) {
    rt::DeviceGuard g(dev);
    if (!items.empty()) {
      std::unique_ptr<Uploader> own;
      Uploader* u = up;
      if (!u) {
        own = std::make_unique<Uploader>(ck);
        u = own.get();
      }
      u->upload_many(std::move(items), st);
      u->sync();
      items.clear();
    }
    if (deq_dst.size() != deq.size()) throw std::logic_error("hy4 load: dequant list out of step");
    for (size_t i = 0; i < deq.size(); ++i) {
      Hy4Linear& l = *deq_dst[i];
      k::hy4_dequant_fp8(deq[i].q.p<uint8_t>(), deq[i].s.p<uint16_t>(), l.w.p<uint16_t>(), l.out, l.in, st);
    }
    UH_CHECK(hipStreamSynchronize(st));
    deq.clear();
    slices.clear();
  }
};

// kv_b [H (dn + dv)][r] -> uk_t [H][r][dn] and uv [H][dv][r].
__global__ void split_kv_b_kernel(const uint16_t* kvb, uint16_t* uk_t, uint16_t* uv, int H, int dn, int dv, int r) {
  const size_t n = size_t(H) * (dn + dv) * r;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int col = int(i % r), row = int(i / r), h = row / (dn + dv), j = row % (dn + dv);
    if (j < dn) uk_t[(size_t(h) * r + col) * dn + j] = kvb[i];
    else uv[(size_t(h) * dv + j - dn) * r + col] = kvb[i];
  }
}

size_t align256(size_t b) { return (b + 255) / 256 * 256; }

}  // namespace

size_t hy4_expert_bytes(const Hy4Config& c, Hy4ExpertFormat f) {
  const size_t I = c.moe_inter, Lt = c.latent;
  if (f == Hy4ExpertFormat::Mxfp4) return 3 * (I * Lt / 2 + I * Lt / 32);
  if (f == Hy4ExpertFormat::Nvfp4) return 3 * (I * Lt / 2 + I * Lt / 16);
  return 2 * I * Lt + (2 * I / 128) * (Lt / 128) * 2 + Lt * I + (Lt / 128) * (I / 128) * 2;
}

int hy4_expert_tile_rows(bool mx, int tokens) {
  if (!mx) return tokens <= kHy4GemvTokens ? k::kHy4GemvRows : k::kHy4TileRows;
  return k::mx_use_gemv(tokens) ? k::kMxGemvTileRows : k::kExl3TileRows;
}

int hy4_max_tiles(const Hy4Config& c, int max_tokens) {
  const size_t P = size_t(max_tokens) * c.n_active;
  const size_t gemv = size_t(std::min(max_tokens, 64)) * c.n_active / k::kMxGemvTileRows;
  return int(std::max(P / k::kHy4TileRows, gemv) + c.n_experts + 1);
}

Hy4LayerWeights load_hy4_layer(const Checkpoint& ck, const Hy4Config& c, int layer, int dev, hipStream_t st,
                               const Hy4LoadOptions& opt) {
  const std::string p = kPrefix + "layers." + std::to_string(layer) + ".";
  LoadCtx lc{ck, dev, opt.up};
  Hy4LayerWeights w;
  w.layer = layer;
  w.sparse = c.is_sparse(layer);
  w.dense = c.is_dense(layer);
  w.engram = c.engram[size_t(layer)];
  const std::string m = p + "mlp.";
  std::vector<Hy4Linear*> deq;
  DevTensor kv_b;
  lc.keep_fp8 = opt.dense_fp8;
  auto lin = [&](Hy4Linear& l, const std::string& name) {
    const bool fp8 = ck.at(name + ".weight").dtype == DType::F8_E4M3 && !opt.dense_fp8;
    l = lc.linear(name);
    if (fp8) deq.push_back(&l);
  };
  if (opt.load_dense) {
    lc.f32(p + "hc_attn_layer.hc_pre.hc_fn", w.hc_attn_fn);
    lc.f32(p + "hc_attn_layer.hc_pre.hc_base", w.hc_attn_base);
    lc.f32(p + "hc_attn_layer.hc_pre.hc_scale", w.hc_attn_scale);
    lc.f32(p + "hc_mlp_layer.hc_pre.hc_fn", w.hc_ffn_fn);
    lc.f32(p + "hc_mlp_layer.hc_pre.hc_base", w.hc_ffn_base);
    lc.f32(p + "hc_mlp_layer.hc_pre.hc_scale", w.hc_ffn_scale);
    lc.bf16(p + "input_layernorm.weight", w.input_norm);
    lc.bf16(p + "post_attention_layernorm.weight", w.post_norm);
    const std::string a = p + "self_attn.";
    lin(w.q_a, a + "q_a_proj");
    lin(w.q_b, a + "q_b_proj");
    lin(w.kv_a, a + "kv_a_proj_with_mqa");
    lin(w.gate, a + "linear_gate");
    lin(w.o, a + "o_proj");
    lc.bf16(a + "q_a_layernorm.weight", w.q_a_norm);
    lc.bf16(a + "kv_a_layernorm.weight", w.kv_a_norm);
    lc.bf16(a + "post_proj_norm.weight", w.post_proj_norm);
    lc.f32(a + "learnable_sink_param", w.sinks);
    lc.bf16(a + "kv_b_proj.weight", kv_b);
    if (w.sparse) {
      const std::string ip = a + "compressed_indexer.";
      lin(w.idx_q, ip + "query_projection");
      lin(w.idx_k, ip + "token_key_projection");
      lin(w.idx_w, ip + "head_weight_projection");
      lc.bf16(ip + "token_key_norm.weight", w.idx_tok_norm);
      lc.f32(ip + "key_value_projection.weight", w.idx_wv);
      lc.f32(ip + "key_gate_projection.weight", w.idx_wg);
      lc.f32(ip + "intra_group_ape", w.idx_ape);
      lc.f32(ip + "key_norm.weight", w.idx_knorm);
    }
    if (w.dense) {
      lin(w.gate_p, m + "gate_proj");
      lin(w.up_p, m + "up_proj");
      lin(w.down_p, m + "down_proj");
      lc.bf16(m + "post_fc2_norm.weight", w.post_mlp_norm);
    } else {
      lin(w.router, m + "gate");
      lc.f32(m + "gate.e_score_correction_bias", w.router_bias);
      lin(w.pre_expert, m + "routed_expert_down_proj");
      lin(w.post_expert, m + "routed_expert_up_proj");
      lin(w.sh_gate, m + "shared_experts.gate_proj");
      lin(w.sh_up, m + "shared_experts.up_proj");
      lin(w.sh_down, m + "shared_experts.down_proj");
      lc.bf16(m + "combine_output_norm.weight", w.combine_norm);
    }
    if (w.engram >= 0) {
      const std::string e = p + "engram_layer.";
      const size_t D = size_t(c.dim), Ein = 1536;
      w.eng_kv.out = int(5 * D);
      w.eng_kv.in = int(Ein);
      w.eng_kv.w = lc.alloc(5 * D * Ein * 2, DType::BF16, {int64_t(5 * D), int64_t(Ein)});
      w.eng_bias = lc.alloc(5 * D * 2, DType::BF16, {5, int64_t(D)});
      w.eng_nk = lc.alloc(4 * D * 2, DType::BF16, {4, int64_t(D)});
      w.eng_nq = lc.alloc(4 * D * 2, DType::BF16, {4, int64_t(D)});
      w.eng_hcs = lc.alloc(4 * D * 2, DType::BF16, {4, int64_t(D)});
      w.eng_lookup_norm = lc.alloc(16 * 96 * 2, DType::BF16, {16, 96});
      auto part = [&](const std::string& name, DevTensor& dst, size_t at, size_t bytes) {
        const TensorInfo& t = ck.at(name);
        if (t.dtype != DType::BF16 || t.nbytes != bytes) throw std::runtime_error(name + ": unexpected tensor");
        lc.items.push_back({&t, static_cast<uint8_t*>(dst.ptr) + at});
      };
      for (int i = 0; i < 5; ++i) {
        const std::string pr = i < 4 ? e + "key_projs." + std::to_string(i) : e + "value_proj";
        part(pr + ".weight", w.eng_kv.w, i * D * Ein * 2, D * Ein * 2);
        part(pr + ".bias", w.eng_bias, i * D * 2, D * 2);
      }
      for (int i = 0; i < 4; ++i) {
        part(e + "norm_key." + std::to_string(i) + ".weight", w.eng_nk, i * D * 2, D * 2);
        part(e + "norm_query." + std::to_string(i) + ".weight", w.eng_nq, i * D * 2, D * 2);
        part(e + "unified_per_hc_scales." + std::to_string(i), w.eng_hcs, i * D * 2, D * 2);
      }
      for (int h = 0; h < 16; ++h) part(e + "lookup_per_head_norms." + std::to_string(h) + ".weight",
                                        w.eng_lookup_norm, size_t(h) * 96 * 2, 96 * 2);
      lc.bf16(e + "unified_pre_gamma", w.eng_pre_gamma);
    }
  }
  if (!w.dense && opt.load_experts && opt.mxfp4 && !opt.experts_gguf) {
    // MXFP4: each batch of experts lands in e4m3 in a staging buffer, then is converted into the final layout and
    // repacked tile-major (mx_repack) one matrix at a time.
    const size_t E = size_t(c.n_experts), I = c.moe_inter, Lt = c.latent;
    const size_t gu_b = 2 * I * Lt, gus_b = (2 * I / 128) * (Lt / 128) * 2, dn_b = Lt * I,
                 dns_b = (Lt / 128) * (I / 128) * 2;
    const size_t mw = I * Lt / 2, ms = I * Lt / 32;  // one matrix's codes / scales (all three are 1536 x 1536)
    std::vector<int> here;
    for (size_t e = 0; e < E; ++e)
      if (!opt.expert_mask || (*opt.expert_mask)[e]) here.push_back(int(e));
    const size_t n = here.size();
    std::vector<const void*> tab(6 * E, nullptr);
    std::vector<uint8_t> res(E, 0);
    if (n) {
      size_t off[6], at = 0;
      for (int f = 0; f < 6; ++f) {
        off[f] = at;
        at += align256(n * (f % 2 == 0 ? mw : ms));
      }
      w.experts = rt::DeviceBuffer(dev, at);
      uint8_t* base = w.experts.as<uint8_t>();
      constexpr size_t kBatch = 32;
      rt::DeviceBuffer stage(dev, kBatch * (gu_b + gus_b + dn_b + dns_b)), tmp(dev, mw + ms);
      rt::DeviceGuard g(dev);
      for (size_t b0 = 0; b0 < n; b0 += kBatch) {
        const size_t nb = std::min(kBatch, n - b0);
        LoadCtx sl{ck, dev, opt.up};
        uint8_t* sb = stage.as<uint8_t>();
        const auto at_of = [&](size_t i, int part) {
          const size_t sizes[4] = {gu_b, gus_b, dn_b, dns_b};
          size_t o = 0;
          for (int q = 0; q < part; ++q) o += kBatch * sizes[q];
          return sb + o + i * sizes[part];
        };
        for (size_t i = 0; i < nb; ++i) {
          const size_t e = size_t(here[b0 + i]);
          sl.slice(m + "experts.gate_up_proj", gu_b, e, 1, at_of(i, 0));
          sl.slice(m + "experts.gate_up_proj_scale_inv", gus_b, e, 1, at_of(i, 1));
          sl.slice(m + "experts.down_proj", dn_b, e, 1, at_of(i, 2));
          sl.slice(m + "experts.down_proj_scale_inv", dns_b, e, 1, at_of(i, 3));
        }
        sl.run(st, {});
        for (size_t i = 0; i < nb; ++i) {
          const size_t idx = b0 + i, e = size_t(here[idx]);
          uint8_t* dst[6];
          for (int f = 0; f < 6; ++f) dst[f] = base + off[f] + idx * (f % 2 == 0 ? mw : ms);
          const auto* gs = reinterpret_cast<const uint16_t*>(at_of(i, 1));
          const auto* ds = reinterpret_cast<const uint16_t*>(at_of(i, 3));
          k::hy4_fp8_to_mxfp4(at_of(i, 0), gs, int(Lt / 128), 0, int(I), int(Lt), dst[0], dst[1], st);   // gate
          k::hy4_fp8_to_mxfp4(at_of(i, 0), gs, int(Lt / 128), int(I), int(I), int(Lt), dst[2], dst[3], st);  // up
          k::hy4_fp8_to_mxfp4(at_of(i, 2), ds, int(I / 128), 0, int(Lt), int(I), dst[4], dst[5], st);  // down
          k::mx_repack(dst[0], dst[1], int(I), int(Lt), tmp.as<uint8_t>(), st);
          k::mx_repack(dst[2], dst[3], int(I), int(Lt), tmp.as<uint8_t>(), st);
          k::mx_repack(dst[4], dst[5], int(Lt), int(I), tmp.as<uint8_t>(), st);
          for (int f = 0; f < 6; ++f) tab[size_t(f) * E + e] = dst[f];
          res[e] = 1;
        }
        UH_CHECK(hipStreamSynchronize(st));
      }
    }
    w.mx = true;
    w.n_resident = int(n);
    w.tab = rt::DeviceBuffer(dev, tab.size() * sizeof(void*));
    w.resident = rt::DeviceBuffer(dev, E);
    rt::DeviceGuard g(dev);
    UH_CHECK(hipMemcpy(w.tab.data(), tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(w.resident.data(), res.data(), E, hipMemcpyHostToDevice));
  }
  if (!w.dense && opt.load_experts && opt.experts_gguf) {
    // NVFP4 from the GGUF: each batch of experts' blocks into a staging buffer, then split into codes and scales.
    const Checkpoint& gk = *opt.experts_gguf;
    const size_t E = size_t(c.n_experts), I = c.moe_inter, Lt = c.latent;
    const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    const int rows_of[3] = {int(I), int(I), int(Lt)}, k_of[3] = {int(Lt), int(Lt), int(I)};
    size_t blk_bytes[3];
    for (int m3 = 0; m3 < 3; ++m3) {
      const TensorInfo& t = gk.at("blk." + std::to_string(layer) + "." + names[m3] + ".weight");
      if (t.gguf_type != 40 || t.shape.size() != 3 || t.shape[0] != int64_t(E) || t.shape[1] != rows_of[m3] ||
          t.shape[2] != k_of[m3])
        throw std::runtime_error(t.name + ": expected NVFP4 [" + std::to_string(E) + ", N, K]");
      blk_bytes[m3] = size_t(rows_of[m3]) * (k_of[m3] / 64) * 36;
    }
    std::vector<int> here;
    for (size_t e = 0; e < E; ++e)
      if (!opt.expert_mask || (*opt.expert_mask)[e]) here.push_back(int(e));
    const size_t n = here.size();
    std::vector<const void*> tab(6 * E, nullptr);
    std::vector<uint8_t> res(E, 0);
    if (n) {
      const size_t cw = I * Lt / 2, cs = I * Lt / 16;  // one matrix's codes / scales (all three are 1536 x 1536)
      size_t off[6], at = 0;
      for (int f = 0; f < 6; ++f) {
        off[f] = at;
        at += align256(n * (f % 2 == 0 ? cw : cs));
      }
      w.experts = rt::DeviceBuffer(dev, at);
      uint8_t* base = w.experts.as<uint8_t>();
      constexpr size_t kBatch = 32;
      rt::DeviceBuffer stage(dev, kBatch * (blk_bytes[0] + blk_bytes[1] + blk_bytes[2]));
      std::unique_ptr<Uploader> own;
      Uploader* gup = opt.experts_up;
      if (!gup) {
        own = std::make_unique<Uploader>(gk);
        gup = own.get();
      }
      rt::DeviceGuard g(dev);
      for (size_t b0 = 0; b0 < n; b0 += kBatch) {
        const size_t nb = std::min(kBatch, n - b0);
        LoadCtx sl{gk, dev, gup};
        uint8_t* sb = stage.as<uint8_t>();
        const auto at_of = [&](size_t i, int m3) {
          size_t o = 0;
          for (int q = 0; q < m3; ++q) o += kBatch * blk_bytes[q];
          return sb + o + i * blk_bytes[m3];
        };
        for (size_t i = 0; i < nb; ++i)
          for (int m3 = 0; m3 < 3; ++m3)
            sl.slice("blk." + std::to_string(layer) + "." + names[m3] + ".weight", blk_bytes[m3],
                     size_t(here[b0 + i]), 1, at_of(i, m3));
        sl.run(st, {});
        for (size_t i = 0; i < nb; ++i) {
          const size_t idx = b0 + i, e = size_t(here[idx]);
          for (int m3 = 0; m3 < 3; ++m3) {
            uint8_t* codes = base + off[2 * m3] + idx * cw;
            uint8_t* scales = base + off[2 * m3 + 1] + idx * cs;
            k::hy4_nvfp4_split(at_of(i, m3), rows_of[m3], k_of[m3], codes, scales, st);
            tab[size_t(2 * m3) * E + e] = codes;
            tab[size_t(2 * m3 + 1) * E + e] = scales;
          }
          res[e] = 1;
        }
        UH_CHECK(hipStreamSynchronize(st));
      }
    }
    w.nv = true;
    w.n_resident = int(n);
    w.tab = rt::DeviceBuffer(dev, tab.size() * sizeof(void*));
    w.resident = rt::DeviceBuffer(dev, E);
    rt::DeviceGuard g(dev);
    UH_CHECK(hipMemcpy(w.tab.data(), tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(w.resident.data(), res.data(), E, hipMemcpyHostToDevice));
  }
  if (!w.dense && opt.load_experts && !opt.mxfp4 && !opt.experts_gguf) {
    const size_t E = size_t(c.n_experts), I = c.moe_inter, Lt = c.latent;
    const size_t gu_b = 2 * I * Lt, gus_b = (2 * I / 128) * (Lt / 128) * 2, dn_b = Lt * I,
                 dns_b = (Lt / 128) * (I / 128) * 2;
    std::vector<int> here;
    for (size_t e = 0; e < E; ++e)
      if (!opt.expert_mask || (*opt.expert_mask)[e]) here.push_back(int(e));
    const size_t n = here.size();
    const size_t o_gus = align256(n * gu_b), o_dn = o_gus + align256(n * gus_b), o_dns = o_dn + align256(n * dn_b);
    const size_t total = o_dns + align256(n * dns_b);
    if (n) w.experts = rt::DeviceBuffer(dev, total);
    uint8_t* base = w.experts.as<uint8_t>();
    std::vector<const void*> tab(4 * E, nullptr);
    std::vector<uint8_t> res(E, 0);
    for (size_t i = 0; i < n; ++i) {
      const size_t e = size_t(here[i]);
      uint8_t* gw = base + i * gu_b;
      uint8_t* gs = base + o_gus + i * gus_b;
      uint8_t* dw = base + o_dn + i * dn_b;
      uint8_t* ds = base + o_dns + i * dns_b;
      lc.slice(m + "experts.gate_up_proj", gu_b, e, 1, gw);
      lc.slice(m + "experts.gate_up_proj_scale_inv", gus_b, e, 1, gs);
      lc.slice(m + "experts.down_proj", dn_b, e, 1, dw);
      lc.slice(m + "experts.down_proj_scale_inv", dns_b, e, 1, ds);
      tab[e] = gw;
      tab[E + e] = gs;
      tab[2 * E + e] = dw;
      tab[3 * E + e] = ds;
      res[e] = 1;
    }
    w.n_resident = int(n);
    w.tab = rt::DeviceBuffer(dev, tab.size() * sizeof(void*));
    w.resident = rt::DeviceBuffer(dev, E);
    rt::DeviceGuard g(dev);
    UH_CHECK(hipMemcpy(w.tab.data(), tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice));
    UH_CHECK(hipMemcpy(w.resident.data(), res.data(), E, hipMemcpyHostToDevice));
  }
  lc.run(st, deq);
  if (kv_b) {  // the absorbed attention's per-head weights; the original is dropped
    const int H = c.heads(layer), dn = c.nope, dv = c.v_dim, r = c.kv_lora;
    if (kv_b.shape.size() != 2 || kv_b.shape[0] != int64_t(H) * (dn + dv) || kv_b.shape[1] != r)
      throw std::runtime_error(p + "kv_b_proj: unexpected shape");
    w.w_ukT = lc.alloc(size_t(H) * r * dn * 2, DType::BF16, {H, r, dn});
    w.w_uv = lc.alloc(size_t(H) * dv * r * 2, DType::BF16, {H, dv, r});
    rt::DeviceGuard g(dev);
    split_kv_b_kernel<<<4096, 256, 0, st>>>(kv_b.p<uint16_t>(), w.w_ukT.p<uint16_t>(), w.w_uv.p<uint16_t>(), H, dn, dv,
                                            r);
    UH_CHECK_LAUNCH();
    UH_CHECK(hipStreamSynchronize(st));
  }
  return w;
}

Hy4HeadWeights load_hy4_head(const Checkpoint& ck, const Hy4Config& c, int dev, hipStream_t st, bool embed,
                             bool head) {
  (void)c;
  LoadCtx lc{ck, dev};
  Hy4HeadWeights h;
  std::vector<Hy4Linear*> deq;
  if (embed) lc.bf16(kPrefix + "embed_tokens.weight", h.embed);
  if (head) {
    lc.f32(kPrefix + "hc_head.hc_head_fn", h.hc_fn);
    lc.f32(kPrefix + "hc_head.hc_head_base", h.hc_base);
    lc.f32(kPrefix + "hc_head.hc_head_scale", h.hc_scale);
    lc.bf16(kPrefix + "norm.weight", h.norm);
    h.head = lc.linear("lm_head");
    if (h.head.w.dtype != DType::BF16) throw std::runtime_error("lm_head: expected bf16");
  }
  lc.run(st, deq);
  return h;
}

bool hy4_has_dfly(const Checkpoint& ck) { return ck.find("model.dfly_layers.0.self_attn.q_a_proj.weight") != nullptr; }

size_t hy4_dfly_bytes(const Checkpoint& ck) {
  size_t b = ck.at("model.embed_tokens.weight").nbytes;
  for (const auto& t : ck.tensors())
    if (t.name.rfind("model.dfly_layers.", 0) == 0 && t.name.find("confidence_head") == std::string::npos)
      b += t.nbytes;  // fp8 kept; kv_b split per head (the original dropped)
  return b;
}

Hy4DflyWeights load_hy4_dfly(const Checkpoint& ck, const Hy4Config& c, int dev, hipStream_t st) {
  const auto& cfg = ck.config();
  const auto ints = [&](const char* key) {
    std::vector<int> v;
    const auto* a = cfg.find(key);
    if (!a) throw std::runtime_error(std::string("config: missing ") + key);
    for (const auto& e : a->items()) v.push_back(int(e.as_double()));
    return v;
  };
  Hy4DflyWeights w;
  w.target = ints("dfly_target_kv_layers");
  if (ints("dfly_target_index_layer") != w.target) throw std::runtime_error("dfly: KV and index layers differ");
  const int n = int(cfg.member_double("dfly_num_hidden_layers"));
  w.mask_id = int(cfg.member_double("dfly_mask_token_id"));
  if (n != int(w.target.size()) || n <= 0) throw std::runtime_error("dfly: layer count");
  for (int t : w.target)
    if (t < 0 || t >= c.n_layers || !c.is_sparse(t)) throw std::runtime_error("dfly: target layers must be sparse");
  if (const auto* lw = cfg.find("dfly_context_local_window"); lw && !lw->is_null() && lw->as_double() != 0)
    throw std::runtime_error("dfly: a local window is not supported");
  LoadCtx lc{ck, dev};
  lc.keep_fp8 = true;
  std::vector<DevTensor> kvb(static_cast<size_t>(n)), sinks16(static_cast<size_t>(n));
  w.layers.resize(size_t(n));
  for (int i = 0; i < n; ++i) {
    Hy4DflyLayer& L = w.layers[size_t(i)];
    const std::string p = "model.dfly_layers." + std::to_string(i) + ".";
    L.q_a = lc.linear(p + "self_attn.q_a_proj");
    L.q_b = lc.linear(p + "self_attn.q_b_proj");
    L.kv_a = lc.linear(p + "self_attn.kv_a_proj_with_mqa");
    L.gate = lc.linear(p + "self_attn.linear_gate");
    L.o = lc.linear(p + "self_attn.o_proj");
    L.mg = lc.linear(p + "mlp.gate_proj");
    L.mu = lc.linear(p + "mlp.up_proj");
    L.md = lc.linear(p + "mlp.down_proj");
    lc.bf16(p + "input_layernorm.weight", L.in_norm);
    lc.bf16(p + "post_attention_layernorm.weight", L.post_norm);
    lc.bf16(p + "self_attn.q_a_layernorm.weight", L.q_a_norm);
    lc.bf16(p + "self_attn.kv_a_layernorm.weight", L.kv_a_norm);
    lc.bf16(p + "self_attn.post_proj_norm.weight", L.post_proj_norm);
    lc.bf16(p + "self_attn.learnable_sink_param", sinks16[size_t(i)]);
    lc.bf16(p + "self_attn.kv_b_proj.weight", kvb[size_t(i)]);
  }
  const std::string f = "model.dfly_layers.";
  lc.bf16(f + "mtp_fusion.enorm.weight", w.enorm);
  lc.bf16(f + "mtp_fusion.hnorm.weight", w.hnorm);
  lc.bf16(f + "mtp_fusion.gate", w.fgate);
  w.eh = lc.linear(f + "mtp_fusion.eh_proj");
  lc.bf16(f + "hidden_correction.hidden_norm.weight", w.c_hnorm);
  lc.bf16(f + "hidden_correction.embed_norm.weight", w.c_enorm);
  w.cg = lc.linear(f + "hidden_correction.gate_proj");
  w.cu = lc.linear(f + "hidden_correction.up_proj");
  w.cd = lc.linear(f + "hidden_correction.down_proj");
  lc.bf16(f + "final_norm.weight", w.final_norm);
  lc.bf16("model.embed_tokens.weight", w.embed);
  lc.run(st, {});
  rt::DeviceGuard g(dev);
  const int H = c.heads_sparse, dn = c.nope, dv = c.v_dim, r = c.kv_lora;
  for (int i = 0; i < n; ++i) {
    Hy4DflyLayer& L = w.layers[size_t(i)];
    if (L.q_b.out != H * (dn + c.rope) || kvb[size_t(i)].shape[0] != int64_t(H) * (dn + dv))
      throw std::runtime_error("dfly: unexpected attention shapes");
    L.sinks = lc.alloc(size_t(H) * 4, DType::F32, {H});
    k::cast_bf16_f32(sinks16[size_t(i)].p<uint16_t>(), L.sinks.p<float>(), size_t(H), st);
    L.w_ukT = lc.alloc(size_t(H) * r * dn * 2, DType::BF16, {H, r, dn});
    L.w_uv = lc.alloc(size_t(H) * dv * r * 2, DType::BF16, {H, dv, r});
    split_kv_b_kernel<<<4096, 256, 0, st>>>(kvb[size_t(i)].p<uint16_t>(), L.w_ukT.p<uint16_t>(), L.w_uv.p<uint16_t>(),
                                            H, dn, dv, r);
    UH_CHECK_LAUNCH();
  }
  UH_CHECK(hipStreamSynchronize(st));
  return w;
}

size_t hy4_dense_layer_bytes(const Checkpoint& ck, const Hy4Config& c, int layer, bool dense_fp8) {
  const std::string p = kPrefix + "layers." + std::to_string(layer) + ".";
  size_t b = 0;
  for (const auto& t : ck.tensors()) {
    if (t.name.compare(0, p.size(), p) != 0) continue;
    if (t.name.find(".mlp.experts.") != std::string::npos) continue;
    if (t.name.find("multi_head_embedding") != std::string::npos || t.name.find("lookup_table") != std::string::npos)
      continue;
    if (t.name.find("_scale_inv") != std::string::npos) {
      if (dense_fp8) b += t.nbytes;  // kept with their fp8 linears
      continue;
    }
    b += t.dtype == DType::F8_E4M3 && !dense_fp8 ? t.nbytes * 2 : t.nbytes;  // (kv_b: split, original dropped)
  }
  (void)c;
  return b;
}

// ---------------------------------------------------------------------------------------------------
// Engram store

namespace {

std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

const gufo::json::Value& req(const gufo::json::Value& o, const char* key) {
  const gufo::json::Value* v = o.find(key);
  if (!v) throw std::runtime_error(std::string("engram pack meta.json: missing ") + key);
  return *v;
}

}  // namespace

Hy4Engram::Hy4Engram(const std::string& dir, const Hy4Config& c) {
  const gufo::json::Value meta = gufo::json::parse(slurp(dir + "/meta.json"));
  pad_ = int(req(meta, "pad_id").as_double());
  if (int(req(meta, "head_dim").as_double()) != 96 || int(req(meta, "max_ngram").as_double()) != 3 ||
      int(req(meta, "heads_per_ngram").as_double()) != 8)
    throw std::runtime_error("engram pack: unexpected shape");
  mem_.resize(size_t(c.n_engram()));
  for (int l = 0; l < c.n_layers; ++l) {
    const int mi = c.engram[size_t(l)];
    if (mi < 0) continue;
    const auto* lj = req(meta, "layers").find(std::to_string(l));
    if (!lj) throw std::runtime_error("engram pack: no memory for layer " + std::to_string(l));
    Mem& m = mem_[size_t(mi)];
    m.layer = l;
    for (const auto& v : req(*lj, "multipliers").items()) m.mult.push_back(std::stoll(v.get_str()));
    for (const auto& v : req(*lj, "moduli").items()) m.moduli.push_back(int64_t(v.as_double()));
    for (const auto& v : req(*lj, "offsets").items()) m.offsets.push_back(int64_t(v.as_double()));
    if (m.mult.size() != 3 || m.moduli.size() != 16 || m.offsets.size() != 16)
      throw std::runtime_error("engram pack: unexpected hash parameters");
    const std::string lk = slurp(dir + "/layer_" + std::to_string(l) + ".lookup.i32");
    m.lookup.resize(lk.size() / 4);
    std::memcpy(m.lookup.data(), lk.data(), m.lookup.size() * 4);
    m.qfd = ::open((dir + "/layer_" + std::to_string(l) + ".q").c_str(), O_RDONLY | O_CLOEXEC);
    m.sfd = ::open((dir + "/layer_" + std::to_string(l) + ".s").c_str(), O_RDONLY | O_CLOEXEC);
    if (m.qfd < 0 || m.sfd < 0) throw std::runtime_error("engram pack: cannot open layer " + std::to_string(l));
    posix_fadvise(m.qfd, 0, 0, POSIX_FADV_RANDOM);
    posix_fadvise(m.sfd, 0, 0, POSIX_FADV_RANDOM);
  }
}

// The reader pool: persistent threads, one batch at a time. A batch's jobs are (memory, position, head) in memory-major
// order: each computes its row and reads its 96 + 2 bytes; the last job of a memory sets that memory's flag.
struct Hy4Engram::Pool {
  struct Batch {
    std::vector<int32_t> ngram;
    std::vector<uint8_t> mask;
    int T = 0, M = 0, total = 0;
    uint8_t* q[4] = {};
    uint16_t* s[4] = {};
    uint32_t* flags[4] = {};
    uint32_t ticket = 0;
    std::atomic<int> next{0}, all_left{0};
    std::atomic<int> left[4];
    std::mutex err_mu;
    std::exception_ptr err;
  };
  std::vector<std::thread> threads;
  std::mutex mu;
  std::condition_variable cv, done;
  uint64_t gen = 0;
  bool stop = false;
  std::shared_ptr<Batch> cur;
  std::exception_ptr pending;  // the last finished batch's error, for finish()

  static void set_flag(uint32_t* f, uint32_t v) {
    if (f) std::atomic_ref<uint32_t>(*f).store(v, std::memory_order_release);
  }
};

Hy4Engram::~Hy4Engram() {
  if (pool_) {
    {
      std::lock_guard lk(pool_->mu);
      pool_->stop = true;
    }
    pool_->cv.notify_all();
    for (auto& t : pool_->threads) t.join();
  }
  for (auto& m : mem_) {
    if (m.qfd >= 0) ::close(m.qfd);
    if (m.sfd >= 0) ::close(m.sfd);
  }
}

void Hy4Engram::fill_head(int mi, const int32_t* ngram, const uint8_t* mask, int t, int h, uint8_t* q,
                          uint16_t* s) const {
  const Mem& m = mem_[size_t(mi)];
  uint8_t* qh = q + (size_t(t) * 16 + h) * 96;
  uint16_t* sh = s + size_t(t) * 16 + h;
  if (!mask[t]) {
    std::memset(qh, 0, 96);
    *sh = 0;
    return;
  }
  const int64_t size = int64_t(m.lookup.size());
  const auto comp = [&](int32_t tok) -> int64_t {
    if (tok < 0 || tok >= size) return m.lookup[size_t(pad_)];
    return m.lookup[size_t(tok)];
  };
  int64_t mix = (comp(ngram[size_t(t) * 3]) * m.mult[0]) ^ (comp(ngram[size_t(t) * 3 + 1]) * m.mult[1]);
  if (h >= 8) mix ^= comp(ngram[size_t(t) * 3 + 2]) * m.mult[2];
  int64_t r = mix % m.moduli[size_t(h)];
  if (r < 0) r += m.moduli[size_t(h)];  // torch's % (remainder): the divisor's sign
  const int64_t row = r + m.offsets[size_t(h)];
  if (::pread(m.qfd, qh, 96, off_t(row) * 96) != 96 || ::pread(m.sfd, sh, 2, off_t(row) * 2) != 2) {
    std::memset(qh, 0, 96);
    *sh = 0;
    throw std::runtime_error("engram: short read");
  }
}

void Hy4Engram::fill(int mi, const int32_t* ngram, const uint8_t* mask, int t, uint8_t* q, uint16_t* s) const {
  for (int h = 0; h < 16; ++h) fill_head(mi, ngram, mask, t, h, q, s);
}

void Hy4Engram::lookup(int mi, const int32_t* ngram, const uint8_t* mask, int T, uint8_t* q, uint16_t* s) const {
  if (mi < 0 || mi >= memories()) throw std::runtime_error("engram: no such memory");
  for (int t = 0; t < T; ++t) fill(mi, ngram, mask, t, q, s);
}

void Hy4Engram::lookup_all(const int32_t* ngram, const uint8_t* mask, int T, uint8_t* const* q,
                           uint16_t* const* s) const {
  lookup_async(ngram, mask, T, q, s, nullptr, 0);
  finish();
}

void Hy4Engram::finish() const {
  if (!pool_) return;
  Pool& p = *pool_;
  std::unique_lock lk(p.mu);
  p.done.wait(lk, [&] { return !p.cur || p.cur->all_left.load(std::memory_order_acquire) == 0; });
  if (p.cur) {
    std::lock_guard el(p.cur->err_mu);
    if (p.cur->err && !p.pending) p.pending = p.cur->err;
  }
  if (p.pending) {
    const auto e = p.pending;
    p.pending = nullptr;
    std::rethrow_exception(e);
  }
}

void Hy4Engram::rethrow_pending() const {
  if (!pool_) return;
  std::lock_guard lk(pool_->mu);
  if (pool_->pending) {
    const auto e = pool_->pending;
    pool_->pending = nullptr;
    std::rethrow_exception(e);
  }
}

void Hy4Engram::lookup_async(const int32_t* ngram, const uint8_t* mask, int T, uint8_t* const* q,
                             uint16_t* const* s, uint32_t* const* flags, uint32_t ticket) const {
  auto* self = const_cast<Hy4Engram*>(this);
  const int M = memories();
  if (!pool_) {  // the readers: enough threads for a decode step's reads to be in flight together
    const char* e = std::getenv("UNHOLY_HY4_ENGRAM_THREADS");
    const int n = std::clamp(e ? std::atoi(e) : 64, 1, 512);
    self->pool_ = std::make_unique<Pool>();
    Pool* p = self->pool_.get();
    for (int i = 0; i < n; ++i)
      p->threads.emplace_back([this, p] {
        uint64_t seen = 0;
        for (;;) {
          std::shared_ptr<Pool::Batch> b;
          {
            std::unique_lock lk(p->mu);
            p->cv.wait(lk, [&] { return p->stop || p->gen != seen; });
            if (p->stop) return;
            seen = p->gen;
            b = p->cur;
          }
          for (int i; (i = b->next.fetch_add(1, std::memory_order_relaxed)) < b->total;) {
            const int per = b->T * 16, m = i / per, t = (i % per) / 16, h = i % 16;
            try {
              fill_head(m, b->ngram.data(), b->mask.data(), t, h, b->q[m], b->s[m]);
            } catch (...) {
              std::lock_guard el(b->err_mu);
              if (!b->err) b->err = std::current_exception();
            }
            if (b->left[m].fetch_sub(1, std::memory_order_acq_rel) == 1) Pool::set_flag(b->flags[m], b->ticket);
            if (b->all_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
              std::lock_guard lk(p->mu);
              p->done.notify_all();
            }
          }
        }
      });
  }
  Pool& p = *pool_;
  {  // the batch before must be done (its buffers and flags belong to the other flight)
    std::unique_lock lk(p.mu);
    p.done.wait(lk, [&] { return !p.cur || p.cur->all_left.load(std::memory_order_acquire) == 0; });
    if (p.cur) {
      std::lock_guard el(p.cur->err_mu);
      if (p.cur->err && !p.pending) p.pending = p.cur->err;
    }
  }
  auto b = std::make_shared<Pool::Batch>();
  b->ngram.assign(ngram, ngram + size_t(T) * 3);
  b->mask.assign(mask, mask + T);
  b->T = T;
  b->M = M;
  b->total = M * T * 16;
  b->ticket = ticket;
  for (int m = 0; m < M; ++m) {
    b->q[m] = q[m];
    b->s[m] = s[m];
    b->flags[m] = flags ? flags[m] : nullptr;
    b->left[m].store(T * 16, std::memory_order_relaxed);
  }
  b->all_left.store(b->total, std::memory_order_relaxed);
  if (b->total == 0) {
    for (int m = 0; m < M; ++m) Pool::set_flag(b->flags[m], ticket);
    return;
  }
  {
    std::lock_guard lk(p.mu);
    p.cur = std::move(b);
    ++p.gen;
  }
  p.cv.notify_all();
}

// ---------------------------------------------------------------------------------------------------
// State and scratch

size_t hy4_cache_bytes(const Hy4Config& c, bool sparse, int slots, int num_pages, bool dfly) {
  if (!sparse) return size_t(slots) * (k::kHy4Window + kHy4MaxVerify) * k::kHy4Row * 2;
  const size_t rows = size_t(num_pages) * k::kHy4PageRows;
  return rows * k::kHy4Row * 2 + rows / c.idx_pool * c.idx_dim * 2 +
         size_t(slots) * (2 * c.idx_pool + kHy4MaxVerify) * c.idx_dim * 2 +
         (dfly ? size_t(slots) * kHy4MaxVerify * (2048 + 1) * 4 : 0);
}

void hy4_restore_slot(Hy4Cache& k, int slot, int s0, int from, int to, hipStream_t st) {
  if (to <= from) return;
  if (k.sparse) {
    const size_t T = size_t(kHy4Pool) * kHy4IdxDim, V = size_t(kHy4MaxVerify) * kHy4IdxDim;
    k::hy4_tail_restore(k.tail.as<uint16_t>() + slot * T, k.tail_bak.as<uint16_t>() + slot * T,
                        k.vkeys.as<uint16_t>() + slot * V, s0, from, st);
  } else {
    k::hy4_ring_restore(k.ring.as<uint16_t>() + size_t(slot) * k::kHy4Window * k::kHy4Row,
                        k.ring_bak.as<uint16_t>() + size_t(slot) * kHy4MaxVerify * k::kHy4Row, s0, from, to, st);
  }
}

Hy4Cache make_hy4_cache(const Hy4Config& c, int layer, int slots, int num_pages, int dev, hipStream_t st,
                        bool dfly) {
  Hy4Cache k;
  k.layer = layer;
  k.slots = slots;
  k.num_pages = num_pages;
  k.sparse = c.is_sparse(layer);
  rt::DeviceGuard g(dev);
  auto zeroed = [&](size_t bytes) {
    rt::DeviceBuffer b(dev, bytes);
    UH_CHECK(hipMemsetAsync(b.data(), 0, bytes, st));
    return b;
  };
  if (k.sparse) {
    const size_t rows = size_t(num_pages) * k::kHy4PageRows;
    k.rows = zeroed(rows * k::kHy4Row * 2);
    k.pooled = zeroed(rows / c.idx_pool * c.idx_dim * 2);
    k.tail = zeroed(size_t(slots) * c.idx_pool * c.idx_dim * 2);
    k.tail_bak = zeroed(size_t(slots) * c.idx_pool * c.idx_dim * 2);
    k.vkeys = zeroed(size_t(slots) * kHy4MaxVerify * c.idx_dim * 2);
    if (dfly) {
      k.dsel = zeroed(size_t(slots) * kHy4MaxVerify * 2048 * 4);
      k.dcount = zeroed(size_t(slots) * kHy4MaxVerify * 4);
    }
  } else {
    k.ring = zeroed(size_t(slots) * k::kHy4Window * k::kHy4Row * 2);
    k.ring_bak = zeroed(size_t(slots) * kHy4MaxVerify * k::kHy4Row * 2);
  }
  UH_CHECK(hipStreamSynchronize(st));
  return k;
}

namespace {

std::vector<size_t> scratch_sizes(const Hy4Config& c, int max_tokens, int max_seq) {
  const size_t T = size_t(max_tokens), D = c.dim, H = std::max(c.heads_sparse, c.heads_window);
  const size_t P = T * c.n_active, E = c.n_experts, Lt = c.latent, I = c.moe_inter;
  const size_t max_tiles = size_t(hy4_max_tiles(c, max_tokens));
  const size_t npool = size_t(max_seq) / c.idx_pool + 1, kg = c.idx_groups();
  std::vector<size_t> sizes = {
      T * D * 2,                          // h
      T * D * 2,                          // hn
      T * c.hc * 4,                       // post
      T * D * 2,                          // out
      T * c.q_lora * 2,                   // q_resid
      T * H * 256 * 2,                    // q
      T * H * 192 * 2,                    // q nope
      T * H * 512 * 2,                    // absorbed q
      T * H * k::kHy4Row * 2,             // packed q
      T * k::kHy4Row * 2,                 // key rows
      T * H * 512 * 2,                    // latent out
      T * H * 256 * 2,                    // attention out
      T * H * 256 * 2,                    // gate
      T * c.idx_dim * 2,                  // token keys
      T * c.idx_heads * c.idx_dim * 2,    // indexer q
      T * c.idx_heads * 2,                // head weights
      size_t(kHy4IdxBlock) * npool * 4,   // index scores
      T * kg * 4,                         // recalled groups
      T * size_t(c.idx_width()) * 4,      // selection
      k::glm_topk_cand_bytes(kHy4IdxBlock, int(std::max(npool, size_t(8) * k::kGlmTopkSeg)), int(kg)),
      T * E * 4,                          // router logits
      P * 4, P * 4,                       // sel, routing weights
      T * Lt * 2,                         // latent
      P * 4, P * 4, P * 4, P * 4,         // eidx, ewt, row expert, pidx
      2 * T * 4,                          // tok2u, u2t
      P * Lt * 2,                         // expert inputs
      P * 2 * I * 2,                      // gate / up
      P * I * 2,                          // swiglu
      P * Lt * 2,                         // expert outputs
      T * Lt * 2,                         // routed sum
      T * D * 2,                          // routed out
      T * D * 2,                          // shared out
      T * I * 2, T * I * 2, T * I * 2,    // shared gate, up, h
      E * 4, E * 4, E * 4,                // MoE count, off, fill
      2 * sizeof(k::MoePlan),
      (max_tiles + E + 1) * sizeof(k::Exl3Tile),
      (max_tiles + E + 1) * sizeof(k::Exl3Tile),
      size_t(c.n_layers) * E * 4,         // routing histogram
      T * c.inter * 2, T * c.inter * 2, T * c.inter * 2,  // dense MLP
      T * 1536 * 2,                       // Engram rows
      T * 5 * D * 4,                      // Engram projections (fp32)
      (T / k::kHy4GemvRows + T / k::kHy4TileRows + 4) * sizeof(k::Exl3Tile),  // fp8 linears' tiles (8-row, 64-row)
      k::hy4_attn_split_bytes(std::max(c.heads_sparse, c.heads_window)),       // split attention partials
  };
  if (sizes.size() != size_t(kHy4ScratchBufs)) throw std::logic_error("hy4 scratch: buffer list out of date");
  return sizes;
}

}  // namespace

size_t hy4_scratch_bytes(const Hy4Config& c, int max_tokens, int max_seq) {
  size_t b = 0;
  for (size_t s : scratch_sizes(c, max_tokens, max_seq)) b += align256(std::max<size_t>(s, 16));
  return b;
}

Hy4Scratch make_hy4_scratch(const Hy4Config& c, int dev, int max_tokens, int max_seq) {
  Hy4Scratch s;
  s.max_tokens = max_tokens;
  s.max_seq = max_seq;
  rt::DeviceGuard g(dev);
  for (size_t b : scratch_sizes(c, max_tokens, max_seq)) {
    s.bufs.emplace_back(dev, std::max<size_t>(b, 16));
    UH_CHECK(hipMemset(s.bufs.back().data(), 0, s.bufs.back().size()));  // MoE counts start at 0
  }
  return s;
}

}  // namespace unholy::model
