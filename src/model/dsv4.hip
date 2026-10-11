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
};

DevTensor upload(LoadCtx& lc, const std::string& name) {
  const TensorInfo& t = lc.ck.at(name);
  DevTensor d;
  d.dtype = t.dtype;
  d.shape = t.shape;
  d.buf = rt::DeviceBuffer(lc.dev, t.nbytes);
  lc.items.push_back({&t, d.buf.data()});
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
  d.buf = rt::DeviceBuffer(lc.dev, host.size() * 4);
  rt::DeviceGuard g(lc.dev);
  UH_CHECK(hipMemcpy(d.buf.data(), host.data(), host.size() * 4, hipMemcpyHostToDevice));
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
                        Uploader* up, const Checkpoint* exl3, Uploader* exl3_up) {
  const std::string p = "layers." + std::to_string(layer) + ".";
  LoadCtx lc{ck, dev, {}};
  LayerWeights w;
  w.layer = layer;
  w.ratio = c.ratio(layer);
  w.hash = layer < c.n_hash_layers;
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
  if (w.ratio == 4) {
    w.has_indexer = true;
    w.idx_wq_b = fp8(lc, p + "attn.indexer.wq_b");
    w.idx_wproj = upload(lc, p + "attn.indexer.weights_proj.weight");
    w.idx_comp = compressor(lc, p + "attn.indexer.compressor", 4, c.idx_dim, true);
  }
  w.gate_w = upload(lc, p + "ffn.gate.weight");
  w.gate_bias = upload(lc, p + "ffn.gate.bias");
  if (w.hash) w.tid2eid = upload_i64_as_i32(lc, p + "ffn.gate.tid2eid");
  LoadCtx lx{exl3 ? *exl3 : ck, dev, {}};
  w.exl3_experts = exl3 != nullptr;
  for (int e = 0; e < c.n_experts; ++e) {
    const std::string ep = p + "ffn.experts." + std::to_string(e) + ".";
    if (exl3) {
      w.x1.push_back(exl3_linear(lx, ep + "w1"));
      w.x2.push_back(exl3_linear(lx, ep + "w2"));
      w.x3.push_back(exl3_linear(lx, ep + "w3"));
    } else {
      w.w1.push_back(fp4(lc, ep + "w1"));
      w.w2.push_back(fp4(lc, ep + "w2"));
      w.w3.push_back(fp4(lc, ep + "w3"));
    }
  }
  w.sw1 = fp8(lc, p + "ffn.shared_experts.w1");
  w.sw2 = fp8(lc, p + "ffn.shared_experts.w2");
  w.sw3 = fp8(lc, p + "ffn.shared_experts.w3");

  rt::DeviceGuard g(dev);
  std::unique_ptr<Uploader> own;
  if (!up) {
    own = std::make_unique<Uploader>(ck);
    up = own.get();
  }
  up->upload_many(std::move(lc.items), st);
  up->sync();
  if (exl3) {
    std::unique_ptr<Uploader> own_x;
    if (!exl3_up) {
      own_x = std::make_unique<Uploader>(*exl3);
      exl3_up = own_x.get();
    }
    exl3_up->upload_many(std::move(lx.items), st);
    exl3_up->sync();
  }
  return w;
}

// ---------------------------------------------------------------------------------------------------
// State and scratch

LayerState make_layer_state(const Dsv4Config& c, const LayerWeights& w, int dev, int max_seq, hipStream_t st) {
  LayerState s;
  s.layer = w.layer;
  rt::DeviceGuard g(dev);
  s.window = rt::DeviceBuffer(dev, size_t(c.window) * c.head_dim * 2);
  UH_CHECK(hipMemsetAsync(s.window.data(), 0, s.window.size(), st));
  auto ring = [&](const CompressorWeights& cw, rt::DeviceBuffer& kv, rt::DeviceBuffer& sc) {
    const size_t bytes = size_t(cw.ring()) * cw.width() * 4;
    kv = rt::DeviceBuffer(dev, bytes);
    sc = rt::DeviceBuffer(dev, bytes);
    UH_CHECK(hipMemsetAsync(kv.data(), 0, bytes, st));
    UH_CHECK(hipMemsetAsync(sc.data(), 0, bytes, st));
  };
  if (w.ratio) {
    ring(*w.comp, s.comp_ring_kv, s.comp_ring_sc);
    s.max_rows = (max_seq / w.ratio + s.rows_per_page - 1) / s.rows_per_page * s.rows_per_page;
    s.comp_pool = rt::DeviceBuffer(dev, size_t(s.max_rows) * c.head_dim * 2);
    std::vector<int32_t> table(size_t(s.max_rows / s.rows_per_page));
    for (size_t i = 0; i < table.size(); ++i) table[i] = int32_t(i);
    s.page_table = rt::DeviceBuffer(dev, table.size() * 4);
    UH_CHECK(hipMemcpy(s.page_table.data(), table.data(), table.size() * 4, hipMemcpyHostToDevice));
  }
  if (w.has_indexer) {
    ring(*w.idx_comp, s.idx_ring_kv, s.idx_ring_sc);
    s.idx_pool = rt::DeviceBuffer(dev, size_t(s.max_rows) * c.idx_dim * 2);
  }
  UH_CHECK(hipStreamSynchronize(st));
  return s;
}

namespace {

enum Buf : size_t {
  kY, kA, kPost, kComb, kQBuf, kQSc, kQA, kQR, kQ, kKV, kAF, kPK, kPS, kRowsF, kRowsB, kRowPos, kQI, kWI,
  kISc, kIdx, kO, kOA, kAO, kLG, kSel, kWt, kYF, kG, kH1, kH3, kHH, kOE, kEIdx, kEWt, kY2F, kNumBufs
};

}  // namespace

Scratch make_scratch(const Dsv4Config& c, int dev, int max_tokens, int max_seq) {
  const size_t N = size_t(max_tokens);
  const size_t rows = N / 4 + 2;
  const size_t keys = size_t(max_seq / 4 + 1);
  std::vector<size_t> sz(kNumBufs, 0);
  sz[kY] = N * c.dim * 2;
  sz[kA] = N * c.dim * 2;
  sz[kPost] = N * c.hc_mult * 4;
  sz[kComb] = N * c.hc_mult * c.hc_mult * 4;
  sz[kQBuf] = N * size_t(c.o_groups * c.o_lora);  // widest quantized activation (8192)
  sz[kQSc] = N * size_t(c.o_groups * c.o_lora / 32);
  sz[kQA] = N * c.q_lora * 2;
  sz[kQR] = N * c.q_lora * 2;
  sz[kQ] = N * c.n_heads * c.head_dim * 2;
  sz[kKV] = N * c.head_dim * 2;
  sz[kAF] = N * c.dim * 4;
  sz[kPK] = N * 2 * c.head_dim * 4;
  sz[kPS] = N * 2 * c.head_dim * 4;
  sz[kRowsF] = rows * c.head_dim * 4;
  sz[kRowsB] = rows * c.head_dim * 2;
  sz[kRowPos] = rows * 4;
  sz[kQI] = N * c.idx_heads * c.idx_dim * 2;
  sz[kWI] = N * c.idx_heads * 2;
  sz[kISc] = N * keys * 4;
  sz[kIdx] = N * c.idx_topk * 4;
  sz[kO] = N * c.n_heads * c.head_dim * 2;
  sz[kOA] = N * c.o_groups * c.o_lora * 2;
  sz[kAO] = N * c.dim * 2;
  sz[kLG] = N * c.n_experts * 4;
  sz[kSel] = N * c.n_active * 4;
  sz[kWt] = N * c.n_active * 4;
  sz[kYF] = N * c.dim * 4;
  sz[kG] = N * c.dim * 2;
  sz[kH1] = N * c.moe_inter * 2;
  sz[kH3] = N * c.moe_inter * 2;
  sz[kHH] = N * c.moe_inter * 2;
  sz[kOE] = N * c.dim * 2;
  sz[kEIdx] = N * c.n_active * 4;
  sz[kEWt] = N * c.n_active * 4;
  sz[kY2F] = N * c.dim * 4;
  Scratch s;
  s.max_tokens = max_tokens;
  s.max_keys = int(keys);
  for (size_t i = 0; i < kNumBufs; ++i) s.bufs.emplace_back(dev, std::max<size_t>(sz[i], 256));
  return s;
}

// ---------------------------------------------------------------------------------------------------
// Forward

namespace {

template <typename T>
T* B(const Scratch& s, Buf b) {
  return static_cast<T*>(s.get(b));
}

// UNHOLY_DUMP=dir: raw intermediates of the first (s0 == 0) chunk of each layer, named like the
// oracle's captures (tools/ref/oracle/run.py --capture-layer), for tools/ref/compare_dump.py.
void dump(int layer, int s0, const char* name, const void* dptr, size_t bytes, hipStream_t st) {
  static const char* dir = std::getenv("UNHOLY_DUMP");
  if (!dir || s0 != 0) return;
  std::vector<char> h(bytes);
  UH_CHECK(hipStreamSynchronize(st));
  UH_CHECK(hipMemcpy(h.data(), dptr, bytes, hipMemcpyDeviceToHost));
  char path[512];
  std::snprintf(path, sizeof path, "%s/L%03d_%s.bin", dir, layer, name);
  if (FILE* f = std::fopen(path, "wb")) {
    std::fwrite(h.data(), 1, bytes, f);
    std::fclose(f);
  }
}

// Compressor over chunk [s0, s0+n): projections of a_f32 [n, dim], emitted rows normed, roped,
// quant-simulated and stored to `pool`. Returns the number of rows emitted.
int run_compressor(const Dsv4Config& c, const CompressorWeights& cw, rt::DeviceBuffer& ring_kv,
                   rt::DeviceBuffer& ring_sc, bf16* pool, const int32_t* table, int rows_per_page, const Scratch& scr,
                   const RopeTables& rope, int s0, int n, hipStream_t st) {
  const int width = cw.width(), d = cw.d;
  float* pk = B<float>(scr, kPK);
  float* ps = B<float>(scr, kPS);
  k::gemm_f32(B<float>(scr, kAF), c.dim, cw.wkv.p<bf16>(), pk, width, n, width, c.dim, st);
  k::gemm_f32(B<float>(scr, kAF), c.dim, cw.wgate.p<bf16>(), ps, width, n, width, c.dim, st);
  k::add_ape(ps, cw.ape.p<float>(), s0, n, cw.ratio, width, st);
  const int row0 = s0 / cw.ratio, nrows = (s0 + n) / cw.ratio - row0;
  if (nrows > 0) {
    float* rf = B<float>(scr, kRowsF);
    bf16* rb = B<bf16>(scr, kRowsB);
    k::compress_rows(pk, ps, ring_kv.as<float>(), ring_sc.as<float>(), rf, s0, n, row0, nrows, cw.ratio, d,
                     cw.overlap, cw.ring(), st);
    k::cast_f32_bf16(rf, rb, size_t(nrows) * d, st);
    k::rmsnorm(rb, d, cw.norm.p<bf16>(), rb, d, nrows, d, c.norm_eps, st);
    std::vector<int32_t> rpos(static_cast<size_t>(nrows));
    for (int j = 0; j < nrows; ++j) rpos[size_t(j)] = (row0 + j) * cw.ratio;
    int32_t* dpos = B<int32_t>(scr, kRowPos);
    UH_CHECK(hipMemcpyAsync(dpos, rpos.data(), rpos.size() * 4, hipMemcpyHostToDevice, st));
    k::rope(rb, d, nrows, 1, d, d - c.rope_dim, c.rope_dim / 2, dpos, rope.yarn.as<float2>(), false, st);
    if (cw.rotate) {
      k::hadamard128(rb, size_t(nrows) * d / 128, st);
      k::fake_quant_fp4(rb, d, nrows, d, 32, st);
    } else {
      k::fake_quant_fp8(rb, d, nrows, d - c.rope_dim, 64, st);
    }
    k::store_rows(rb, nrows, d, k::PagedRows{pool, table, rows_per_page, d}, row0, st);
    UH_CHECK(hipStreamSynchronize(st));  // rpos lives on the host stack
  }
  k::ring_update(pk, ps, ring_kv.as<float>(), ring_sc.as<float>(), s0, n, cw.ring(), width, st);
  return nrows;
}

void attention_half(const Dsv4Config& c, const LayerWeights& w, LayerState& s, const Scratch& scr,
                    const RopeTables& rope, bf16* x, const int32_t* pos, int s0, int n, hipStream_t st) {
  const int D = c.dim, H = c.n_heads, HD = c.head_dim, rd = c.rope_dim;
  const float2* freqs = (w.ratio ? rope.yarn : rope.plain).as<float2>();
  bf16* y = B<bf16>(scr, kY);
  bf16* a = B<bf16>(scr, kA);
  float* post = B<float>(scr, kPost);
  float* comb = B<float>(scr, kComb);
  uint8_t* qb = B<uint8_t>(scr, kQBuf);
  uint8_t* qs = B<uint8_t>(scr, kQSc);

  k::hc_pre(x, w.hc_attn_fn.p<float>(), w.hc_attn_scale.p<float>(), w.hc_attn_base.p<float>(), y, post, comb, n,
            c.hc_mult, D, c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  const int L = w.layer;
  dump(L, s0, "hcpre_attn_y", y, size_t(n) * D * 2, st);
  dump(L, s0, "hcpre_attn_post", post, size_t(n) * c.hc_mult * 4, st);
  dump(L, s0, "hcpre_attn_comb", comb, size_t(n) * c.hc_mult * c.hc_mult * 4, st);
  k::rmsnorm(y, D, w.attn_norm.p<bf16>(), a, D, n, D, c.norm_eps, st);
  dump(L, s0, "attn_in", a, size_t(n) * D * 2, st);

  // q: wq_a -> q_norm (qr) -> wq_b -> per-head rms -> rope
  bf16* qa = B<bf16>(scr, kQA);
  bf16* qr = B<bf16>(scr, kQR);
  bf16* q = B<bf16>(scr, kQ);
  k::act_quant_fp8(a, D, qb, qs, n, D, 128, st);
  k::gemm_w8a8(qb, qs, w.wq_a.w.p<uint8_t>(), w.wq_a.s.p<uint8_t>(), qa, c.q_lora, n, c.q_lora, D, st);
  dump(L, s0, "wq_a_out", qa, size_t(n) * c.q_lora * 2, st);
  // kv: wkv from the same quantized input
  bf16* kv = B<bf16>(scr, kKV);
  k::gemm_w8a8(qb, qs, w.wkv.w.p<uint8_t>(), w.wkv.s.p<uint8_t>(), kv, HD, n, HD, D, st);
  dump(L, s0, "wkv_out", kv, size_t(n) * HD * 2, st);
  k::rmsnorm(qa, c.q_lora, w.q_norm.p<bf16>(), qr, c.q_lora, n, c.q_lora, c.norm_eps, st);
  k::act_quant_fp8(qr, c.q_lora, qb, qs, n, c.q_lora, 128, st);
  k::gemm_w8a8(qb, qs, w.wq_b.w.p<uint8_t>(), w.wq_b.s.p<uint8_t>(), q, H * HD, n, H * HD, c.q_lora, st);
  dump(L, s0, "wq_b_out", q, size_t(n) * H * HD * 2, st);
  k::head_rms_bf16(q, n, H, HD, c.norm_eps, st);
  k::rope(q, H * HD, n, H, HD, HD - rd, rd / 2, pos, freqs, false, st);

  k::rmsnorm(kv, HD, w.kv_norm.p<bf16>(), kv, HD, n, HD, c.norm_eps, st);
  k::rope(kv, HD, n, 1, HD, HD - rd, rd / 2, pos, freqs, false, st);
  k::fake_quant_fp8(kv, HD, n, HD - rd, 64, st);

  k::AttnKv akv;
  akv.chunk_kv = kv;
  akv.ring = s.window.as<bf16>();
  akv.s0 = s0;
  akv.ratio = w.ratio;
  if (w.ratio) {
    k::cast_bf16_f32(a, B<float>(scr, kAF), size_t(n) * D, st);
    const int32_t* table = s.page_table.as<int32_t>();
    if (w.has_indexer) {
      // Indexer q from qr (still quantized in qb/qs), weights from x, its compressor, scores, top-k.
      bf16* qi = B<bf16>(scr, kQI);
      bf16* wi = B<bf16>(scr, kWI);
      const int IH = c.idx_heads, ID = c.idx_dim;
      k::gemm_w8a8(qb, qs, w.idx_wq_b.w.p<uint8_t>(), w.idx_wq_b.s.p<uint8_t>(), qi, IH * ID, n, IH * ID, c.q_lora,
                   st);
      k::rope(qi, IH * ID, n, IH, ID, ID - rd, rd / 2, pos, freqs, false, st);
      k::hadamard128(qi, size_t(n) * IH * ID / 128, st);
      k::fake_quant_fp4(qi, IH * ID, n, IH * ID, 32, st);
      k::gemm_bf16(a, D, w.idx_wproj.p<bf16>(), wi, IH, n, IH, D, st);
      // weights_proj(x) * (softmax_scale * n_heads^-0.5): the Python double, applied as fp32.
      const double wscale = (1.0 / std::sqrt(double(ID))) * (1.0 / std::sqrt(double(IH)));
      k::scale_bf16(wi, float(wscale), size_t(n) * IH, st);
      run_compressor(c, *w.idx_comp, s.idx_ring_kv, s.idx_ring_sc, s.idx_pool.as<bf16>(), table, s.rows_per_page,
                     scr, rope, s0, n, st);
      const int nkeys = (s0 + n) / 4;
      const int kk = std::min(c.idx_topk, nkeys);
      if (kk > 0) {
        float* sc = B<float>(scr, kISc);
        k::indexer_scores(qi, wi, k::PagedRows{s.idx_pool.as<bf16>(), table, s.rows_per_page, ID}, pos, sc, n, IH,
                          ID, nkeys, 4, st);
        k::topk_rows(sc, n, nkeys, kk, B<int32_t>(scr, kIdx), st);
        dump(L, s0, "indexer_out", B<int32_t>(scr, kIdx), size_t(n) * kk * 4, st);
      }
      akv.comp_idx = B<int32_t>(scr, kIdx);
      akv.ncomp = kk;
    }
    run_compressor(c, *w.comp, s.comp_ring_kv, s.comp_ring_sc, s.comp_pool.as<bf16>(), table, s.rows_per_page, scr,
                   rope, s0, n, st);
    akv.comp = k::PagedRows{s.comp_pool.as<bf16>(), table, s.rows_per_page, HD};
  }

  bf16* o = B<bf16>(scr, kO);
  dump(L, s0, "sa_q", q, size_t(n) * H * HD * 2, st);
  dump(L, s0, "sa_kv", kv, size_t(n) * HD * 2, st);
  k::sparse_attn(q, pos, akv, w.sink.p<float>(), o, n, H, HD, float(1.0 / std::sqrt(double(HD))), st);
  dump(L, s0, "sa_o", o, size_t(n) * H * HD * 2, st);
  k::window_update(kv, s.window.as<bf16>(), s0, n, HD, st);
  k::rope(o, H * HD, n, H, HD, HD - rd, rd / 2, pos, freqs, true, st);

  // o: grouped wo_a (bf16 einsum on dequantized fp8 weights), then wo_b.
  bf16* oa = B<bf16>(scr, kOA);
  bf16* ao = B<bf16>(scr, kAO);
  const int G = c.o_groups, R = c.o_lora, GK = H * HD / G;
  k::gemm_grouped_fp8w(o, w.wo_a.w.p<uint8_t>(), w.wo_a.s.p<uint8_t>(), oa, n, G, R, GK, st);
  dump(L, s0, "wo_b_in", oa, size_t(n) * G * R * 2, st);
  k::act_quant_fp8(oa, G * R, qb, qs, n, G * R, 128, st);
  k::gemm_w8a8(qb, qs, w.wo_b.w.p<uint8_t>(), w.wo_b.s.p<uint8_t>(), ao, D, n, D, G * R, st);
  dump(L, s0, "attn_out", ao, size_t(n) * D * 2, st);

  k::hc_post(ao, x, post, comb, x, n, c.hc_mult, D, st);
}

void moe_half(const Dsv4Config& c, const LayerWeights& w, const Scratch& scr, bf16* x, const int32_t* ids, int s0, int n,
              hipStream_t st) {
  const int D = c.dim, I = c.moe_inter, E = c.n_experts, K = c.n_active;
  bf16* y = B<bf16>(scr, kY);
  bf16* f = B<bf16>(scr, kA);
  float* post = B<float>(scr, kPost);
  float* comb = B<float>(scr, kComb);
  uint8_t* qb = B<uint8_t>(scr, kQBuf);
  uint8_t* qs = B<uint8_t>(scr, kQSc);
  k::hc_pre(x, w.hc_ffn_fn.p<float>(), w.hc_ffn_scale.p<float>(), w.hc_ffn_base.p<float>(), y, post, comb, n,
            c.hc_mult, D, c.sinkhorn_iters, c.norm_eps, c.hc_eps, st);
  const int L = w.layer;
  dump(L, s0, "hcpre_ffn_y", y, size_t(n) * D * 2, st);
  dump(L, s0, "hcpre_ffn_post", post, size_t(n) * c.hc_mult * 4, st);
  dump(L, s0, "hcpre_ffn_comb", comb, size_t(n) * c.hc_mult * c.hc_mult * 4, st);
  k::rmsnorm(y, D, w.ffn_norm.p<bf16>(), f, D, n, D, c.norm_eps, st);
  dump(L, s0, "ffn_in", f, size_t(n) * D * 2, st);

  // Router (fp32 linear on x.float()).
  float* af = B<float>(scr, kAF);
  float* lg = B<float>(scr, kLG);
  int32_t* sel = B<int32_t>(scr, kSel);
  float* wt = B<float>(scr, kWt);
  k::cast_bf16_f32(f, af, size_t(n) * D, st);
  k::gemm_f32(af, D, w.gate_w.p<bf16>(), lg, E, n, E, D, st);
  k::route(lg, w.gate_bias.p<float>(), w.hash ? w.tid2eid.p<int32_t>() : nullptr, ids, sel, wt, n, E, K,
           c.route_scale, st);
  dump(L, s0, "gate_out0", wt, size_t(n) * K * 4, st);
  dump(L, s0, "gate_out1", sel, size_t(n) * K * 4, st);

  // Group (token, weight) by expert, ascending expert id (the reference's accumulation order).
  std::vector<int32_t> hsel(size_t(n) * K);
  std::vector<float> hwt(size_t(n) * K);
  UH_CHECK(hipMemcpyAsync(hsel.data(), sel, hsel.size() * 4, hipMemcpyDeviceToHost, st));
  UH_CHECK(hipMemcpyAsync(hwt.data(), wt, hwt.size() * 4, hipMemcpyDeviceToHost, st));
  UH_CHECK(hipStreamSynchronize(st));
  std::vector<std::vector<std::pair<int32_t, float>>> per(static_cast<size_t>(E));
  for (int t = 0; t < n; ++t)
    for (int j = 0; j < K; ++j) per[size_t(hsel[size_t(t) * K + j])].push_back({t, hwt[size_t(t) * K + j]});
  std::vector<int32_t> eidx;
  std::vector<float> ewt;
  std::vector<int> off(size_t(E) + 1, 0);
  for (int e = 0; e < E; ++e) {
    off[size_t(e)] = int(eidx.size());
    for (auto& [t, v] : per[size_t(e)]) {
      eidx.push_back(t);
      ewt.push_back(v);
    }
  }
  off[size_t(E)] = int(eidx.size());
  int32_t* deidx = B<int32_t>(scr, kEIdx);
  float* dewt = B<float>(scr, kEWt);
  UH_CHECK(hipMemcpyAsync(deidx, eidx.data(), eidx.size() * 4, hipMemcpyHostToDevice, st));
  UH_CHECK(hipMemcpyAsync(dewt, ewt.data(), ewt.size() * 4, hipMemcpyHostToDevice, st));

  float* yf = B<float>(scr, kYF);
  UH_CHECK(hipMemsetAsync(yf, 0, size_t(n) * D * 4, st));
  bf16* g = B<bf16>(scr, kG);
  bf16* h1 = B<bf16>(scr, kH1);
  bf16* h3 = B<bf16>(scr, kH3);
  bf16* hh = B<bf16>(scr, kHH);
  bf16* oe = B<bf16>(scr, kOE);
  for (int e = 0; e < E; ++e) {
    const int cnt = off[size_t(e) + 1] - off[size_t(e)];
    if (!cnt) continue;
    const int32_t* rows = deidx + off[size_t(e)];
    k::gather_rows(f, rows, g, cnt, D, st);
    if (w.exl3_experts) {
      // EXL3 weights on the fp8-quantized activations the experts were trained with (A8 fake quant).
      float* xf = B<float>(scr, kAF);
      float* y2 = B<float>(scr, kY2F);
      auto lin = [&](const Exl3Weights& xw, const bf16* in, int in_dim, bf16* out, int out_dim) {
        k::exl3_in(in, in_dim, xw.suh.p<uint16_t>(), xf, cnt, in_dim, st);
        k::gemm_exl3_inner(xf, xw.trellis.p<uint16_t>(), xw.bits, y2, cnt, out_dim, in_dim, st);
        k::exl3_out(y2, xw.svh.p<uint16_t>(), out, out_dim, cnt, out_dim, st);
      };
      k::fake_quant_fp8(g, D, cnt, D, 128, st);
      lin(w.x1[size_t(e)], g, D, h1, I);
      lin(w.x3[size_t(e)], g, D, h3, I);
      k::swiglu(h1, h3, dewt + off[size_t(e)], hh, cnt, I, c.swiglu_limit, st);
      k::fake_quant_fp8(hh, I, cnt, I, 128, st);
      lin(w.x2[size_t(e)], hh, I, oe, D);
    } else {
      k::act_quant_fp8(g, D, qb, qs, cnt, D, 128, st);
      k::gemm_a8w4(qb, qs, w.w1[size_t(e)].w.p<uint8_t>(), w.w1[size_t(e)].s.p<uint8_t>(), h1, I, cnt, I, D, st);
      k::gemm_a8w4(qb, qs, w.w3[size_t(e)].w.p<uint8_t>(), w.w3[size_t(e)].s.p<uint8_t>(), h3, I, cnt, I, D, st);
      k::swiglu(h1, h3, dewt + off[size_t(e)], hh, cnt, I, c.swiglu_limit, st);
      k::act_quant_fp8(hh, I, qb, qs, cnt, I, 128, st);
      k::gemm_a8w4(qb, qs, w.w2[size_t(e)].w.p<uint8_t>(), w.w2[size_t(e)].s.p<uint8_t>(), oe, D, cnt, D, I, st);
    }
    k::scatter_add(oe, rows, yf, cnt, D, st);
  }
  // Shared expert (fp8 weights, no routing weight).
  k::act_quant_fp8(f, D, qb, qs, n, D, 128, st);
  k::gemm_w8a8(qb, qs, w.sw1.w.p<uint8_t>(), w.sw1.s.p<uint8_t>(), h1, I, n, I, D, st);
  k::gemm_w8a8(qb, qs, w.sw3.w.p<uint8_t>(), w.sw3.s.p<uint8_t>(), h3, I, n, I, D, st);
  k::swiglu(h1, h3, nullptr, hh, n, I, c.swiglu_limit, st);
  k::act_quant_fp8(hh, I, qb, qs, n, I, 128, st);
  k::gemm_w8a8(qb, qs, w.sw2.w.p<uint8_t>(), w.sw2.s.p<uint8_t>(), oe, D, n, D, I, st);
  dump(L, s0, "shared_out", oe, size_t(n) * D * 2, st);
  k::add_bf16_to_f32(oe, yf, size_t(n) * D, st);
  bf16* fo = B<bf16>(scr, kAO);
  k::cast_f32_bf16(yf, fo, size_t(n) * D, st);
  dump(L, s0, "ffn_out", fo, size_t(n) * D * 2, st);
  k::hc_post(fo, x, post, comb, x, n, c.hc_mult, D, st);
  UH_CHECK(hipStreamSynchronize(st));  // eidx/ewt host vectors
}

}  // namespace

void layer_forward(const Dsv4Config& c, const LayerWeights& w, LayerState& s, const Scratch& scr,
                   const RopeTables& rope, uint16_t* x, const int32_t* ids, const int32_t* pos, int s0, int n,
                   hipStream_t st) {
  if (n > scr.max_tokens) throw std::runtime_error("layer_forward: chunk larger than scratch");
  if ((s0 + n) / 4 > scr.max_keys) throw std::runtime_error("layer_forward: context larger than scratch");
  attention_half(c, w, s, scr, rope, x, pos, s0, n, st);
  moe_half(c, w, scr, x, ids, s0, n, st);
}

}  // namespace unholy::model
