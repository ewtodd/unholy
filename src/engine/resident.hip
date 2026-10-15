#include "engine/resident.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/dspark.hpp"
#include "model/vision.hpp"
#include "kernels/hc_fast.hpp"
#include "kernels/ref.hpp"
#include "model/loader.hpp"
#include "rt/hip.hpp"

namespace unholy::engine {

// The vision tower on the APU: weights, scratch, its stream, the output rows, one image at a time.
struct ResidentModel::Vision {
  model::VisionWeights w;
  model::VisionScratch s;
  rt::DeviceBuffer out;
  hipStream_t st = nullptr;
  int dev = 0;
  std::mutex mu;
};

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// ApuExperts

ApuExperts::ApuExperts(const model::Dsv4Config& c, int dev, size_t capacity_rows, std::shared_ptr<Layers> layers,
                       bool exl3)
    : c_(c), dev_(dev), layers_(std::move(layers)) {
  rt::DeviceGuard g(dev);
  UH_CHECK(hipStreamCreateWithFlags(&st_, hipStreamNonBlocking));
  this->capacity_rows = capacity_rows;
  const unsigned flags = hipHostMallocPortable | hipHostMallocMapped;
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in_host), capacity_rows * c.dim * 2, flags));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&out_host), capacity_rows * c.dim * 2, flags));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&w_host), capacity_rows * 4, flags));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&e_host), capacity_rows * 4, flags));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&e_dev_), e_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&in_dev_), in_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&out_dev_), out_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&w_dev_), w_host, 0));
  // Compact fp8 inputs: at most one row per token of a step (capacity_rows / n_active tokens).
  if (const char* q = std::getenv("UNHOLY_REMOTE_Q")) quant_in_ = std::string(q) != "0";
  const size_t cap_tokens = (capacity_rows + size_t(c.n_active) - 1) / size_t(c.n_active);
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&inq_host), cap_tokens * c.dim, flags));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ins_host), cap_tokens * (c.dim / 128), flags));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&p2u_host), capacity_rows * 4, flags));
  combine_out_ = quant_in_;
  if (const char* m = std::getenv("UNHOLY_REMOTE_COMBINE")) combine_out_ = quant_in_ && std::string(m) != "0";
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&u2p_host), cap_tokens * size_t(c.n_active) * 4, flags));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&u2p_dev_), u2p_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&inq_dev_), inq_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ins_dev_), ins_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&p2u_dev_), p2u_host, 0));
  // Each job is one expert: at most every token of a step.
  if (model::ref_experts_needed()) scratch_ = model::make_expert_scratch(c, dev, int(capacity_rows));  // run()
  xscratch_ = model::make_exl3_moe_scratch(c, dev, int(capacity_rows), exl3);
  max_tiles = xscratch_.max_tiles;
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&plan_host), sizeof(kernels::MoePlan), flags));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&tiles_host), size_t(max_tiles) * sizeof(kernels::Exl3Tile), flags));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&plan_dev_), plan_host, 0));
  UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&tiles_dev_), tiles_host, 0));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&req_flag_), 64, flags | hipHostMallocCoherent));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&done_flag_), 64, flags | hipHostMallocCoherent));
  *req_flag_ = 0;
  *done_flag_ = 0;
  if (layers_->size() < size_t(c.n_layers)) layers_->resize(size_t(c.n_layers));
}

ApuExperts::~ApuExperts() {
  rt::DeviceGuard g(dev_);
  if (st_) (void)hipStreamDestroy(st_);
  if (in_host) (void)hipHostFree(in_host);
  if (inq_host) (void)hipHostFree(inq_host);
  if (ins_host) (void)hipHostFree(ins_host);
  if (p2u_host) (void)hipHostFree(p2u_host);
  if (u2p_host) (void)hipHostFree(u2p_host);
  if (out_host) (void)hipHostFree(out_host);
  if (w_host) (void)hipHostFree(w_host);
  if (e_host) (void)hipHostFree(e_host);
  if (plan_host) (void)hipHostFree(plan_host);
  if (tiles_host) (void)hipHostFree(tiles_host);
  if (req_flag_) (void)hipHostFree(req_flag_);
  if (done_flag_) (void)hipHostFree(done_flag_);
}

void ApuExperts::add_layer(model::LayerWeights w) {
  const int L = w.layer;
  if (layers_->size() <= size_t(L)) layers_->resize(size_t(L) + 1);
  layers_->at(size_t(L)) = std::make_unique<model::LayerWeights>(std::move(w));
}

void ApuExperts::run(int layer, const std::vector<Job>& jobs) {
  const auto& w = *layers_->at(size_t(layer));
  rt::DeviceGuard g(dev_);
  if (w.exl3_experts && model::exl3_fast_enabled()) {
    std::vector<std::array<int, 3>> runs;
    int P = 0;
    for (const auto& j : jobs) {
      if (!w.has_expert(j.expert)) throw std::runtime_error("ApuExperts: expert not resident on the APU");
      runs.push_back({j.expert, j.row0, j.cnt});
      P = std::max(P, j.row0 + j.cnt);
    }
    model::exl3_moe_forward(c_, w, in_dev_, w_dev_, e_dev_, model::exl3_tiles(runs), P, xscratch_, out_dev_, st_);
    UH_CHECK(hipStreamSynchronize(st_));
    return;
  }
  for (const auto& j : jobs) {
    if (!w.has_expert(j.expert)) throw std::runtime_error("ApuExperts: expert not resident on the APU");
    const size_t o = size_t(j.row0);
    model::expert_forward(c_, w, j.expert, in_dev_ + o * c_.dim, w_dev_ + o, j.cnt, out_dev_ + o * c_.dim, scratch_,
                          st_);
  }
  UH_CHECK(hipStreamSynchronize(st_));
}

// Device-ordered handoff: the card stream raises req after writing the inputs; the APU stream waits for
// it, runs the remote tiles, raises done; join() makes the card stream wait for done. Flags are
// sequence numbers (>= waits), so no reset is needed and nothing blocks the host.
void ApuExperts::enqueue(int layer, hipStream_t card_st, int P) {
  const auto& w = *layers_->at(size_t(layer));
  ++seq_;
  UH_CHECK(hipStreamWriteValue32(card_st, req_flag_, seq_, 0));
  rt::DeviceGuard g(dev_);
  UH_CHECK(hipStreamWaitValue32(st_, req_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
  if (!w.exl3_experts && quant_in_)  // the card sent compact fp8 inputs (moe_device_dispatch)
    model::mx_moe_forward_dev_q(c_, w, inq_dev_, ins_dev_, p2u_dev_, plan_dev_, w_dev_, e_dev_, tiles_dev_, max_tiles,
                                &plan_dev_->n_remote_tiles, P, xscratch_, out_dev_, st_, combine_out_ ? u2p_dev_ : nullptr);
  else
    model::exl3_moe_forward_dev(c_, w, in_dev_, w_dev_, e_dev_, tiles_dev_, max_tiles, &plan_dev_->n_remote_tiles, P,
                                xscratch_, out_dev_, st_);
  UH_CHECK(hipStreamWriteValue32(st_, done_flag_, seq_, 0));
}

void ApuExperts::join(hipStream_t card_st) {
  UH_CHECK(hipStreamWaitValue32(card_st, done_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
}

// ---------------------------------------------------------------------------------------------------
// ResidentModel

namespace {
rt::DeviceBuffer upload_tensor(model::Uploader& up, const model::Checkpoint& ck, const char* name, int dev,
                               hipStream_t st) {
  const auto& t = ck.at(name);
  rt::DeviceBuffer b(dev, t.nbytes);
  up.upload(t, b.data(), st);
  up.sync();
  return b;
}
}  // namespace

ResidentModel::ResidentModel(const model::Dsv4Config& c, const model::Checkpoint& dense,
                             const model::Checkpoint* exl3, const Plan& plan, const model::CacheGeometry& geo,
                             int max_tokens, int max_seq, int max_logit_rows, int snapshots, bool dspark,
                             bool vision)
    : c_(c), plan_(plan), geo_(geo), max_tokens_(max_tokens),
      max_logit_rows_(max_logit_rows > 0 ? std::min(max_logit_rows, max_tokens) : max_tokens), snapshots_(snapshots),
      dspark_(dspark) {
  model::Uploader up(dense);
  std::unique_ptr<model::Uploader> xup;
  if (exl3) xup = std::make_unique<model::Uploader>(*exl3);
  {
    auto layers = std::make_shared<ApuExperts::Layers>();
    for (auto& ch : chan_)
      ch = std::make_unique<ApuExperts>(c, plan.apu, size_t(max_tokens) * c.n_active, layers, exl3 != nullptr);
    if (dspark_)
      dchan_ = std::make_unique<ApuExperts>(c, plan.apu, size_t(c.dspark_block) * geo.max_slots * c.n_active, layers, false);
  }
  if (const char* v = std::getenv("UNHOLY_INFLIGHT")) inflight_ = std::clamp(std::atoi(v), 1, 2);
  if (const char* v = std::getenv("UNHOLY_MICROBATCH")) microbatch_ = std::string(v) == "1";
  if (std::getenv("UNHOLY_CANARY")) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&canary_), sizeof(*canary_), hipHostMallocPortable | hipHostMallocMapped));
    std::memset(canary_, 0, sizeof(*canary_));
    k::hc_set_canary(canary_);
  }
  for (int k = 0; k < 2; ++k) card_arena_[k] = std::make_unique<rt::Arena>(plan.card[k], size_t(256) << 20);
  apu_arena_ = std::make_unique<rt::Arena>(plan.apu, size_t(256) << 20);

  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    cd.dev = plan.card[k];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipStreamCreateWithFlags(&cd.st, hipStreamNonBlocking));
    cd.scratch = model::make_scratch(c, cd.dev, max_tokens, max_seq, exl3 != nullptr);
    cd.rope = model::make_rope_tables(c, cd.dev, max_seq + 128);
    cd.x = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    cd.ids = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.pos = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.tables = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
    cd.vis = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 2 * 4);
  }
  for (int L = 0; L < c.n_layers; ++L) {
    Card& cd = card_[plan.layer_card[size_t(L)]];
    rt::DeviceGuard g(cd.dev);
    // Card: dense + hot experts.
    model::LoadOptions lo;
    lo.up = &up;
    lo.exl3 = exl3;
    lo.exl3_up = xup.get();
    lo.experts = &plan.on_card[size_t(L)];
    lo.arena = card_arena_[plan.layer_card[size_t(L)]].get();
    cd.layers.push_back(L);
    cd.w.push_back(model::load_layer(dense, c, L, cd.dev, cd.st, lo));
    cd.cache.push_back(model::make_layer_cache(c, cd.w.back(), geo, cd.dev, cd.st));
    // APU: the cold experts.
    std::vector<uint8_t> cold(size_t(c.n_experts));
    for (int e = 0; e < c.n_experts; ++e) cold[size_t(e)] = !plan.on_card[size_t(L)][size_t(e)];
    model::LoadOptions la;
    la.up = &up;
    la.exl3 = exl3;
    la.exl3_up = xup.get();
    la.dense = false;
    la.experts = &cold;
    la.arena = apu_arena_.get();
    {
      rt::DeviceGuard ga(plan.apu);
      chan_[0]->add_layer(model::load_layer(dense, c, L, plan.apu, chan_[0]->stream(), la));
    }
  }
  {
    rt::DeviceGuard g(card_[0].dev);
    embed_ = upload_tensor(up, dense, "embed.weight", card_[0].dev, card_[0].st);
  }
  {
    Card& cd = card_[1];
    rt::DeviceGuard g(cd.dev);
    head_ = upload_tensor(up, dense, "head.weight", cd.dev, cd.st);
    norm_ = upload_tensor(up, dense, "norm.weight", cd.dev, cd.st);
    hc_fn_ = upload_tensor(up, dense, "hc_head_fn", cd.dev, cd.st);
    hc_scale_ = upload_tensor(up, dense, "hc_head_scale", cd.dev, cd.st);
    hc_base_ = upload_tensor(up, dense, "hc_head_base", cd.dev, cd.st);
    hrows_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    hb_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.dim * 2);
    hf_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.dim * 4);
    logits_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * c.vocab * 4);
    rows_idx_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
  }
  if (std::getenv("UNHOLY_DEBUG_ADDRS")) {  // diagnostics: buffer ranges, to attribute GPU faults
    const auto pr = [](const char* what, int k, const void* p, size_t n) {
      std::fprintf(stderr, "addr %-14s card %d  %p .. %p\n", what, k, p, static_cast<const char*>(p) + n);
    };
    for (int k = 0; k < 2; ++k) {
      const Card& cd = card_[k];
      pr("x", k, cd.x.data(), cd.x.size());
      pr("ids", k, cd.ids.data(), cd.ids.size());
      pr("tables", k, cd.tables.data(), cd.tables.size());
      for (size_t i = 0; i < cd.scratch.bufs.size(); ++i) pr("scratch", k, cd.scratch.bufs[i].data(), cd.scratch.bufs[i].size());
      for (size_t i = 0; i < cd.cache.size(); ++i) {
        pr("cache.comp", k, cd.cache[i].comp.data(), cd.cache[i].comp.size());
        pr("cache.idx", k, cd.cache[i].idx.data(), cd.cache[i].idx.size());
        pr("cache.window", k, cd.cache[i].window_rows.data(), cd.cache[i].window_rows.size());
      }
    }
  }
  if (vision && c.vision_layers > 0) {
    vision_ = std::make_unique<Vision>();
    vision_->dev = plan.apu;
    rt::DeviceGuard g(plan.apu);
    UH_CHECK(hipStreamCreateWithFlags(&vision_->st, hipStreamNonBlocking));
    vision_->w = model::load_vision(dense, c, plan.apu, vision_->st);
    vision_->s = model::make_vision_scratch(c, plan.apu);
    vision_->out = rt::DeviceBuffer(plan.apu, size_t(c.vision_max_tokens + 16) * c.dim * 2);
  }
  const auto pin_host = [](auto** p, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(p), bytes, hipHostMallocPortable | hipHostMallocMapped));
  };
  if (dspark_) {
    for (const int L : c.dspark_target_layers)
      if (plan.layer_card[size_t(L)] != 1) throw std::runtime_error("dspark: target layers must be on card 1");
    Card& cd = card_[1];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipStreamCreateWithFlags(&dst_, hipStreamNonBlocking));
    const std::vector<uint8_t> none(size_t(c.n_experts), 0), all(size_t(c.n_experts), 1);
    model::LoadOptions lo;
    lo.up = &up;
    lo.experts = &none;
    lo.arena = card_arena_[1].get();
    dw_ = model::load_dspark(dense, c, cd.dev, cd.st, lo);
    for (const auto& w : dw_.stages) dcache_.push_back(model::make_layer_cache(c, w, geo, cd.dev, cd.st));
    model::LoadOptions la;
    la.up = &up;
    la.dense = false;
    la.experts = &all;
    la.arena = apu_arena_.get();
    {
      rt::DeviceGuard ga(plan.apu);
      auto apu = model::load_dspark(dense, c, plan.apu, chan_[0]->stream(), la);
      for (auto& w : apu.stages) chan_[0]->add_layer(std::move(w));
    }
    const int rows = c.dspark_block * geo.max_slots;
    dscr_ = model::make_scratch(c, cd.dev, rows, 4096, false);  // the drafter's experts are native
    cap_max_ = geo.max_slots * 128;
    mh_ = rt::DeviceBuffer(cd.dev, size_t(cap_max_) * c.dspark_target_layers.size() * c.dim * 2);
    cap_dev_ = rt::DeviceBuffer(cd.dev, size_t(cap_max_) * 3 * 4);
    d_x_ = rt::DeviceBuffer(cd.dev, size_t(rows) * c.hc_mult * c.dim * 2);
    d_in_ = rt::DeviceBuffer(cd.dev, size_t(rows) * 2 * 4);
    d_logits_ = rt::DeviceBuffer(cd.dev, size_t(rows) * c.vocab * 4);
    d_tok_ = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * (c.dspark_block + 1) * 4);
    d_conf_ = rt::DeviceBuffer(cd.dev, size_t(rows) * 4);
    for (auto& p : cap_host_) pin_host(&p, size_t(cap_max_) * 3 * 4);
    pin_host(&d_x_host_, size_t(rows) * c.hc_mult * c.dim * 2);
    pin_host(&d_io_host_, (size_t(rows) * 2 + size_t(geo.max_slots) * (c.dspark_block + 1)) * 4);
    pin_host(&d_conf_host_, size_t(rows) * 4);
    embed_host_ = dense.as<uint16_t>(dense.at("embed.weight"));
    // Ring backups for verify chunks: every per-slot ring of each card (and the stages' windows).
    for (int k = 0; k < 2; ++k) {
      std::vector<k::RingDesc> descs;
      size_t off = 0;
      const auto add = [&](const rt::DeviceBuffer& b, int ring, int row_bytes) {
        descs.push_back({static_cast<uint8_t*>(const_cast<void*>(b.data())), size_t(ring) * row_bytes, ring, row_bytes, off});
        off += size_t(row_bytes);
      };
      const auto add_cache = [&](const model::LayerCache& lc) {
        add(lc.window_rows, c.window, c.head_dim * 2);
        if (lc.ratio) {
          add(lc.ring_kv, lc.ring, lc.width * 4);
          add(lc.ring_sc, lc.ring, lc.width * 4);
        }
        if (lc.indexer) {
          add(lc.iring_kv, lc.iring, lc.iwidth * 4);
          add(lc.iring_sc, lc.iring, lc.iwidth * 4);
        }
      };
      for (const auto& lc : card_[k].cache) add_cache(lc);
      if (k == 1)
        for (const auto& lc : dcache_) add_cache(lc);
      rt::DeviceGuard gk(card_[k].dev);
      n_rings_[k] = int(descs.size());
      ring_pos_bytes_[k] = off;
      ring_desc_[k] = rt::DeviceBuffer(card_[k].dev, descs.size() * sizeof(k::RingDesc));
      UH_CHECK(hipMemcpy(ring_desc_[k].data(), descs.data(), descs.size() * sizeof(k::RingDesc), hipMemcpyHostToDevice));
      ring_bak_[k] = rt::DeviceBuffer(card_[k].dev, size_t(geo.max_slots) * (c.dspark_block + 1) * off);
    }
  }
  for (Flight& f : flight_) {
    const auto pinned = [](auto** p, size_t bytes) {
      UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(p), bytes, hipHostMallocPortable | hipHostMallocMapped));
    };
    pinned(&f.x_host, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    pinned(&f.logits_host, size_t(max_logit_rows_) * c.vocab * 4);
    pinned(&f.ids_host, size_t(max_tokens) * 3 * 4);
    pinned(&f.tables_host, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
    pinned(&f.vis_host, size_t(max_tokens) * 2 * 4);
    pinned(&f.img_host, size_t(max_tokens) * c.dim * 2);
    {
      rt::DeviceGuard g(card_[0].dev);
      UH_CHECK(hipEventCreateWithFlags(&f.handoff, hipEventDisableTiming));
      UH_CHECK(hipEventCreateWithFlags(&f.handoff2, hipEventDisableTiming));
    }
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventCreateWithFlags(&f.consumed, hipEventDisableTiming));
    UH_CHECK(hipEventCreateWithFlags(&f.done, hipEventDisableTiming));
    if (const char* path = std::getenv("UNHOLY_DEBUG_TENSORS"); path && &f == &flight_[0]) {
      // Every per-layer tensor of both cards: "card layer name begin end" (attributing GPU faults).
      if (FILE* out = std::fopen(path, "w")) {
        const auto put = [&](int k, int L, const std::string& name, const model::DevTensor& t, size_t bytes) {
          if (t.ptr) std::fprintf(out, "%d %d %s %p %p\n", k, L, name.c_str(), t.ptr, static_cast<char*>(t.ptr) + bytes);
        };
        for (int k = 0; k < 2; ++k)
          for (const auto& w : card_[k].w) {
            const int L = w.layer;
            put(k, L, "hc_attn_fn", w.hc_attn_fn, size_t(24) * c.hc_mult * c.dim * 4);
            put(k, L, "hc_ffn_fn", w.hc_ffn_fn, size_t(24) * c.hc_mult * c.dim * 4);
            for (size_t e = 0; e < w.w1.size(); ++e)
              for (const auto& [nm, l] : {std::pair<const char*, const model::Fp4Linear*>{"w1", &w.w1[e]}, {"w2", &w.w2[e]},
                                          {"w3", &w.w3[e]}}) {
                put(k, L, std::string(nm) + "." + std::to_string(e) + ".w", l->w, size_t(l->out) * l->in / 2);
                put(k, L, std::string(nm) + "." + std::to_string(e) + ".s", l->s, size_t(l->out) * (l->in / 32));
              }
            for (const auto& [nm, l] : {std::pair<const char*, const model::Fp8Linear*>{"wq_a", &w.wq_a}, {"wq_b", &w.wq_b},
                                        {"wkv", &w.wkv}, {"wo_a", &w.wo_a}, {"wo_b", &w.wo_b}, {"sw1", &w.sw1},
                                        {"sw2", &w.sw2}, {"sw3", &w.sw3}})
              put(k, L, nm, l->w, size_t(l->out) * l->in);
          }
        std::fclose(out);
      }
    }
    if (std::getenv("UNHOLY_DEBUG_ADDRS") && &f == &flight_[0]) {
      const auto span = [](const char* what, const rt::Arena& a) {
        for (const auto& sl : a.slabs())
          std::fprintf(stderr, "addr %-14s card %d  %p .. %p\n", what, a.device(), sl.data(),
                       static_cast<const char*>(sl.data()) + sl.size());
      };
      span("arena", *card_arena_[0]);
      span("arena", *card_arena_[1]);
      span("apu_arena", *apu_arena_);
      for (int k = 0; k < 2; ++k)
        std::fprintf(stderr, "addr chan %d in %p out %p w %p e %p plan %p tiles %p\n", k,
                     static_cast<void*>(chan_[k]->in_host), static_cast<void*>(chan_[k]->out_host),
                     static_cast<void*>(chan_[k]->w_host), static_cast<void*>(chan_[k]->e_host),
                     static_cast<void*>(chan_[k]->plan_host), static_cast<void*>(chan_[k]->tiles_host));
    }
    if (std::getenv("UNHOLY_DEBUG_ADDRS"))
      std::fprintf(stderr, "addr flight x_host %p .. %p, logits %p, ids %p, tables %p\n", static_cast<void*>(f.x_host),
                   static_cast<void*>(f.x_host + size_t(max_tokens) * c.hc_mult * c.dim), static_cast<void*>(f.logits_host),
                   static_cast<void*>(f.ids_host), static_cast<void*>(f.tables_host));
  }
  if (snapshots_ > 0) {
    for (int k = 0; k < 2; ++k) {
      size_t off = 0;
      const auto add = [&](const rt::DeviceBuffer& b, size_t per_slot) {
        snap_segs_[k].push_back({static_cast<uint8_t*>(const_cast<void*>(b.data())), per_slot, off});
        off += per_slot;
      };
      if (k == 1)  // the drafter's windows are per-slot state too
        for (const model::LayerCache& lc : dcache_) add(lc.window_rows, size_t(c.window) * c.head_dim * 2);
      for (const model::LayerCache& lc : card_[k].cache) {
        add(lc.window_rows, size_t(c.window) * c.head_dim * 2);
        if (lc.ratio == 4) {
          add(lc.ring_kv, size_t(lc.ring) * lc.width * 4);
          add(lc.ring_sc, size_t(lc.ring) * lc.width * 4);
        }
        if (lc.indexer) {
          add(lc.iring_kv, size_t(lc.iring) * lc.iwidth * 4);
          add(lc.iring_sc, size_t(lc.iring) * lc.iwidth * 4);
        }
      }
      snap_bytes_[k] = off;
      UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&snap_host_[k]), off * size_t(snapshots_),
                             hipHostMallocPortable | hipHostMallocMapped));
    }
  }
}

// Restores (before the step's layers) or saves (after them) slot states on card k's stream.
void ResidentModel::state_copies(int k, const std::vector<StateOp>& ops, bool save) {
  Card& cd = card_[k];
  for (const StateOp& op : ops) {
    if (op.snapshot < 0 || op.snapshot >= snapshots_ || op.slot < 0 || op.slot >= geo_.max_slots)
      throw std::runtime_error("step: bad state op");
    uint8_t* snap = snap_host_[k] + size_t(op.snapshot) * snap_bytes_[k];
    for (const SnapSeg& s : snap_segs_[k]) {
      uint8_t* slot = s.dev + size_t(op.slot) * s.bytes;
      if (save) k::copy_kernel(snap + s.off, slot, s.bytes, cd.st);
      else k::copy_kernel(slot, snap + s.off, s.bytes, cd.st);
    }
  }
}

void ResidentModel::encode_image(const vision::Patches& p, const vision::ImageBlock& block, std::vector<uint16_t>& rows) {
  if (!vision_) throw std::runtime_error("no vision tower loaded");
  std::lock_guard lk(vision_->mu);
  rt::DeviceGuard g(vision_->dev);
  const size_t n = block.types.size();
  if (n * size_t(c_.dim) * 2 > vision_->out.size()) throw std::runtime_error("image block too long");
  model::vision_encode(c_, vision_->w, vision_->s, p, block, vision_->out.as<uint16_t>(), vision_->st);
  rows.resize(n * size_t(c_.dim));
  UH_CHECK(hipMemcpyAsync(rows.data(), vision_->out.data(), rows.size() * 2, hipMemcpyDeviceToHost, vision_->st));
  UH_CHECK(hipStreamSynchronize(vision_->st));
}

// Verify-chunk ring backups (before the layers) and restores of rejected positions, on card k.
void ResidentModel::ring_ops(int k, hipStream_t st, const std::vector<RingBackup>* backup,
                             const std::vector<RingRestore>* restore) {
  const auto* descs = ring_desc_[k].as<k::RingDesc>();
  const size_t per_slot = size_t(c_.dspark_block + 1) * ring_pos_bytes_[k];
  uint8_t* bak = ring_bak_[k].as<uint8_t>();
  if (backup)
    for (const RingBackup& b : *backup) {
      if (b.n > c_.dspark_block + 1 || b.slot < 0 || b.slot >= geo_.max_slots) throw std::runtime_error("step: bad backup");
      k::ring_backup(descs, n_rings_[k], b.slot, b.s0, b.n, bak + size_t(b.slot) * per_slot, ring_pos_bytes_[k], st);
    }
  if (restore)
    for (const RingRestore& r : *restore) {
      if (r.to - r.s0 > c_.dspark_block + 1 || r.from < r.s0 || r.slot < 0 || r.slot >= geo_.max_slots)
        throw std::runtime_error("draft: bad restore");
      k::ring_restore(descs, n_rings_[k], r.slot, r.s0, r.from, r.to, bak + size_t(r.slot) * per_slot,
                      ring_pos_bytes_[k], st);
    }
}

// Restores first (card 0 on its stream, ahead of its next step; card 1 on the draft stream), then the
// draft blocks through the stages and the head on card 1's draft stream, its own scratch and APU
// channel — alongside whatever step card 1 is running for other sequences.
void ResidentModel::draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs,
                          std::vector<int32_t>& tokens, std::vector<float>& conf) {
  tokens.clear();
  conf.clear();
  if (!dspark_) throw std::runtime_error("draft: no drafter loaded");
  if (!restore.empty()) {
    rt::DeviceGuard g0(card_[0].dev);
    ring_ops(0, card_[0].st, nullptr, &restore);
  }
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  if (!restore.empty()) ring_ops(1, dst_, nullptr, &restore);
  const int R = int(reqs.size()), Bk = c_.dspark_block, T = R * Bk, D = c_.dim, HC = c_.hc_mult;
  if (R > geo_.max_slots) throw std::runtime_error("draft: too many requests");
  if (R > 0) {
    int32_t* ids = d_io_host_;
    int32_t* pos = ids + T;
    int32_t* tok = pos + T;
    model::DraftDesc dd;
    dd.T = T;
    for (int r = 0; r < R; ++r) {
      dd.slots.push_back(reqs[size_t(r)].slot);
      dd.start.push_back(reqs[size_t(r)].start);
      tok[r * (Bk + 1)] = reqs[size_t(r)].token;
      for (int i = 0; i < Bk; ++i) {
        const int32_t id = i == 0 ? reqs[size_t(r)].token : c_.dspark_noise_token;
        ids[r * Bk + i] = id;
        pos[r * Bk + i] = reqs[size_t(r)].start + 1 + i;
        const uint16_t* e = embed_host_.data() + size_t(id) * D;
        for (int h = 0; h < HC; ++h) std::memcpy(d_x_host_ + (size_t(r * Bk + i) * HC + h) * D, e, size_t(D) * 2);
      }
    }
    k::copy_kernel(d_x_.data(), d_x_host_, size_t(T) * HC * D * 2, dst_);
    k::copy_kernel(d_in_.data(), ids, size_t(T) * 2 * 4, dst_);
    k::copy_kernel(d_tok_.data(), tok, size_t(R) * (Bk + 1) * 4, dst_);
    dd.ids = d_in_.as<int32_t>();
    dd.pos = d_in_.as<int32_t>() + T;
    for (size_t s = 0; s < dw_.stages.size(); ++s)
      model::dspark_stage(c_, dw_.stages[s], dcache_[s], dscr_, cd.rope, d_x_.as<uint16_t>(), dd, dst_, dchan_.get());
    model::dspark_head(c_, dw_, dscr_, d_x_.as<uint16_t>(), head_.as<uint16_t>(), R, d_logits_.as<float>(),
                       d_tok_.as<int32_t>(), d_conf_.as<float>(), dst_);
    k::copy_kernel(tok, d_tok_.data(), size_t(R) * (Bk + 1) * 4, dst_);
    k::copy_kernel(d_conf_host_, d_conf_.data(), size_t(T) * 4, dst_);
  }
  UH_CHECK(hipStreamSynchronize(dst_));
  const int32_t* tok = d_io_host_ + 2 * T;
  for (int r = 0; r < R; ++r)
    for (int i = 1; i <= Bk; ++i) tokens.push_back(tok[r * (Bk + 1) + i]);
  conf.assign(d_conf_host_, d_conf_host_ + T);
}

ResidentModel::~ResidentModel() {
  if (vision_ && vision_->st) {
    rt::DeviceGuard g(vision_->dev);
    (void)hipStreamDestroy(vision_->st);
  }
  if (dst_) {
    rt::DeviceGuard g(card_[1].dev);
    (void)hipStreamDestroy(dst_);
  }
  for (void* p : {static_cast<void*>(cap_host_[0]), static_cast<void*>(cap_host_[1]), static_cast<void*>(d_x_host_),
                  static_cast<void*>(d_io_host_), static_cast<void*>(d_conf_host_)})
    if (p) (void)hipHostFree(p);
  for (uint8_t* p : snap_host_)
    if (p) (void)hipHostFree(p);
  for (auto& cd : card_)
    if (cd.st) {
      rt::DeviceGuard g(cd.dev);
      (void)hipStreamDestroy(cd.st);
    }
  for (Flight& f : flight_) {
    for (void* p : {static_cast<void*>(f.x_host), static_cast<void*>(f.logits_host), static_cast<void*>(f.ids_host),
                    static_cast<void*>(f.tables_host), static_cast<void*>(f.vis_host), static_cast<void*>(f.img_host)})
      if (p) (void)hipHostFree(p);
    for (hipEvent_t e : {f.handoff, f.handoff2, f.consumed, f.done})
      if (e) (void)hipEventDestroy(e);
  }
}

std::vector<std::vector<uint64_t>> ResidentModel::routing_histogram() const {
  std::vector<std::vector<uint64_t>> h(size_t(c_.n_layers), std::vector<uint64_t>(size_t(c_.n_experts), 0));
  std::vector<uint32_t> buf(size_t(c_.n_layers) * c_.n_experts);
  for (const auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemcpy(buf.data(), cd.scratch.route_hist.data(), buf.size() * 4, hipMemcpyDeviceToHost));
    for (size_t L = 0; L < h.size(); ++L)
      for (size_t e = 0; e < h[L].size(); ++e) h[L][e] += buf[L * c_.n_experts + e];
  }
  return h;
}

void ResidentModel::reset_routing_histogram() {
  for (auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemset(cd.scratch.route_hist.data(), 0, cd.scratch.route_hist.size()));
  }
}

void ResidentModel::set_block_tables(const std::vector<int32_t>& tables) {
  if (tables.size() != size_t(geo_.max_slots) * geo_.max_pages_per_seq)
    throw std::runtime_error("set_block_tables: wrong size");
  tables_ = tables;  // uploaded on the card streams by the next submit (steps in flight keep theirs)
  tables_dirty_ = true;
}

std::vector<const void*> ResidentModel::layer_ptrs(int k) const {
  std::vector<const void*> v;
  for (const auto& w : card_[k].w) {
    for (const auto* t : {&w.hc_attn_fn, &w.hc_attn_base, &w.hc_attn_scale, &w.hc_ffn_fn, &w.hc_ffn_base,
                          &w.hc_ffn_scale, &w.attn_norm, &w.ffn_norm, &w.sink, &w.q_norm, &w.kv_norm, &w.gate_w,
                          &w.gate_bias})
      v.push_back(t->ptr);
    for (const auto* l : {&w.wq_a, &w.wq_b, &w.wkv, &w.wo_a, &w.wo_b, &w.sw1, &w.sw2, &w.sw3}) {
      v.push_back(l->w.ptr);
      v.push_back(l->s.ptr);
    }
    v.push_back(w.exl3_tab.data());
  }
  return v;
}

uint64_t ResidentModel::submit(const StepInput& in) {
  static const bool check_ptrs = std::getenv("UNHOLY_CHECK_PTRS") != nullptr;
  if (check_ptrs) {
    if (ptr_snapshot_.empty())
      for (int k = 0; k < 2; ++k) ptr_snapshot_.push_back(layer_ptrs(k));
    for (int k = 0; k < 2; ++k) {
      const auto now = layer_ptrs(k);
      for (size_t i = 0; i < now.size(); ++i)
        if (now[i] != ptr_snapshot_[size_t(k)][i]) {
          std::fprintf(stderr, "CHECK_PTRS: step %llu card %d slot %zu changed %p -> %p\n",
                       static_cast<unsigned long long>(steps_), k, i, ptr_snapshot_[size_t(k)][i], now[i]);
          std::abort();
        }
    }
  }
  ++steps_;
  static const bool timing = std::getenv("UNHOLY_STEP_TIMING") != nullptr;
  const auto t_begin = std::chrono::steady_clock::now();
  const int T = int(in.ids.size()), R = int(in.logit_rows.size());
  if (T == 0 || T > max_tokens_) throw std::runtime_error("step: bad token count");
  if (R > max_logit_rows_) throw std::runtime_error("step: too many logit rows");
  const uint64_t ticket = next_ticket_;
  Flight& f = flight_[ticket % 2];
  if (f.busy || (inflight_ == 1 && flight_[(ticket + 1) % 2].busy))
    throw std::runtime_error("step: too many steps in flight");
  const size_t row = size_t(c_.hc_mult) * c_.dim;
  // Inputs into this flight's pinned staging: ids, positions, logit rows.
  std::memcpy(f.ids_host, in.ids.data(), size_t(T) * 4);
  std::memcpy(f.ids_host + max_tokens_, in.pos.data(), size_t(T) * 4);
  std::memcpy(f.ids_host + 2 * max_tokens_, in.logit_rows.data(), size_t(R) * 4);
  const bool tables = tables_dirty_;
  if (tables) std::memcpy(f.tables_host, tables_.data(), tables_.size() * 4);
  tables_dirty_ = false;
  // Image spans: every row's window range (model.py get_image_visible / get_window_topk_idxs_visible);
  // image rows into this flight's pinned buffer.
  const bool vis = !in.image_spans.empty();
  if (vis) {
    const int max_img = std::max(c_.vision_max_tokens, 1);
    for (const Chunk& ch : in.chunks)
      for (int i = 0; i < ch.n; ++i) {
        const int p = ch.s0 + i;
        int lo = std::max(0, p - (c_.window - 1)), hi = p;
        for (const ImageSpan& sp : in.image_spans)
          if (sp.slot == ch.slot && p >= sp.start && p <= sp.end) {
            const int left = std::min(p - sp.start, max_img - 1), right = std::min(sp.end - p, max_img);
            lo = std::max(0, p - (c_.window - 1) - std::max(0, left - (c_.window - 1)));
            hi = p + right;
          }
        f.vis_host[2 * (ch.t0 + i)] = lo;
        f.vis_host[2 * (ch.t0 + i) + 1] = hi;
      }
  }
  {
    size_t off = 0;
    for (const ImageRows& im : in.image_rows) {
      if (im.t0 < 0 || im.t0 + im.n > T) throw std::runtime_error("step: image rows outside the batch");
      std::memcpy(f.img_host + off, im.rows, size_t(im.n) * c_.dim * 2);
      off += size_t(im.n) * c_.dim;
    }
  }
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    // Staging moves by kernels pulling from / pushing to pinned host memory (DECISIONS 2026-10-12).
    k::copy_kernel(cd.ids.data(), f.ids_host, size_t(T) * 4, cd.st);
    k::copy_kernel(cd.pos.data(), f.ids_host + max_tokens_, size_t(T) * 4, cd.st);
    if (tables) k::copy_kernel(cd.tables.data(), f.tables_host, tables_.size() * 4, cd.st);
    if (vis) k::copy_kernel(cd.vis.data(), f.vis_host, size_t(T) * 2 * 4, cd.st);
  }
  // DSpark: the rows whose target hidden states feed the drafter's main-KV (each chunk's last <= 128).
  int ncap = 0;
  if (dspark_) {
    int32_t* cap = cap_host_[ticket % 2];
    for (const Chunk& ch : in.chunks)
      for (int t = ch.t0 + std::max(0, ch.n - 128); t < ch.t0 + ch.n; ++t) {
        if (ncap == cap_max_) throw std::runtime_error("step: too many dspark capture rows");
        cap[ncap] = t;
        cap[cap_max_ + ncap] = ch.slot;
        cap[2 * cap_max_ + ncap] = ch.s0 + (t - ch.t0);
        ++ncap;
      }
    rt::DeviceGuard g(card_[1].dev);
    k::copy_kernel(cap_dev_.data(), cap, size_t(cap_max_) * 3 * 4, card_[1].st);
  }
  if (!in.backup.empty() && !dspark_) throw std::runtime_error("step: verify chunks without a drafter");
  // Two decode micro-batches within a step (UNHOLY_MICROBATCH=1; off by default): a step of decode / verify
  // chunks of distinct sequences runs as two halves of its chunks, and card 0 hands the first half to card 1
  // before computing the second, so the cards work on different halves at once. Every kernel path is chosen
  // by the whole step's rows (model::ClassRows) and a row's arithmetic within a path does not depend on the
  // other rows, so each row is computed exactly as in the unsplit step (bench --out: bit-identical). Slower
  // here, though: decode reads weights, each half reads all of them again (DECISIONS 2026-10-15).
  struct Part {
    int c0, c1, r0, r1, cap0, cap1;  // chunks, rows, DSpark capture entries
  };
  std::vector<Part> parts;
  {
    bool split = microbatch_ && in.chunks.size() >= 2 && in.image_rows.empty() && in.image_spans.empty();
    for (size_t i = 0; split && i < in.chunks.size(); ++i) {
      split = in.chunks[i].n <= c_.dspark_block + 1;
      for (size_t j = 0; split && j < i; ++j) split = in.chunks[j].slot != in.chunks[i].slot;
    }
    const int nch = int(in.chunks.size());
    if (split) {
      const int ca = (nch + 1) / 2, ra = in.chunks[size_t(ca)].t0;
      int capa = 0;
      while (capa < ncap && cap_host_[ticket % 2][capa] < ra) ++capa;
      parts.push_back({0, ca, 0, ra, 0, capa});
      parts.push_back({ca, nch, ra, T, capa, ncap});
    } else {
      parts.push_back({0, nch, 0, T, 0, ncap});
    }
  }
  std::chrono::steady_clock::time_point t_enq[2];
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    if (k == 0) {
      k::embed_hc(cd.ids.as<int32_t>(), embed_.as<uint16_t>(), cd.x.as<uint16_t>(), T, c_.dim, c_.hc_mult, cd.st,
                  c_.vocab);
      // Image blocks: their rows (staged in this flight's pinned buffer) over the zero embeddings.
      size_t off = 0;
      for (const ImageRows& im : in.image_rows) {
        k::put_rows_hc(f.img_host + off, im.n, cd.x.as<uint16_t>(), im.t0, c_.dim, c_.hc_mult, cd.st);
        off += size_t(im.n) * c_.dim;
      }
    }
    state_copies(k, in.restore, false);
    if (!in.backup.empty()) ring_ops(k, cd.st, &in.backup, nullptr);
    const int nt = int(c_.dspark_target_layers.size());
    const auto* cap_rows = cap_dev_.as<int32_t>();
    for (size_t pi = 0; pi < parts.size(); ++pi) {
      const Part& pt = parts[pi];
      const size_t xoff = size_t(pt.r0) * row;
      if (k == 1) {  // residual from card 0 through this flight's pinned buffer, ordered by events
        UH_CHECK(hipStreamWaitEvent(cd.st, pi == 0 ? f.handoff : f.handoff2, 0));
        k::copy_kernel(cd.x.as<uint16_t>() + xoff, f.x_host + xoff, size_t(pt.r1 - pt.r0) * row * 2, cd.st);
        if (pi + 1 == parts.size()) UH_CHECK(hipEventRecord(f.consumed, cd.st));
      }
      model::BatchDesc b;
      for (int c = pt.c0; c < pt.c1; ++c) {
        const Chunk& ch = in.chunks[size_t(c)];
        b.chunks.push_back({ch.slot, ch.s0, ch.n, ch.t0 - pt.r0});
      }
      b.T = pt.r1 - pt.r0;
      b.max_pages_per_seq = geo_.max_pages_per_seq;
      b.ids = cd.ids.as<int32_t>() + pt.r0;
      b.pos = cd.pos.as<int32_t>() + pt.r0;
      b.block_tables = cd.tables.as<int32_t>();
      b.vis = in.image_spans.empty() ? nullptr : cd.vis.as<int32_t>() + 2 * pt.r0;
      const model::ClassRows whole_step(parts.size() > 1 ? T : 0);  // 0: no override
      for (size_t i = 0; i < cd.layers.size(); ++i) {
        model::layer_forward(c_, cd.w[i], cd.cache[i], cd.scratch, cd.rope, cd.x.as<uint16_t>() + xoff, b, cd.st,
                             chan_[k].get());
        for (int j = 0; dspark_ && pt.cap1 > pt.cap0 && j < nt; ++j)
          if (cd.layers[i] == c_.dspark_target_layers[size_t(j)])
            k::hc_mean_rows(cd.x.as<uint16_t>(), c_.hc_mult, c_.dim, cap_rows + pt.cap0, pt.cap1 - pt.cap0,
                            mh_.as<uint16_t>() + size_t(pt.cap0) * nt * c_.dim + size_t(j) * c_.dim, nt * c_.dim, cd.st);
      }
      if (k == 0) {
        // The flight's buffer may still be read by card 1 for the step two tickets back.
        if (pi == 0) UH_CHECK(hipStreamWaitEvent(cd.st, f.consumed, 0));
        k::copy_kernel(f.x_host + xoff, cd.x.as<uint16_t>() + xoff, size_t(pt.r1 - pt.r0) * row * 2, cd.st);
        UH_CHECK(hipEventRecord(pi == 0 ? f.handoff : f.handoff2, cd.st));
      }
    }
    if (dspark_ && k == 1)
      model::dspark_main_kv(c_, dw_, dcache_, cd.scratch, cd.rope, mh_.as<uint16_t>(), cap_rows + cap_max_,
                            cap_rows + 2 * cap_max_, ncap, cd.st);
    state_copies(k, in.save, true);
    if (timing) t_enq[k] = std::chrono::steady_clock::now();
  }
  // Head on card 1 for the requested rows.
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  if (R > 0) {
    k::copy_kernel(rows_idx_.data(), f.ids_host + 2 * max_tokens_, size_t(R) * 4, cd.st);
    k::gather_rows(cd.x.as<uint16_t>(), rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, int(row), cd.st);
    k::hc_head(hrows_.as<uint16_t>(), hc_fn_.as<float>(), hc_scale_.as<float>(), hc_base_.as<float>(),
               hb_.as<uint16_t>(), R, c_.hc_mult, c_.dim, c_.norm_eps, c_.hc_eps, cd.st);
    k::rmsnorm(hb_.as<uint16_t>(), c_.dim, norm_.as<uint16_t>(), hb_.as<uint16_t>(), c_.dim, R, c_.dim, c_.norm_eps,
               cd.st);
    // The reference head is an fp32 linear on the bf16-valued normed state: bf16 WMMA, exact products.
    if (R <= k::gemv_max_rows())
      k::gemv_bf16_f32out(hb_.as<uint16_t>(), c_.dim, head_.as<uint16_t>(), logits_.as<float>(), c_.vocab, R,
                          c_.vocab, c_.dim, cd.st);
    else
      k::gemm_bf16_f32out_wmma(hb_.as<uint16_t>(), c_.dim, head_.as<uint16_t>(), logits_.as<float>(), c_.vocab, R,
                               c_.vocab, c_.dim, cd.st);
    k::copy_kernel(f.logits_host, logits_.data(), size_t(R) * c_.vocab * 4, cd.st);
  }
  UH_CHECK(hipEventRecord(f.done, cd.st));
  if (timing)
    std::fprintf(stderr, "submit T=%d: enqueue card0 %.1f ms, card1 %.1f ms\n", T,
                 std::chrono::duration<double, std::milli>(t_enq[0] - t_begin).count(),
                 std::chrono::duration<double, std::milli>(t_enq[1] - t_enq[0]).count());
  f.busy = true;
  f.ticket = ticket;
  f.rows = R;
  return next_ticket_++;
}

void ResidentModel::wait(uint64_t ticket, std::vector<float>& out) {
  Flight& f = flight_[ticket % 2];
  if (!f.busy || f.ticket != ticket) throw std::runtime_error("step: wait for an unknown ticket");
  {
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventSynchronize(f.done));
  }
  if (canary_ && canary_->count != canary_seen_) {
    const unsigned now = canary_->count;
    for (unsigned i = canary_seen_; i < now && i < kernels::HcCanaryLog::kMax; ++i) {
      const auto& e = canary_->ev[i];
      std::fprintf(stderr,
                   "canary: hc_pre_tiled load address %#lx, expected %#lx (xor %#lx) at k0 %u i %u; HW_ID1 %#x: SE %u SA %u "
                   "WGP %u SIMD %u wave %u\n",
                   (unsigned long)e.got, (unsigned long)e.want, (unsigned long)(e.got ^ e.want), e.where >> 16,
                   e.where & 0xffff, e.hw_id1, (e.hw_id1 >> 18) & 7, (e.hw_id1 >> 16) & 1, (e.hw_id1 >> 10) & 15,
                   (e.hw_id1 >> 8) & 3, e.hw_id1 & 31);
    }
    if (now > kernels::HcCanaryLog::kMax) std::fprintf(stderr, "canary: %u mismatches in all\n", now);
    canary_seen_ = now;
  }
  out.resize(size_t(f.rows) * c_.vocab);
  std::memcpy(out.data(), f.logits_host, out.size() * 4);
  f.busy = false;
}

}  // namespace unholy::engine
