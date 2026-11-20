#include "engine/hy4_resident.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "kernels/elem_fast.hpp"
#include "kernels/glm_fast.hpp"
#include "kernels/ref.hpp"
#include "model/loader.hpp"
#include "rt/hip.hpp"

namespace unholy::engine {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// Hy4ApuExperts

size_t Hy4ApuExperts::work_bytes(const model::Hy4Config& c, size_t cap) {
  const size_t Lt = c.latent, I = c.moe_inter;
  return cap * Lt * 2 + cap * 2 * I * 2 + cap * I * 2 + cap * Lt * 2;
}

size_t Hy4ApuExperts::staging_bytes(const model::Hy4Config& c, size_t cap) {
  const size_t tokens = cap / size_t(c.n_active), Lt = c.latent;
  return tokens * Lt * (2 + 4) + cap * 4 * 4 + sizeof(k::MoePlan) +
         size_t(model::hy4_max_tiles(c, int(tokens))) * sizeof(k::Exl3Tile);
}

Hy4ApuExperts::Hy4ApuExperts(const model::Hy4Config& c, int dev, size_t capacity_rows, std::shared_ptr<Layers> layers)
    : c_(c), dev_(dev), layers_(std::move(layers)) {
  rt::DeviceGuard g(dev);
  this->capacity_rows = capacity_rows;
  const size_t Lt = c.latent, I = c.moe_inter, cap = capacity_rows, tokens = cap / size_t(c.n_active);
  max_tiles = model::hy4_max_tiles(c, int(tokens));
  const unsigned flags = hipHostMallocPortable | hipHostMallocMapped;
  const auto pinned = [&](auto** host, auto** dev_ptr, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(host), bytes, flags));
    if (dev_ptr) UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(dev_ptr), *host, 0));
  };
  pinned(&in_host, &in_dev_, tokens * Lt * 2);    // compact token latent rows (bf16)
  pinned(&out_host, &out_dev_, tokens * Lt * 4);  // per-token partial sums (fp32)
  pinned(&w_host, &w_dev_, cap * 4);
  pinned(&e_host, &e_dev_, cap * 4);
  pinned(&p2u_host, &p2u_dev_, cap * 4);
  pinned(&u2p_host, &u2p_dev_, cap * 4);
  pinned(&plan_host, &plan_dev_, sizeof(k::MoePlan));
  pinned(&tiles_host, &tiles_dev_, size_t(max_tiles) * sizeof(k::Exl3Tile));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&req_flag_), 64, flags | hipHostMallocCoherent));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&done_flag_), 64, flags | hipHostMallocCoherent));
  *req_flag_ = 0;
  *done_flag_ = 0;
  UH_CHECK(hipStreamCreateWithFlags(&st_, hipStreamNonBlocking));
  xa_ = rt::DeviceBuffer(dev, cap * Lt * 2);
  gu_ = rt::DeviceBuffer(dev, cap * 2 * I * 2);
  eh_ = rt::DeviceBuffer(dev, cap * I * 2);
  pout_ = rt::DeviceBuffer(dev, cap * Lt * 2);
  if (layers_->size() < size_t(c.n_layers)) layers_->resize(size_t(c.n_layers));
}

Hy4ApuExperts::~Hy4ApuExperts() {
  rt::DeviceGuard g(dev_);
  if (st_) (void)hipStreamDestroy(st_);
  for (void* p : {static_cast<void*>(in_host), static_cast<void*>(out_host), static_cast<void*>(w_host),
                  static_cast<void*>(e_host), static_cast<void*>(plan_host), static_cast<void*>(tiles_host),
                  static_cast<void*>(p2u_host), static_cast<void*>(u2p_host), static_cast<void*>(req_flag_),
                  static_cast<void*>(done_flag_)})
    if (p) (void)hipHostFree(p);
}

void Hy4ApuExperts::add_layer(model::Hy4LayerWeights w) {
  const int L = w.layer;
  if (layers_->size() <= size_t(L)) layers_->resize(size_t(L) + 1);
  layers_->at(size_t(L)) = std::make_unique<model::Hy4LayerWeights>(std::move(w));
}

void Hy4ApuExperts::run(int, const std::vector<Job>&) {
  throw std::runtime_error("Hy4ApuExperts: host-synchronous jobs are not supported");
}

// The card stream raises req after staging; the APU stream waits for it, expands the compact token rows into its pair
// rows, runs the remote tiles (gate / up, clamped SwiGLU, down, routing weights), sums each token's pair rows into one
// fp32 row of out_host and raises done; join() makes the card stream wait for done.
void Hy4ApuExperts::enqueue(int layer, hipStream_t card_st, int P) {
  const auto* w = layers_->at(size_t(layer)).get();
  if (!w) throw std::runtime_error("Hy4ApuExperts: no experts of layer " + std::to_string(layer) + " on the APU");
  ++seq_;
  UH_CHECK(hipStreamWriteValue32(card_st, req_flag_, seq_, 0));
  rt::DeviceGuard g(dev_);
  UH_CHECK(hipStreamWaitValue32(st_, req_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
  const int Lt = c_.latent;
  uint16_t* xa = xa_.as<uint16_t>();
  uint16_t* pout = pout_.as<uint16_t>();
  k::hy4_gather_remote(in_dev_, p2u_dev_, plan_dev_, xa, Lt, P, st_);
  model::hy4_experts_dev(c_, *w, xa, w_dev_, tiles_dev_, max_tiles, &plan_dev_->n_remote_tiles, P, gu_.as<uint16_t>(),
                         eh_.as<uint16_t>(), pout, st_, &plan_dev_->remote0, nullptr);
  k::glm_remote_partial(pout, u2p_dev_, &plan_dev_->n_remote_tokens, reinterpret_cast<float*>(out_dev_),
                        c_.n_active, Lt, P / c_.n_active, st_);
  UH_CHECK(hipStreamWriteValue32(st_, done_flag_, seq_, 0));
}

void Hy4ApuExperts::join(hipStream_t card_st) {
  UH_CHECK(hipStreamWaitValue32(card_st, done_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
}

// ---------------------------------------------------------------------------------------------------
// Plan

Plan make_hy4_plan(const Hy4PlanInput& in) {
  const auto& c = *in.cfg;
  const auto& ck = *in.ck;
  const auto devs = query_devices();
  Plan pl;
  int ncard = 0;
  for (const auto& d : devs) {
    if (d.integrated) pl.apu = d.id;
    else if (ncard < 2) pl.card[ncard++] = d.id;
  }
  if (ncard < 2 || pl.apu < 0) throw std::runtime_error("plan: need two discrete cards and the APU");
  const auto info = [&](int id) -> const DeviceInfo& { return devs[size_t(id)]; };

  std::vector<size_t> dense(size_t(c.n_layers)), cache(size_t(c.n_layers));
  size_t total = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    dense[size_t(L)] = model::hy4_dense_layer_bytes(ck, c, L, true);
    cache[size_t(L)] = model::hy4_cache_bytes(c, c.is_sparse(L), in.geo.max_slots, in.geo.num_pages, in.dfly);
    total += dense[size_t(L)] + cache[size_t(L)];
  }
  pl.layer_card.assign(size_t(c.n_layers), 1);
  size_t acc = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    const size_t b = dense[size_t(L)] + cache[size_t(L)];
    if (acc + b / 2 > total / 2) break;
    pl.layer_card[size_t(L)] = 0;
    acc += b;
  }
  for (int L = 0; L < c.n_layers; ++L) {
    const int kc = pl.layer_card[size_t(L)];
    pl.dense[kc] += dense[size_t(L)];
    pl.cache[kc] += cache[size_t(L)];
  }
  pl.dense[0] += ck.at("model.embed_tokens.weight").nbytes;
  pl.dense[1] += ck.at("lm_head.weight").nbytes + ck.at("model.norm.weight").nbytes;
  if (in.dfly) {  // the drafter on card 1: weights + the embedding table, scratch, hidden-state rings, head rows
    const int reqs = std::max(in.geo.max_slots, 1);
    pl.dense[1] += model::hy4_dfly_bytes(ck) + model::hy4_dfly_scratch_bytes(c, reqs) +
                   size_t(in.geo.max_slots) * model::kHy4MaxVerify * c.dim * 2 +
                   size_t(in.max_logit_rows > 0 ? in.max_logit_rows : in.max_tokens) * c.dim * 2;
  }
  const int logit_rows = in.max_logit_rows > 0 ? in.max_logit_rows : in.max_tokens;
  for (int kc = 0; kc < 2; ++kc) {
    pl.scratch[kc] = model::hy4_scratch_bytes(c, in.max_tokens, in.max_seq) +
                     size_t(in.max_tokens) * c.hc * c.dim * 2 + size_t(in.max_tokens) * 16 +
                     size_t(in.geo.max_slots) * in.geo.max_pages_per_seq * 4 +
                     (kc == 1 ? size_t(logit_rows) * (c.vocab * 4 + size_t(c.hc) * c.dim * 2) : 0);
    const size_t used = pl.dense[kc] + pl.cache[kc] + pl.scratch[kc] + in.card_margin;
    const size_t free = info(pl.card[kc]).free_bytes;
    if (used > free) throw std::runtime_error("plan: card " + std::to_string(kc) + " cannot hold its dense layers");
    pl.budget[kc] = free - used;
  }

  struct Cand {
    double share;
    int layer, expert;
  };
  const size_t eb = model::hy4_expert_bytes(c, in.nvfp4   ? model::Hy4ExpertFormat::Nvfp4
                                              : in.mxfp4 ? model::Hy4ExpertFormat::Mxfp4
                                                         : model::Hy4ExpertFormat::Fp8);
  std::vector<Cand> cands[2];
  double layers_moe = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    if (c.is_dense(L)) continue;
    double sum = 0;
    for (double v : in.usage[size_t(L)]) sum += v;
    for (int e = 0; e < c.n_experts; ++e)
      cands[pl.layer_card[size_t(L)]].push_back({in.usage[size_t(L)][size_t(e)] / sum, L, e});
    layers_moe += 1.0;
  }
  pl.on_card.assign(size_t(c.n_layers), std::vector<uint8_t>(size_t(c.n_experts), 0));
  double mass_cards = 0;
  for (int kc = 0; kc < 2; ++kc) {
    std::stable_sort(cands[kc].begin(), cands[kc].end(), [](const Cand& a, const Cand& b) { return a.share > b.share; });
    for (const auto& cd : cands[kc]) {
      if (pl.experts[kc] + eb > pl.budget[kc]) continue;
      pl.experts[kc] += eb;
      pl.on_card[size_t(cd.layer)][size_t(cd.expert)] = 1;
      mass_cards += cd.share;
    }
  }
  for (int kc = 0; kc < 2; ++kc)
    for (const auto& cd : cands[kc])
      if (!pl.on_card[size_t(cd.layer)][size_t(cd.expert)]) pl.apu_experts += eb;
  pl.routed_on_cards = mass_cards / std::max(layers_moe, 1.0);
  pl.apu_free = info(pl.apu).free_bytes;
  const size_t cap = size_t(in.max_tokens) * c.n_active;
  size_t apu_work = 2 * (Hy4ApuExperts::work_bytes(c, cap) + Hy4ApuExperts::staging_bytes(c, cap));
  if (in.vision) {  // the vision tower: weights, a scratch for the largest image, its rows
    const auto vc = model::Hy4VisionConfig::from_hf(ck.config());
    apu_work += model::hy4_vision_weight_bytes(ck) + model::hy4_vision_scratch_bytes(vc, vc.max_patches) +
                size_t(vc.max_patches) * vc.out_dim * 2;
  }
  const size_t apu_need = pl.apu_experts + apu_work + in.apu_margin;
  if (apu_need > pl.apu_free) {
    char msg[256];
    std::snprintf(msg, sizeof msg, "plan: the APU cannot hold the cold experts (%.2f GB + %.2f GB working + %.2f GB "
                  "margin > %.2f GB free)", pl.apu_experts / 1e9, apu_work / 1e9, in.apu_margin / 1e9,
                  pl.apu_free / 1e9);
    throw std::runtime_error(msg);
  }
  if (in.snapshots > 0) {
    pl.snapshot_bytes = hy4_snapshot_bytes(c, pl, 0) + hy4_snapshot_bytes(c, pl, 1) + 4096;
    const size_t left = pl.apu_free - apu_need;
    pl.snapshots = int(std::min<size_t>(size_t(in.snapshots), left / pl.snapshot_bytes));
  }
  return pl;
}

size_t hy4_snapshot_bytes(const model::Hy4Config& c, const Plan& p, int kc) {
  size_t b = 0;
  for (int L = 0; L < c.n_layers; ++L)
    if (p.layer_card[size_t(L)] == kc) b += c.is_sparse(L) ? size_t(c.idx_pool) * c.idx_dim * 2
                                                           : size_t(k::kHy4Window) * k::kHy4Row * 2;
  return (b + 4095) / 4096 * 4096;
}

// ---------------------------------------------------------------------------------------------------
// Hy4ResidentModel

struct Hy4ResidentModel::Vision {
  int dev = 0;
  hipStream_t st = nullptr;
  std::mutex mu;
  model::Hy4VisionConfig c;
  model::Hy4VisionWeights w;
  model::Hy4VisionScratch s;
  rt::DeviceBuffer out;
  ~Vision() {
    rt::DeviceGuard g(dev);
    if (st) (void)hipStreamDestroy(st);
  }
};

Hy4ResidentModel::Hy4ResidentModel(const model::Hy4Config& c, const model::Checkpoint& ck,
                                   const std::string& engram_dir, const Plan& plan, const model::CacheGeometry& geo,
                                   int max_tokens, int max_seq, int max_logit_rows, int snapshots, bool mxfp4,
                                   const model::Checkpoint* experts_gguf, bool dfly, bool vision)
    : c_(c), plan_(plan), geo_(geo), max_tokens_(max_tokens),
      max_logit_rows_(max_logit_rows > 0 ? std::min(max_logit_rows, max_tokens) : max_tokens),
      snapshots_(snapshots) {
  if (geo.max_pages_per_seq * k::kHy4PageRows < max_seq) throw std::runtime_error("hy4: block table too narrow");
  engram_ = std::make_unique<model::Hy4Engram>(engram_dir, c);
  last2_.assign(size_t(geo.max_slots) * 2, -1);
  dfly_ = dfly;
  std::vector<uint8_t> dfly_layer(size_t(c.n_layers), 0);
  if (dfly_) {
    if (!model::hy4_has_dfly(ck)) throw std::runtime_error("hy4: the checkpoint has no DFly2 drafter");
    const auto* tl = ck.config().find("dfly_target_kv_layers");
    if (!tl) throw std::runtime_error("hy4: config has no dfly_target_kv_layers");
    for (const auto& e : tl->items()) {
      const int L = int(e.as_double());
      if (L < 0 || L >= c.n_layers || plan.layer_card[size_t(L)] != 1)
        throw std::runtime_error("hy4: DFly2's target layers must be on card 1");
      dfly_layer[size_t(L)] = 1;
    }
    vs0_.assign(size_t(geo.max_slots), -1);
    vids_.assign(size_t(geo.max_slots), {});
    vprev_.assign(size_t(geo.max_slots) * 2, -1);
  }
  model::Uploader up(ck);
  std::unique_ptr<model::Uploader> gup;
  if (experts_gguf) gup = std::make_unique<model::Uploader>(*experts_gguf);
  {
    auto layers = std::make_shared<Hy4ApuExperts::Layers>();
    for (auto& ch : chan_) ch = std::make_unique<Hy4ApuExperts>(c, plan.apu, size_t(max_tokens) * c.n_active, layers);
  }
  for (int kc = 0; kc < 2; ++kc) {
    Card& cd = card_[kc];
    cd.dev = plan.card[kc];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipStreamCreateWithFlags(&cd.st, hipStreamNonBlocking));
    cd.scratch = model::make_hy4_scratch(c, cd.dev, max_tokens, max_seq);
    cd.x = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.hc * c.dim * 2);
    cd.ids = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.pos = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.pos3 = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 3 * 4);
    cd.tables = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
  }
  for (int L = 0; L < c.n_layers; ++L) {
    const int kc = plan.layer_card[size_t(L)];
    Card& cd = card_[kc];
    {
      rt::DeviceGuard g(cd.dev);
      model::Hy4LoadOptions lo;
      lo.expert_mask = &plan.on_card[size_t(L)];
      lo.mxfp4 = mxfp4;
      lo.experts_gguf = experts_gguf;
      lo.experts_up = gup.get();
      lo.up = &up;
      cd.w.push_back(model::load_hy4_layer(ck, c, L, cd.dev, cd.st, lo));
      cd.cache.push_back(model::make_hy4_cache(c, L, geo.max_slots, geo.num_pages, cd.dev, cd.st,
                                               dfly_layer[size_t(L)] != 0));
    }
    if (c.is_dense(L)) continue;
    std::vector<uint8_t> cold(size_t(c.n_experts));
    bool any = false;
    for (int e = 0; e < c.n_experts; ++e) any |= (cold[size_t(e)] = !plan.on_card[size_t(L)][size_t(e)]) != 0;
    if (!any) continue;
    rt::DeviceGuard ga(plan.apu);
    model::Hy4LoadOptions la;
    la.load_dense = false;
    la.expert_mask = &cold;
    la.mxfp4 = mxfp4;
    la.experts_gguf = experts_gguf;
    la.experts_up = gup.get();
    la.up = &up;
    chan_[0]->add_layer(model::load_hy4_layer(ck, c, L, plan.apu, chan_[0]->stream(), la));
  }
  {
    rt::DeviceGuard g(card_[0].dev);
    embed_ = model::load_hy4_head(ck, c, card_[0].dev, card_[0].st, true, false);
  }
  {
    Card& cd = card_[1];
    rt::DeviceGuard g(cd.dev);
    head_ = model::load_hy4_head(ck, c, cd.dev, cd.st, false, true);
    hrows_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * c.hc * c.dim * 2);
    logits_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * c.vocab * 4);
    rows_idx_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * 4);
    if (dfly_) {
      dw_ = model::load_hy4_dfly(ck, c, cd.dev, cd.st);
      dscr_ = model::make_hy4_dfly_scratch(c, cd.dev, geo.max_slots);
      UH_CHECK(hipStreamCreateWithFlags(&dst_, hipStreamNonBlocking));
      UH_CHECK(hipEventCreateWithFlags(&drafted_, hipEventDisableTiming));
      hidden_ = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * model::kHy4MaxVerify * c.dim * 2);
      UH_CHECK(hipMemset(hidden_.data(), 0, hidden_.size()));
      hn_rows_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * c.dim * 2);
      dtok_ = rt::DeviceBuffer(cd.dev, size_t(model::kHy4DraftTokens) * geo.max_slots * 8);  // tokens, then conf
      UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&dtok_host_),
                             size_t(model::kHy4DraftTokens) * geo.max_slots * 8, hipHostMallocPortable));
      for (auto& hc : cd.cache)
        if (hc.dsel.data()) dtarget_.push_back(&hc);
      if (dtarget_.size() != dw_.target.size()) throw std::runtime_error("hy4: DFly2 target caches");
      // in the drafter's layer order
      std::sort(dtarget_.begin(), dtarget_.end(), [](auto* a, auto* b) { return a->layer < b->layer; });
      for (size_t i = 0; i < dtarget_.size(); ++i)
        if (dtarget_[i]->layer != dw_.target[i]) throw std::runtime_error("hy4: DFly2 target layer order");
    }
  }
  if (vision) {
    vision_ = std::make_unique<Vision>();
    Vision& v = *vision_;
    v.dev = plan.apu;
    v.c = model::Hy4VisionConfig::from_hf(ck.config());
    if (v.c.depth <= 0) throw std::runtime_error("hy4: the config has no vision tower");
    if (v.c.out_dim != c.dim) throw std::runtime_error("hy4: the vision tower's output width is not the model's");
    rt::DeviceGuard g(v.dev);
    UH_CHECK(hipStreamCreateWithFlags(&v.st, hipStreamNonBlocking));
    v.w = model::load_hy4_vision(ck, v.c, v.dev, v.st);
    v.s = model::make_hy4_vision_scratch(v.c, v.c.max_patches, v.dev);
    v.out = rt::DeviceBuffer(v.dev, size_t(v.c.max_patches) * v.c.out_dim * 2);
  }
  const auto pinned = [](auto** p, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(p), std::max<size_t>(bytes, 64),
                           hipHostMallocPortable | hipHostMallocMapped));
  };
  for (Flight& f : flight_) {
    pinned(&f.x_host, size_t(max_tokens) * c.hc * c.dim * 2);
    pinned(&f.logits_host, size_t(max_logit_rows_) * c.vocab * 4);
    pinned(&f.io_host, size_t(max_tokens) * 3 * 4);
    pinned(&f.pos3_host, size_t(max_tokens) * 3 * 4);
    if (vision) pinned(&f.img_host, size_t(max_tokens) * c.dim * 2);
    pinned(&f.tables_host, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
    for (int m = 0; m < engram_->memories(); ++m) {
      pinned(&f.eng_q[m], size_t(max_tokens) * 16 * 96);
      pinned(&f.eng_s[m], size_t(max_tokens) * 16 * 2);
    }
    pinned(&f.eng_mask, size_t(max_tokens));
    {
      rt::DeviceGuard g(card_[0].dev);
      UH_CHECK(hipEventCreateWithFlags(&f.handoff, hipEventDisableTiming));
      UH_CHECK(hipEventCreateWithFlags(&f.saved, hipEventDisableTiming));
    }
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventCreateWithFlags(&f.consumed, hipEventDisableTiming));
    UH_CHECK(hipEventCreateWithFlags(&f.done, hipEventDisableTiming));
  }
  if (snapshots_ > 0) {  // at a page boundary the indexer tails hold no partial group; the rings the last 512 rows
    for (int kc = 0; kc < 2; ++kc) {
      size_t off = 0;
      for (const model::Hy4Cache& hc : card_[kc].cache) {
        const rt::DeviceBuffer& b = hc.sparse ? hc.tail : hc.ring;
        const size_t per = b.size() / size_t(geo.max_slots);
        snap_segs_[kc].push_back({static_cast<uint8_t*>(const_cast<void*>(b.data())), per, off});
        off += per;
      }
      snap_bytes_[kc] = (off + 4095) / 4096 * 4096;
      pinned(&snap_host_[kc], snap_bytes_[kc] * size_t(snapshots_));
    }
    pinned(&snap_tok_host_, size_t(snapshots_) * 4096);
  }
}

Hy4ResidentModel::~Hy4ResidentModel() {
  if (dst_) {
    rt::DeviceGuard g(card_[1].dev);
    (void)hipStreamDestroy(dst_);
  }
  if (drafted_) (void)hipEventDestroy(drafted_);
  if (dtok_host_) (void)hipHostFree(dtok_host_);
  for (uint8_t* p : snap_host_)
    if (p) (void)hipHostFree(p);
  if (snap_tok_host_) (void)hipHostFree(snap_tok_host_);
  for (auto& cd : card_)
    if (cd.st) {
      rt::DeviceGuard g(cd.dev);
      (void)hipStreamDestroy(cd.st);
    }
  for (Flight& f : flight_) {
    for (void* p : {static_cast<void*>(f.x_host), static_cast<void*>(f.logits_host), static_cast<void*>(f.io_host),
                    static_cast<void*>(f.tables_host), static_cast<void*>(f.eng_mask), static_cast<void*>(f.pos3_host),
                    static_cast<void*>(f.img_host)})
      if (p) (void)hipHostFree(p);
    for (int m = 0; m < 4; ++m) {
      if (f.eng_q[m]) (void)hipHostFree(f.eng_q[m]);
      if (f.eng_s[m]) (void)hipHostFree(f.eng_s[m]);
    }
    for (hipEvent_t e : {f.handoff, f.saved, f.consumed, f.done})
      if (e) (void)hipEventDestroy(e);
  }
}

void Hy4ResidentModel::snapshot_host(int id, std::vector<std::pair<uint8_t*, size_t>>& segs) const {
  for (int kc = 0; kc < 2; ++kc)
    if (snap_bytes_[kc] > 0) segs.emplace_back(snap_host_[kc] + size_t(id) * snap_bytes_[kc], snap_bytes_[kc]);
  if (snap_tok_host_) segs.emplace_back(snap_tok_host_ + size_t(id) * 4096, size_t(4096));
}

std::vector<std::vector<uint64_t>> Hy4ResidentModel::routing_histogram() const {
  std::vector<std::vector<uint64_t>> h(size_t(c_.n_layers), std::vector<uint64_t>(size_t(c_.n_experts), 0));
  std::vector<uint32_t> buf(size_t(c_.n_layers) * c_.n_experts);
  for (const auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemcpy(buf.data(), cd.scratch.get(model::kBufRouteHist), buf.size() * 4, hipMemcpyDeviceToHost));
    for (size_t L = 0; L < h.size(); ++L)
      for (size_t e = 0; e < h[L].size(); ++e) h[L][e] += buf[L * c_.n_experts + e];
  }
  return h;
}

std::string Hy4ResidentModel::routing_share() const {
  const auto h = routing_histogram();
  uint64_t card = 0, all = 0;
  std::string lines;
  char buf[64];
  for (int L = 0; L < c_.n_layers; ++L) {
    if (c_.is_dense(L)) continue;
    uint64_t lc = 0, la = 0;
    for (int e = 0; e < c_.n_experts; ++e) {
      la += h[size_t(L)][size_t(e)];
      if (plan_.on_card[size_t(L)][size_t(e)]) lc += h[size_t(L)][size_t(e)];
    }
    std::snprintf(buf, sizeof buf, "L%d %.4f\n", L, la ? double(lc) / double(la) : 0.0);
    lines += buf;
    card += lc;
    all += la;
  }
  std::snprintf(buf, sizeof buf, "total %.4f (%llu pairs)\n", all ? double(card) / double(all) : 0.0,
                static_cast<unsigned long long>(all));
  return buf + lines;
}

void Hy4ResidentModel::reset_routing_histogram() {
  for (auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemset(cd.scratch.get(model::kBufRouteHist), 0, cd.scratch.bufs[model::kBufRouteHist].size()));
  }
}

void Hy4ResidentModel::set_block_tables(const std::vector<int32_t>& tables) {
  if (tables.size() != size_t(geo_.max_slots) * geo_.max_pages_per_seq)
    throw std::runtime_error("set_block_tables: wrong size");
  tables_ = tables;
  tables_dirty_ = true;
}

void Hy4ResidentModel::state_copies(int kc, const std::vector<StateOp>& ops, bool save) {
  Card& cd = card_[kc];
  for (const StateOp& op : ops) {
    uint8_t* snap = snap_host_[kc] + size_t(op.snapshot) * snap_bytes_[kc];
    for (const SnapSeg& s : snap_segs_[kc]) {
      uint8_t* slot = s.dev + size_t(op.slot) * s.bytes;
      if (save) k::copy_kernel(snap + s.off, slot, s.bytes, cd.st);
      else k::copy_kernel(slot, snap + s.off, s.bytes, cd.st);
    }
  }
}

uint64_t Hy4ResidentModel::submit(const StepInput& in) {
  const int T = int(in.ids.size()), R = int(in.logit_rows.size()), D = c_.dim;
  if (T == 0 || T > max_tokens_) throw std::runtime_error("step: bad token count");
  if (R > max_logit_rows_) throw std::runtime_error("step: too many logit rows");
  if (!in.image_spans.empty()) throw std::runtime_error("step: Hy4 image tokens are causal (no spans)");
  if (!in.image_rows.empty() && !vision_) throw std::runtime_error("step: image rows without a vision tower");
  if (!in.backup.empty() && !dfly_) throw std::runtime_error("step: verify chunks without a drafter");
  for (const auto* ops : {&in.restore, &in.save})
    for (const StateOp& op : *ops)
      if (op.snapshot < 0 || op.snapshot >= snapshots_ || op.slot < 0 || op.slot >= geo_.max_slots)
        throw std::runtime_error("step: bad state op");
  const uint64_t ticket = next_ticket_;
  Flight& f = flight_[ticket % 2];
  if (f.busy) throw std::runtime_error("step: too many steps in flight");
  const size_t row = size_t(c_.hc) * D;

  // Engram inputs: each row's n-gram (the slot's last two tokens before its chunk), looked up into this flight's rows.
  for (const StateOp& op : in.restore)
    std::memcpy(&last2_[size_t(op.slot) * 2], snap_tok_host_ + size_t(op.snapshot) * 4096, 8);
  std::vector<int32_t> ngram(size_t(T) * 3);
  const auto is_verify = [&](const Chunk& ch) {
    for (const RingBackup& bk : in.backup)
      if (bk.slot == ch.slot && bk.s0 == ch.s0) return true;
    return false;
  };
  for (const Chunk& ch : in.chunks) {
    if (ch.s0 == 0) last2_[size_t(ch.slot) * 2] = last2_[size_t(ch.slot) * 2 + 1] = -1;
    if (dfly_ && is_verify(ch)) {  // for rolling the Engram history back past rejected drafts
      vs0_[size_t(ch.slot)] = ch.s0;
      vids_[size_t(ch.slot)].assign(in.ids.begin() + ch.t0, in.ids.begin() + ch.t0 + ch.n);
      vprev_[size_t(ch.slot) * 2] = last2_[size_t(ch.slot) * 2];
      vprev_[size_t(ch.slot) * 2 + 1] = last2_[size_t(ch.slot) * 2 + 1];
    }
    int32_t p1 = last2_[size_t(ch.slot) * 2], p2 = last2_[size_t(ch.slot) * 2 + 1];  // tokens at s0 - 1, s0 - 2
    for (int i = 0; i < ch.n; ++i) {
      const int t = ch.t0 + i;
      const int32_t id = in.ids[size_t(t)];
      ngram[size_t(t) * 3] = id;
      ngram[size_t(t) * 3 + 1] = p1 >= 0 ? p1 : c_.engram_pad;
      ngram[size_t(t) * 3 + 2] = p2 >= 0 ? p2 : c_.engram_pad;
      f.eng_mask[t] = id != c_.image_token && id != c_.video_token && id != c_.engram_pad;
      p2 = p1;
      p1 = id;
    }
    last2_[size_t(ch.slot) * 2] = p1;
    last2_[size_t(ch.slot) * 2 + 1] = p2;
  }
  for (const StateOp& op : in.save)
    std::memcpy(snap_tok_host_ + size_t(op.snapshot) * 4096, &last2_[size_t(op.slot) * 2], 8);
  engram_->lookup_all(ngram.data(), f.eng_mask, T, f.eng_q, f.eng_s);

  // Image rows into this flight's pinned buffer, in order.
  {
    size_t off = 0;
    for (const ImageRows& im : in.image_rows) {
      if (im.t0 < 0 || im.n <= 0 || im.t0 + im.n > T || off + size_t(im.n) > size_t(T))
        throw std::runtime_error("step: image rows outside the batch");
      std::memcpy(f.img_host + off * D, im.rows, size_t(im.n) * D * 2);
      off += size_t(im.n);
    }
  }
  // M-RoPE: when a chunk's sequence has images, every row's three planes (text rows: their slot, shifted past images)
  bool mrope = false;
  for (const Chunk& ch : in.chunks) mrope |= ch.mrope != nullptr;
  if (mrope)
    for (const Chunk& ch : in.chunks)
      for (int i = 0; i < ch.n; ++i) mrope_position(ch.mrope, ch.s0 + i, f.pos3_host + size_t(ch.t0 + i) * 3);
  std::memcpy(f.io_host, in.ids.data(), size_t(T) * 4);
  std::memcpy(f.io_host + max_tokens_, in.pos.data(), size_t(T) * 4);
  std::memcpy(f.io_host + 2 * max_tokens_, in.logit_rows.data(), size_t(R) * 4);
  const bool tables = tables_dirty_;
  if (tables) std::memcpy(f.tables_host, tables_.data(), tables_.size() * 4);
  tables_dirty_ = false;
  model::Hy4Batch b;
  for (const Chunk& ch : in.chunks) {
    b.chunks.push_back({ch.slot, ch.s0, ch.n, ch.t0, is_verify(ch)});
    if (mrope) mrope_position(ch.mrope, ch.s0 / c_.idx_pool * c_.idx_pool, b.chunks.back().g0);
  }
  b.T = T;
  b.max_pages = geo_.max_pages_per_seq;
  b.pos_host = in.pos.data();
  for (int m = 0; m < engram_->memories(); ++m) {
    b.eng_q[m] = f.eng_q[m];
    b.eng_s[m] = f.eng_s[m];
  }
  b.eng_mask = f.eng_mask;
  for (int kc = 0; kc < 2; ++kc) {
    Card& cd = card_[kc];
    rt::DeviceGuard g(cd.dev);
    if (kc == 0) k::copy_kernel(cd.ids.data(), f.io_host, size_t(T) * 4, cd.st);
    k::copy_kernel(cd.pos.data(), f.io_host + max_tokens_, size_t(T) * 4, cd.st);
    if (mrope) k::copy_kernel(cd.pos3.data(), f.pos3_host, size_t(T) * 3 * 4, cd.st);
    if (tables) k::copy_kernel(cd.tables.data(), f.tables_host, tables_.size() * 4, cd.st);
    b.pos = cd.pos.as<int32_t>();
    b.pos3 = mrope ? cd.pos3.as<int32_t>() : nullptr;
    b.tables = cd.tables.as<int32_t>();
    if (kc == 0) {
      model::hy4_embed(c_, embed_, cd.ids.as<int32_t>(), cd.x.as<uint16_t>(), T, cd.st);
      size_t off = 0;  // image rows over their pad tokens' embeddings (every iHC stream), from the pinned buffer
      for (const ImageRows& im : in.image_rows) {
        k::put_rows_hc(f.img_host + off * D, im.n, cd.x.as<uint16_t>(), im.t0, D, c_.hc, cd.st);
        off += size_t(im.n);
      }
    } else {
      if (dfly_) UH_CHECK(hipStreamWaitEvent(cd.st, drafted_, 0));  // the last draft()'s restores and reads
      UH_CHECK(hipStreamWaitEvent(cd.st, f.handoff, 0));
      k::copy_kernel(cd.x.data(), f.x_host, size_t(T) * row * 2, cd.st);
      UH_CHECK(hipEventRecord(f.consumed, cd.st));
    }
    state_copies(kc, in.restore, false);
    for (size_t i = 0; i < cd.w.size(); ++i)
      model::hy4_layer_forward(c_, cd.w[i], cd.cache[i], cd.scratch, cd.x.as<uint16_t>(), b, cd.st, chan_[kc].get());
    if (kc == 0) {
      UH_CHECK(hipStreamWaitEvent(cd.st, f.consumed, 0));
      k::copy_kernel(f.x_host, cd.x.data(), size_t(T) * row * 2, cd.st);
      UH_CHECK(hipEventRecord(f.handoff, cd.st));
      state_copies(0, in.save, true);
      f.saved_wait = in.sync_saves && !in.save.empty();
      if (f.saved_wait) UH_CHECK(hipEventRecord(f.saved, cd.st));
    }
  }
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  state_copies(1, in.save, true);
  if (R > 0) {
    k::copy_kernel(rows_idx_.data(), f.io_host + 2 * max_tokens_, size_t(R) * 4, cd.st);
    k::gather_rows(cd.x.as<uint16_t>(), rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, int(row), cd.st);
    model::hy4_head_forward(c_, head_, cd.scratch, hrows_.as<uint16_t>(), R, logits_.as<float>(), cd.st,
                            dfly_ ? hn_rows_.as<uint16_t>() : nullptr);
    if (dfly_)  // each logit row's final hidden state into its slot's ring (row pos % 6): DFly2's input
      for (int i = 0; i < R; ++i) {
        const int t = in.logit_rows[size_t(i)];
        for (const Chunk& ch : in.chunks)
          if (t >= ch.t0 && t < ch.t0 + ch.n) {
            const int pos = ch.s0 + t - ch.t0;
            UH_CHECK(hipMemcpyAsync(hidden_.as<uint16_t>() + (size_t(ch.slot) * model::kHy4MaxVerify +
                                                              pos % model::kHy4MaxVerify) * D,
                                    hn_rows_.as<uint16_t>() + size_t(i) * D, size_t(D) * 2, hipMemcpyDeviceToDevice,
                                    cd.st));
            break;
          }
      }
    k::copy_kernel(f.logits_host, logits_.data(), size_t(R) * c_.vocab * 4, cd.st);
  }
  UH_CHECK(hipEventRecord(f.done, cd.st));
  f.busy = true;
  f.ticket = ticket;
  f.rows = R;
  return next_ticket_++;
}

// Restores first (card 0's caches on its stream, ahead of its next step; card 1's on the draft stream, which card 1's
// next step waits for; the Engram history on the host), then the drafts on the draft stream.
void Hy4ResidentModel::draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs,
                             std::vector<int32_t>& tokens, std::vector<float>& conf) {
  tokens.clear();
  conf.clear();
  if (!dfly_) throw std::runtime_error("draft: no DFly2 drafter loaded");
  for (const RingRestore& r : restore) {
    if (r.slot < 0 || r.slot >= geo_.max_slots || r.from <= r.s0 || r.to <= r.from ||
        r.to - r.s0 > model::kHy4MaxVerify)
      throw std::runtime_error("draft: bad restore");
    {
      Card& c0 = card_[0];
      rt::DeviceGuard g(c0.dev);
      for (auto& cache : c0.cache) model::hy4_restore_slot(cache, r.slot, r.s0, r.from, r.to, c0.st);
    }
    {
      rt::DeviceGuard g(card_[1].dev);
      for (auto& cache : card_[1].cache) model::hy4_restore_slot(cache, r.slot, r.s0, r.from, r.to, dst_);
    }
    const auto& ids = vids_[size_t(r.slot)];
    if (vs0_[size_t(r.slot)] != r.s0 || int(ids.size()) < r.from - r.s0)
      throw std::runtime_error("draft: restore without its verify chunk");
    const auto tok_at = [&](int p) {
      return p >= r.s0 ? ids[size_t(p - r.s0)] : vprev_[size_t(r.slot) * 2 + (r.s0 - 1 - p)];
    };
    last2_[size_t(r.slot) * 2] = tok_at(r.from - 1);
    last2_[size_t(r.slot) * 2 + 1] = r.from - 2 >= 0 ? tok_at(r.from - 2) : -1;
  }
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  const int R = int(reqs.size());
  if (R > 0) {
    std::vector<model::Hy4DflyReq> dr;
    for (const DraftReq& q : reqs) {
      if (q.token < 0 || q.token >= c_.vocab) throw std::runtime_error("draft: token id outside the vocabulary");
      int32_t rp[3];
      mrope_position(q.mrope, q.start + 1, rp);  // text after any images: the three planes are equal
      dr.push_back({q.slot, q.start + 1, q.token, rp[0]});
    }
    model::hy4_dfly_draft(c_, dw_, head_, dtarget_, cd.tables.as<int32_t>(), geo_.max_pages_per_seq,
                          hidden_.as<uint16_t>(), dscr_, dr, dtok_.as<int32_t>(), dst_,
                          reinterpret_cast<float*>(dtok_.as<int32_t>() + size_t(model::kHy4DraftTokens) * R));
    UH_CHECK(hipMemcpyAsync(dtok_host_, dtok_.data(), size_t(model::kHy4DraftTokens) * R * 8, hipMemcpyDeviceToHost,
                            dst_));
  }
  UH_CHECK(hipEventRecord(drafted_, dst_));
  UH_CHECK(hipStreamSynchronize(dst_));
  // The drafter's probability of its token as the confidence (its confidence head is unused, as in vLLM and SGLang).
  tokens.resize(size_t(R) * model::kHy4DraftTokens);
  conf.resize(tokens.size());
  const float* cf = reinterpret_cast<const float*>(dtok_host_ + size_t(model::kHy4DraftTokens) * R);
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < model::kHy4DraftTokens; ++t) {
      tokens[size_t(r) * model::kHy4DraftTokens + t] = dtok_host_[size_t(t) * R + r];
      conf[size_t(r) * model::kHy4DraftTokens + t] = cf[size_t(t) * R + r];
    }
}

void Hy4ResidentModel::encode_image(const vision::Hy4Patches& p, std::vector<uint16_t>& rows) {
  if (!vision_) throw std::runtime_error("no vision tower loaded");
  Vision& v = *vision_;
  const int n = (p.grid_h / v.c.merge) * (p.grid_w / v.c.merge + 1);
  if (p.grid_h * p.grid_w > v.c.max_patches) throw std::runtime_error("image of more patches than the vision tower takes");
  std::lock_guard lk(v.mu);
  rt::DeviceGuard g(v.dev);
  model::hy4_vision_encode(v.c, v.w, v.s, p.grid_h, p.grid_w, p.data, v.out.as<uint16_t>(), v.st);
  rows.resize(size_t(n) * v.c.out_dim);
  UH_CHECK(hipMemcpyAsync(rows.data(), v.out.data(), rows.size() * 2, hipMemcpyDeviceToHost, v.st));
  UH_CHECK(hipStreamSynchronize(v.st));
}

void Hy4ResidentModel::wait(uint64_t ticket, std::vector<float>& out) {
  Flight& f = flight_[ticket % 2];
  if (!f.busy || f.ticket != ticket) throw std::runtime_error("step: wait for an unknown ticket");
  {
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventSynchronize(f.done));
  }
  if (f.saved_wait) {
    rt::DeviceGuard g(card_[0].dev);
    UH_CHECK(hipEventSynchronize(f.saved));
    f.saved_wait = false;
  }
  out.resize(size_t(f.rows) * c_.vocab);
  std::memcpy(out.data(), f.logits_host, out.size() * 4);
  f.busy = false;
}

}  // namespace unholy::engine
