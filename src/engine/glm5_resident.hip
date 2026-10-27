#include "engine/glm5_resident.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/elem_fast.hpp"
#include "kernels/ref.hpp"
#include "model/loader.hpp"
#include "rt/hip.hpp"

namespace unholy::engine {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// Glm5ApuExperts

size_t Glm5ApuExperts::work_bytes(const model::Glm5Config& c, size_t cap) {
  const size_t D = c.dim, I = c.moe_inter;
  return 2 * cap * D * 2 + 3 * cap * I * 2 + std::min<size_t>(cap, 64) * std::max(D, 2 * I) * 4 +
         (64 * std::max(D, 3 * I) / 128 + D / 128) * 4;
}

size_t Glm5ApuExperts::staging_bytes(const model::Glm5Config& c, size_t cap) {
  const size_t tokens = cap / size_t(c.n_active), D = c.dim;
  return tokens * D * (2 + 4) + cap * 4 * 4 + sizeof(k::MoePlan) +
         (cap / k::kExl3TileRows + c.n_experts + 1) * sizeof(k::Exl3Tile);
}

Glm5ApuExperts::Work::~Work() {
  rt::DeviceGuard g(dev);
  if (st) (void)hipStreamDestroy(st);
}

Glm5ApuExperts::Glm5ApuExperts(const model::Glm5Config& c, int dev, size_t capacity_rows,
                               std::shared_ptr<Layers> layers, const Glm5ApuExperts* share)
    : c_(c), dev_(dev), layers_(std::move(layers)) {
  rt::DeviceGuard g(dev);
  this->capacity_rows = capacity_rows;
  const size_t D = c.dim, I = c.moe_inter, cap = capacity_rows, tokens = cap / size_t(c.n_active);
  max_tiles = int(cap / k::kExl3TileRows + c.n_experts + 1);
  const unsigned flags = hipHostMallocPortable | hipHostMallocMapped;
  const auto pinned = [&](auto** host, auto** dev_ptr, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(host), bytes, flags));
    if (dev_ptr) UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(dev_ptr), *host, 0));
  };
  pinned(&in_host, &in_dev_, tokens * D * 2);   // compact token rows (bf16)
  pinned(&out_host, &out_dev_, tokens * D * 4);  // per-token partial sums (fp32)
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
  if (share) {
    if (share->capacity_rows < cap) throw std::runtime_error("Glm5ApuExperts: shared working buffers too small");
    work_ = share->work_;
  } else {
    work_ = std::make_shared<Work>();
    work_->dev = dev;
    UH_CHECK(hipStreamCreateWithFlags(&work_->st, hipStreamNonBlocking));
    work_->xh1 = rt::DeviceBuffer(dev, cap * D * 2);
    work_->xh3 = rt::DeviceBuffer(dev, cap * D * 2);
    work_->xh2 = rt::DeviceBuffer(dev, cap * I * 2);
    work_->h1 = rt::DeviceBuffer(dev, cap * I * 2);
    work_->h3 = rt::DeviceBuffer(dev, cap * I * 2);
    work_->yp = rt::DeviceBuffer(dev, std::min<size_t>(cap, 64) * std::max(D, 2 * I) * 4);
    work_->cnt = rt::DeviceBuffer(dev, (64 * std::max(D, 3 * I) / 128 + D / 128) * 4);
    UH_CHECK(hipMemset(work_->cnt.data(), 0, work_->cnt.size()));
  }
  if (layers_->size() < size_t(c.n_layers)) layers_->resize(size_t(c.n_layers));
}

Glm5ApuExperts::~Glm5ApuExperts() {
  rt::DeviceGuard g(dev_);
  for (void* p : {static_cast<void*>(in_host), static_cast<void*>(out_host), static_cast<void*>(w_host),
                  static_cast<void*>(e_host), static_cast<void*>(plan_host), static_cast<void*>(tiles_host),
                  static_cast<void*>(p2u_host), static_cast<void*>(u2p_host), static_cast<void*>(req_flag_),
                  static_cast<void*>(done_flag_)})
    if (p) (void)hipHostFree(p);
}

void Glm5ApuExperts::add_layer(model::Glm5LayerWeights w) {
  const int L = w.layer;
  if (layers_->size() <= size_t(L)) layers_->resize(size_t(L) + 1);
  layers_->at(size_t(L)) = std::make_unique<model::Glm5LayerWeights>(std::move(w));
}

void Glm5ApuExperts::run(int, const std::vector<Job>&) {
  throw std::runtime_error("Glm5ApuExperts: host-synchronous jobs are not supported");
}

// The card stream raises req after staging the inputs; the APU stream waits for it, runs the remote tiles (their
// pair rows into the working xh1, free once the gate projection is done), sums each token's pair rows into one fp32
// row of out_host, raises done; join() makes the card stream wait for done (sequence numbers: no resets, no host
// waits).
void Glm5ApuExperts::enqueue(int layer, hipStream_t card_st, int P) {
  const auto* w = layers_->at(size_t(layer)).get();
  if (!w) throw std::runtime_error("Glm5ApuExperts: no experts of layer " + std::to_string(layer) + " on the APU");
  ++seq_;
  UH_CHECK(hipStreamWriteValue32(card_st, req_flag_, seq_, 0));
  rt::DeviceGuard g(dev_);
  Work& wk = *work_;
  UH_CHECK(hipStreamWaitValue32(wk.st, req_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
  const model::Glm5ExpertBufs b{wk.xh1.as<uint16_t>(), wk.xh3.as<uint16_t>(), wk.xh2.as<uint16_t>(),
                                wk.h1.as<uint16_t>(), wk.h3.as<uint16_t>(), wk.yp.as<float>(), wk.cnt.as<uint32_t>()};
  const k::Exl3TokenSums sums{u2p_dev_, &plan_dev_->n_remote_tokens, reinterpret_cast<float*>(out_dev_), c_.n_active};
  model::glm5_experts_dev(c_, *w, in_dev_, w_dev_, e_dev_, tiles_dev_, max_tiles, &plan_dev_->n_remote_tiles, P, b,
                          wk.xh1.as<uint16_t>(), wk.st, &plan_dev_->remote0, nullptr, p2u_dev_, &sums);
  UH_CHECK(hipStreamWriteValue32(wk.st, done_flag_, seq_, 0));
}

void Glm5ApuExperts::join(hipStream_t card_st) {
  UH_CHECK(hipStreamWaitValue32(card_st, done_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
}

// ---------------------------------------------------------------------------------------------------
// Plan

namespace {

const char* const kEmbed = "model.language_model.embed_tokens.weight";

size_t head_bytes(const model::Checkpoint& ck) {
  size_t b = ck.at("model.language_model.norm.weight").nbytes;
  for (const char* s : {".trellis", ".suh", ".svh"}) b += ck.at(std::string("lm_head") + s).nbytes;
  return b;
}

// A step's second part (glm5_layers_fast): at most half the step plus a page.
int second_part_tokens(int max_tokens) { return max_tokens / 2 + k::kGlmPageRows; }

// Card-1 bytes MTP drafting needs besides its weights and cache: the draft scratch and rows, the step's cache-row
// inputs, the slots' hidden states.
size_t mtp_work_bytes(const model::Glm5Config& c, const model::CacheGeometry& g, int max_tokens, int max_seq) {
  const size_t D = c.dim, R = size_t(std::max(g.max_slots, 8));
  return model::glm5_scratch_bytes(c, int(R), max_seq, true) + 4 * size_t(max_tokens) * D * 2 +
         2 * size_t(g.max_slots) * D * 2 + R * (4 * D * 2 + size_t(c.vocab) * 4 + 16);
}

}  // namespace

Plan make_glm5_plan(const Glm5PlanInput& in) {
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

  // Layer split: balance dense + cache bytes; card 0 also holds the embedding, card 1 the head.
  std::vector<size_t> dense(size_t(c.n_layers)), cache(size_t(c.n_layers));
  size_t total = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    dense[size_t(L)] = model::glm5_dense_layer_bytes(ck, L);
    cache[size_t(L)] = model::glm5_paged_cache_bytes(c, c.is_kda(L), in.geo.max_slots, in.geo.num_pages, in.mtp);
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
    const int k = pl.layer_card[size_t(L)];
    pl.dense[k] += dense[size_t(L)];
    pl.cache[k] += cache[size_t(L)];
  }
  pl.dense[0] += ck.at(kEmbed).nbytes;
  pl.dense[1] += head_bytes(ck);
  if (in.mtp) {  // the MTP layer's dense part, DSA rows and working buffers on card 1 (its experts on the APU)
    pl.dense[1] += model::glm5_dense_layer_bytes(ck, c.n_layers) + mtp_work_bytes(c, in.geo, in.max_tokens, in.max_seq);
    pl.cache[1] += model::glm5_paged_cache_bytes(c, false, in.geo.max_slots, in.geo.num_pages);
  }
  const int logit_rows = in.max_logit_rows > 0 ? in.max_logit_rows : in.max_tokens;
  for (int k = 0; k < 2; ++k) {
    pl.scratch[k] = model::glm5_scratch_bytes(c, in.max_tokens, in.max_seq, true) +
                    size_t(in.max_tokens) * c.hc_mult * c.dim * 2 + size_t(in.max_tokens) * 16 +
                    size_t(in.geo.max_slots) * in.geo.max_pages_per_seq * 4 +
                    (k == 1 ? size_t(logit_rows) * (c.vocab * 4 + size_t(c.hc_mult) * c.dim * 2) : 0);
    const size_t used = pl.dense[k] + pl.cache[k] + pl.scratch[k] + in.card_margin;
    const size_t free = info(pl.card[k]).free_bytes;
    if (used > free) throw std::runtime_error("plan: card " + std::to_string(k) + " cannot hold its dense layers");
    pl.budget[k] = free - used;
  }

  // Hot experts: greedy by per-layer routing share, per card budget.
  struct Cand {
    double share;
    int layer, expert;
    size_t bytes;
  };
  std::vector<Cand> cands[2];
  double layers_moe = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    if (c.is_dense(L)) continue;
    double sum = 0;
    for (double v : in.usage[size_t(L)]) sum += v;
    for (int e = 0; e < c.n_experts; ++e)
      cands[pl.layer_card[size_t(L)]].push_back(
          {in.usage[size_t(L)][size_t(e)] / sum, L, e, model::glm5_expert_bytes(ck, L, e)});
    layers_moe += 1.0;
  }
  pl.on_card.assign(size_t(c.n_layers), std::vector<uint8_t>(size_t(c.n_experts), 0));
  double mass_cards = 0;
  for (int k = 0; k < 2; ++k) {
    std::sort(cands[k].begin(), cands[k].end(), [](const Cand& a, const Cand& b) { return a.share > b.share; });
    for (const auto& cd : cands[k]) {
      if (pl.experts[k] + cd.bytes > pl.budget[k]) continue;
      pl.experts[k] += cd.bytes;
      pl.on_card[size_t(cd.layer)][size_t(cd.expert)] = 1;
      mass_cards += cd.share;
    }
  }
  for (int k = 0; k < 2; ++k)
    for (const auto& cd : cands[k])
      if (!pl.on_card[size_t(cd.layer)][size_t(cd.expert)]) pl.apu_experts += cd.bytes;
  pl.routed_on_cards = mass_cards / std::max(layers_moe, 1.0);
  pl.apu_free = info(pl.apu).free_bytes;
  // Per card: a channel for a step (or its first part) and one for a step's second part on its working buffers; their
  // pinned staging (it takes the same RAM).
  const size_t cap = size_t(in.max_tokens) * c.n_active, cap2 = size_t(second_part_tokens(in.max_tokens)) * c.n_active;
  size_t apu_work = 2 * (Glm5ApuExperts::work_bytes(c, cap) + Glm5ApuExperts::staging_bytes(c, cap) +
                         Glm5ApuExperts::staging_bytes(c, cap2));
  if (in.mtp) {  // the draft channel
    for (int e = 0; e < c.n_experts; ++e) pl.apu_experts += model::glm5_expert_bytes(ck, c.n_layers, e);
    const size_t dcap = size_t(std::max(in.geo.max_slots, 8)) * c.n_active;
    apu_work += Glm5ApuExperts::work_bytes(c, dcap) + Glm5ApuExperts::staging_bytes(c, dcap);
  }
  if (in.vision_tokens > 0) {  // the vision tower's weights, scratch and output rows; the flights' pinned image rows
    for (const auto& t : ck.tensors())
      if (t.name.rfind("vision.", 0) == 0) pl.vision_apu += t.nbytes;
    pl.vision_apu += model::glm5_vision_scratch_bytes(*in.vision, in.vision_tokens) +
                     size_t(in.vision_tokens) * in.vision->out_dim * 2 + 2 * size_t(in.max_tokens) * c.dim * 2;
  }
  const size_t apu_need = pl.apu_experts + apu_work + pl.vision_apu + in.apu_margin;
  if (apu_need > pl.apu_free) {
    char msg[256];
    std::snprintf(msg, sizeof msg, "plan: the APU cannot hold the cold experts (%.2f GB + %.2f GB working + %.2f GB "
                  "vision + %.2f GB margin > %.2f GB free)", pl.apu_experts / 1e9, apu_work / 1e9, pl.vision_apu / 1e9,
                  in.apu_margin / 1e9, pl.apu_free / 1e9);
    throw std::runtime_error(msg);
  }
  // Prefix snapshots: pinned host memory is TTM pages too, under the same page limit as the APU's (the limit its free
  // bytes report; MemAvailable is no guide, TTM's pool keeps freed pages); as many as fit beside the margin.
  if (in.snapshots > 0) {
    pl.snapshot_bytes = glm5_snapshot_bytes(c, in.mtp);
    const size_t left = pl.apu_free - apu_need;
    pl.snapshots = int(std::min<size_t>(size_t(in.snapshots), left / pl.snapshot_bytes));
  }
  return pl;
}

size_t glm5_snapshot_bytes(const model::Glm5Config& c, bool mtp) {
  size_t b = mtp ? size_t(c.dim) * 2 : 0;  // the slot's last hidden state (the next MTP cache row's)
  for (int L = 0; L < c.n_layers; ++L)
    if (c.is_kda(L)) b += model::glm5_paged_cache_bytes(c, true, 1, 0);
  return b;
}

// ---------------------------------------------------------------------------------------------------
// Glm5ResidentModel

// The vision tower on the APU: weights, scratch, its stream, the output rows; one image at a time.
struct Glm5ResidentModel::Vision {
  int dev = 0, max_tokens = 0;
  hipStream_t st = nullptr;
  std::mutex mu;
  model::Glm5VisionConfig c;
  model::Glm5VisionWeights w;
  model::Glm5VisionScratch s;
  rt::DeviceBuffer out;
  ~Vision() {
    rt::DeviceGuard g(dev);
    if (st) (void)hipStreamDestroy(st);
  }
};

Glm5ResidentModel::Glm5ResidentModel(const model::Glm5Config& c, const model::Checkpoint& ck, const Plan& plan,
                                     const model::CacheGeometry& geo, int max_tokens, int max_seq, int max_logit_rows,
                                     bool mtp, int snapshots, int vision_tokens)
    : c_(c), plan_(plan), geo_(geo), max_tokens_(max_tokens),
      max_logit_rows_(max_logit_rows > 0 ? std::min(max_logit_rows, max_tokens) : max_tokens), mtp_(mtp),
      snapshots_(snapshots) {
  if (geo.max_pages_per_seq * k::kGlmPageRows < max_seq) throw std::runtime_error("glm5: block table too narrow");
  model::Uploader up(ck);
  const int draft_rows = std::max(geo.max_slots, 8);
  {
    auto layers = std::make_shared<Glm5ApuExperts::Layers>();
    for (auto& ch : chan_) ch = std::make_unique<Glm5ApuExperts>(c, plan.apu, size_t(max_tokens) * c.n_active, layers);
    for (int k = 0; k < 2; ++k)
      chan2_[k] = std::make_unique<Glm5ApuExperts>(c, plan.apu, size_t(second_part_tokens(max_tokens)) * c.n_active,
                                                   layers, chan_[k].get());
    if (mtp_) dchan_ = std::make_unique<Glm5ApuExperts>(c, plan.apu, size_t(draft_rows) * c.n_active, layers);
  }
  for (int k = 0; k < 2; ++k) card_arena_[k] = std::make_unique<rt::Arena>(plan.card[k], size_t(256) << 20);
  apu_arena_ = std::make_unique<rt::Arena>(plan.apu, size_t(256) << 20);
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    cd.dev = plan.card[k];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipStreamCreateWithFlags(&cd.st, hipStreamNonBlocking));
    cd.scratch = model::make_glm5_scratch(c, cd.dev, max_tokens, max_seq, true);
    cd.x = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    cd.ids = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.pos = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.tables = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
  }
  for (int L = 0; L < c.n_layers; ++L) {
    const int kc = plan.layer_card[size_t(L)];
    Card& cd = card_[kc];
    {  // card: dense + hot experts
      rt::DeviceGuard g(cd.dev);
      model::Glm5LoadOptions lo;
      lo.expert_mask = &plan.on_card[size_t(L)];
      lo.arena = card_arena_[kc].get();
      lo.up = &up;
      cd.w.push_back(model::load_glm5_layer(ck, c, L, cd.dev, cd.st, lo));
      model::prepare_glm5_fast(c, cd.w.back(), cd.dev, cd.st);
      cd.cache.push_back(model::make_glm5_paged_cache(c, L, geo.max_slots, geo.num_pages, cd.dev, cd.st, mtp_));
    }
    if (c.is_dense(L)) continue;
    std::vector<uint8_t> cold(size_t(c.n_experts));
    bool any = false;
    for (int e = 0; e < c.n_experts; ++e) any |= (cold[size_t(e)] = !plan.on_card[size_t(L)][size_t(e)]) != 0;
    if (!any) continue;
    rt::DeviceGuard ga(plan.apu);  // APU: the cold experts
    model::Glm5LoadOptions la;
    la.load_dense = false;
    la.expert_mask = &cold;
    la.arena = apu_arena_.get();
    la.up = &up;
    auto w = model::load_glm5_layer(ck, c, L, plan.apu, chan_[0]->stream(), la);
    model::prepare_glm5_fast(c, w, plan.apu, chan_[0]->stream());
    chan_[0]->add_layer(std::move(w));
  }
  {
    rt::DeviceGuard g(card_[0].dev);
    embed_ = model::load_glm5_head(ck, card_[0].dev, card_[0].st, true, false);
  }
  {
    Card& cd = card_[1];
    rt::DeviceGuard g(cd.dev);
    head_ = model::load_glm5_head(ck, cd.dev, cd.st, false, true);
    model::prepare_glm5_fast_head(head_, cd.dev, cd.st);
    hrows_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * c.hc_mult * c.dim * 2);
    logits_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * c.vocab * 4);
    rows_idx_ = rt::DeviceBuffer(cd.dev, size_t(max_logit_rows_) * 4);
  }
  if (mtp_) {  // the MTP layer: dense part + DSA rows on card 1, experts on the APU
    Card& cd = card_[1];
    rt::DeviceGuard g(cd.dev);
    const std::vector<uint8_t> none(size_t(c.n_experts), 0), all(size_t(c.n_experts), 1);
    model::Glm5LoadOptions lo;
    lo.expert_mask = &none;
    lo.arena = card_arena_[1].get();
    lo.up = &up;
    mtpw_ = model::load_glm5_mtp(ck, c, cd.dev, cd.st, lo);
    model::prepare_glm5_fast_mtp(c, mtpw_, cd.dev, cd.st);
    mcache_ = model::make_glm5_paged_cache(c, c.n_layers, geo.max_slots, geo.num_pages, cd.dev, cd.st);
    UH_CHECK(hipStreamCreateWithFlags(&dst_, hipStreamNonBlocking));
    UH_CHECK(hipEventCreateWithFlags(&drafted_, hipEventDisableTiming));
    dscr_ = model::make_glm5_scratch(c, cd.dev, draft_rows, max_seq, true);
    const size_t D = c.dim;
    h_all_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * D * 2);
    m_emb_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * D * 2);
    m_h_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * D * 2);
    m_x0_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * D * 2);
    h_last_ = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * D * 2);
    h_prev_ = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * D * 2);
    d_emb_ = rt::DeviceBuffer(cd.dev, size_t(draft_rows) * D * 2);
    d_h_ = rt::DeviceBuffer(cd.dev, size_t(draft_rows) * D * 2);
    d_x_ = rt::DeviceBuffer(cd.dev, size_t(draft_rows) * D * 2);
    d_hn_ = rt::DeviceBuffer(cd.dev, size_t(draft_rows) * D * 2);
    d_logits_ = rt::DeviceBuffer(cd.dev, size_t(draft_rows) * c.vocab * 4);
    d_tok_ = rt::DeviceBuffer(cd.dev, size_t(draft_rows) * 3 * 4);  // tokens, confidences, positions
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&d_emb_host_), size_t(draft_rows) * D * 2,
                           hipHostMallocPortable | hipHostMallocMapped));
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&d_tok_host_), size_t(draft_rows) * 3 * 4,
                           hipHostMallocPortable | hipHostMallocMapped));
    embed_host_ = ck.as<uint16_t>(ck.at(kEmbed));
    rt::DeviceGuard ga(plan.apu);
    model::Glm5LoadOptions la;
    la.load_dense = false;
    la.expert_mask = &all;
    la.arena = apu_arena_.get();
    la.up = &up;
    auto w = model::load_glm5_mtp(ck, c, plan.apu, dchan_->stream(), la);
    model::prepare_glm5_fast(c, w.layer, plan.apu, dchan_->stream());
    dchan_->add_layer(std::move(w.layer));
  }
  if (vision_tokens > 0) {
    vision_ = std::make_unique<Vision>();
    Vision& v = *vision_;
    v.dev = plan.apu;
    v.max_tokens = vision_tokens;
    v.c = model::Glm5VisionConfig::from_hf(ck.config());
    if (v.c.out_dim != c.dim) throw std::runtime_error("glm5: the vision tower's output width is not the model's");
    rt::DeviceGuard g(v.dev);
    UH_CHECK(hipStreamCreateWithFlags(&v.st, hipStreamNonBlocking));
    v.w = model::load_glm5_vision(ck, v.c, v.dev, v.st);
    v.s = model::make_glm5_vision_scratch(v.c, vision_tokens, v.dev);
    v.out = rt::DeviceBuffer(v.dev, size_t(vision_tokens) * v.c.out_dim * 2);
  }
  const auto pinned = [](auto** p, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(p), bytes, hipHostMallocPortable | hipHostMallocMapped));
  };
  for (Flight& f : flight_) {
    pinned(&f.x_host, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    pinned(&f.logits_host, size_t(max_logit_rows_) * c.vocab * 4);
    pinned(&f.io_host, size_t(max_tokens) * 3 * 4);
    pinned(&f.tables_host, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
    if (mtp_) pinned(&f.emb_host, size_t(max_tokens) * c.dim * 2);
    if (vision_) pinned(&f.img_host, size_t(max_tokens) * c.dim * 2);
    {
      rt::DeviceGuard g(card_[0].dev);
      UH_CHECK(hipEventCreateWithFlags(&f.handoff, hipEventDisableTiming));
    }
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventCreateWithFlags(&f.consumed, hipEventDisableTiming));
    UH_CHECK(hipEventCreateWithFlags(&f.done, hipEventDisableTiming));
  }
  if (snapshots_ > 0) {  // a slot's state at a page boundary: the DSA tails hold only complete pools there
    for (int k = 0; k < 2; ++k) {
      size_t off = 0;
      const auto add = [&](const rt::DeviceBuffer& b, size_t per_slot) {
        snap_segs_[k].push_back({static_cast<uint8_t*>(const_cast<void*>(b.data())), per_slot, off});
        off += per_slot;
      };
      for (const model::Glm5PagedCache& pc : card_[k].cache)
        if (pc.kda) {
          add(pc.conv, pc.conv.size() / size_t(geo.max_slots));
          add(pc.state, pc.state.size() / size_t(geo.max_slots));
        }
      if (k == 1 && mtp_) add(h_last_, size_t(c.dim) * 2);
      snap_bytes_[k] = off;
      pinned(&snap_host_[k], off * size_t(snapshots_));
    }
  }
}

Glm5ResidentModel::~Glm5ResidentModel() {
  if (dst_) {
    rt::DeviceGuard g(card_[1].dev);
    (void)hipStreamDestroy(dst_);
  }
  if (drafted_) (void)hipEventDestroy(drafted_);
  for (void* p : {static_cast<void*>(d_emb_host_), static_cast<void*>(d_tok_host_)})
    if (p) (void)hipHostFree(p);
  for (uint8_t* p : snap_host_)
    if (p) (void)hipHostFree(p);
  for (auto& cd : card_)
    if (cd.st) {
      rt::DeviceGuard g(cd.dev);
      (void)hipStreamDestroy(cd.st);
    }
  for (Flight& f : flight_) {
    for (void* p : {static_cast<void*>(f.x_host), static_cast<void*>(f.logits_host), static_cast<void*>(f.io_host),
                    static_cast<void*>(f.tables_host), static_cast<void*>(f.emb_host), static_cast<void*>(f.img_host)})
      if (p) (void)hipHostFree(p);
    for (hipEvent_t e : {f.handoff, f.consumed, f.done})
      if (e) (void)hipEventDestroy(e);
  }
}

const model::Glm5VisionConfig& Glm5ResidentModel::vision_config() const {
  if (!vision_) throw std::runtime_error("no vision tower loaded");
  return vision_->c;
}

int Glm5ResidentModel::vision_tokens() const { return vision_ ? vision_->max_tokens : 0; }

void Glm5ResidentModel::encode_image(const vision::GlmPatches& p, std::vector<uint16_t>& rows) {
  if (!vision_) throw std::runtime_error("no vision tower loaded");
  Vision& v = *vision_;
  const int tokens = p.grid_h * p.grid_w / (v.c.merge * v.c.merge);
  if (tokens > v.max_tokens) throw std::runtime_error("image of more tokens than the vision tower holds");
  std::lock_guard lk(v.mu);
  rt::DeviceGuard g(v.dev);
  model::glm5_vision_encode(v.c, v.w, v.s, p, v.out.as<uint16_t>(), v.st);
  rows.resize(size_t(tokens) * v.c.out_dim);
  UH_CHECK(hipMemcpyAsync(rows.data(), v.out.data(), rows.size() * 2, hipMemcpyDeviceToHost, v.st));
  UH_CHECK(hipStreamSynchronize(v.st));
}

std::vector<std::vector<uint64_t>> Glm5ResidentModel::routing_histogram() const {
  std::vector<std::vector<uint64_t>> h(size_t(c_.n_layers), std::vector<uint64_t>(size_t(c_.n_experts), 0));
  std::vector<uint32_t> buf(size_t(c_.n_layers) * c_.n_experts);
  for (const auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemcpy(buf.data(), cd.scratch.get(model::kRouteHist), buf.size() * 4, hipMemcpyDeviceToHost));
    for (size_t L = 0; L < h.size(); ++L)
      for (size_t e = 0; e < h[L].size(); ++e) h[L][e] += buf[L * c_.n_experts + e];
  }
  return h;
}

std::string Glm5ResidentModel::routing_share() const {
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

void Glm5ResidentModel::reset_routing_histogram() {
  for (auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemset(cd.scratch.get(model::kRouteHist), 0, cd.scratch.bufs[model::kRouteHist].size()));
  }
}

void Glm5ResidentModel::set_block_tables(const std::vector<int32_t>& tables) {
  if (tables.size() != size_t(geo_.max_slots) * geo_.max_pages_per_seq)
    throw std::runtime_error("set_block_tables: wrong size");
  tables_ = tables;  // uploaded on the card streams by the next submit (steps in flight keep theirs)
  tables_dirty_ = true;
}

// Restores (before the step's layers) or saves (after them; card 1: after the hidden states) slot states on card k's
// stream.
void Glm5ResidentModel::state_copies(int k, const std::vector<StateOp>& ops, bool save) {
  Card& cd = card_[k];
  for (const StateOp& op : ops) {
    uint8_t* snap = snap_host_[k] + size_t(op.snapshot) * snap_bytes_[k];
    for (const SnapSeg& s : snap_segs_[k]) {
      uint8_t* slot = s.dev + size_t(op.slot) * s.bytes;
      if (save) k::copy_kernel(snap + s.off, slot, s.bytes, cd.st);
      else k::copy_kernel(slot, snap + s.off, s.bytes, cd.st);
    }
  }
}

uint64_t Glm5ResidentModel::submit(const StepInput& in) {
  const int T = int(in.ids.size()), R = int(in.logit_rows.size()), D = c_.dim;
  if (T == 0 || T > max_tokens_) throw std::runtime_error("step: bad token count");
  if (R > max_logit_rows_) throw std::runtime_error("step: too many logit rows");
  if (!in.image_spans.empty()) throw std::runtime_error("step: GLM image tokens are causal (no spans)");
  if (!in.image_rows.empty() && !vision_) throw std::runtime_error("step: image rows without a vision tower");
  for (const auto* ops : {&in.restore, &in.save})
    for (const StateOp& op : *ops)
      if (op.snapshot < 0 || op.snapshot >= snapshots_ || op.slot < 0 || op.slot >= geo_.max_slots)
        throw std::runtime_error("step: bad state op");
  if (!in.backup.empty() && !mtp_) throw std::runtime_error("step: verify chunks without a drafter");
  const uint64_t ticket = next_ticket_;
  Flight& f = flight_[ticket % 2];
  if (f.busy) throw std::runtime_error("step: too many steps in flight");
  const size_t row = size_t(c_.hc_mult) * D;
  // Image rows into this flight's pinned buffer, in order; img_at: batch row -> its image row (null: a token's).
  std::vector<const uint16_t*> img_at;
  if (!in.image_rows.empty()) {
    img_at.assign(size_t(T), nullptr);
    size_t off = 0;
    for (const ImageRows& im : in.image_rows) {
      if (im.t0 < 0 || im.n <= 0 || im.t0 + im.n > T || off + size_t(im.n) > size_t(T))
        throw std::runtime_error("step: image rows outside the batch");
      std::memcpy(f.img_host + off * D, im.rows, size_t(im.n) * D * 2);
      for (int i = 0; i < im.n; ++i) img_at[size_t(im.t0 + i)] = im.rows + size_t(i) * D;
      off += size_t(im.n);
    }
  }
  std::memcpy(f.io_host, in.ids.data(), size_t(T) * 4);
  std::memcpy(f.io_host + max_tokens_, in.pos.data(), size_t(T) * 4);
  std::memcpy(f.io_host + 2 * max_tokens_, in.logit_rows.data(), size_t(R) * 4);
  const bool tables = tables_dirty_;
  if (tables) std::memcpy(f.tables_host, tables_.data(), tables_.size() * 4);
  tables_dirty_ = false;
  model::Glm5Batch b;
  for (const Chunk& ch : in.chunks) b.chunks.push_back({ch.slot, ch.s0, ch.n, ch.t0});
  // A verify chunk (its token and one draft) backs up the KDA states after its first token.
  for (const RingBackup& bk : in.backup) {
    if (bk.n > 2) throw std::runtime_error("step: GLM verifies one draft per chunk");
    for (auto& ch : b.chunks)
      if (ch.slot == bk.slot && ch.s0 == bk.s0) ch.save_after = 0;
  }
  b.T = T;
  b.max_pages = geo_.max_pages_per_seq;
  // MTP cache rows of the step: per chunk, positions s0 - 1 (from the slot's saved hidden state) .. s0 + n - 2, each
  // from its hidden state and the next position's embedding (host rows from the checkpoint's table, or its image row).
  model::Glm5Batch mb;
  int M = 0;
  if (mtp_) {
    for (const Chunk& ch : in.chunks) {
      const bool last = ch.s0 > 0;
      const int cnt = ch.n - 1 + (last ? 1 : 0);
      if (cnt <= 0) continue;
      const int first_tok = last ? ch.t0 : ch.t0 + 1;  // batch row of the first row's next-position token
      for (int j = 0; j < cnt; ++j) {
        const int32_t tok = in.ids[size_t(first_tok + j)];
        if (tok < 0 || tok >= c_.vocab) throw std::runtime_error("step: token id outside the vocabulary");
        const uint16_t* img = img_at.empty() ? nullptr : img_at[size_t(first_tok + j)];
        std::memcpy(f.emb_host + size_t(M + j) * D, img ? img : embed_host_.data() + size_t(tok) * D, size_t(D) * 2);
      }
      mb.chunks.push_back({ch.slot, last ? ch.s0 - 1 : ch.s0, cnt, M});
      M += cnt;
    }
    mb.T = M;
    mb.max_pages = geo_.max_pages_per_seq;
  }
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    // Staging moves by kernels pulling from pinned host memory, ordered with the stream (DECISIONS 2026-10-12).
    if (k == 0) k::copy_kernel(cd.ids.data(), f.io_host, size_t(T) * 4, cd.st);
    k::copy_kernel(cd.pos.data(), f.io_host + max_tokens_, size_t(T) * 4, cd.st);
    if (tables) k::copy_kernel(cd.tables.data(), f.tables_host, tables_.size() * 4, cd.st);
    b.pos = cd.pos.as<int32_t>();
    b.tables = cd.tables.as<int32_t>();
    if (k == 0) {
      k::embed_hc(cd.ids.as<int32_t>(), embed_.embed.p<uint16_t>(), cd.x.as<uint16_t>(), T, D, c_.hc_mult, cd.st,
                  c_.vocab);
      size_t off = 0;  // image rows over their tokens' embeddings, from this flight's pinned buffer
      for (const ImageRows& im : in.image_rows) {
        k::put_rows_hc(f.img_host + off * D, im.n, cd.x.as<uint16_t>(), im.t0, D, c_.hc_mult, cd.st);
        off += size_t(im.n);
      }
    } else {  // the residual from card 0 through this flight's pinned buffer
      if (mtp_) UH_CHECK(hipStreamWaitEvent(cd.st, drafted_, 0));  // the last draft()'s restores
      UH_CHECK(hipStreamWaitEvent(cd.st, f.handoff, 0));
      k::copy_kernel(cd.x.data(), f.x_host, size_t(T) * row * 2, cd.st);
      UH_CHECK(hipEventRecord(f.consumed, cd.st));
    }
    state_copies(k, in.restore, false);
    static const bool split = [] {
      const char* e = std::getenv("UNHOLY_GLM_SPLIT");
      return !e || std::string(e) != "0";
    }();
    model::glm5_layers_fast(c_, cd.w, cd.cache, cd.scratch, cd.x.as<uint16_t>(), b, cd.st, chan_[k].get(),
                            split ? chan2_[k].get() : nullptr);
    if (k == 0) {
      UH_CHECK(hipStreamWaitEvent(cd.st, f.consumed, 0));  // card 1 may still read it for the step two back
      k::copy_kernel(f.x_host, cd.x.data(), size_t(T) * row * 2, cd.st);
      UH_CHECK(hipEventRecord(f.handoff, cd.st));
      state_copies(0, in.save, true);  // after the handoff: card 1 need not wait for it
    }
  }
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  uint16_t* h_all = h_all_.as<uint16_t>();
  if (mtp_) {  // every row's final-normed hidden state (the head's input, the MTP layer's)
    k::hc_mean(cd.x.as<uint16_t>(), m_x0_.as<uint16_t>(), T, c_.hc_mult, D, cd.st);
    k::rmsnorm_slices(m_x0_.as<uint16_t>(), D, head_.norm.p<uint16_t>(), h_all, D, T, D, c_.norm_eps, cd.st);
    if (M > 0) {
      k::copy_kernel(m_emb_.data(), f.emb_host, size_t(M) * D * 2, cd.st);
      uint16_t* mh = m_h_.as<uint16_t>();
      size_t ci = 0;
      for (const Chunk& ch : in.chunks) {
        const bool last = ch.s0 > 0;
        if (ch.n - 1 + (last ? 1 : 0) <= 0) continue;
        const int r0 = mb.chunks[ci++].t0;
        if (last)
          UH_CHECK(hipMemcpyAsync(mh + size_t(r0) * D, h_last_.as<uint16_t>() + size_t(ch.slot) * D, size_t(D) * 2,
                                  hipMemcpyDeviceToDevice, cd.st));
        if (ch.n > 1)
          UH_CHECK(hipMemcpyAsync(mh + size_t(r0 + last) * D, h_all + size_t(ch.t0) * D, size_t(ch.n - 1) * D * 2,
                                  hipMemcpyDeviceToDevice, cd.st));
      }
      mb.pos = cd.pos.as<int32_t>();
      mb.tables = cd.tables.as<int32_t>();
      model::glm5_mtp_in(c_, mtpw_, m_emb_.as<uint16_t>(), mh, m_x0_.as<uint16_t>(), M, cd.scratch, cd.st);
      model::glm5_mtp_cache(c_, mtpw_, mcache_, cd.scratch, m_x0_.as<uint16_t>(), mb, cd.st);
    }
    for (const Chunk& ch : in.chunks) {  // the slots' last two hidden states (the previous one for a rollback)
      uint16_t* last = h_last_.as<uint16_t>() + size_t(ch.slot) * D;
      uint16_t* prev = h_prev_.as<uint16_t>() + size_t(ch.slot) * D;
      UH_CHECK(hipMemcpyAsync(prev, ch.n >= 2 ? h_all + size_t(ch.t0 + ch.n - 2) * D : last, size_t(D) * 2,
                              hipMemcpyDeviceToDevice, cd.st));
      UH_CHECK(hipMemcpyAsync(last, h_all + size_t(ch.t0 + ch.n - 1) * D, size_t(D) * 2, hipMemcpyDeviceToDevice,
                              cd.st));
    }
  }
  state_copies(1, in.save, true);
  if (R > 0) {
    k::copy_kernel(rows_idx_.data(), f.io_host + 2 * max_tokens_, size_t(R) * 4, cd.st);
    if (mtp_) {  // the normed rows are at hand
      k::gather_rows(h_all, rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, D, cd.st);
      model::glm5_head_logits(c_, head_, cd.scratch, hrows_.as<uint16_t>(), R, logits_.as<float>(), cd.st);
    } else {
      k::gather_rows(cd.x.as<uint16_t>(), rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, int(row), cd.st);
      model::glm5_head_forward_fast(c_, head_, cd.scratch, hrows_.as<uint16_t>(), R, logits_.as<float>(), cd.st);
    }
    k::copy_kernel(f.logits_host, logits_.data(), size_t(R) * c_.vocab * 4, cd.st);
  }
  UH_CHECK(hipEventRecord(f.done, cd.st));
  f.busy = true;
  f.ticket = ticket;
  f.rows = R;
  return next_ticket_++;
}

// Restores first (card 0's KDA states on its stream, ahead of its next step; card 1's and the slots' hidden states on
// the draft stream, which card 1's next step waits for), then one MTP draft per request on the draft stream, its own
// scratch and APU channel, alongside a step card 1 may be running for other sequences.
void Glm5ResidentModel::draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs,
                              std::vector<int32_t>& tokens, std::vector<float>& conf) {
  tokens.clear();
  conf.clear();
  if (!mtp_) throw std::runtime_error("draft: no MTP layer loaded");
  const int D = c_.dim, R = int(reqs.size());
  for (const RingRestore& r : restore)
    if (r.from != r.s0 + 1 || r.slot < 0 || r.slot >= geo_.max_slots)
      throw std::runtime_error("draft: GLM restores only a rejected single draft");
  if (!restore.empty()) {
    Card& c0 = card_[0];
    rt::DeviceGuard g0(c0.dev);
    for (const RingRestore& r : restore)
      for (auto& cache : c0.cache) model::glm5_restore_kda(cache, r.slot, c0.st);
  }
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  for (const RingRestore& r : restore) {
    for (auto& cache : cd.cache) model::glm5_restore_kda(cache, r.slot, dst_);
    UH_CHECK(hipMemcpyAsync(h_last_.as<uint16_t>() + size_t(r.slot) * D, h_prev_.as<uint16_t>() + size_t(r.slot) * D,
                            size_t(D) * 2, hipMemcpyDeviceToDevice, dst_));
  }
  if (R > dscr_.max_tokens) throw std::runtime_error("draft: too many requests");
  if (R > 0) {
    int32_t* pos_host = d_tok_host_ + 2 * R;
    model::Glm5Batch b;
    for (int r = 0; r < R; ++r) {
      const DraftReq& q = reqs[size_t(r)];
      if (q.token < 0 || q.token >= c_.vocab) throw std::runtime_error("draft: token id outside the vocabulary");
      std::memcpy(d_emb_host_ + size_t(r) * D, embed_host_.data() + size_t(q.token) * D, size_t(D) * 2);
      pos_host[r] = q.start;
      b.chunks.push_back({q.slot, q.start, 1, r});
      UH_CHECK(hipMemcpyAsync(d_h_.as<uint16_t>() + size_t(r) * D, h_last_.as<uint16_t>() + size_t(q.slot) * D,
                              size_t(D) * 2, hipMemcpyDeviceToDevice, dst_));
    }
    k::copy_kernel(d_emb_.data(), d_emb_host_, size_t(R) * D * 2, dst_);
    int32_t* dpos = d_tok_.as<int32_t>() + 2 * R;
    k::copy_kernel(dpos, pos_host, size_t(R) * 4, dst_);
    b.T = R;
    b.pos = dpos;
    b.tables = cd.tables.as<int32_t>();
    b.max_pages = geo_.max_pages_per_seq;
    model::glm5_mtp_in(c_, mtpw_, d_emb_.as<uint16_t>(), d_h_.as<uint16_t>(), d_x_.as<uint16_t>(), R, dscr_, dst_);
    model::glm5_mtp_forward(c_, mtpw_, mcache_, dscr_, d_x_.as<uint16_t>(), d_hn_.as<uint16_t>(), b, dst_,
                            dchan_.get());
    model::glm5_head_logits(c_, head_, dscr_, d_hn_.as<uint16_t>(), R, d_logits_.as<float>(), dst_);
    k::glm_argmax_conf(d_logits_.as<float>(), R, c_.vocab, d_tok_.as<int32_t>(),
                       reinterpret_cast<float*>(d_tok_.as<int32_t>() + R), dst_);
    k::copy_kernel(d_tok_host_, d_tok_.data(), size_t(R) * 2 * 4, dst_);
  }
  UH_CHECK(hipEventRecord(drafted_, dst_));
  UH_CHECK(hipStreamSynchronize(dst_));
  tokens.assign(d_tok_host_, d_tok_host_ + R);
  const float* cf = reinterpret_cast<const float*>(d_tok_host_ + R);
  conf.assign(cf, cf + R);
}

void Glm5ResidentModel::wait(uint64_t ticket, std::vector<float>& out) {
  Flight& f = flight_[ticket % 2];
  if (!f.busy || f.ticket != ticket) throw std::runtime_error("step: wait for an unknown ticket");
  {
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventSynchronize(f.done));
  }
  out.resize(size_t(f.rows) * c_.vocab);
  std::memcpy(out.data(), f.logits_host, out.size() * 4);
  f.busy = false;
}

}  // namespace unholy::engine
