#include "engine/glm5_resident.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/ref.hpp"
#include "model/loader.hpp"
#include "rt/hip.hpp"

namespace unholy::engine {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// Glm5ApuExperts

Glm5ApuExperts::Glm5ApuExperts(const model::Glm5Config& c, int dev, size_t capacity_rows,
                               std::shared_ptr<Layers> layers)
    : c_(c), dev_(dev), layers_(std::move(layers)) {
  rt::DeviceGuard g(dev);
  UH_CHECK(hipStreamCreateWithFlags(&st_, hipStreamNonBlocking));
  this->capacity_rows = capacity_rows;
  const size_t D = c.dim, I = c.moe_inter, cap = capacity_rows;
  max_tiles = int(cap / k::kExl3TileRows + c.n_experts + 1);
  const unsigned flags = hipHostMallocPortable | hipHostMallocMapped;
  const auto pinned = [&](auto** host, auto** dev_ptr, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(host), bytes, flags));
    if (dev_ptr) UH_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(dev_ptr), *host, 0));
  };
  pinned(&in_host, &in_dev_, cap * D * 2);
  pinned(&out_host, &out_dev_, cap * D * 2);
  pinned(&w_host, &w_dev_, cap * 4);
  pinned(&e_host, &e_dev_, cap * 4);
  pinned(&plan_host, &plan_dev_, sizeof(k::MoePlan));
  pinned(&tiles_host, &tiles_dev_, size_t(max_tiles) * sizeof(k::Exl3Tile));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&req_flag_), 64, flags | hipHostMallocCoherent));
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&done_flag_), 64, flags | hipHostMallocCoherent));
  *req_flag_ = 0;
  *done_flag_ = 0;
  xh1_ = rt::DeviceBuffer(dev, cap * D * 2);
  xh3_ = rt::DeviceBuffer(dev, cap * D * 2);
  xh2_ = rt::DeviceBuffer(dev, cap * I * 2);
  h1_ = rt::DeviceBuffer(dev, cap * I * 2);
  h3_ = rt::DeviceBuffer(dev, cap * I * 2);
  yp_ = rt::DeviceBuffer(dev, std::min<size_t>(cap, 64) * std::max(D, I) * 4);
  if (layers_->size() < size_t(c.n_layers)) layers_->resize(size_t(c.n_layers));
}

Glm5ApuExperts::~Glm5ApuExperts() {
  rt::DeviceGuard g(dev_);
  if (st_) (void)hipStreamDestroy(st_);
  for (void* p : {static_cast<void*>(in_host), static_cast<void*>(out_host), static_cast<void*>(w_host),
                  static_cast<void*>(e_host), static_cast<void*>(plan_host), static_cast<void*>(tiles_host),
                  static_cast<void*>(req_flag_), static_cast<void*>(done_flag_)})
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

// The card stream raises req after staging the inputs; the APU stream waits for it, runs the remote tiles, raises
// done; join() makes the card stream wait for done (sequence numbers: no resets, no host waits).
void Glm5ApuExperts::enqueue(int layer, hipStream_t card_st, int P) {
  const auto* w = layers_->at(size_t(layer)).get();
  if (!w) throw std::runtime_error("Glm5ApuExperts: no experts of layer " + std::to_string(layer) + " on the APU");
  ++seq_;
  UH_CHECK(hipStreamWriteValue32(card_st, req_flag_, seq_, 0));
  rt::DeviceGuard g(dev_);
  UH_CHECK(hipStreamWaitValue32(st_, req_flag_, seq_, hipStreamWaitValueGte, 0xffffffffu));
  const model::Glm5ExpertBufs b{xh1_.as<uint16_t>(), xh3_.as<uint16_t>(), xh2_.as<uint16_t>(), h1_.as<uint16_t>(),
                                h3_.as<uint16_t>(), yp_.as<float>()};
  model::glm5_experts_dev(c_, *w, in_dev_, w_dev_, e_dev_, tiles_dev_, max_tiles, &plan_dev_->n_remote_tiles, P, b,
                          out_dev_, st_);
  UH_CHECK(hipStreamWriteValue32(st_, done_flag_, seq_, 0));
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

// APU bytes besides the cold experts: both channels' scratch.
size_t channel_bytes(const model::Glm5Config& c, size_t cap) {
  const size_t D = c.dim, I = c.moe_inter;
  return 2 * (2 * cap * D * 2 + 3 * cap * I * 2 + std::min<size_t>(cap, 64) * std::max(D, I) * 4);
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
    cache[size_t(L)] = model::glm5_paged_cache_bytes(c, c.is_kda(L), in.geo.max_slots, in.geo.num_pages);
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
  const int logit_rows = in.max_logit_rows > 0 ? in.max_logit_rows : in.max_tokens;
  for (int k = 0; k < 2; ++k) {
    pl.scratch[k] = model::glm5_scratch_bytes(c, in.max_tokens, in.max_seq) +
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
  const size_t apu_work = channel_bytes(c, size_t(in.max_tokens) * c.n_active);
  if (pl.apu_experts + apu_work + in.apu_margin > pl.apu_free)
    throw std::runtime_error("plan: the APU cannot hold the cold experts");
  return pl;
}

// ---------------------------------------------------------------------------------------------------
// Glm5ResidentModel

Glm5ResidentModel::Glm5ResidentModel(const model::Glm5Config& c, const model::Checkpoint& ck, const Plan& plan,
                                     const model::CacheGeometry& geo, int max_tokens, int max_seq, int max_logit_rows)
    : c_(c), plan_(plan), geo_(geo), max_tokens_(max_tokens),
      max_logit_rows_(max_logit_rows > 0 ? std::min(max_logit_rows, max_tokens) : max_tokens) {
  if (geo.max_pages_per_seq * k::kGlmPageRows < max_seq) throw std::runtime_error("glm5: block table too narrow");
  model::Uploader up(ck);
  {
    auto layers = std::make_shared<Glm5ApuExperts::Layers>();
    for (auto& ch : chan_) ch = std::make_unique<Glm5ApuExperts>(c, plan.apu, size_t(max_tokens) * c.n_active, layers);
  }
  for (int k = 0; k < 2; ++k) card_arena_[k] = std::make_unique<rt::Arena>(plan.card[k], size_t(256) << 20);
  apu_arena_ = std::make_unique<rt::Arena>(plan.apu, size_t(256) << 20);
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    cd.dev = plan.card[k];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipStreamCreateWithFlags(&cd.st, hipStreamNonBlocking));
    cd.scratch = model::make_glm5_scratch(c, cd.dev, max_tokens, max_seq);
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
      cd.cache.push_back(model::make_glm5_paged_cache(c, L, geo.max_slots, geo.num_pages, cd.dev, cd.st));
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
  const auto pinned = [](auto** p, size_t bytes) {
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(p), bytes, hipHostMallocPortable | hipHostMallocMapped));
  };
  for (Flight& f : flight_) {
    pinned(&f.x_host, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    pinned(&f.logits_host, size_t(max_logit_rows_) * c.vocab * 4);
    pinned(&f.io_host, size_t(max_tokens) * 3 * 4);
    pinned(&f.tables_host, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
    {
      rt::DeviceGuard g(card_[0].dev);
      UH_CHECK(hipEventCreateWithFlags(&f.handoff, hipEventDisableTiming));
    }
    rt::DeviceGuard g(card_[1].dev);
    UH_CHECK(hipEventCreateWithFlags(&f.consumed, hipEventDisableTiming));
    UH_CHECK(hipEventCreateWithFlags(&f.done, hipEventDisableTiming));
  }
}

Glm5ResidentModel::~Glm5ResidentModel() {
  for (auto& cd : card_)
    if (cd.st) {
      rt::DeviceGuard g(cd.dev);
      (void)hipStreamDestroy(cd.st);
    }
  for (Flight& f : flight_) {
    for (void* p : {static_cast<void*>(f.x_host), static_cast<void*>(f.logits_host), static_cast<void*>(f.io_host),
                    static_cast<void*>(f.tables_host)})
      if (p) (void)hipHostFree(p);
    for (hipEvent_t e : {f.handoff, f.consumed, f.done})
      if (e) (void)hipEventDestroy(e);
  }
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

uint64_t Glm5ResidentModel::submit(const StepInput& in) {
  const int T = int(in.ids.size()), R = int(in.logit_rows.size());
  if (T == 0 || T > max_tokens_) throw std::runtime_error("step: bad token count");
  if (R > max_logit_rows_) throw std::runtime_error("step: too many logit rows");
  if (!in.restore.empty() || !in.save.empty() || !in.backup.empty() || !in.image_rows.empty() ||
      !in.image_spans.empty())
    throw std::runtime_error("step: GLM serves without prefix snapshots, drafts or images");
  const uint64_t ticket = next_ticket_;
  Flight& f = flight_[ticket % 2];
  if (f.busy) throw std::runtime_error("step: too many steps in flight");
  const size_t row = size_t(c_.hc_mult) * c_.dim;
  std::memcpy(f.io_host, in.ids.data(), size_t(T) * 4);
  std::memcpy(f.io_host + max_tokens_, in.pos.data(), size_t(T) * 4);
  std::memcpy(f.io_host + 2 * max_tokens_, in.logit_rows.data(), size_t(R) * 4);
  const bool tables = tables_dirty_;
  if (tables) std::memcpy(f.tables_host, tables_.data(), tables_.size() * 4);
  tables_dirty_ = false;
  model::Glm5Batch b;
  for (const Chunk& ch : in.chunks) b.chunks.push_back({ch.slot, ch.s0, ch.n, ch.t0});
  b.T = T;
  b.max_pages = geo_.max_pages_per_seq;
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
      k::embed_hc(cd.ids.as<int32_t>(), embed_.embed.p<uint16_t>(), cd.x.as<uint16_t>(), T, c_.dim, c_.hc_mult, cd.st,
                  c_.vocab);
    } else {  // the residual from card 0 through this flight's pinned buffer
      UH_CHECK(hipStreamWaitEvent(cd.st, f.handoff, 0));
      k::copy_kernel(cd.x.data(), f.x_host, size_t(T) * row * 2, cd.st);
      UH_CHECK(hipEventRecord(f.consumed, cd.st));
    }
    for (size_t i = 0; i < cd.w.size(); ++i)
      model::glm5_layer_forward_fast(c_, cd.w[i], cd.cache[i], cd.scratch, cd.x.as<uint16_t>(), b, cd.st,
                                     chan_[k].get());
    if (k == 0) {
      UH_CHECK(hipStreamWaitEvent(cd.st, f.consumed, 0));  // card 1 may still read it for the step two back
      k::copy_kernel(f.x_host, cd.x.data(), size_t(T) * row * 2, cd.st);
      UH_CHECK(hipEventRecord(f.handoff, cd.st));
    }
  }
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  if (R > 0) {
    k::copy_kernel(rows_idx_.data(), f.io_host + 2 * max_tokens_, size_t(R) * 4, cd.st);
    k::gather_rows(cd.x.as<uint16_t>(), rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, int(row), cd.st);
    model::glm5_head_forward_fast(c_, head_, cd.scratch, hrows_.as<uint16_t>(), R, logits_.as<float>(), cd.st);
    k::copy_kernel(f.logits_host, logits_.data(), size_t(R) * c_.vocab * 4, cd.st);
  }
  UH_CHECK(hipEventRecord(f.done, cd.st));
  f.busy = true;
  f.ticket = ticket;
  f.rows = R;
  return next_ticket_++;
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
