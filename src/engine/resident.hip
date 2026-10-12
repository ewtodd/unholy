#include "engine/resident.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels/gemm_wmma.hpp"
#include "kernels/gemv.hpp"
#include "kernels/ref.hpp"
#include "model/loader.hpp"
#include "rt/hip.hpp"

namespace unholy::engine {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// ApuExperts

ApuExperts::ApuExperts(const model::Dsv4Config& c, int dev, size_t capacity_rows, std::shared_ptr<Layers> layers)
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
  // Each job is one expert: at most every token of a step.
  scratch_ = model::make_expert_scratch(c, dev, int(capacity_rows));
  xscratch_ = model::make_exl3_moe_scratch(c, dev, int(capacity_rows));
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
                             int max_tokens, int max_seq, int max_logit_rows)
    : c_(c), plan_(plan), geo_(geo), max_tokens_(max_tokens),
      max_logit_rows_(max_logit_rows > 0 ? std::min(max_logit_rows, max_tokens) : max_tokens) {
  model::Uploader up(dense);
  std::unique_ptr<model::Uploader> xup;
  if (exl3) xup = std::make_unique<model::Uploader>(*exl3);
  {
    auto layers = std::make_shared<ApuExperts::Layers>();
    for (auto& ch : chan_) ch = std::make_unique<ApuExperts>(c, plan.apu, size_t(max_tokens) * c.n_active, layers);
  }
  if (const char* v = std::getenv("UNHOLY_INFLIGHT")) inflight_ = std::clamp(std::atoi(v), 1, 2);
  for (int k = 0; k < 2; ++k) card_arena_[k] = std::make_unique<rt::Arena>(plan.card[k], size_t(256) << 20);
  apu_arena_ = std::make_unique<rt::Arena>(plan.apu, size_t(256) << 20);

  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    cd.dev = plan.card[k];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipStreamCreateWithFlags(&cd.st, hipStreamNonBlocking));
    cd.scratch = model::make_scratch(c, cd.dev, max_tokens, max_seq);
    cd.rope = model::make_rope_tables(c, cd.dev, max_seq + 128);
    cd.x = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    cd.ids = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.pos = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
    cd.tables = rt::DeviceBuffer(cd.dev, size_t(geo.max_slots) * geo.max_pages_per_seq * 4);
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
  for (Flight& f : flight_) {
    const auto pinned = [](auto** p, size_t bytes) {
      UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(p), bytes, hipHostMallocPortable));
    };
    pinned(&f.x_host, size_t(max_tokens) * c.hc_mult * c.dim * 2);
    pinned(&f.logits_host, size_t(max_logit_rows_) * c.vocab * 4);
    pinned(&f.ids_host, size_t(max_tokens) * 3 * 4);
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

ResidentModel::~ResidentModel() {
  for (auto& cd : card_)
    if (cd.st) {
      rt::DeviceGuard g(cd.dev);
      (void)hipStreamDestroy(cd.st);
    }
  for (Flight& f : flight_) {
    for (void* p : {static_cast<void*>(f.x_host), static_cast<void*>(f.logits_host), static_cast<void*>(f.ids_host),
                    static_cast<void*>(f.tables_host)})
      if (p) (void)hipHostFree(p);
    for (hipEvent_t e : {f.handoff, f.consumed, f.done})
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

uint64_t ResidentModel::submit(const StepInput& in) {
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
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemcpyAsync(cd.ids.data(), f.ids_host, size_t(T) * 4, hipMemcpyHostToDevice, cd.st));
    UH_CHECK(hipMemcpyAsync(cd.pos.data(), f.ids_host + max_tokens_, size_t(T) * 4, hipMemcpyHostToDevice, cd.st));
    if (tables)
      UH_CHECK(hipMemcpyAsync(cd.tables.data(), f.tables_host, tables_.size() * 4, hipMemcpyHostToDevice, cd.st));
  }
  model::BatchDesc b;
  for (const Chunk& ch : in.chunks) b.chunks.push_back({ch.slot, ch.s0, ch.n, ch.t0});
  b.T = T;
  b.max_pages_per_seq = geo_.max_pages_per_seq;
  std::chrono::steady_clock::time_point t_enq[2];
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    if (k == 0) {
      k::embed_hc(cd.ids.as<int32_t>(), embed_.as<uint16_t>(), cd.x.as<uint16_t>(), T, c_.dim, c_.hc_mult, cd.st);
    } else {  // residual from card 0 through this flight's pinned buffer, ordered by events
      UH_CHECK(hipStreamWaitEvent(cd.st, f.handoff, 0));
      UH_CHECK(hipMemcpyAsync(cd.x.data(), f.x_host, size_t(T) * row * 2, hipMemcpyHostToDevice, cd.st));
      UH_CHECK(hipEventRecord(f.consumed, cd.st));
    }
    b.ids = cd.ids.as<int32_t>();
    b.pos = cd.pos.as<int32_t>();
    b.block_tables = cd.tables.as<int32_t>();
    for (size_t i = 0; i < cd.layers.size(); ++i)
      model::layer_forward(c_, cd.w[i], cd.cache[i], cd.scratch, cd.rope, cd.x.as<uint16_t>(), b, cd.st,
                           chan_[k].get());
    if (k == 0) {
      // The flight's buffer may still be read by card 1 for the step two tickets back.
      UH_CHECK(hipStreamWaitEvent(cd.st, f.consumed, 0));
      UH_CHECK(hipMemcpyAsync(f.x_host, cd.x.data(), size_t(T) * row * 2, hipMemcpyDeviceToHost, cd.st));
      UH_CHECK(hipEventRecord(f.handoff, cd.st));
    }
    if (timing) t_enq[k] = std::chrono::steady_clock::now();
  }
  // Head on card 1 for the requested rows.
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  if (R > 0) {
    UH_CHECK(hipMemcpyAsync(rows_idx_.data(), f.ids_host + 2 * max_tokens_, size_t(R) * 4, hipMemcpyHostToDevice,
                            cd.st));
    k::gather_rows(cd.x.as<uint16_t>(), rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, int(row), cd.st);
    k::hc_head(hrows_.as<uint16_t>(), hc_fn_.as<float>(), hc_scale_.as<float>(), hc_base_.as<float>(),
               hb_.as<uint16_t>(), R, c_.hc_mult, c_.dim, c_.norm_eps, c_.hc_eps, cd.st);
    k::rmsnorm(hb_.as<uint16_t>(), c_.dim, norm_.as<uint16_t>(), hb_.as<uint16_t>(), c_.dim, R, c_.dim, c_.norm_eps,
               cd.st);
    // The reference head is an fp32 linear on the bf16-valued normed state: bf16 WMMA, exact products.
    if (R <= k::kGemvMaxRows)
      k::gemv_bf16_f32out(hb_.as<uint16_t>(), c_.dim, head_.as<uint16_t>(), logits_.as<float>(), c_.vocab, R,
                          c_.vocab, c_.dim, cd.st);
    else
      k::gemm_bf16_f32out_wmma(hb_.as<uint16_t>(), c_.dim, head_.as<uint16_t>(), logits_.as<float>(), c_.vocab, R,
                               c_.vocab, c_.dim, cd.st);
    UH_CHECK(hipMemcpyAsync(f.logits_host, logits_.data(), size_t(R) * c_.vocab * 4, hipMemcpyDeviceToHost, cd.st));
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
  out.resize(size_t(f.rows) * c_.vocab);
  std::memcpy(out.data(), f.logits_host, out.size() * 4);
  f.busy = false;
}

}  // namespace unholy::engine
