#include "engine/resident.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "kernels/ref.hpp"
#include "model/loader.hpp"
#include "rt/hip.hpp"

namespace unholy::engine {

namespace k = unholy::kernels;

// ---------------------------------------------------------------------------------------------------
// ApuExperts

ApuExperts::ApuExperts(const model::Dsv4Config& c, int dev, size_t capacity_rows) : c_(c), dev_(dev) {
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
  layers_.resize(size_t(c.n_layers));
}

ApuExperts::~ApuExperts() {
  rt::DeviceGuard g(dev_);
  if (st_) (void)hipStreamDestroy(st_);
  if (in_host) (void)hipHostFree(in_host);
  if (out_host) (void)hipHostFree(out_host);
  if (w_host) (void)hipHostFree(w_host);
  if (e_host) (void)hipHostFree(e_host);
}

void ApuExperts::add_layer(model::LayerWeights w) {
  const int L = w.layer;
  layers_.at(size_t(L)) = std::make_unique<model::LayerWeights>(std::move(w));
}

void ApuExperts::run(int layer, const std::vector<Job>& jobs) {
  const auto& w = *layers_.at(size_t(layer));
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
                             int max_tokens, int max_seq)
    : c_(c), plan_(plan), geo_(geo), max_tokens_(max_tokens) {
  model::Uploader up(dense);
  std::unique_ptr<model::Uploader> xup;
  if (exl3) xup = std::make_unique<model::Uploader>(*exl3);
  apu_ = std::make_unique<ApuExperts>(c, plan.apu, size_t(max_tokens) * c.n_active);
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
      apu_->add_layer(model::load_layer(dense, c, L, plan.apu, apu_->stream(), la));
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
    logits_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * c.vocab * 4);
    rows_idx_ = rt::DeviceBuffer(cd.dev, size_t(max_tokens) * 4);
  }
  UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&x_host_), size_t(max_tokens) * c.hc_mult * c.dim * 2,
                         hipHostMallocPortable));
}

ResidentModel::~ResidentModel() {
  for (auto& cd : card_)
    if (cd.st) {
      rt::DeviceGuard g(cd.dev);
      (void)hipStreamDestroy(cd.st);
    }
  if (x_host_) (void)hipHostFree(x_host_);
}

void ResidentModel::set_block_tables(const std::vector<int32_t>& tables) {
  if (tables.size() != size_t(geo_.max_slots) * geo_.max_pages_per_seq)
    throw std::runtime_error("set_block_tables: wrong size");
  for (auto& cd : card_) {
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemcpy(cd.tables.data(), tables.data(), tables.size() * 4, hipMemcpyHostToDevice));
  }
}

std::vector<float> ResidentModel::step(const StepInput& in) {
  const int T = int(in.ids.size());
  if (T == 0 || T > max_tokens_) throw std::runtime_error("step: bad token count");
  const size_t row = size_t(c_.hc_mult) * c_.dim;
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    UH_CHECK(hipMemcpyAsync(cd.ids.data(), in.ids.data(), size_t(T) * 4, hipMemcpyHostToDevice, cd.st));
    UH_CHECK(hipMemcpyAsync(cd.pos.data(), in.pos.data(), size_t(T) * 4, hipMemcpyHostToDevice, cd.st));
  }
  // Embedding on card 0.
  {
    Card& cd = card_[0];
    rt::DeviceGuard g(cd.dev);
    k::embed_hc(cd.ids.as<int32_t>(), embed_.as<uint16_t>(), cd.x.as<uint16_t>(), T, c_.dim, c_.hc_mult, cd.st);
  }
  for (int k = 0; k < 2; ++k) {
    Card& cd = card_[k];
    rt::DeviceGuard g(cd.dev);
    if (k == 1) {  // residual handoff through pinned host memory
      UH_CHECK(hipMemcpyAsync(cd.x.data(), x_host_, size_t(T) * row * 2, hipMemcpyHostToDevice, cd.st));
    }
    model::BatchDesc b;
    b.chunks = in.chunks;
    b.T = T;
    b.ids = cd.ids.as<int32_t>();
    b.pos = cd.pos.as<int32_t>();
    b.block_tables = cd.tables.as<int32_t>();
    b.max_pages_per_seq = geo_.max_pages_per_seq;
    for (size_t i = 0; i < cd.layers.size(); ++i)
      model::layer_forward(c_, cd.w[i], cd.cache[i], cd.scratch, cd.rope, cd.x.as<uint16_t>(), b, cd.st, apu_.get());
    if (k == 0) {
      UH_CHECK(hipMemcpyAsync(x_host_, cd.x.data(), size_t(T) * row * 2, hipMemcpyDeviceToHost, cd.st));
      UH_CHECK(hipStreamSynchronize(cd.st));
    }
  }
  // Head on card 1 for the requested rows.
  const int R = int(in.logit_rows.size());
  std::vector<float> out(size_t(R) * c_.vocab);
  if (R == 0) return out;
  Card& cd = card_[1];
  rt::DeviceGuard g(cd.dev);
  UH_CHECK(hipMemcpyAsync(rows_idx_.data(), in.logit_rows.data(), size_t(R) * 4, hipMemcpyHostToDevice, cd.st));
  k::gather_rows(cd.x.as<uint16_t>(), rows_idx_.as<int32_t>(), hrows_.as<uint16_t>(), R, int(row), cd.st);
  k::hc_head(hrows_.as<uint16_t>(), hc_fn_.as<float>(), hc_scale_.as<float>(), hc_base_.as<float>(), hb_.as<uint16_t>(),
             R, c_.hc_mult, c_.dim, c_.norm_eps, c_.hc_eps, cd.st);
  k::rmsnorm(hb_.as<uint16_t>(), c_.dim, norm_.as<uint16_t>(), hb_.as<uint16_t>(), c_.dim, R, c_.dim, c_.norm_eps,
             cd.st);
  k::cast_bf16_f32(hb_.as<uint16_t>(), hf_.as<float>(), size_t(R) * c_.dim, cd.st);
  k::gemm_f32(hf_.as<float>(), c_.dim, head_.as<uint16_t>(), logits_.as<float>(), c_.vocab, R, c_.vocab, c_.dim,
              cd.st);
  UH_CHECK(hipMemcpyAsync(out.data(), logits_.data(), out.size() * 4, hipMemcpyDeviceToHost, cd.st));
  UH_CHECK(hipStreamSynchronize(cd.st));
  return out;
}

}  // namespace unholy::engine
