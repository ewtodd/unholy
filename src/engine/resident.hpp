// The whole model resident across the two cards and the APU, per a Plan, on the reference kernels.
//
// card 0: embedding + its layers; card 1: its layers + head; the APU: cold experts of every layer,
// computed from / into pinned host memory (zero-copy for the APU; the cards DMA to it). The residual
// crosses from card 0 to card 1 once per step through pinned host memory. Orchestration is
// host-synchronous here; the fast path replaces it with device-side signalling.
#pragma once

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

#include "engine/plan.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"

namespace unholy::engine {

class ApuExperts final : public model::RemoteExperts {
 public:
  ApuExperts(const model::Dsv4Config& c, int dev, size_t capacity_rows);
  ~ApuExperts() override;
  void add_layer(model::LayerWeights w);  // experts-only weights of one layer, on this device
  void run(int layer, const std::vector<Job>& jobs) override;
  int device() const { return dev_; }
  hipStream_t stream() const { return st_; }

 private:
  const model::Dsv4Config& c_;
  int dev_;
  hipStream_t st_ = nullptr;
  std::vector<std::unique_ptr<model::LayerWeights>> layers_;  // [layer]
  model::ExpertScratch scratch_;
  uint16_t* in_dev_ = nullptr;
  float* w_dev_ = nullptr;
  uint16_t* out_dev_ = nullptr;
};

// One step's work: chunks over slots, flattened token ids/positions, and the batch rows whose logits
// are wanted.
struct StepInput {
  std::vector<model::ChunkDesc> chunks;
  std::vector<int32_t> ids, pos;
  std::vector<int32_t> logit_rows;
};

class ResidentModel {
 public:
  ResidentModel(const model::Dsv4Config& c, const model::Checkpoint& dense, const model::Checkpoint* exl3,
                const Plan& plan, const model::CacheGeometry& geo, int max_tokens, int max_seq);
  ~ResidentModel();

  // Block tables [max_slots][max_pages_per_seq] (host), uploaded to both cards.
  void set_block_tables(const std::vector<int32_t>& tables);
  // Runs one step; returns logits [logit_rows.size()][vocab] fp32.
  std::vector<float> step(const StepInput& in);

  const Plan& plan() const { return plan_; }

 private:
  struct Card {
    int dev = 0;
    hipStream_t st = nullptr;
    std::vector<int> layers;
    std::vector<model::LayerWeights> w;
    std::vector<model::LayerCache> cache;
    model::Scratch scratch;
    model::RopeTables rope;
    rt::DeviceBuffer x, ids, pos, tables;
  };
  const model::Dsv4Config& c_;
  Plan plan_;
  model::CacheGeometry geo_;
  int max_tokens_ = 0;
  Card card_[2];
  std::unique_ptr<ApuExperts> apu_;
  std::unique_ptr<rt::Arena> card_arena_[2], apu_arena_;  // weights (256 MiB slabs)
  rt::DeviceBuffer embed_, head_, norm_, hc_fn_, hc_scale_, hc_base_;  // embed on card 0, the rest card 1
  rt::DeviceBuffer hrows_, hb_, hf_, logits_, rows_idx_;              // head scratch on card 1
  uint16_t* x_host_ = nullptr;                                        // pinned residual handoff
};

}  // namespace unholy::engine
