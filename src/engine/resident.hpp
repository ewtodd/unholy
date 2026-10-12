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
#include "engine/step.hpp"
#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"

namespace unholy::engine {

// The APU's cold experts as seen by one card: its own stream, flags, pinned staging and scratch, so
// both cards can have expert work in flight at once; the expert weights are shared between channels.
class ApuExperts final : public model::RemoteExperts {
 public:
  using Layers = std::vector<std::unique_ptr<model::LayerWeights>>;  // [layer]
  ApuExperts(const model::Dsv4Config& c, int dev, size_t capacity_rows, std::shared_ptr<Layers> layers);
  ~ApuExperts() override;
  void add_layer(model::LayerWeights w);  // experts-only weights of one layer, on this device
  void run(int layer, const std::vector<Job>& jobs) override;
  bool can_enqueue() const override { return true; }
  void enqueue(int layer, hipStream_t card_st, int P) override;
  void join(hipStream_t card_st) override;
  int device() const { return dev_; }
  hipStream_t stream() const { return st_; }

 private:
  const model::Dsv4Config& c_;
  int dev_;
  hipStream_t st_ = nullptr;
  std::shared_ptr<Layers> layers_;
  model::ExpertScratch scratch_;
  model::Exl3MoeScratch xscratch_;
  int32_t* e_dev_ = nullptr;
  kernels::MoePlan* plan_dev_ = nullptr;   // device views of plan_host / tiles_host
  kernels::Exl3Tile* tiles_dev_ = nullptr;
  uint32_t* req_flag_ = nullptr;           // pinned, coherent: card -> APU
  uint32_t* done_flag_ = nullptr;          // APU -> card
  uint32_t seq_ = 0;
  uint16_t* in_dev_ = nullptr;
  float* w_dev_ = nullptr;
  uint16_t* out_dev_ = nullptr;
};

class ResidentModel final : public StepModel {
 public:
  // max_logit_rows: rows of logits one step can return (0: max_tokens).
  ResidentModel(const model::Dsv4Config& c, const model::Checkpoint& dense, const model::Checkpoint* exl3,
                const Plan& plan, const model::CacheGeometry& geo, int max_tokens, int max_seq,
                int max_logit_rows = 0);
  ~ResidentModel() override;

  int vocab() const override { return c_.vocab; }
  int max_tokens() const override { return max_tokens_; }
  int max_logit_rows() const override { return max_logit_rows_; }
  int max_slots() const override { return geo_.max_slots; }
  int max_pages_per_seq() const override { return geo_.max_pages_per_seq; }
  int num_pages() const override { return geo_.num_pages; }
  int page_tokens() const override { return model::kPageTokens; }
  // Block tables [max_slots][max_pages_per_seq] (host), uploaded to both cards.
  void set_block_tables(const std::vector<int32_t>& tables) override;
  // Two steps in flight: card 0 runs step k + 1 while card 1 finishes step k (UNHOLY_INFLIGHT=1: one).
  int max_inflight() const override { return inflight_; }
  uint64_t submit(const StepInput& in) override;
  void wait(uint64_t ticket, std::vector<float>& logits) override;

  const Plan& plan() const { return plan_; }
  // Pairs routed to each expert since load (or the last reset), [layer][expert]; usage-profile format
  // via format_usage_profile(). Thread-safe against step() (its own stream; counts are approximate
  // while a step runs).
  std::vector<std::vector<uint64_t>> routing_histogram() const;
  void reset_routing_histogram();

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
  int max_tokens_ = 0, max_logit_rows_ = 0;
  Card card_[2];
  std::unique_ptr<ApuExperts> chan_[2];  // card k's channel to the APU experts
  std::unique_ptr<rt::Arena> card_arena_[2], apu_arena_;  // weights (256 MiB slabs)
  rt::DeviceBuffer embed_, head_, norm_, hc_fn_, hc_scale_, hc_base_;  // embed on card 0, the rest card 1
  rt::DeviceBuffer hrows_, hb_, hf_, logits_, rows_idx_;              // head scratch on card 1
  // Per in-flight step (ticket % 2): pinned staging and the events ordering the two cards.
  struct Flight {
    uint16_t* x_host = nullptr;   // residual handoff card 0 -> card 1
    float* logits_host = nullptr; // [max_logit_rows][vocab]
    int32_t* ids_host = nullptr;  // [max_tokens] ids, then positions, then logit rows
    int32_t* tables_host = nullptr;
    hipEvent_t handoff = nullptr;   // card 0: x_host written
    hipEvent_t consumed = nullptr;  // card 1: x_host read
    hipEvent_t done = nullptr;      // card 1: logits_host written
    uint64_t ticket = 0;
    int rows = 0;
    bool busy = false;
  };
  Flight flight_[2];
  uint64_t next_ticket_ = 1;
  int inflight_ = 2;
  std::vector<int32_t> tables_;
  bool tables_dirty_ = false;
};

}  // namespace unholy::engine
