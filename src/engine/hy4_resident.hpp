// Hy4 flash resident across the two cards and the APU (docs/PLAN-hy4-flash.md): the Hy4 counterpart of
// Glm5ResidentModel, on model/hy4.hpp.
//
// card 0: the embedding and its layers; card 1: its layers and the head; each layer's dense weights (bf16), its
// paged attention rows / window rings, and its hot routed experts (e4m3) on its card; the cold experts on the APU
// (Hy4ApuExperts: one channel per card, device-ordered flags, no host waits; one compact bf16 latent row per token
// crosses to the APU, one fp32 latent row per token comes back). Two steps in flight: card 0 runs step k + 1 while
// card 1 finishes step k; the residual crosses once per step through pinned memory. The Engram rows of a step are
// looked up on the host at submit (the store reads the SSD pack) into pinned memory the cards read directly; each
// slot keeps its last two tokens for the n-grams of its next chunk. Prefix snapshots: a slot's window rings, its
// indexer tails and its last two tokens, in pinned host memory.
//
// Not yet: speculative decoding (DFly2), the vision tower, steps split into overlapping parts.
#pragma once

#include <hip/hip_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "engine/plan.hpp"
#include "engine/step.hpp"
#include "kernels/hy4.hpp"
#include "kernels/moe_dispatch.hpp"
#include "model/hy4.hpp"

namespace unholy::engine {

class Hy4ApuExperts final : public model::RemoteExperts {
 public:
  using Layers = std::vector<std::unique_ptr<model::Hy4LayerWeights>>;  // [layer]
  Hy4ApuExperts(const model::Hy4Config& c, int dev, size_t capacity_rows, std::shared_ptr<Layers> layers);
  ~Hy4ApuExperts() override;
  void add_layer(model::Hy4LayerWeights w);
  void run(int, const std::vector<Job>&) override;
  bool can_enqueue() const override { return true; }
  void enqueue(int layer, hipStream_t card_st, int P) override;
  void join(hipStream_t card_st) override;
  hipStream_t stream() const { return st_; }
  static size_t work_bytes(const model::Hy4Config& c, size_t capacity_rows);
  static size_t staging_bytes(const model::Hy4Config& c, size_t capacity_rows);

 private:
  const model::Hy4Config& c_;
  int dev_;
  hipStream_t st_ = nullptr;
  std::shared_ptr<Layers> layers_;
  rt::DeviceBuffer xa_, gu_, eh_, pout_;
  kernels::MoePlan* plan_dev_ = nullptr;
  kernels::Exl3Tile* tiles_dev_ = nullptr;
  uint16_t* in_dev_ = nullptr;
  uint16_t* out_dev_ = nullptr;
  float* w_dev_ = nullptr;
  int32_t* e_dev_ = nullptr;
  int32_t* p2u_dev_ = nullptr;
  int32_t* u2p_dev_ = nullptr;
  uint32_t* req_flag_ = nullptr;
  uint32_t* done_flag_ = nullptr;
  uint32_t seq_ = 0;
};

struct Hy4PlanInput {
  const model::Hy4Config* cfg = nullptr;
  const model::Checkpoint* ck = nullptr;
  UsageProfile usage;
  model::CacheGeometry geo;
  int max_tokens = 0, max_seq = 0, max_logit_rows = 0;
  size_t card_margin = size_t(2) << 30, apu_margin = size_t(4) << 30;
  int snapshots = 0;
};
Plan make_hy4_plan(const Hy4PlanInput& in);
// Pinned host bytes of one prefix snapshot on card k (its window rings and indexer tails of one slot), and the
// tokens block.
size_t hy4_snapshot_bytes(const model::Hy4Config& c, const Plan& p, int k);

class Hy4ResidentModel final : public StepModel {
 public:
  Hy4ResidentModel(const model::Hy4Config& c, const model::Checkpoint& ck, const std::string& engram_dir,
                   const Plan& plan, const model::CacheGeometry& geo, int max_tokens, int max_seq,
                   int max_logit_rows = 0, int snapshots = 0);
  ~Hy4ResidentModel() override;

  int vocab() const override { return c_.vocab; }
  int max_tokens() const override { return max_tokens_; }
  int max_logit_rows() const override { return max_logit_rows_; }
  int max_slots() const override { return geo_.max_slots; }
  int max_pages_per_seq() const override { return geo_.max_pages_per_seq; }
  int num_pages() const override { return geo_.num_pages; }
  int page_tokens() const override { return kernels::kHy4PageRows; }
  int max_inflight() const override { return 2; }
  int num_snapshots() const override { return snapshots_; }
  void snapshot_host(int id, std::vector<std::pair<uint8_t*, size_t>>& segs) const override;
  bool whole_image_blocks() const override { return false; }
  void set_block_tables(const std::vector<int32_t>& tables) override;
  uint64_t submit(const StepInput& in) override;
  void wait(uint64_t ticket, std::vector<float>& logits) override;

  const Plan& plan() const { return plan_; }
  std::vector<std::vector<uint64_t>> routing_histogram() const;
  void reset_routing_histogram();
  std::string routing_share() const;

 private:
  struct Card {
    int dev = 0;
    hipStream_t st = nullptr;
    std::vector<model::Hy4LayerWeights> w;
    std::vector<model::Hy4Cache> cache;
    model::Hy4Scratch scratch;
    rt::DeviceBuffer x, ids, pos, tables;
  };
  const model::Hy4Config& c_;
  Plan plan_;
  model::CacheGeometry geo_;
  int max_tokens_ = 0, max_logit_rows_ = 0;
  Card card_[2];
  std::unique_ptr<Hy4ApuExperts> chan_[2];
  model::Hy4HeadWeights embed_, head_;
  rt::DeviceBuffer hrows_, logits_, rows_idx_;
  std::unique_ptr<model::Hy4Engram> engram_;
  std::vector<int32_t> last2_;  // [slot][2]: the slot's last two tokens (-1: before the start)

  struct Flight {
    uint16_t* x_host = nullptr;
    float* logits_host = nullptr;
    int32_t* io_host = nullptr;
    int32_t* tables_host = nullptr;
    uint8_t* eng_q[4] = {};
    uint16_t* eng_s[4] = {};
    uint8_t* eng_mask = nullptr;
    hipEvent_t handoff = nullptr, consumed = nullptr, done = nullptr, saved = nullptr;
    uint64_t ticket = 0;
    int rows = 0;
    bool busy = false, saved_wait = false;
  };
  Flight flight_[2];
  struct SnapSeg {
    uint8_t* dev;
    size_t bytes, off;
  };
  int snapshots_ = 0;
  std::vector<SnapSeg> snap_segs_[2];
  size_t snap_bytes_[2] = {0, 0};
  uint8_t* snap_host_[2] = {nullptr, nullptr};
  uint8_t* snap_tok_host_ = nullptr;  // [snapshots][4096]: the slot's last two tokens
  void state_copies(int k, const std::vector<StateOp>& ops, bool save);
  uint64_t next_ticket_ = 1;
  std::vector<int32_t> tables_;
  bool tables_dirty_ = false;
};

}  // namespace unholy::engine
