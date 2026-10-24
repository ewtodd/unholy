// GLM-5.3-Flash resident across the two cards and the APU (docs/PLAN-glm-dsv41.md, G5): the GLM counterpart of
// ResidentModel, on the fast path of model/glm5.hpp.
//
// card 0: the embedding and its layers; card 1: its layers and the head; each layer's dense weights, paged DSA rows
// or per-slot KDA state, and hot routed experts on its card; the cold experts on the APU (Glm5ApuExperts: one
// channel per card, device-ordered flags, no host waits). Two steps in flight: card 0 runs step k + 1 while card 1
// finishes step k; the residual crosses once per step through pinned memory.
//
// MTP drafting (mtp = true): the nextn layer's dense part and DSA rows on card 1, its experts on the APU (a third
// channel); every step writes its cache rows from the trunk's final-normed hidden states, draft() runs it for one draft per
// request on card 1's draft stream; a verify chunk backs up the KDA states after its first token, restored when the
// draft is rejected. Prefix snapshots (snapshots > 0): a slot's KDA conv tails and states (148 MB) and its
// MTP hidden row, in pinned host memory (TTM pages, under the limit the APU's memory comes from). Not yet: images.
#pragma once

#include <hip/hip_runtime.h>

#include <memory>
#include <span>
#include <vector>

#include "engine/plan.hpp"
#include "engine/step.hpp"
#include "kernels/glm_fast.hpp"
#include "kernels/moe_dispatch.hpp"
#include "model/glm5.hpp"

namespace unholy::engine {

// The APU's cold experts as seen by one card (own stream, flags, pinned staging, scratch; shared weights).
class Glm5ApuExperts final : public model::RemoteExperts {
 public:
  using Layers = std::vector<std::unique_ptr<model::Glm5LayerWeights>>;  // [layer]
  Glm5ApuExperts(const model::Glm5Config& c, int dev, size_t capacity_rows, std::shared_ptr<Layers> layers);
  ~Glm5ApuExperts() override;
  void add_layer(model::Glm5LayerWeights w);
  void run(int, const std::vector<Job>&) override;  // host-synchronous jobs: not supported (enqueue only)
  bool can_enqueue() const override { return true; }
  void enqueue(int layer, hipStream_t card_st, int P) override;
  void join(hipStream_t card_st) override;
  hipStream_t stream() const { return st_; }

 private:
  const model::Glm5Config& c_;
  int dev_;
  hipStream_t st_ = nullptr;
  std::shared_ptr<Layers> layers_;
  rt::DeviceBuffer xh1_, xh3_, xh2_, h1_, h3_, yp_;
  kernels::MoePlan* plan_dev_ = nullptr;
  kernels::Exl3Tile* tiles_dev_ = nullptr;
  uint16_t* in_dev_ = nullptr;
  uint16_t* out_dev_ = nullptr;
  float* w_dev_ = nullptr;
  int32_t* e_dev_ = nullptr;
  int32_t* p2u_dev_ = nullptr;  // pair row -> compact token row of in_dev_
  uint32_t* req_flag_ = nullptr;  // pinned, coherent: card -> APU
  uint32_t* done_flag_ = nullptr;  // APU -> card
  uint32_t seq_ = 0;
};

struct Glm5PlanInput {
  const model::Glm5Config* cfg = nullptr;
  const model::Checkpoint* ck = nullptr;  // the EXL3 checkpoint served
  UsageProfile usage;
  model::CacheGeometry geo;
  int max_tokens = 0, max_seq = 0, max_logit_rows = 0;
  size_t card_margin = size_t(2) << 30, apu_margin = size_t(4) << 30;
  bool mtp = false;  // MTP drafting: its layer on card 1 / the APU, the KDA state backups
  int snapshots = 0;  // prefix snapshots in pinned host memory (taken from the APU's RAM)
};
Plan make_glm5_plan(const Glm5PlanInput& in);
// Pinned host bytes of one prefix snapshot (every KDA layer's slot state; with MTP, the slot's last hidden state).
size_t glm5_snapshot_bytes(const model::Glm5Config& c, bool mtp);

class Glm5ResidentModel final : public StepModel {
 public:
  Glm5ResidentModel(const model::Glm5Config& c, const model::Checkpoint& ck, const Plan& plan,
                    const model::CacheGeometry& geo, int max_tokens, int max_seq, int max_logit_rows = 0,
                    bool mtp = false, int snapshots = 0);
  ~Glm5ResidentModel() override;

  int vocab() const override { return c_.vocab; }
  int max_tokens() const override { return max_tokens_; }
  int max_logit_rows() const override { return max_logit_rows_; }
  int max_slots() const override { return geo_.max_slots; }
  int max_pages_per_seq() const override { return geo_.max_pages_per_seq; }
  int num_pages() const override { return geo_.num_pages; }
  int page_tokens() const override { return kernels::kGlmPageRows; }
  int max_inflight() const override { return 2; }
  int num_snapshots() const override { return snapshots_; }
  int draft_block() const override { return mtp_ ? 1 : 0; }
  int max_verify() const override { return mtp_ ? 2 : 0; }
  void draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs, std::vector<int32_t>& tokens,
             std::vector<float>& conf) override;
  void set_block_tables(const std::vector<int32_t>& tables) override;
  uint64_t submit(const StepInput& in) override;
  void wait(uint64_t ticket, std::vector<float>& logits) override;

  const Plan& plan() const { return plan_; }
  // Pairs routed to each expert since load (or the last reset), [layer][expert] (usage-profile format via
  // format_usage_profile).
  std::vector<std::vector<uint64_t>> routing_histogram() const;
  void reset_routing_histogram();

 private:
  struct Card {
    int dev = 0;
    hipStream_t st = nullptr;
    std::vector<model::Glm5LayerWeights> w;
    std::vector<model::Glm5PagedCache> cache;
    model::Glm5Scratch scratch;
    rt::DeviceBuffer x, ids, pos, tables;
  };
  const model::Glm5Config& c_;
  Plan plan_;
  model::CacheGeometry geo_;
  int max_tokens_ = 0, max_logit_rows_ = 0;
  Card card_[2];
  std::unique_ptr<Glm5ApuExperts> chan_[2];
  std::unique_ptr<rt::Arena> card_arena_[2], apu_arena_;
  model::Glm5HeadWeights embed_, head_;   // embedding on card 0, final norm + LM head on card 1
  rt::DeviceBuffer hrows_, logits_, rows_idx_;  // head inputs / outputs on card 1
  // MTP (card 1): weights, cache, the draft stream / scratch / APU channel, the hidden states it reads, staging.
  bool mtp_ = false;
  model::Glm5MtpWeights mtpw_;
  model::Glm5PagedCache mcache_;
  hipStream_t dst_ = nullptr;
  hipEvent_t drafted_ = nullptr;           // dst_: the last draft() (restores included) done
  model::Glm5Scratch dscr_;
  std::unique_ptr<Glm5ApuExperts> dchan_;
  rt::DeviceBuffer h_all_, h_last_, h_prev_;  // [max_tokens][dim] this step's final-normed rows; [slot][dim]
  rt::DeviceBuffer m_emb_, m_h_, m_x0_;       // step: MTP cache rows' embeddings, hidden states, inputs
  rt::DeviceBuffer d_emb_, d_h_, d_x_, d_hn_, d_logits_, d_tok_;  // draft rows
  uint16_t* d_emb_host_ = nullptr;            // [max_slots][dim]
  int32_t* d_tok_host_ = nullptr;             // [max_slots] tokens, then [max_slots] confidences (float)
  std::span<const uint16_t> embed_host_;      // the checkpoint's embedding table (mmap)

  struct Flight {
    uint16_t* x_host = nullptr;     // residual handoff card 0 -> card 1
    float* logits_host = nullptr;   // [max_logit_rows][vocab]
    int32_t* io_host = nullptr;     // [max_tokens] ids, positions, logit rows
    int32_t* tables_host = nullptr;
    uint16_t* emb_host = nullptr;    // MTP: [max_tokens][dim] embeddings of the step's cache rows
    hipEvent_t handoff = nullptr, consumed = nullptr, done = nullptr;
    uint64_t ticket = 0;
    int rows = 0;
    bool busy = false;
  };
  Flight flight_[2];
  // Prefix snapshots: per card, the slot slices of the per-slot state (KDA conv / recurrent, card 1: the MTP hidden
  // row), packed per snapshot in pinned host memory.
  struct SnapSeg {
    uint8_t* dev;       // the buffer; a slot's slice at dev + slot * bytes
    size_t bytes, off;  // per slot; offset in the snapshot
  };
  int snapshots_ = 0;
  std::vector<SnapSeg> snap_segs_[2];
  size_t snap_bytes_[2] = {0, 0};
  uint8_t* snap_host_[2] = {nullptr, nullptr};
  void state_copies(int k, const std::vector<StateOp>& ops, bool save);
  uint64_t next_ticket_ = 1;
  std::vector<int32_t> tables_;
  bool tables_dirty_ = false;
};

}  // namespace unholy::engine
