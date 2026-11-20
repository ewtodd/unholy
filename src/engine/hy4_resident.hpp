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
// Speculative decoding with the checkpoint's DFly2 drafter (model/hy4.hpp; card 1, its own stream): draft() rolls back
// what a verify chunk's rejected drafts left (window ring rows and indexer tails from per-slot backups, the Engram
// history on the host), then drafts 5 tokens per request from each slot's saved final hidden state and the indexer
// picks of target layers 25 + 4 i, with the drafter's probabilities as confidences.
//
// Images (vision): the vision tower (model/hy4_vision.hpp) on the APU encodes a request's images when the server
// parses it; a step's image rows replace their pad tokens' embeddings on card 0; chunks of sequences with images carry
// their M-RoPE layout (Chunk::mrope), from which the three rotary planes of the rows, of the indexer groups' first
// slots and of the drafter's block are computed per step.
//
// Not yet: steps split into overlapping parts.
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
#include "model/hy4_vision.hpp"
#include "vision/image.hpp"

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
  bool mxfp4 = false;  // routed experts converted to MXFP4 at load
  bool nvfp4 = false;  // routed experts in NVFP4 from a GGUF build
  bool dfly = false;   // the DFly2 drafter (card 1: its weights, the embedding table, its scratch)
  bool vision = false;  // the vision tower (the APU: its weights and a scratch for the largest image)
};
Plan make_hy4_plan(const Hy4PlanInput& in);
// Pinned host bytes of one prefix snapshot on card k (its window rings and indexer tails of one slot), and the
// tokens block.
size_t hy4_snapshot_bytes(const model::Hy4Config& c, const Plan& p, int k);

class Hy4ResidentModel final : public StepModel {
 public:
  Hy4ResidentModel(const model::Hy4Config& c, const model::Checkpoint& ck, const std::string& engram_dir,
                   const Plan& plan, const model::CacheGeometry& geo, int max_tokens, int max_seq,
                   int max_logit_rows = 0, int snapshots = 0, bool mxfp4 = false,
                   const model::Checkpoint* experts_gguf = nullptr, bool dfly = false, bool vision = false);
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
  int draft_block() const override { return dfly_ ? model::kHy4DraftTokens : 0; }
  int max_verify() const override { return dfly_ ? model::kHy4DraftRows : 0; }
  void draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs, std::vector<int32_t>& tokens,
             std::vector<float>& conf) override;
  void set_block_tables(const std::vector<int32_t>& tables) override;
  uint64_t submit(const StepInput& in) override;
  void wait(uint64_t ticket, std::vector<float>& logits) override;

  bool has_vision() const { return bool(vision_); }
  // One image's rows [(gh / 2) * (gw / 2 + 1)][dim] bf16 (thread-safe; the APU's vision stream).
  void encode_image(const vision::Hy4Patches& p, std::vector<uint16_t>& rows);

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
    rt::DeviceBuffer x, ids, pos, pos3, tables;
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
  // DFly2 (card 1): weights, scratch, its stream; the slots' final hidden states of their last 6 positions; the head's
  // normed rows of a step; the drafted tokens; the target caches it reads; each slot's last verify chunk (s0, its
  // tokens, the two tokens before it) for rolling the Engram history back.
  bool dfly_ = false;
  model::Hy4DflyWeights dw_;
  model::Hy4DflyScratch dscr_;
  hipStream_t dst_ = nullptr;
  hipEvent_t drafted_ = nullptr;
  rt::DeviceBuffer hidden_, hn_rows_, dtok_;
  int32_t* dtok_host_ = nullptr;
  std::vector<model::Hy4Cache*> dtarget_;
  std::vector<int> vs0_;
  std::vector<std::vector<int32_t>> vids_;
  std::vector<int32_t> vprev_;

  struct Flight {
    uint16_t* x_host = nullptr;
    float* logits_host = nullptr;
    int32_t* io_host = nullptr;
    int32_t* pos3_host = nullptr;  // [max_tokens][3] M-RoPE planes
    uint16_t* img_host = nullptr;  // [max_tokens][dim] the step's image rows
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
  struct Vision;
  std::unique_ptr<Vision> vision_;
  uint64_t next_ticket_ = 1;
  std::vector<int32_t> tables_;
  bool tables_dirty_ = false;
};

}  // namespace unholy::engine
