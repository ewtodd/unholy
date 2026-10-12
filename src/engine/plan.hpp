// Placement planner: which device holds what.
//
// Layers are split across the two cards (pipeline); every layer's dense weights, caches and the
// layer's hot routed experts live on its card; the remaining (cold) experts live on the APU, which
// computes them. Hot experts are chosen greedily by measured routing frequency, which maximises the
// routing mass served from VRAM for the bytes available. All byte counts come from the checkpoint
// and the actual allocation sizes, not estimates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/dsv4.hpp"
#include "model/forward_ref.hpp"
#include "model/safetensors.hpp"

namespace unholy::engine {

struct DeviceInfo {
  int id = 0;
  std::string arch;
  bool integrated = false;
  size_t free_bytes = 0, total_bytes = 0;
};
std::vector<DeviceInfo> query_devices();

// Routing frequency per [layer][expert] (any scale; normalised per layer by the planner).
using UsageProfile = std::vector<std::vector<double>>;
// "L<n> c0 c1 ... c255" lines (dsv4-engine's expert_profile.txt format, and ours).
UsageProfile load_usage_profile(const std::string& path, int n_layers, int n_experts);
// Counts [layer][expert] in that format.
std::string format_usage_profile(const std::vector<std::vector<uint64_t>>& counts);

struct PlanInput {
  const model::Dsv4Config* cfg = nullptr;
  const model::Checkpoint* dense = nullptr;
  const model::Checkpoint* exl3 = nullptr;  // routed experts from here when set
  UsageProfile usage;
  model::CacheGeometry geo;
  int max_tokens = 0, max_seq = 0;
  int max_logit_rows = 0;  // rows of logits a step returns (0: max_tokens)
  size_t card_margin = size_t(2) << 30;  // runtime, fragmentation, HIP's own allocations (unholy serve --card-margin-gb)
  size_t apu_margin = size_t(4) << 30;
  bool dspark = false;  // the DSpark drafter: dense parts and windows on card 1, routed experts on the APU
  bool vision = false;  // the vision tower on the APU
};

struct Plan {
  int card[2] = {-1, -1};
  int apu = -1;
  std::vector<int> layer_card;               // [layer] -> 0 / 1
  std::vector<std::vector<uint8_t>> on_card;  // [layer][expert] 1 = on the layer's card
  size_t dense[2] = {0, 0}, cache[2] = {0, 0}, scratch[2] = {0, 0}, experts[2] = {0, 0}, budget[2] = {0, 0};
  size_t apu_experts = 0, apu_free = 0;
  size_t dspark_card = 0, dspark_apu = 0;
  size_t vision_apu = 0;  // vision tower weights + scratch  // the drafter's bytes on card 1 / the APU
  double routed_on_cards = 0;  // predicted share of routing served by card-resident experts
  std::string summary() const;
};

Plan make_plan(const PlanInput& in);

// Bytes of one routed expert (w1 + w2 + w3) of `layer` in the format that will be loaded.
size_t expert_bytes(const model::Checkpoint& ck, int layer, int expert, bool exl3);
// Bytes of a layer's dense part (everything but routed experts) as uploaded.
size_t dense_layer_bytes(const model::Checkpoint& ck, int layer);
// The DSpark stages' bytes (mtp.*): everything but the routed experts / the routed experts.
size_t dspark_dense_bytes(const model::Checkpoint& ck);
size_t dspark_expert_bytes(const model::Checkpoint& ck);
// Card-1 bytes the drafter needs besides its weights: stage windows, main-hidden capture, draft scratch
// and logits, ring backups (both cards' share counted here).
size_t dspark_work_bytes(const model::Dsv4Config& c, const model::CacheGeometry& g);

}  // namespace unholy::engine
