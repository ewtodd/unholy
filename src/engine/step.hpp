// The interface between the scheduler and a model: one batched step over chunks of sequences.
//
// A chunk feeds positions [s0, s0 + n) of the sequence in `slot` from batch rows [t0, t0 + n). Each
// slot's chunks arrive in position order; a chunk with s0 == 0 starts a new sequence in that slot. Per
// slot state that is not paged (window ring, compressor rings) belongs to the slot; paged state (the
// compressed rows) lives in physical pages named by the slot's row of the block table.
#pragma once

#include <cstdint>
#include <vector>

namespace unholy::engine {

struct Chunk {
  int slot = 0, s0 = 0, n = 0, t0 = 0;
};

struct StepInput {
  std::vector<Chunk> chunks;
  std::vector<int32_t> ids, pos;     // [T]
  std::vector<int32_t> logit_rows;   // batch rows whose logits are wanted
};

class StepModel {
 public:
  virtual ~StepModel() = default;
  virtual int vocab() const = 0;
  virtual int max_tokens() const = 0;         // rows per step
  virtual int max_logit_rows() const = 0;
  virtual int max_slots() const = 0;
  virtual int max_pages_per_seq() const = 0;  // block-table width
  virtual int num_pages() const = 0;          // physical pages
  virtual int page_tokens() const = 0;
  // Block tables [max_slots][max_pages_per_seq] of physical page ids.
  virtual void set_block_tables(const std::vector<int32_t>& tables) = 0;
  // Runs the step; logits [logit_rows.size()][vocab] fp32.
  virtual void step(const StepInput& in, std::vector<float>& logits) = 0;
};

}  // namespace unholy::engine
