// The interface between the scheduler and a model: batched steps over chunks of sequences.
//
// A chunk feeds positions [s0, s0 + n) of the sequence in `slot` from batch rows [t0, t0 + n). Each
// slot's chunks arrive in position order; a chunk with s0 == 0 starts a new sequence in that slot. Per
// slot state that is not paged (window ring, compressor rings) belongs to the slot; paged state (the
// compressed rows) lives in physical pages named by the slot's row of the block table.
//
// Steps are asynchronous: submit() enqueues one and returns a ticket, wait() blocks until its logits
// are ready. Up to max_inflight() steps may be outstanding (the resident model overlaps one step's
// second card with the next step's first); they execute in submission order, so a step sees every
// cache write of the steps submitted before it, and tickets are waited for in that order.
#pragma once

#include <cstdint>
#include <vector>

namespace unholy::engine {

struct Chunk {
  int slot = 0, s0 = 0, n = 0, t0 = 0;
};

// Prefix reuse: a snapshot holds a slot's unpaged state (window ring, compressor rings) as it was at a
// page boundary. `restore` loads one into a slot before the step's chunks run (the slot then continues
// its sequence at the snapshot's position, over pages it shares with the sequence that saved it);
// `save` copies a slot's state into one after them.
struct StateOp {
  int slot = 0, snapshot = 0;
};

// Speculative decoding. A verify chunk feeds a sequence's next token followed by drafted tokens, with
// logits for every row. Before it runs, `backup` copies the ring entries (attention windows, compressor
// and drafter rings) that its positions overwrite; once the drafts are judged, a RingRestore puts back
// the entries of the rejected positions [from, to) (s0: the backed-up chunk's first position).
struct RingBackup {
  int slot = 0, s0 = 0, n = 0;
};
struct RingRestore {
  int slot = 0, s0 = 0, from = 0, to = 0;
};
// A draft block: the sequence's token at position start + 1 (the last one sampled), drafting the
// following draft_block() tokens from the target's state at positions <= start.
struct DraftReq {
  int slot = 0, start = 0;
  int32_t token = 0;
};

// Images: an image block's embedding rows replace its tokens' embeddings (rows [n][dim] bf16 in host memory
// that outlives the step, for batch rows [t0, t0 + n)); within a span [start, end] (IMAGE_START ..
// IMAGE_END positions of the slot's sequence) tokens see each other both ways. A step holds whole blocks.
struct ImageRows {
  int t0 = 0, n = 0;
  const uint16_t* rows = nullptr;
};
struct ImageSpan {
  int slot = 0, start = 0, end = 0;
};

struct StepInput {
  std::vector<Chunk> chunks;
  std::vector<int32_t> ids, pos;     // [T]
  std::vector<int32_t> logit_rows;   // batch rows whose logits are wanted
  std::vector<StateOp> restore, save;
  std::vector<RingBackup> backup;
  std::vector<ImageRows> image_rows;
  std::vector<ImageSpan> image_spans;
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
  virtual int max_inflight() const { return 1; }
  virtual int num_snapshots() const { return 0; }  // 0: no prefix reuse
  virtual int draft_block() const { return 0; }    // tokens one draft proposes; 0: no drafter
  virtual int max_verify() const { return 0; }     // longest verify chunk (1 + drafts) backups cover
  // Applies `restore` (both cards' rings), then drafts for each request: tokens [reqs][draft_block()]
  // and their confidences. Runs after every submitted step that wrote these slots has completed.
  virtual void draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs,
                     std::vector<int32_t>& tokens, std::vector<float>& conf) {
    (void)restore, (void)reqs, (void)tokens, (void)conf;
  }
  // Block tables [max_slots][max_pages_per_seq] of physical page ids, for steps submitted from now on.
  virtual void set_block_tables(const std::vector<int32_t>& tables) = 0;
  virtual uint64_t submit(const StepInput& in) = 0;
  // Logits [logit_rows.size()][vocab] fp32 of the step `ticket` (the oldest outstanding one).
  virtual void wait(uint64_t ticket, std::vector<float>& logits) = 0;
  void step(const StepInput& in, std::vector<float>& logits) { wait(submit(in), logits); }
};

}  // namespace unholy::engine
