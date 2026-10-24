// Continuous-batching engine: requests in, token streams out.
//
// One thread owns the model. Every iteration it builds a step from the running requests — every
// request that is decoding contributes its next token, then prefills are packed in arrival order into
// the remaining token budget as chunks — runs it, samples the requests whose chunk reached their end,
// and delivers the new tokens. Requests take a slot (per-sequence model state) and pages
// (page_tokens positions of paged cache each) from a shared pool as their positions advance; when the
// pool runs dry the most recently admitted request is preempted (its cache dropped, recomputed later
// from prompt + generated tokens, as vLLM's recompute preemption).
//
// Prefix caching (when the model keeps snapshots): every full page is named by a hash chained over the
// tokens up to its end, and a finished request's pages stay in the pool, unreferenced, until they are
// needed (least recently used first, the tail of a sequence before its head). DSV4 also keeps unpaged
// per-slot state (the attention window, compressor rings), so a prefix is reusable only up to a page
// boundary where that state was saved: prefill chunks that end on a page boundary save it (every
// chunk up to 64k tokens, then sparser), the last full page of the prompt always does, and decoding
// saves at every page it completes (each superseding the last). A new request takes the deepest
// boundary its tokens match, restores the snapshot into its slot and computes the rest.
//
// Token-level stops (EOS, stop token ids, max_tokens, the context limit) are decided here; text-level
// stop strings belong to the frontend, which aborts the request when it sees one.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/sampler.hpp"
#include "engine/step.hpp"

namespace unholy::engine {

enum class FinishReason { None, Stop, Length, Abort, Error };
const char* finish_reason_name(FinishReason r);

// An image of the prompt: its token block [start, start + len) (ids vocab + type), the span whose tokens
// see each other ([span_start, span_end]: IMAGE_START .. IMAGE_END), a hash of the image (prefix caching
// keys on it) and the block's embedding rows [len][dim] bf16 from the vision tower.
struct GenImage {
  int start = 0, len = 0, span_start = 0, span_end = 0;
  uint64_t hash = 0;
  std::shared_ptr<const std::vector<uint16_t>> rows;
};

struct GenRequest {
  std::vector<int32_t> prompt;
  std::vector<GenImage> images;  // in position order
  SamplingParams sampling;
  int max_tokens = 0;  // completion tokens; <= 0: until the context limit
  std::vector<int32_t> stop_token_ids;
  bool ignore_eos = false;
  std::string tag;  // names the request in the request log (the API's completion id)
};

// Delivered incrementally: the tokens generated since the previous delivery.
struct GenOutput {
  std::vector<int32_t> tokens;
  std::vector<Sampled> samples;  // parallel to tokens (logprobs when asked for)
  bool finished = false;
  FinishReason reason = FinishReason::None;
  int32_t stop_token = -1;  // the token that stopped it (not included in `tokens` when it is EOS)
  std::string error;
  int prompt_tokens = 0, completion_tokens = 0;
  int cached_tokens = 0;  // prompt tokens served from the prefix cache
};

// One finished request, for the request log: token counts, where its time went (queue: submitted ->
// admitted; ttft: submitted -> first token; prefill: admitted -> first token; decode: first token ->
// end; negative when it never got there) and how its drafts fared.
struct RequestLog {
  uint64_t id = 0;
  std::string tag;
  FinishReason reason = FinishReason::None;
  std::string error;
  int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0, images = 0, preemptions = 0;
  double queue_s = -1, ttft_s = -1, prefill_s = -1, decode_s = -1;
  int spec_steps = 0, spec_drafted = 0, spec_accepted = 0;
};

// Order in which prefill chunks are packed into a step's token budget (decodes always go first).
//   Shortest: least remaining prompt first, so a continuation with a few hundred uncached tokens is not
//             queued behind a 160k-token prefill; the oldest prefill keeps a quarter of each step.
//   Fcfs:     arrival order (vLLM's default: a long prefill takes the whole budget until it is done).
enum class PrefillOrder { Shortest, Fcfs };

struct EngineConfig {
  int max_batch_tokens = 2048;  // rows per step (<= the model's max_tokens)
  int max_running = 0;          // concurrent sequences (<= the model's slots); 0 = all slots
  int eos_id = 1;
  std::vector<int32_t> eos_ids;  // more end-of-sequence tokens (GLM: <|user|>, <|observation|>)
  int max_context = 0;          // positions per sequence; 0 = the block table's reach
  uint64_t seed = 0;            // base for requests without a seed
  bool prefix_cache = true;      // when the model keeps snapshots
  int spec_drafts = 0;           // speculative decoding: drafts verified per step (0: off; model's drafter)
  float spec_min_conf = -1e30f;  // verify only the leading drafts whose drafter confidence is at least this
  std::function<void(const RequestLog&)> on_finish;  // every request that ends; on the engine thread
  PrefillOrder prefill_order = PrefillOrder::Shortest;
  // vLLM's --long-prefill-token-threshold: with more than one request running or waiting, no request
  // takes more than this many tokens of a step (0: off). Adaptive: never below budget / requests.
  int long_prefill_threshold = 0;
  bool long_prefill_adaptive = false;
  // Two decode micro-batches: a step submitted with nothing in flight takes half of the decoding
  // sequences, the other half go into the next step, so with two steps in flight (the resident model) the
  // cards work on different halves at once instead of taking turns on the whole batch. Off by default: it
  // changes which rows share a step, and a row's values depend on its step's size (GEMVs up to 8 rows, WMMA
  // GEMMs above), so the tokens differ from the original engine's for the same requests.
  bool decode_split = false;
};

struct EngineStats {
  uint64_t steps = 0, prefill_tokens = 0, decode_tokens = 0, preemptions = 0;
  uint64_t prompt_tokens = 0, cached_tokens = 0;  // admitted prompt tokens, those served from the cache
  int running = 0, waiting = 0, free_pages = 0, total_pages = 0;  // free: unused or cached-unreferenced
  int cached_pages = 0, snapshots = 0;
  uint64_t spec_steps = 0, spec_drafted = 0, spec_accepted = 0;  // verify chunks, drafts in them, accepted
  std::array<uint64_t, 8> spec_pos_drafted{}, spec_pos_accepted{};  // by draft position
  double busy_seconds = 0;
};

class Engine {
 public:
  using Sink = std::function<void(GenOutput&&)>;  // called on the engine thread; must not block

  Engine(StepModel& model, EngineConfig cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Thread-safe. Requests that can never fit (prompt longer than the context) finish with Error.
  uint64_t submit(GenRequest req, Sink sink);
  // `why` is the reason the request log gives (Stop when the frontend matched a stop string).
  void abort(uint64_t id, FinishReason why = FinishReason::Abort);
  EngineStats stats() const;
  // Diagnostics (tests): one line per running request (positions, flags, drafts, images).
  std::string debug_state() const;
  int max_context() const { return max_context_; }

  // Runs one scheduling iteration on the calling thread (tests drive the engine this way when it is
  // constructed with start = false). Returns false when there was nothing to do.
  bool iterate();
  Engine(StepModel& model, EngineConfig cfg, bool start);

 private:
  struct Seq;
  void loop();
  void intake();
  bool alloc_pages(Seq& s, int upto);  // pages for positions [0, upto)
  int available_pages() const { return int(free_pages_.size() + evictable_.size()); }
  int32_t take_page();
  void write_table(const Seq& s);
  uint64_t token_key(const Seq& s, int p) const;
  void hash_pages(Seq& s, int npages);
  void match_prefix(Seq& s);
  void register_pages(Seq& s);
  int take_snapshot();
  void put_snapshot(int id);
  void commit_snapshot(Seq& s, int id, int end);
  void clear_cache();
  bool save_at(const Seq& s, int end, bool decode) const;
  void release(Seq& s);
  void preempt(Seq& s);
  void finish(Seq& s, FinishReason r, const std::string& err = "");
  bool submit_step(int budget);
  void complete_step();
  void fail_all(const std::string& why);

  struct InFlight {
    struct Item {
      Seq* s;
      int n;
      bool logits;
      int row;  // logit row, -1 if none
      int restore = -1, save = -1;  // snapshots loaded before / saved after the chunk
      int end = 0;                  // the chunk's end position
      int ndraft = 0;               // a verify chunk: the fed token, then this many drafts
    };
    uint64_t ticket = 0;
    std::vector<Item> items;
  };

  StepModel& model_;
  EngineConfig cfg_;
  int max_context_ = 0, page_tokens_ = 0, slots_ = 0;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<Seq>> incoming_;
  std::vector<std::pair<uint64_t, FinishReason>> aborts_;
  bool stop_ = false;
  uint64_t next_id_ = 1;

  // Engine-thread state.
  std::deque<std::unique_ptr<Seq>> waiting_;
  std::vector<std::unique_ptr<Seq>> running_;  // admission order
  std::deque<InFlight> inflight_;              // submitted steps, oldest first
  std::vector<int32_t> free_pages_, free_slots_;
  // Prefix cache. A page is free (in free_pages_), referenced by running sequences, or cached and
  // unreferenced (in evictable_, oldest first). Snapshots likewise (free_snaps_, snap_lru_).
  struct PageInfo {
    uint64_t hash = 0;
    int refs = 0;
    bool cached = false;
    std::list<int32_t>::iterator lru;
  };
  struct SnapInfo {
    uint64_t hash = 0;
    int refs = 0;
    bool cached = false;
    std::list<int>::iterator lru;
  };
  bool cache_on_ = false;
  bool spec_on_ = false;  // speculative decoding with the model's drafter
  int spec_max_ = 0;      // drafts verified per step at most
  std::vector<PageInfo> page_info_;
  std::list<int32_t> evictable_;
  std::unordered_map<uint64_t, int32_t> page_by_hash_;
  std::vector<SnapInfo> snap_info_;
  std::vector<int> free_snaps_;
  std::list<int> snap_lru_;
  std::unordered_map<uint64_t, int> snap_by_hash_;
  std::vector<int32_t> tables_;
  bool tables_dirty_ = true;
  std::vector<float> logits_;
  uint64_t admit_counter_ = 0;

  mutable std::mutex stats_mu_;
  EngineStats stats_;
  std::thread thread_;
};

}  // namespace unholy::engine
