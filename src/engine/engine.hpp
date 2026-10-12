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
// Token-level stops (EOS, stop token ids, max_tokens, the context limit) are decided here; text-level
// stop strings belong to the frontend, which aborts the request when it sees one.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "engine/sampler.hpp"
#include "engine/step.hpp"

namespace unholy::engine {

enum class FinishReason { None, Stop, Length, Abort, Error };
const char* finish_reason_name(FinishReason r);

struct GenRequest {
  std::vector<int32_t> prompt;
  SamplingParams sampling;
  int max_tokens = 0;  // completion tokens; <= 0: until the context limit
  std::vector<int32_t> stop_token_ids;
  bool ignore_eos = false;
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
};

struct EngineConfig {
  int max_batch_tokens = 2048;  // rows per step (<= the model's max_tokens)
  int max_running = 0;          // concurrent sequences (<= the model's slots); 0 = all slots
  int eos_id = 1;
  int max_context = 0;          // positions per sequence; 0 = the block table's reach
  uint64_t seed = 0;            // base for requests without a seed
};

struct EngineStats {
  uint64_t steps = 0, prefill_tokens = 0, decode_tokens = 0, preemptions = 0;
  int running = 0, waiting = 0, free_pages = 0, total_pages = 0;
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
  void abort(uint64_t id);
  EngineStats stats() const;
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
  std::vector<uint64_t> aborts_;
  bool stop_ = false;
  uint64_t next_id_ = 1;

  // Engine-thread state.
  std::deque<std::unique_ptr<Seq>> waiting_;
  std::vector<std::unique_ptr<Seq>> running_;  // admission order
  std::deque<InFlight> inflight_;              // submitted steps, oldest first
  std::vector<int32_t> free_pages_, free_slots_;
  std::vector<int32_t> tables_;
  bool tables_dirty_ = true;
  std::vector<float> logits_;
  uint64_t admit_counter_ = 0;

  mutable std::mutex stats_mu_;
  EngineStats stats_;
  std::thread thread_;
};

}  // namespace unholy::engine
