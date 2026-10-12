#include "engine/engine.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace unholy::engine {

const char* finish_reason_name(FinishReason r) {
  switch (r) {
    case FinishReason::None: return "";
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Abort: return "abort";
    case FinishReason::Error: return "error";
  }
  return "";
}

struct Engine::Seq {
  uint64_t id = 0;
  GenRequest req;
  Sink sink;
  Sampler sampler;
  std::vector<int32_t> tokens;  // prompt + generated
  int n_prompt = 0;
  int computed = 0;             // positions submitted to the model (cache valid once their steps complete)
  int slot = -1;
  std::vector<int32_t> pages;   // logical page -> physical page
  uint64_t admitted = 0;
  GenOutput out;                // pending delivery
  bool done = false;
  int inflight = 0;             // submitted steps that include this sequence and have not completed
  bool awaiting_logits = false; // one of them samples its next token
  bool abort_pending = false;   // aborted while in flight: finished when the last such step completes

  Seq(uint64_t id_, GenRequest r, Sink s, uint64_t seed)
      : id(id_), req(std::move(r)), sink(std::move(s)), sampler(req.sampling, seed) {}
  int generated() const { return int(tokens.size()) - n_prompt; }
};

Engine::Engine(StepModel& model, EngineConfig cfg) : Engine(model, cfg, true) {}

Engine::Engine(StepModel& model, EngineConfig cfg, bool start) : model_(model), cfg_(cfg) {
  page_tokens_ = model_.page_tokens();
  max_context_ = model_.max_pages_per_seq() * page_tokens_;
  if (cfg_.max_context > 0) max_context_ = std::min(max_context_, cfg_.max_context);
  cfg_.max_batch_tokens = std::min(cfg_.max_batch_tokens > 0 ? cfg_.max_batch_tokens : model_.max_tokens(),
                                   model_.max_tokens());
  slots_ = model_.max_slots();
  if (cfg_.max_running > 0) slots_ = std::min(slots_, cfg_.max_running);
  for (int s = slots_ - 1; s >= 0; --s) free_slots_.push_back(s);
  for (int p = model_.num_pages() - 1; p >= 0; --p) free_pages_.push_back(p);
  tables_.assign(size_t(model_.max_slots()) * model_.max_pages_per_seq(), 0);
  stats_.total_pages = stats_.free_pages = model_.num_pages();
  if (start) thread_ = std::thread([this] { loop(); });
}

Engine::~Engine() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  while (!inflight_.empty()) {
    try {
      model_.wait(inflight_.front().ticket, logits_);
    } catch (const std::exception&) {
    }
    inflight_.pop_front();
  }
  for (auto& s : incoming_) finish(*s, FinishReason::Abort);
  for (auto& s : waiting_) finish(*s, FinishReason::Abort);
  for (auto& s : running_) finish(*s, FinishReason::Abort);
}

uint64_t Engine::submit(GenRequest req, Sink sink) {
  std::lock_guard lk(mu_);
  const uint64_t id = next_id_++;
  auto s = std::make_unique<Seq>(id, std::move(req), std::move(sink), cfg_.seed * 0x9e3779b97f4a7c15ull + id);
  s->tokens = s->req.prompt;
  s->n_prompt = int(s->tokens.size());
  s->out.prompt_tokens = s->n_prompt;
  s->sampler.set_prompt(s->tokens);
  incoming_.push_back(std::move(s));
  cv_.notify_all();
  return id;
}

void Engine::abort(uint64_t id) {
  std::lock_guard lk(mu_);
  aborts_.push_back(id);
  cv_.notify_all();
}

EngineStats Engine::stats() const {
  std::lock_guard lk(stats_mu_);
  return stats_;
}

void Engine::finish(Seq& s, FinishReason r, const std::string& err) {
  if (s.done) return;
  s.done = true;
  release(s);
  s.out.finished = true;
  s.out.reason = r;
  s.out.error = err;
  s.out.completion_tokens = s.generated();
  if (s.sink) s.sink(std::move(s.out));
  s.out = GenOutput{};
}

void Engine::release(Seq& s) {
  for (const int32_t p : s.pages) free_pages_.push_back(p);
  s.pages.clear();
  if (s.slot >= 0) free_slots_.push_back(s.slot);
  s.slot = -1;
  s.computed = 0;
}

void Engine::preempt(Seq& s) {
  release(s);
  {
    std::lock_guard lk(stats_mu_);
    ++stats_.preemptions;
  }
}

bool Engine::alloc_pages(Seq& s, int upto) {
  const size_t need = size_t((upto + page_tokens_ - 1) / page_tokens_);
  if (need <= s.pages.size()) return true;
  if (need - s.pages.size() > free_pages_.size()) return false;
  while (s.pages.size() < need) {
    s.pages.push_back(free_pages_.back());
    free_pages_.pop_back();
  }
  const size_t w = size_t(model_.max_pages_per_seq());
  std::copy(s.pages.begin(), s.pages.end(), tables_.begin() + ptrdiff_t(size_t(s.slot) * w));
  tables_dirty_ = true;
  return true;
}

void Engine::intake() {
  std::vector<std::unique_ptr<Seq>> in;
  std::vector<uint64_t> aborts;
  {
    std::lock_guard lk(mu_);
    in.swap(incoming_);
    aborts.swap(aborts_);
  }
  for (auto& s : in) {
    if (s->n_prompt == 0) finish(*s, FinishReason::Error, "empty prompt");
    else if (s->n_prompt >= max_context_)
      finish(*s, FinishReason::Error,
             "prompt of " + std::to_string(s->n_prompt) + " tokens does not fit the context of " +
                 std::to_string(max_context_));
    else waiting_.push_back(std::move(s));
  }
  for (const uint64_t id : aborts) {
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
      if ((*it)->id == id) {
        finish(**it, FinishReason::Abort);
        waiting_.erase(it);
        break;
      }
    for (auto it = running_.begin(); it != running_.end(); ++it)
      if ((*it)->id == id) {
        if ((*it)->inflight > 0) {
          (*it)->abort_pending = true;  // its step still uses the slot's state; finish on completion
        } else {
          finish(**it, FinishReason::Abort);
          running_.erase(it);
        }
        break;
      }
  }
}

bool Engine::iterate() {
  intake();
  const auto t_start = std::chrono::steady_clock::now();
  const int budget0 = cfg_.max_batch_tokens;

  // Admission: a free slot, pages for the request's first chunk, and a page of headroom for every
  // running sequence (so decoders crossing a page boundary do not immediately preempt the newcomer).
  while (!waiting_.empty() && !free_slots_.empty()) {
    Seq& s = *waiting_.front();
    const int first = std::min(int(s.tokens.size()), budget0);
    const size_t need = size_t((first + page_tokens_ - 1) / page_tokens_) + running_.size();
    if (need > free_pages_.size()) break;
    s.slot = free_slots_.back();
    free_slots_.pop_back();
    s.admitted = ++admit_counter_;
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
  if (running_.empty() && inflight_.empty()) {
    if (!waiting_.empty()) {
      // Nothing runs and the head does not fit an empty pool: it never will.
      finish(*waiting_.front(), FinishReason::Error, "request does not fit the page pool");
      waiting_.pop_front();
      return true;
    }
    return false;
  }
  const bool submitted = int(inflight_.size()) < model_.max_inflight() && submit_step(budget0);
  bool completed = false;
  if (!inflight_.empty() && (int(inflight_.size()) >= model_.max_inflight() || !submitted)) {
    complete_step();
    completed = true;
  }
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::lock_guard lk(stats_mu_);
  stats_.busy_seconds += dt;
  stats_.running = int(running_.size());
  stats_.waiting = int(waiting_.size());
  stats_.free_pages = int(free_pages_.size());
  return submitted || completed;
}

// Plans and submits one step over the sequences that are not waiting for logits: decoding sequences
// first (one token each), then prefill chunks in admission order. Returns false if nothing was
// submitted.
bool Engine::submit_step(int budget0) {
  std::vector<InFlight::Item> plan;
  const auto build = [&]() {
    plan.clear();
    int budget = budget0, logit_rows = model_.max_logit_rows();
    for (int pass = 0; pass < 2; ++pass)
      for (auto& sp : running_) {
        Seq& s = *sp;
        if (s.awaiting_logits || s.abort_pending || s.done) continue;
        const int remaining = int(s.tokens.size()) - s.computed;
        if (remaining <= 0 || (remaining == 1) != (pass == 0) || budget == 0) continue;
        const int n = std::min(remaining, budget);
        const bool logits = n == remaining;
        if (logits && logit_rows == 0) continue;
        plan.push_back({&s, n, logits, -1});
        budget -= n;
        logit_rows -= logits;
      }
  };
  for (;;) {
    build();
    size_t need = 0;
    for (const auto& it : plan) {
      const size_t want = size_t((it.s->computed + it.n + page_tokens_ - 1) / page_tokens_);
      need += want > it.s->pages.size() ? want - it.s->pages.size() : 0;
    }
    if (need <= free_pages_.size()) break;
    // Preempt the most recently admitted request with nothing in flight and plan again; if every
    // candidate is in flight, let those steps complete first.
    auto victim = running_.end();
    for (auto it = running_.begin(); it != running_.end(); ++it)
      if ((*it)->inflight == 0 && (victim == running_.end() || (*it)->admitted > (*victim)->admitted)) victim = it;
    if (victim == running_.end()) return false;
    if (running_.size() == 1 && inflight_.empty()) {
      finish(**victim, FinishReason::Error, "request does not fit the page pool");
      running_.clear();
      return false;
    }
    if (inflight_.empty() || running_.size() > 1) {
      preempt(**victim);
      waiting_.push_front(std::move(*victim));
      running_.erase(victim);
    } else {
      return false;
    }
  }
  if (plan.empty()) return false;

  StepInput in;
  for (auto& it : plan) {
    Seq& s = *it.s;
    if (!alloc_pages(s, s.computed + it.n)) throw std::logic_error("engine: page accounting");
    const int t0 = int(in.ids.size());
    in.chunks.push_back({s.slot, s.computed, it.n, t0});
    for (int k = 0; k < it.n; ++k) {
      in.ids.push_back(s.tokens[size_t(s.computed + k)]);
      in.pos.push_back(s.computed + k);
    }
    if (it.logits) {
      it.row = int(in.logit_rows.size());
      in.logit_rows.push_back(t0 + it.n - 1);
    }
  }
  if (tables_dirty_) {
    model_.set_block_tables(tables_);
    tables_dirty_ = false;
  }
  uint64_t ticket = 0;
  try {
    ticket = model_.submit(in);
  } catch (const std::exception& e) {
    fail_all(std::string("step failed: ") + e.what());
    return false;
  }
  for (auto& it : plan) {
    it.s->computed += it.n;
    ++it.s->inflight;
    it.s->awaiting_logits |= it.logits;
  }
  inflight_.push_back({ticket, std::move(plan)});
  return true;
}

// Waits for the oldest submitted step, samples the sequences it ends, delivers their tokens.
void Engine::complete_step() {
  InFlight f = std::move(inflight_.front());
  inflight_.pop_front();
  try {
    model_.wait(f.ticket, logits_);
  } catch (const std::exception& e) {
    fail_all(std::string("step failed: ") + e.what());
    return;
  }
  int prefill = 0, decode = 0;
  const int V = model_.vocab();
  for (const auto& it : f.items) {
    Seq& s = *it.s;
    --s.inflight;
    (it.n == 1 && it.logits ? decode : prefill) += it.n;
    if (!it.logits) continue;
    s.awaiting_logits = false;
    if (s.abort_pending || s.done) continue;
    const Sampled smp = s.sampler.sample(logits_.data() + size_t(it.row) * V, V);
    const int32_t tok = smp.token;
    s.tokens.push_back(tok);
    s.sampler.accept(tok);
    const bool eos = tok == cfg_.eos_id && !s.req.ignore_eos;
    const bool stop_id = std::find(s.req.stop_token_ids.begin(), s.req.stop_token_ids.end(), tok) !=
                         s.req.stop_token_ids.end();
    if (eos || stop_id) {
      s.out.stop_token = tok;
      s.done = true;  // finished below, once nothing of it is in flight
      s.out.reason = FinishReason::Stop;
      continue;
    }
    s.out.tokens.push_back(tok);
    s.out.samples.push_back(smp);
    if ((s.req.max_tokens > 0 && s.generated() >= s.req.max_tokens) || int(s.tokens.size()) >= max_context_) {
      s.done = true;
      s.out.reason = FinishReason::Length;
    }
  }
  for (auto& s : running_) {
    if (s->inflight > 0) continue;
    if (s->abort_pending) {
      finish(*s, FinishReason::Abort);
    } else if (s->done && !s->out.finished) {
      const FinishReason r = s->out.reason;
      s->done = false;  // finish() sets it again
      finish(*s, r);
    } else if (!s->done && !s->out.tokens.empty()) {
      s->out.completion_tokens = s->generated();
      s->sink(std::move(s->out));
      s->out = GenOutput{};
      s->out.prompt_tokens = s->n_prompt;
    }
  }
  running_.erase(std::remove_if(running_.begin(), running_.end(),
                                [](const auto& s) { return s->done && s->inflight == 0; }),
                 running_.end());
  std::lock_guard lk(stats_mu_);
  ++stats_.steps;
  stats_.prefill_tokens += uint64_t(prefill);
  stats_.decode_tokens += uint64_t(decode);
}

// A failed step leaves the device state unknown: everything running or in flight fails.
void Engine::fail_all(const std::string& why) {
  inflight_.clear();
  for (auto& s : running_) {
    s->inflight = 0;
    finish(*s, FinishReason::Error, why);
  }
  running_.clear();
}

void Engine::loop() {
  for (;;) {
    {
      std::unique_lock lk(mu_);
      cv_.wait(lk, [&] {
        return stop_ || !incoming_.empty() || !aborts_.empty() || !running_.empty() || !waiting_.empty() ||
               !inflight_.empty();
      });
      if (stop_) return;
    }
    iterate();
  }
}

}  // namespace unholy::engine
