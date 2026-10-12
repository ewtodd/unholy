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
  bool admitted_once = false;
  // Prefix cache.
  std::vector<uint64_t> hashes;  // chained hash of each full page of `tokens`, computed as needed
  int registered = 0;            // leading pages entered in the cache (or found there)
  int done_pos = 0;              // positions whose steps have completed
  int restore = -1;              // snapshot to load into the slot with the next chunk
  int cut = 0;                   // the prompt's last page boundary: a prefill chunk ends there to save state
  int last_save = 0;             // position of the last state save
  int rolling = -1;              // the snapshot decoding saved last, superseded by the next
  uint64_t rolling_hash = 0;
  int cached = 0;                // prompt tokens served from the cache at the first admission

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
  cache_on_ = cfg_.prefix_cache && model_.num_snapshots() > 0;
  page_info_.resize(size_t(model_.num_pages()));
  if (cache_on_) {
    snap_info_.resize(size_t(model_.num_snapshots()));
    for (int s = model_.num_snapshots() - 1; s >= 0; --s) free_snaps_.push_back(s);
  }
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
  // Tail first: a sequence's last pages become evictable before its first ones.
  for (auto it = s.pages.rbegin(); it != s.pages.rend(); ++it) {
    PageInfo& pi = page_info_[size_t(*it)];
    if (--pi.refs > 0) continue;
    if (pi.cached) pi.lru = evictable_.insert(evictable_.end(), *it);
    else free_pages_.push_back(*it);
  }
  s.pages.clear();
  if (s.restore >= 0) put_snapshot(s.restore);
  s.restore = -1;
  if (s.slot >= 0) free_slots_.push_back(s.slot);
  s.slot = -1;
  s.computed = s.done_pos = s.registered = s.cut = s.last_save = 0;
  s.rolling = -1;
}

int32_t Engine::take_page() {
  int32_t p;
  if (!free_pages_.empty()) {
    p = free_pages_.back();
    free_pages_.pop_back();
  } else {
    p = evictable_.front();
    evictable_.pop_front();
    PageInfo& pi = page_info_[size_t(p)];
    page_by_hash_.erase(pi.hash);
    pi.cached = false;
  }
  page_info_[size_t(p)].refs = 1;
  return p;
}

void Engine::write_table(const Seq& s) {
  const size_t w = size_t(model_.max_pages_per_seq());
  std::copy(s.pages.begin(), s.pages.end(), tables_.begin() + ptrdiff_t(size_t(s.slot) * w));
  tables_dirty_ = true;
}

namespace {
uint64_t mix64(uint64_t x) {  // splitmix64 finalizer
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
}  // namespace

void Engine::hash_pages(Seq& s, int npages) {
  while (int(s.hashes.size()) < npages) {
    const size_t i = s.hashes.size();
    uint64_t h = i ? s.hashes.back() : 0x243f6a8885a308d3ull;
    for (int k = 0; k < page_tokens_; ++k)
      h = mix64(h + 0x9e3779b97f4a7c15ull + uint32_t(s.tokens[i * size_t(page_tokens_) + size_t(k)]));
    s.hashes.push_back(h);
  }
}

// The deepest page boundary (short of the last token, which must be computed for its logits) whose
// pages are all cached and whose state was saved: take the pages and the snapshot.
void Engine::match_prefix(Seq& s) {
  const int limit = (int(s.tokens.size()) - 1) / page_tokens_;
  if (!cache_on_ || limit <= 0) return;
  hash_pages(s, limit);
  int best = 0, snap = -1;
  for (int i = 0; i < limit; ++i) {
    if (!page_by_hash_.count(s.hashes[size_t(i)])) break;
    if (const auto it = snap_by_hash_.find(s.hashes[size_t(i)]); it != snap_by_hash_.end()) {
      best = i + 1;
      snap = it->second;
    }
  }
  if (best == 0) return;
  for (int i = 0; i < best; ++i) {
    const int32_t p = page_by_hash_.at(s.hashes[size_t(i)]);
    PageInfo& pi = page_info_[size_t(p)];
    if (pi.refs++ == 0) evictable_.erase(pi.lru);
    s.pages.push_back(p);
  }
  SnapInfo& si = snap_info_[size_t(snap)];
  if (si.refs++ == 0) snap_lru_.erase(si.lru);
  s.restore = snap;
  s.computed = s.done_pos = s.last_save = best * page_tokens_;
  s.registered = best;
}

// Enters the sequence's completed full pages into the cache (a page whose hash is already there stays
// private to the sequence and is freed with it).
void Engine::register_pages(Seq& s) {
  const int full = std::min(s.done_pos / page_tokens_, int(s.pages.size()));
  if (full <= s.registered) return;
  hash_pages(s, full);
  for (int i = s.registered; i < full; ++i) {
    const int32_t p = s.pages[size_t(i)];
    PageInfo& pi = page_info_[size_t(p)];
    if (!pi.cached && page_by_hash_.emplace(s.hashes[size_t(i)], p).second) {
      pi.hash = s.hashes[size_t(i)];
      pi.cached = true;
    }
  }
  s.registered = full;
}

int Engine::take_snapshot() {
  int id;
  if (!free_snaps_.empty()) {
    id = free_snaps_.back();
    free_snaps_.pop_back();
  } else if (!snap_lru_.empty()) {
    id = snap_lru_.front();
    snap_lru_.pop_front();
    snap_by_hash_.erase(snap_info_[size_t(id)].hash);
    snap_info_[size_t(id)].cached = false;
  } else {
    return -1;
  }
  snap_info_[size_t(id)].refs = 1;
  return id;
}

void Engine::put_snapshot(int id) {
  SnapInfo& si = snap_info_[size_t(id)];
  if (--si.refs > 0) return;
  if (si.cached) si.lru = snap_lru_.insert(snap_lru_.end(), id);
  else free_snaps_.push_back(id);
}

// Whether a chunk ending at `end` saves the slot's state: on page boundaries, at the prompt's last one,
// at every one decoding completes, and every 16 pages of prefill up to 256 pages (64 to 1024, then 256;
// with 256-token pages: every 4k tokens up to 64k, 16k up to 256k, then 64k).
bool Engine::save_at(const Seq& s, int end, bool decode) const {
  if (end % page_tokens_ != 0) return false;
  if (decode || end == s.cut) return true;
  const int pages = end / page_tokens_;
  const int stride = page_tokens_ * (pages <= 256 ? 16 : pages <= 1024 ? 64 : 256);
  return end - s.last_save >= stride;
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
  if (need - s.pages.size() > size_t(available_pages())) return false;
  while (s.pages.size() < need) s.pages.push_back(take_page());
  write_table(s);
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

  // Admission: a free slot, pages for the request's first chunk (after any cached prefix), and a page of
  // headroom for every running sequence (so decoders crossing a page boundary do not immediately
  // preempt the newcomer).
  while (!waiting_.empty() && !free_slots_.empty()) {
    Seq& s = *waiting_.front();
    match_prefix(s);
    const int first = std::min(int(s.tokens.size()) - s.computed, budget0);
    const int need =
        (s.computed + first + page_tokens_ - 1) / page_tokens_ - int(s.pages.size()) + int(running_.size());
    if (need > available_pages()) {
      release(s);  // gives the matched prefix back
      break;
    }
    s.slot = free_slots_.back();
    free_slots_.pop_back();
    write_table(s);
    s.admitted = ++admit_counter_;
    if (cache_on_) {
      s.cut = (int(s.tokens.size()) - 1) / page_tokens_ * page_tokens_;
      if (s.cut <= s.computed) s.cut = 0;
    }
    if (!s.admitted_once) {
      s.admitted_once = true;
      s.cached = s.computed;
      s.out.cached_tokens = s.cached;
      std::lock_guard lk(stats_mu_);
      stats_.prompt_tokens += uint64_t(s.n_prompt);
      stats_.cached_tokens += uint64_t(s.cached);
    }
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
  stats_.free_pages = available_pages();
  stats_.cached_pages = int(page_by_hash_.size());
  stats_.snapshots = int(snap_by_hash_.size());
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
        int n = std::min(remaining, budget);
        if (s.cut > s.computed && s.computed + n > s.cut) n = s.cut - s.computed;
        const bool logits = n == remaining;
        if (logits && logit_rows == 0) continue;
        plan.push_back({&s, n, logits, -1, -1, -1, s.computed + n});
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
    if (need <= size_t(available_pages())) break;
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
    if (s.restore >= 0) {
      in.restore.push_back({s.slot, s.restore});
      it.restore = s.restore;
      s.restore = -1;
    }
    if (cache_on_ && save_at(s, it.end, it.n == 1)) {
      it.save = take_snapshot();
      if (it.save >= 0) {
        in.save.push_back({s.slot, it.save});
        s.last_save = it.end;
      }
    }
    if (it.end == s.cut) s.cut = 0;
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
    if (it.restore >= 0) put_snapshot(it.restore);
    if (cache_on_) {
      s.done_pos = std::max(s.done_pos, it.end);
      register_pages(s);
      if (it.save >= 0) commit_snapshot(s, it.save, it.end);
    }
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
      s->out.cached_tokens = s->cached;
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
  clear_cache();
}

// Forgets every cached page and snapshot (after a failed step the device contents are unknown).
void Engine::clear_cache() {
  for (const int32_t p : evictable_) {
    page_info_[size_t(p)].cached = false;
    free_pages_.push_back(p);
  }
  evictable_.clear();
  for (auto& pi : page_info_) pi.cached = false;
  page_by_hash_.clear();
  snap_by_hash_.clear();
  snap_lru_.clear();
  free_snaps_.clear();
  for (size_t i = snap_info_.size(); i-- > 0;) {
    snap_info_[i] = SnapInfo{};
    free_snaps_.push_back(int(i));
  }
}

// A state save completed: the snapshot names the boundary `end` of the sequence's tokens. Decoding saves
// at every page it completes; the save it supersedes is moved to the front of the eviction order.
void Engine::commit_snapshot(Seq& s, int id, int end) {
  hash_pages(s, end / page_tokens_);
  const uint64_t h = s.hashes[size_t(end / page_tokens_ - 1)];
  SnapInfo& si = snap_info_[size_t(id)];
  if (snap_by_hash_.emplace(h, id).second) {
    si.hash = h;
    si.cached = true;
  }
  put_snapshot(id);
  if (!si.cached || end <= s.n_prompt) return;
  if (s.rolling >= 0 && s.rolling != id) {
    const auto f = snap_by_hash_.find(s.rolling_hash);
    SnapInfo& old = snap_info_[size_t(s.rolling)];
    if (f != snap_by_hash_.end() && f->second == s.rolling && old.refs == 0) {
      snap_lru_.erase(old.lru);
      old.lru = snap_lru_.insert(snap_lru_.begin(), s.rolling);
    }
  }
  s.rolling = id;
  s.rolling_hash = h;
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
