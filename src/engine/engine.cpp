#include "engine/engine.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
  FinishReason abort_reason = FinishReason::Abort;
  // Request log.
  std::chrono::steady_clock::time_point t_submit, t_admit, t_first;
  bool first_token = false;
  int preemptions = 0, spec_steps = 0, spec_drafted = 0, spec_accepted = 0;
  // Prefix cache.
  std::vector<uint64_t> hashes;  // chained hash of each full page of `tokens`, computed as needed
  int registered = 0;            // leading pages entered in the cache (or found there)
  int done_pos = 0;              // positions whose steps have completed
  int restore = -1;              // snapshot to load into the slot with the next chunk
  bool loading = false;          // its restore snapshot is being read from disk (load_id): not scheduled until done
  int load_dslot = -1;           // the disk slot it reads (a reference on it), once matched
  uint64_t load_id = 0, load_hash = 0;
  bool cached_by_load = false;   // its first admission counted the disk prefix as cached tokens
  bool waiting_slot = false;       // its save at the prompt's last boundary waits for a RAM slot (since slot_wait)
  std::chrono::steady_clock::time_point slot_wait;
  int cut = 0;                   // the prompt's last page boundary: a prefill chunk ends there to save state
  int last_save = 0;             // position of the last state save
  int rolling = -1;              // the snapshot decoding saved last, superseded by the next
  uint64_t rolling_hash = 0;
  int cached = 0;                // prompt tokens served from the cache at the first admission
  // Speculative decoding: the drafter's proposals for the tokens after the last sampled one.
  std::vector<int32_t> drafts;
  std::vector<float> draft_conf;

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
  spec_max_ = std::min({cfg_.spec_drafts, model_.draft_block(), model_.max_verify() - 1});
  spec_on_ = spec_max_ > 0;
  page_info_.resize(size_t(model_.num_pages()));
  if (cache_on_) {
    snap_info_.resize(size_t(model_.num_snapshots()));
    for (int s = model_.num_snapshots() - 1; s >= 0; --s) free_snaps_.push_back(s);
  }
  if (cache_on_) init_disk();
  slots_ = model_.max_slots();
  if (cfg_.max_running > 0) slots_ = std::min(slots_, cfg_.max_running);
  for (int s = slots_ - 1; s >= 0; --s) free_slots_.push_back(s);
  for (int p = model_.num_pages() - 1; p >= 0; --p) free_pages_.push_back(p);
  tables_.assign(size_t(model_.max_slots()) * model_.max_pages_per_seq(), 0);
  stats_.total_pages = stats_.free_pages = model_.num_pages();
  stats_.disk_slots = disk_on_ ? cfg_.disk_snapshots : 0;
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
  if (io_thread_.joinable()) {
    {
      std::lock_guard lk(io_mu_);
      io_stop_ = true;
    }
    io_cv_.notify_all();
    io_thread_.join();
  }
  if (fd_ >= 0) {
    ::close(fd_);
    ::unlink(path_.c_str());
  }
}

uint64_t Engine::submit(GenRequest req, Sink sink) {
  std::lock_guard lk(mu_);
  const uint64_t id = next_id_++;
  auto s = std::make_unique<Seq>(id, std::move(req), std::move(sink), cfg_.seed * 0x9e3779b97f4a7c15ull + id);
  s->tokens = s->req.prompt;
  s->n_prompt = int(s->tokens.size());
  s->out.prompt_tokens = s->n_prompt;
  s->sampler.set_prompt(s->tokens);
  s->t_submit = std::chrono::steady_clock::now();
  incoming_.push_back(std::move(s));
  cv_.notify_all();
  return id;
}

void Engine::abort(uint64_t id, FinishReason why) {
  std::lock_guard lk(mu_);
  aborts_.emplace_back(id, why);
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
  if (cfg_.on_finish) {
    const auto now = std::chrono::steady_clock::now();
    const auto sec = [](auto d) { return std::chrono::duration<double>(d).count(); };
    RequestLog l;
    l.id = s.id;
    l.tag = s.req.tag;
    l.reason = r;
    l.error = err;
    l.prompt_tokens = s.n_prompt;
    l.cached_tokens = s.cached;
    l.completion_tokens = s.out.completion_tokens;
    l.images = int(s.req.images.size());
    l.preemptions = s.preemptions;
    l.queue_s = sec((s.admitted_once ? s.t_admit : now) - s.t_submit);
    if (s.first_token) {
      l.ttft_s = sec(s.t_first - s.t_submit);
      l.prefill_s = sec(s.t_first - s.t_admit);
      l.decode_s = sec(now - s.t_first);
    }
    l.spec_steps = s.spec_steps;
    l.spec_drafted = s.spec_drafted;
    l.spec_accepted = s.spec_accepted;
    cfg_.on_finish(l);
  }
  if (s.sink) s.sink(std::move(s.out));
  s.out = GenOutput{};
}

void Engine::unref_pages(Seq& s) {
  // Tail first: a sequence's last pages become evictable before its first ones.
  for (auto it = s.pages.rbegin(); it != s.pages.rend(); ++it) {
    PageInfo& pi = page_info_[size_t(*it)];
    if (--pi.refs > 0) continue;
    if (pi.cached) pi.lru = evictable_.insert(evictable_.end(), *it);
    else free_pages_.push_back(*it);
  }
  s.pages.clear();
}

void Engine::release(Seq& s) {
  unref_pages(s);
  // A started read keeps its disk slot and RAM slot until it completes (poll_io); one not started gives them back.
  if (s.load_dslot >= 0 && !s.loading) put_disk(s.load_dslot);
  s.load_dslot = -1;
  s.loading = false;
  if (s.restore >= 0) put_snapshot(s.restore);
  s.restore = -1;
  if (s.slot >= 0) free_slots_.push_back(s.slot);
  s.slot = -1;
  s.computed = s.done_pos = s.registered = s.cut = s.last_save = 0;
  s.rolling = -1;
  s.drafts.clear();
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

// What a page hash takes for position p: the token, or for an image block's positions the image's hash
// and the offset in the block (every image block's ids are the same few type ids).
uint64_t Engine::token_key(const Seq& s, int p) const {
  for (const GenImage& im : s.req.images)
    if (p >= im.start && p < im.start + im.len) return mix64(im.hash + uint64_t(p - im.start));
  return uint32_t(s.tokens[size_t(p)]);
}

void Engine::hash_pages(Seq& s, int npages) {
  while (int(s.hashes.size()) < npages) {
    const size_t i = s.hashes.size();
    uint64_t h = i ? s.hashes.back() : 0x243f6a8885a308d3ull;
    for (int k = 0; k < page_tokens_; ++k)
      h = mix64(h + 0x9e3779b97f4a7c15ull + token_key(s, int(i) * page_tokens_ + k));
    s.hashes.push_back(h);
  }
}

// The deepest page boundary (short of the last token, which must be computed for its logits) whose
// pages are all cached and whose state was saved: take the pages and the snapshot.
void Engine::match_prefix(Seq& s) {
  const int limit = (int(s.tokens.size()) - 1) / page_tokens_;
  if (!cache_on_ || limit <= 0) return;
  hash_pages(s, limit);
  int best = 0, snap = -1, dbest = 0, dslot = -1;  // the deepest RAM snapshot; the deepest one only on disk
  for (int i = 0; i < limit; ++i) {
    if (!page_by_hash_.count(s.hashes[size_t(i)])) break;
    // Never resume inside a whole image block: its tokens see the whole block.
    const int boundary = (i + 1) * page_tokens_;
    bool inside = false;
    if (model_.whole_image_blocks())
      for (const GenImage& im : s.req.images) inside |= boundary > im.start && boundary < im.start + im.len;
    if (inside) continue;
    if (const auto it = snap_by_hash_.find(s.hashes[size_t(i)]); it != snap_by_hash_.end()) {
      best = i + 1;
      snap = it->second;
    } else if (const auto dt = disk_on_ ? disk_by_hash_.find(s.hashes[size_t(i)]) : disk_by_hash_.end();
               dt != disk_by_hash_.end()) {
      dbest = i + 1;
      dslot = dt->second;
    }
  }
  // A disk hit deeper than the RAM ones is read into a free RAM slot (none: the RAM hit, if any, is used).
  const bool from_disk = dbest > best && !free_snaps_.empty();
  if (from_disk) {
    best = dbest;
    snap = free_snaps_.back();
    free_snaps_.pop_back();
    SnapInfo& si = snap_info_[size_t(snap)];
    si.refs = 1;
    si.cached = si.stale = false;
    DiskInfo& di = disk_[size_t(dslot)];
    if (di.refs++ == 0) disk_lru_.erase(di.lru);
    s.load_dslot = dslot;
    s.load_hash = s.hashes[size_t(best - 1)];
  } else if (best == 0) {
    return;
  }
  for (int i = 0; i < best; ++i) {
    const int32_t p = page_by_hash_.at(s.hashes[size_t(i)]);
    PageInfo& pi = page_info_[size_t(p)];
    if (pi.refs++ == 0) evictable_.erase(pi.lru);
    s.pages.push_back(p);
  }
  if (!from_disk) {
    SnapInfo& si = snap_info_[size_t(snap)];
    if (si.refs++ == 0 && !si.io) snap_lru_.erase(si.lru);  // (a spilling one is not in the LRU)
    si.stale = false;
  }
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

// A cached snapshot that can be discarded without losing the prefix (it is superseded, or the disk has it).
bool Engine::droppable(int id) const {
  const SnapInfo& si = snap_info_[size_t(id)];
  return si.stale || disk_by_hash_.count(si.hash) > 0;
}

int Engine::take_snapshot() {
  int id;
  if (!free_snaps_.empty()) {
    id = free_snaps_.back();
    free_snaps_.pop_back();
  } else if (disk_on_) {  // only a cached snapshot that loses nothing: superseded, or already on disk
    const auto it = std::find_if(snap_lru_.begin(), snap_lru_.end(), [&](int i) { return droppable(i); });
    if (it == snap_lru_.end()) return -1;
    id = *it;
    snap_lru_.erase(it);
    snap_by_hash_.erase(snap_info_[size_t(id)].hash);
    snap_info_[size_t(id)].cached = false;
    snap_info_[size_t(id)].stale = false;
  } else if (!snap_lru_.empty()) {
    id = snap_lru_.front();
    snap_lru_.pop_front();
    snap_by_hash_.erase(snap_info_[size_t(id)].hash);
    snap_info_[size_t(id)].cached = false;
    snap_info_[size_t(id)].stale = false;
  } else {
    return -1;
  }
  snap_info_[size_t(id)].refs = 1;
  return id;
}

void Engine::put_snapshot(int id) {
  SnapInfo& si = snap_info_[size_t(id)];
  if (--si.refs > 0 || si.io) return;  // an I/O in flight settles it when it completes
  settle_snapshot(id);
}

// An unreferenced snapshot with no I/O in flight: cached (evictable) or free.
void Engine::settle_snapshot(int id) {
  SnapInfo& si = snap_info_[size_t(id)];
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

// The prompt's last page boundary, where a prefill chunk ends to save state (0: none past the cached prefix).
void Engine::set_cut(Seq& s) {
  s.cut = (int(s.tokens.size()) - 1) / page_tokens_ * page_tokens_;
  for (const GenImage& im : s.req.images)  // not inside a whole image block
    if (model_.whole_image_blocks() && s.cut > im.start && s.cut < im.start + im.len) s.cut = im.start / page_tokens_ * page_tokens_;
  if (s.cut <= s.computed) s.cut = 0;
}

// A prefix whose snapshot could not be read: the sequence starts over from position 0 in its slot.
void Engine::drop_prefix(Seq& s) {
  unref_pages(s);
  if (s.restore >= 0) put_snapshot(s.restore);
  s.restore = -1;
  s.computed = s.done_pos = s.registered = s.last_save = 0;
  set_cut(s);
}

void Engine::preempt(Seq& s) {
  release(s);
  ++s.preemptions;
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
  std::vector<std::pair<uint64_t, FinishReason>> aborts;
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
    else if (std::any_of(s->req.images.begin(), s->req.images.end(), [&](const GenImage& im) {
               return (model_.whole_image_blocks() && im.len > cfg_.max_batch_tokens - slots_) || im.start < 0 ||
                      im.start + im.len > s->n_prompt || !im.rows;
             }))
      finish(*s, FinishReason::Error, "an image block does not fit one step (or lies outside the prompt)");
    else waiting_.push_back(std::move(s));
  }
  for (const auto& [id, why] : aborts) {
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
      if ((*it)->id == id) {
        finish(**it, why);
        waiting_.erase(it);
        break;
      }
    for (auto it = running_.begin(); it != running_.end(); ++it)
      if ((*it)->id == id) {
        if ((*it)->inflight > 0) {
          (*it)->abort_pending = true;  // its step still uses the slot's state; finish on completion
          (*it)->abort_reason = why;
        } else {
          finish(**it, why);
          running_.erase(it);
        }
        break;
      }
  }
}

bool Engine::iterate() {
  intake();
  const bool polled = poll_io();
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
    if (cache_on_) set_cut(s);
    if (!s.admitted_once) {
      s.admitted_once = true;
      s.t_admit = std::chrono::steady_clock::now();
      s.cached = s.computed;
      s.cached_by_load = s.load_dslot >= 0;
      s.out.cached_tokens = s.cached;
      std::lock_guard lk(stats_mu_);
      stats_.prompt_tokens += uint64_t(s.n_prompt);
      stats_.cached_tokens += uint64_t(s.cached);
    }
    if (s.load_dslot >= 0) start_load(s);
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
  bool submitted = false, completed = false;
  if (running_.empty() && inflight_.empty()) {
    if (!waiting_.empty()) {
      // Nothing runs and the head does not fit an empty pool: it never will.
      finish(*waiting_.front(), FinishReason::Error, "request does not fit the page pool");
      waiting_.pop_front();
      completed = true;
    }
  } else {
    submitted = int(inflight_.size()) < model_.max_inflight() && submit_step(budget0);
    if (!inflight_.empty() && (int(inflight_.size()) >= model_.max_inflight() || !submitted)) {
      complete_step();
      completed = true;
    }
  }
  spill();
  // Counts refreshed on every path (an abort during intake can empty the running set).
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::lock_guard lk(stats_mu_);
  if (submitted || completed) stats_.busy_seconds += dt;
  stats_.running = int(running_.size());
  stats_.waiting = int(waiting_.size());
  stats_.free_pages = available_pages();
  stats_.cached_pages = int(page_by_hash_.size());
  stats_.snapshots = int(snap_by_hash_.size());
  stats_.free_snapshots = int(free_snaps_.size() + snap_lru_.size());
  stats_.disk_snapshots = int(disk_by_hash_.size());
  stats_.snapshot_io = io_pending_;
  return submitted || completed || polled;
}

// Plans and submits one step over the sequences that are not waiting for logits: decoding sequences
// first (one token each), then prefill chunks in admission order. Returns false if nothing was
// submitted.
bool Engine::submit_step(int budget0) {
  std::vector<InFlight::Item> plan;
  const auto build = [&]() {
    plan.clear();
    int musts = 0, snap_budget = 0;  // saves at a prompt's last boundary planned; RAM slots they can take
    if (disk_on_) {
      snap_budget = int(free_snaps_.size());
      for (const int id : snap_lru_) snap_budget += droppable(id);
    }
    int budget = budget0, logit_rows = model_.max_logit_rows();
    const auto rem = [](const Seq* s) { return int(s->tokens.size()) - s->computed; };
    // Plans up to `cap` (<= budget) tokens of s.
    const auto take = [&](Seq& s, int cap, bool decode) {
      const int remaining = rem(&s);
      int n = std::min(remaining, cap);
      // A prefill chunk that stops short of the prompt ends on a page boundary when that leaves it any
      // tokens (state is saved for the prefix cache only there).
      if (!decode && n < remaining) {
        const int aligned = (s.computed + n) / page_tokens_ * page_tokens_ - s.computed;
        if (aligned > 0) n = aligned;
      }
      if (s.cut > s.computed && s.computed + n > s.cut) n = s.cut - s.computed;
      // An image block goes into one chunk whole: stop before it, or take all of it when the step
      // has room (even past the request's cap), else wait for a step that does.
      for (const GenImage& im : s.req.images) {
        if (!model_.whole_image_blocks()) break;
        const int a = im.start, e = im.start + im.len;
        if (n <= 0 || s.computed >= e || s.computed + n <= a || s.computed + n >= e) continue;
        if (a > s.computed) n = a - s.computed;
        else n = e - s.computed <= budget ? e - s.computed : 0;
      }
      if (n <= 0) return;
      const bool logits = n == remaining;
      if (logits && logit_rows == 0) return;
      // With the disk tier a save at the prompt's last boundary needs a RAM slot that loses nothing (a free one, or a
      // droppable one): when there is none and slots are about to free (spills, steps), wait for them instead.
      if (disk_on_ && !decode && s.cut > 0 && s.computed + n == s.cut) {
        if (musts >= snap_budget && (io_pending_ > 0 || !inflight_.empty())) {
          // (not for long: a stalled disk must not hold a request back; the save is then skipped)
          const auto t = std::chrono::steady_clock::now();
          if (!s.waiting_slot) s.waiting_slot = true, s.slot_wait = t;
          if (t - s.slot_wait < std::chrono::milliseconds(cfg_.save_wait_ms)) return;
        }
        ++musts;
      }
      s.waiting_slot = false;
      // Decoding with drafts: verify as many as the step, the context and max_tokens allow.
      int k = 0;
      if (decode && spec_on_ && !s.drafts.empty()) {
        int confident = 0;
        while (confident < int(s.drafts.size()) && s.draft_conf[size_t(confident)] >= cfg_.spec_min_conf) ++confident;
        k = std::min({confident, spec_max_, budget - 1, logit_rows - 1, max_context_ - 1 - s.computed});
        if (s.req.max_tokens > 0) k = std::min(k, s.req.max_tokens - s.generated() - 1);
        k = std::max(k, 0);
      }
      plan.push_back({&s, n + k, logits, -1, -1, -1, s.computed + n + k, k});
      budget -= n + k;
      logit_rows -= logits ? 1 + k : 0;
    };
    // Decodes first (admission order), then prefill chunks in the configured order.
    std::vector<Seq*> prefills, decodes;
    for (auto& sp : running_) {
      Seq& s = *sp;
      if (s.awaiting_logits || s.abort_pending || s.done || s.loading || rem(&s) <= 0) continue;
      (rem(&s) > 1 ? prefills : decodes).push_back(&s);
    }
    // Two micro-batches (cfg_.decode_split): a step takes half of the decoders, counting those of the step
    // in flight; the ones held back are not awaiting logits, so the next step takes them. Two halves that
    // alternate (each submitted when the other's predecessor completes) keep both cards busy; groups that
    // arrive unbalanced (one request, then seven) even out within a few steps.
    size_t ndec = decodes.size();
    if (cfg_.decode_split && model_.max_inflight() > 1 && ndec >= 2) {
      size_t fly = 0;
      for (const auto& f : inflight_)
        for (const auto& it : f.items) fly += it.logits && it.n == 1 + it.ndraft;
      ndec = std::min(ndec, std::max<size_t>(1, (fly + ndec + 1) / 2));
    }
    for (size_t i = 0; i < ndec && budget > 0; ++i) take(*decodes[i], budget, true);
    if (prefills.empty() || budget == 0) return;
    const bool shortest = cfg_.prefill_order == PrefillOrder::Shortest;
    if (shortest)  // stable: admission order among equals
      std::stable_sort(prefills.begin(), prefills.end(), [&](const Seq* a, const Seq* b) { return rem(a) < rem(b); });
    const int eligible = int(running_.size() + waiting_.size());
    int thr = eligible > 1 ? cfg_.long_prefill_threshold : 0;
    if (thr > 0 && cfg_.long_prefill_adaptive) thr = std::max(thr, cfg_.max_batch_tokens / eligible);
    // Shortest first: the earliest-submitted prefill keeps a quarter of the step (at least a page), so a
    // stream of short prompts slows it but cannot stall it.
    Seq* oldest = nullptr;
    int reserve = 0;
    if (shortest && prefills.size() > 1) {
      for (Seq* s : prefills)
        if (!oldest || s->id < oldest->id) oldest = s;
      reserve = std::min({rem(oldest), budget, std::max(page_tokens_, budget / 4 / page_tokens_ * page_tokens_)});
    }
    for (Seq* s : prefills) {
      if (budget == 0) break;
      int cap = s == oldest ? budget : budget - reserve;
      if (thr > 0) cap = std::min(cap, thr);
      if (cap > 0) take(*s, cap, false);
      if (s == oldest) reserve = 0;
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
  in.sync_saves = disk_on_;  // spills read the saves from host memory
  if (disk_on_)  // the saves at a prompt's last boundary take their slots before the others do
    for (auto& it : plan)
      if (cache_on_ && it.end == it.s->cut && save_at(*it.s, it.end, it.n == 1 || it.ndraft > 0)) it.save = take_snapshot();
  for (auto& it : plan) {
    Seq& s = *it.s;
    if (!alloc_pages(s, s.computed + it.n)) throw std::logic_error("engine: page accounting");
    const int t0 = int(in.ids.size());
    if (s.restore >= 0) {
      in.restore.push_back({s.slot, s.restore});
      it.restore = s.restore;
      s.restore = -1;
    }
    if (cache_on_ && save_at(s, it.end, it.n == 1 || it.ndraft > 0)) {
      if (it.save < 0) it.save = take_snapshot();
      if (it.save >= 0) {
        in.save.push_back({s.slot, it.save});
        s.last_save = it.end;
      }
    }
    if (it.end == s.cut) s.cut = 0;
    in.chunks.push_back({s.slot, s.computed, it.n, t0});
    for (const GenImage& im : s.req.images) {  // whole blocks lie wholly inside chunks (planning above)
      const int a = std::max(im.start, s.computed), e = std::min(im.start + im.len, s.computed + it.n - it.ndraft);
      if (a >= e) continue;
      const size_t dim = im.rows->size() / size_t(im.len);
      in.image_rows.push_back({t0 + a - s.computed, e - a, im.rows->data() + size_t(a - im.start) * dim});
      if (model_.whole_image_blocks()) in.image_spans.push_back({s.slot, im.span_start, im.span_end});
    }
    const int fed = it.n - it.ndraft;  // tokens of the sequence; the rest are drafts
    for (int k = 0; k < it.n; ++k) {
      in.ids.push_back(k < fed ? s.tokens[size_t(s.computed + k)] : s.drafts[size_t(k - fed)]);
      in.pos.push_back(s.computed + k);
    }
    if (it.ndraft > 0) in.backup.push_back({s.slot, s.computed, it.n});
    if (it.logits) {  // a verify chunk wants every row (its token and each draft are judged)
      it.row = int(in.logit_rows.size());
      for (int k = it.ndraft > 0 ? 0 : it.n - 1; k < it.n; ++k) in.logit_rows.push_back(t0 + k);
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
  // Appends a sampled token; true when it ends the sequence.
  const auto emit = [&](Seq& s, const Sampled& smp) {
    const int32_t tok = smp.token;
    s.tokens.push_back(tok);
    s.sampler.accept(tok);
    if (!s.first_token) {
      s.first_token = true;
      s.t_first = std::chrono::steady_clock::now();
    }
    const auto& ends = cfg_.eos_ids;
    const bool eos = (tok == cfg_.eos_id || std::find(ends.begin(), ends.end(), tok) != ends.end()) && !s.req.ignore_eos;
    const bool stop_id = std::find(s.req.stop_token_ids.begin(), s.req.stop_token_ids.end(), tok) !=
                         s.req.stop_token_ids.end();
    if (eos || stop_id) {
      s.out.stop_token = tok;
      s.done = true;  // finished below, once nothing of it is in flight
      s.out.reason = FinishReason::Stop;
      return true;
    }
    s.out.tokens.push_back(tok);
    s.out.samples.push_back(smp);
    if ((s.req.max_tokens > 0 && s.generated() >= s.req.max_tokens) || int(s.tokens.size()) >= max_context_) {
      s.done = true;
      s.out.reason = FinishReason::Length;
      return true;
    }
    return false;
  };
  std::vector<RingRestore> restores;
  std::vector<Seq*> sampled;
  for (const auto& it : f.items) {
    Seq& s = *it.s;
    --s.inflight;
    if (it.restore >= 0) put_snapshot(it.restore);
    const int s0 = it.end - it.n;
    // A verify chunk: the fed token, then drafts while the target agrees (speculative sampling), then
    // the target's own token (a replacement, or the bonus after the last draft).
    int valid_end = it.end, accepted = 0;
    if (it.ndraft > 0 && !s.abort_pending && !s.done) {
      bool stopped = false;
      for (int j = 0; j <= it.ndraft && !stopped; ++j) {
        const float* row = logits_.data() + size_t(it.row + j) * V;
        bool ok = false;
        const Sampled smp = j < it.ndraft ? s.sampler.verify(row, V, s.drafts[size_t(j)], ok) : s.sampler.sample(row, V);
        stopped = emit(s, smp);
        if (!ok) break;
        ++accepted;
      }
      valid_end = s0 + 1 + accepted;
      s.computed = valid_end;
      if (valid_end < it.end && !s.done) restores.push_back({s.slot, s0, valid_end, it.end});
      ++s.spec_steps;
      s.spec_drafted += it.ndraft;
      s.spec_accepted += accepted;
      std::lock_guard lk(stats_mu_);
      ++stats_.spec_steps;
      stats_.spec_drafted += uint64_t(it.ndraft);
      stats_.spec_accepted += uint64_t(accepted);
      for (int j = 0; j < std::min(it.ndraft, int(stats_.spec_pos_drafted.size())); ++j) {
        ++stats_.spec_pos_drafted[size_t(j)];
        stats_.spec_pos_accepted[size_t(j)] += j < accepted;
      }
    }
    if (cache_on_) {
      s.done_pos = std::max(s.done_pos, valid_end);
      register_pages(s);
      if (it.save >= 0) {
        if (valid_end == it.end) commit_snapshot(s, it.save, it.end);
        else put_snapshot(it.save);  // saved past a rejected draft: not the sequence's state
      }
    }
    (it.n == 1 + it.ndraft && it.logits ? decode : prefill) += it.ndraft ? 1 + accepted : it.n;
    if (!it.logits) continue;
    s.awaiting_logits = false;
    s.drafts.clear();
    if (s.abort_pending || s.done) continue;
    if (it.ndraft == 0) emit(s, s.sampler.sample(logits_.data() + size_t(it.row) * V, V));
    if (!s.done) sampled.push_back(&s);
  }
  // Drafts for the sequences that just sampled (and the restores of rejected positions first).
  if (spec_on_ && (!sampled.empty() || !restores.empty())) {
    std::vector<DraftReq> reqs;
    for (Seq* s : sampled)
      if (int(s->tokens.size()) - s->computed == 1 && s->computed + 1 < max_context_)
        reqs.push_back({s->slot, s->computed - 1, s->tokens.back()});
    std::vector<int32_t> toks;
    std::vector<float> conf;
    try {
      model_.draft(restores, reqs, toks, conf);
    } catch (const std::exception& e) {
      fail_all(std::string("draft failed: ") + e.what());
      return;
    }
    const size_t Bd = size_t(model_.draft_block());
    size_t r = 0;
    for (Seq* s : sampled) {
      if (int(s->tokens.size()) - s->computed != 1 || s->computed + 1 >= max_context_) continue;
      s->drafts.assign(toks.begin() + ptrdiff_t(r * Bd), toks.begin() + ptrdiff_t((r + 1) * Bd));
      s->draft_conf.assign(conf.begin() + ptrdiff_t(r * Bd), conf.begin() + ptrdiff_t((r + 1) * Bd));
      ++r;
    }
  }
  for (auto& s : running_) {
    if (s->inflight > 0) continue;
    if (s->abort_pending) {
      finish(*s, s->abort_reason);
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
  // Disk reads and writes in flight finish first (an error path: blocking here is fine).
  while (io_pending_ > 0) {
    if (!poll_io()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  disk_by_hash_.clear();
  disk_lru_.clear();
  free_disk_.clear();
  for (size_t i = disk_.size(); i-- > 0;) {
    disk_[i] = DiskInfo{};
    free_disk_.push_back(int(i));
  }
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
    if (f != snap_by_hash_.end() && f->second == s.rolling && old.refs == 0 && !old.io) {
      snap_lru_.erase(old.lru);
      old.lru = snap_lru_.insert(snap_lru_.begin(), s.rolling);
      old.stale = true;
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
               !inflight_.empty() || io_ready_;
      });
      if (stop_) return;
    }
    // Nothing to do but wait for the disk (sequences loading): sleep until a read finishes or work arrives.
    if (!iterate() && io_pending_ > 0) {
      std::unique_lock lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(2),
                   [&] { return stop_ || !incoming_.empty() || !aborts_.empty() || io_ready_; });
    }
  }
}

namespace {
// 64 bits over a buffer (8 independent lanes; the I/O thread checks what it read against what it wrote).
uint64_t sum_bytes(const uint8_t* p, size_t n) {
  uint64_t h[8];
  for (int l = 0; l < 8; ++l) h[l] = 0x9e3779b97f4a7c15ull * uint64_t(l + 1);
  size_t i = 0;
  for (; i + 64 <= n; i += 64)
    for (int l = 0; l < 8; ++l) {
      uint64_t w;
      std::memcpy(&w, p + i + size_t(l) * 8, 8);
      h[l] = (h[l] ^ w) * 0xff51afd7ed558ccdull;
      h[l] ^= h[l] >> 32;
    }
  uint64_t t = n;
  for (int l = 0; l < 8; ++l) t = mix64(t + h[l]);
  for (; i < n; ++i) t = mix64(t + p[i]);
  return t;
}
}  // namespace

void Engine::disk_fail(const std::string& why) {
  if (!disk_on_) return;
  disk_on_ = false;
  std::fprintf(stderr, "engine: snapshot disk tier off: %s\n", why.c_str());
}

// Creates the file (truncated, preallocated) and starts the I/O thread. Failures turn the tier off.
void Engine::init_disk() {
  if (cfg_.disk_snapshots <= 0 && cfg_.snapshot_dir.empty()) return;
  if (cfg_.disk_snapshots <= 0 || cfg_.snapshot_dir.empty()) {
    std::fprintf(stderr, "engine: snapshot disk tier needs both a directory and a slot count; off\n");
    return;
  }
  disk_on_ = true;
  std::vector<std::pair<uint8_t*, size_t>> segs;
  model_.snapshot_host(0, segs);
  size_t bytes = 0;
  bool aligned = true;
  for (const auto& [p, n] : segs) {
    bytes += n;
    aligned &= uintptr_t(p) % 4096 == 0 && n % 4096 == 0;
  }
  if (bytes == 0) return disk_fail("the model's snapshots are not in host memory");
  slot_bytes_ = bytes;
  std::error_code ec;
  std::filesystem::create_directories(cfg_.snapshot_dir, ec);
  path_ = cfg_.snapshot_dir + "/snapshots.bin";
  direct_ = aligned;
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | (direct_ ? O_DIRECT : 0), 0600);
  if (fd_ < 0 && direct_ && errno == EINVAL) {  // tmpfs and the like
    direct_ = false;
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  }
  if (fd_ < 0) return disk_fail("open " + path_ + ": " + std::strerror(errno));
  const off_t total = off_t(slot_bytes_) * cfg_.disk_snapshots;
  if (const int e = ::posix_fallocate(fd_, 0, total); e == EOPNOTSUPP || e == EINVAL) {
    if (::ftruncate(fd_, total) != 0) return disk_fail(std::string("ftruncate: ") + std::strerror(errno));
  } else if (e != 0) {
    return disk_fail(std::string("allocating ") + std::to_string(total >> 20) + " MiB in " + path_ + ": " + std::strerror(e));
  }
  disk_.resize(size_t(cfg_.disk_snapshots));
  for (int d = cfg_.disk_snapshots - 1; d >= 0; --d) free_disk_.push_back(d);
  spill_reserve_ = std::clamp(model_.num_snapshots() - 1, 1, 2);
  io_thread_ = std::thread([this] { io_loop(); });
  std::fprintf(stderr, "engine: snapshot disk tier: %d slots of %.1f MB in %s (%s), %d RAM snapshots kept free\n",
               cfg_.disk_snapshots, double(slot_bytes_) / 1e6, path_.c_str(),
               direct_ ? "O_DIRECT" : "buffered, pages dropped after each I/O", spill_reserve_);
}

// The I/O thread: one operation at a time, in queue order.
void Engine::io_loop() {
  for (;;) {
    IoDone d;
    {
      std::unique_lock lk(io_mu_);
      io_cv_.wait(lk, [&] { return io_stop_ || !io_q_.empty(); });
      if (io_stop_) return;
      d.op = std::move(io_q_.front());
      io_q_.pop_front();
    }
    if (cfg_.io_hook) cfg_.io_hook(d.op.write);
    d.ok = true;
    const off_t base = off_t(d.op.dslot) * off_t(slot_bytes_);
    off_t off = base;
    uint64_t sum = 0;
    for (const auto& [p, n] : d.op.segs) {
      for (size_t done = 0; done < n;) {
        const ssize_t r = d.op.write ? ::pwrite(fd_, p + done, n - done, off + off_t(done))
                                     : ::pread(fd_, p + done, n - done, off + off_t(done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
          d.ok = false;
          d.err = std::string(d.op.write ? "write" : "read") + " of snapshot slot " + std::to_string(d.op.dslot) + ": " +
                  (r < 0 ? std::strerror(errno) : "end of file");
          break;
        }
        done += size_t(r);
      }
      if (!d.ok) break;
      off += off_t(n);
      sum = mix64(sum ^ sum_bytes(p, n));
    }
    if (!direct_) ::posix_fadvise(fd_, base, off_t(slot_bytes_), POSIX_FADV_DONTNEED);
    if (d.ok && d.op.write) d.sum = sum;
    if (d.ok && !d.op.write && sum != d.op.sum) {
      d.ok = false;
      d.err = "snapshot slot " + std::to_string(d.op.dslot) + " read back differently from what was written";
    }
    {
      std::lock_guard lk(io_mu_);
      io_done_.push_back(std::move(d));
    }
    {
      std::lock_guard lk(mu_);
      io_ready_ = true;
    }
    cv_.notify_all();
  }
}

void Engine::queue_io(bool write, int snap, int dslot, uint64_t id, uint64_t sum) {
  IoOp op;
  op.write = write;
  op.snap = snap;
  op.dslot = dslot;
  op.id = id;
  op.sum = sum;
  model_.snapshot_host(snap, op.segs);
  ++io_pending_;
  {
    std::lock_guard lk(io_mu_);
    io_q_.push_back(std::move(op));
  }
  io_cv_.notify_one();
}

// Reads the sequence's disk snapshot into the RAM slot match_prefix took; the sequence waits (loading).
void Engine::start_load(Seq& s) {
  SnapInfo& si = snap_info_[size_t(s.restore)];
  si.io = true;
  si.hash = s.load_hash;
  s.loading = true;
  s.load_id = ++io_ids_;
  queue_io(false, s.restore, s.load_dslot, s.load_id, disk_[size_t(s.load_dslot)].sum);
  std::lock_guard lk(stats_mu_);
  ++stats_.disk_hits;
}

int Engine::take_disk_slot() {
  int d;
  if (!free_disk_.empty()) {
    d = free_disk_.back();
    free_disk_.pop_back();
  } else if (!disk_lru_.empty()) {  // (slots being read are not in the LRU)
    d = disk_lru_.front();
    disk_lru_.pop_front();
    disk_by_hash_.erase(disk_[size_t(d)].hash);
    disk_[size_t(d)].valid = false;
  } else {
    return -1;
  }
  return d;
}

void Engine::put_disk(int d) {
  DiskInfo& di = disk_[size_t(d)];
  if (--di.refs == 0 && di.valid) di.lru = disk_lru_.insert(disk_lru_.end(), d);
}

// Keeps spill_reserve_ RAM snapshots free: the least recently used cached ones are written to disk (their RAM
// slots go free when the write completes), or dropped when a later rolling snapshot superseded them or the disk
// has them already. Only unreferenced cached snapshots are in the LRU, so none is being saved into or restored.
void Engine::spill() {
  if (!disk_on_) return;
  while (int(free_snaps_.size()) + spills_out_ < spill_reserve_ && !snap_lru_.empty()) {
    const int id = snap_lru_.front();
    SnapInfo& si = snap_info_[size_t(id)];
    const auto dt = disk_by_hash_.find(si.hash);
    if (si.stale || dt != disk_by_hash_.end()) {
      if (dt != disk_by_hash_.end() && disk_[size_t(dt->second)].refs == 0) {  // used again: the disk copy too
        disk_lru_.erase(disk_[size_t(dt->second)].lru);
        disk_[size_t(dt->second)].lru = disk_lru_.insert(disk_lru_.end(), dt->second);
      }
      snap_lru_.pop_front();
      snap_by_hash_.erase(si.hash);
      si.cached = si.stale = false;
      free_snaps_.push_back(id);
      continue;
    }
    const int d = take_disk_slot();
    if (d < 0) break;  // every disk slot is being read
    snap_lru_.pop_front();
    si.io = true;
    ++spills_out_;
    queue_io(true, id, d, 0, 0);
  }
}

// Handles the finished disk operations; true if there were any.
bool Engine::poll_io() {
  io_ready_ = false;  // before taking the completions: one that lands after sets it again
  if (io_pending_ == 0) return false;
  std::vector<IoDone> done;
  {
    std::lock_guard lk(io_mu_);
    done.swap(io_done_);
  }
  for (IoDone& d : done) {
    --io_pending_;
    SnapInfo& si = snap_info_[size_t(d.op.snap)];
    si.io = false;
    if (!d.ok) disk_fail(d.err);
    if (!d.op.write) {
      const int dslot = d.op.dslot;
      const uint64_t h = disk_[size_t(dslot)].hash;
      put_disk(dslot);
      Seq* seq = nullptr;  // none if it was preempted, aborted or finished meanwhile
      for (auto& r : running_)
        if (r->loading && r->load_id == d.op.id) seq = r.get();
      if (d.ok) {
        si.hash = h;
        if (snap_by_hash_.emplace(h, d.op.snap).second) si.cached = true;  // the RAM copy is cached too
        std::lock_guard lk(stats_mu_);
        stats_.disk_read_bytes += slot_bytes_;
      }
      if (seq) {
        seq->loading = false;
        seq->load_dslot = -1;
        seq->load_id = 0;
        if (!d.ok) {  // recomputed from the start
          drop_prefix(*seq);
          if (seq->cached_by_load) {
            std::lock_guard lk(stats_mu_);
            stats_.cached_tokens -= uint64_t(seq->cached);
            seq->cached = seq->out.cached_tokens = 0;
            seq->cached_by_load = false;
          }
        }
      } else if (si.refs == 0) {
        settle_snapshot(d.op.snap);
      }
    } else {
      --spills_out_;
      DiskInfo& di = disk_[size_t(d.op.dslot)];
      const bool stored = d.ok && disk_on_ && disk_by_hash_.emplace(si.hash, d.op.dslot).second;
      if (stored) {
        di.hash = si.hash;
        di.sum = d.sum;
        di.valid = true;
        di.refs = 0;
        di.lru = disk_lru_.insert(disk_lru_.end(), d.op.dslot);
        std::lock_guard lk(stats_mu_);
        ++stats_.spills;
        stats_.disk_write_bytes += slot_bytes_;
      } else {
        free_disk_.push_back(d.op.dslot);
      }
      if (si.refs == 0) {  // (a request that hit it meanwhile keeps the RAM copy; it is cached again when released)
        if (stored) {
          if (const auto f = snap_by_hash_.find(si.hash); f != snap_by_hash_.end() && f->second == d.op.snap)
            snap_by_hash_.erase(f);
          si.cached = false;
          free_snaps_.push_back(d.op.snap);
        } else {
          settle_snapshot(d.op.snap);
        }
      }
    }
  }
  return !done.empty();
}

}  // namespace unholy::engine

namespace unholy::engine {

std::string Engine::debug_state() const {
  std::string out = "waiting " + std::to_string(waiting_.size()) + ", running " + std::to_string(running_.size()) +
                    ", inflight " + std::to_string(inflight_.size()) + ", free slots " +
                    std::to_string(free_slots_.size()) + ", available pages " + std::to_string(available_pages()) +
                    ", cached pages " + std::to_string(page_by_hash_.size()) + "\n";
  for (const auto& s : waiting_)
    out += "waiting seq " + std::to_string(s->id) + ": tokens " + std::to_string(s->tokens.size()) + " computed " +
           std::to_string(s->computed) + " pages " + std::to_string(s->pages.size()) + "\n";
  for (const auto& s : running_) {
    out += "seq " + std::to_string(s->id) + ": tokens " + std::to_string(s->tokens.size()) + " prompt " +
           std::to_string(s->n_prompt) + " computed " + std::to_string(s->computed) + " inflight " +
           std::to_string(s->inflight) + " awaiting " + std::to_string(s->awaiting_logits) + " done " +
           std::to_string(s->done) + " abort " + std::to_string(s->abort_pending) + " drafts " +
           std::to_string(s->drafts.size()) + " cut " + std::to_string(s->cut) + " pages " +
           std::to_string(s->pages.size()) + " images";
    for (const auto& im : s->req.images) out += " [" + std::to_string(im.start) + "," + std::to_string(im.start + im.len) + ")";
    out += "\n";
  }
  return out;
}

}  // namespace unholy::engine
