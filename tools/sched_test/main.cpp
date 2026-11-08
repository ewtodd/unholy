// unholy-sched-test: the engine's scheduler against a fake model, on the CPU.
//
// The fake model keeps a paged token store addressed through the block tables the engine hands it,
// checks every chunk (positions continue the slot's sequence, rows are contiguous), and produces
// one-hot "logits" from a hash of the whole sequence read back through the block table. Greedy outputs
// must then equal a standalone simulation of each request, whatever the batching, chunking,
// preemption and slot/page reuse were — any aliasing of pages between live sequences or a lost
// chunk changes the hash. With prefix caching on, half the requests extend an earlier prompt (or prompt
// and reply), and the fake keeps per-slot state (a running hash of the slot's tokens) that snapshots
// save and restore: a restore that does not match the pages it continues over, or a chunk that does
// not continue the restored state, fails. The snapshots are bytes in aligned host buffers (StepModel::
// snapshot_host) that encode the state, so the disk tier's cases can check that what comes back from the
// file is what was saved.
//
//   unholy-sched-test [--requests N] [--seed S] [--disk-dir DIR]
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "engine/engine.hpp"
#include "server/metrics.hpp"

using namespace unholy::engine;

namespace {

constexpr int kVocab = 512, kEos = 1;

constexpr uint64_t kFnvBasis = 1469598103934665603ull;
uint64_t fnv_step(uint64_t h, int32_t t) { return (h ^ uint64_t(uint32_t(t))) * 1099511628211ull; }
uint64_t fnv(const int32_t* t, size_t n) {
  uint64_t h = kFnvBasis;
  for (size_t i = 0; i < n; ++i) h = fnv_step(h, t[i]);
  return h;
}
int32_t next_token(const std::vector<int32_t>& seq) {
  const uint64_t h = fnv(seq.data(), seq.size());
  if (h % 53 == 0) return kEos;
  return int32_t(2 + (h >> 8) % (kVocab - 2));
}

class FakeModel final : public StepModel {
 public:
  std::chrono::microseconds wait_delay{0};  // each wait() takes this long (the device time of a step)
  FakeModel(int slots, int pages_per_seq, int pages, int page_tokens, int max_tokens, int inflight, int snapshots,
            int draft_block)
      : inflight_(inflight), slots_(slots), ppseq_(pages_per_seq), pages_(pages), page_(page_tokens), max_tokens_(max_tokens),
        snapshots_(snapshots), draft_block_(draft_block), store_(size_t(pages) * page_tokens, -1),
        len_(size_t(slots), 0), state_(size_t(slots), kFnvBasis), backup_(size_t(slots)) {
    if (snapshots > 0) {
      host_ = static_cast<uint8_t*>(std::aligned_alloc(4096, size_t(snapshots) * kSnapBytes));
      std::memset(host_, 0, size_t(snapshots) * kSnapBytes);
    }
  }
  ~FakeModel() override { std::free(host_); }
  int num_snapshots() const override { return snapshots_; }
  // Two 4 KiB regions per snapshot (as the real ones are per card).
  void snapshot_host(int id, std::vector<std::pair<uint8_t*, size_t>>& segs) const override {
    segs.emplace_back(host_ + size_t(id) * kSnapBytes, 4096);
    segs.emplace_back(host_ + size_t(id) * kSnapBytes + 4096, 4096);
  }
  int draft_block() const override { return draft_block_; }
  int max_verify() const override { return draft_block_ ? draft_block_ + 1 : 0; }
  // Restores truncate the slot back to the accepted prefix (they must match the last backup); drafts
  // are the true continuation with a quarter of the tokens wrong (deterministically), so verification
  // accepts some and rejects others.
  void draft(const std::vector<RingRestore>& restore, const std::vector<DraftReq>& reqs, std::vector<int32_t>& tokens,
             std::vector<float>& conf) override {
    for (const RingRestore& r : restore) {
      Backup& b = backup_.at(size_t(r.slot));
      if (b.n == 0 || r.s0 != b.s0 || r.to != b.s0 + b.n || r.from <= r.s0 || r.from >= r.to)
        throw std::runtime_error("fake: restore does not match the slot's backup");
      len_[size_t(r.slot)] = r.from;
      uint64_t h = kFnvBasis;
      for (int p = 0; p < r.from; ++p) h = fnv_step(h, at(r.slot, p));
      state_[size_t(r.slot)] = h;
      b.n = 0;
      ++rollbacks;
    }
    tokens.clear();
    conf.clear();
    for (const DraftReq& q : reqs) {
      if (len_[size_t(q.slot)] != q.start + 1) throw std::runtime_error("fake: draft from a state that is not the sequence's");
      std::vector<int32_t> seq(size_t(q.start + 1));
      for (int p = 0; p <= q.start; ++p) seq[size_t(p)] = at(q.slot, p);
      seq.push_back(q.token);
      for (int i = 0; i < draft_block_; ++i) {
        const uint64_t h = fnv(seq.data(), seq.size());
        const int32_t t = (h >> 17) % 4 == 0 ? int32_t(2 + (h >> 29) % (kVocab - 2)) : next_token(seq);
        tokens.push_back(t);
        conf.push_back(1.0f);
        seq.push_back(t);
      }
      ++drafts;
    }
  }
  int vocab() const override { return kVocab; }
  int max_tokens() const override { return max_tokens_; }
  int max_logit_rows() const override { return 64; }
  int max_slots() const override { return slots_; }
  int max_pages_per_seq() const override { return ppseq_; }
  int num_pages() const override { return pages_; }
  int page_tokens() const override { return page_; }
  void set_block_tables(const std::vector<int32_t>& t) override {
    if (t.size() != size_t(slots_) * ppseq_) throw std::runtime_error("fake: table size");
    tables_ = t;
  }
  int max_inflight() const override { return inflight_; }
  // Executes at submission (steps run in submission order, as on the devices); wait() hands back the
  // logits, checking the engine waits in order and never exceeds the in-flight limit.
  uint64_t submit(const StepInput& in) override {
    if (int(pending_.size()) >= inflight_) throw std::runtime_error("fake: too many steps in flight");
    std::vector<float> logits;
    run(in, logits);
    pending_.push_back({next_, std::move(logits)});
    max_seen = std::max(max_seen, int(pending_.size()));
    return next_++;
  }
  void wait(uint64_t ticket, std::vector<float>& logits) override {
    if (wait_delay.count() > 0) std::this_thread::sleep_for(wait_delay);
    if (pending_.empty() || pending_.front().first != ticket) throw std::runtime_error("fake: out-of-order wait");
    logits = std::move(pending_.front().second);
    pending_.pop_front();
  }
  void run(const StepInput& in, std::vector<float>& logits) {
    ++steps;
    if (in.ids.size() != in.pos.size() || int(in.ids.size()) > max_tokens_) throw std::runtime_error("fake: T");
    int t = 0;
    std::vector<bool> seen(size_t(slots_), false);
    std::vector<int> chunk_of_row(in.ids.size());
    for (const StateOp& r : in.restore) {
      const Snap sn = load_snap(r.snapshot);
      state_[size_t(r.slot)] = sn.state;
      len_[size_t(r.slot)] = sn.len;
      uint64_t h = kFnvBasis;  // the pages the slot continues over must hold what the state saw
      for (int p = 0; p < sn.len; ++p) h = fnv_step(h, at(r.slot, p));
      if (h != sn.state) throw std::runtime_error("fake: restored state does not match the slot's pages");
      ++restores;
    }
    std::vector<uint64_t> row_state(in.ids.size());
    std::vector<bool> verify(size_t(slots_), false);
    for (const RingBackup& b : in.backup) {
      verify[size_t(b.slot)] = true;
      backup_[size_t(b.slot)] = {b.s0, b.n};
    }
    for (size_t ci = 0; ci < in.chunks.size(); ++ci) {
      const Chunk& c = in.chunks[ci];
      if (c.t0 != t || c.n <= 0 || seen[size_t(c.slot)]) throw std::runtime_error("fake: chunk layout");
      seen[size_t(c.slot)] = true;
      if (c.s0 != 0 && c.s0 != len_[size_t(c.slot)])
        throw std::runtime_error("fake: slot " + std::to_string(c.slot) + " chunk at " + std::to_string(c.s0) +
                                 " but the sequence has " + std::to_string(len_[size_t(c.slot)]));
      for (int k = 0; k < c.n; ++k) {
        const int p = c.s0 + k;
        if (in.pos[size_t(t + k)] != p) throw std::runtime_error("fake: pos");
        at(c.slot, p) = in.ids[size_t(t + k)];
        chunk_of_row[size_t(t + k)] = int(ci);
      }
      if (c.s0 == 0) state_[size_t(c.slot)] = kFnvBasis;
      for (int k = 0; k < c.n; ++k) {
        state_[size_t(c.slot)] = fnv_step(state_[size_t(c.slot)], in.ids[size_t(t + k)]);
        row_state[size_t(t + k)] = state_[size_t(c.slot)];
      }
      for (const RingBackup& b : in.backup)
        if (b.slot == c.slot && (b.s0 != c.s0 || b.n != c.n)) throw std::runtime_error("fake: backup is not the chunk");
      len_[size_t(c.slot)] = c.s0 + c.n;
      t += c.n;
      max_batch = std::max(max_batch, t);
    }
    logits.assign(in.logit_rows.size() * kVocab, -10.0f);
    for (size_t r = 0; r < in.logit_rows.size(); ++r) {
      const Chunk& c = in.chunks[size_t(chunk_of_row[size_t(in.logit_rows[r])])];
      const int row = in.logit_rows[r];
      if (row != c.t0 + c.n - 1 && !verify[size_t(c.slot)])
        throw std::runtime_error("fake: logit row is not a chunk end (and the chunk is not a verify chunk)");
      std::vector<int32_t> seq(size_t(c.s0 + (row - c.t0) + 1));
      for (size_t p = 0; p < seq.size(); ++p) seq[p] = at(c.slot, int(p));
      if (fnv(seq.data(), seq.size()) != row_state[size_t(row)]) throw std::runtime_error("fake: slot state is stale");
      logits[r * kVocab + size_t(next_token(seq))] = 10.0f;
    }
    for (const ImageRows& im : in.image_rows) {  // whole blocks, inside one chunk
      bool inside = false;
      for (const Chunk& c : in.chunks) inside |= im.t0 >= c.t0 && im.t0 + im.n <= c.t0 + c.n;
      if (!inside || !im.rows) throw std::runtime_error("fake: image rows across chunks");
      ++image_blocks;
    }
    for (const ImageSpan& sp : in.image_spans) {
      bool inside = false;
      for (const Chunk& c : in.chunks) inside |= c.slot == sp.slot && sp.start >= c.s0 && sp.end < c.s0 + c.n;
      if (!inside) throw std::runtime_error("fake: image span across chunks");
    }
    for (const StateOp& sv : in.save) {
      if (len_[size_t(sv.slot)] % page_ != 0) throw std::runtime_error("fake: save off a page boundary");
      store_snap(sv.snapshot, {state_[size_t(sv.slot)], len_[size_t(sv.slot)]});
      ++saves;
    }
  }
  int steps = 0, max_batch = 0, max_seen = 0, restores = 0, saves = 0, rollbacks = 0, drafts = 0, image_blocks = 0;

 private:
  int32_t& at(int slot, int p) {
    const int lp = p / page_;
    if (lp >= ppseq_) throw std::runtime_error("fake: position beyond the block table");
    const int32_t phys = tables_[size_t(slot) * ppseq_ + size_t(lp)];
    if (phys < 0 || phys >= pages_) throw std::runtime_error("fake: bad page id");
    return store_[size_t(phys) * page_ + size_t(p % page_)];
  }
  int inflight_;
  std::deque<std::pair<uint64_t, std::vector<float>>> pending_;
  uint64_t next_ = 1;
  int slots_, ppseq_, pages_, page_, max_tokens_, snapshots_, draft_block_;
  std::vector<int32_t> store_, tables_;
  std::vector<int> len_;
  // A snapshot's bytes: a magic, the state and the length, then words derived from the state and their index, so a
  // slot that was overwritten, torn, or read from the wrong place fails the check in load_snap.
  static constexpr size_t kSnapBytes = 8192;
  static constexpr uint64_t kMagic = 0x534e415046414b45ull;
  struct Snap {
    uint64_t state = 0;
    int len = 0;
  };
  static uint64_t word(uint64_t state, size_t i) {
    uint64_t x = state + 0x9e3779b97f4a7c15ull * (i + 1);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  }
  void store_snap(int id, const Snap& sn) {
    uint64_t* w = reinterpret_cast<uint64_t*>(host_ + size_t(id) * kSnapBytes);
    w[0] = kMagic;
    w[1] = sn.state;
    w[2] = uint64_t(sn.len);
    for (size_t i = 3; i < kSnapBytes / 8; ++i) w[i] = word(sn.state, i);
  }
  Snap load_snap(int id) const {
    if (id < 0 || id >= snapshots_) throw std::runtime_error("fake: bad snapshot id");
    const uint64_t* w = reinterpret_cast<const uint64_t*>(host_ + size_t(id) * kSnapBytes);
    if (w[0] != kMagic) throw std::runtime_error("fake: restore of an unsaved snapshot");
    for (size_t i = 3; i < kSnapBytes / 8; ++i)
      if (w[i] != word(w[1], i)) throw std::runtime_error("fake: restored snapshot bytes differ from what was saved");
    return {w[1], int(w[2])};
  }
  uint8_t* host_ = nullptr;
  std::vector<uint64_t> state_;
  struct Backup {
    int s0 = 0, n = 0;
  };
  std::vector<Backup> backup_;
};

struct Result {
  std::vector<int32_t> tokens;
  FinishReason reason = FinishReason::None;
  int32_t stop_token = -1;
  int deliveries = 0;
  bool finished = false;
};

std::vector<int32_t> reference(const std::vector<int32_t>& prompt, int max_tokens, int max_context,
                               FinishReason& reason) {
  std::vector<int32_t> seq = prompt, out;
  for (;;) {
    const int32_t t = next_token(seq);
    if (t == kEos) {
      reason = FinishReason::Stop;
      return out;
    }
    seq.push_back(t);
    out.push_back(t);
    if ((max_tokens > 0 && int(out.size()) >= max_tokens) || int(seq.size()) >= max_context) {
      reason = FinishReason::Length;
      return out;
    }
  }
}

// Head-of-line blocking: a long prompt, then (one step later) a short one. Returns the step at which
// each request finished (both with max_tokens = 1), driving the engine by hand.
std::pair<int, int> race(PrefillOrder order, int threshold, bool adaptive) {
  FakeModel model(4, 256, 512, 16, 64, 1, 0, 0);
  EngineConfig ec;
  ec.max_batch_tokens = 64;
  ec.prefill_order = order;
  ec.long_prefill_threshold = threshold;
  ec.long_prefill_adaptive = adaptive;
  Engine eng(model, ec, false);
  int step = 0, done_long = -1, done_short = -1;
  const auto submit = [&](int len, int* done) {
    GenRequest r;
    for (int i = 0; i < len; ++i) r.prompt.push_back(int32_t(2 + (i * 7919) % (kVocab - 2)));
    r.max_tokens = 1;
    r.sampling.temperature = 0.0f;
    eng.submit(std::move(r), [&step, done](GenOutput&& o) {
      if (o.finished) *done = step;
    });
  };
  submit(2000, &done_long);
  eng.iterate();
  ++step;
  submit(20, &done_short);
  while ((done_long < 0 || done_short < 0) && step < 10000) {
    eng.iterate();
    ++step;
  }
  return {done_long, done_short};
}

// ---- The disk tier ------------------------------------------------------------------------------------------

struct Workload {
  std::vector<std::vector<int32_t>> prompts;
  std::vector<int> max_tokens;
};

// Requests that share prefixes: `groups` bases of 6-15 pages, each request one base (whole, or its first pages) and a
// few new tokens, so a base's snapshots are needed again long after more snapshots than the RAM holds were saved.
Workload make_workload(uint64_t seed, int n, int groups, int page) {
  std::mt19937_64 rng(seed);
  std::vector<std::vector<int32_t>> base;
  for (int g = 0; g < groups; ++g) {
    base.emplace_back();
    const int len = page * (6 + int(rng() % 10));
    for (int i = 0; i < len; ++i) base.back().push_back(int32_t(2 + rng() % (kVocab - 2)));
  }
  Workload w;
  for (int i = 0; i < n; ++i) {
    std::vector<int32_t> p = base[rng() % size_t(groups)];
    if (rng() % 3 == 0) p.resize(size_t(page) * (1 + rng() % (p.size() / size_t(page))));
    const int extra = 1 + int(rng() % 30);
    for (int k = 0; k < extra; ++k) p.push_back(int32_t(2 + rng() % (kVocab - 2)));
    w.prompts.push_back(std::move(p));
    w.max_tokens.push_back(1 + int(rng() % 24));
  }
  return w;
}

struct Run {
  std::vector<Result> res;
  std::vector<int> cached;
  EngineStats st;
  int steps = 0, restores = 0, saves = 0;
  bool ok = true;
  std::string err;
};

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Drives `eng` until `done()` (sleeping while it waits for the disk); false on a timeout.
template <class F>
bool drive(Engine& eng, F done, double timeout_s = 60) {
  const double t0 = now_s();
  while (!done()) {
    if (!eng.iterate()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    if (now_s() - t0 > timeout_s) return false;
  }
  return true;
}
// The workload through an engine with `ram` snapshots and (dir, slots) of disk; `abort_every`: a recent request is
// aborted every n-th pass of the loop; `read_delay_us`: the I/O thread sleeps before each read.
// pass of the loop a recent request; `read_delay_us`: the I/O thread sleeps before each read.
Run run_workload(const Workload& w, int ram, const std::string& dir, int slots, int nspec = 0, int abort_every = 0,
                 int read_delay_us = 0) {
  Run r;
  const int page = 16;
  FakeModel model(8, 64, 2048, page, 256, 2, ram, nspec ? 5 : 0);
  EngineConfig ec;
  ec.max_batch_tokens = 256;
  ec.spec_drafts = nspec;
  ec.snapshot_dir = dir;
  ec.disk_snapshots = slots;
  if (read_delay_us > 0)
    ec.io_hook = [read_delay_us](bool write) {
      if (!write) std::this_thread::sleep_for(std::chrono::microseconds(read_delay_us));
    };
  std::mt19937_64 rng(7);
  const size_t n = w.prompts.size();
  r.res.resize(n);
  r.cached.assign(n, 0);
  std::vector<uint64_t> ids;
  try {
    Engine eng(model, ec, false);
    size_t sub = 0;
    const auto all_done = [&] {
      const auto s = eng.stats();
      return sub == n && s.running + s.waiting == 0;
    };
    const double t0 = now_s();
    while (!all_done()) {
      for (int b = int(rng() % 4); b > 0 && sub < n; --b, ++sub) {
        GenRequest q;
        q.prompt = w.prompts[sub];
        q.max_tokens = w.max_tokens[sub];
        q.sampling.temperature = 0.0f;
        Result* out = &r.res[sub];
        int* cached = &r.cached[sub];
        ids.push_back(eng.submit(std::move(q), [out, cached](GenOutput&& o) {
          out->tokens.insert(out->tokens.end(), o.tokens.begin(), o.tokens.end());
          if (o.finished) {
            out->finished = true;
            out->reason = o.reason;
            *cached = o.cached_tokens;
          }
        }));
      }
      if (abort_every > 0 && !ids.empty() && rng() % size_t(abort_every) == 0)  // one of the last few: often a loading one
        eng.abort(ids[ids.size() - 1 - rng() % std::min<size_t>(ids.size(), 6)]);
      // The fake model's steps take no time, so the disk would never keep up: give an operation in flight a moment
      // (a real step takes tens of milliseconds, a spill about a hundred).
      if (!eng.iterate() || eng.stats().snapshot_io > 0) std::this_thread::sleep_for(std::chrono::microseconds(300));
      if (now_s() - t0 > 120) throw std::runtime_error("timeout:\n" + eng.debug_state());
    }
    eng.iterate();  // drain: the last spills
    if (!drive(eng, [&] { return eng.stats().snapshot_io == 0; })) throw std::runtime_error("I/O never finished");
    r.st = eng.stats();
  } catch (const std::exception& e) {
    r.ok = false;
    r.err = e.what();
  }
  r.steps = model.steps;
  r.restores = model.restores;
  r.saves = model.saves;
  return r;
}

// A run against the reference outputs (and `base`, a run with unlimited RAM, when given); returns the failures.
int check_run(const char* name, const Workload& w, const Run& r, const Run* base, int ram, bool expect_disk, int disk_slots) {
  int bad = 0;
  if (!r.ok) {
    std::printf("  %s: exception: %s\n", name, r.err.c_str());
    return 1;
  }
  int aborted = 0;
  for (size_t i = 0; i < w.prompts.size(); ++i) {
    const Result& x = r.res[i];
    FinishReason why;
    const auto want = reference(w.prompts[i], w.max_tokens[i], 64 * 16, why);
    if (!x.finished) {
      ++bad;
      std::printf("  %s: request %zu never finished\n", name, i);
    } else if (x.reason == FinishReason::Abort) {
      ++aborted;
      if (x.tokens.size() > want.size() || !std::equal(x.tokens.begin(), x.tokens.end(), want.begin())) ++bad;
    } else if (x.tokens != want || x.reason != why) {
      if (++bad <= 3) std::printf("  %s: request %zu differs from the reference\n", name, i);
    } else if (base && base->res[i].reason != FinishReason::Abort && x.tokens != base->res[i].tokens) {
      if (++bad <= 3) std::printf("  %s: request %zu differs from the unlimited-RAM run\n", name, i);
    }
  }
  if (r.st.free_pages != r.st.total_pages) std::printf("  %s: page leak\n", name), ++bad;
  if (r.st.free_snapshots != ram) std::printf("  %s: %d of %d RAM snapshots unreferenced at the end\n", name, r.st.free_snapshots, ram), ++bad;
  if (r.st.disk_snapshots > disk_slots) std::printf("  %s: more disk entries than slots\n", name), ++bad;
  if (expect_disk) {
    if (r.st.spills == 0 || r.st.disk_hits == 0 || r.st.disk_read_bytes == 0 || r.st.disk_write_bytes == 0) {
      std::printf("  %s: no spills or no disk hits (%llu, %llu)\n", name, (unsigned long long)r.st.spills,
                  (unsigned long long)r.st.disk_hits);
      ++bad;
    }
  } else if (r.st.spills != 0 || r.st.disk_hits != 0) {
    std::printf("  %s: disk activity with the tier off\n", name), ++bad;
  }
  std::printf("%-26s ram %4d disk %3d: %d requests (%d aborted), %d steps, %d restores, %d saves, cached %llu of %llu "
              "tokens, spills %llu, disk hits %llu, on disk %d, read %.1f MB, written %.1f MB: %s\n",
              name, ram, disk_slots, int(w.prompts.size()), aborted, r.steps, r.restores, r.saves,
              (unsigned long long)r.st.cached_tokens, (unsigned long long)r.st.prompt_tokens,
              (unsigned long long)r.st.spills, (unsigned long long)r.st.disk_hits, r.st.disk_snapshots,
              double(r.st.disk_read_bytes) / 1e6, double(r.st.disk_write_bytes) / 1e6, bad ? "BAD" : "ok");
  return bad;
}

// A prompt of `len` tokens, different for each `k` from its first page on.
std::vector<int32_t> distinct_prompt(int k, int len) {
  std::vector<int32_t> p;
  for (int i = 0; i < len; ++i) p.push_back(int32_t(2 + (uint64_t(k) * 7919 + uint64_t(i) * 104729 + uint64_t(i) * uint64_t(i) * uint64_t(k)) % (kVocab - 2)));
  return p;
}

// A prefix on disk is read back while other requests keep running; a corrupt file is noticed and the request
// recomputed (the tier turns itself off).
int blocking_tests(const std::string& dir, bool corrupt) {
  int bad = 0;
  std::atomic<bool> gate{false};  // reads wait for it
  const std::string file = dir + "/snapshots.bin";
  FakeModel model(4, 64, 1024, 16, 128, 2, 3, 0);
  EngineConfig ec;
  ec.max_batch_tokens = 128;
  ec.snapshot_dir = dir;
  ec.save_wait_ms = 5;
  ec.disk_snapshots = 16;
  ec.io_hook = [&](bool write) {
    if (write) return;
    for (int i = 0; i < 5000 && !gate; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (corrupt) {
      const int fd = ::open(file.c_str(), O_RDWR);
      const uint64_t junk = ~0ull;
      for (int d = 0; fd >= 0 && d < 16; ++d) (void)!::pwrite(fd, &junk, 8, off_t(d) * 8192 + 16);
      if (fd >= 0) ::close(fd);
    }
  };
  struct Open {  // never leave the I/O thread waiting
    std::atomic<bool>& g;
    ~Open() { g = true; }
  } opener{gate};
  try {
    Engine eng(model, ec, false);
    std::vector<std::unique_ptr<Result>> res;
    std::vector<int> cached;
    cached.reserve(64);
    std::vector<std::vector<int32_t>> prompts;
    const auto submit = [&](std::vector<int32_t> p, int max_tokens) {
      GenRequest q;
      q.prompt = p;
      q.max_tokens = max_tokens;
      q.sampling.temperature = 0.0f;
      prompts.push_back(p);
      res.push_back(std::make_unique<Result>());
      cached.push_back(0);
      Result* out = res.back().get();
      int* c = &cached.back();
      eng.submit(std::move(q), [out, c](GenOutput&& o) {
        out->tokens.insert(out->tokens.end(), o.tokens.begin(), o.tokens.end());
        if (o.finished) {
          out->finished = true;
          out->reason = o.reason;
          *c = o.cached_tokens;
        }
      });
      return res.size() - 1;
    };
    // Six distinct prompts of 12 pages + 3 tokens, one at a time: three RAM snapshots keep one beyond the free pair.
    for (int k = 1; k <= 6; ++k) {
      const size_t i = submit(distinct_prompt(k, 195), 1);
      if (!drive(eng, [&] { return res[i]->finished; })) throw std::runtime_error("phase 1 timed out");
      if (!drive(eng, [&] { return eng.stats().snapshot_io == 0; })) throw std::runtime_error("spills never finished");
    }
    if (!drive(eng, [&] { return eng.stats().snapshot_io == 0; })) throw std::runtime_error("spills never finished");
    const auto s1 = eng.stats();
    if (s1.spills < 3) {
      std::printf("  blocking: only %llu spills\n", (unsigned long long)s1.spills);
      ++bad;
    }
    // The first prompt again, extended: its snapshot is on disk only, the read waits at the gate. Meanwhile a fresh
    // prompt must run to its end.
    std::vector<int32_t> again = distinct_prompt(1, 195);
    for (int k = 0; k < 5; ++k) again.push_back(int32_t(2 + 31 * k));
    const size_t a = submit(again, 3);
    const size_t q = submit(distinct_prompt(9, 300), 20);
    int iters = 0;
    while (!res[q]->finished && iters < 4000) {
      eng.iterate();
      ++iters;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    const auto s2 = eng.stats();
    if (!res[q]->finished || res[a]->finished || s2.disk_hits != s1.disk_hits + 1 || s2.running != 1) {
      std::printf("  blocking: fresh request finished %d after %d iterations, loading one finished %d, disk hits %llu, "
                  "running %d\n", int(res[q]->finished), iters, int(res[a]->finished), (unsigned long long)s2.disk_hits, s2.running);
      ++bad;
    }
    // Not even a step is spent on the loading sequence while its read is out.
    const int steps_before = model.steps;
    for (int i = 0; i < 50; ++i) eng.iterate();
    if (model.steps != steps_before || res[a]->finished) {
      std::printf("  blocking: the loading sequence was scheduled\n");
      ++bad;
    }
    gate = true;
    if (!drive(eng, [&] { return res[a]->finished; })) throw std::runtime_error("the loading request never finished");
    const auto s3 = eng.stats();
    for (const size_t i : {a, q}) {
      FinishReason why;
      const auto want = reference(prompts[i], i == a ? 3 : 20, 64 * 16, why);
      if (res[i]->tokens != want || res[i]->reason != why) {
        std::printf("  blocking: request %zu differs from the reference\n", i);
        ++bad;
      }
    }
    if (!corrupt && (cached[a] != 192 || s3.disk_read_bytes == 0)) {
      std::printf("  blocking: %d cached tokens, %llu bytes read\n", cached[a], (unsigned long long)s3.disk_read_bytes);
      ++bad;
    }
    if (corrupt && (cached[a] != 0 || s3.disk_read_bytes != 0)) {
      std::printf("  corrupt: %d cached tokens counted, %llu bytes read\n", cached[a], (unsigned long long)s3.disk_read_bytes);
      ++bad;
    }
    // Afterwards: with the tier off (corrupt) nothing more is spilled; with it on, the engine still works.
    const auto sp = eng.stats().spills;
    for (int k = 10; k < 16; ++k) {
      const size_t i = submit(distinct_prompt(k, 195), 2);
      if (!drive(eng, [&] { return res[i]->finished; })) throw std::runtime_error("phase 3 timed out");
      FinishReason why;
      if (res[i]->tokens != reference(prompts[i], 2, 64 * 16, why)) ++bad;
    }
    drive(eng, [&] { return eng.stats().snapshot_io == 0; });
    if (corrupt ? eng.stats().spills != sp : eng.stats().spills == sp) {
      std::printf("  %s: spills %llu -> %llu\n", corrupt ? "corrupt" : "blocking", (unsigned long long)sp,
                  (unsigned long long)eng.stats().spills);
      ++bad;
    }
    std::printf("%-26s fresh request done in %d iterations while a read was out; spills %llu, disk hits %llu, read %.1f MB: %s\n",
                corrupt ? "disk corrupted" : "loading does not block", iters, (unsigned long long)eng.stats().spills,
                (unsigned long long)eng.stats().disk_hits, double(eng.stats().disk_read_bytes) / 1e6, bad ? "BAD" : "ok");
  } catch (const std::exception& e) {
    std::printf("  blocking: exception: %s\n", e.what());
    ++bad;
  }
  return bad;
}

// A request aborted while its snapshot is being read: the read finishes into its RAM slot, which is then cached (not
// freed under the read, not leaked), and the next request with that prefix is served from it.
int abort_loading_test(const std::string& dir) {
  int bad = 0;
  std::atomic<bool> gate{false};
  FakeModel model(4, 64, 1024, 16, 128, 2, 3, 0);
  EngineConfig ec;
  ec.max_batch_tokens = 128;
  ec.snapshot_dir = dir;
  ec.disk_snapshots = 16;
  ec.io_hook = [&](bool write) {
    for (int i = 0; i < 5000 && !write && !gate; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  };
  struct Open {
    std::atomic<bool>& g;
    ~Open() { g = true; }
  } opener{gate};
  try {
    Engine eng(model, ec, false);
    Result r[3];
    int cached[3] = {0, 0, 0};
    uint64_t id[3] = {0, 0, 0};
    const auto submit = [&](int i, std::vector<int32_t> p) {
      GenRequest q;
      q.prompt = std::move(p);
      q.max_tokens = 3;
      q.sampling.temperature = 0.0f;
      id[i] = eng.submit(std::move(q), [&r, &cached, i](GenOutput&& o) {
        r[i].tokens.insert(r[i].tokens.end(), o.tokens.begin(), o.tokens.end());
        if (o.finished) {
          r[i].finished = true;
          r[i].reason = o.reason;
          cached[i] = o.cached_tokens;
        }
      });
    };
    gate = true;
    for (int k = 1; k <= 6; ++k) {
      GenRequest q;
      q.prompt = distinct_prompt(k, 195);
      q.max_tokens = 1;
      q.sampling.temperature = 0.0f;
      bool fin = false;
      eng.submit(std::move(q), [&fin](GenOutput&& o) { fin |= o.finished; });
      if (!drive(eng, [&] { return fin; }) || !drive(eng, [&] { return eng.stats().snapshot_io == 0; }))
        throw std::runtime_error("setup timed out");
    }
    gate = false;
    std::vector<int32_t> again = distinct_prompt(1, 195);
    again.push_back(5);
    submit(0, again);
    for (int i = 0; i < 20; ++i) eng.iterate();
    const auto s1 = eng.stats();
    eng.abort(id[0]);
    for (int i = 0; i < 5; ++i) eng.iterate();
    if (s1.disk_hits == 0 || !r[0].finished || r[0].reason != FinishReason::Abort || eng.stats().running != 0 ||
        eng.stats().snapshot_io < 1) {  // (the read, and a spill queued behind it)
      std::printf("  abort while loading: hits %llu, finished %d, running %d, I/Os %d\n", (unsigned long long)s1.disk_hits,
                  int(r[0].finished), eng.stats().running, eng.stats().snapshot_io);
      ++bad;
    }
    gate = true;
    if (!drive(eng, [&] { return eng.stats().snapshot_io == 0; })) throw std::runtime_error("the read never finished");
    submit(1, again);
    if (!drive(eng, [&] { return r[1].finished; })) throw std::runtime_error("the second request never finished");
    const auto s2 = eng.stats();
    FinishReason why;
    if (r[1].tokens != reference(again, 3, 64 * 16, why) || cached[1] != 192 || s2.disk_hits != s1.disk_hits ||
        s2.free_snapshots != 3 || s2.disk_read_bytes == 0) {
      std::printf("  abort while loading: second request cached %d tokens, disk hits %llu -> %llu, %d RAM snapshots unreferenced\n",
                  cached[1], (unsigned long long)s1.disk_hits, (unsigned long long)s2.disk_hits, s2.free_snapshots);
      ++bad;
    }
    std::printf("%-26s the abandoned read's RAM slot was cached and served the next request: %s\n", "abort while loading",
                bad ? "BAD" : "ok");
  } catch (const std::exception& e) {
    std::printf("  abort while loading: exception: %s\n", e.what());
    ++bad;
  }
  return bad;
}

// The GLM pattern: 256-token pages, 3072-token prefill steps, 2 RAM snapshots, a prompt of 32791 tokens with decode
// after it, then followers that share its first 32768 tokens. Every follower must resume at 32768: the snapshot at the
// prompt's last boundary may not be lost to the saves (every 16 pages, rolling decode ones) made while spills are out.
int glm_pattern_test(const std::string& dir, int write_ms, int step_ms) {
  int bad = 0;
  FakeModel model(4, 160, 1200, 256, 3072, 2, 2, 0);
  EngineConfig ec;
  ec.max_batch_tokens = 3072;
  ec.snapshot_dir = dir;
  ec.disk_snapshots = 64;
  ec.io_hook = [write_ms](bool write) {
    if (write) std::this_thread::sleep_for(std::chrono::milliseconds(write_ms));
  };
  try {
    Engine eng(model, ec, false);
    std::vector<int32_t> base = distinct_prompt(3, 32791);
    Result r[5];
    int cached[5] = {};
    std::vector<std::vector<int32_t>> prompts;
    const auto submit = [&](int i, std::vector<int32_t> p, int max_tokens) {
      GenRequest q;
      prompts.push_back(p);
      q.prompt = std::move(p);
      q.max_tokens = max_tokens;
      q.sampling.temperature = 0.0f;
      eng.submit(std::move(q), [&r, &cached, i](GenOutput&& o) {
        r[i].tokens.insert(r[i].tokens.end(), o.tokens.begin(), o.tokens.end());
        if (o.finished) {
          r[i].finished = true;
          r[i].reason = o.reason;
          cached[i] = o.cached_tokens;
        }
      });
    };
    const auto run_until = [&](auto done) {
      const double t0 = now_s();
      while (!done()) {
        if (!eng.iterate()) std::this_thread::sleep_for(std::chrono::microseconds(200));
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
        if (now_s() - t0 > 120) throw std::runtime_error("timeout");
      }
    };
    submit(0, base, 700);  // decode crosses pages: rolling saves
    run_until([&] { return r[0].finished; });
    for (int i = 1; i <= 4; ++i) {
      std::vector<int32_t> p(base.begin(), base.begin() + 32768);
      for (int k = 0; k < 10; ++k) p.push_back(int32_t(2 + (i * 37 + k * 11) % 400));
      submit(i, p, 2);
      run_until([&] { return r[i].finished; });
    }
    for (int i = 1; i <= 4; ++i) {
      FinishReason why;
      if (cached[i] != 32768 || r[i].tokens != reference(prompts[size_t(i)], 2, 160 * 256, why)) {
        std::printf("  glm pattern: follower %d resumed at %d\n", i, cached[i]);
        ++bad;
      }
    }
    const auto st = eng.stats();
    std::printf("%-26s write %d ms, step %d ms: followers resumed at %d %d %d %d, spills %llu, disk hits %llu: %s\n",
                "GLM save pattern", write_ms, step_ms, cached[1], cached[2], cached[3], cached[4],
                (unsigned long long)st.spills, (unsigned long long)st.disk_hits, bad ? "BAD" : "ok");
  } catch (const std::exception& e) {
    std::printf("  glm pattern: exception: %s\n", e.what());
    ++bad;
  }
  return bad;
}

int disk_tests(uint64_t seed, int n_req, const std::string& base_dir) {
  char tmpl[512];
  std::snprintf(tmpl, sizeof tmpl, "%s/snap-disk-test-XXXXXX", base_dir.c_str());
  if (!mkdtemp(tmpl)) {
    std::printf("mkdtemp(%s): %s\n", tmpl, std::strerror(errno));
    return 1;
  }
  const std::string dir = tmpl;
  int bad = 0;
  const Workload w = make_workload(seed, std::max(n_req, 120), 12, 16);
  const int plenty = 4000;  // never short of RAM snapshots
  const Run unlimited = run_workload(w, plenty, "", 0);
  bad += check_run("unlimited RAM", w, unlimited, nullptr, plenty, false, 0);
  // The tier set up but not used must be today's behaviour bit for bit: same steps, same cache hits.
  const Run off_dir = run_workload(w, plenty, dir, 0);
  const Run idle_disk = run_workload(w, plenty, dir, 64);
  bad += check_run("dir without slots", w, off_dir, &unlimited, plenty, false, 0);
  bad += check_run("disk, RAM never short", w, idle_disk, &unlimited, plenty, false, 64);
  for (const Run* r : {&off_dir, &idle_disk})
    if (r->steps != unlimited.steps || r->restores != unlimited.restores || r->saves != unlimited.saves ||
        r->st.cached_tokens != unlimited.st.cached_tokens) {
      std::printf("  not the same run as without the tier\n");
      ++bad;
    }
  const Run ram_only = run_workload(w, 3, "", 0);
  bad += check_run("3 RAM, no disk", w, ram_only, &unlimited, 3, false, 0);
  const Run few = run_workload(w, 3, dir, 200);
  bad += check_run("3 RAM, 200 disk", w, few, &unlimited, 3, true, 200);
  if (few.st.cached_tokens <= ram_only.st.cached_tokens) {
    std::printf("  the disk tier added no cache hits (%llu vs %llu cached tokens)\n", (unsigned long long)few.st.cached_tokens,
                (unsigned long long)ram_only.st.cached_tokens);
    ++bad;
  }
  const Run tiny = run_workload(w, 3, dir, 10);  // disk slots evicted too
  bad += check_run("3 RAM, 10 disk", w, tiny, &unlimited, 3, true, 10);
  const Run two = run_workload(w, 2, dir, 64, 0, 0, 200);
  bad += check_run("2 RAM, 64 disk", w, two, &unlimited, 2, true, 64);
  const Run spec = run_workload(w, 3, dir, 200, 4);
  bad += check_run("3 RAM, 200 disk, spec 4", w, spec, nullptr, 3, true, 200);
  const Run aborts = run_workload(w, 3, dir, 64, 0, 3, 300);  // aborts while reads are out
  bad += check_run("3 RAM, 64 disk, aborts", w, aborts, nullptr, 3, true, 64);
  bad += blocking_tests(dir, false);
  bad += blocking_tests(dir, true);
  bad += abort_loading_test(dir);
  bad += glm_pattern_test(dir, 20, 5);
  bad += glm_pattern_test(dir, 5, 5);
  ::rmdir(dir.c_str());
  return bad;
}

// The sequence-seconds, first-token and /metrics counters llmtop reads. Each wait() takes 3 ms, so a step is
// ~3 ms whatever the batch: one sequence decoding N tokens over T seconds of decode steps gives N/T, and so
// does each of several sequences decoding together (the per-stream speed), while the aggregate is nseq times it.
int metrics_test() {
  int bad = 0;
  for (const int depth : {1, 2})
    for (const int nseq : {1, 4}) {
      FakeModel model(8, 64, 512, 16, 256, depth, 0, 0);
      model.wait_delay = std::chrono::microseconds(3000);
      EngineConfig ec;
      ec.max_batch_tokens = 256;
      Engine eng(model, ec, false);
      std::vector<Result> res(size_t(nseq) + 1);
      int with_tokens = 0;
      for (int i = 0; i < nseq; ++i) {
        GenRequest q;
        for (int k = 0; k < 8; ++k) q.prompt.push_back(int32_t(2 + (i * 13 + k * 7) % 400));
        q.max_tokens = 40;
        q.ignore_eos = true;
        q.sampling.temperature = 0.0f;
        Result* out = &res[size_t(i)];
        eng.submit(std::move(q), [out](GenOutput&& o) {
          out->tokens.insert(out->tokens.end(), o.tokens.begin(), o.tokens.end());
          if (o.finished) out->finished = true;
        });
      }
      // A request aborted before the engine ever ran it produces no token: not counted.
      {
        GenRequest q;
        q.prompt = {2, 3, 4};
        q.max_tokens = 5;
        const uint64_t id = eng.submit(std::move(q), [](GenOutput&&) {});
        eng.abort(id);
      }
      const bool ok = drive(eng, [&] { return eng.stats().running + eng.stats().waiting == 0 && eng.stats().steps > 0; });
      for (int i = 0; i < nseq; ++i) with_tokens += !res[size_t(i)].tokens.empty();
      const auto s = eng.stats();
      const double dec_rate = s.decode_seq_seconds > 0 ? double(s.decode_tokens) / s.decode_seq_seconds : 0;
      const double pre_rate = s.prefill_seq_seconds > 0 ? double(s.prefill_tokens) / s.prefill_seq_seconds : 0;
      std::printf("%-26s depth %d, %d seq: decode %.0f tok/s per stream (%llu tokens / %.3f s), prefill %.0f tok/s, "
                  "first tokens %llu (%.3f s), busy %.3f s\n",
                  "sequence-seconds", depth, nseq, dec_rate, (unsigned long long)s.decode_tokens, s.decode_seq_seconds,
                  pre_rate, (unsigned long long)s.first_tokens, s.first_token_seconds, s.busy_seconds);
      bool good = ok && s.prefill_seq_seconds > 0 && s.decode_seq_seconds > 0 && s.first_tokens == uint64_t(with_tokens) &&
                  with_tokens == nseq && s.first_token_seconds > 0 && s.slots == 8;
      // ~333 tok/s per stream with 3 ms steps; sleeps overshoot, never undershoot.
      good = good && dec_rate > 150 && dec_rate < 340;
      // A sequence is in one phase at a time and the engine thread is one: seconds cannot exceed busy x sequences.
      good = good && s.prefill_seq_seconds + s.decode_seq_seconds <= s.busy_seconds * nseq * 1.001;
      // Mean time to first token: at least the prefill step, far under the whole run.
      good = good && s.first_token_seconds / double(s.first_tokens) >= 0.003 &&
             s.first_token_seconds / double(s.first_tokens) < s.busy_seconds;
      if (!good) {
        std::printf("  sequence-seconds / first-token counters inconsistent\n");
        ++bad;
      }
    }
  // The text.
  {
    EngineStats s;
    s.slots = 4;
    s.total_pages = 200;
    s.free_pages = 50;
    s.prefill_seq_seconds = 1.5;
    s.decode_seq_seconds = 2.25;
    s.first_token_seconds = 0.75;
    s.first_tokens = 3;
    const std::string t = unholy::server::format_metrics(s, 8192, "a\"b\\c\nd", 7, 1);
    const char* want[] = {"unholy_info{model=\"a\\\"b\\\\c\\nd\"} 1\n", "unholy_slots 4\n", "unholy_context_tokens 8192\n",
                          "unholy_kv_usage_ratio 0.750000\n", "unholy_prefill_sequence_seconds_total 1.500\n",
                          "unholy_decode_sequence_seconds_total 2.250\n", "unholy_first_token_seconds_total 0.750\n",
                          "unholy_first_tokens_total 3\n", "unholy_http_requests_total 7\n", "unholy_http_errors_total 1\n",
                          "unholy_steps_total 0\n", "unholy_busy_seconds_total 0.000\n", "unholy_free_pages 50\n",
                          "unholy_spec_accepted_total 0\n", "unholy_cached_snapshots 0\n"};
    for (const char* w : want)
      if (t.find(w) == std::string::npos) {
        std::printf("  /metrics text lacks: %s", w);
        ++bad;
      }
    EngineStats z;
    if (unholy::server::format_metrics(z, 1, "m", 0, 0).find("unholy_kv_usage_ratio 0.000000\n") == std::string::npos) {
      std::printf("  kv usage with an empty pool is not 0\n");
      ++bad;
    }
    std::printf("%-26s %s\n", "/metrics text", bad ? "BAD" : "ok");
  }
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  int n_req = 300;
  uint64_t seed = 1;
  std::string disk_dir = "/tmp";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--requests") && i + 1 < argc) n_req = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--disk-dir") && i + 1 < argc) disk_dir = argv[++i];
  }
  int failures = 0;
  // Configurations: (slots, pages per seq, pool pages, page tokens, step tokens, batch budget).
  const int configs[][6] = {{8, 64, 512, 16, 256, 256}, {4, 32, 40, 16, 128, 100}, {16, 16, 60, 8, 64, 64},
                            {2, 128, 30, 32, 512, 512}, {32, 20, 200, 16, 96, 96}};
  // Scheduling modes: (prefill order, long-prefill threshold, adaptive).
  const struct Mode { PrefillOrder order; int thr; bool adaptive; const char* name; } modes[] = {
      {PrefillOrder::Shortest, 0, false, "shortest"},
      {PrefillOrder::Fcfs, 0, false, "fcfs"},
      {PrefillOrder::Fcfs, 24, true, "fcfs+thr24a"}};
  for (const Mode& mode : modes)
  for (const int nspec : {0, 4})
  for (const int snaps : {0, 6, 64})
  for (const int depth : {1, 2})
  for (const auto& cfgv : configs) {
    std::mt19937_64 rng(seed);
    FakeModel model(cfgv[0], cfgv[1], cfgv[2], cfgv[3], cfgv[4], depth, snaps, nspec ? 5 : 0);
    EngineConfig ec;
    ec.spec_drafts = nspec;
    ec.max_batch_tokens = cfgv[5];
    ec.prefill_order = mode.order;
    ec.long_prefill_threshold = mode.thr;
    ec.long_prefill_adaptive = mode.adaptive;
    std::unordered_map<uint64_t, RequestLog> logs;
    int log_dups = 0;
    ec.on_finish = [&](const RequestLog& l) { log_dups += !logs.emplace(l.id, l).second; };
    Engine eng(model, ec, false);
    int submitted = 0;
    const int max_ctx = eng.max_context();
    std::vector<std::unique_ptr<Result>> res;
    std::vector<std::pair<std::vector<int32_t>, int>> spec;
    std::vector<std::vector<GenImage>> spec_images;
    const int max_image = std::min(40, cfgv[5] - cfgv[0] - 1);  // fits a step even with every slot decoding
    std::vector<uint64_t> rid;
    auto submit = [&]() {
      GenRequest r;
      const int cap = std::min(max_ctx - 1, 700);
      if (snaps > 0 && !spec.empty() && rng() % 2 == 0) {
        // Extend an earlier request: its prompt, or its prompt and reply (a following chat turn), whole or
        // cut somewhere past the middle, plus new tokens.
        const size_t bi = spec.size() - 1 - rng() % std::min<size_t>(spec.size(), 8);  // a recent one
        const auto& base = spec[bi];
        std::vector<int32_t> prefix = base.first;
        if (rng() % 2 == 0) {
          FinishReason why;
          const auto reply = reference(base.first, base.second, max_ctx, why);
          prefix.insert(prefix.end(), reply.begin(), reply.end());
        }
        prefix.resize(std::min(prefix.size(), size_t(cap) - 1));
        if (rng() % 2 == 0) prefix.resize(prefix.size() / 2 + rng() % (prefix.size() / 2 + 1));
        for (const GenImage& im : spec_images[bi]) {  // the base's images that the prefix keeps whole
          if (int(prefix.size()) > im.start && int(prefix.size()) < im.start + im.len) prefix.resize(size_t(im.start));
          if (im.start + im.len <= int(prefix.size())) r.images.push_back(im);
        }
        r.prompt = prefix;
        const int extra = 1 + int(rng() % 40);
        for (int i = 0; i < extra && int(r.prompt.size()) < cap; ++i)
          r.prompt.push_back(int32_t(2 + rng() % (kVocab - 2)));
        if (r.prompt.empty()) r.prompt.push_back(2);
      } else {
        const int len = 1 + int(rng() % uint64_t(cap));
        for (int i = 0; i < len; ++i) r.prompt.push_back(int32_t(2 + rng() % (kVocab - 2)));
        // A third of the fresh prompts carry an image block (ids past the vocabulary, as the real ones).
        if (max_image >= 8 && len > max_image + 2 && rng() % 3 == 0) {
          GenImage im;
          im.len = 8 + int(rng() % uint64_t(max_image - 7));
          im.start = int(rng() % uint64_t(len - im.len));
          im.span_start = im.start + 1;
          im.span_end = im.start + im.len - 1;
          im.hash = rng();
          im.rows = std::make_shared<std::vector<uint16_t>>(size_t(im.len), 0);
          for (int i = 0; i < im.len; ++i) r.prompt[size_t(im.start + i)] = int32_t(kVocab + i % 5);
          r.images.push_back(im);
        }
      }
      r.max_tokens = rng() % 4 == 0 ? 0 : 1 + int(rng() % 200);
      r.sampling.temperature = 0.0f;
      spec.emplace_back(r.prompt, r.max_tokens);
      spec_images.push_back(r.images);
      res.push_back(std::make_unique<Result>());
      Result* out = res.back().get();
      rid.push_back(eng.submit(std::move(r), [out](GenOutput&& o) {
        if (out->finished) throw std::runtime_error("delivery after finish");
        out->tokens.insert(out->tokens.end(), o.tokens.begin(), o.tokens.end());
        ++out->deliveries;
        if (o.finished) {
          out->finished = true;
          out->reason = o.reason;
          out->stop_token = o.stop_token;
        }
      }));
      ++submitted;
    };
    try {
      int idle = 0;
      while (submitted < n_req || eng.stats().running + eng.stats().waiting > 0 || submitted == 0) {
        if (++idle > 200000) {
          const auto s = eng.stats();
          throw std::runtime_error("stuck: " + std::to_string(s.running) + " running, " + std::to_string(s.waiting) +
                                   " waiting, " + std::to_string(s.free_pages) + " free pages, steps " +
                                   std::to_string(s.steps) + "\n" + eng.debug_state());
        }
        const int burst = submitted < n_req ? int(rng() % 4) : 0;
        for (int b = 0; b < burst && submitted < n_req; ++b) submit();
        if (rng() % 50 == 0 && !rid.empty()) {
          eng.abort(rid[rng() % rid.size()]);
        }
        if (eng.iterate()) idle = 0;
        if (submitted >= n_req && eng.stats().running + eng.stats().waiting == 0) {
          eng.iterate();  // drain aborts and intake
          if (eng.stats().running + eng.stats().waiting == 0) break;
        }
      }
    } catch (const std::exception& e) {
      std::printf("config %d/%d: exception: %s\n", cfgv[0], cfgv[2], e.what());
      ++failures;
      continue;
    }
    int ok = 0, bad = 0, aborts_seen = 0, errors = 0, log_bad = log_dups;
    uint64_t log_drafted = 0, log_accepted = 0;
    for (size_t i = 0; i < res.size(); ++i) {
      const Result& r = *res[i];
      if (!r.finished) {
        ++bad;
        std::printf("  request %zu never finished\n", i);
        continue;
      }
      // The request log: one entry per request, agreeing with what was delivered.
      if (const auto it = logs.find(rid[i]); it == logs.end()) {
        ++log_bad;
      } else {
        const RequestLog& l = it->second;
        log_drafted += uint64_t(l.spec_drafted);
        log_accepted += uint64_t(l.spec_accepted);
        const int delivered = int(r.tokens.size()) + (r.reason == FinishReason::Stop);
        const bool timing = l.queue_s >= 0 && (l.ttft_s < 0 || (l.ttft_s >= l.queue_s && l.prefill_s >= 0 &&
                                                                  l.decode_s >= 0));
        if (l.reason != r.reason || (r.reason != FinishReason::Abort && l.completion_tokens != delivered) ||
            l.prompt_tokens != int(spec[i].first.size()) || !timing || (l.ttft_s < 0) != (l.completion_tokens == 0)) {
          if (++log_bad <= 3)
            std::printf("  request %zu: log says %s, %d completion tokens, queue %.3f ttft %.3f; delivered %d (%s)\n", i,
                        finish_reason_name(l.reason), l.completion_tokens, l.queue_s, l.ttft_s, delivered,
                        finish_reason_name(r.reason));
        }
      }
      FinishReason want_reason;
      const auto want = reference(spec[i].first, spec[i].second, max_ctx, want_reason);
      if (r.reason == FinishReason::Abort) {
        ++aborts_seen;
        if (r.tokens.size() > want.size() || !std::equal(r.tokens.begin(), r.tokens.end(), want.begin())) {
          ++bad;
          std::printf("  request %zu (aborted) diverged\n", i);
        }
        continue;
      }
      if (r.reason == FinishReason::Error) {
        ++errors;
        continue;
      }
      if (r.tokens != want || r.reason != want_reason) {
        ++bad;
        if (bad <= 5)
          std::printf("  request %zu: got %zu tokens (%s), want %zu (%s)\n", i, r.tokens.size(),
                      finish_reason_name(r.reason), want.size(), finish_reason_name(want_reason));
      } else {
        ++ok;
      }
    }
    const auto st = eng.stats();
    std::printf("%s spec %d snaps %d depth %d (max seen %d) slots=%d pool=%d page=%d budget=%d: %d ok, %d bad, %d aborted, "
                "%d errors; %d steps, max batch %d, %llu preemptions, free pages %d/%d; cached %llu of %llu prompt "
                "tokens, %d restores, %d saves; drafts %d, verified %llu of %llu drafted, %d rollbacks; %d image blocks\n",
                mode.name, nspec, snaps, depth, model.max_seen, cfgv[0], cfgv[2], cfgv[3], cfgv[5], ok, bad, aborts_seen, errors,
                model.steps, model.max_batch, (unsigned long long)st.preemptions, st.free_pages, st.total_pages,
                (unsigned long long)st.cached_tokens, (unsigned long long)st.prompt_tokens, model.restores,
                model.saves, model.drafts, (unsigned long long)st.spec_accepted, (unsigned long long)st.spec_drafted,
                model.rollbacks, model.image_blocks);
    if (log_bad > 0 || logs.size() != res.size() || log_drafted != st.spec_drafted ||
        log_accepted != st.spec_accepted || st.spec_pos_drafted[0] != st.spec_steps) {
      std::printf("  request log: %d bad, %zu logged of %zu, drafts %llu/%llu vs %llu/%llu, position 0 drafted %llu of %llu "
                  "verify steps\n",
                  log_bad, logs.size(), res.size(), (unsigned long long)log_accepted, (unsigned long long)log_drafted,
                  (unsigned long long)st.spec_accepted, (unsigned long long)st.spec_drafted,
                  (unsigned long long)st.spec_pos_drafted[0], (unsigned long long)st.spec_steps);
      ++failures;
    }
    if (nspec > 0 && (st.spec_accepted == 0 || model.rollbacks == 0)) {
      std::printf("  speculative decoding never accepted or never rolled back\n");
      ++failures;
    }
    if (snaps > 0 && st.cached_tokens == 0) {
      std::printf("  the prefix cache was never used\n");
      ++failures;
    }
    if (st.free_pages != st.total_pages) {
      std::printf("  page leak\n");
      ++failures;
    }
    failures += bad;
  }
  failures += disk_tests(seed, n_req, disk_dir);
  failures += metrics_test();
  // A 2000-token prompt (64-token steps: ~32 steps of prefill), then a 20-token one a step later.
  // Shortest-first and the threshold let the short one finish within a few steps; FCFS queues it behind
  // the whole long prefill (vLLM's default).
  {
    const auto [l1, s1] = race(PrefillOrder::Shortest, 0, false);
    const auto [l2, s2] = race(PrefillOrder::Fcfs, 0, false);
    const auto [l3, s3] = race(PrefillOrder::Fcfs, 16, false);
    std::printf("race (long, short finish step): shortest %d, %d; fcfs %d, %d; fcfs+thr16 %d, %d\n", l1, s1, l2, s2,
                l3, s3);
    if (s1 < 0 || s1 > 4 || l1 < 0 || s2 < l2 || s3 < 0 || s3 > 4) {
      std::printf("  head-of-line check failed\n");
      ++failures;
    }
    if (l1 > l2 + 2) {
      std::printf("  shortest-first delayed the long prompt by more than a step or two\n");
      ++failures;
    }
  }
  std::printf("%s\n", failures ? "FAILED" : "all ok");
  return failures ? 1 : 0;
}
