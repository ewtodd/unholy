// unholy-sched-test: the engine's scheduler against a fake model, on the CPU.
//
// The fake model keeps a paged token store addressed through the block tables the engine hands it,
// checks every chunk (positions continue the slot's sequence, rows are contiguous), and produces
// one-hot "logits" from a hash of the whole sequence read back through the block table. Greedy outputs
// must then equal a standalone simulation of each request, whatever the batching, chunking,
// preemption and slot/page reuse were — any aliasing of pages between live sequences or a lost
// chunk changes the hash.
//
//   unholy-sched-test [--requests N] [--seed S]
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>

#include "engine/engine.hpp"

using namespace unholy::engine;

namespace {

constexpr int kVocab = 512, kEos = 1;

uint64_t fnv(const int32_t* t, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ uint64_t(uint32_t(t[i]))) * 1099511628211ull;
  return h;
}
int32_t next_token(const std::vector<int32_t>& seq) {
  const uint64_t h = fnv(seq.data(), seq.size());
  if (h % 53 == 0) return kEos;
  return int32_t(2 + (h >> 8) % (kVocab - 2));
}

class FakeModel final : public StepModel {
 public:
  FakeModel(int slots, int pages_per_seq, int pages, int page_tokens, int max_tokens, int inflight)
      : inflight_(inflight), slots_(slots), ppseq_(pages_per_seq), pages_(pages), page_(page_tokens), max_tokens_(max_tokens),
        store_(size_t(pages) * page_tokens, -1), len_(size_t(slots), 0) {}
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
      len_[size_t(c.slot)] = c.s0 + c.n;
      t += c.n;
      max_batch = std::max(max_batch, t);
    }
    logits.assign(in.logit_rows.size() * kVocab, -10.0f);
    for (size_t r = 0; r < in.logit_rows.size(); ++r) {
      const Chunk& c = in.chunks[size_t(chunk_of_row[size_t(in.logit_rows[r])])];
      if (in.logit_rows[r] != c.t0 + c.n - 1) throw std::runtime_error("fake: logit row is not a chunk end");
      std::vector<int32_t> seq(size_t(c.s0 + c.n));
      for (int p = 0; p < c.s0 + c.n; ++p) seq[size_t(p)] = at(c.slot, p);
      logits[r * kVocab + size_t(next_token(seq))] = 10.0f;
    }
  }
  int steps = 0, max_batch = 0, max_seen = 0;

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
  int slots_, ppseq_, pages_, page_, max_tokens_;
  std::vector<int32_t> store_, tables_;
  std::vector<int> len_;
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

}  // namespace

int main(int argc, char** argv) {
  int n_req = 300;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--requests") && i + 1 < argc) n_req = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
  }
  int failures = 0;
  // Configurations: (slots, pages per seq, pool pages, page tokens, step tokens, batch budget).
  const int configs[][6] = {{8, 64, 512, 16, 256, 256}, {4, 32, 40, 16, 128, 100}, {16, 16, 60, 8, 64, 64},
                            {2, 128, 30, 32, 512, 512}, {32, 20, 200, 16, 96, 96}};
  for (const int depth : {1, 2})
  for (const auto& cfgv : configs) {
    std::mt19937_64 rng(seed);
    FakeModel model(cfgv[0], cfgv[1], cfgv[2], cfgv[3], cfgv[4], depth);
    EngineConfig ec;
    ec.max_batch_tokens = cfgv[5];
    Engine eng(model, ec, false);
    int submitted = 0;
    const int max_ctx = eng.max_context();
    std::vector<std::unique_ptr<Result>> res;
    std::vector<std::pair<std::vector<int32_t>, int>> spec;
    std::vector<uint64_t> rid;
    auto submit = [&]() {
      GenRequest r;
      const int len = 1 + int(rng() % uint64_t(std::min(max_ctx - 1, 700)));
      for (int i = 0; i < len; ++i) r.prompt.push_back(int32_t(2 + rng() % (kVocab - 2)));
      r.max_tokens = rng() % 4 == 0 ? 0 : 1 + int(rng() % 200);
      r.sampling.temperature = 0.0f;
      spec.emplace_back(r.prompt, r.max_tokens);
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
      while (submitted < n_req || eng.stats().running + eng.stats().waiting > 0 || submitted == 0) {
        const int burst = submitted < n_req ? int(rng() % 4) : 0;
        for (int b = 0; b < burst && submitted < n_req; ++b) submit();
        if (rng() % 50 == 0 && !rid.empty()) {
          eng.abort(rid[rng() % rid.size()]);
        }
        eng.iterate();
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
    int ok = 0, bad = 0, aborts_seen = 0, errors = 0;
    for (size_t i = 0; i < res.size(); ++i) {
      const Result& r = *res[i];
      if (!r.finished) {
        ++bad;
        std::printf("  request %zu never finished\n", i);
        continue;
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
    std::printf("depth %d (max seen %d) slots=%d pool=%d page=%d budget=%d: %d ok, %d bad, %d aborted, %d errors; "
                "%d steps, max batch %d, %llu preemptions, free pages %d/%d\n",
                depth, model.max_seen, cfgv[0], cfgv[2], cfgv[3], cfgv[5], ok, bad, aborts_seen, errors, model.steps,
                model.max_batch,
                (unsigned long long)st.preemptions, st.free_pages, st.total_pages);
    if (st.free_pages != st.total_pages) {
      std::printf("  page leak\n");
      ++failures;
    }
    failures += bad;
  }
  std::printf("%s\n", failures ? "FAILED" : "all ok");
  return failures ? 1 : 0;
}
