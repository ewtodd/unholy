// unholy: the serving binary.
//
//   unholy plan  ...the same options: print the placement for the geometry without loading anything
//   unholy serve --model DIR [--exl3 DIR] [--profile FILE] [--host H] [--port P]
//                [--served-model-name NAME] [--max-num-seqs N] [--max-model-len N] [--kv-tokens N]
//                [--max-batch-tokens N] [--api-key KEY] [--http-threads N] [--max-tokens-default N]
//
// The KV cache is one pool of --kv-tokens positions (pages of 256) shared by all requests, as in vLLM:
// a request takes pages as it grows, up to --max-model-len; --max-num-seqs caps the requests running at
// once (each holds ~17 MB of fixed state: the attention window and compressor rings). When the pool
// runs dry the newest request is preempted and recomputed later. Default pool: 2 x --max-model-len.
// (--max-slots, --max-seq and --num-pages are the older names of the same three settings.)
//
// Prefix caching keeps finished requests' pages in the pool until they are needed and --prefix-snapshots
// (default 128, ~7 MB of pinned host memory each; 0 turns prefix caching off) slot-state snapshots at
// page boundaries, so a request that repeats an earlier one's prefix (the next turn of a chat, a shared
// system prompt) starts from the deepest matching boundary instead of position 0.
//
// --spec-drafts N (default 0: off) turns on speculative decoding with the checkpoint's DSpark drafter
// (its mtp.* stages: dense parts on card 1, ~10 GB of routed experts on the APU): each decode step
// verifies up to N drafted tokens (N <= 5) and keeps the ones the model would have produced.
// --spec-min-conf C: verify only the leading drafts whose confidence-head score is >= C (default 0.5: on
// short-context code and prose, 5 drafts at 0.5 gave 33.6 / 23.9 tok/s vs 30.0 / 19.1 verifying all).
//
// Prefill scheduling: decodes always go first; prefill chunks fill the rest of the step's token budget
// (--max-batch-tokens) in --prefill-order: `shortest` (default) packs the least remaining prompt first,
// so a continuation with a few hundred uncached tokens is not queued behind a long prefill (the oldest
// prefill keeps a quarter of each step); `fcfs` is arrival order, vLLM's default.
// --long-prefill-token-threshold N / --long-prefill-token-threshold-adaptive 1: vLLM's per-request cap
// (when more than one request is running or waiting; adaptive: never below budget / requests). Off.
// --decode-split 1: decoding requests go in two alternating steps, so the two cards work on different
// halves at once (default 0: every decoder in every step, as the 2026-10-12 build forms its steps; the
// split changes which rows share a step and with it their values).
// --exact 1: the 2026-10-12 build's outputs for the same steps (bit for bit). By default serve trades that
// for speed, with changes at the noise level (DECISIONS 2026-10-15): --reclaim-scratch 1 (plan the hot
// experts with the scratch this build allocates, ~0.9 GB more per card for experts, instead of that build's
// count), UNHOLY_MX_GEMV_TOKENS=64 (the experts' GEMVs for steps of up to 64 tokens: verify steps of several
// streams) and UNHOLY_DSPARK_WMMA_ATTN=1 (the drafter's attention on the WMMA kernel). Each can be set on its
// own (--reclaim-scratch 0, UNHOLY_MX_GEMV_TOKENS=8, UNHOLY_DSPARK_WMMA_ATTN=0). Other options with the same
// trade, off: --decode-split 1, UNHOLY_GEMV_ROWS=16 (slower than this build's GEMMs at 9-16 rows).
//
// Logging (stdout, so journald under systemd): a line per finished request (its tokens, queue time, time
// to first token, prefill and decode tok/s, draft acceptance) and, every --log-interval seconds (default
// 10; 0: off) while there is traffic, the engine's throughput, queue, KV pool use, prefix-cache hit rate
// and draft acceptance (rate, mean tokens per verify step, by draft position) over the interval.
//
// --model is the native checkpoint directory (config, tokenizer, dense weights); --exl3 the EXL3
// quant whose routed experts are used instead of the native MXFP4 ones. Placement comes from the
// planner (src/engine/plan.hpp) for the requested cache geometry.
#include <chrono>
#include <condition_variable>
#include <stop_token>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "engine/engine.hpp"
#include "engine/glm5_resident.hpp"
#include "engine/plan.hpp"
#include "engine/resident.hpp"
#include "model/dsv4.hpp"
#include "model/family.hpp"
#include "model/safetensors.hpp"
#include "server/openai.hpp"
#include "text/chat.hpp"
#include "text/glm_chat.hpp"
#include "text/tokenizer.hpp"

#ifndef UNHOLY_DATA_DIR
#define UNHOLY_DATA_DIR "data"
#endif

using namespace unholy;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: unholy serve --model DIR [--exl3 DIR] [--profile FILE] [--host H] [--port P]\n"
               "                    [--served-model-name NAME] [--max-num-seqs N] [--max-model-len N] [--kv-tokens N]\n"
               "                    [--max-batch-tokens N] [--api-key KEY] [--http-threads N]\n"
               "                    [--max-tokens-default N] [--card-margin-gb G] [--prefix-snapshots N]\n"
               "                    [--spec-drafts N] [--spec-min-conf C] [--log-interval S]\n"
               "                    [--prefill-order shortest|fcfs] [--long-prefill-token-threshold N]\n"
               "                    [--long-prefill-token-threshold-adaptive 0|1] [--decode-split 0|1]\n"
               "                    [--reclaim-scratch 0|1] [--exact 0|1]\n");
  return 2;
}

// Log lines are written whole under one lock, each prefixed with the local time.
std::mutex g_log_mu;

void log_line(const std::string& line) {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char ts[16];
  std::strftime(ts, sizeof(ts), "%H:%M:%S", &tm);
  std::lock_guard lk(g_log_mu);
  std::printf("%s %s\n", ts, line.c_str());
  std::fflush(stdout);
}

std::string fmt(const char* f, auto... a) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a...);
  return buf;
}

std::string request_line(const engine::RequestLog& l) {
  std::string out = "request " + (l.tag.empty() ? std::to_string(l.id) : l.tag) + ": " +
                    engine::finish_reason_name(l.reason);
  if (!l.error.empty()) out += " (" + l.error + ")";
  out += fmt(" | prompt %d", l.prompt_tokens);
  if (l.cached_tokens > 0) out += fmt(" (%d cached)", l.cached_tokens);
  if (l.images > 0) out += fmt(", %d image%s", l.images, l.images > 1 ? "s" : "");
  out += fmt(", completion %d | queue %.2f s", l.completion_tokens, l.queue_s);
  if (l.ttft_s >= 0) {
    out += fmt(", ttft %.2f s", l.ttft_s);
    const int computed = l.prompt_tokens - l.cached_tokens;
    if (l.prefill_s > 0) out += fmt(", prefill %.1f tok/s", computed / l.prefill_s);
    if (l.completion_tokens > 1 && l.decode_s > 0)
      out += fmt(", decode %.1f tok/s", (l.completion_tokens - 1) / l.decode_s);
  }
  if (l.spec_drafted > 0)
    out += fmt(" | drafts %d/%d accepted (%.1f%%), %.2f tok/step", l.spec_accepted, l.spec_drafted,
               100.0 * l.spec_accepted / l.spec_drafted, 1.0 + double(l.spec_accepted) / l.spec_steps);
  if (l.preemptions > 0) out += fmt(" | preempted %d", l.preemptions);
  return out;
}

// The engine over the interval since `a` (vLLM's periodic stats line).
std::string engine_line(const engine::EngineStats& a, const engine::EngineStats& b, double dt, bool spec) {
  const auto d = [](uint64_t x, uint64_t y) { return double(y - x); };
  std::string out = fmt("engine: prompt %.1f tok/s, generation %.1f tok/s | running %d, waiting %d",
                        d(a.prefill_tokens, b.prefill_tokens) / dt, d(a.decode_tokens, b.decode_tokens) / dt,
                        b.running, b.waiting);
  const int used = b.total_pages - b.free_pages;
  out += fmt(" | kv %.1f%% (%d/%d pages, %d cached)", 100.0 * used / std::max(b.total_pages, 1), used, b.total_pages,
             b.cached_pages);
  if (b.prompt_tokens > a.prompt_tokens)
    out += fmt(" | prefix hit %.1f%%", 100.0 * d(a.cached_tokens, b.cached_tokens) / d(a.prompt_tokens, b.prompt_tokens));
  if (b.preemptions > a.preemptions) out += fmt(" | preempted %llu", (unsigned long long)(b.preemptions - a.preemptions));
  if (spec && b.spec_steps > a.spec_steps) {
    const double drafted = d(a.spec_drafted, b.spec_drafted), accepted = d(a.spec_accepted, b.spec_accepted);
    out += fmt(" | drafts accepted %.1f%%, %.2f tok/step, by position", drafted > 0 ? 100.0 * accepted / drafted : 0.0,
               1.0 + accepted / d(a.spec_steps, b.spec_steps));
    for (size_t j = 0; j < b.spec_pos_drafted.size(); ++j) {
      const double n = d(a.spec_pos_drafted[j], b.spec_pos_drafted[j]);
      if (n <= 0) break;
      out += fmt(" %.2f", d(a.spec_pos_accepted[j], b.spec_pos_accepted[j]) / n);
    }
  }
  return out;
}

// Everything the command line sets.
struct Options {
  std::string model_dir, exl3_dir, profile = std::string(UNHOLY_DATA_DIR) + "/expert_profile_unholy_mixed.txt";
  bool profile_set = false;  // --profile given (otherwise each family has its default)
  server::ServerConfig scfg;
  int max_slots = 8, max_seq = 65536, num_pages = 0, max_batch = 2048;
  long long kv_tokens = 0;
  int prefix_snapshots = 128;
  int spec_drafts = 0;
  bool vision = true;
  float spec_min_conf = 0.5f;
  double card_margin_gb = 2.0;
  double log_interval = 10;
  engine::PrefillOrder prefill_order = engine::PrefillOrder::Shortest;
  int long_prefill_threshold = 0;
  bool long_prefill_adaptive = false;
  bool decode_split = false;
  int reclaim_scratch = -1;  // -1: !exact
  bool exact = false;
};

// The serving loop every model family shares: the engine over `model`, the periodic stats line, the OpenAI
// server (`extras` adds the family's server settings once the engine exists) and the signal thread.
// eos: the end-of-sequence tokens (empty: the tokenizer's EOS).
int run_server(engine::StepModel& model, const text::Tokenizer& tok, const text::ChatFormat& chat, Options& o,
               int max_seq, sigset_t sigs, const std::function<void(server::ServerConfig&)>& extras,
               const std::vector<int32_t>& eos = {}) {
  engine::EngineConfig ecfg;
  ecfg.max_batch_tokens = o.max_batch;
  ecfg.eos_id = eos.empty() ? tok.eos_id() : eos[0];
  if (eos.size() > 1) ecfg.eos_ids.assign(eos.begin() + 1, eos.end());
  ecfg.max_context = max_seq;
  ecfg.spec_drafts = o.spec_drafts;
  ecfg.spec_min_conf = o.spec_min_conf;
  ecfg.seed = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
  ecfg.on_finish = [](const engine::RequestLog& l) { log_line(request_line(l)); };
  ecfg.prefill_order = o.prefill_order;
  ecfg.long_prefill_threshold = o.long_prefill_threshold;
  ecfg.long_prefill_adaptive = o.long_prefill_adaptive;
  ecfg.decode_split = o.decode_split;
  engine::Engine eng(model, ecfg);
  // Logs while there is traffic, plus the interval after it ends (throughput back to zero). A jthread
  // declared after the engine: stopped and joined before the engine goes away.
  std::jthread stats_thread;
  if (o.log_interval > 0) {
    stats_thread = std::jthread([&eng, log_interval = o.log_interval, spec = o.spec_drafts > 0](std::stop_token stop) {
      std::mutex m;
      std::condition_variable_any cv;
      auto prev = eng.stats();
      auto t_prev = std::chrono::steady_clock::now();
      bool was_active = false;
      for (;;) {
        {
          std::unique_lock lk(m);
          cv.wait_for(lk, stop, std::chrono::duration<double>(log_interval), [] { return false; });
        }
        if (stop.stop_requested()) return;
        const auto cur = eng.stats();
        const auto now = std::chrono::steady_clock::now();
        const bool active = cur.steps != prev.steps || cur.running > 0 || cur.waiting > 0;
        if (active || was_active)
          log_line(engine_line(prev, cur, std::chrono::duration<double>(now - t_prev).count(), spec));
        was_active = active;
        prev = cur;
        t_prev = now;
      }
    });
  }
  extras(o.scfg);
  server::OpenAIServer srv(eng, tok, chat, o.scfg);

  std::thread sig([&] {
    int s = 0;
    sigwait(&sigs, &s);
    std::printf("signal %d: stopping\n", s);
    std::fflush(stdout);
    srv.stop();
  });
  sig.detach();
  std::printf("serving %s on http://%s:%d\n", o.scfg.model_name.c_str(), o.scfg.host.c_str(), o.scfg.port);
  std::fflush(stdout);
  srv.run();
  return 0;
}

// DeepSeek-V4: the planner places the layers and experts, ResidentModel runs them, text::dsv4_chat_format().
int serve_dsv4(Options& o, std::unique_ptr<model::Checkpoint> ck, bool plan_only, sigset_t sigs) {
  const auto tok = text::Tokenizer::load(o.model_dir + "/tokenizer.json");
  std::unique_ptr<model::Checkpoint> xck;
  if (!o.exl3_dir.empty()) xck = model::Checkpoint::open(o.exl3_dir);
  const auto c = model::Dsv4Config::from_hf(ck->config());

  const int max_seq = (o.max_seq + model::kPageTokens - 1) / model::kPageTokens * model::kPageTokens;
  model::CacheGeometry geo;
  geo.max_slots = o.max_slots;
  geo.max_pages_per_seq = max_seq / model::kPageTokens;
  long long kv_tokens = o.kv_tokens;
  if (kv_tokens <= 0) kv_tokens = o.num_pages > 0 ? (long long)(o.num_pages) * model::kPageTokens : 2LL * max_seq;
  geo.num_pages = int((kv_tokens + model::kPageTokens - 1) / model::kPageTokens);

  engine::PlanInput pin;
  pin.cfg = &c;
  pin.dense = ck.get();
  pin.exl3 = xck.get();
  pin.usage = engine::load_usage_profile(o.profile, c.n_layers, c.n_experts);
  pin.geo = geo;
  pin.max_tokens = o.max_batch;
  pin.max_seq = max_seq;
  pin.dspark = o.spec_drafts > 0;
  const bool vision = o.vision && c.vision_layers > 0;
  pin.vision = vision;
  pin.reclaim_scratch = o.reclaim_scratch < 0 ? !o.exact : o.reclaim_scratch > 0;
  // A verify chunk wants logits for its token and every draft.
  pin.max_logit_rows = std::max(o.max_slots * (o.spec_drafts > 0 ? c.dspark_block + 1 : 1), 8);
  // VRAM left free on each card for the HIP runtime (kernel scratch, kernarg pools, staging). 2 GB: the
  // faults once blamed on headroom were c7 under the 230 W power cap (DECISIONS 2026-10-12); 2 GB moves
  // ~2 GB of experts off the APU, which three 512k requests + DSpark + the vision tower need.
  pin.card_margin = size_t(o.card_margin_gb * double(1ull << 30));
  if (plan_only) {
    std::printf("geometry: kv pool %lld tokens (%d pages), up to %d tokens per request, %d requests at once; step %d "
                "tokens; experts %s\n",
                (long long)(geo.num_pages) * model::kPageTokens, geo.num_pages, max_seq, o.max_slots, o.max_batch,
                xck ? "EXL3" : "native (MXFP4)");
    try {
      std::printf("%s", engine::make_plan(pin).summary().c_str());
    } catch (const std::exception& e) {
      std::printf("does not fit: %s\n", e.what());
      return 1;
    }
    return 0;
  }
  const auto plan = engine::make_plan(pin);
  std::printf("%s", plan.summary().c_str());
  std::fflush(stdout);

  const auto t0 = std::chrono::steady_clock::now();
  engine::ResidentModel model(c, *ck, xck.get(), plan, geo, o.max_batch, max_seq, pin.max_logit_rows,
                              o.prefix_snapshots, o.spec_drafts > 0, vision);
  std::printf("loaded in %.1f s; kv pool %lld tokens, up to %d per request, %d requests at once\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              (long long)(geo.num_pages) * model::kPageTokens, max_seq, o.max_slots);

  return run_server(model, tok, text::dsv4_chat_format(), o, max_seq, sigs, [&](server::ServerConfig& scfg) {
    if (model.has_vision()) {
      scfg.encode_image = [&model](const vision::Patches& p, const vision::ImageBlock& b, std::vector<uint16_t>& rows) {
        model.encode_image(p, b, rows);
      };
      scfg.image_token = tok.token_id("<｜deepseek_image｜>");
      scfg.vocab = c.vocab;
      scfg.vision = vision::VisionConfig{c.vision_patch, c.vision_downsample, c.vision_max_tokens, c.vision_min_pixels,
                                         c.vision_max_wh_ratio};
      if (scfg.image_token < 0) throw std::runtime_error("the tokenizer has no image placeholder token");
    }
    // Routing histogram of the traffic served so far, in the --profile format (for re-planning).
    scfg.extra_get.emplace_back("/unholy/routing_profile",
                                [&model] { return engine::format_usage_profile(model.routing_histogram()); });
    scfg.extra_post.emplace_back("/unholy/routing_profile/reset", [&model] {
      model.reset_routing_histogram();
      return std::string("ok\n");
    });
  });
}

// GLM-5.3-Flash, an EXL3 quant (build B, docs/PLAN-glm-dsv41.md): make_glm5_plan places the layers and experts,
// Glm5ResidentModel runs them, text::glm5_chat_format(). No prefix snapshots, drafts or images yet.
int serve_glm5(Options& o, std::unique_ptr<model::Checkpoint> ck, bool plan_only, sigset_t sigs) {
  const auto tok = text::Tokenizer::load(o.model_dir + "/tokenizer.json");
  const auto c = model::Glm5Config::from_hf(ck->config());
  if (o.spec_drafts > 0 || o.vision)
    std::printf("GLM-5.3-Flash: speculative drafts and images are not served yet; serving without them\n");
  o.spec_drafts = 0;
  o.prefix_snapshots = 0;
  if (!o.profile_set) o.profile = std::string(UNHOLY_DATA_DIR) + "/expert_profile_glm53_mixed.txt";

  const int max_seq = (o.max_seq + model::kPageTokens - 1) / model::kPageTokens * model::kPageTokens;
  model::CacheGeometry geo;
  geo.max_slots = o.max_slots;
  geo.max_pages_per_seq = max_seq / model::kPageTokens;
  long long kv_tokens = o.kv_tokens;
  if (kv_tokens <= 0) kv_tokens = o.num_pages > 0 ? (long long)(o.num_pages) * model::kPageTokens : 2LL * max_seq;
  geo.num_pages = int((kv_tokens + model::kPageTokens - 1) / model::kPageTokens);

  engine::Glm5PlanInput pin;
  pin.cfg = &c;
  pin.ck = ck.get();
  pin.usage = engine::load_usage_profile(o.profile, c.n_layers, c.n_experts);
  pin.geo = geo;
  pin.max_tokens = o.max_batch;
  pin.max_seq = max_seq;
  pin.max_logit_rows = std::max(o.max_slots, 8);
  pin.card_margin = size_t(o.card_margin_gb * double(1ull << 30));
  if (plan_only) {
    std::printf("geometry: kv pool %lld tokens (%d pages), up to %d tokens per request, %d requests at once; step %d "
                "tokens; GLM-5.3-Flash EXL3\n",
                (long long)(geo.num_pages) * model::kPageTokens, geo.num_pages, max_seq, o.max_slots, o.max_batch);
    try {
      std::printf("%s", engine::make_glm5_plan(pin).summary().c_str());
    } catch (const std::exception& e) {
      std::printf("does not fit: %s\n", e.what());
      return 1;
    }
    return 0;
  }
  const auto plan = engine::make_glm5_plan(pin);
  std::printf("%s", plan.summary().c_str());
  std::fflush(stdout);
  const auto t0 = std::chrono::steady_clock::now();
  engine::Glm5ResidentModel model(c, *ck, plan, geo, o.max_batch, max_seq, pin.max_logit_rows);
  std::printf("loaded in %.1f s; kv pool %lld tokens, up to %d per request, %d requests at once\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              (long long)(geo.num_pages) * model::kPageTokens, max_seq, o.max_slots);
  std::vector<int32_t> eos;
  for (const char* t : {"<|endoftext|>", "<|user|>", "<|observation|>"}) {
    const int32_t id = tok.token_id(t);
    if (id < 0) throw std::runtime_error(std::string("GLM tokenizer: no ") + t);
    eos.push_back(id);
  }
  return run_server(
      model, tok, text::glm5_chat_format(), o, max_seq, sigs,
      [&](server::ServerConfig& scfg) {
        scfg.extra_get.emplace_back("/unholy/routing_profile",
                                    [&model] { return engine::format_usage_profile(model.routing_histogram()); });
        scfg.extra_post.emplace_back("/unholy/routing_profile/reset", [&model] {
          model.reset_routing_histogram();
          return std::string("ok\n");
        });
      },
      eos);
}

int serve(int argc, char** argv, bool plan_only) {
  Options o;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) return usage();
    const std::string v = argv[++i];
    if (a == "--model") o.model_dir = v;
    else if (a == "--exl3") o.exl3_dir = v;
    else if (a == "--profile") o.profile = v, o.profile_set = true;
    else if (a == "--host") o.scfg.host = v;
    else if (a == "--port") o.scfg.port = std::stoi(v);
    else if (a == "--served-model-name") o.scfg.model_name = v;
    else if (a == "--api-key") o.scfg.api_key = v;
    else if (a == "--http-threads") o.scfg.threads = std::stoi(v);
    else if (a == "--max-tokens-default") o.scfg.default_max_tokens = std::stoi(v);
    else if (a == "--max-num-seqs" || a == "--max-slots") o.max_slots = std::stoi(v);
    else if (a == "--max-model-len" || a == "--max-seq") o.max_seq = std::stoi(v);
    else if (a == "--kv-tokens") o.kv_tokens = std::stoll(v);
    else if (a == "--prefix-snapshots") o.prefix_snapshots = std::stoi(v);
    else if (a == "--spec-drafts") o.spec_drafts = std::stoi(v);
    else if (a == "--vision") o.vision = v != "0" && v != "off" && v != "false";
    else if (a == "--spec-min-conf") o.spec_min_conf = std::stof(v);
    else if (a == "--num-pages") o.num_pages = std::stoi(v);
    else if (a == "--max-batch-tokens") o.max_batch = std::stoi(v);
    else if (a == "--card-margin-gb") o.card_margin_gb = std::stod(v);
    else if (a == "--log-interval") o.log_interval = std::stod(v);
    else if (a == "--prefill-order") {
      if (v == "shortest") o.prefill_order = engine::PrefillOrder::Shortest;
      else if (v == "fcfs") o.prefill_order = engine::PrefillOrder::Fcfs;
      else return usage();
    } else if (a == "--long-prefill-token-threshold") o.long_prefill_threshold = std::stoi(v);
    else if (a == "--long-prefill-token-threshold-adaptive") o.long_prefill_adaptive = v != "0" && v != "off" && v != "false";
    else if (a == "--decode-split") o.decode_split = v != "0" && v != "off" && v != "false";
    else if (a == "--reclaim-scratch") o.reclaim_scratch = v != "0" && v != "off" && v != "false";
    else if (a == "--exact") o.exact = v != "0" && v != "off" && v != "false";
    else return usage();
  }
  if (o.model_dir.empty()) return usage();
  // The kernel options of the default (not --exact) trade; an explicit environment setting wins. Before
  // anything reads them (they are read once).
  if (!o.exact) {
    setenv("UNHOLY_MX_GEMV_TOKENS", "64", 0);
    setenv("UNHOLY_DSPARK_WMMA_ATTN", "1", 0);
  }
  if (o.scfg.api_key.empty())
    if (const char* k = std::getenv("UNHOLY_API_KEY")) o.scfg.api_key = k;

  // Block SIGINT/SIGTERM in every thread; one thread waits for them and stops the server.
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, SIGINT);
  sigaddset(&sigs, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

  auto ck = model::Checkpoint::open(o.model_dir);
  switch (model::detect_family(ck->config())) {
    case model::Family::Dsv4:
      return serve_dsv4(o, std::move(ck), plan_only, sigs);
    case model::Family::Glm5Next:
      return serve_glm5(o, std::move(ck), plan_only, sigs);
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  try {
    if (!std::strcmp(argv[1], "serve")) return serve(argc, argv, false);
    if (!std::strcmp(argv[1], "plan")) return serve(argc, argv, true);  // placement only, nothing loaded
    return usage();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
