// unholy-engine-batch: OpenAI chat requests through the serving engine with no HTTP and no second thread.
// The engine is constructed with start = false and driven by iterate() on this thread after every request
// is submitted, so the steps it forms -- which requests share a step, how prompts are chunked, the prefix
// cache's hits -- are a function of the requests and the flags alone, not of wall-clock timing. Two builds
// that compute every step alike therefore produce the same tokens at any --concurrency: the token-for-token
// check of a change under concurrency, which a load test against the server cannot give (its step
// composition depends on when requests arrive and steps complete).
//
//   unholy-engine-batch --model DIR --requests FILE.jsonl --out FILE.jsonl [--concurrency N]
//                       [--profile FILE] [--exl3 DIR] [--max-num-seqs N] [--max-model-len N] [--kv-tokens N]
//                       [--max-batch-tokens N] [--prefix-snapshots N] [--spec-drafts N] [--spec-min-conf C]
//                       [--card-margin-gb G] [--vision 0|1] [--decode-split 0|1] [--reclaim-scratch 0|1]
//                       [--logprobs 0|1] [--glm-mtp 0|1]
//
// FILE.jsonl: one chat-completions request body per line (messages, chat_template_kwargs, temperature,
// top_p, top_k, min_p, penalties, seed, max_tokens, ignore_eos, stop_token_ids; text only), rendered and
// tokenized as the server does. --concurrency caps the requests running at once (default --max-num-seqs);
// the rest wait and are admitted as others finish, like a client holding N requests open. Model flags and
// their defaults are serve's. Output: one line per request in input order (index, prompt / cached /
// completion tokens, finish reason, token ids, text), then the engine's counters on stdout. --logprobs 1 adds
// each sampled token's logprob (bit pattern in hex): a fingerprint of the logits, finer than the tokens, for
// checking that repeated runs compute alike (tools/engine_batch/compare.py compares runs and builds).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "engine/engine.hpp"
#include "engine/glm5_resident.hpp"
#include "engine/plan.hpp"
#include "engine/resident.hpp"
#include "model/dsv4.hpp"
#include "model/family.hpp"
#include "model/glm5.hpp"
#include "model/safetensors.hpp"
#include "text/chat.hpp"
#include "text/glm_chat.hpp"
#include "text/json.hpp"
#include "text/tokenizer.hpp"

using namespace unholy;
using json::Value;

namespace {

// Options a build may not have (the comparison runs this file against older trees).
template <class C> void set_decode_split(C& c, std::optional<bool> v) {
  if constexpr (requires { c.decode_split; }) {
    if (v) c.decode_split = *v;
  } else if (v) {
    throw std::runtime_error("this build has no --decode-split");
  }
}
template <class P> void set_reclaim_scratch(P& p, std::optional<bool> v) {
  if constexpr (requires { p.reclaim_scratch; }) {
    if (v) p.reclaim_scratch = *v;
  } else if (v) {
    throw std::runtime_error("this build has no --reclaim-scratch");
  }
}
template <class C> std::string decode_split_name(const C& c) {
  if constexpr (requires { c.decode_split; }) return c.decode_split ? "on" : "off";
  else return "n/a";
}

double num(const Value& req, std::string_view key, double def) {
  const Value* v = req.find(key);
  return v && v->is_number() ? v->as_double() : def;
}

// server/openai.cpp parse_params (chat), the generation fields.
engine::GenRequest gen_request(const Value& body, const text::Tokenizer& tok, const text::ChatFormat& fmt) {
  engine::GenRequest g;
  auto& s = g.sampling;
  s.temperature = float(num(body, "temperature", 1.0));
  s.top_p = float(num(body, "top_p", 1.0));
  s.top_k = int(num(body, "top_k", 0));
  s.min_p = float(num(body, "min_p", 0.0));
  s.presence_penalty = float(num(body, "presence_penalty", 0.0));
  s.frequency_penalty = float(num(body, "frequency_penalty", 0.0));
  s.repetition_penalty = float(num(body, "repetition_penalty", 1.0));
  if (const Value* v = body.find("seed"); v && v->is_number()) s.seed = uint64_t(v->as_int());
  const Value* mt = body.find("max_completion_tokens");
  if (!mt || mt->is_null()) mt = body.find("max_tokens");
  g.max_tokens = mt && mt->is_number() ? int(mt->as_int()) : 0;
  if (const Value* v = body.find("stop_token_ids"); v && v->is_array())
    for (const Value& x : v->items()) g.stop_token_ids.push_back(int32_t(x.as_int()));
  if (const Value* v = body.find("ignore_eos")) g.ignore_eos = v->truthy();
  const text::RenderedChat r = fmt.render(body);
  if (!r.images.empty()) throw std::runtime_error("images are not supported here");
  g.prompt = tok.encode(r.prompt);
  return g;
}

bool flag(const std::string& v) { return v != "0" && v != "off" && v != "false"; }

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, exl3_dir, requests_path, out_path;
  std::string profile = "data/expert_profile_unholy_mixed.txt";
  int max_slots = 8, max_seq = 65536, max_batch = 2048, prefix_snapshots = 128, spec_drafts = 0, conc = 0;
  int glm_mtp = -1;  // GLM: load the MTP layer (-1: when drafting)
  long long kv_tokens = 0;
  float spec_min_conf = 0.5f;
  double card_margin_gb = 2.0;
  bool vision = true, logprobs = false;
  std::optional<bool> decode_split, reclaim_scratch;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s: missing value (see the header of tools/engine_batch/main.cpp)\n", a.c_str());
      return 2;
    }
    const std::string v = argv[++i];
    if (a == "--model") model_dir = v;
    else if (a == "--exl3") exl3_dir = v;
    else if (a == "--profile") profile = v;
    else if (a == "--requests") requests_path = v;
    else if (a == "--out") out_path = v;
    else if (a == "--concurrency") conc = std::stoi(v);
    else if (a == "--max-num-seqs") max_slots = std::stoi(v);
    else if (a == "--max-model-len") max_seq = std::stoi(v);
    else if (a == "--kv-tokens") kv_tokens = std::stoll(v);
    else if (a == "--max-batch-tokens") max_batch = std::stoi(v);
    else if (a == "--prefix-snapshots") prefix_snapshots = std::stoi(v);
    else if (a == "--spec-drafts") spec_drafts = std::stoi(v);
    else if (a == "--glm-mtp") glm_mtp = std::stoi(v);
    else if (a == "--spec-min-conf") spec_min_conf = std::stof(v);
    else if (a == "--card-margin-gb") card_margin_gb = std::stod(v);
    else if (a == "--vision") vision = flag(v);
    else if (a == "--decode-split") decode_split = flag(v);
    else if (a == "--reclaim-scratch") reclaim_scratch = flag(v);
    else if (a == "--logprobs") logprobs = flag(v);
    else {
      std::fprintf(stderr, "unknown option %s (see the header of tools/engine_batch/main.cpp)\n", a.c_str());
      return 2;
    }
  }
  if (model_dir.empty() || requests_path.empty() || out_path.empty()) {
    std::fprintf(stderr, "usage: %s --model DIR --requests FILE.jsonl --out FILE.jsonl [options]\n", argv[0]);
    return 2;
  }
  try {
    const auto tok = text::Tokenizer::load(model_dir + "/tokenizer.json");
    auto ck = model::Checkpoint::open(model_dir, {"mtp.safetensors"});
    const bool glm = model::detect_family(ck->config()) == model::Family::Glm5Next;
    const text::ChatFormat& fmt = glm ? text::glm5_chat_format() : text::dsv4_chat_format();
    std::vector<engine::GenRequest> reqs;
    {
      std::ifstream f(requests_path);
      if (!f) throw std::runtime_error("cannot read " + requests_path);
      for (std::string line; std::getline(f, line);)
        if (line.find_first_not_of(" \t\r") != std::string::npos) {
          reqs.push_back(gen_request(json::parse(line), tok, fmt));
          if (logprobs) reqs.back().sampling.logprobs = true;
        }
    }
    std::unique_ptr<model::Checkpoint> xck;
    if (!exl3_dir.empty()) xck = model::Checkpoint::open(exl3_dir);

    // serve's geometry and plan.
    max_seq = (max_seq + model::kPageTokens - 1) / model::kPageTokens * model::kPageTokens;
    model::CacheGeometry geo;
    geo.max_slots = max_slots;
    geo.max_pages_per_seq = max_seq / model::kPageTokens;
    if (kv_tokens <= 0) kv_tokens = 2LL * max_seq;
    geo.num_pages = int((kv_tokens + model::kPageTokens - 1) / model::kPageTokens);
    // The model by family (serve's plans; the configs outlive the model, which keeps a reference).
    std::unique_ptr<model::Dsv4Config> dc;
    std::unique_ptr<model::Glm5Config> gc;
    std::unique_ptr<engine::StepModel> model_ptr;
    std::vector<int32_t> eos;
    if (glm) {
      gc = std::make_unique<model::Glm5Config>(model::Glm5Config::from_hf(ck->config()));
      // --glm-mtp 1 loads the MTP layer (and its state backups) without drafting when --spec-drafts is 0: the same
      // placement as a drafting run, so the two compute alike (a check of the drafts' rollback).
      const bool mtp = (glm_mtp >= 0 ? glm_mtp > 0 : spec_drafts > 0) && gc->n_mtp > 0;
      spec_drafts = mtp && spec_drafts > 0 ? 1 : 0;
      engine::Glm5PlanInput pin;
      pin.cfg = gc.get();
      pin.ck = ck.get();
      pin.usage = engine::load_usage_profile(profile, gc->n_layers, gc->n_experts);
      pin.geo = geo;
      pin.max_tokens = max_batch;
      pin.max_seq = max_seq;
      pin.max_logit_rows = std::max(max_slots * (mtp ? 2 : 1), 8);
      pin.card_margin = size_t(card_margin_gb * double(1ull << 30));
      pin.mtp = mtp;
      pin.snapshots = prefix_snapshots;
      const auto plan = engine::make_glm5_plan(pin);
      std::printf("%s", plan.summary().c_str());
      std::fflush(stdout);
      model_ptr = std::make_unique<engine::Glm5ResidentModel>(*gc, *ck, plan, geo, max_batch, max_seq,
                                                               pin.max_logit_rows, mtp, plan.snapshots);
      for (const char* t : {"<|endoftext|>", "<|user|>", "<|observation|>"}) eos.push_back(tok.token_id(t));
    } else {
      dc = std::make_unique<model::Dsv4Config>(model::Dsv4Config::from_hf(ck->config()));
      const auto& c = *dc;
      engine::PlanInput pin;
      pin.cfg = &c;
      pin.dense = ck.get();
      pin.exl3 = xck.get();
      pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
      pin.geo = geo;
      pin.max_tokens = max_batch;
      pin.max_seq = max_seq;
      pin.dspark = spec_drafts > 0;
      vision = vision && c.vision_layers > 0;
      pin.vision = vision;
      set_reclaim_scratch(pin, reclaim_scratch);
      pin.max_logit_rows = std::max(max_slots * (spec_drafts > 0 ? c.dspark_block + 1 : 1), 8);
      pin.card_margin = size_t(card_margin_gb * double(1ull << 30));
      const auto plan = engine::make_plan(pin);
      std::printf("%s", plan.summary().c_str());
      std::fflush(stdout);
      model_ptr = std::make_unique<engine::ResidentModel>(c, *ck, xck.get(), plan, geo, max_batch, max_seq,
                                                          pin.max_logit_rows, prefix_snapshots, spec_drafts > 0,
                                                          vision);
    }
    engine::StepModel& model = *model_ptr;

    engine::EngineConfig ecfg;
    ecfg.max_batch_tokens = max_batch;
    ecfg.max_running = conc;
    ecfg.eos_id = eos.empty() ? tok.eos_id() : eos[0];
    if (eos.size() > 1) ecfg.eos_ids.assign(eos.begin() + 1, eos.end());
    ecfg.max_context = max_seq;
    ecfg.spec_drafts = spec_drafts;
    ecfg.spec_min_conf = spec_min_conf;
    ecfg.seed = 1;
    set_decode_split(ecfg, decode_split);
    engine::Engine eng(model, ecfg, false);

    struct Result {
      std::vector<int32_t> tokens;
      std::vector<float> logprobs;
      engine::GenOutput last;
      bool done = false;
    };
    std::vector<Result> res(reqs.size());
    size_t finished = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < reqs.size(); ++i)
      eng.submit(std::move(reqs[i]), [&res, &finished, i](engine::GenOutput&& o) {
        Result& r = res[i];
        r.tokens.insert(r.tokens.end(), o.tokens.begin(), o.tokens.end());
        for (const auto& sm : o.samples) r.logprobs.push_back(sm.logprob);
        if (o.finished) {
          r.done = true;
          ++finished;
        }
        r.last = std::move(o);
      });
    while (finished < res.size())
      if (!eng.iterate() && finished < res.size()) throw std::runtime_error("the engine stopped with requests left");
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::ofstream out(out_path);
    size_t completion = 0;
    for (size_t i = 0; i < res.size(); ++i) {
      const Result& r = res[i];
      Value line = Value::object();
      line.set("index", Value(static_cast<long long>(i)));
      line.set("prompt_tokens", Value(r.last.prompt_tokens));
      line.set("cached_tokens", Value(r.last.cached_tokens));
      line.set("completion_tokens", Value(static_cast<long long>(r.tokens.size())));
      line.set("finish", Value(engine::finish_reason_name(r.last.reason)));
      if (!r.last.error.empty()) line.set("error", Value(r.last.error));
      Value ids = Value::array();
      for (int32_t t : r.tokens) ids.push_back(Value(t));
      line.set("tokens", std::move(ids));
      if (logprobs) {
        std::string hex;
        char buf[16];
        for (float f : r.logprobs) {
          uint32_t u;
          std::memcpy(&u, &f, 4);
          std::snprintf(buf, sizeof buf, "%08x", u);
          hex += buf;
        }
        line.set("logprobs", Value(hex));
      }
      line.set("text", Value(tok.decode(r.tokens)));
      out << line.dump() << "\n";
      completion += r.tokens.size();
    }
    const auto s = eng.stats();
    std::printf("%zu requests, concurrency %d, decode split %s: %zu completion tokens in %.1f s (%.1f tok/s); "
                "%llu steps, %llu prefill / %llu decode tokens, %llu cached prompt tokens, drafts %llu / %llu "
                "accepted, %llu preemptions\n",
                res.size(), conc > 0 ? conc : max_slots, decode_split_name(ecfg).c_str(), completion, wall,
                double(completion) / wall, (unsigned long long)s.steps, (unsigned long long)s.prefill_tokens,
                (unsigned long long)s.decode_tokens, (unsigned long long)s.cached_tokens,
                (unsigned long long)s.spec_accepted, (unsigned long long)s.spec_drafted,
                (unsigned long long)s.preemptions);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
