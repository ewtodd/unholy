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
// --model is the native checkpoint directory (config, tokenizer, dense weights); --exl3 the EXL3
// quant whose routed experts are used instead of the native MXFP4 ones. Placement comes from the
// planner (src/engine/plan.hpp) for the requested cache geometry.
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "engine/engine.hpp"
#include "engine/plan.hpp"
#include "engine/resident.hpp"
#include "model/dsv4.hpp"
#include "model/safetensors.hpp"
#include "server/openai.hpp"
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
               "                    [--max-tokens-default N] [--card-margin-gb G] [--prefix-snapshots N]\n");
  return 2;
}

int serve(int argc, char** argv, bool plan_only) {
  std::string model_dir, exl3_dir, profile = std::string(UNHOLY_DATA_DIR) + "/expert_profile_unholy_mixed.txt";
  server::ServerConfig scfg;
  int max_slots = 8, max_seq = 65536, num_pages = 0, max_batch = 2048;
  long long kv_tokens = 0;
  int prefix_snapshots = 128;
  double card_margin_gb = 3.0;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) return usage();
    const std::string v = argv[++i];
    if (a == "--model") model_dir = v;
    else if (a == "--exl3") exl3_dir = v;
    else if (a == "--profile") profile = v;
    else if (a == "--host") scfg.host = v;
    else if (a == "--port") scfg.port = std::stoi(v);
    else if (a == "--served-model-name") scfg.model_name = v;
    else if (a == "--api-key") scfg.api_key = v;
    else if (a == "--http-threads") scfg.threads = std::stoi(v);
    else if (a == "--max-tokens-default") scfg.default_max_tokens = std::stoi(v);
    else if (a == "--max-num-seqs" || a == "--max-slots") max_slots = std::stoi(v);
    else if (a == "--max-model-len" || a == "--max-seq") max_seq = std::stoi(v);
    else if (a == "--kv-tokens") kv_tokens = std::stoll(v);
    else if (a == "--prefix-snapshots") prefix_snapshots = std::stoi(v);
    else if (a == "--num-pages") num_pages = std::stoi(v);
    else if (a == "--max-batch-tokens") max_batch = std::stoi(v);
    else if (a == "--card-margin-gb") card_margin_gb = std::stod(v);
    else return usage();
  }
  if (model_dir.empty()) return usage();
  if (scfg.api_key.empty())
    if (const char* k = std::getenv("UNHOLY_API_KEY")) scfg.api_key = k;

  // Block SIGINT/SIGTERM in every thread; one thread waits for them and stops the server.
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, SIGINT);
  sigaddset(&sigs, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

  const auto tok = text::Tokenizer::load(model_dir + "/tokenizer.json");
  auto ck = model::Checkpoint::open(model_dir);
  std::unique_ptr<model::Checkpoint> xck;
  if (!exl3_dir.empty()) xck = model::Checkpoint::open(exl3_dir);
  const auto c = model::Dsv4Config::from_hf(ck->config());

  max_seq = (max_seq + model::kPageTokens - 1) / model::kPageTokens * model::kPageTokens;
  model::CacheGeometry geo;
  geo.max_slots = max_slots;
  geo.max_pages_per_seq = max_seq / model::kPageTokens;
  if (kv_tokens <= 0) kv_tokens = num_pages > 0 ? (long long)(num_pages) * model::kPageTokens : 2LL * max_seq;
  geo.num_pages = int((kv_tokens + model::kPageTokens - 1) / model::kPageTokens);

  engine::PlanInput pin;
  pin.cfg = &c;
  pin.dense = ck.get();
  pin.exl3 = xck.get();
  pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
  pin.geo = geo;
  pin.max_tokens = max_batch;
  pin.max_seq = max_seq;
  pin.max_logit_rows = std::max(max_slots, 8);
  // VRAM left free on each card for the HIP runtime (kernel scratch, kernarg pools, staging). With
  // 1.5 GB the cards ended up ~300 MB from full and the driver faulted on weight pages.
  pin.card_margin = size_t(card_margin_gb * double(1ull << 30));
  if (plan_only) {
    std::printf("geometry: kv pool %lld tokens (%d pages), up to %d tokens per request, %d requests at once; step %d "
                "tokens; experts %s\n",
                (long long)(geo.num_pages) * model::kPageTokens, geo.num_pages, max_seq, max_slots, max_batch,
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
  engine::ResidentModel model(c, *ck, xck.get(), plan, geo, max_batch, max_seq, pin.max_logit_rows,
                              prefix_snapshots);
  std::printf("loaded in %.1f s; kv pool %lld tokens, up to %d per request, %d requests at once\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              (long long)(geo.num_pages) * model::kPageTokens, max_seq, max_slots);

  engine::EngineConfig ecfg;
  ecfg.max_batch_tokens = max_batch;
  ecfg.eos_id = tok.eos_id();
  ecfg.max_context = max_seq;
  ecfg.seed = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
  engine::Engine eng(model, ecfg);
  // Routing histogram of the traffic served so far, in the --profile format (for re-planning).
  scfg.extra_get.emplace_back("/unholy/routing_profile",
                              [&model] { return engine::format_usage_profile(model.routing_histogram()); });
  scfg.extra_post.emplace_back("/unholy/routing_profile/reset", [&model] {
    model.reset_routing_histogram();
    return std::string("ok\n");
  });
  server::OpenAIServer srv(eng, tok, scfg);

  std::thread sig([&] {
    int s = 0;
    sigwait(&sigs, &s);
    std::printf("signal %d: stopping\n", s);
    std::fflush(stdout);
    srv.stop();
  });
  sig.detach();
  std::printf("serving %s on http://%s:%d\n", scfg.model_name.c_str(), scfg.host.c_str(), scfg.port);
  std::fflush(stdout);
  srv.run();
  return 0;
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
