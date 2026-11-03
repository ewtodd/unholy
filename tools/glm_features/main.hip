// unholy-glm-features: per prompt, a prefill and a greedy continuation through the resident GLM-5.3-Flash, writing
// the tokens and the hidden states a DFlash drafter conditions on (the mean of the hc streams after each given layer),
// for tools/ref/dflash2_ref.py's acceptance simulation.
//
//   unholy-glm-features --model DIR --profile FILE --out DIR [--gen 512] [--chunk 3072]
//                       [--layers 5,14,24,33,42] [--eos 154820,154827,154829] PROMPT.i32...
//
// Each prompt (raw int32 token ids) gets OUT/<file stem>/: tokens.i32 (the prompt, then the continuation up to an end
// token or --gen tokens; every token went through the model), features.bf16 ([T][layers][dim]: position t's state
// after each layer) and meta.json. Steps run one at a time (prefill chunks, then one token per decode step); the
// continuation is the argmax of each step's logits (ties: the lower id).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "engine/glm5_resident.hpp"
#include "engine/plan.hpp"
#include "model/family.hpp"
#include "model/glm5.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

using namespace unholy;
namespace fs = std::filesystem;

namespace {

std::vector<int> parse_list(const std::string& v) {
  std::vector<int> out;
  std::stringstream ss(v);
  for (std::string t; std::getline(ss, t, ',');) out.push_back(std::stoi(t));
  return out;
}

std::vector<int32_t> read_ids(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<int32_t> ids;
  for (int32_t x; f.read(reinterpret_cast<char*>(&x), 4);) ids.push_back(x);
  if (ids.empty()) throw std::runtime_error("no token ids in " + path);
  return ids;
}

template <typename T>
void write_raw(const fs::path& path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
  if (!f) throw std::runtime_error("cannot write " + path.string());
}

int32_t argmax(const float* row, int n) {
  int32_t best = 0;
  for (int i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, profile, out_dir;
  std::vector<int> layers = {5, 14, 24, 33, 42}, eos = {154820, 154827, 154829};
  std::vector<std::string> prompts;
  int gen = 512, chunk = 3072;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--", 0) != 0) {
      prompts.push_back(a);
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s needs a value\n", a.c_str());
      return 2;
    }
    const std::string v = argv[++i];
    if (a == "--model") model_dir = v;
    else if (a == "--profile") profile = v;
    else if (a == "--out") out_dir = v;
    else if (a == "--gen") gen = std::stoi(v);
    else if (a == "--chunk") chunk = std::stoi(v);
    else if (a == "--layers") layers = parse_list(v);
    else if (a == "--eos") eos = parse_list(v);
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (model_dir.empty() || out_dir.empty() || prompts.empty()) {
    std::fprintf(stderr, "usage: unholy-glm-features --model DIR --profile FILE --out DIR [options] PROMPT.i32...\n");
    return 2;
  }
  try {
    std::vector<std::vector<int32_t>> ids;
    size_t longest = 0;
    for (const auto& p : prompts) {
      ids.push_back(read_ids(p));
      longest = std::max(longest, ids.back().size());
    }
    auto ck = model::Checkpoint::open(model_dir);
    if (model::detect_family(ck->config()) != model::Family::Glm5Next)
      throw std::runtime_error(model_dir + " is not a GLM-5.3-Flash checkpoint");
    model::Glm5Config c = model::Glm5Config::from_hf(ck->config());
    const int max_seq = int((longest + size_t(gen) + 16 + model::kPageTokens - 1) / model::kPageTokens *
                            model::kPageTokens);
    model::CacheGeometry geo{1, max_seq / model::kPageTokens, max_seq / model::kPageTokens};
    engine::Glm5PlanInput pin;
    pin.cfg = &c;
    pin.ck = ck.get();
    pin.usage = engine::load_usage_profile(profile, c.n_layers, c.n_experts);
    pin.geo = geo;
    pin.max_tokens = chunk;
    pin.max_seq = max_seq;
    pin.max_logit_rows = 8;
    const auto plan = engine::make_glm5_plan(pin);
    std::printf("%s", plan.summary().c_str());
    engine::Glm5ResidentModel m(c, *ck, plan, geo, chunk, max_seq, 8);
    std::vector<int32_t> tables(size_t(geo.max_pages_per_seq));
    for (size_t i = 0; i < tables.size(); ++i) tables[i] = int32_t(i);
    m.set_block_tables(tables);

    const size_t L = layers.size(), D = size_t(c.dim);
    uint16_t* cap = nullptr;  // [L][chunk][dim], written by both cards
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&cap), L * size_t(chunk) * D * 2,
                           hipHostMallocPortable | hipHostMallocMapped));
    model::glm5_set_feature_capture(layers, cap, chunk);

    std::vector<float> logits;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
      const auto& prompt = ids[pi];
      std::vector<int32_t> tokens;
      std::vector<uint16_t> feats;  // [T][L][D]
      int32_t next = -1;
      // One step over tokens t0 .. t0 + n - 1 at positions pos .. pos + n - 1; their features join feats.
      const auto step = [&](const int32_t* t0, int n, int pos) {
        engine::StepInput in;
        in.chunks = {{0, pos, n, 0}};
        for (int k = 0; k < n; ++k) {
          in.ids.push_back(t0[k]);
          in.pos.push_back(pos + k);
        }
        in.logit_rows = {n - 1};
        m.wait(m.submit(in), logits);
        const size_t base = feats.size();
        feats.resize(base + size_t(n) * L * D);
        for (int r = 0; r < n; ++r)
          for (size_t j = 0; j < L; ++j)
            std::memcpy(&feats[base + (size_t(r) * L + j) * D], cap + (j * size_t(chunk) + size_t(r)) * D, D * 2);
        next = argmax(logits.data(), c.vocab);
      };
      for (size_t pos = 0; pos < prompt.size(); pos += size_t(chunk)) {
        const int n = int(std::min(prompt.size() - pos, size_t(chunk)));
        step(prompt.data() + pos, n, int(pos));
      }
      tokens = prompt;
      for (int g = 0; g < gen && std::find(eos.begin(), eos.end(), next) == eos.end(); ++g) {
        tokens.push_back(next);
        step(&tokens.back(), 1, int(tokens.size()) - 1);
      }
      const fs::path dir = fs::path(out_dir) / fs::path(prompts[pi]).stem();
      fs::create_directories(dir);
      write_raw(dir / "tokens.i32", tokens);
      write_raw(dir / "features.bf16", feats);
      std::ofstream meta(dir / "meta.json");
      meta << "{\"prompt_len\": " << prompt.size() << ", \"target_layer_ids\": [";
      for (size_t j = 0; j < L; ++j) meta << (j ? ", " : "") << layers[j];
      meta << "], \"label\": \"" << fs::path(prompts[pi]).stem().string() << "\"}\n";
      std::printf("%s: %zu prompt + %zu generated tokens\n", dir.string().c_str(), prompt.size(),
                  tokens.size() - prompt.size());
      std::fflush(stdout);
    }
    model::glm5_set_feature_capture({}, nullptr, 0);
    UH_CHECK(hipHostFree(cap));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
