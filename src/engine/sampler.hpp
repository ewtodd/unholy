// Token sampling on the host, per request (penalties, temperature, top-k / top-p / min-p, seeds,
// logprobs). Logits are the model's raw fp32 row; logprobs report that raw distribution (as vLLM's
// default raw_logprobs mode does).
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

namespace unholy::engine {

struct SamplingParams {
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = 0;  // <= 0: off
  float min_p = 0.0f;
  float presence_penalty = 0.0f, frequency_penalty = 0.0f;  // over generated tokens
  float repetition_penalty = 1.0f;                          // over prompt + generated tokens
  std::optional<uint64_t> seed;
  bool logprobs = false;
  int top_logprobs = 0;
};

struct TokenLogprob {
  int32_t token = 0;
  float logprob = 0.0f;
};

struct Sampled {
  int32_t token = 0;
  float logprob = 0.0f;              // filled when logprobs were asked for
  std::vector<TokenLogprob> top;     // top_logprobs alternatives, best first
};

class Sampler {
 public:
  Sampler(const SamplingParams& p, uint64_t default_seed);
  // Penalty state: the prompt (for the repetition penalty) and every accepted token.
  void set_prompt(const std::vector<int32_t>& prompt);
  void accept(int32_t token);
  Sampled sample(const float* logits, int vocab);
  // Forget generated tokens (recompute after preemption replays them through accept()).
  void reset_output();

 private:
  SamplingParams p_;
  std::mt19937_64 rng_;
  std::unordered_map<int32_t, int> out_counts_;
  std::vector<int32_t> prompt_unique_;
  std::vector<float> work_;
  std::vector<std::pair<float, int32_t>> cand_;
  bool penalties_ = false;
};

}  // namespace unholy::engine
