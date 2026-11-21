#include "engine/sampler.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>

namespace unholy::engine {

Sampler::Sampler(const SamplingParams& p, uint64_t default_seed) : p_(p), rng_(p.seed ? *p.seed : default_seed) {
  penalties_ = p_.presence_penalty != 0.0f || p_.frequency_penalty != 0.0f || p_.repetition_penalty != 1.0f;
}

void Sampler::set_prompt(const std::vector<int32_t>& prompt) {
  if (p_.repetition_penalty == 1.0f) return;
  prompt_unique_ = prompt;
  std::sort(prompt_unique_.begin(), prompt_unique_.end());
  prompt_unique_.erase(std::unique(prompt_unique_.begin(), prompt_unique_.end()), prompt_unique_.end());
}

void Sampler::accept(int32_t token) {
  if (penalties_) ++out_counts_[token];
}

void Sampler::reset_output() { out_counts_.clear(); }

float Sampler::exp_nonpos(float x) {
  x = x < -87.0f ? -87.0f : x;
  const float t = x * 1.44269504088896341f;
  const float n = (t + 12582912.0f) - 12582912.0f;  // nearest integer (|t| < 2^22): no floor, which baseline SSE lacks
  float r = x - n * 0.693359375f;
  r = r - n * -2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * (r * r) + r + 1.0f;
  return p * __builtin_bit_cast(float, uint32_t(int32_t(n) + 127) << 23);
}

double Sampler::weights_pass(const float* l, int n, float m, float inv_t, float* w, double* blocks) {
  for (int i = 0; i < n; ++i) {
    const float e = exp_nonpos((l[i] - m) * inv_t);
    w[i] = l[i] > -INFINITY ? e : 0.0f;
  }
  double total = 0.0;
  for (int b0 = 0, b = 0; b0 < n; b0 += kWeightBlock, ++b) {
    const int b1 = std::min(n, b0 + kWeightBlock);
    double acc[8] = {};
    int i = b0;
    for (; i + 8 <= b1; i += 8)
      for (int j = 0; j < 8; ++j) acc[j] += double(w[i + j]);
    for (; i < b1; ++i) acc[0] += double(w[i]);
    blocks[b] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    total += blocks[b];
  }
  return total;
}

Sampled Sampler::sample(const float* logits, int V) { return sample_impl(logits, V, -1, nullptr); }

Sampled Sampler::verify(const float* logits, int V, int32_t draft, bool& accepted) {
  return sample_impl(logits, V, draft, &accepted);
}

Sampled Sampler::sample_impl(const float* logits, int V, int32_t draft, bool* accepted) {
  Sampled s;
  if (accepted) *accepted = false;
  const float* l = logits;
  if (penalties_) {
    work_.assign(logits, logits + V);
    const float rp = p_.repetition_penalty;
    const auto repeat = [&](int32_t t) {
      float& x = work_[size_t(t)];
      x = x > 0.0f ? x / rp : x * rp;
    };
    if (rp != 1.0f) {
      for (const int32_t t : prompt_unique_)
        if (t >= 0 && t < V) repeat(t);
      for (const auto& [t, n] : out_counts_)
        if (t >= 0 && t < V && !std::binary_search(prompt_unique_.begin(), prompt_unique_.end(), t)) repeat(t);
    }
    for (const auto& [t, n] : out_counts_)
      if (t >= 0 && t < V) work_[size_t(t)] -= p_.frequency_penalty * float(n) + p_.presence_penalty;
    l = work_.data();
  }

  if (p_.temperature < 1e-5f || p_.top_k == 1) {
    s.token = int32_t(std::max_element(l, l + V) - l);  // first maximum
    if (accepted) *accepted = s.token == draft;
  } else if ((p_.top_k <= 0 || p_.top_k >= V) && p_.top_p >= 1.0f && p_.min_p <= 0.0f) {
    // The whole vocabulary (the usual case, e.g. temperature only): one fp32 pass of weights, the draw walks them in
    // token order (as the general path walks its candidates). A -inf logit weighs 0 (it is never drawn).
    const size_t im = size_t(std::max_element(l, l + V) - l);
    const float m = l[im];
    const int nb = (V + kWeightBlock - 1) / kWeightBlock;
    weights_.resize(size_t(V));
    blocks_.resize(size_t(nb));
    float* w = weights_.data();
    const double total = weights_pass(l, V, m, float(1.0 / double(p_.temperature)), w, blocks_.data());
    const bool has_draft = accepted && draft >= 0 && draft < V;
    const double pd = has_draft ? double(w[draft]) : 0.0;
    bool take_draft = false;
    if (accepted && pd > 0.0) take_draft = std::uniform_real_distribution<double>(0.0, total)(rng_) < pd;
    if (take_draft) {
      s.token = draft;
      *accepted = true;
    } else {
      // Rejected (or no draft): draw without the draft, by blocks, then within the block.
      double u = std::uniform_real_distribution<double>(0.0, total - pd)(rng_);
      s.token = -1;
      for (int b = 0; b < nb && s.token < 0; ++b) {
        const int b0 = b * kWeightBlock, b1 = std::min(V, b0 + kWeightBlock);
        const double bs = blocks_[size_t(b)] - (has_draft && draft >= b0 && draft < b1 ? pd : 0.0);
        if (u >= bs && b + 1 < nb) {
          u -= bs;
          continue;
        }
        for (int i = b0; i < b1; ++i) {
          if (w[i] == 0.0f || (has_draft && i == draft)) continue;
          s.token = int32_t(i);
          u -= double(w[i]);
          if (u < 0.0) break;
        }
      }
      if (s.token < 0) s.token = int32_t(im);
    }
  } else {
    const double T = p_.temperature;
    const float m = *std::max_element(l, l + V);
    // min_p: p_i >= min_p * p_max  <=>  l_i >= m + T * log(min_p).
    const float cut = p_.min_p > 0.0f ? m + float(T * std::log(double(p_.min_p))) : -INFINITY;
    cand_.clear();
    for (int i = 0; i < V; ++i)
      if (l[i] >= cut && l[i] > -INFINITY) cand_.emplace_back(l[i], int32_t(i));
    const auto greater = std::greater<std::pair<float, int32_t>>();
    size_t keep = cand_.size();
    const size_t k = p_.top_k > 0 ? std::min(size_t(p_.top_k), cand_.size()) : cand_.size();
    if (k < cand_.size() && p_.top_p >= 1.0f) {
      std::nth_element(cand_.begin(), cand_.begin() + ptrdiff_t(k), cand_.end(), greater);
      keep = k;
    } else if (p_.top_p < 1.0f) {
      // Nucleus over the top-k set: grow a sorted prefix until it holds top_p of the mass.
      if (k < cand_.size()) {
        std::nth_element(cand_.begin(), cand_.begin() + ptrdiff_t(k), cand_.end(), greater);
        cand_.resize(k);
      }
      double Z = 0.0;
      for (const auto& [x, i] : cand_) Z += std::exp((double(x) - m) / T);
      const double target = double(p_.top_p) * Z;
      size_t want = std::min<size_t>(64, cand_.size());
      for (;;) {
        std::partial_sort(cand_.begin(), cand_.begin() + ptrdiff_t(want), cand_.end(), greater);
        double cum = 0.0;
        size_t n = 0;
        while (n < want && cum < target) cum += std::exp((double(cand_[n++].first) - m) / T);
        if (cum >= target || want == cand_.size()) {
          keep = std::max<size_t>(n, 1);
          break;
        }
        want = std::min(cand_.size(), want * 16);
      }
    }
    double total = 0.0, pd = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      const double w = std::exp((double(cand_[i].first) - m) / T);
      total += w;
      if (cand_[i].second == draft) pd = w;
    }
    bool take_draft = false;
    if (accepted && pd > 0.0)  // accept with probability p(draft)
      take_draft = std::uniform_real_distribution<double>(0.0, total)(rng_) < pd;
    if (take_draft) {
      s.token = draft;
      *accepted = true;
    } else {
      // Rejected (or no draft): draw from the distribution without the draft.
      double u = std::uniform_real_distribution<double>(0.0, total - pd)(rng_);
      s.token = -1;
      for (size_t i = 0; i < keep; ++i) {
        if (accepted && cand_[i].second == draft) continue;
        s.token = cand_[i].second;
        u -= std::exp((double(cand_[i].first) - m) / T);
        if (u < 0.0) break;
      }
      if (s.token < 0) s.token = cand_[0].second;
    }
  }

  if (p_.logprobs || p_.top_logprobs > 0) {
    const float m = *std::max_element(logits, logits + V);
    double z = 0.0;
    for (int i = 0; i < V; ++i) z += std::exp(double(logits[i]) - m);
    const double lse = double(m) + std::log(z);
    s.logprob = float(double(logits[s.token]) - lse);
    if (p_.top_logprobs > 0) {
      const size_t n = std::min<size_t>(size_t(p_.top_logprobs), size_t(V));
      cand_.clear();
      for (int i = 0; i < V; ++i) cand_.emplace_back(logits[i], int32_t(i));
      std::partial_sort(cand_.begin(), cand_.begin() + ptrdiff_t(n), cand_.end(),
                        std::greater<std::pair<float, int32_t>>());
      for (size_t i = 0; i < n; ++i) s.top.push_back({cand_[i].second, float(double(cand_[i].first) - lse)});
    }
  }
  return s;
}

}  // namespace unholy::engine
