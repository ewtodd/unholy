// The /metrics text (Prometheus exposition). Header-only and device-free so the CPU tests can call it.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "engine/engine.hpp"

namespace unholy::server {

// A label value: backslash, double quote and newline escaped.
inline std::string prom_escape(const std::string& v) {
  std::string o;
  for (const char c : v) {
    if (c == '\\') o += "\\\\";
    else if (c == '"') o += "\\\"";
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o;
}

inline std::string format_metrics(const engine::EngineStats& s, int context_tokens, const std::string& model_name,
                                  uint64_t http_requests, uint64_t http_errors) {
  std::string o;
  const auto put = [&o](const char* name, double v, const char* fmt) {
    char b[96];
    std::snprintf(b, sizeof(b), fmt, v);
    o += name;
    o += ' ';
    o += b;
    o += '\n';
  };
  const auto u = [&o](const char* name, uint64_t v) { o += name + std::string(" ") + std::to_string(v) + "\n"; };
  const auto i = [&o](const char* name, int v) { o += name + std::string(" ") + std::to_string(v) + "\n"; };
  u("unholy_steps_total", s.steps);
  u("unholy_prefill_tokens_total", s.prefill_tokens);
  u("unholy_decode_tokens_total", s.decode_tokens);
  u("unholy_preemptions_total", s.preemptions);
  put("unholy_busy_seconds_total", s.busy_seconds, "%.3f");
  i("unholy_running", s.running);
  i("unholy_waiting", s.waiting);
  i("unholy_free_pages", s.free_pages);
  i("unholy_total_pages", s.total_pages);
  u("unholy_http_requests_total", http_requests);
  u("unholy_http_errors_total", http_errors);
  u("unholy_prompt_tokens_total", s.prompt_tokens);
  u("unholy_cached_prompt_tokens_total", s.cached_tokens);
  i("unholy_cached_pages", s.cached_pages);
  i("unholy_cached_snapshots", s.snapshots);
  u("unholy_spec_verify_steps_total", s.spec_steps);
  u("unholy_spec_drafted_total", s.spec_drafted);
  u("unholy_spec_accepted_total", s.spec_accepted);
  // For llmtop (docs: DECISIONS.md 2026-11-08).
  o += "unholy_info{model=\"" + prom_escape(model_name) + "\"} 1\n";
  i("unholy_slots", s.slots);
  i("unholy_context_tokens", context_tokens);
  put("unholy_kv_usage_ratio", s.total_pages > 0 ? double(s.total_pages - s.free_pages) / s.total_pages : 0.0, "%.6f");
  put("unholy_prefill_sequence_seconds_total", s.prefill_seq_seconds, "%.3f");
  put("unholy_decode_sequence_seconds_total", s.decode_seq_seconds, "%.3f");
  put("unholy_first_token_seconds_total", s.first_token_seconds, "%.3f");
  u("unholy_first_tokens_total", s.first_tokens);
  return o;
}

}  // namespace unholy::server
