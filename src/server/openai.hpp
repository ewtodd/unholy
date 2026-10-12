// OpenAI-compatible HTTP frontend over the engine: /v1/chat/completions, /v1/completions (both with
// SSE streaming), /v1/models, /tokenize, /detokenize, /health, /metrics.
//
// Each HTTP request runs on its own server thread: it renders and tokenizes the prompt, submits it,
// and turns the engine's token deliveries into text (streaming detokenizer -> stop strings -> the
// DSV4 output parser for chat). A client that disconnects or a matched stop string aborts the
// engine request.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/engine.hpp"
#include "vision/image.hpp"
#include "text/tokenizer.hpp"

namespace unholy::server {

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8000;
  std::string model_name = "deepseek-v4-flash";
  std::string api_key;  // empty: no auth
  int threads = 64;     // concurrent HTTP requests (each stream holds one)
  int default_max_tokens = 0;  // when a request gives none; 0 = until the context limit
  // Images in chat messages: the vision tower (encode one image's block rows; null: images are refused),
  // the placeholder token, the model's vocabulary size (image tokens are vocab + type) and preprocessing.
  std::function<void(const vision::Patches&, const vision::ImageBlock&, std::vector<uint16_t>&)> encode_image;
  int32_t image_token = -1, vocab = 0;
  vision::VisionConfig vision;
  // Extra plain-text endpoints (path -> handler), e.g. the routing histogram.
  std::vector<std::pair<std::string, std::function<std::string()>>> extra_get, extra_post;
};

class OpenAIServer {
 public:
  OpenAIServer(engine::Engine& eng, const text::Tokenizer& tok, ServerConfig cfg);
  ~OpenAIServer();
  // Blocks until stop().
  void run();
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Stop-string matching over a text stream: emits text that can no longer be the start of a stop
// string, and reports the first stop string found (the text before it is emitted, the rest dropped).
class StopMatcher {
 public:
  explicit StopMatcher(std::vector<std::string> stops);
  // Appends text; returns true once a stop string has been seen (out gets the text before it).
  bool feed(std::string_view text, std::string& out);
  void finish(std::string& out);  // end of stream: the held-back tail
  bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  size_t hold_ = 0;
  std::string buf_;
  bool stopped_ = false;
};

}  // namespace unholy::server
