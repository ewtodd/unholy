// DeepSeek-V4 chat format: prompt rendering and completion parsing.
//
// encode_messages() is a port of encoding_dsv4.encode_messages (the reference shipped with the
// checkpoint, text path), plus the one fix vLLM's copy carries: a system message that is last, or is
// followed by an assistant turn, opens a generation turn. render_request() applies vLLM's
// OpenAI-request semantics on top (tools become a leading system message, thinking defaults on,
// reasoning_effort mapping). Verified byte-for-byte against the Python reference by
// tools/ref/chat_compare.py.
//
// OutputParser is a streaming version of vLLM's DeepSeek-V4 parser state machine: reasoning until
// </think>, content, and DSML tool-call blocks, emitting each tool call when its invoke closes.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "text/chat_format.hpp"
#include "text/json.hpp"

namespace unholy::text {

inline constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
inline constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";
inline constexpr std::string_view kThinkStart = "<think>";
inline constexpr std::string_view kThinkEnd = "</think>";

struct EncodeOptions {
  bool thinking = false;          // thinking_mode "thinking" vs "chat"
  std::string reasoning_effort;   // "", "low", "high", "max" ("" = low)
  bool drop_thinking = true;
  bool add_bos = true;
};

// messages / context: JSON arrays of messages (roles system, developer, user, assistant, tool,
// latest_reminder). Throws ChatError on input the reference rejects.
// Image content blocks become the image placeholder token; their URLs (base64 sources as data: URLs)
// are appended to `images` in prompt order.
std::string encode_messages(const json::Value& messages, const EncodeOptions& opt,
                            const json::Value* context = nullptr, std::vector<std::string>* images = nullptr);

// An OpenAI chat-completions request body, interpreted as vLLM does for DeepSeek-V4.
RenderedChat render_request(const json::Value& request);

class OutputParser final : public ChatParser {
 public:
  // tools: the request's tool list (for unwrapping {"arguments": {...}} wrappers); may be null.
  OutputParser(bool thinking, const json::Value* tools = nullptr);
  void feed(std::string_view text, ParsedDelta& out) override;
  void finish(ParsedDelta& out) override;
  bool saw_tool_calls() const override { return n_calls_ > 0; }

 private:
  enum class State { Reasoning, Content, ToolPreamble, ToolName, ToolArgs, ToolBetween };
  void step(ParsedDelta& out, bool final);
  void emit_call(std::string_view raw_args, bool partial, ParsedDelta& out);

  State state_;
  std::string buf_;
  std::string name_;
  json::Value tools_;
  int n_calls_ = 0;
};

// The DSML parameter list of one invoke -> JSON arguments (vLLM's _dsml_arg_converter +
// _unwrap_wrapper_args). Exposed for tests.
std::string dsml_args_to_json(std::string_view raw_args, bool partial, const json::Value* tools,
                              std::string_view tool_name);

// render_request + OutputParser behind the server's ChatFormat interface.
const ChatFormat& dsv4_chat_format();

}  // namespace unholy::text
