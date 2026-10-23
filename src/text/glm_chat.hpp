// GLM-5.3-Flash chat format: prompt rendering and completion parsing.
//
// glm5_render() is a port of the checkpoint's chat_template.jinja as transformers renders it (trim_blocks,
// lstrip_blocks): "[gMASK]<sop>", the reasoning-effort line, the tools block, <|user|> / <|assistant|> /
// <|observation|> / <|system|> turns, tool results re-ordered to the calls' order when the ids allow it.
// glm5_render_request() applies an OpenAI request on top (tool-call arguments given as JSON text become
// objects, as vLLM does; reasoning_effort from the request or chat_template_kwargs). The template always
// opens <think>: GLM-5.3-Flash has no non-thinking mode, only low / high / max effort.
//
// Glm5OutputParser follows vLLM's GLM-4.7 parser (vllm/parser/glm47_moe.py): reasoning until </think> or a
// tool call (and again after <think>), right-stripped; content up to the first tool call, stripped when a
// call follows, verbatim otherwise; text after the first call dropped;
// <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> with the values coerced to the
// tool schema's types (vllm/tool_parsers/utils.coerce_to_schema_type). Differences, all from streaming or
// deliberate: content that has text before a tool call keeps its leading whitespace (vLLM strips it once it
// has seen the call; a stream has already sent it), calls to tools the request did not declare are dropped
// without reordering the content around them, and a call cut off by the length limit gets valid JSON.
// Verified against transformers and vLLM by tools/ref/glm_chat_compare.py.
#pragma once
#include <string>
#include <string_view>
#include <vector>
#include "text/chat_format.hpp"

namespace unholy::text {

struct Glm5Options {
  std::string reasoning_effort;  // "low" or "high"; anything else renders as max (the template's rule)
  bool clear_thinking = false;   // empty <think></think> for assistant turns before the last user turn
  bool add_generation_prompt = true;
};

// messages / tools as the template receives them (tool-call arguments as objects); tools may be null.
// Image content blocks become the image placeholder; their URLs are appended to `images` in order.
// Throws ChatError on input the template cannot render.
std::string glm5_render(const json::Value& messages, const json::Value* tools, const Glm5Options& opt,
                        std::vector<std::string>* images = nullptr);

RenderedChat glm5_render_request(const json::Value& request);

// vLLM's coerce_to_schema_type(value, extract_types_from_schema(schema)). Exposed for tests.
json::Value glm5_coerce(const std::string& value, const json::Value& schema);

class Glm5OutputParser final : public ChatParser {
 public:
  // tools: the request's tool list (names to accept, schemas to coerce arguments with); may be null.
  explicit Glm5OutputParser(const json::Value* tools = nullptr);
  void feed(std::string_view text, ParsedDelta& out) override;
  void finish(ParsedDelta& out) override;
  bool saw_tool_calls() const override { return n_calls_ > 0; }

 private:
  enum class State { Reasoning, Content, ToolName, ToolArgs };
  void step(ParsedDelta& out, bool final);
  void text_out(std::string_view s, ParsedDelta& out);
  void emit_call(bool partial, ParsedDelta& out);

  State state_ = State::Reasoning;
  bool tools_seen_ = false;  // a call has started: content after it is dropped (vLLM)
  std::string buf_;          // not yet processed
  std::string hold_r_;       // trailing whitespace of the reasoning (dropped at the end: it is right-stripped)
  std::string hold_c_;       // whitespace of the content not yet followed by other text (dropped if a call follows)
  std::string call_ws_;      // hold_c_ as the first call started (back in the content if that call stays empty)
  bool content_text_ = false;  // the content has text other than whitespace
  std::string name_, args_;
  json::Value tools_;
  int n_calls_ = 0;
};

const ChatFormat& glm5_chat_format();

}  // namespace unholy::text
