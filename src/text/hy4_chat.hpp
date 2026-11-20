// Hy4-flash (Tencent, model_type hy_v4_flash) chat format: prompt rendering and completion parsing.
//
// hy4_render() is a port of the HF release's chat_template.jinja as transformers renders it (trim_blocks,
// lstrip_blocks, tojson = json.dumps(ensure_ascii=False)): <｜hy_start:opensource｜>role<｜hy_middle:opensource｜>
// ... <｜hy_end:opensource｜> turns, a system turn always first (synthesized when the request has none) that
// carries the tools block and "<｜reasoning_mode:opensource｜>reasoning_effort:<effort>", consecutive tool
// messages merged into one turn (results re-ordered to the calls' order when the ids allow it), assistant
// turns as "<think></think>content" (the reasoning kept for turns after the last user message, or for all
// turns when tools are given), and the generation prompt "<think>" (high) or "<think></think>" (no_think).
// The AngelSlim GGUF's spelling of the markers (":6124c78e") and its low / max efforts are not used: this is
// the HF release's template. hy4_render_request() applies an OpenAI request on top (tool-call arguments
// given as JSON text become objects, as vLLM does; reasoning_effort from the request or chat_template_kwargs,
// "high" or "no_think", else ChatError; the kwargs preserved_thinking, raw_last_assistant, add_vision_id and
// fallback_strategy are passed to the template as well). Image blocks become the template's
// image_start/image_pad/image_end placeholder, their URLs go to RenderedChat::images in order.
//
// Hy4OutputParser follows vLLM's hy_v4 reasoning parser and strict hy_v4 tool parser (vllm/reasoning/
// hy_v4_reasoning_parser.py, vllm/tool_parsers/hy_v4_tool_parser.py, vLLM 0.29.0): with thinking on, the
// completion is reasoning (verbatim, not stripped) until </think>, content after it; with no_think it is all
// content. In the content, text up to <tool_calls> is content (whitespace-only content before the calls is
// dropped), the first <tool_calls>...</tool_calls> block is the calls, text after it is dropped. A block is
// validated all-or-nothing as vLLM's strict mode does (equal <tool_call> / </tool_call> counts, a non-empty
// name, balanced <arg_key> / <arg_value> tags, every argument <arg_key>k</arg_key><arg_value>v</arg_value>
// with nothing in between, nothing else left over): on any failure the whole content is returned as text,
// markup included, and stays text to the end. Argument values are typed by the tool schema exactly as
// vLLM's _parse_value does (bool, int, number, json.loads, string, then json.loads / ast.literal_eval).
// Calls are emitted when </tool_calls> arrives, as vLLM's non-streaming path decides them.
//
// Differences from vLLM, all from streaming or deliberate:
//  - text before a "<think>" in the reasoning is kept (vLLM drops it; a stream has already sent it); the
//    first <think> is dropped, later ones stay text. Anything after </think> is never read as reasoning.
//  - a block cut off by the length limit (no </tool_calls>) is not returned as text but salvaged: the calls
//    that closed are emitted and the open one gets valid JSON (its finished arguments plus the value being
//    written, as text); a cut block with no usable call is returned as text.
//  - a call whose name is neither declared by the request's tools nor valid ([A-Za-z0-9_-]{1,64}) makes the
//    block fail like a malformed one (vLLM would emit it): the markup goes back as content, ParsedDelta::log
//    says why. Calls to undeclared tools with a valid name are emitted, as vLLM does.
//  - ast.literal_eval of an argument value is a subset (constants, numbers, plain strings, lists, tuples,
//    dicts); sets / bytes / complex numbers make the block fail as vLLM's json.dumps error does; int() and
//    float() accept ASCII digits only.
// Verified against transformers and vLLM by tools/ref/hy4_chat_compare.py; tools/text_test/hy4_parse_test.cpp
// covers the differences, which that comparison cannot.
#pragma once
#include <string>
#include <string_view>
#include <vector>
#include "text/chat_format.hpp"

namespace unholy::text {

struct Hy4Options {
  std::string reasoning_effort = "high";  // "high" or "no_think"; anything else is ChatError
  bool preserved_thinking_set = false;    // template default: on exactly when tools are given
  bool preserved_thinking = false;
  bool raw_last_assistant = false;        // the last assistant message stays open, without think tags
  bool add_vision_id = false;             // "Picture N: " before each image
  bool fallback_retry = false;            // fallback_strategy == 'reasoning_toolcall_retry'
  bool add_generation_prompt = true;
};

// messages / tools as the template receives them (tool-call arguments as objects); tools may be null.
// *thinking: the prompt ends inside the reasoning block. Throws ChatError on input the template rejects.
std::string hy4_render(const json::Value& messages, const json::Value* tools, const Hy4Options& opt,
                       std::vector<std::string>* images = nullptr, bool* thinking = nullptr);

RenderedChat hy4_render_request(const json::Value& request);

// vLLM's _parse_value(value, function_name, key, tools) as a JSON value (throws std::runtime_error where
// vLLM's json.dumps would fail). Exposed for tests.
json::Value hy4_parse_value(const std::string& value, const json::Value* property_schema);

class Hy4OutputParser final : public ChatParser {
 public:
  // thinking: the completion starts inside the reasoning block. tools: the request's tool list; may be null.
  explicit Hy4OutputParser(bool thinking, const json::Value* tools = nullptr);
  void feed(std::string_view text, ParsedDelta& out) override;
  void finish(ParsedDelta& out) override;
  bool saw_tool_calls() const override { return n_calls_ > 0; }

 private:
  enum class State { Reasoning, Content, Block, Raw, Done };
  void step(ParsedDelta& out, bool final);
  void content_out(std::string_view s, ParsedDelta& out);
  void end_block(bool cut, ParsedDelta& out);

  State state_;
  bool think_seen_ = false;   // the first <think> went by
  bool stray_end_ = false;    // a </tool_calls> came before any <tool_calls>: vLLM's strict mode fails
  bool content_text_ = false;  // the content has text other than whitespace
  std::string buf_;            // not yet processed
  std::string hold_c_;         // whitespace-only content so far (dropped if the calls follow)
  std::string block_;          // text of the open <tool_calls> block
  json::Value tools_;
  int n_calls_ = 0;
};

const ChatFormat& hy4_chat_format();

}  // namespace unholy::text
