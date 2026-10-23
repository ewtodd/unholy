// A model family's chat format as the OpenAI server uses it: an OpenAI chat-completions request rendered
// to prompt text, and a streaming parser that splits the completion into reasoning, content and tool calls.
// Each family implements it next to its template (DeepSeek-V4: text/chat.hpp).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "text/json.hpp"

namespace unholy::text {

// Input the family's reference template rejects (the server answers 400).
struct ChatError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct RenderedChat {
  std::string prompt;
  bool thinking = false;  // the completion starts inside the reasoning block
  std::vector<std::string> images;  // image URLs, one per placeholder token in the prompt, in order
};

struct ToolCall {
  std::string name;
  std::string arguments;  // JSON object text
};

struct ParsedDelta {
  std::string reasoning;
  std::string content;
  std::vector<ToolCall> tool_calls;
  bool empty() const { return reasoning.empty() && content.empty() && tool_calls.empty(); }
};

class ChatParser {
 public:
  virtual ~ChatParser() = default;
  virtual void feed(std::string_view text, ParsedDelta& out) = 0;
  // End of stream: releases held-back text and any tool call cut off by the length limit.
  virtual void finish(ParsedDelta& out) = 0;
  virtual bool saw_tool_calls() const = 0;
};

class ChatFormat {
 public:
  virtual ~ChatFormat() = default;
  virtual RenderedChat render(const json::Value& request) const = 0;
  // tools: the request's tool list (argument handling may depend on it); may be null.
  virtual std::unique_ptr<ChatParser> parser(bool thinking, const json::Value* tools) const = 0;
};

}  // namespace unholy::text
