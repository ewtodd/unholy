#include "text/chat.hpp"

#include <algorithm>
#include <map>

namespace unholy::text {

namespace {

using json::Value;

constexpr std::string_view kDsml = "｜DSML｜";
constexpr std::string_view kUser = "<｜User｜>";
constexpr std::string_view kAssistant = "<｜Assistant｜>";
constexpr std::string_view kLatestReminder = "<｜latest_reminder｜>";
constexpr std::string_view kImage = "<｜deepseek_image｜>";

const std::map<std::string, std::string, std::less<>> kTaskTokens = {
    {"action", "<｜action｜>"}, {"query", "<｜query｜>"},   {"authority", "<｜authority｜>"},
    {"domain", "<｜domain｜>"}, {"title", "<｜title｜>"},   {"read_url", "<｜read_url｜>"},
};

const std::map<std::string, std::string, std::less<>> kEffortPrompts = {
    {"low", ""},
    {"high",
     "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
     "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root "
     "cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial "
     "scenarios.\n"
     "Explicitly write out your entire deliberation process, documenting every intermediate step, considered "
     "alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"},
    {"max",
     "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
     "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively "
     "decompose the problem into its most fundamental components, trace every causal chain to its root, and "
     "resolve the underlying cause rather than any surface symptom.\n"
     "Do not stop reasoning until you have independently verified the solution from multiple angles and are "
     "certain that no assumption remains unchecked and no error remains undiscovered.\n\n"},
};

constexpr std::string_view kToolsHead = R"TPL(## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

)TPL";
constexpr std::string_view kToolsTail = R"TPL(

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
)TPL";

const Value kNullValue;

const Value& member(const Value& m, std::string_view key) {
  const Value* v = m.find(key);
  return v ? *v : kNullValue;
}

// Python `x or ""` for a field that is a string (or absent/null).
std::string str_or_empty(const Value& v) {
  if (v.is_string()) return v.str();
  if (v.is_null() || !v.truthy()) return "";
  return v.dump_py();  // non-string content: Python would str() it; JSON is the useful equivalent
}

std::string role_of(const Value& m) { return m.get_str("role"); }

bool has_task(const Value& m) {
  const Value* t = m.find("task");
  return t && !t->is_null();
}

// reasoning_content (the reference) or reasoning (vLLM's chat layer).
std::string reasoning_of(const Value& m) {
  if (const Value* r = m.find("reasoning_content"); r && r->is_string()) return r->str();
  if (const Value* r = m.find("reasoning"); r && r->is_string()) return r->str();
  return "";
}

std::string render_tools(const Value& tools) {
  std::string out(kToolsHead);
  bool first = true;
  for (const Value& t : tools.items()) {
    if (!first) out += '\n';
    first = false;
    const Value* fn = t.find("function");  // tools_from_openai_format
    (fn ? *fn : t).dump_py_to(out);
  }
  out += kToolsTail;
  return out;
}

std::string render_response_format(const Value& rf) {
  return "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" + rf.dump_py();
}

std::string encode_arguments_to_dsml(const Value& arguments) {
  // Strings are parsed as JSON (vLLM also takes an object); anything that is not an object is wrapped.
  Value args;
  if (arguments.is_object()) {
    args = arguments;
  } else {
    const std::string raw = arguments.is_string() ? arguments.str() : arguments.dump_py();
    try {
      args = json::parse(raw);
    } catch (const json::ParseError&) {
    }
    if (!args.is_object()) {
      args = Value::object();
      args.set("arguments", Value(raw));
    }
  }
  std::string out;
  bool first = true;
  for (const auto& [k, v] : args.members()) {
    if (!first) out += '\n';
    first = false;
    out += '<';
    out += kDsml;
    out += "parameter name=\"" + k + "\" string=\"" + (v.is_string() ? "true" : "false") + "\">";
    if (v.is_string()) out += v.str();
    else v.dump_py_to(out);
    out += "</";
    out += kDsml;
    out += "parameter>";
  }
  return out;
}

int find_last_user_index(const std::vector<Value>& msgs) {
  for (int i = int(msgs.size()) - 1; i >= 0; --i) {
    const std::string r = role_of(msgs[size_t(i)]);
    if (r == "user" || r == "developer") return i;
  }
  return -1;
}

std::string tool_result_text(const Value& content) {
  if (!content.is_array()) return str_or_empty(content);
  std::string out;
  bool first = true;
  for (const Value& b : content.items()) {
    if (!first) out += "\n\n";
    first = false;
    const std::string type = b.get_str("type");
    if (type == "text") out += b.get_str("text");
    else out += "[Unsupported " + type + "]";
  }
  return out;
}

std::string render_message(size_t index, const std::vector<Value>& msgs, bool thinking, bool drop_thinking,
                           const std::string& effort, int last_user_idx) {
  const Value& msg = msgs[index];
  const std::string role = role_of(msg);
  std::string prompt;

  const Value& tools = member(msg, "tools");
  const Value& response_format = member(msg, "response_format");
  const Value& tool_calls = member(msg, "tool_calls");
  const bool wo_eos = member(msg, "wo_eos").truthy();

  const auto ef = kEffortPrompts.find(effort.empty() ? "low" : effort);
  if (ef == kEffortPrompts.end()) throw ChatError("invalid reasoning effort '" + effort + "'");
  if (index == 0 && thinking) prompt += ef->second;

  if (role == "system") {
    prompt += str_or_empty(member(msg, "content"));
    if (tools.truthy()) prompt += "\n\n" + render_tools(tools);
    if (response_format.truthy()) prompt += "\n\n" + render_response_format(response_format);
  } else if (role == "developer") {
    const Value& content = member(msg, "content");
    if (!content.truthy()) throw ChatError("developer message without content");
    prompt += kUser;
    prompt += str_or_empty(content);
    if (tools.truthy()) prompt += "\n\n" + render_tools(tools);
    if (response_format.truthy()) prompt += "\n\n" + render_response_format(response_format);
  } else if (role == "user") {
    prompt += kUser;
    const Value& blocks = member(msg, "content_blocks");
    if (blocks.truthy()) {
      bool first = true;
      for (const Value& b : blocks.items()) {
        if (!first) prompt += "\n\n";
        first = false;
        const std::string type = b.get_str("type");
        if (type == "text") {
          prompt += str_or_empty(member(b, "text"));
        } else if (type == "tool_result") {
          prompt += "<tool_result>" + tool_result_text(member(b, "content")) + "</tool_result>";
        } else {
          prompt += "[Unsupported " + type + "]";
        }
      }
    } else {
      prompt += str_or_empty(member(msg, "content"));
    }
  } else if (role == "latest_reminder") {
    prompt += kLatestReminder;
    prompt += str_or_empty(member(msg, "content"));
  } else if (role == "tool") {
    throw ChatError("tool messages must be merged into user messages first");
  } else if (role == "assistant") {
    std::string tc_content;
    if (tool_calls.truthy()) {
      std::string calls;
      bool first = true;
      for (const Value& tc : tool_calls.items()) {
        if (!first) calls += '\n';
        first = false;
        const Value& fn = member(tc, "function");
        calls += '<';
        calls += kDsml;
        calls += "invoke name=\"" + str_or_empty(member(fn, "name")) + "\">\n" +
                 encode_arguments_to_dsml(member(fn, "arguments")) + "\n</";
        calls += kDsml;
        calls += "invoke>";
      }
      tc_content = "\n\n<";
      tc_content += kDsml;
      tc_content += "tool_calls>\n" + calls + "\n</";
      tc_content += kDsml;
      tc_content += "tool_calls>";
    }
    std::string thinking_part;
    const bool prev_has_task = index >= 1 && has_task(msgs[index - 1]);
    if (thinking && !prev_has_task && (!drop_thinking || int(index) > last_user_idx))
      thinking_part = reasoning_of(msg) + std::string(kThinkEnd);
    prompt += thinking_part + str_or_empty(member(msg, "content")) + tc_content;
    if (!wo_eos) prompt += kEos;
  } else {
    throw ChatError("unknown role '" + role + "'");
  }

  // Transition tokens, unless the next message is another turn of the prompt.
  if (index + 1 < msgs.size()) {
    const std::string next = role_of(msgs[index + 1]);
    if (next != "assistant" && next != "latest_reminder") return prompt;
  }
  if (has_task(msg)) {
    const Value& task = msg.at("task");
    const auto it = task.is_string() ? kTaskTokens.find(task.str()) : kTaskTokens.end();
    if (it == kTaskTokens.end()) throw ChatError("invalid task");
    if (it->first != "action") {
      prompt += it->second;
    } else {
      prompt += kAssistant;
      prompt += thinking ? kThinkStart : kThinkEnd;
      prompt += it->second;
    }
  } else if (role == "user" || role == "developer" ||
             (role == "system" && (index + 1 == msgs.size() || role_of(msgs[index + 1]) == "assistant"))) {
    prompt += kAssistant;
    if (!drop_thinking && thinking) prompt += kThinkStart;
    else if (drop_thinking && thinking && int(index) >= last_user_idx) prompt += kThinkStart;
    else prompt += kThinkEnd;
  }
  return prompt;
}

// Image blocks -> text blocks holding the placeholder token, collecting each image's URL in order
// (_process_image_blocks, recursing into tool results). Anthropic-style base64 sources become data URLs.
Value replace_images(const Value& blocks, std::vector<std::string>* images) {
  Value out = Value::array();
  for (const Value& b : blocks.items()) {
    if (!b.is_object()) {
      out.push_back(b);
      continue;
    }
    const std::string type = b.get_str("type");
    if (type == "image" || type == "image_url") {
      std::string url;
      if (type == "image_url") {
        const Value& iu = member(b, "image_url");
        url = iu.is_string() ? iu.str() : member(iu, "url").is_string() ? member(iu, "url").str() : "";
      } else if (const Value& src = member(b, "source"); src.is_object()) {
        if (member(src, "data").is_string())
          url = "data:" + (member(src, "media_type").is_string() ? member(src, "media_type").str() : std::string("image")) +
                ";base64," + member(src, "data").str();
        else if (member(src, "url").is_string())
          url = member(src, "url").str();
      } else if (member(b, "url").is_string()) {
        url = member(b, "url").str();
      } else if (member(b, "data").is_string()) {
        url = "data:image;base64," + member(b, "data").str();
      }
      if (url.empty()) throw ChatError("Image block does not contain a valid source");
      Value t = Value::object();
      t.set("type", "text");
      t.set("text", Value(std::string(kImage)));
      out.push_back(std::move(t));
      if (images) images->push_back(std::move(url));
    } else if (type == "tool_result" && member(b, "content").is_array()) {
      Value nb = b;
      nb.set("content", replace_images(member(b, "content"), images));
      out.push_back(std::move(nb));
    } else {
      if (type == "text" && b.get_str("text").find(kImage) != std::string::npos)
        throw ChatError("text block contains the image placeholder token");
      out.push_back(b);
    }
  }
  return out;
}

// Content lists -> content_blocks (and a "\n\n"-joined text content), as process_image_messages does.
std::vector<Value> preprocess(const Value& messages, std::vector<std::string>* images) {
  if (!messages.is_array()) throw ChatError("messages must be an array");
  std::vector<Value> out;
  for (const Value& m0 : messages.items()) {
    if (!m0.is_object()) throw ChatError("message must be an object");
    Value m = m0;
    for (const char* key : {"content", "reasoning_content"})
      if (const Value* v = m.find(key); v && v->is_string() && v->str().find(kImage) != std::string::npos)
        throw ChatError("message contains the image placeholder token");
    if (Value* c = m.find("content"); c && c->is_array() && !m.contains("content_blocks")) {
      Value blocks = std::move(*c);
      m.erase("content");
      m.set("content_blocks", std::move(blocks));
    }
    if (const Value* b0 = m.find("content_blocks"); b0 && b0->truthy()) {
      Value replaced = replace_images(*b0, images);
      m.set("content_blocks", std::move(replaced));
      const Value* blocks = m.find("content_blocks");
      std::string text;
      bool first = true;
      for (const Value& b : blocks->items()) {
        if (b.get_str("type") == "text") {
          if (!first) text += "\n\n";
          first = false;
          text += b.get_str("text");
        }
      }
      if (!member(m, "content").is_string()) m.set("content", Value(text));
    }
    out.push_back(std::move(m));
  }
  return out;
}

std::vector<Value> merge_tool_messages(std::vector<Value> msgs) {
  std::vector<Value> merged;
  for (Value& msg : msgs) {
    const std::string role = role_of(msg);
    const bool prev_user = !merged.empty() && role_of(merged.back()) == "user" && merged.back().contains("content_blocks");
    if (role == "tool") {
      Value block = Value::object();
      block.set("type", "tool_result");
      block.set("tool_use_id", msg.contains("tool_call_id") ? msg.at("tool_call_id") : Value(""));
      block.set("content", msg.contains("content") ? msg.at("content") : Value(""));
      if (prev_user) {
        merged.back()["content_blocks"].push_back(std::move(block));
      } else {
        Value u = Value::object();
        u.set("role", "user");
        Value blocks = Value::array();
        blocks.push_back(std::move(block));
        u.set("content_blocks", std::move(blocks));
        merged.push_back(std::move(u));
      }
    } else if (role == "user") {
      Value blocks;
      if (const Value* cb = msg.find("content_blocks"); cb && !cb->is_null()) {
        blocks = *cb;
      } else {
        Value t = Value::object();
        t.set("type", "text");
        t.set("text", msg.contains("content") ? msg.at("content") : Value(""));
        blocks = Value::array();
        blocks.push_back(std::move(t));
      }
      if (prev_user && !has_task(merged.back())) {
        for (Value& b : blocks.items()) merged.back()["content_blocks"].push_back(std::move(b));
      } else {
        msg.set("content_blocks", std::move(blocks));
        merged.push_back(std::move(msg));
      }
    } else {
      merged.push_back(std::move(msg));
    }
  }
  return merged;
}

void sort_tool_results_by_call_order(std::vector<Value>& msgs) {
  std::map<std::string, int> order;
  for (Value& msg : msgs) {
    const std::string role = role_of(msg);
    if (role == "assistant" && member(msg, "tool_calls").truthy()) {
      order.clear();
      int idx = 0;
      for (const Value& tc : msg.at("tool_calls").items()) {
        std::string id = tc.get_str("id");
        if (id.empty()) id = member(tc, "function").get_str("id");
        if (!id.empty()) order[id] = idx;
        ++idx;
      }
    } else if (role == "user" && member(msg, "content_blocks").truthy()) {
      auto& blocks = msg["content_blocks"].items();
      std::vector<Value> tool_blocks;
      for (const Value& b : blocks)
        if (b.get_str("type") == "tool_result") tool_blocks.push_back(b);
      if (tool_blocks.size() > 1 && !order.empty()) {
        const auto key = [&](const Value& b) {
          const auto it = order.find(b.get_str("tool_use_id"));
          return it == order.end() ? 0 : it->second;
        };
        std::stable_sort(tool_blocks.begin(), tool_blocks.end(),
                         [&](const Value& a, const Value& b) { return key(a) < key(b); });
        size_t k = 0;
        for (Value& b : blocks)
          if (b.get_str("type") == "tool_result") b = tool_blocks[k++];
      }
    }
  }
}

std::vector<Value> drop_thinking_messages(const std::vector<Value>& msgs) {
  const int last_user = find_last_user_index(msgs);
  std::vector<Value> out;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const std::string role = role_of(msgs[i]);
    if (role == "user" || role == "system" || role == "tool" || role == "latest_reminder" ||
        role == "direct_search_results" || int(i) >= last_user) {
      out.push_back(msgs[i]);
    } else if (role == "assistant") {
      Value m = msgs[i];
      m.erase("reasoning_content");
      m.erase("reasoning");
      out.push_back(std::move(m));
    }
  }
  return out;
}

}  // namespace

std::string encode_messages(const Value& messages_in, const EncodeOptions& opt, const Value* context_in,
                            std::vector<std::string>* images) {
  std::vector<Value> messages = merge_tool_messages(preprocess(messages_in, images));
  std::vector<Value> context;
  // The context's images are not returned (as in the reference).
  if (context_in && context_in->truthy()) context = merge_tool_messages(preprocess(*context_in, nullptr));

  std::vector<Value> full = context;
  full.insert(full.end(), messages.begin(), messages.end());
  sort_tool_results_by_call_order(full);
  // Sorting the context part on its own (as the reference does) gives the same result: the order
  // only depends on earlier messages.

  std::string prompt = opt.add_bos && context.empty() ? std::string(kBos) : std::string();

  bool drop = opt.drop_thinking;
  for (const Value& m : full)
    if (member(m, "tools").truthy()) drop = false;

  size_t n_render, ctx_len;
  if (opt.thinking && drop) {
    full = drop_thinking_messages(full);
    n_render = full.size() - drop_thinking_messages(context).size();
    ctx_len = full.size() - n_render;
  } else {
    n_render = messages.size();
    ctx_len = context.size();
  }
  const int last_user = find_last_user_index(full);
  for (size_t i = 0; i < n_render; ++i)
    prompt += render_message(i + ctx_len, full, opt.thinking, drop, opt.reasoning_effort, last_user);
  return prompt;
}

RenderedChat render_request(const Value& req) {
  const Value& kwargs = member(req, "chat_template_kwargs");
  bool thinking = true;
  if (kwargs.contains("thinking") || kwargs.contains("enable_thinking"))
    thinking = member(kwargs, "thinking").truthy() || member(kwargs, "enable_thinking").truthy();

  Value messages = member(req, "messages");
  if (!messages.is_array()) throw ChatError("'messages' must be an array");
  const Value& tools = member(req, "tools");
  if (tools.is_array() && tools.size() > 0) {
    Value sys = Value::object();
    sys.set("role", "system");
    sys.set("tools", tools);
    messages.items().insert(messages.items().begin(), std::move(sys));
  }

  const Value* effort_v = req.find("reasoning_effort");
  if (!effort_v || effort_v->is_null()) effort_v = kwargs.find("reasoning_effort");
  EncodeOptions opt;
  if (!effort_v || !effort_v->is_string()) {
    opt.reasoning_effort = thinking ? "high" : "";
  } else {
    const std::string& e = effort_v->str();
    if (e == "none") {
      thinking = false;
    } else if (e == "max") {
      opt.reasoning_effort = "max";
    } else if (e == "low" || e == "minimal" || e == "medium") {
      opt.reasoning_effort = "low";
    } else {
      opt.reasoning_effort = "high";
    }
  }
  opt.thinking = thinking;
  if (const Value* d = kwargs.find("drop_thinking")) opt.drop_thinking = d->truthy();
  RenderedChat r;
  r.prompt = encode_messages(messages, opt, nullptr, &r.images);
  r.thinking = thinking;
  return r;
}

// ---- completion parsing -------------------------------------------------------------------------

namespace {

const std::string kToolStart = std::string("<") + std::string(kDsml) + "tool_calls>";
const std::string kToolStartNl = "\n\n" + kToolStart;
const std::string kToolEnd = std::string("</") + std::string(kDsml) + "tool_calls>";
const std::string kInvokePrefix = std::string("<") + std::string(kDsml) + "invoke name=\"";
const std::string kInvokeNameEnd = "\">";
const std::string kInvokeEnd = std::string("</") + std::string(kDsml) + "invoke>";
const std::string kParamOpen = std::string("<") + std::string(kDsml) + "parameter";
const std::string kParamClose = std::string("</") + std::string(kDsml) + "parameter>";

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// One parameter at s[p] (just past "<｜DSML｜parameter"): \s+name="([^"]+)"\s+string="(true|false)">
// Returns the position after "\">" or npos.
size_t parse_param_head(std::string_view s, size_t p, std::string& name, bool& is_str) {
  const size_t start = p;
  while (p < s.size() && is_ws(s[p])) ++p;
  if (p == start || s.substr(p, 6) != "name=\"") return std::string_view::npos;
  p += 6;
  const size_t q = s.find('"', p);
  if (q == std::string_view::npos || q == p) return std::string_view::npos;
  name = std::string(s.substr(p, q - p));
  p = q + 1;
  const size_t ws = p;
  while (p < s.size() && is_ws(s[p])) ++p;
  if (p == ws) return std::string_view::npos;
  if (s.substr(p, 14) == "string=\"true\">") {
    is_str = true;
    return p + 14;
  }
  if (s.substr(p, 15) == "string=\"false\">") {
    is_str = false;
    return p + 15;
  }
  return std::string_view::npos;
}

const Value* tool_properties(const Value* tools, std::string_view name) {
  if (!tools || !tools->is_array()) return nullptr;
  for (const Value& t : tools->items()) {
    const Value* fn = t.find("function");
    const Value& f = fn ? *fn : t;
    if (f.get_str("name") == name)
      if (const Value* params = f.find("parameters"))
        if (const Value* props = params->find("properties"); props && props->is_object()) return props;
  }
  return nullptr;
}

}  // namespace

std::string dsml_args_to_json(std::string_view raw, bool partial, const Value* tools, std::string_view tool_name) {
  Value params = Value::object();
  const auto set_param = [&](std::string name, std::string_view value, bool is_str, bool lenient) {
    if (is_str) {
      params.set(std::move(name), Value(std::string(value)));
      return;
    }
    try {
      params.set(name, json::parse(value));
    } catch (const json::ParseError&) {
      if (lenient) params.set(std::move(name), Value(std::string(value)));
    }
  };
  size_t pos = 0, last_end = 0;
  for (;;) {
    const size_t open = raw.find(kParamOpen, pos);
    if (open == std::string_view::npos) break;
    std::string name;
    bool is_str = false;
    const size_t body = parse_param_head(raw, open + kParamOpen.size(), name, is_str);
    const size_t close = body == std::string_view::npos ? body : raw.find(kParamClose, body);
    if (close == std::string_view::npos) {
      pos = open + 1;
      continue;
    }
    set_param(std::move(name), raw.substr(body, close - body), is_str, true);
    pos = last_end = close + kParamClose.size();
  }
  if (partial) {
    // A parameter still being written: its value runs to the end of the text.
    for (size_t open = raw.find(kParamOpen, last_end); open != std::string_view::npos;
         open = raw.find(kParamOpen, open + 1)) {
      std::string name;
      bool is_str = false;
      const size_t body = parse_param_head(raw, open + kParamOpen.size(), name, is_str);
      if (body == std::string_view::npos) continue;
      set_param(std::move(name), raw.substr(body), is_str, false);
      break;
    }
  }

  // {"arguments": {...}} / {"input": {...}} wrappers around the real parameters.
  if (const Value* props = tool_properties(tools, tool_name); props && props->size() > 0 && params.size() == 1) {
    const std::string& key = params.members()[0].first;
    if ((key == "arguments" || key == "input") && !props->contains(key)) {
      Value inner = params.members()[0].second;
      bool ok = true;
      if (inner.is_string()) {
        try {
          inner = json::parse(inner.str());
        } catch (const json::ParseError&) {
          ok = false;
        }
      }
      if (ok && inner.is_object()) {
        bool subset = true;
        for (const auto& [k, v] : inner.members()) subset = subset && props->contains(k);
        if (subset) return inner.dump_py();
      }
    }
  }
  return params.dump_py();
}

OutputParser::OutputParser(bool thinking, const Value* tools)
    : state_(thinking ? State::Reasoning : State::Content), tools_(tools ? *tools : Value()) {}

void OutputParser::emit_call(std::string_view raw_args, bool partial, ParsedDelta& out) {
  out.tool_calls.push_back({name_, dsml_args_to_json(raw_args, partial, &tools_, name_)});
  ++n_calls_;
}

void OutputParser::feed(std::string_view text, ParsedDelta& out) {
  buf_ += text;
  step(out, false);
}

void OutputParser::finish(ParsedDelta& out) {
  step(out, true);
  if (state_ == State::Reasoning) out.reasoning += buf_;
  else if (state_ == State::Content) out.content += buf_;
  else if (state_ == State::ToolArgs) emit_call(buf_, true, out);
  buf_.clear();
}

void OutputParser::step(ParsedDelta& out, bool final) {
  struct Term {
    const std::string* text;
    int action;
  };
  for (;;) {
    // Terminals of the current state; the earliest occurrence wins (longest at the same position).
    std::vector<Term> terms;
    static const std::string think_start(kThinkStart), think_end(kThinkEnd), eos(kEos);
    switch (state_) {
      case State::Reasoning:
        terms = {{&think_end, 1}, {&think_start, 0}, {&kToolStart, 2}, {&eos, 0}};
        break;
      case State::Content:
        terms = {{&kToolStartNl, 2}, {&kToolStart, 2}, {&think_start, 3}, {&think_end, 0}, {&eos, 0}};
        break;
      case State::ToolPreamble:
      case State::ToolBetween:
        terms = {{&kInvokePrefix, 4}, {&kToolEnd, 1}};
        break;
      case State::ToolName:
        terms = {{&kInvokeNameEnd, 5}};
        break;
      case State::ToolArgs:
        terms = {{&kInvokeEnd, 6}, {&kToolEnd, 7}};
        break;
    }
    size_t best = std::string::npos, best_len = 0;
    int action = -1;
    for (const Term& t : terms) {
      const size_t p = buf_.find(*t.text);
      if (p != std::string::npos && (p < best || (p == best && t.text->size() > best_len))) {
        best = p;
        best_len = t.text->size();
        action = t.action;
      }
    }
    const bool text_state = state_ == State::Reasoning || state_ == State::Content;
    std::string& sink = state_ == State::Reasoning ? out.reasoning : out.content;
    if (best == std::string::npos) {
      if (text_state && !final) {
        // Hold back a suffix that may be the start of a terminal.
        size_t hold = 0;
        for (const Term& t : terms)
          for (size_t k = std::min(t.text->size() - 1, buf_.size()); k > hold; --k)
            if (buf_.compare(buf_.size() - k, k, *t.text, 0, k) == 0) {
              hold = k;
              break;
            }
        sink.append(buf_, 0, buf_.size() - hold);
        buf_.erase(0, buf_.size() - hold);
      } else if (state_ == State::ToolPreamble || state_ == State::ToolBetween) {
        // Text between invokes is dropped, but keep a possible partial terminal.
        const size_t keep = std::min(buf_.size(), std::max(kInvokePrefix.size(), kToolEnd.size()) - 1);
        buf_.erase(0, buf_.size() - keep);
      }
      return;
    }
    const std::string before = buf_.substr(0, best);
    buf_.erase(0, best + best_len);
    if (text_state) sink += before;
    switch (action) {
      case 0: break;                                           // absorbed
      case 1: state_ = State::Content; break;                  // </think> or </tool_calls>
      case 2: state_ = State::ToolPreamble; break;             // tool block opens
      case 3: state_ = State::Reasoning; break;                // <think> in content
      case 4: state_ = State::ToolName; break;                 // invoke name="
      case 5: name_ = before; state_ = State::ToolArgs; break;  // ">
      case 6: emit_call(before, false, out); state_ = State::ToolBetween; break;
      case 7: emit_call(before, false, out); state_ = State::Content; break;
    }
  }
}

}  // namespace unholy::text

namespace unholy::text {

namespace {
class Dsv4ChatFormat final : public ChatFormat {
 public:
  RenderedChat render(const json::Value& request) const override { return render_request(request); }
  std::unique_ptr<ChatParser> parser(bool thinking, const json::Value* tools) const override {
    return std::make_unique<OutputParser>(thinking, tools);
  }
};
}  // namespace

const ChatFormat& dsv4_chat_format() {
  static const Dsv4ChatFormat format;
  return format;
}

}  // namespace unholy::text
