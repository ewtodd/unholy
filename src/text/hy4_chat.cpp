#include "text/hy4_chat.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <set>

namespace unholy::text {

namespace {

using json::Value;

// ---- Python string semantics ----------------------------------------------------------------------

// One code point at s[i] (invalid bytes count as one replacement character), its byte length in n.
uint32_t utf8_at(std::string_view s, size_t i, size_t& n) {
  const auto c = static_cast<unsigned char>(s[i]);
  int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
  if (len == 0 || i + size_t(len) > s.size()) {
    n = 1;
    return 0xFFFD;
  }
  uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
  for (int k = 1; k < len; ++k) {
    const auto d = static_cast<unsigned char>(s[i + size_t(k)]);
    if ((d >> 6) != 2) {
      n = 1;
      return 0xFFFD;
    }
    cp = (cp << 6) | (d & 0x3F);
  }
  n = size_t(len);
  return cp;
}

// str.isspace().
bool py_space(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

size_t lead_space(std::string_view s) {
  size_t i = 0, n = 0;
  while (i < s.size() && py_space(utf8_at(s, i, n))) i += n;
  return i;
}

size_t trail_space(std::string_view s) {  // byte offset where the trailing whitespace starts
  size_t end = 0, i = 0, n = 0;
  while (i < s.size()) {
    const bool ws = py_space(utf8_at(s, i, n));
    i += n;
    if (!ws) end = i;
  }
  return end;
}

std::string_view py_strip(std::string_view s) {
  s = s.substr(0, trail_space(s));
  return s.substr(lead_space(s));
}

std::string ascii_lower(std::string_view s) {
  std::string out(s);
  for (char& c : out)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return out;
}

// str.isprintable() for one code point (the common categories; unassigned code points count as printable).
bool py_printable(uint32_t cp) {
  if (cp < 0x20 || cp == 0x7f) return false;
  if (cp < 0x7f) return true;
  if (cp <= 0xa0 || cp == 0xad) return false;
  if (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200f) || (cp >= 0x2028 && cp <= 0x202f) ||
      (cp >= 0x205f && cp <= 0x206f) || cp == 0x3000 || cp == 0xfeff || (cp >= 0xd800 && cp <= 0xf8ff) ||
      (cp >= 0xfff0 && cp <= 0xfffb) || cp == 0xfffe || cp == 0xffff || (cp >= 0xe0000 && cp <= 0xe007f) ||
      cp >= 0xf0000)
    return false;
  return true;
}

void hex_escape(std::string& out, uint32_t cp) {
  char buf[16];
  if (cp < 0x100) std::snprintf(buf, sizeof buf, "\\x%02x", cp);
  else if (cp < 0x10000) std::snprintf(buf, sizeof buf, "\\u%04x", cp);
  else std::snprintf(buf, sizeof buf, "\\U%08x", cp);
  out += buf;
}

// repr(str)
void py_repr_str(std::string& out, std::string_view s) {
  const bool single = s.find('\'') != std::string_view::npos && s.find('"') == std::string_view::npos;
  const char q = single ? '"' : '\'';
  out += q;
  for (size_t i = 0, n = 0; i < s.size(); i += n) {
    const uint32_t cp = utf8_at(s, i, n);
    if (cp == uint32_t(q) || cp == '\\') {
      out += '\\';
      out += char(cp);
    } else if (cp == '\n') out += "\\n";
    else if (cp == '\r') out += "\\r";
    else if (cp == '\t') out += "\\t";
    else if (!py_printable(cp)) hex_escape(out, cp);
    else out.append(s.substr(i, n));
  }
  out += q;
}

// repr(value) for a value json.loads made.
void py_repr_to(std::string& out, const Value& v) {
  switch (v.type()) {
    case Value::Type::Null: out += "None"; break;
    case Value::Type::Bool: out += v.as_bool() ? "True" : "False"; break;
    case Value::Type::Number:
      if (v.number_is_int()) out += v.str() == "-0" ? "0" : v.str();
      else out += json::py_float_repr(v.as_double());
      break;
    case Value::Type::String: py_repr_str(out, v.str()); break;
    case Value::Type::Array: {
      out += '[';
      for (size_t k = 0; k < v.items().size(); ++k) {
        if (k) out += ", ";
        py_repr_to(out, v.items()[k]);
      }
      out += ']';
      break;
    }
    case Value::Type::Object: {
      out += '{';
      bool first = true;
      for (const auto& [k, x] : v.members()) {
        if (!first) out += ", ";
        first = false;
        py_repr_str(out, k);
        out += ": ";
        py_repr_to(out, x);
      }
      out += '}';
      break;
    }
  }
}

// {{ v }} / v | string: Python str(); an undefined value prints as nothing.
std::string py_str(const Value* v) {
  if (!v) return "";
  if (v->is_string()) return v->str();
  std::string out;
  py_repr_to(out, *v);
  return out;
}

// ---- Jinja value semantics ------------------------------------------------------------------------

const Value* attr(const Value* v, std::string_view key) { return v && v->is_object() ? v->find(key) : nullptr; }
bool truthy(const Value* v) { return v && v->truthy(); }
bool is_str(const Value* v, std::string_view s) { return v && v->is_string() && v->str() == s; }

bool jinja_eq(const Value* a, const Value* b) {
  if (!a || !b) return !a && !b;
  if (a->is_string() && b->is_string()) return a->str() == b->str();
  return a->dump_py() == b->dump_py();
}

// ---- the template -------------------------------------------------------------------------------

const std::string kSuffix = ":opensource";
std::string tag(const char* name) { return std::string("<") + name + kSuffix + ">"; }
std::string tag_end(const char* name) { return std::string("</") + name + kSuffix + ">"; }
std::string bar_tag(const char* name) { return std::string("<\xEF\xBD\x9C") + name + kSuffix + "\xEF\xBD\x9C>"; }

const std::string kHyStart = bar_tag("hy_start"), kHyMiddle = bar_tag("hy_middle"), kHyEnd = bar_tag("hy_end");
const std::string kImageStart = bar_tag("hy_image\xE2\x96\x81start"), kImagePad = bar_tag("hy_image\xE2\x96\x81pad"),
                  kImageEnd = bar_tag("hy_image\xE2\x96\x81" "end");
const std::string kVideoStart = bar_tag("hy_video\xE2\x96\x81start"), kVideoPad = bar_tag("hy_video\xE2\x96\x81pad"),
                  kVideoEnd = bar_tag("hy_video\xE2\x96\x81" "end");
const std::string kThinkOpen = tag("think"), kThinkClose = tag_end("think");
const std::string kCallsOpen = tag("tool_calls"), kCallsClose = tag_end("tool_calls");
const std::string kCallOpen = tag("tool_call"), kCallClose = tag_end("tool_call");
const std::string kKeyOpen = tag("arg_key"), kKeyClose = tag_end("arg_key");
const std::string kValOpen = tag("arg_value"), kValClose = tag_end("arg_value");
const std::string kRespOpen = tag("tool_response"), kRespClose = tag_end("tool_response");
const std::string kReasoningMode = bar_tag("reasoning_mode");

struct Item {  // one message of the template's merged list
  const Value* msg = nullptr;                // null for a cluster of tool messages
  std::vector<const Value*> cluster;
  const Value* role() const { return msg ? attr(msg, "role") : nullptr; }
  bool is_role(std::string_view r) const { return msg ? is_str(attr(msg, "role"), r) : r == "tool"; }
};

struct Renderer {
  const Value* tools;  // null when the template's `tools` is falsy
  const Hy4Options& opt;
  std::string effort;
  std::vector<std::string>* images;
  std::string out;
  int image_count = 0, video_count = 0;
  std::vector<const Value*> prev_ids;  // prev_ns.tool_call_ids

  static const Value* unwrap(const Value& tool) {
    if (tool.is_object()) {
      if (const Value* f = tool.find("function")) return f;
    }
    return &tool;
  }

  void tool_to_json(const Value& tool) {
    if (!tool.is_object()) throw ChatError("a tool must be an object");
    out += '{';
    bool first_member = true;
    for (const auto& [k, v] : tool.members()) {
      if (k == "defer_loading" || k == "strict") continue;
      if (!first_member) out += ", ";
      first_member = false;
      out += '"';
      out += k;
      out += "\": ";
      v.dump_py_to(out);
    }
    out += '}';
  }

  void tools_prompt() {
    out += "# Tools\n\nYou may call one or more functions to assist with the user query.";
    out += "\n\nYou are provided with function signatures within <tools></tools> XML tags:";
    out += "\n<tools>\n";
    bool first = true;
    for (const Value& tool : tools->items()) {
      const Value* t = unwrap(tool);
      const Value* d = attr(t, "defer_loading");
      if (d && d->truthy()) continue;
      if (!first) out += '\n';
      first = false;
      tool_to_json(*t);
    }
    out += "\n</tools>\n\n";
    out += "For function call returns, you should first print " + kCallsOpen;
    out += "\nFor each function call, you should return object like:\n";
    out += kCallOpen + "{function-name}";
    out += kKeyOpen + "{arg-key-1}" + kKeyClose + kValOpen + "{arg-value-1}" + kValClose;
    out += kKeyOpen + "{arg-key-2}" + kKeyClose + kValOpen + "{arg-value-2}" + kValClose;
    out += "...";
    out += kCallClose;
    out += "\nAt the end of function call returns, you should print " + kCallsClose;
  }

  // One element of render_content's loop.
  void render_item(const Value& item) {
    if (item.is_object()) {
      const Value* type = item.find("type");
      if (is_str(type, "text")) {
        out += py_str(item.find("text"));
        return;
      }
      if (item.find("image") || item.find("image_url") || is_str(type, "image") || is_str(type, "image_url")) {
        ++image_count;
        if (opt.add_vision_id) out += "Picture " + std::to_string(image_count) + ": ";
        out += kImageStart + kImagePad + kImageEnd;
        if (images) {
          const Value* u = item.find("image_url");
          if (u && u->is_object()) u = u->find("url");
          if (!u) u = item.find("image");
          images->push_back(u && u->is_string() ? u->str() : std::string());
        }
        return;
      }
      if (item.find("video") || item.find("video_url") || is_str(type, "video") || is_str(type, "video_url")) {
        ++video_count;
        if (opt.add_vision_id) out += "Video " + std::to_string(video_count) + ": ";
        out += kVideoStart + kVideoPad + kVideoEnd;
        return;
      }
      py_repr_to(out, item);
      return;
    }
    out += py_str(&item);
  }

  void render_content(const Value* c) {
    if (!c || c->is_null()) return;
    if (c->is_string()) out += c->str();
    else if (c->is_array())
      for (const Value& item : c->items()) render_item(item);
    else out += py_str(c);
  }

  std::string content_string(const Value* c) {
    std::string saved;
    std::swap(out, saved);
    render_content(c);
    std::swap(out, saved);
    return saved;
  }

  void tool_response(const Value& m) {
    out += kRespOpen;
    const Value* c = attr(&m, "content");
    if (!c) {
    } else if (c->is_string()) {
      out += c->str();
    } else if (c->is_array()) {
      bool open = false;
      for (const Value& item : c->items()) {
        const bool search = item.is_object() && is_str(item.find("type"), "tool_search_output");
        if (search && !open) {
          out += "<tools>\n";
          open = true;
        } else if (!search && open) {
          out += "</tools>";
          open = false;
        }
        if (search) {
          const Value* ts = item.find("tools");
          if (ts && ts->is_array()) {
            for (const Value& t : ts->items()) {
              tool_to_json(*unwrap(t));
              out += '\n';
            }
          } else if (ts && ((ts->is_string() && !ts->str().empty()) || (ts->is_object() && ts->size() > 0))) {
            throw ChatError("tool_search_output.tools must be a list of tools");
          }
        } else if (item.is_object() && item.find("output")) {
          render_content(item.find("output"));
        } else {
          render_item(item);
        }
      }
      if (open) out += "</tools>";
    } else {
      render_content(c);
    }
    out += kRespClose;
  }

  void tool_cluster(const std::vector<const Value*>& msgs) {
    std::vector<const Value*> ordered;
    bool matched = false;
    if (!prev_ids.empty() && prev_ids.size() == msgs.size()) {
      matched = true;
      for (const Value* id : prev_ids) {
        int count = 0;
        const Value* hit = nullptr;
        for (const Value* m : msgs) {
          const Value* mid = attr(m, "tool_call_id");
          if (!mid) mid = attr(m, "id");
          if (mid && jinja_eq(mid, id)) {
            ++count;
            hit = m;
          }
        }
        if (count == 1) ordered.push_back(hit);
        else matched = false;
      }
    }
    for (const Value* m : matched ? ordered : msgs) tool_response(*m);
  }

  void assistant(const Item& it, size_t index, long long last_user, bool last, const std::string& fallback_tail) {
    const Value& m = *it.msg;
    const std::string body = content_string(attr(&m, "content"));
    std::string reasoning;
    if (const Value* r = attr(&m, "reasoning"); r && r->is_string() && !r->str().empty()) reasoning = r->str();
    else if (const Value* rc = attr(&m, "reasoning_content"); rc && rc->is_string() && !rc->str().empty())
      reasoning = rc->str();
    const bool preserved = opt.preserved_thinking_set ? opt.preserved_thinking : tools != nullptr;
    std::string content;
    if (effort != "no_think" && (preserved || (long long)index > last_user) && !reasoning.empty())
      content = kThinkOpen + reasoning + kThinkClose + body;
    else content = kThinkOpen + kThinkClose + body;
    out += kHyStart + "assistant" + kHyMiddle;
    const Value* calls = attr(&m, "tool_calls");
    if (truthy(calls)) {
      if (!calls->is_array()) throw ChatError("tool_calls must be an array");
      std::vector<const Value*> ids;
      bool complete = true;
      for (const Value& tc : calls->items()) {
        const Value* id = attr(&tc, "id");
        if (!truthy(id)) id = attr(&tc, "tool_call_id");
        if (truthy(id)) ids.push_back(id);
        else complete = false;
      }
      prev_ids = complete ? ids : std::vector<const Value*>{};
      out += content;
      out += kCallsOpen;
      for (const Value& tc : calls->items()) {
        const Value* f = unwrap(tc);
        const Value* args = attr(f, "arguments");
        if (!args || !args->is_object()) throw ChatError("tool call arguments must be a JSON object");
        out += kCallOpen + py_str(attr(f, "name"));
        for (const auto& [k, v] : args->members()) {
          out += kKeyOpen + k + kKeyClose + kValOpen;
          if (v.is_string()) out += v.str();
          else v.dump_py_to(out);
          out += kValClose;
        }
        out += kCallClose;
      }
      out += kCallsClose;
    } else {
      out += last && opt.raw_last_assistant ? body : content;
    }
    if (!(last && opt.raw_last_assistant) && !(last && opt.fallback_retry)) out += kHyEnd;
    (void)fallback_tail;
  }

  bool run(bool& thinking) {
    // merge_tool_responses
    std::vector<Item> items;
    for (const Value* m : msgs_) {
      if (is_str(attr(m, "role"), "tool")) {
        if (!items.empty() && !items.back().msg) items.back().cluster.push_back(m);
        else items.push_back(Item{nullptr, {m}});
      } else {
        items.push_back(Item{m, {}});
      }
    }
    long long last_user = -1;
    for (size_t i = 0; i < items.size(); ++i)
      if (items[i].is_role("user")) last_user = (long long)i;
    const bool leading_system = !items.empty() && items[0].is_role("system");
    if (!leading_system) {
      out += kHyStart + "system" + kHyMiddle;
      if (tools) tools_prompt();
      out += kReasoningMode + "reasoning_effort:" + effort;
      out += kHyEnd;
    }
    for (size_t i = 0; i < items.size(); ++i) {
      const Item& it = items[i];
      const bool last = i + 1 == items.size();
      if (it.is_role("system")) {
        out += kHyStart + "system" + kHyMiddle;
        if (i == 0) {
          if (tools) tools_prompt();
          const std::string content = content_string(attr(it.msg, "content"));
          if (tools && !content.empty()) out += "\n\n";
          out += content;
          out += kReasoningMode + "reasoning_effort:" + effort;
        } else {
          render_content(attr(it.msg, "content"));
        }
        out += kHyEnd;
      } else if (it.is_role("user")) {
        out += kHyStart + "user" + kHyMiddle;
        render_content(attr(it.msg, "content"));
        out += kHyEnd;
      } else if (it.is_role("assistant")) {
        assistant(it, i, last_user, last, "");
      } else if (!it.msg) {
        out += kHyStart + "tool" + kHyMiddle;
        tool_cluster(it.cluster);
        out += kHyEnd;
      } else {
        out += kHyStart + py_str(it.role()) + kHyMiddle;
        render_content(attr(it.msg, "content"));
        out += kHyEnd;
      }
      if (!it.is_role("assistant") || !truthy(attr(it.msg, "tool_calls"))) prev_ids.clear();
    }
    const bool last_assistant = !items.empty() && items.back().is_role("assistant");
    thinking = false;
    if (opt.add_generation_prompt && !opt.fallback_retry && !last_assistant) {
      out += kHyStart + "assistant" + kHyMiddle + kThinkOpen;
      if (effort == "no_think") out += kThinkClose;
      else thinking = true;
    }
    return true;
  }

  std::vector<const Value*> msgs_;
};

// ---- tool-call arguments ------------------------------------------------------------------------

// A failure vLLM's strict extraction turns into "the whole content is text".
struct BlockError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

size_t count_of(std::string_view s, std::string_view t) {
  size_t n = 0;
  for (size_t p = s.find(t); p != std::string_view::npos; p = s.find(t, p + t.size())) ++n;
  return n;
}

// func_args_regex.finditer: <arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>, DOTALL, nothing in between.
// Calls f(key, value) per match and returns the text with the matches removed.
template <class F>
std::string arg_matches(std::string_view raw, F&& f) {
  std::string rest;
  size_t pos = 0, s = raw.find(kKeyOpen);
  while (s != std::string_view::npos) {
    bool matched = false;
    for (size_t e = raw.find(kKeyClose, s + kKeyOpen.size()); e != std::string_view::npos;
         e = raw.find(kKeyClose, e + 1)) {
      const size_t v0 = e + kKeyClose.size();
      if (raw.substr(v0, kValOpen.size()) != kValOpen) continue;
      const size_t v1 = raw.find(kValClose, v0 + kValOpen.size());
      if (v1 == std::string_view::npos) break;
      f(raw.substr(s + kKeyOpen.size(), e - s - kKeyOpen.size()),
        raw.substr(v0 + kValOpen.size(), v1 - v0 - kValOpen.size()));
      rest.append(raw.substr(pos, s - pos));
      pos = v1 + kValClose.size();
      matched = true;
      break;
    }
    s = raw.find(kKeyOpen, matched ? pos : s + 1);
  }
  rest.append(raw.substr(pos));
  return rest;
}

// _normalize_type
std::string normalize_type(const std::string& raw) {
  static const std::pair<const char*, const char*> kAliases[] = {
      {"str", "string"}, {"text", "string"}, {"varchar", "string"}, {"char", "string"}, {"enum", "string"},
      {"bool", "boolean"}, {"binary", "boolean"}, {"int", "integer"}, {"float", "number"}, {"double", "number"},
      {"list", "array"}, {"dict", "object"}, {"map", "object"}};
  for (const auto& [a, b] : kAliases)
    if (raw == a) return b;
  const std::string lower = ascii_lower(raw);
  for (const char* p : {"int", "uint", "long", "short", "unsigned"})
    if (lower.rfind(p, 0) == 0) return "integer";
  for (const char* p : {"num", "float"})
    if (lower.rfind(p, 0) == 0) return "number";
  return raw;
}

// _get_types(arg_schema)
std::set<std::string> schema_types(const Value* schema) {
  std::vector<std::string> raw;
  const auto type_of = [](const Value& s) {
    const Value* t = s.is_object() ? s.find("type") : nullptr;
    return t && t->is_string() ? t->str() : std::string("string");
  };
  if (schema && schema->is_object() && schema->find("type")) {
    const Value* t = schema->find("type");
    if (t->is_array()) {
      for (const Value& x : t->items()) raw.push_back(x.is_string() ? x.str() : std::string("string"));
    } else {
      raw.push_back(t->is_string() ? t->str() : std::string("string"));
    }
  } else if (const Value* a = schema && schema->is_object() ? schema->find("anyOf") : nullptr; a && a->is_array()) {
    for (const Value& x : a->items()) raw.push_back(type_of(x));
  } else if (const Value* o = schema && schema->is_object() ? schema->find("oneOf") : nullptr; o && o->is_array()) {
    for (const Value& x : o->items()) raw.push_back(type_of(x));
  } else {
    raw.push_back("string");
  }
  std::set<std::string> out;
  for (const std::string& t : raw) out.insert(normalize_type(t));
  out.erase("null");
  return out;
}

// Python int(s) for ASCII input: sign, digits with single underscores between them, surrounding whitespace.
bool py_int(std::string_view s, std::string& out) {
  s = py_strip(s);
  std::string sign;
  if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
    if (s[0] == '-') sign = "-";
    s.remove_prefix(1);
  }
  std::string digits;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] >= '0' && s[i] <= '9') digits += s[i];
    else if (s[i] == '_' && i > 0 && i + 1 < s.size() && s[i - 1] != '_' && s[i + 1] >= '0' && s[i + 1] <= '9') continue;
    else return false;
  }
  if (digits.empty()) return false;
  const size_t nz = digits.find_first_not_of('0');
  digits = nz == std::string::npos ? "0" : digits.substr(nz);
  out = digits == "0" ? digits : sign + digits;
  return true;
}

// Python float(s) for decimal literals (inf / nan never get here: they contain no '.', 'e' or 'E').
bool py_float(std::string_view s, double& out) {
  s = py_strip(s);
  std::string t;
  bool digit = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c >= '0' && c <= '9') {
      t += c;
      digit = true;
    } else if (c == '_') {
      const bool ok = i > 0 && i + 1 < s.size() && s[i - 1] >= '0' && s[i - 1] <= '9' && s[i + 1] >= '0' && s[i + 1] <= '9';
      if (!ok) return false;
    } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
      t += c;
    } else {
      return false;
    }
  }
  if (!digit) return false;
  char* end = nullptr;
  out = std::strtod(t.c_str(), &end);
  return end && *end == '\0' && end != t.c_str();
}

// json.loads(value): any JSON value, with Python's NaN / Infinity.
std::optional<Value> json_loads(const std::string& s) {
  try {
    return json::parse(s);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// ast.literal_eval, for the literals a model writes (see the header for what is left out).
struct Literal {
  struct Fail {};  // not a literal: literal_eval raises
  std::string_view s;
  size_t i = 0;

  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r' || s[i] == '\f')) ++i;
  }
  bool at(char c) const { return i < s.size() && s[i] == c; }

  Value number(bool negative) {
    const size_t start = i;
    int base = 10;
    if (at('0') && i + 1 < s.size() && (s[i + 1] == 'x' || s[i + 1] == 'X')) base = 16;
    else if (at('0') && i + 1 < s.size() && (s[i + 1] == 'o' || s[i + 1] == 'O')) base = 8;
    else if (at('0') && i + 1 < s.size() && (s[i + 1] == 'b' || s[i + 1] == 'B')) base = 2;
    if (base != 10) {
      i += 2;
      std::string digits;
      while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) {
        if (s[i] != '_') digits += s[i];
        ++i;
      }
      if (digits.empty()) throw Fail{};
      char* end = nullptr;
      const unsigned long long v = std::strtoull(digits.c_str(), &end, base);
      if (*end != '\0' || v > 0x7fffffffffffffffULL) throw Fail{};
      return Value::number_text((negative && v ? "-" : "") + std::to_string(v));
    }
    std::string t;
    bool is_float = false;
    while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '_' || s[i] == '.' || s[i] == 'e' || s[i] == 'E' ||
                            ((s[i] == '+' || s[i] == '-') && i > start && (s[i - 1] == 'e' || s[i - 1] == 'E')))) {
      if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') is_float = true;
      if (s[i] != '_') t += s[i];
      ++i;
    }
    if (i < s.size() && (s[i] == 'j' || s[i] == 'J')) throw std::runtime_error("complex literal");
    if (t.empty()) throw Fail{};
    if (!is_float) {
      if (t.size() > 1 && t[0] == '0' && t.find_first_not_of('0') != std::string::npos) throw Fail{};
      const size_t nz = t.find_first_not_of('0');
      t = nz == std::string::npos ? "0" : t.substr(nz);
      return Value::number_text(negative && t != "0" ? "-" + t : t);
    }
    char* end = nullptr;
    const double d = std::strtod(t.c_str(), &end);
    if (*end != '\0') throw Fail{};
    return Value(negative ? -d : d);
  }

  std::string string_literal() {
    const char q = s[i++];
    std::string out;
    while (i < s.size() && s[i] != q) {
      char c = s[i++];
      if (c == '\n') throw Fail{};
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) throw Fail{};
      c = s[i++];
      switch (c) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case '\\': case '\'': case '"': out += c; break;
        case '\n': break;
        case 'x': case 'u': case 'U': case 'N': case '0': case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case 'a': case 'b': case 'f': case 'v':
          throw Fail{};
        default: out += '\\'; out += c;
      }
    }
    if (i >= s.size()) throw Fail{};
    ++i;
    return out;
  }

  Value value() {
    ws();
    if (i >= s.size()) throw Fail{};
    const char c = s[i];
    if (c == '[' || c == '(') {
      const bool tuple = c == '(';
      const char close = tuple ? ')' : ']';
      ++i;
      Value arr = Value::array();
      bool comma = false;
      for (;;) {
        ws();
        if (at(close)) {
          ++i;
          break;
        }
        arr.push_back(value());
        ws();
        if (at(',')) {
          ++i;
          comma = true;
          continue;
        }
        if (at(close)) {
          ++i;
          break;
        }
        throw Fail{};
      }
      if (tuple && arr.size() == 1 && !comma) return arr.items()[0];
      return arr;
    }
    if (c == '{') {
      ++i;
      Value obj = Value::object();
      ws();
      if (at('}')) {
        ++i;
        return obj;
      }
      for (bool first = true;; first = false) {
        ws();
        Value k = value();
        ws();
        if (!at(':')) {
          if (first) throw std::runtime_error("set literal");  // json.dumps fails on a set
          throw Fail{};
        }
        ++i;
        if (!k.is_string()) throw Fail{};
        Value v = value();
        obj.set(k.str(), std::move(v));
        ws();
        if (at(',')) {
          ++i;
          ws();
          if (at('}')) {
            ++i;
            return obj;
          }
          continue;
        }
        if (at('}')) {
          ++i;
          return obj;
        }
        throw Fail{};
      }
    }
    if (c == '-' || c == '+') {
      ++i;
      ws();
      if (i >= s.size() || !((s[i] >= '0' && s[i] <= '9') || s[i] == '.')) throw Fail{};
      return number(c == '-');
    }
    if ((c >= '0' && c <= '9') || c == '.') return number(false);
    // string prefixes
    size_t p = i;
    while (p < s.size() && p < i + 2 && std::isalpha(static_cast<unsigned char>(s[p]))) ++p;
    if (p < s.size() && (s[p] == '\'' || s[p] == '"') && p > i) {
      const std::string prefix = ascii_lower(s.substr(i, p - i));
      if (prefix.find('b') != std::string::npos) throw std::runtime_error("bytes literal");
      if (prefix != "u") throw Fail{};
      i = p;
    }
    if (at('\'') || at('"')) return Value(string_literal());
    size_t e = i;
    while (e < s.size() && (std::isalnum(static_cast<unsigned char>(s[e])) || s[e] == '_')) ++e;
    const std::string_view id = s.substr(i, e - i);
    i = e;
    if (id == "None") return Value();
    if (id == "True") return Value(true);
    if (id == "False") return Value(false);
    throw Fail{};
  }
};

std::optional<Value> literal_eval(const std::string& text) {
  Literal l{text};
  while (l.i < text.size() && (text[l.i] == ' ' || text[l.i] == '\t')) ++l.i;
  try {
    Value v = l.value();
    l.ws();
    if (l.i != text.size()) return std::nullopt;
    return v;
  } catch (const Literal::Fail&) {
    return std::nullopt;
  }
}

// _deserialize
Value deserialize(const std::string& value) {
  if (auto v = json_loads(value)) return *v;
  if (auto v = literal_eval(value)) return *v;
  return Value(value);
}

// find_tool_properties: the first tool with this name; its parameters.properties.
const Value* tool_properties(const Value& tools, std::string_view name) {
  if (!tools.is_array()) return nullptr;
  for (const Value& t : tools.items()) {
    const Value* f = t.is_object() ? t.find("function") : nullptr;
    const Value& fn = f ? *f : t;
    if (fn.get_str("name") != name) continue;
    const Value* params = fn.is_object() ? fn.find("parameters") : nullptr;
    const Value* props = params && params->is_object() ? params->find("properties") : nullptr;
    return props && props->is_object() ? props : nullptr;
  }
  return nullptr;
}

bool tool_declared(const Value& tools, std::string_view name) {
  if (!tools.is_array()) return false;
  for (const Value& t : tools.items()) {
    const Value* f = t.is_object() ? t.find("function") : nullptr;
    if ((f ? *f : t).get_str("name") == name) return true;
  }
  return false;
}

// [A-Za-z0-9_-]{1,64}: clients keep the names they are given and replay them.
bool tool_name_ok(std::string_view s) {
  if (s.empty() || s.size() > 64) return false;
  for (const char c : s)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-'))
      return false;
  return true;
}

// The first n bytes of s, without splitting a UTF-8 character.
std::string_view utf8_prefix(std::string_view s, size_t n) {
  if (s.size() <= n) return s;
  while (n > 0 && (static_cast<unsigned char>(s[n]) >> 6) == 2) --n;
  return s.substr(0, n);
}

// One <tool_call> body. partial: the call was cut off (lenient, the open value is kept as text).
ToolCall parse_call(std::string_view body, bool partial, const Value& tools) {
  const size_t arg_start = body.find(kKeyOpen);
  const std::string name(py_strip(arg_start == std::string_view::npos ? body : body.substr(0, arg_start)));
  if (name.empty()) throw BlockError("empty function name");
  if (!tool_declared(tools, name) && !tool_name_ok(name)) throw BlockError("invalid tool name " + Value(std::string(utf8_prefix(name, 80))).dump());
  const std::string args =
      arg_start == std::string_view::npos ? std::string() : std::string(py_strip(body.substr(arg_start)));
  if (!partial && (count_of(args, kKeyOpen) != count_of(args, kKeyClose) ||
                   count_of(args, kValOpen) != count_of(args, kValClose)))
    throw BlockError("unbalanced argument tags");
  const Value* props = tool_properties(tools, name);
  Value params = Value::object();
  const std::string rest = arg_matches(args, [&](std::string_view k, std::string_view v) {
    const Value* schema = props ? props->find(k) : nullptr;
    params.set(std::string(k), hy4_parse_value(std::string(v), schema));
  });
  if (!partial) {
    if (!py_strip(rest).empty()) throw BlockError("unparsed argument payload");
  } else {
    // the value being written: <arg_key>(.*?)</arg_key><arg_value>(.*)$ in what the full matches left
    for (size_t s = rest.find(kKeyOpen); s != std::string::npos; s = rest.find(kKeyOpen, s + 1)) {
      bool found = false;
      for (size_t e = rest.find(kKeyClose, s + kKeyOpen.size()); e != std::string::npos;
           e = rest.find(kKeyClose, e + 1)) {
        const size_t v0 = e + kKeyClose.size();
        if (rest.compare(v0, kValOpen.size(), kValOpen) != 0) continue;
        params.set(rest.substr(s + kKeyOpen.size(), e - s - kKeyOpen.size()), Value(rest.substr(v0 + kValOpen.size())));
        found = true;
        break;
      }
      if (found) break;
    }
  }
  return ToolCall{name, params.dump_py()};
}

// vLLM's strict extraction of the text between <tool_calls> and </tool_calls> (cut: no closing tag; the calls
// that closed are strict, the last one may be open). Throws BlockError.
std::vector<ToolCall> parse_block(std::string_view block, bool cut, const Value& tools) {
  const size_t n_start = count_of(block, kCallOpen), n_end = count_of(block, kCallClose);
  if (!cut && n_start != n_end) throw BlockError("unequal <tool_call> and </tool_call> counts");
  std::vector<std::string_view> bodies;
  std::optional<std::string_view> open;
  for (size_t pos = 0;;) {
    const size_t s = block.find(kCallOpen, pos);
    if (s == std::string_view::npos) break;
    const size_t b = s + kCallOpen.size();
    const size_t e = block.find(kCallClose, b);
    if (e == std::string_view::npos) {
      if (cut) open = block.substr(b);
      break;
    }
    bodies.push_back(block.substr(b, e - b));
    pos = e + kCallClose.size();
  }
  if (!cut && bodies.size() != n_start) throw BlockError("malformed <tool_call> body");
  std::vector<ToolCall> calls;
  for (std::string_view body : bodies) calls.push_back(parse_call(body, false, tools));
  if (open) calls.push_back(parse_call(*open, true, tools));
  if (calls.empty()) throw BlockError("no tool calls");
  return calls;
}

}  // namespace

// ---- rendering ----------------------------------------------------------------------------------

std::string hy4_render(const Value& messages, const Value* tools, const Hy4Options& opt,
                       std::vector<std::string>* images, bool* thinking) {
  if (!messages.is_array()) throw ChatError("'messages' must be an array");
  if (messages.items().empty()) throw ChatError("cannot apply the chat template to an empty conversation");  // transformers' check
  Renderer r{tools && tools->is_array() && !tools->items().empty() ? tools : nullptr, opt, opt.reasoning_effort,
             images, {}, 0, 0, {}, {}};
  if (r.effort != "no_think" && r.effort != "high")
    throw ChatError("reasoning_effort error : " + r.effort + ", should be no_think/high");
  if (opt.fallback_retry) r.effort = "high";
  for (const Value& m : messages.items()) {
    if (!m.is_object()) throw ChatError("every message must be an object");
    r.msgs_.push_back(&m);
  }
  bool think = false;
  r.run(think);
  if (thinking) *thinking = think;
  return std::move(r.out);
}

RenderedChat hy4_render_request(const Value& req) {
  static const Value kEmpty = Value::object();
  const Value* kw = req.find("chat_template_kwargs");
  const Value& kwargs = kw && kw->is_object() ? *kw : kEmpty;
  const Value* m = req.find("messages");
  if (!m || !m->is_array()) throw ChatError("'messages' must be an array");
  Value messages = *m;
  // vLLM: assistant tool-call arguments given as JSON text become objects before the template sees them.
  for (Value& msg : messages.items()) {
    Value* calls = msg.is_object() && msg.get_str("role") == "assistant" ? msg.find("tool_calls") : nullptr;
    if (!calls || !calls->is_array()) continue;
    for (Value& tc : calls->items()) {
      Value* f = tc.is_object() ? tc.find("function") : nullptr;
      Value* args = f && f->is_object() ? f->find("arguments") : nullptr;
      if (!args || !args->is_string()) continue;
      try {
        *args = json::parse(args->str());
      } catch (const std::exception&) {
        throw ChatError("tool call arguments are not valid JSON");
      }
    }
  }
  Hy4Options opt;
  // vLLM merges the top-level field over the kwargs, ignoring null and "auto".
  const Value* top = req.find("reasoning_effort");
  if (top && !top->is_null() && !is_str(top, "auto")) {
    if (!top->is_string()) throw ChatError("reasoning_effort error : should be no_think/high");
    opt.reasoning_effort = top->str();
  } else if (const Value* e = kwargs.find("reasoning_effort")) {
    if (!e->is_string()) throw ChatError("reasoning_effort error : should be no_think/high");
    opt.reasoning_effort = e->str();
  }
  // OpenAI's name for no reasoning (what maitred and its clients send); the template itself knows no_think / high.
  if (opt.reasoning_effort == "none") opt.reasoning_effort = "no_think";
  if (const Value* v = kwargs.find("preserved_thinking")) {
    opt.preserved_thinking_set = true;
    opt.preserved_thinking = v->truthy();
  }
  if (const Value* v = kwargs.find("raw_last_assistant")) opt.raw_last_assistant = v->truthy();
  if (const Value* v = kwargs.find("add_vision_id")) opt.add_vision_id = v->truthy();
  opt.fallback_retry = is_str(kwargs.find("fallback_strategy"), "reasoning_toolcall_retry");
  RenderedChat r;
  r.prompt = hy4_render(messages, req.find("tools"), opt, &r.images, &r.thinking);
  return r;
}

Value hy4_parse_value(const std::string& value, const Value* schema) {
  const std::set<std::string> types = schema_types(schema);
  const auto has = [&](const char* t) { return types.count(t) > 0; };
  if (has("boolean")) {
    const std::string l = ascii_lower(value);
    if (l == "true") return Value(true);
    if (l == "false") return Value(false);
  }
  if (has("integer")) {
    std::string digits;
    if (py_int(value, digits)) return Value::number_text(digits);
  }
  if (has("number")) {
    if (value.find_first_of(".eE") != std::string::npos) {
      double d = 0;
      if (py_float(value, d)) return Value(d);
    } else {
      std::string digits;
      if (py_int(value, digits)) return Value::number_text(digits);
    }
  }
  for (const std::string& t : types) {
    if (t != "string" && t != "boolean" && t != "integer" && t != "number") {
      if (auto v = json_loads(value)) return *v;
      break;
    }
  }
  if (has("string")) return Value(value);
  try {
    return deserialize(value);
  } catch (const std::runtime_error& e) {
    throw BlockError(e.what());
  }
}

// ---- completion parsing -------------------------------------------------------------------------

namespace {

// Bytes at the end of s held back until more text arrives: the start of a terminal, or an incomplete UTF-8
// character.
size_t held_tail(std::string_view s, const std::vector<const std::string*>& terminals) {
  size_t best = 0;
  for (const std::string* t : terminals)
    for (size_t k = std::min(t->size() - 1, s.size()); k > best; --k)
      if (s.substr(s.size() - k) == std::string_view(*t).substr(0, k)) {
        best = k;
        break;
      }
  for (size_t k = 1; k <= std::min<size_t>(3, s.size()); ++k) {
    const auto c = static_cast<unsigned char>(s[s.size() - k]);
    if ((c >> 6) == 2) continue;
    const size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (len > k) best = std::max(best, k);
    break;
  }
  return best;
}

}  // namespace

Hy4OutputParser::Hy4OutputParser(bool thinking, const Value* tools)
    : state_(thinking ? State::Reasoning : State::Content) {
  if (tools && tools->is_array() && !tools->items().empty()) tools_ = *tools;
}

void Hy4OutputParser::feed(std::string_view text, ParsedDelta& out) {
  buf_.append(text);
  step(out, false);
}

void Hy4OutputParser::finish(ParsedDelta& out) {
  step(out, true);
  if (state_ == State::Block) end_block(true, out);
  if (state_ == State::Content || state_ == State::Raw) out.content += hold_c_;
  hold_c_.clear();
}

// Content text; whitespace-only content is held until other text follows (vLLM drops it when calls follow).
void Hy4OutputParser::content_out(std::string_view s, ParsedDelta& out) {
  if (s.empty()) return;
  if (content_text_) {
    out.content.append(s);
    return;
  }
  hold_c_.append(s);
  if (lead_space(hold_c_) == hold_c_.size()) return;
  out.content += hold_c_;
  hold_c_.clear();
  content_text_ = true;
}

void Hy4OutputParser::end_block(bool cut, ParsedDelta& out) {
  std::vector<ToolCall> calls;
  std::string why;
  try {
    calls = parse_block(block_, cut, tools_);
  } catch (const BlockError& e) {
    why = e.what();
  }
  if (why.empty()) {
    hold_c_.clear();
    for (ToolCall& c : calls) out.tool_calls.push_back(std::move(c));
    n_calls_ += int(calls.size());
    state_ = State::Done;
  } else {
    // vLLM's strict mode: the whole content, markup included, is text.
    out.content += hold_c_;
    out.content += kCallsOpen;
    out.content += block_;
    if (!cut) out.content += kCallsClose;
    hold_c_.clear();
    content_text_ = true;
    state_ = State::Raw;
    out.log.push_back("tool block returned as content (" + why + "): " +
                      Value(std::string(utf8_prefix(block_, 200))).dump());
  }
  block_.clear();
}

void Hy4OutputParser::step(ParsedDelta& out, bool final) {
  for (;;) {
    std::vector<const std::string*> terms;
    switch (state_) {
      case State::Reasoning:
        if (!think_seen_) terms.push_back(&kThinkOpen);
        terms.push_back(&kThinkClose);
        break;
      case State::Content: terms = {&kCallsOpen, &kCallsClose}; break;
      case State::Block: terms = {&kCallsClose}; break;
      default: break;
    }
    size_t pos = std::string::npos;
    const std::string* hit = nullptr;
    for (const std::string* t : terms) {
      const size_t p = buf_.find(*t);
      if (p < pos) pos = p, hit = t;
    }
    const size_t n = pos != std::string::npos ? pos : final ? buf_.size() : buf_.size() - held_tail(buf_, terms);
    const std::string_view text = std::string_view(buf_).substr(0, n);
    switch (state_) {
      case State::Reasoning: out.reasoning.append(text); break;
      case State::Content: content_out(text, out); break;
      case State::Block: block_.append(text); break;
      case State::Raw:
        out.content.append(text);
        break;
      case State::Done: break;
    }
    if (pos == std::string::npos) {
      buf_.erase(0, n);
      return;
    }
    buf_.erase(0, pos + hit->size());
    if (hit == &kThinkOpen) {
      think_seen_ = true;
    } else if (hit == &kThinkClose) {
      state_ = State::Content;
    } else if (hit == &kCallsOpen) {
      if (stray_end_) {
        out.content += hold_c_;
        out.content += kCallsOpen;
        hold_c_.clear();
        content_text_ = true;
        state_ = State::Raw;
      } else {
        block_.clear();
        state_ = State::Block;
      }
    } else if (hit == &kCallsClose) {
      if (state_ == State::Block) {
        end_block(false, out);
      } else {
        stray_end_ = true;
        content_out(kCallsClose, out);
      }
    }
  }
}

namespace {
class Hy4ChatFormat final : public ChatFormat {
 public:
  RenderedChat render(const Value& request) const override { return hy4_render_request(request); }
  std::unique_ptr<ChatParser> parser(bool thinking, const Value* tools) const override {
    return std::make_unique<Hy4OutputParser>(thinking, tools);
  }
};
}  // namespace

const ChatFormat& hy4_chat_format() {
  static const Hy4ChatFormat format;
  return format;
}

}  // namespace unholy::text
