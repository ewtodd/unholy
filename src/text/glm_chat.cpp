#include "text/glm_chat.hpp"

#include <algorithm>
#include <cmath>
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

// str.isspace() / regex \s on str patterns.
bool py_space(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

// Byte length of the leading whitespace of s.
size_t lead_space(std::string_view s) {
  size_t i = 0, n = 0;
  while (i < s.size() && py_space(utf8_at(s, i, n))) i += n;
  return i;
}

// Byte offset where the trailing whitespace of s starts.
size_t trail_space(std::string_view s) {
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

// Python str.split(sep)[0] and [-1].
std::string_view before_first(std::string_view s, std::string_view sep) {
  const size_t p = s.find(sep);
  return p == std::string_view::npos ? s : s.substr(0, p);
}
std::string_view after_last(std::string_view s, std::string_view sep) {
  const size_t p = s.rfind(sep);
  return p == std::string_view::npos ? s : s.substr(p + sep.size());
}

// ---- Jinja value semantics ------------------------------------------------------------------------

// Attribute / item access: nullptr is Jinja's undefined.
const Value* attr(const Value* v, std::string_view key) { return v && v->is_object() ? v->find(key) : nullptr; }
const Value* first(const Value* v) { return v && v->is_array() && !v->items().empty() ? &v->items()[0] : nullptr; }
bool truthy(const Value* v) { return v && v->truthy(); }
bool is_str(const Value* v, std::string_view s) { return v && v->is_string() && v->str() == s; }

// {{ v }}: Python str() of the value.
std::string jinja_str(const Value* v) {
  if (!v) return "";
  if (v->is_string()) return v->str();
  if (v->is_null()) return "None";
  if (v->is_bool()) return v->as_bool() ? "True" : "False";
  if (v->is_number()) return v->number_is_int() ? v->str() : json::py_float_repr(v->as_double());
  throw ChatError("cannot render a JSON object or array as text");
}

// Jinja ==: strings and numbers by value, two undefineds are equal.
bool jinja_eq(const Value* a, const Value* b) {
  if (!a || !b) return !a && !b;
  if (a->is_string() && b->is_string()) return a->str() == b->str();
  return a->dump_py() == b->dump_py();
}

// ---- the template -------------------------------------------------------------------------------

constexpr std::string_view kToolsHead =
    "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
constexpr std::string_view kToolsTail =
    "</tools>\n\nFor each function call, output the function name and arguments within the following XML "
    "format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

struct Renderer {
  const std::vector<Value>& msgs;
  const Value* tools;  // null when the template's `tools` is falsy
  const Glm5Options& opt;
  std::vector<std::string>* images;
  std::string out;

  static const Value& unwrap_function(const Value& tool) {
    const Value* f = attr(&tool, "function");
    return f ? *f : tool;
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

  void visible_text(const Value* content) {
    if (!content || content->is_null()) return;
    if (content->is_string()) {
      out += content->str();
      return;
    }
    if (!content->is_array()) {
      out += jinja_str(content);
      return;
    }
    for (const Value& item : content->items()) {
      if (item.is_string()) {
        out += item.str();
        continue;
      }
      if (!item.is_object()) continue;
      const Value* type = item.find("type");
      if (is_str(type, "text")) {
        out += jinja_str(item.find("text"));
      } else if (is_str(type, "image") || is_str(type, "image_url")) {
        out += "<|begin_of_image|><|image|><|end_of_image|>";
        if (images) {
          const Value* u = item.find("image_url");
          if (u && u->is_object()) u = u->find("url");
          if (!u) u = item.find("image");
          images->push_back(u && u->is_string() ? u->str() : std::string());
        }
      } else if (is_str(type, "video") || is_str(type, "video_url")) {
        out += "<|begin_of_video|><|video|><|end_of_video|>";
      } else if (is_str(type, "audio") || is_str(type, "audio_url") || is_str(type, "input_audio")) {
        out += "<|begin_of_audio|><|end_of_audio|>";
      }
    }
  }

  void tool_response(const Value* content_or_output) {
    out += "<tool_response>";
    visible_text(content_or_output);
    out += "</tool_response>";
  }

  void tool_references(const Value& refs) {
    if (!tools) throw ChatError("tool_reference results need the request's tools");
    out += "<tool_response><tools>\n";
    for (const Value& tr : refs.items())
      for (const Value& tool : tools->items()) {
        const Value& t = unwrap_function(tool);
        if (jinja_eq(attr(&t, "name"), attr(&tr, "name"))) {
          tool_to_json(t);
          out += '\n';
        }
      }
    out += "</tools></tool_response>";
  }

  static bool is_reference_list(const Value* v) {
    return v && v->is_array() && !v->items().empty() && is_str(attr(first(v), "type"), "tool_reference");
  }

  static bool list_of_outputs(const Value& m) {
    const Value* c = attr(&m, "content");
    return truthy(c) && attr(first(c), "output") != nullptr;
  }

  static std::string id_of(const Value* obj) {
    if (const Value* id = attr(obj, "tool_call_id"); truthy(id)) return jinja_str(id);
    if (const Value* id = attr(obj, "id"); truthy(id)) return jinja_str(id);
    return "";
  }

  void render_tool_response(const Value& m) {
    const Value* c = attr(&m, "content");
    if (c && c->is_string()) {
      tool_response(c);
    } else if (truthy(c) && !c->is_object() && is_str(attr(first(c), "type"), "tool_reference")) {
      tool_references(*c);
    } else if (list_of_outputs(m)) {
      for (const Value& tr : c->items()) {
        const Value* o = attr(&tr, "output");
        if (is_reference_list(o)) tool_references(*o);
        else tool_response(o);
      }
    } else {
      tool_response(c);
    }
  }

  // A run of tool messages [start, end]: in the preceding assistant turn's call order when every result
  // id matches exactly one call, else in message order.
  void tool_block(size_t start) {
    out += "<|observation|>";
    size_t end = start;
    while (end + 1 < msgs.size() && msgs[end + 1].get_str("role") == "tool") ++end;
    const Value* calls = nullptr;
    if (start > 0 && msgs[start - 1].get_str("role") == "assistant" && truthy(attr(&msgs[start - 1], "tool_calls")))
      calls = attr(&msgs[start - 1], "tool_calls");

    std::vector<std::string> res, tcs;
    std::vector<size_t> mk;
    std::vector<int> ek;
    for (size_t k = start; k <= end; ++k) {
      const Value& m = msgs[k];
      if (list_of_outputs(m)) {
        int e = 0;
        for (const Value& entry : attr(&m, "content")->items()) {
          res.push_back(id_of(&entry));
          mk.push_back(k);
          ek.push_back(e++);
        }
      } else {
        res.push_back(id_of(&m));
        mk.push_back(k);
        ek.push_back(-1);
      }
    }
    bool can_sort = calls != nullptr && calls->is_array();
    if (can_sort) {
      for (const Value& tc : calls->items()) tcs.push_back(id_of(&tc));
      const auto has_dups = [](std::vector<std::string> v) {
        std::sort(v.begin(), v.end());
        return std::adjacent_find(v.begin(), v.end()) != v.end();
      };
      const auto contains = [](const std::vector<std::string>& v, const std::string& x) {
        return std::find(v.begin(), v.end(), x) != v.end();
      };
      if (contains(res, "") || contains(tcs, "") || has_dups(res) || has_dups(tcs)) can_sort = false;
      else
        for (const std::string& id : res)
          if (!contains(tcs, id)) {
            can_sort = false;
            break;
          }
    }
    if (!can_sort) {
      for (size_t k = start; k <= end; ++k) render_tool_response(msgs[k]);
      return;
    }
    // ids are unique and every result's id is a call's: the results in call order.
    for (const std::string& id : tcs) {
      const auto it = std::find(res.begin(), res.end(), id);
      if (it == res.end()) continue;
      const size_t p = size_t(it - res.begin());
      const Value& m = msgs[mk[p]];
      if (ek[p] >= 0) {
        const Value& entry = attr(&m, "content")->items()[size_t(ek[p])];
        const Value* o = attr(&entry, "output");
        if (is_reference_list(o)) tool_references(*o);
        else tool_response(o);
      } else {
        render_tool_response(m);
      }
    }
  }

  void assistant(size_t i, long long last_user) {
    const Value& m = msgs[i];
    out += "<|assistant|>";
    std::string content;
    std::swap(out, content);
    visible_text(attr(&m, "content"));
    std::swap(out, content);
    std::optional<std::string> reasoning;
    if (const Value* rc = attr(&m, "reasoning_content"); rc && rc->is_string()) {
      reasoning = rc->str();
    } else if (const Value* r = attr(&m, "reasoning"); r && r->is_string()) {
      reasoning = r->str();  // vLLM's chat layer name for it
    } else if (content.find("</think>") != std::string::npos) {
      reasoning = std::string(after_last(before_first(content, "</think>"), "<think>"));
      content = std::string(after_last(content, "</think>"));
    }
    if ((!opt.clear_thinking || (long long)i > last_user) && reasoning) out += "<think>" + *reasoning + "</think>";
    else out += "<think></think>";
    out += py_strip(content);
    const Value* calls = attr(&m, "tool_calls");
    if (!truthy(calls)) return;
    if (!calls->is_array()) throw ChatError("tool_calls must be an array");
    for (const Value& tc : calls->items()) {
      const Value* f = truthy(attr(&tc, "function")) ? attr(&tc, "function") : &tc;
      out += "<tool_call>" + jinja_str(attr(f, "name"));
      const Value* args = attr(f, "arguments");
      if (!args || !args->is_object()) throw ChatError("tool call arguments must be a JSON object");
      for (const auto& [k, v] : args->members()) {
        out += "<arg_key>" + k + "</arg_key><arg_value>";
        if (v.is_string()) out += v.str();
        else v.dump_py_to(out);
        out += "</arg_value>";
      }
      out += "</tool_call>";
    }
  }

  void run() {
    out = "[gMASK]<sop><|system|>Reasoning Effort: ";
    out += opt.reasoning_effort == "low" ? "Low" : opt.reasoning_effort == "high" ? "High" : "Max";
    if (tools) {
      out += kToolsHead;
      for (const Value& tool : tools->items()) {
        const Value& t = unwrap_function(tool);
        if (truthy(attr(&t, "defer_loading"))) continue;
        tool_to_json(t);
        out += '\n';
      }
      out += kToolsTail;
    }
    long long last_user = -1;  // the template's ns.last_user_index
    for (size_t i = 0; i < msgs.size(); ++i)
      if (msgs[i].get_str("role") == "user") last_user = (long long)i;
    for (size_t i = 0; i < msgs.size(); ++i) {
      const Value& m = msgs[i];
      const std::string role = m.get_str("role");
      if (role == "user") {
        out += "<|user|>";
        visible_text(attr(&m, "content"));
      } else if (role == "assistant") {
        assistant(i, last_user);
      } else if (role == "tool") {
        if (i == 0 || msgs[i - 1].get_str("role") != "tool") tool_block(i);
      } else if (role == "system") {
        out += "<|system|>";
        visible_text(attr(&m, "content"));
      }
    }
    if (opt.add_generation_prompt) out += "<|assistant|><think>";
  }
};

// ---- tool-call arguments ------------------------------------------------------------------------

constexpr std::string_view kKeyOpen = "<arg_key>", kKeyClose = "</arg_key>";
constexpr std::string_view kValOpen = "<arg_value>", kValClose = "</arg_value>";

// _ARG_RE.finditer: <arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>, DOTALL. Calls f(key, value) per
// match and returns the text with the matches removed (_ARG_RE.sub("", raw)).
template <class F>
std::string arg_matches(std::string_view raw, F&& f) {
  std::string rest;
  size_t pos = 0, s = raw.find(kKeyOpen);
  while (s != std::string_view::npos) {
    bool matched = false;
    // the lazy key: the first </arg_key> followed by \s*<arg_value> and, later, a </arg_value>
    for (size_t e = raw.find(kKeyClose, s + kKeyOpen.size()); e != std::string_view::npos;
         e = raw.find(kKeyClose, e + 1)) {
      const size_t j = e + kKeyClose.size();
      const size_t v0 = j + lead_space(raw.substr(j));
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

// _glm47_arg_converter: every value kept as text.
Value convert_args(std::string_view raw, bool partial) {
  Value params = Value::object();
  const std::string rest = arg_matches(raw, [&](std::string_view k, std::string_view v) {
    params.set(std::string(py_strip(k)), Value(std::string(v)));
  });
  if (partial) {
    // _PARTIAL_ARG_RE.search: <arg_key>(.*?)</arg_key>\s*<arg_value>(.*)$ in what the full matches left
    for (size_t s = rest.find(kKeyOpen); s != std::string::npos; s = rest.find(kKeyOpen, s + 1)) {
      bool found = false;
      for (size_t e = rest.find(kKeyClose, s + kKeyOpen.size()); e != std::string::npos;
           e = rest.find(kKeyClose, e + 1)) {
        const size_t j = e + kKeyClose.size();
        const size_t v0 = j + lead_space(std::string_view(rest).substr(j));
        if (rest.compare(v0, kValOpen.size(), kValOpen) != 0) continue;
        const std::string key(py_strip(std::string_view(rest).substr(s + kKeyOpen.size(), e - s - kKeyOpen.size())));
        if (!key.empty()) params.set(key, Value(rest.substr(v0 + kValOpen.size())));
        found = true;
        break;
      }
      if (found) break;
    }
  }
  return params;
}

// extract_types_from_schema
void schema_types(const Value& schema, std::set<std::string>& types) {
  if (!schema.is_object()) return;
  if (const Value* t = schema.find("type")) {
    if (t->is_string()) types.insert(t->str());
    else if (t->is_array())
      for (const Value& x : t->items())
        if (x.is_string()) types.insert(x.str());
  }
  if (const Value* e = schema.find("enum"); e && e->is_array())
    for (const Value& x : e->items())
      types.insert(x.is_null() ? "null" : x.is_bool() ? "boolean" : x.is_number() ? (x.number_is_int() ? "integer" : "number")
                   : x.is_string() ? "string" : x.is_array() ? "array" : "object");
  for (const char* choice : {"anyOf", "oneOf", "allOf"})
    if (const Value* c = schema.find(choice); c && c->is_array())
      for (const Value& x : c->items()) schema_types(x, types);
}

std::string type_alias(std::string t) {
  t = ascii_lower(py_strip(t));
  static const std::pair<const char*, const char*> kAliases[] = {
      {"str", "string"},     {"text", "string"},     {"varchar", "string"},  {"char", "string"},   {"enum", "string"},
      {"int", "integer"},    {"int32", "integer"},   {"int64", "integer"},   {"uint", "integer"},  {"uint32", "integer"},
      {"uint64", "integer"}, {"long", "integer"},    {"short", "integer"},   {"unsigned", "integer"},
      {"float", "number"},   {"float32", "number"},  {"float64", "number"},  {"double", "number"}, {"bool", "boolean"},
      {"dict", "object"},    {"arr", "array"},       {"list", "array"},      {"sequence", "array"}};
  for (const auto& [a, b] : kAliases)
    if (t == a) return b;
  return t;
}

// Python int(s): optional sign, digits with single underscores between them, surrounding whitespace.
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

// Python float(s) for finite decimal literals (inf / nan are rejected by the caller anyway).
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

// The decimal digits of an integral double, exactly (Python int(float)).
std::string exact_integer(double v) {
  if (std::fabs(v) < 9007199254740992.0) return std::to_string(static_cast<long long>(v));
  int exp = 0;
  const double m = std::frexp(std::fabs(v), &exp);  // v = m * 2^exp, 0.5 <= m < 1
  auto mant = static_cast<unsigned long long>(std::ldexp(m, 53));
  exp -= 53;
  std::vector<int> dec;  // little-endian base-10 digits
  while (mant) {
    dec.push_back(int(mant % 10));
    mant /= 10;
  }
  for (int k = 0; k < exp; ++k) {
    int carry = 0;
    for (int& d : dec) {
      const int x = d * 2 + carry;
      d = x % 10;
      carry = x / 10;
    }
    if (carry) dec.push_back(carry);
  }
  std::string out = v < 0 ? "-" : "";
  for (auto it = dec.rbegin(); it != dec.rend(); ++it) out += char('0' + *it);
  return out;
}

bool json_finite(const Value& v) {
  if (v.is_number()) return std::isfinite(v.as_double());
  if (v.is_array())
    for (const Value& x : v.items())
      if (!json_finite(x)) return false;
  if (v.is_object())
    for (const auto& [k, x] : v.members())
      if (!json_finite(x)) return false;
  return true;
}

std::optional<Value> try_json(const std::string& s) {
  try {
    Value v = json::parse(s);
    if (json_finite(v)) return v;
  } catch (const std::exception&) {
  }
  return std::nullopt;
}

Value coerce(const std::string& value, const std::set<std::string>& raw_types) {
  std::set<std::string> types;
  for (const std::string& t : raw_types) types.insert(type_alias(t));
  for (const char* cand : {"null", "integer", "number", "boolean", "object", "array", "string"}) {
    if (!types.count(cand)) continue;
    const std::string_view c = cand;
    if (c == "null") {
      if (ascii_lower(value) == "null") return Value();
    } else if (c == "string") {
      return Value(value);
    } else if (c == "integer") {
      std::string digits;
      if (py_int(value, digits)) return Value::number_text(digits);
    } else if (c == "number") {
      double d = 0;
      if (py_float(value, d) && std::isfinite(d)) return d == std::trunc(d) ? Value::number_text(exact_integer(d)) : Value(d);
    } else if (c == "boolean") {
      const std::string l(py_strip(ascii_lower(value)));
      if (l == "true" || l == "1") return Value(true);
      if (l == "false" || l == "0") return Value(false);
    } else if (auto v = try_json(value)) {  // object, array: any JSON value json.loads accepts
      return *v;
    }
  }
  if (auto v = try_json(value)) return *v;
  return Value(value);
}

// find_tool_properties
const Value* tool_properties(const Value& tools, std::string_view name) {
  if (!tools.is_array()) return nullptr;
  for (const Value& t : tools.items()) {
    const Value* f = t.find("function");
    const Value& fn = f ? *f : t;
    if (fn.get_str("name") != name) continue;
    const Value* params = fn.find("parameters");
    const Value* props = params ? params->find("properties") : nullptr;
    return props && props->is_object() ? props : nullptr;
  }
  return nullptr;
}

bool tool_declared(const Value& tools, std::string_view name) {
  for (const Value& t : tools.items()) {
    const Value* f = t.find("function");
    if ((f ? *f : t).get_str("name") == name) return true;
  }
  return false;
}

}  // namespace

std::string glm5_render(const Value& messages, const Value* tools, const Glm5Options& opt,
                        std::vector<std::string>* images) {
  if (!messages.is_array()) throw ChatError("'messages' must be an array");
  const Value* t = tools && tools->is_array() && !tools->items().empty() ? tools : nullptr;
  Renderer r{messages.items(), t, opt, images, {}};
  r.run();
  return std::move(r.out);
}

RenderedChat glm5_render_request(const Value& req) {
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
  Glm5Options opt;
  const Value* effort = req.find("reasoning_effort");
  if (!effort || !effort->is_string()) effort = kwargs.find("reasoning_effort");
  if (effort && effort->is_string()) opt.reasoning_effort = effort->str();
  if (const Value* c = kwargs.find("clear_thinking")) opt.clear_thinking = c->truthy();
  RenderedChat r;
  r.prompt = glm5_render(messages, req.find("tools"), opt, &r.images);
  r.thinking = true;
  return r;
}

Value glm5_coerce(const std::string& value, const Value& schema) {
  std::set<std::string> types;
  schema_types(schema, types);
  if (types.empty()) types.insert("string");
  return coerce(value, types);
}

// ---- completion parsing -------------------------------------------------------------------------

namespace {
constexpr std::string_view kThinkOpen = "<think>", kThinkClose = "</think>";
constexpr std::string_view kCallOpen = "<tool_call>", kCallClose = "</tool_call>";
constexpr std::string_view kTerminals[] = {kThinkOpen, kThinkClose, kCallOpen, kCallClose, kKeyOpen};

// Bytes at the end of s held back until more text arrives: the start of a terminal, or an incomplete UTF-8
// character (whitespace is judged per character).
size_t held_tail(std::string_view s) {
  size_t best = 0;
  for (std::string_view t : kTerminals)
    for (size_t k = std::min(t.size() - 1, s.size()); k > best; --k)
      if (s.substr(s.size() - k) == t.substr(0, k)) {
        best = k;
        break;
      }
  for (size_t k = 1; k <= std::min<size_t>(3, s.size()); ++k) {
    const auto c = static_cast<unsigned char>(s[s.size() - k]);
    if ((c >> 6) == 2) continue;  // continuation byte
    const size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (len > k) best = std::max(best, k);
    break;
  }
  return best;
}
}  // namespace

Glm5OutputParser::Glm5OutputParser(const Value* tools) {
  if (tools && tools->is_array() && !tools->items().empty()) tools_ = *tools;
}

void Glm5OutputParser::feed(std::string_view text, ParsedDelta& out) {
  buf_.append(text);
  step(out, false);
}

void Glm5OutputParser::finish(ParsedDelta& out) {
  step(out, true);
  if (!tools_seen_) out.content += hold_c_;  // no call followed: the content is kept as it was
  hold_c_.clear();
  hold_r_.clear();
  if (state_ == State::ToolName && py_strip(name_).empty()) {  // a bare <tool_call>: no call
    if (content_text_) out.content += call_ws_;
  } else if (state_ == State::ToolName || state_ == State::ToolArgs) {
    emit_call(true, out);
  }
}

// Reasoning / content text; whitespace is held until other text follows it.
void Glm5OutputParser::text_out(std::string_view s, ParsedDelta& out) {
  if (s.empty() || (state_ == State::Content && tools_seen_)) return;
  std::string& hold = state_ == State::Reasoning ? hold_r_ : hold_c_;
  std::string all = hold + std::string(s);
  const size_t keep = trail_space(all);
  (state_ == State::Reasoning ? out.reasoning : out.content).append(all, 0, keep);
  if (state_ == State::Content && keep > 0) content_text_ = true;
  hold = all.substr(keep);
}

void Glm5OutputParser::emit_call(bool partial, ParsedDelta& out) {
  const std::string name(py_strip(name_));
  name_.clear();
  const std::string raw = std::move(args_);
  args_.clear();
  if (tools_.is_array() && !tool_declared(tools_, name)) return;  // vLLM: validate_tool_names
  Value params = convert_args(raw, partial);
  if (const Value* props = tools_.is_array() ? tool_properties(tools_, name) : nullptr)
    for (auto& [k, v] : params.members())
      if (const Value* schema = props->find(k)) v = glm5_coerce(v.str(), *schema);
  out.tool_calls.push_back({name, params.dump_py()});
  ++n_calls_;
}

void Glm5OutputParser::step(ParsedDelta& out, bool final) {
  for (;;) {
    // the terminals this state reacts to (the others are ordinary text here)
    size_t pos = std::string::npos;
    std::string_view hit;
    const auto look = [&](std::string_view t) {
      const size_t p = buf_.find(t);
      if (p < pos) pos = p, hit = t;
    };
    if (state_ == State::ToolName) look(kKeyOpen), look(kCallClose);
    else if (state_ == State::ToolArgs) look(kCallClose);
    else look(kThinkOpen), look(kThinkClose), look(kCallOpen);
    const size_t n = pos != std::string::npos ? pos : final ? buf_.size() : buf_.size() - held_tail(buf_);
    const std::string_view text = std::string_view(buf_).substr(0, n);
    if (state_ == State::ToolName) name_.append(text);
    else if (state_ == State::ToolArgs) args_.append(text);
    else text_out(text, out);
    if (pos == std::string::npos) {
      buf_.erase(0, n);
      return;
    }
    buf_.erase(0, pos + hit.size());
    if (hit == kCallOpen) {  // ends the content (whitespace before it dropped) and the reasoning
      call_ws_ = tools_seen_ ? std::string() : std::move(hold_c_);
      hold_c_.clear();
      name_.clear();
      args_.clear();
      tools_seen_ = true;
      state_ = State::ToolName;
    } else if (hit == kThinkClose) {
      state_ = State::Content;
    } else if (hit == kThinkOpen) {
      state_ = State::Reasoning;
    } else if (hit == kKeyOpen) {
      args_.assign(kKeyOpen);
      state_ = State::ToolArgs;
    } else if (hit == kCallClose) {
      emit_call(false, out);
      state_ = State::Content;
    }
  }
}

namespace {
class Glm5ChatFormat final : public ChatFormat {
 public:
  RenderedChat render(const Value& request) const override { return glm5_render_request(request); }
  std::unique_ptr<ChatParser> parser(bool thinking, const Value* tools) const override {
    (void)thinking;  // the template always opens <think>
    return std::make_unique<Glm5OutputParser>(tools);
  }
};
}  // namespace

const ChatFormat& glm5_chat_format() {
  static const Glm5ChatFormat format;
  return format;
}

}  // namespace unholy::text
