#include "text/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace unholy::json {

namespace {


void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += char(cp);
  } else if (cp < 0x800) {
    out += char(0xc0 | (cp >> 6));
    out += char(0x80 | (cp & 0x3f));
  } else if (cp < 0x10000) {
    out += char(0xe0 | (cp >> 12));
    out += char(0x80 | ((cp >> 6) & 0x3f));
    out += char(0x80 | (cp & 0x3f));
  } else {
    out += char(0xf0 | (cp >> 18));
    out += char(0x80 | ((cp >> 12) & 0x3f));
    out += char(0x80 | ((cp >> 6) & 0x3f));
    out += char(0x80 | (cp & 0x3f));
  }
}

struct Parser {
  std::string_view s;
  size_t i = 0;
  static constexpr int kMaxDepth = 512;

  [[noreturn]] void fail(const char* what) const {
    throw ParseError(std::string("json: ") + what + " at offset " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  bool lit(std::string_view w) {
    if (s.substr(i, w.size()) != w) return false;
    i += w.size();
    return true;
  }
  Value value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    ws();
    if (i >= s.size()) fail("unexpected end");
    const char c = s[i];
    if (c == '{') return object(depth);
    if (c == '[') return array(depth);
    if (c == '"') return Value(string());
    if (lit("true")) return Value(true);
    if (lit("false")) return Value(false);
    if (lit("null")) return Value();
    // Python's json accepts these too.
    if (lit("NaN")) return Value::number_text("NaN");
    if (lit("Infinity")) return Value::number_text("Infinity");
    if (lit("-Infinity")) return Value::number_text("-Infinity");
    if (c == '-' || (c >= '0' && c <= '9')) return number();
    fail("unexpected character");
  }
  Value object(int depth) {
    ++i;
    Value v = Value::object();
    ws();
    if (i < s.size() && s[i] == '}') {
      ++i;
      return v;
    }
    // Duplicate keys: the last value wins in the first key's position (Python dict). Linear lookup
    // while small, a hash index once an object gets large (tokenizer.json's vocab has 128k members).
    std::unordered_map<std::string, size_t> index;
    auto& members = v.members();
    for (;;) {
      ws();
      if (i >= s.size() || s[i] != '"') fail("expected key");
      std::string k = string();
      ws();
      if (i >= s.size() || s[i] != ':') fail("expected ':'");
      ++i;
      Value item = value(depth + 1);
      size_t dup = members.size();
      if (members.size() < 32) {
        for (size_t m = 0; m < members.size(); ++m)
          if (members[m].first == k) dup = m;
      } else {
        if (index.empty())
          for (size_t m = 0; m < members.size(); ++m) index.emplace(members[m].first, m);
        const auto [it, fresh] = index.emplace(k, members.size());
        if (!fresh) dup = it->second;
      }
      if (dup < members.size()) members[dup].second = std::move(item);
      else members.emplace_back(std::move(k), std::move(item));
      ws();
      if (i < s.size() && s[i] == ',') {
        ++i;
        continue;
      }
      if (i < s.size() && s[i] == '}') {
        ++i;
        return v;
      }
      fail("expected ',' or '}'");
    }
  }
  Value array(int depth) {
    ++i;
    Value v = Value::array();
    ws();
    if (i < s.size() && s[i] == ']') {
      ++i;
      return v;
    }
    for (;;) {
      v.push_back(value(depth + 1));
      ws();
      if (i < s.size() && s[i] == ',') {
        ++i;
        continue;
      }
      if (i < s.size() && s[i] == ']') {
        ++i;
        return v;
      }
      fail("expected ',' or ']'");
    }
  }
  uint32_t hex4() {
    if (i + 4 > s.size()) fail("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else fail("bad \\u escape");
    }
    return v;
  }
  std::string string() {
    ++i;
    std::string out;
    for (;;) {
      const size_t start = i;
      while (i < s.size() && s[i] != '"' && s[i] != '\\' && static_cast<unsigned char>(s[i]) >= 0x20) ++i;
      out.append(s.data() + start, i - start);
      if (i >= s.size()) fail("unterminated string");
      const char c = s[i];
      if (c == '"') {
        ++i;
        return out;
      }
      if (c != '\\') fail("control character in string");
      ++i;
      if (i >= s.size()) fail("unterminated escape");
      switch (s[i++]) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xd800 && cp < 0xdc00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            const size_t save = i;
            i += 2;
            const uint32_t lo = hex4();
            if (lo >= 0xdc00 && lo < 0xe000) cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            else i = save;  // lone high surrogate; the next escape stands on its own
          }
          append_utf8(out, cp);  // lone surrogates come out as WTF-8, as Python would hold them
          break;
        }
        default: fail("bad escape");
      }
    }
  }
  Value number() {
    const size_t start = i;
    if (s[i] == '-') ++i;
    if (i >= s.size()) fail("bad number");
    if (s[i] == '0') {
      ++i;
    } else if (s[i] >= '1' && s[i] <= '9') {
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    } else {
      fail("bad number");
    }
    if (i < s.size() && s[i] == '.') {
      ++i;
      if (i >= s.size() || s[i] < '0' || s[i] > '9') fail("bad fraction");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
      if (i >= s.size() || s[i] < '0' || s[i] > '9') fail("bad exponent");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    return Value::number_text(std::string(s.substr(start, i - start)));
  }
};

}  // namespace

std::string py_float_repr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  // Shortest round-trip digits in scientific form, then Python's float_repr_style='short' layout:
  // exponent form when the decimal point position is <= -4 or > 16, else positional with ".0".
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
  std::string sci(buf, r.ptr);
  std::string out;
  size_t p = 0;
  if (sci[0] == '-') {
    out += '-';
    p = 1;
  }
  const size_t epos = sci.find('e');
  std::string digits;
  for (size_t k = p; k < epos; ++k)
    if (sci[k] != '.') digits += sci[k];
  const int exp10 = std::atoi(sci.c_str() + epos + 1);
  const int decpt = exp10 + 1, n = int(digits.size());
  if (decpt <= -4 || decpt > 16) {
    out += digits[0];
    if (n > 1) {
      out += '.';
      out.append(digits, 1);
    }
    char e[16];
    std::snprintf(e, sizeof(e), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    out += e;
  } else if (decpt <= 0) {
    out += "0.";
    out.append(size_t(-decpt), '0');
    out += digits;
  } else if (decpt < n) {
    out.append(digits, 0, size_t(decpt));
    out += '.';
    out.append(digits, size_t(decpt));
  } else {
    out += digits;
    out.append(size_t(decpt - n), '0');
    out += ".0";
  }
  return out;
}

Value::Value(double v) : type_(Type::Number), s_(py_float_repr(v)) {}

Value Value::number_text(std::string text) {
  Value v;
  v.type_ = Type::Number;
  v.s_ = std::move(text);
  return v;
}

bool Value::truthy() const {
  switch (type_) {
    case Type::Null: return false;
    case Type::Bool: return b_;
    case Type::Number: return as_double() != 0.0;
    case Type::String: return !s_.empty();
    case Type::Array: return !arr_.empty();
    case Type::Object: return !obj_.empty();
  }
  return false;
}

double Value::as_double() const {
  if (type_ != Type::Number) return 0.0;
  if (s_ == "NaN") return NAN;
  if (s_ == "Infinity") return INFINITY;
  if (s_ == "-Infinity") return -INFINITY;
  return std::strtod(s_.c_str(), nullptr);
}

long long Value::as_int() const {
  if (type_ != Type::Number) return 0;
  if (number_is_int()) return std::strtoll(s_.c_str(), nullptr, 10);
  return static_cast<long long>(as_double());
}

bool Value::number_is_int() const {
  return type_ == Type::Number && s_.find_first_of(".eEIN") == std::string::npos;
}

const Value* Value::find(std::string_view key) const {
  if (type_ != Type::Object) return nullptr;
  for (const auto& [k, v] : obj_)
    if (k == key) return &v;
  return nullptr;
}

Value* Value::find(std::string_view key) {
  return const_cast<Value*>(static_cast<const Value*>(this)->find(key));
}

Value& Value::set(std::string key, Value v) {
  if (type_ == Type::Null) type_ = Type::Object;
  if (type_ != Type::Object) throw std::runtime_error("json: set on non-object");
  if (Value* e = find(key)) return *e = std::move(v);
  obj_.emplace_back(std::move(key), std::move(v));
  return obj_.back().second;
}

bool Value::erase(std::string_view key) {
  for (auto it = obj_.begin(); it != obj_.end(); ++it)
    if (it->first == key) {
      obj_.erase(it);
      return true;
    }
  return false;
}

Value& Value::operator[](std::string_view key) {
  if (Value* e = find(key)) return *e;
  return set(std::string(key), Value());
}

const Value& Value::at(std::string_view key) const {
  if (const Value* e = find(key)) return *e;
  throw std::runtime_error("json: missing key '" + std::string(key) + "'");
}

std::string Value::get_str(std::string_view key, std::string_view def) const {
  const Value* e = find(key);
  return e && e->is_string() ? e->s_ : std::string(def);
}

void Value::push_back(Value v) {
  if (type_ == Type::Null) type_ = Type::Array;
  if (type_ != Type::Array) throw std::runtime_error("json: push_back on non-array");
  arr_.push_back(std::move(v));
}

void escape_to(std::string& out, std::string_view s) {
  out += '"';
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char u[8];
          std::snprintf(u, sizeof(u), "\\u%04x", c);
          out += u;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

void Value::dump_to(std::string& out) const {
  switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Number: out += s_; break;
    case Type::String: escape_to(out, s_); break;
    case Type::Array:
      out += '[';
      for (size_t k = 0; k < arr_.size(); ++k) {
        if (k) out += ',';
        arr_[k].dump_to(out);
      }
      out += ']';
      break;
    case Type::Object:
      out += '{';
      for (size_t k = 0; k < obj_.size(); ++k) {
        if (k) out += ',';
        escape_to(out, obj_[k].first);
        out += ':';
        obj_[k].second.dump_to(out);
      }
      out += '}';
      break;
  }
}

std::string Value::dump() const {
  std::string out;
  dump_to(out);
  return out;
}

void Value::dump_py_to(std::string& out) const {
  switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Number:
      if (s_ == "NaN" || s_ == "Infinity" || s_ == "-Infinity") {
        out += s_;
      } else if (number_is_int()) {
        // Python int: "-0" prints as "0".
        if (s_ == "-0") out += '0';
        else out += s_;
      } else {
        out += py_float_repr(std::strtod(s_.c_str(), nullptr));
      }
      break;
    case Type::String: escape_to(out, s_); break;
    case Type::Array:
      out += '[';
      for (size_t k = 0; k < arr_.size(); ++k) {
        if (k) out += ", ";
        arr_[k].dump_py_to(out);
      }
      out += ']';
      break;
    case Type::Object:
      out += '{';
      for (size_t k = 0; k < obj_.size(); ++k) {
        if (k) out += ", ";
        escape_to(out, obj_[k].first);
        out += ": ";
        obj_[k].second.dump_py_to(out);
      }
      out += '}';
      break;
  }
}

std::string Value::dump_py() const {
  std::string out;
  dump_py_to(out);
  return out;
}

Value parse(std::string_view text) {
  Parser p{text};
  Value v = p.value(0);
  p.ws();
  if (p.i != text.size()) p.fail("trailing characters");
  return v;
}

}  // namespace unholy::json
