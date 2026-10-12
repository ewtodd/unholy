// Ordered JSON for the serving layer.
//
// Two things the vendored gufo JSON does not do and the DSV4 chat format needs: numbers keep their
// source text (tool arguments are re-serialized, and Python's json round trip prints ints as ints and
// floats as repr(float)), and dump_py() reproduces json.dumps(v, ensure_ascii=False) byte for byte
// (", " and ": " separators), which is what encoding_dsv4.py embeds in prompts.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace unholy::json {

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Value {
 public:
  enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };
  using Array = std::vector<Value>;
  using Object = std::vector<std::pair<std::string, Value>>;

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : type_(Type::Bool), b_(b) {}
  Value(int v) : Value(static_cast<long long>(v)) {}
  Value(long v) : Value(static_cast<long long>(v)) {}
  Value(long long v) : type_(Type::Number), s_(std::to_string(v)) {}
  Value(unsigned long v) : type_(Type::Number), s_(std::to_string(v)) {}
  Value(unsigned long long v) : type_(Type::Number), s_(std::to_string(v)) {}
  Value(double v);
  Value(const char* s) : type_(Type::String), s_(s) {}
  Value(std::string s) : type_(Type::String), s_(std::move(s)) {}
  Value(std::string_view s) : type_(Type::String), s_(s) {}
  static Value number_text(std::string text);  // a JSON number literal, kept verbatim
  static Value array() { Value v; v.type_ = Type::Array; return v; }
  static Value object() { Value v; v.type_ = Type::Object; return v; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }
  // Python truthiness: null/false/0/""/[]/{} are false.
  bool truthy() const;

  bool as_bool() const { return b_; }
  const std::string& str() const { return s_; }        // string payload, or a number's text
  double as_double() const;
  long long as_int() const;
  bool number_is_int() const;  // no '.', 'e' or 'E' in the literal (Python json -> int)

  // Objects: insertion order; set() replaces an existing key in place (Python dict semantics).
  const Object& members() const { return obj_; }
  Object& members() { return obj_; }
  const Value* find(std::string_view key) const;
  Value* find(std::string_view key);
  bool contains(std::string_view key) const { return find(key) != nullptr; }
  Value& set(std::string key, Value v);
  bool erase(std::string_view key);
  Value& operator[](std::string_view key);  // inserts null when absent (object or null only)
  const Value& at(std::string_view key) const;  // throws when absent
  // Member as string: def when absent, null, or not a string.
  std::string get_str(std::string_view key, std::string_view def = "") const;

  const Array& items() const { return arr_; }
  Array& items() { return arr_; }
  void push_back(Value v);
  size_t size() const { return type_ == Type::Array ? arr_.size() : type_ == Type::Object ? obj_.size() : 0; }

  // Compact JSON (no spaces), numbers verbatim.
  std::string dump() const;
  void dump_to(std::string& out) const;
  // json.dumps(self, ensure_ascii=False) as Python prints it.
  std::string dump_py() const;
  void dump_py_to(std::string& out) const;

 private:
  Type type_ = Type::Null;
  bool b_ = false;
  std::string s_;
  Array arr_;
  Object obj_;
};

Value parse(std::string_view text);
// JSON string escaping as both compact dump and Python use it (ensure_ascii=False).
void escape_to(std::string& out, std::string_view s);
// Python repr(float(text)) for a JSON number literal.
std::string py_float_repr(double v);

}  // namespace unholy::json
