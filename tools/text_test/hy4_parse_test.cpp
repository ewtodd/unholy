// unholy-hy4-parse-test: Hy4OutputParser against the expected results in hy4_parse_cases.json, each case fed in
// every chunk size of the file. Needs neither vLLM nor a GPU (tools/ref/hy4_chat_compare.py needs vLLM and cannot
// cover the departures from it: cut-off blocks, invalid names, text before <think>):
//
//   g++ -std=c++20 -Isrc tools/text_test/hy4_parse_test.cpp src/text/hy4_chat.cpp src/text/json.cpp -o hy4-parse-test
//   ./hy4-parse-test tools/text_test/hy4_parse_cases.json
//
// A case: {name, text, thinking?, tools?: false, reasoning, content, tool_calls: [{name, arguments}], log: [prefix]}.
// Omitted results are expected empty; "$" in text / reasoning / content / log is the marker suffix ":opensource".
// saw_tool_calls() must agree with tool_calls: it is what makes a stream's finish_reason tool_calls.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "text/hy4_chat.hpp"
#include "text/json.hpp"

using namespace unholy;
using json::Value;

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string suffixed(std::string s) {
  for (size_t p = s.find('$'); p != std::string::npos; p = s.find('$', p + 11)) s.replace(p, 1, ":opensource");
  return s;
}

Value parse(const std::string& t, size_t chunk, bool thinking, const Value* tools, bool& saw) {
  text::Hy4OutputParser p(thinking, tools);
  text::ParsedDelta d;
  for (size_t i = 0; i < t.size(); i += chunk) p.feed(std::string_view(t).substr(i, chunk), d);
  p.finish(d);
  saw = p.saw_tool_calls();
  Value r = Value::object();
  r.set("reasoning", Value(d.reasoning));
  r.set("content", Value(d.content));
  Value calls = Value::array();
  for (const auto& tc : d.tool_calls) {
    Value call = Value::object();
    call.set("name", Value(tc.name));
    call.set("arguments", Value(tc.arguments));
    calls.push_back(std::move(call));
  }
  r.set("tool_calls", std::move(calls));
  Value log = Value::array();
  for (const auto& line : d.log) log.push_back(Value(line));
  r.set("log", std::move(log));
  return r;
}

// Same shape as parse()'s result, with each log line cut to the length of the expected prefix.
bool same(const Value& want, const Value& got, const Value& want_log) {
  for (const char* k : {"reasoning", "content"})
    if (want.get_str(k) != got.get_str(k)) return false;
  if (want.at("tool_calls").dump() != got.at("tool_calls").dump()) return false;
  const auto& g = got.at("log").items();
  if (want_log.size() != g.size()) return false;
  for (size_t i = 0; i < g.size(); ++i)
    if (g[i].str().rfind(want_log.items()[i].str(), 0) != 0) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: unholy-hy4-parse-test <hy4_parse_cases.json>\n");
    return 2;
  }
  try {
    const Value file = json::parse(slurp(argv[1]));
    const Value& tools = file.at("tools");
    int runs = 0, failed = 0;
    for (const Value& c : file.at("cases").items()) {
      const std::string name = c.at("name").str();
      Value want = Value::object();
      want.set("reasoning", Value(suffixed(c.get_str("reasoning"))));
      want.set("content", Value(suffixed(c.get_str("content"))));
      const Value* wc = c.find("tool_calls");
      want.set("tool_calls", wc ? *wc : Value::array());
      Value want_log = Value::array();
      if (const Value* l = c.find("log"))
        for (const Value& line : l->items()) want_log.push_back(Value(suffixed(line.str())));
      const Value* use = c.find("tools");
      const bool with_tools = !use || use->truthy();
      const Value* th = c.find("thinking");
      const bool thinking = !th || th->truthy();
      for (const Value& ch : file.at("chunks").items()) {
        const auto chunk = size_t(ch.as_int());
        bool saw = false;
        const Value got = parse(suffixed(c.at("text").str()), chunk, thinking, with_tools ? &tools : nullptr, saw);
        ++runs;
        if (!same(want, got, want_log) || saw != (got.at("tool_calls").size() > 0)) {
          ++failed;
          std::printf("FAIL %s (chunk %zu, saw_tool_calls %d)\n  want %s log %s\n  got  %s\n", name.c_str(), chunk,
                      int(saw), want.dump().c_str(), want_log.dump().c_str(), got.dump().c_str());
        }
      }
      std::printf("%-78s %s\n", name.c_str(), want.at("tool_calls").size() > 0 ? "call" : "no call");
    }
    std::printf("%d runs, %d failed\n", runs, failed);
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
