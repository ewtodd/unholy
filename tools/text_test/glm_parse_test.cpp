// unholy-glm-parse-test: Glm5OutputParser against the expected results in glm_parse_cases.json, each case fed
// in every chunk size of the file. Needs neither vLLM nor a GPU (tools/ref/glm_chat_compare.py needs vLLM and
// cannot cover calls to undeclared tools, where the parser departs from it on purpose):
//
//   g++ -std=c++20 -Isrc tools/text_test/glm_parse_test.cpp src/text/glm_chat.cpp src/text/json.cpp -o glm-parse-test
//   ./glm-parse-test tools/text_test/glm_parse_cases.json
//
// A case: {name, text, reasoning, content, tool_calls: [{name, arguments}], log: [line], tools?: false}. Omitted
// results are expected empty; tools: false runs the case on a request without tools. saw_tool_calls() must agree
// with tool_calls: it is what makes a stream's finish_reason tool_calls.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "text/glm_chat.hpp"
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

Value parse(const std::string& t, size_t chunk, const Value* tools, bool& saw) {
  text::Glm5OutputParser p(tools);
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

Value expected(const Value& c) {
  Value r = Value::object();
  r.set("reasoning", Value(c.get_str("reasoning")));
  r.set("content", Value(c.get_str("content")));
  for (const char* k : {"tool_calls", "log"}) {
    const Value* v = c.find(k);
    r.set(k, v ? *v : Value::array());
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: unholy-glm-parse-test <glm_parse_cases.json>\n");
    return 2;
  }
  try {
    const Value file = json::parse(slurp(argv[1]));
    const Value& tools = file.at("tools");
    int runs = 0, failed = 0;
    for (const Value& c : file.at("cases").items()) {
      const std::string name = c.at("name").str();
      const Value want = expected(c);
      const Value* use = c.find("tools");
      const bool with_tools = !use || use->truthy();
      for (const Value& ch : file.at("chunks").items()) {
        const auto chunk = size_t(ch.as_int());
        bool saw = false;
        const Value got = parse(c.at("text").str(), chunk, with_tools ? &tools : nullptr, saw);
        ++runs;
        const bool ok = got.dump() == want.dump() && saw == (got.at("tool_calls").size() > 0);
        if (!ok) {
          ++failed;
          std::printf("FAIL %s (chunk %zu, saw_tool_calls %d)\n  want %s\n  got  %s\n", name.c_str(), chunk, int(saw),
                      want.dump().c_str(), got.dump().c_str());
        }
      }
      std::printf("%-58s %s\n", name.c_str(), want.at("tool_calls").size() > 0 ? "call" : "no call");
    }
    std::printf("%d runs, %d failed\n", runs, failed);
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
