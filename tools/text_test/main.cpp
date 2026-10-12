// unholy-text-test: drives the tokenizer and the chat format for comparison against the Python
// references (tools/ref/tok_compare.py, tools/ref/chat_compare.py).
//
//   unholy-text-test tokenize <tokenizer.json> <strings.json>   JSON array of strings -> array of id arrays
//   unholy-text-test decode <tokenizer.json> <ids.json>         array of id arrays -> array of strings
//   unholy-text-test chat <cases.json>                          array of encode cases -> array of prompts
//   unholy-text-test request <requests.json>                    array of OpenAI requests -> array of prompts
//   unholy-text-test parse <cases.json>                         array of {text, thinking, chunk} -> parsed
//   unholy-text-test bench <tokenizer.json> <file>              tokenization throughput on a text file
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "text/chat.hpp"
#include "text/json.hpp"
#include "text/tokenizer.hpp"

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

Value error_value(const std::exception& e) {
  Value v = Value::object();
  v.set("error", Value(std::string(e.what())));
  return v;
}

int usage() {
  std::fprintf(stderr, "usage: unholy-text-test tokenize|decode|chat|request|parse|bench ...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string cmd = argv[1];
  try {
    Value out = Value::array();
    if (cmd == "tokenize" && argc == 4) {
      const auto tok = text::Tokenizer::load(argv[2]);
      const Value in = json::parse(slurp(argv[3]));
      for (const Value& s : in.items()) {
        Value ids = Value::array();
        for (const int32_t id : tok.encode(s.str())) ids.push_back(Value(int(id)));
        out.push_back(std::move(ids));
      }
    } else if (cmd == "decode" && argc == 4) {
      const auto tok = text::Tokenizer::load(argv[2]);
      const Value in = json::parse(slurp(argv[3]));
      for (const Value& a : in.items()) {
        std::vector<int32_t> ids;
        for (const Value& v : a.items()) ids.push_back(int32_t(v.as_int()));
        // Streamed through the Detokenizer, which must agree with a lossy decode of the whole.
        text::Detokenizer dt(tok);
        std::string streamed;
        for (const int32_t id : ids) dt.push(id, streamed);
        dt.finish(streamed);
        if (streamed != text::sanitize_utf8(tok.decode(ids))) throw std::runtime_error("streamed decode differs");
        out.push_back(Value(streamed));
      }
    } else if (cmd == "chat" && argc == 3) {
      const Value in = json::parse(slurp(argv[2]));
      for (const Value& c : in.items()) {
        try {
          text::EncodeOptions opt;
          opt.thinking = c.get_str("thinking_mode", "chat") == "thinking";
          if (const Value* e = c.find("reasoning_effort"); e && e->is_string()) opt.reasoning_effort = e->str();
          if (const Value* d = c.find("drop_thinking")) opt.drop_thinking = d->truthy();
          if (const Value* b = c.find("add_bos")) opt.add_bos = b->truthy();
          const Value* ctx = c.find("context");
          out.push_back(Value(text::encode_messages(c.at("messages"), opt, ctx && !ctx->is_null() ? ctx : nullptr)));
        } catch (const std::exception& e) {
          out.push_back(error_value(e));
        }
      }
    } else if (cmd == "request" && argc == 3) {
      const Value in = json::parse(slurp(argv[2]));
      for (const Value& r : in.items()) {
        try {
          out.push_back(Value(text::render_request(r).prompt));
        } catch (const std::exception& e) {
          out.push_back(error_value(e));
        }
      }
    } else if (cmd == "parse" && argc == 3) {
      const Value in = json::parse(slurp(argv[2]));
      for (const Value& c : in.items()) {
        const std::string& t = c.at("text").str();
        const auto chunk = size_t(std::max<long long>(1, c.contains("chunk") ? c.at("chunk").as_int() : 1 << 30));
        const Value* tools = c.find("tools");
        text::OutputParser p(c.at("thinking").truthy(), tools);
        text::ParsedDelta d;
        for (size_t i = 0; i < t.size(); i += chunk) p.feed(std::string_view(t).substr(i, chunk), d);
        p.finish(d);
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
        out.push_back(std::move(r));
      }
    } else if (cmd == "bench" && argc == 4) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto tok = text::Tokenizer::load(argv[2]);
      const auto t1 = std::chrono::steady_clock::now();
      const std::string s = slurp(argv[3]);
      const auto ids = tok.encode(s);
      const auto t2 = std::chrono::steady_clock::now();
      const double load_s = std::chrono::duration<double>(t1 - t0).count();
      const double enc_s = std::chrono::duration<double>(t2 - t1).count();
      std::printf("load %.3f s; %zu bytes -> %zu tokens in %.3f s (%.1f MB/s, %.0f tok/s)\n", load_s, s.size(),
                  ids.size(), enc_s, s.size() / enc_s / 1e6, ids.size() / enc_s);
      return 0;
    } else {
      return usage();
    }
    std::string s = out.dump();
    s += '\n';
    std::fwrite(s.data(), 1, s.size(), stdout);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
