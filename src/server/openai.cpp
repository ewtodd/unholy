#include "server/openai.hpp"

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <random>

#include "text/chat.hpp"
#include "text/json.hpp"

namespace unholy::server {

using json::Value;

// ---- stop strings -------------------------------------------------------------------------------

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
  stops_.erase(std::remove_if(stops_.begin(), stops_.end(), [](const std::string& s) { return s.empty(); }),
               stops_.end());
  for (const auto& s : stops_) hold_ = std::max(hold_, s.size() - 1);
}

bool StopMatcher::feed(std::string_view text, std::string& out) {
  if (stopped_) return true;
  if (stops_.empty()) {
    out += text;
    return false;
  }
  const size_t scan_from = buf_.size() > hold_ ? buf_.size() - hold_ : 0;
  buf_ += text;
  size_t best = std::string::npos;
  for (const auto& s : stops_) best = std::min(best, buf_.find(s, scan_from));
  if (best != std::string::npos) {
    out.append(buf_, 0, best);
    buf_.clear();
    stopped_ = true;
    return true;
  }
  // Keep the last hold_ bytes, moved back to a UTF-8 character boundary.
  size_t emit = buf_.size() > hold_ ? buf_.size() - hold_ : 0;
  while (emit > 0 && emit < buf_.size() && (static_cast<unsigned char>(buf_[emit]) & 0xc0) == 0x80) --emit;
  out.append(buf_, 0, emit);
  buf_.erase(0, emit);
  return false;
}

void StopMatcher::finish(std::string& out) {
  if (!stopped_) out += buf_;
  buf_.clear();
}

namespace {

int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string random_hex(size_t n) {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string s(n, '0');
  for (auto& c : s) c = hex[rng() & 15];
  return s;
}

struct HttpError : std::runtime_error {
  int status;
  std::string type;
  HttpError(int st, std::string msg, std::string t = "BadRequestError")
      : std::runtime_error(std::move(msg)), status(st), type(std::move(t)) {}
};

Value error_body(const std::string& msg, const std::string& type, int code) {
  Value e = Value::object();
  e.set("message", Value(msg));
  e.set("type", Value(type));
  e.set("param", Value());
  e.set("code", Value(code));
  Value b = Value::object();
  b.set("error", std::move(e));
  return b;
}

// Engine deliveries for one request, consumed by its HTTP thread.
struct Channel {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<engine::GenOutput> q;
  void push(engine::GenOutput&& o) {
    {
      std::lock_guard lk(mu);
      q.push_back(std::move(o));
    }
    cv.notify_one();
  }
  bool pop(engine::GenOutput& o, std::chrono::milliseconds wait) {
    std::unique_lock lk(mu);
    if (!cv.wait_for(lk, wait, [&] { return !q.empty(); })) return false;
    o = std::move(q.front());
    q.pop_front();
    return true;
  }
};

double num(const Value& req, std::string_view key, double def) {
  const Value* v = req.find(key);
  if (!v || v->is_null()) return def;
  if (!v->is_number()) throw HttpError(400, "'" + std::string(key) + "' must be a number");
  return v->as_double();
}

struct Params {
  engine::GenRequest gen;
  std::vector<std::string> stop;
  bool stream = false, include_usage = false;
  bool logprobs = false;
  int top_logprobs = 0;
  bool echo = false;
};

Params parse_params(const Value& req, bool chat, int default_max_tokens) {
  Params p;
  auto& s = p.gen.sampling;
  s.temperature = float(num(req, "temperature", 1.0));
  s.top_p = float(num(req, "top_p", 1.0));
  s.top_k = int(num(req, "top_k", 0));
  s.min_p = float(num(req, "min_p", 0.0));
  s.presence_penalty = float(num(req, "presence_penalty", 0.0));
  s.frequency_penalty = float(num(req, "frequency_penalty", 0.0));
  s.repetition_penalty = float(num(req, "repetition_penalty", 1.0));
  if (s.temperature < 0 || s.top_p <= 0 || s.top_p > 1 || s.min_p < 0 || s.min_p > 1 || s.repetition_penalty <= 0)
    throw HttpError(400, "sampling parameter out of range");
  if (const Value* v = req.find("seed"); v && v->is_number()) s.seed = uint64_t(v->as_int());
  const Value* mt = chat ? req.find("max_completion_tokens") : nullptr;
  if (!mt || mt->is_null()) mt = req.find("max_tokens");
  p.gen.max_tokens = mt && mt->is_number() ? int(mt->as_int()) : default_max_tokens;
  if (mt && mt->is_number() && p.gen.max_tokens <= 0) throw HttpError(400, "max_tokens must be at least 1");
  if (const Value* v = req.find("stop"); v && !v->is_null()) {
    if (v->is_string()) p.stop.push_back(v->str());
    else if (v->is_array())
      for (const Value& x : v->items()) p.stop.push_back(x.str());
    else throw HttpError(400, "'stop' must be a string or an array of strings");
  }
  if (const Value* v = req.find("stop_token_ids"); v && v->is_array())
    for (const Value& x : v->items()) p.gen.stop_token_ids.push_back(int32_t(x.as_int()));
  if (const Value* v = req.find("ignore_eos")) p.gen.ignore_eos = v->truthy();
  if (const Value* v = req.find("n"); v && v->is_number() && v->as_int() != 1)
    throw HttpError(400, "only n = 1 is supported");
  if (const Value* v = req.find("stream")) p.stream = v->truthy();
  if (const Value* so = req.find("stream_options"); so && so->is_object())
    if (const Value* v = so->find("include_usage")) p.include_usage = v->truthy();
  if (chat) {
    if (const Value* v = req.find("logprobs")) p.logprobs = v->truthy();
    if (const Value* v = req.find("top_logprobs"); v && v->is_number()) p.top_logprobs = int(v->as_int());
  } else {
    if (const Value* v = req.find("logprobs"); v && v->is_number()) {
      p.logprobs = true;
      p.top_logprobs = int(v->as_int());
    }
    if (const Value* v = req.find("echo")) p.echo = v->truthy();
  }
  if (p.top_logprobs < 0 || p.top_logprobs > 20) throw HttpError(400, "top_logprobs must be in [0, 20]");
  s.logprobs = p.logprobs;
  s.top_logprobs = p.top_logprobs;
  return p;
}

Value usage_json(int prompt, int completion) {
  Value u = Value::object();
  u.set("prompt_tokens", Value(prompt));
  u.set("completion_tokens", Value(completion));
  u.set("total_tokens", Value(prompt + completion));
  return u;
}

std::string sse(const Value& v) { return "data: " + v.dump() + "\n\n"; }

}  // namespace

// ---- the server ---------------------------------------------------------------------------------

struct OpenAIServer::Impl {
  engine::Engine& eng;
  const text::Tokenizer& tok;
  ServerConfig cfg;
  httplib::Server svr;
  std::atomic<uint64_t> requests{0}, errors{0};

  Impl(engine::Engine& e, const text::Tokenizer& t, ServerConfig c) : eng(e), tok(t), cfg(std::move(c)) {}

  // Per request text state: detokenizer -> stop strings -> (chat) output parser.
  struct TextState {
    text::Detokenizer detok;
    StopMatcher stop;
    std::optional<text::OutputParser> parser;
    TextState(const text::Tokenizer& t, std::vector<std::string> stops, bool chat, bool thinking, const Value* tools)
        : detok(t, true), stop(std::move(stops)) {
      if (chat) parser.emplace(thinking, tools);
    }
    // Feeds tokens; returns new raw text (completions) or fills the parsed delta (chat).
    void push(const std::vector<int32_t>& toks, std::string& raw, text::ParsedDelta& d) {
      std::string piece, safe;
      for (const int32_t t : toks) detok.push(t, piece);
      stop.feed(piece, safe);
      emit(safe, raw, d);
    }
    void finish(std::string& raw, text::ParsedDelta& d) {
      std::string piece, safe;
      if (!stop.stopped()) {
        detok.finish(piece);
        stop.feed(piece, safe);
        stop.finish(safe);
      }
      emit(safe, raw, d);
      if (parser) parser->finish(d);
    }
    void emit(const std::string& s, std::string& raw, text::ParsedDelta& d) {
      if (parser) parser->feed(s, d);
      else raw += s;
    }
  };

  Value logprob_entry(const engine::Sampled& smp, bool chat) const {
    const auto piece = [&](int32_t id) { return text::sanitize_utf8(tok.piece(id)); };
    Value e = Value::object();
    e.set("token", Value(piece(smp.token)));
    e.set("logprob", Value(double(smp.logprob)));
    if (chat) {
      Value bytes = Value::array();
      for (const char c : tok.piece(smp.token)) bytes.push_back(Value(int(static_cast<unsigned char>(c))));
      e.set("bytes", std::move(bytes));
      Value top = Value::array();
      for (const auto& t : smp.top) {
        Value x = Value::object();
        x.set("token", Value(piece(t.token)));
        x.set("logprob", Value(double(t.logprob)));
        top.push_back(std::move(x));
      }
      e.set("top_logprobs", std::move(top));
    }
    return e;
  }

  void check_auth(const httplib::Request& req) const {
    if (cfg.api_key.empty()) return;
    if (req.get_header_value("Authorization") != "Bearer " + cfg.api_key)
      throw HttpError(401, "invalid API key", "AuthenticationError");
  }

  static void send_json(httplib::Response& res, const Value& v, int status = 200) {
    res.status = status;
    res.set_content(v.dump(), "application/json");
  }

  template <typename F>
  httplib::Server::Handler wrap(F f) {
    return [this, f](const httplib::Request& req, httplib::Response& res) {
      ++requests;
      try {
        f(req, res);
      } catch (const HttpError& e) {
        ++errors;
        send_json(res, error_body(e.what(), e.type, e.status), e.status);
      } catch (const text::ChatError& e) {
        ++errors;
        send_json(res, error_body(e.what(), "BadRequestError", 400), 400);
      } catch (const json::ParseError& e) {
        ++errors;
        send_json(res, error_body(std::string("invalid JSON: ") + e.what(), "BadRequestError", 400), 400);
      } catch (const std::exception& e) {
        ++errors;
        send_json(res, error_body(e.what(), "InternalServerError", 500), 500);
      }
    };
  }

  Value parse_body(const httplib::Request& req) const {
    Value v = json::parse(req.body);
    if (!v.is_object()) throw HttpError(400, "request body must be a JSON object");
    return v;
  }

  // Shared driver: submits, then streams or collects.
  struct Job {
    bool chat = false, thinking = false;
    Value tools;
    std::vector<int32_t> prompt;
    std::string prompt_text;  // completions echo
    Params p;
    std::string id;
    int64_t created = 0;
  };

  void run_job(const std::shared_ptr<Job>& job, const httplib::Request& req, httplib::Response& res) {
    auto ch = std::make_shared<Channel>();
    engine::GenRequest gen = job->p.gen;
    gen.prompt = job->prompt;
    const int ctx = eng.max_context();
    if (int(gen.prompt.size()) >= ctx)
      throw HttpError(400, "this model's maximum context length is " + std::to_string(ctx) +
                               " tokens; the prompt has " + std::to_string(gen.prompt.size()));
    const uint64_t rid = eng.submit(std::move(gen), [ch](engine::GenOutput&& o) { ch->push(std::move(o)); });
    auto state = std::make_shared<TextState>(tok, job->p.stop, job->chat, job->thinking,
                                             job->tools.is_null() ? nullptr : &job->tools);
    if (job->p.stream) stream(job, ch, state, rid, res);
    else collect(job, ch, state, rid, req, res);
  }

  Value chunk_base(const Job& job) const {
    Value c = Value::object();
    c.set("id", Value(job.id));
    c.set("object", Value(job.chat ? "chat.completion.chunk" : "text_completion"));
    c.set("created", Value(static_cast<long long>(job.created)));
    c.set("model", Value(cfg.model_name));
    return c;
  }

  std::string finish_name(engine::FinishReason r, bool stopped_by_string, bool tool_calls) const {
    if (stopped_by_string) return "stop";
    if (r == engine::FinishReason::Stop) return tool_calls ? "tool_calls" : "stop";
    return engine::finish_reason_name(r);
  }

  void stream(std::shared_ptr<Job> job, std::shared_ptr<Channel> ch, std::shared_ptr<TextState> st, uint64_t rid,
              httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, job, ch, st, rid](size_t, httplib::DataSink& sink) {
          const auto write = [&](const std::string& s) { return sink.write(s.data(), s.size()); };
          int n_calls = 0, prompt_tokens = int(job->prompt.size()), completion = 0;
          bool sent_role = false;
          const auto send_delta = [&](const text::ParsedDelta& d, const std::string& raw, const Value* lp) {
            Value choice = Value::object();
            choice.set("index", Value(0));
            if (job->chat) {
              Value delta = Value::object();
              if (!sent_role) {
                delta.set("role", Value("assistant"));
                sent_role = true;
              }
              if (!d.reasoning.empty()) {
                delta.set("reasoning", Value(d.reasoning));
                delta.set("reasoning_content", Value(d.reasoning));
              }
              if (!d.content.empty() || delta.contains("role")) delta.set("content", Value(d.content));
              if (!d.tool_calls.empty()) {
                Value calls = Value::array();
                for (const auto& tc : d.tool_calls) {
                  Value c = Value::object();
                  c.set("index", Value(n_calls++));
                  c.set("id", Value("call_" + random_hex(24)));
                  c.set("type", Value("function"));
                  Value fn = Value::object();
                  fn.set("name", Value(tc.name));
                  fn.set("arguments", Value(tc.arguments));
                  c.set("function", std::move(fn));
                  calls.push_back(std::move(c));
                }
                delta.set("tool_calls", std::move(calls));
              }
              if (delta.size() == 0 && !lp) return true;
              choice.set("delta", std::move(delta));
              Value lpv;
              if (lp) {
                lpv = Value::object();
                lpv.set("content", *lp);
              }
              choice.set("logprobs", std::move(lpv));
            } else {
              if (raw.empty() && !lp) return true;
              choice.set("text", Value(raw));
              choice.set("logprobs", lp ? *lp : Value());
            }
            choice.set("finish_reason", Value());
            Value c = chunk_base(*job);
            Value choices = Value::array();
            choices.push_back(std::move(choice));
            c.set("choices", std::move(choices));
            return write(sse(c));
          };
          const auto lp_json = [&](const engine::GenOutput& o) -> std::optional<Value> {
            if (!job->p.logprobs) return std::nullopt;
            if (job->chat) {
              Value arr = Value::array();
              for (const auto& smp : o.samples) arr.push_back(logprob_entry(smp, true));
              return arr;
            }
            Value l = Value::object(), toks = Value::array(), lps = Value::array(), tops = Value::array();
            for (const auto& smp : o.samples) {
              toks.push_back(Value(text::sanitize_utf8(tok.piece(smp.token))));
              lps.push_back(Value(double(smp.logprob)));
              Value t = Value::object();
              for (const auto& a : smp.top) t.set(text::sanitize_utf8(tok.piece(a.token)), Value(double(a.logprob)));
              tops.push_back(std::move(t));
            }
            l.set("tokens", std::move(toks));
            l.set("token_logprobs", std::move(lps));
            l.set("top_logprobs", std::move(tops));
            return l;
          };
          if (job->chat && !send_delta({}, "", nullptr)) {  // role chunk first
            eng.abort(rid);
            return false;
          }
          engine::FinishReason reason = engine::FinishReason::None;
          std::string error;
          for (;;) {
            engine::GenOutput o;
            if (!ch->pop(o, std::chrono::milliseconds(200))) {
              if (!sink.is_writable()) {
                eng.abort(rid);
                return false;
              }
              continue;
            }
            completion = std::max(completion, o.completion_tokens);
            text::ParsedDelta d;
            std::string raw;
            st->push(o.tokens, raw, d);
            if (o.finished || st->stop.stopped()) st->finish(raw, d);
            const auto lp = lp_json(o);
            if (!send_delta(d, raw, lp ? &*lp : nullptr)) {
              eng.abort(rid);
              return false;
            }
            if (st->stop.stopped() && !o.finished) eng.abort(rid);
            if (o.finished || st->stop.stopped()) {
              reason = o.finished ? o.reason : engine::FinishReason::Stop;
              error = o.error;
              break;
            }
          }
          if (reason == engine::FinishReason::Error) {
            write(sse(error_body(error, "InternalServerError", 500)));
          } else {
            const bool calls = st->parser && st->parser->saw_tool_calls();
            Value choice = Value::object();
            choice.set("index", Value(0));
            if (job->chat) choice.set("delta", Value::object());
            else choice.set("text", Value(""));
            choice.set("logprobs", Value());
            choice.set("finish_reason", Value(finish_name(reason, st->stop.stopped(), calls)));
            Value c = chunk_base(*job);
            Value choices = Value::array();
            choices.push_back(std::move(choice));
            c.set("choices", std::move(choices));
            write(sse(c));
            if (job->p.include_usage) {
              Value u = chunk_base(*job);
              u.set("choices", Value::array());
              u.set("usage", usage_json(prompt_tokens, completion));
              write(sse(u));
            }
          }
          write("data: [DONE]\n\n");
          sink.done();
          return true;
        },
        [this, rid](bool success) {
          if (!success) eng.abort(rid);
        });
  }

  void collect(const std::shared_ptr<Job>& job, const std::shared_ptr<Channel>& ch,
               const std::shared_ptr<TextState>& st, uint64_t rid, const httplib::Request& req,
               httplib::Response& res) {
    text::ParsedDelta all;
    std::string raw = job->p.echo ? job->prompt_text : "";
    std::vector<engine::Sampled> samples;
    engine::FinishReason reason = engine::FinishReason::None;
    std::string error;
    int completion = 0;
    for (;;) {
      engine::GenOutput o;
      if (!ch->pop(o, std::chrono::milliseconds(500))) {
        if (req.is_connection_closed()) {
          eng.abort(rid);
          throw HttpError(499, "client closed the connection");
        }
        continue;
      }
      completion = std::max(completion, o.completion_tokens);
      if (job->p.logprobs) samples.insert(samples.end(), o.samples.begin(), o.samples.end());
      st->push(o.tokens, raw, all);
      if (o.finished || st->stop.stopped()) {
        st->finish(raw, all);
        if (!o.finished) eng.abort(rid);
        reason = o.finished ? o.reason : engine::FinishReason::Stop;
        error = o.error;
        break;
      }
    }
    if (reason == engine::FinishReason::Error) throw HttpError(500, error, "InternalServerError");
    const bool calls = !all.tool_calls.empty();
    Value choice = Value::object();
    choice.set("index", Value(0));
    if (job->chat) {
      Value msg = Value::object();
      msg.set("role", Value("assistant"));
      msg.set("content", calls && all.content.empty() ? Value() : Value(all.content));
      msg.set("reasoning", all.reasoning.empty() ? Value() : Value(all.reasoning));
      msg.set("reasoning_content", all.reasoning.empty() ? Value() : Value(all.reasoning));
      Value tcs = Value::array();
      for (const auto& tc : all.tool_calls) {
        Value c = Value::object();
        c.set("id", Value("call_" + random_hex(24)));
        c.set("type", Value("function"));
        Value fn = Value::object();
        fn.set("name", Value(tc.name));
        fn.set("arguments", Value(tc.arguments));
        c.set("function", std::move(fn));
        tcs.push_back(std::move(c));
      }
      msg.set("tool_calls", std::move(tcs));
      choice.set("message", std::move(msg));
      if (job->p.logprobs) {
        Value content = Value::array();
        for (const auto& smp : samples) content.push_back(logprob_entry(smp, true));
        Value l = Value::object();
        l.set("content", std::move(content));
        choice.set("logprobs", std::move(l));
      } else {
        choice.set("logprobs", Value());
      }
    } else {
      choice.set("text", Value(raw));
      if (job->p.logprobs) {
        Value l = Value::object(), toks = Value::array(), lps = Value::array(), tops = Value::array();
        for (const auto& smp : samples) {
          toks.push_back(Value(text::sanitize_utf8(tok.piece(smp.token))));
          lps.push_back(Value(double(smp.logprob)));
          Value t = Value::object();
          for (const auto& a : smp.top) t.set(text::sanitize_utf8(tok.piece(a.token)), Value(double(a.logprob)));
          tops.push_back(std::move(t));
        }
        l.set("tokens", std::move(toks));
        l.set("token_logprobs", std::move(lps));
        l.set("top_logprobs", std::move(tops));
        choice.set("logprobs", std::move(l));
      } else {
        choice.set("logprobs", Value());
      }
    }
    choice.set("finish_reason", Value(finish_name(reason, st->stop.stopped(), calls)));
    Value body = Value::object();
    body.set("id", Value(job->id));
    body.set("object", Value(job->chat ? "chat.completion" : "text_completion"));
    body.set("created", Value(static_cast<long long>(job->created)));
    body.set("model", Value(cfg.model_name));
    Value choices = Value::array();
    choices.push_back(std::move(choice));
    body.set("choices", std::move(choices));
    body.set("usage", usage_json(int(job->prompt.size()), completion));
    send_json(res, body);
  }

  void chat(const httplib::Request& req, httplib::Response& res) {
    check_auth(req);
    Value body = parse_body(req);
    auto job = std::make_shared<Job>();
    job->chat = true;
    job->p = parse_params(body, true, cfg.default_max_tokens);
    if (const Value* tc = body.find("tool_choice"); tc && tc->is_string() && tc->str() == "none") body.erase("tools");
    if (const Value* t = body.find("tools"); t && t->is_array()) job->tools = *t;
    const text::RenderedChat r = text::render_request(body);
    job->thinking = r.thinking;
    job->prompt = tok.encode(r.prompt);
    job->id = "chatcmpl-" + random_hex(32);
    job->created = unix_now();
    run_job(job, req, res);
  }

  void completions(const httplib::Request& req, httplib::Response& res) {
    check_auth(req);
    const Value body = parse_body(req);
    auto job = std::make_shared<Job>();
    job->p = parse_params(body, false, cfg.default_max_tokens);
    const Value& pr = body.at("prompt");
    if (pr.is_string()) {
      job->prompt_text = pr.str();
      job->prompt = tok.encode(pr.str());
    } else if (pr.is_array() && pr.size() > 0 && pr.items()[0].is_number()) {
      for (const Value& v : pr.items()) {
        const auto id = int32_t(v.as_int());
        if (id < 0 || id >= tok.vocab_size()) throw HttpError(400, "token id out of range");
        job->prompt.push_back(id);
      }
      job->prompt_text = tok.decode(job->prompt);
    } else if (pr.is_array() && pr.size() == 1 && pr.items()[0].is_string()) {
      job->prompt_text = pr.items()[0].str();
      job->prompt = tok.encode(job->prompt_text);
    } else {
      throw HttpError(400, "'prompt' must be a string, one string in an array, or token ids");
    }
    if (job->p.echo && job->p.stream) throw HttpError(400, "echo is not supported with streaming");
    job->id = "cmpl-" + random_hex(32);
    job->created = unix_now();
    run_job(job, req, res);
  }

  void setup() {
    svr.new_task_queue = [n = size_t(std::max(4, cfg.threads))] { return new httplib::ThreadPool(n, n); };
    svr.set_read_timeout(3600, 0);
    svr.set_write_timeout(3600, 0);
    svr.set_payload_max_length(size_t(512) << 20);
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
                             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.status = 200; });
    svr.Get("/v1/models", wrap([this](const httplib::Request& req, httplib::Response& res) {
              check_auth(req);
              Value m = Value::object();
              m.set("id", Value(cfg.model_name));
              m.set("object", Value("model"));
              m.set("created", Value(static_cast<long long>(unix_now())));
              m.set("owned_by", Value("unholy"));
              m.set("max_model_len", Value(eng.max_context()));
              Value data = Value::array();
              data.push_back(std::move(m));
              Value b = Value::object();
              b.set("object", Value("list"));
              b.set("data", std::move(data));
              send_json(res, b);
            }));
    svr.Post("/v1/chat/completions",
             wrap([this](const httplib::Request& req, httplib::Response& res) { chat(req, res); }));
    svr.Post("/v1/completions",
             wrap([this](const httplib::Request& req, httplib::Response& res) { completions(req, res); }));
    svr.Post("/tokenize", wrap([this](const httplib::Request& req, httplib::Response& res) {
               check_auth(req);
               const Value body = parse_body(req);
               std::vector<int32_t> ids;
               if (body.contains("messages")) ids = tok.encode(text::render_request(body).prompt);
               else ids = tok.encode(body.get_str("prompt"));
               Value arr = Value::array();
               for (const int32_t t : ids) arr.push_back(Value(int(t)));
               Value b = Value::object();
               b.set("count", Value(int(ids.size())));
               b.set("max_model_len", Value(eng.max_context()));
               b.set("tokens", std::move(arr));
               send_json(res, b);
             }));
    svr.Post("/detokenize", wrap([this](const httplib::Request& req, httplib::Response& res) {
               check_auth(req);
               const Value body = parse_body(req);
               std::vector<int32_t> ids;
               for (const Value& v : body.at("tokens").items()) ids.push_back(int32_t(v.as_int()));
               Value b = Value::object();
               b.set("prompt", Value(text::sanitize_utf8(tok.decode(ids))));
               send_json(res, b);
             }));
    svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
      const auto s = eng.stats();
      char buf[1024];
      std::snprintf(buf, sizeof(buf),
                    "unholy_steps_total %llu\nunholy_prefill_tokens_total %llu\nunholy_decode_tokens_total %llu\n"
                    "unholy_preemptions_total %llu\nunholy_busy_seconds_total %.3f\nunholy_running %d\n"
                    "unholy_waiting %d\nunholy_free_pages %d\nunholy_total_pages %d\nunholy_http_requests_total %llu\n"
                    "unholy_http_errors_total %llu\n",
                    (unsigned long long)s.steps, (unsigned long long)s.prefill_tokens,
                    (unsigned long long)s.decode_tokens, (unsigned long long)s.preemptions, s.busy_seconds, s.running,
                    s.waiting, s.free_pages, s.total_pages, (unsigned long long)requests.load(),
                    (unsigned long long)errors.load());
      res.set_content(buf, "text/plain; version=0.0.4");
    });
  }
};

OpenAIServer::OpenAIServer(engine::Engine& eng, const text::Tokenizer& tok, ServerConfig cfg)
    : impl_(std::make_unique<Impl>(eng, tok, std::move(cfg))) {
  impl_->setup();
}

OpenAIServer::~OpenAIServer() = default;

void OpenAIServer::run() {
  if (!impl_->svr.listen(impl_->cfg.host, impl_->cfg.port))
    throw std::runtime_error("cannot listen on " + impl_->cfg.host + ":" + std::to_string(impl_->cfg.port));
}

void OpenAIServer::stop() { impl_->svr.stop(); }

}  // namespace unholy::server
