#define PCRE2_CODE_UNIT_WIDTH 8
#include "text/tokenizer.hpp"

#include <pcre2.h>

#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "text/json.hpp"

namespace unholy::text {

namespace {

// ---- UTF-8 ----------------------------------------------------------------------------------------

// Classifies the sequence at p[0..n): 1 = valid (len = its length), 0 = invalid (len = the maximal
// invalid subpart, replaced by one U+FFFD), -1 = a valid prefix cut off by the end of the buffer.
int utf8_seq(const unsigned char* p, size_t n, size_t& len) {
  const unsigned c = p[0];
  if (c < 0x80) {
    len = 1;
    return 1;
  }
  int need;
  unsigned lo = 0x80, hi = 0xbf;
  if (c >= 0xc2 && c <= 0xdf) need = 1;
  else if (c == 0xe0) need = 2, lo = 0xa0;
  else if ((c >= 0xe1 && c <= 0xec) || c == 0xee || c == 0xef) need = 2;
  else if (c == 0xed) need = 2, hi = 0x9f;
  else if (c == 0xf0) need = 3, lo = 0x90;
  else if (c >= 0xf1 && c <= 0xf3) need = 3;
  else if (c == 0xf4) need = 3, hi = 0x8f;
  else {
    len = 1;
    return 0;
  }
  for (int k = 1; k <= need; ++k) {
    if (size_t(k) >= n) {
      len = size_t(k);
      return -1;
    }
    const unsigned b = p[k];
    const unsigned l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xbf;
    if (b < l || b > h) {
      len = size_t(k);
      return 0;
    }
  }
  len = size_t(need + 1);
  return 1;
}

constexpr std::string_view kReplacement = "\xef\xbf\xbd";

// ---- GPT-2 byte-level alphabet -------------------------------------------------------------------

std::vector<uint32_t> byte_to_cp() {
  std::vector<uint32_t> m(256, 0);
  std::vector<bool> direct(256, false);
  for (int b = '!'; b <= '~'; ++b) direct[b] = true;
  for (int b = 0xa1; b <= 0xac; ++b) direct[b] = true;
  for (int b = 0xae; b <= 0xff; ++b) direct[b] = true;
  uint32_t n = 0;
  for (int b = 0; b < 256; ++b) m[b] = direct[b] ? uint32_t(b) : 256 + n++;
  return m;
}

void put_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += char(cp);
  } else if (cp < 0x800) {
    out += char(0xc0 | (cp >> 6));
    out += char(0x80 | (cp & 0x3f));
  } else {
    out += char(0xe0 | (cp >> 12));
    out += char(0x80 | ((cp >> 6) & 0x3f));
    out += char(0x80 | (cp & 0x3f));
  }
}

// Byte-level token text -> raw bytes; false if some character is outside the alphabet.
bool unmap_bytes(std::string_view s, const std::unordered_map<uint32_t, uint8_t>& cp_to_byte, std::string& out) {
  out.clear();
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  for (size_t i = 0; i < s.size();) {
    size_t len;
    if (utf8_seq(p + i, s.size() - i, len) != 1) return false;
    uint32_t cp;
    if (len == 1) cp = p[i];
    else if (len == 2) cp = ((p[i] & 0x1fu) << 6) | (p[i + 1] & 0x3fu);
    else if (len == 3) cp = ((p[i] & 0x0fu) << 12) | ((p[i + 1] & 0x3fu) << 6) | (p[i + 2] & 0x3fu);
    else return false;
    const auto it = cp_to_byte.find(cp);
    if (it == cp_to_byte.end()) return false;
    out += char(it->second);
    i += len;
  }
  return true;
}

// ---- added-token matcher -------------------------------------------------------------------------

// Leftmost-longest over a byte trie (HF uses Aho-Corasick with MatchKind::LeftmostLongest; for a
// non-overlapping scan the two agree: earliest start with any match, longest match there).
class Trie {
 public:
  void add(std::string_view key, int32_t id) {
    int32_t node = 0;
    for (const char ch : key) {
      const auto c = static_cast<uint8_t>(ch);
      int32_t nx = -1;
      for (const auto& [b, to] : nodes_[node].next)
        if (b == c) nx = to;
      if (nx < 0) {
        nx = int32_t(nodes_.size());
        nodes_[node].next.emplace_back(c, nx);
        nodes_.emplace_back();
      }
      node = nx;
    }
    nodes_[node].id = id;
    if (!key.empty()) first_[static_cast<uint8_t>(key[0])] = true;
  }
  bool empty() const { return nodes_.size() == 1; }
  // Longest match starting at s[i]; returns its length (0 if none) and sets id.
  size_t match_at(std::string_view s, size_t i, int32_t& id) const {
    int32_t node = 0;
    size_t best = 0;
    for (size_t k = i; k < s.size(); ++k) {
      const auto c = static_cast<uint8_t>(s[k]);
      int32_t nx = -1;
      for (const auto& [b, to] : nodes_[node].next)
        if (b == c) nx = to;
      if (nx < 0) break;
      node = nx;
      if (nodes_[node].id >= 0) {
        best = k + 1 - i;
        id = nodes_[node].id;
      }
    }
    return best;
  }
  bool may_start(uint8_t c) const { return first_[c]; }

 private:
  struct Node {
    int32_t id = -1;
    std::vector<std::pair<uint8_t, int32_t>> next;
  };
  std::vector<Node> nodes_{1};
  bool first_[256] = {};
};

struct Segment {
  size_t off, len;
  int32_t added;  // -1: plain text
};

void split_added(const Trie& trie, std::string_view text, const Segment& seg, std::vector<Segment>& out) {
  if (seg.added >= 0 || trie.empty()) {
    out.push_back(seg);
    return;
  }
  const std::string_view s = text.substr(0, seg.off + seg.len);
  size_t start = seg.off;
  for (size_t i = seg.off; i < s.size();) {
    int32_t id;
    size_t m = 0;
    if (trie.may_start(static_cast<uint8_t>(s[i]))) m = trie.match_at(s, i, id);
    if (!m) {
      ++i;
      continue;
    }
    if (i > start) out.push_back({start, i - start, -1});
    out.push_back({i, m, id});
    i += m;
    start = i;
  }
  if (s.size() > start) out.push_back({start, s.size() - start, -1});
}

// ---- PCRE2 --------------------------------------------------------------------------------------

// Oniguruma's \s is the Unicode White_Space property; PCRE2's UCP \s is \p{Z} + \h + \v, which also
// has U+180E (Mongolian vowel separator; tok_compare.py catches the difference). Spell the property
// out instead.
constexpr std::string_view kWhiteSpace = "\\t-\\r \\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}\\x{2028}\\x{2029}"
                                         "\\x{202F}\\x{205F}\\x{3000}";

std::string onig_to_pcre2(std::string_view p) {
  std::string out;
  bool in_class = false;
  for (size_t i = 0; i < p.size(); ++i) {
    const char c = p[i];
    if (c == '\\' && i + 1 < p.size()) {
      const char e = p[i + 1];
      ++i;
      if (e == 's') {
        if (in_class) out += kWhiteSpace;
        else out += "[" + std::string(kWhiteSpace) + "]";
      } else if (e == 'S') {
        if (in_class) throw std::runtime_error("tokenizer: \\S inside a character class is not supported");
        out += "[^" + std::string(kWhiteSpace) + "]";
      } else {
        out += c;
        out += e;
      }
      continue;
    }
    if (c == '[' && !in_class) {
      in_class = true;
      out += c;
      if (i + 1 < p.size() && p[i + 1] == '^') out += p[++i];
      if (i + 1 < p.size() && p[i + 1] == ']') out += p[++i];  // a leading ']' is literal
      continue;
    }
    if (c == ']' && in_class) in_class = false;
    out += c;
  }
  return out;
}

struct Regex {
  pcre2_code* code = nullptr;
  explicit Regex(const std::string& onig_pattern) {
    const std::string pattern = onig_to_pcre2(onig_pattern);
    int err;
    PCRE2_SIZE off;
    code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP, &err,
                         &off, nullptr);
    if (!code) {
      PCRE2_UCHAR msg[256];
      pcre2_get_error_message(err, msg, sizeof(msg));
      throw std::runtime_error("tokenizer: regex compile failed at " + std::to_string(off) + ": " +
                               reinterpret_cast<char*>(msg));
    }
    pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);  // falls back to the interpreter if JIT is unavailable
  }
  Regex(Regex&& o) noexcept : code(o.code) { o.code = nullptr; }
  Regex(const Regex&) = delete;
  ~Regex() {
    if (code) pcre2_code_free(code);
  }
};

}  // namespace

// ---- Tokenizer ----------------------------------------------------------------------------------

struct Tokenizer::Impl {
  std::vector<std::string> pieces;  // id -> raw bytes
  std::vector<uint8_t> kind;        // 0 none, 1 model, 2 added, 3 special added
  int32_t byte_id[256];
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;  // (a,b) -> (rank, merged id)
  Trie pass1, pass2;  // special/non-normalized added tokens; normalized added tokens
  std::vector<Regex> splits;
  std::unordered_map<std::string, int32_t> added_ids;
  int32_t bos = -1, eos = -1;

  static uint64_t key(int32_t a, int32_t b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }

  void bpe(std::string_view word, std::vector<int32_t>& out) const {
    struct Sym {
      int32_t id, prev, next;
      bool live;
    };
    const size_t n = word.size();
    if (n == 1) {
      out.push_back(byte_id[static_cast<uint8_t>(word[0])]);
      return;
    }
    std::vector<Sym> sym(n);
    for (size_t i = 0; i < n; ++i)
      sym[i] = {byte_id[static_cast<uint8_t>(word[i])], int32_t(i) - 1, i + 1 < n ? int32_t(i + 1) : -1, true};
    struct Merge {
      int32_t pos, rank, id;
      bool operator<(const Merge& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Merge> q;
    for (size_t i = 0; i + 1 < n; ++i) {
      const auto it = merges.find(key(sym[i].id, sym[i + 1].id));
      if (it != merges.end()) q.push({int32_t(i), it->second.first, it->second.second});
    }
    while (!q.empty()) {
      const Merge top = q.top();
      q.pop();
      Sym& cur = sym[top.pos];
      if (!cur.live || cur.next < 0) continue;
      const Sym right = sym[cur.next];
      const auto it = merges.find(key(cur.id, right.id));
      if (it == merges.end() || it->second.second != top.id) continue;  // stale entry
      cur.id = top.id;
      sym[cur.next].live = false;
      cur.next = right.next;
      if (right.next >= 0) sym[right.next].prev = top.pos;
      if (cur.prev >= 0) {
        const auto p = merges.find(key(sym[cur.prev].id, cur.id));
        if (p != merges.end()) q.push({cur.prev, p->second.first, p->second.second});
      }
      if (cur.next >= 0) {
        const auto p = merges.find(key(cur.id, sym[cur.next].id));
        if (p != merges.end()) q.push({top.pos, p->second.first, p->second.second});
      }
    }
    for (int32_t i = 0; i >= 0; i = sym[i].next) out.push_back(sym[i].id);
  }

  void pretokenize_bpe(std::string_view text, std::vector<int32_t>& out, pcre2_match_data* md) const {
    // Each Split (behavior Isolated) cuts the pieces of the previous one into matches and gaps.
    std::vector<std::pair<size_t, size_t>> pieces{{0, text.size()}}, next;
    for (const Regex& re : splits) {
      next.clear();
      for (const auto& [off, len] : pieces) {
        const auto* subj = reinterpret_cast<PCRE2_SPTR>(text.data() + off);
        size_t pos = 0, gap = 0;
        while (pos < len) {
          const int rc = pcre2_match(re.code, subj, len, pos, PCRE2_NO_UTF_CHECK, md, nullptr);
          if (rc < 0) break;
          const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
          const size_t ms = ov[0], me = ov[1];
          if (me == ms) {  // empty match: step one character (never happens with these patterns)
            size_t l = 1;
            utf8_seq(reinterpret_cast<const unsigned char*>(text.data() + off + ms), len - ms, l);
            pos = ms + l;
            continue;
          }
          if (ms > gap) next.emplace_back(off + gap, ms - gap);
          next.emplace_back(off + ms, me - ms);
          pos = gap = me;
        }
        if (len > gap) next.emplace_back(off + gap, len - gap);
      }
      pieces.swap(next);
    }
    for (const auto& [off, len] : pieces) bpe(text.substr(off, len), out);
  }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Tokenizer Tokenizer::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const json::Value root = json::parse(ss.str());
  auto impl = std::make_unique<Impl>();

  const json::Value& model = root.at("model");
  if (model.get_str("type") != "BPE") throw std::runtime_error("tokenizer: only BPE models are supported");
  for (const char* flag : {"byte_fallback", "ignore_merges"})
    if (const json::Value* v = model.find(flag); v && v->truthy())
      throw std::runtime_error(std::string("tokenizer: unsupported BPE option ") + flag);
  if (const json::Value* v = model.find("continuing_subword_prefix"); v && !v->is_null())
    throw std::runtime_error("tokenizer: unsupported continuing_subword_prefix");
  if (const json::Value* v = model.find("end_of_word_suffix"); v && !v->is_null())
    throw std::runtime_error("tokenizer: unsupported end_of_word_suffix");

  const std::vector<uint32_t> b2c = byte_to_cp();
  std::unordered_map<uint32_t, uint8_t> c2b;
  for (int b = 0; b < 256; ++b) c2b[b2c[b]] = uint8_t(b);

  int32_t max_id = -1;
  for (const auto& [k, v] : model.at("vocab").members()) max_id = std::max(max_id, int32_t(v.as_int()));
  for (const auto& a : root.at("added_tokens").items()) max_id = std::max(max_id, int32_t(a.at("id").as_int()));
  impl->pieces.assign(size_t(max_id + 1), {});
  impl->kind.assign(size_t(max_id + 1), 0);

  std::unordered_map<std::string, int32_t> vocab;
  vocab.reserve(model.at("vocab").size());
  std::string bytes;
  for (const auto& [k, v] : model.at("vocab").members()) {
    const auto id = int32_t(v.as_int());
    vocab.emplace(k, id);
    if (unmap_bytes(k, c2b, bytes)) {
      impl->pieces[size_t(id)] = bytes;
      impl->kind[size_t(id)] = 1;
    }
  }
  for (int b = 0; b < 256; ++b) {
    std::string ch;
    put_utf8(ch, b2c[b]);
    const auto it = vocab.find(ch);
    if (it == vocab.end()) throw std::runtime_error("tokenizer: byte " + std::to_string(b) + " missing from vocab");
    impl->byte_id[b] = it->second;
  }

  const auto& merges = model.at("merges").items();
  impl->merges.reserve(merges.size() * 2);
  for (size_t r = 0; r < merges.size(); ++r) {
    std::string a, b;
    if (merges[r].is_string()) {
      const std::string& m = merges[r].str();
      const size_t sp = m.find(' ', 1);
      if (sp == std::string::npos) throw std::runtime_error("tokenizer: bad merge '" + m + "'");
      a = m.substr(0, sp);
      b = m.substr(sp + 1);
    } else {
      a = merges[r].items().at(0).str();
      b = merges[r].items().at(1).str();
    }
    const auto ia = vocab.find(a), ib = vocab.find(b), ic = vocab.find(a + b);
    if (ia == vocab.end() || ib == vocab.end() || ic == vocab.end())
      throw std::runtime_error("tokenizer: merge '" + a + " " + b + "' references unknown tokens");
    impl->merges.emplace(Impl::key(ia->second, ib->second), std::make_pair(int32_t(r), ic->second));
  }

  for (const auto& a : root.at("added_tokens").items()) {
    const auto id = int32_t(a.at("id").as_int());
    const std::string& content = a.at("content").str();
    const bool special = a.at("special").as_bool(), normalized = a.at("normalized").as_bool();
    for (const char* flag : {"lstrip", "rstrip", "single_word"})
      if (a.at(flag).as_bool())
        throw std::runtime_error(std::string("tokenizer: added token option ") + flag + " is not supported");
    impl->pieces[size_t(id)] = content;
    impl->kind[size_t(id)] = special ? 3 : 2;
    impl->added_ids[content] = id;
    (special || !normalized ? impl->pass1 : impl->pass2).add(content, id);
  }

  if (const json::Value* n = root.find("normalizer"); n && !n->is_null()) {
    const bool empty_seq = n->get_str("type") == "Sequence" && n->at("normalizers").size() == 0;
    if (!empty_seq) throw std::runtime_error("tokenizer: only an empty normalizer is supported");
  }
  const json::Value& pre = root.at("pre_tokenizer");
  std::vector<const json::Value*> pres;
  if (pre.get_str("type") == "Sequence")
    for (const auto& p : pre.at("pretokenizers").items()) pres.push_back(&p);
  else
    pres.push_back(&pre);
  bool byte_level = false;
  for (const json::Value* p : pres) {
    const std::string type = p->get_str("type");
    if (type == "Split") {
      if (byte_level) throw std::runtime_error("tokenizer: Split after ByteLevel is not supported");
      if (p->get_str("behavior") != "Isolated" || p->at("invert").as_bool())
        throw std::runtime_error("tokenizer: only non-inverted Isolated splits are supported");
      const json::Value& pat = p->at("pattern");
      if (!pat.contains("Regex")) throw std::runtime_error("tokenizer: only Regex split patterns are supported");
      impl->splits.emplace_back(pat.at("Regex").str());
    } else if (type == "ByteLevel") {
      if (p->at("add_prefix_space").as_bool() || p->at("use_regex").as_bool())
        throw std::runtime_error("tokenizer: ByteLevel with prefix space or its own regex is not supported");
      byte_level = true;
    } else {
      throw std::runtime_error("tokenizer: unsupported pre-tokenizer " + type);
    }
  }
  if (!byte_level) throw std::runtime_error("tokenizer: expected a ByteLevel pre-tokenizer");

  const auto find = [&](const char* s) {
    const auto it = impl->added_ids.find(s);
    return it == impl->added_ids.end() ? -1 : it->second;
  };
  impl->bos = find("<｜begin▁of▁sentence｜>");
  impl->eos = find("<｜end▁of▁sentence｜>");
  return Tokenizer(std::move(impl));
}

void Tokenizer::encode_to(std::string_view input, std::vector<int32_t>& out) const {
  std::string fixed;
  if (!valid_utf8(input)) {
    fixed = sanitize_utf8(input);
    input = fixed;
  }
  std::vector<Segment> s1, s2;
  split_added(impl_->pass1, input, {0, input.size(), -1}, s1);
  for (const Segment& s : s1) split_added(impl_->pass2, input, s, s2);
  pcre2_match_data* md = impl_->splits.empty() ? nullptr : pcre2_match_data_create_from_pattern(impl_->splits[0].code, nullptr);
  for (const Segment& s : s2) {
    if (s.added >= 0) out.push_back(s.added);
    else impl_->pretokenize_bpe(input.substr(s.off, s.len), out, md);
  }
  if (md) pcre2_match_data_free(md);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<int32_t> out;
  encode_to(text, out);
  return out;
}

std::string_view Tokenizer::piece(int32_t id) const {
  if (id < 0 || size_t(id) >= impl_->pieces.size()) return {};
  return impl_->pieces[size_t(id)];
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
  std::string out;
  for (const int32_t id : ids)
    if (!(skip_special && is_special(id))) out += piece(id);
  return out;
}

bool Tokenizer::is_special(int32_t id) const {
  return id >= 0 && size_t(id) < impl_->kind.size() && impl_->kind[size_t(id)] == 3;
}
bool Tokenizer::is_added(int32_t id) const {
  return id >= 0 && size_t(id) < impl_->kind.size() && impl_->kind[size_t(id)] >= 2;
}
int32_t Tokenizer::token_id(std::string_view content) const {
  const auto it = impl_->added_ids.find(std::string(content));
  return it == impl_->added_ids.end() ? -1 : it->second;
}
int32_t Tokenizer::vocab_size() const { return int32_t(impl_->pieces.size()); }
int32_t Tokenizer::bos_id() const { return impl_->bos; }
int32_t Tokenizer::eos_id() const { return impl_->eos; }

// ---- streaming detokenization -------------------------------------------------------------------

void Detokenizer::push(int32_t id, std::string& out) {
  if (skip_special_ && tok_->is_special(id)) return;
  pending_ += tok_->piece(id);
  const auto* p = reinterpret_cast<const unsigned char*>(pending_.data());
  size_t i = 0;
  while (i < pending_.size()) {
    size_t len;
    const int r = utf8_seq(p + i, pending_.size() - i, len);
    if (r < 0) break;  // incomplete tail: wait for the next token
    if (r == 1) out.append(pending_, i, len);
    else out += kReplacement;
    i += len;
  }
  pending_.erase(0, i);
}

void Detokenizer::finish(std::string& out) {
  if (!pending_.empty()) out += sanitize_utf8(pending_);
  pending_.clear();
}

bool valid_utf8(std::string_view s) {
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  for (size_t i = 0; i < s.size();) {
    if (p[i] < 0x80) {
      ++i;
      continue;
    }
    size_t len;
    if (utf8_seq(p + i, s.size() - i, len) != 1) return false;
    i += len;
  }
  return true;
}

std::string sanitize_utf8(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  for (size_t i = 0; i < s.size();) {
    size_t len;
    const int r = utf8_seq(p + i, s.size() - i, len);
    if (r == 1) out.append(s.data() + i, len);
    else out += kReplacement;  // invalid subpart, or a truncated sequence at the very end
    i += len;
  }
  return out;
}

}  // namespace unholy::text
