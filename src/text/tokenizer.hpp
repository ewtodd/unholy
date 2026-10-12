// Byte-level BPE tokenizer reproducing HF `tokenizers` for the DSV4 tokenizer.json.
//
// Pipeline (as HF runs it with split_special_tokens off): added tokens are cut out first, special and
// non-normalized ones in one leftmost-longest pass, normalized ones in a second pass over what is left
// (the normalizer is empty, so "normalized" text is the input); every remaining span goes through the
// Split pre-tokenizers (PCRE2 in UTF+UCP mode standing in for Oniguruma, each Split applied to the
// pieces of the previous one), the byte-level alphabet, and BPE by merge rank. Verified against HF
// with tools/text_test + tools/ref/tok_compare.py.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace unholy::text {

class Tokenizer {
 public:
  static Tokenizer load(const std::string& tokenizer_json);
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  ~Tokenizer();

  // Text -> ids. Invalid UTF-8 is replaced by U+FFFD first (HF only ever sees valid strings).
  std::vector<int32_t> encode(std::string_view text) const;
  void encode_to(std::string_view text, std::vector<int32_t>& out) const;

  // Raw bytes of one token: the byte-level-decoded piece, or an added token's content.
  std::string_view piece(int32_t id) const;
  // Concatenated pieces (skipping special added tokens if asked). May end inside a UTF-8 sequence;
  // see Detokenizer for streaming.
  std::string decode(std::span<const int32_t> ids, bool skip_special = false) const;

  bool is_special(int32_t id) const;  // special added token (bos/eos/pad/placeholders...)
  bool is_added(int32_t id) const;
  int32_t token_id(std::string_view added_token_content) const;  // -1 when absent
  int32_t vocab_size() const;  // max id + 1 (embedding rows)
  int32_t bos_id() const;
  int32_t eos_id() const;

 private:
  struct Impl;
  explicit Tokenizer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Streaming ids -> UTF-8: holds back an incomplete trailing sequence so every emitted chunk is valid
// text (bytes that can never complete are flushed as U+FFFD, as HF's lossy decode does).
class Detokenizer {
 public:
  explicit Detokenizer(const Tokenizer& tok, bool skip_special = false) : tok_(&tok), skip_special_(skip_special) {}
  // Appends the text that became complete with this token.
  void push(int32_t id, std::string& out);
  // Flushes whatever is pending at end of stream.
  void finish(std::string& out);

 private:
  const Tokenizer* tok_;
  bool skip_special_;
  std::string pending_;
};

// Replaces invalid UTF-8 with U+FFFD (per maximal invalid subpart, as Python/Rust lossy decoding).
std::string sanitize_utf8(std::string_view s);
bool valid_utf8(std::string_view s);

}  // namespace unholy::text
