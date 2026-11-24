#pragma once

#include "perception/tokenizer_config.h"
#include "perception/bpe_engine.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace opengp {
namespace responder {

// Streaming token-ID -> UTF-8 detokenizer supporting Byte-Level BPE & SentencePiece Unigram
class Detokenizer {
public:
    Detokenizer() = default;

    void init(const perception::TokenizerConfig & config,
              const perception::BpeEngine &       bpe);

    // Converts a single token ID into complete, valid UTF-8 text (buffers trailing partial UTF-8 bytes)
    std::string push_token(int32_t token_id);

    // Flushes any remaining pending bytes (with U+FFFD replacement if truncated) and clears buffer
    std::string flush();

    void reset();

private:
    void decode_piece_bytes(const std::string & piece, std::vector<uint8_t> & out_bytes) const;

    perception::TokenizerType type_ = perception::TokenizerType::BPE;
    const perception::BpeEngine * bpe_ = nullptr;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    std::vector<uint8_t> pending_bytes_;
};

} // namespace responder
} // namespace opengp
