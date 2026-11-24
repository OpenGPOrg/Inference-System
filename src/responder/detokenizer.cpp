#include "responder/detokenizer.h"

#include <cstdlib>

namespace opengp {
namespace responder {

namespace {

size_t utf8_sequence_len(uint8_t lead) {
    if ((lead & 0x80) == 0x00) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0; // Continuation or invalid lead byte
}

bool parse_spm_hex_byte(const std::string & piece, uint8_t & out_byte) {
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        char * endp = nullptr;
        unsigned long val = std::strtoul(piece.substr(3, 2).c_str(), &endp, 16);
        if (endp && *endp == '\0' && val <= 0xFF) {
            out_byte = static_cast<uint8_t>(val);
            return true;
        }
    }
    return false;
}

} // namespace

void Detokenizer::init(const perception::TokenizerConfig & config,
                       const perception::BpeEngine &       bpe) {
    type_            = config.tokenizer_type;
    bpe_             = &bpe;
    unicode_to_byte_ = bpe.unicode_to_byte();
    pending_bytes_.clear();
}

void Detokenizer::reset() {
    pending_bytes_.clear();
}

void Detokenizer::decode_piece_bytes(const std::string & piece, std::vector<uint8_t> & out_bytes) const {
    if (piece.empty()) return;

    // Handle SentencePiece hex fallback tokens (<0x0A>, etc.) regardless of mode
    uint8_t hex_byte = 0;
    if (parse_spm_hex_byte(piece, hex_byte)) {
        out_bytes.push_back(hex_byte);
        return;
    }

    if (type_ == perception::TokenizerType::SPM_UNIGRAM) {
        // Replace SentencePiece U+2581 ('▁', UTF-8 bytes E2 96 81) with ASCII space
        for (size_t i = 0; i < piece.size();) {
            if (i + 2 < piece.size() &&
                static_cast<uint8_t>(piece[i]) == 0xE2 &&
                static_cast<uint8_t>(piece[i + 1]) == 0x96 &&
                static_cast<uint8_t>(piece[i + 2]) == 0x81) {
                out_bytes.push_back(static_cast<uint8_t>(' '));
                i += 3;
            } else {
                out_bytes.push_back(static_cast<uint8_t>(piece[i]));
                ++i;
            }
        }
        return;
    }

    // Byte-Level BPE: reverse GPT-2 byte-to-Unicode mapping one UTF-8 character at a time
    for (size_t i = 0; i < piece.size();) {
        size_t len = utf8_sequence_len(static_cast<uint8_t>(piece[i]));
        if (len == 0 || i + len > piece.size()) {
            len = 1;
        }
        std::string ch = piece.substr(i, len);
        auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            out_bytes.push_back(it->second);
        } else {
            for (size_t k = 0; k < len; ++k) {
                out_bytes.push_back(static_cast<uint8_t>(piece[i + k]));
            }
        }
        i += len;
    }
}

std::string Detokenizer::push_token(int32_t token_id) {
    if (!bpe_) return "";

    std::string piece = bpe_->id_to_token(token_id);
    decode_piece_bytes(piece, pending_bytes_);

    // Flush only complete valid UTF-8 codepoints; retain trailing partial multi-byte prefixes
    std::string ready;
    size_t consumed = 0;
    while (consumed < pending_bytes_.size()) {
        const size_t seq_len = utf8_sequence_len(pending_bytes_[consumed]);
        if (seq_len == 0) {
            // Invalid standalone lead byte — emit directly as-is and move past
            ready.push_back(static_cast<char>(pending_bytes_[consumed]));
            consumed += 1;
            continue;
        }
        if (consumed + seq_len > pending_bytes_.size()) {
            // Incomplete multi-byte UTF-8 sequence at end of buffer — wait for next token
            break;
        }
        bool valid_cont = true;
        for (size_t k = 1; k < seq_len; ++k) {
            if ((pending_bytes_[consumed + k] & 0xC0) != 0x80) {
                valid_cont = false;
                break;
            }
        }
        if (!valid_cont) {
            ready.push_back(static_cast<char>(pending_bytes_[consumed]));
            consumed += 1;
            continue;
        }
        for (size_t k = 0; k < seq_len; ++k) {
            ready.push_back(static_cast<char>(pending_bytes_[consumed + k]));
        }
        consumed += seq_len;
    }

    if (consumed > 0) {
        pending_bytes_.erase(pending_bytes_.begin(), pending_bytes_.begin() + consumed);
    }
    return ready;
}

std::string Detokenizer::flush() {
    std::string rem(pending_bytes_.begin(), pending_bytes_.end());
    pending_bytes_.clear();
    return rem;
}

} // namespace responder
} // namespace opengp
