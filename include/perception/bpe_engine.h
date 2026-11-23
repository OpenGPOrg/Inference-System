#pragma once

#include "perception/tokenizer_config.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct gguf_context;

namespace opengp {
namespace perception {

class BpeEngine {
public:
    BpeEngine();
    ~BpeEngine();

    // Load vocabulary, merges, and byte mappings from GGUF
    bool load_from_gguf(const struct gguf_context * gctx, const TokenizerConfig & config);

    // Tokenize pre-tokenized chunks into final Token IDs
    std::vector<int32_t> tokenize_chunks(const std::vector<std::string> & chunks) const;

    // Direct string lookup
    int32_t token_to_id(const std::string & token) const;
    std::string id_to_token(int32_t id) const;

    const std::unordered_map<uint8_t, std::string> & byte_to_unicode() const { return byte_to_unicode_; }

private:
    void init_byte_mappings();

    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    std::unordered_map<std::string, int32_t> vocab_map_;
    std::vector<std::string> id_to_vocab_;

    // 64-bit integer merge map: ((uint64_t)id1 << 32) | id2 -> rank & merged_id
    struct MergeResult {
        int32_t rank;
        int32_t merged_id;
    };
    std::unordered_map<uint64_t, MergeResult> merges_map_;

    TokenizerType type_ = TokenizerType::BPE;

    // Helper: BPE tournament on a single chunk
    void bpe_tournament(const std::string & chunk, std::vector<int32_t> & out_ids) const;
};

} // namespace perception
} // namespace opengp
