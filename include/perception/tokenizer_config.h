#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct gguf_context;

namespace opengp {
namespace perception {

enum class TokenizerType {
    BPE,
    SPM_UNIGRAM,
    UNKNOWN
};

struct TokenizerConfig {
    std::string model_architecture; // e.g. "llama", "gpt2"
    std::string pre_tokenizer;      // e.g. "llama3", "qwen2", "llama-bpe", "default"
    std::string chat_template;      // Jinja / ChatML / LLaMA-3 template string
    
    TokenizerType tokenizer_type = TokenizerType::UNKNOWN;

    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t unk_token_id = -1;
    int32_t pad_token_id = -1;
    std::vector<int32_t> eot_token_ids; // e.g. 128009 for LLaMA-3, 151645 for Qwen

    uint32_t vocab_size = 0;
    uint32_t n_merges = 0;

    // Load and populate dynamically from any arbitrary GGUF context
    bool parse_from_gguf(const struct gguf_context * gctx);
};

} // namespace perception
} // namespace opengp
