#include "perception/tokenizer_config.h"
#include "gguf.h"
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace opengp {
namespace perception {

bool TokenizerConfig::parse_from_gguf(const struct gguf_context * gctx) {
    if (!gctx) return false;

    // 1. Model architecture
    int64_t model_key = gguf_find_key(gctx, "tokenizer.ggml.model");
    if (model_key >= 0) {
        model_architecture = gguf_get_val_str(gctx, model_key);
    } else {
        model_architecture = "unknown";
    }

    // 2. Pre-tokenizer flavor string directly from model metadata
    int64_t pre_key = gguf_find_key(gctx, "tokenizer.ggml.pre");
    if (pre_key >= 0) {
        pre_tokenizer = gguf_get_val_str(gctx, pre_key);
    } else {
        pre_tokenizer = "default";
    }

    // 3. Chat template
    int64_t tmpl_key = gguf_find_key(gctx, "tokenizer.chat_template");
    if (tmpl_key >= 0) {
        chat_template = gguf_get_val_str(gctx, tmpl_key);
    } else {
        chat_template = "";
    }

    // 4. Counts
    int64_t tokens_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    vocab_size = (tokens_key >= 0) ? (uint32_t)gguf_get_arr_n(gctx, tokens_key) : 0;

    int64_t merges_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
    n_merges = (merges_key >= 0) ? (uint32_t)gguf_get_arr_n(gctx, merges_key) : 0;

    // 5. Special token IDs from metadata
    int64_t bos_key = gguf_find_key(gctx, "tokenizer.ggml.bos_token_id");
    bos_token_id = (bos_key >= 0) ? (int32_t)gguf_get_val_u32(gctx, bos_key) : -1;

    int64_t eos_key = gguf_find_key(gctx, "tokenizer.ggml.eos_token_id");
    eos_token_id = (eos_key >= 0) ? (int32_t)gguf_get_val_u32(gctx, eos_key) : -1;

    int64_t unk_key = gguf_find_key(gctx, "tokenizer.ggml.unknown_token_id");
    unk_token_id = (unk_key >= 0) ? (int32_t)gguf_get_val_u32(gctx, unk_key) : -1;

    int64_t pad_key = gguf_find_key(gctx, "tokenizer.ggml.padding_token_id");
    pad_token_id = (pad_key >= 0) ? (int32_t)gguf_get_val_u32(gctx, pad_key) : -1;

    // 6. Tokenizer type resolution
    if (n_merges > 0 || model_architecture == "gpt2" || pre_tokenizer == "llama3" || pre_tokenizer == "qwen2") {
        tokenizer_type = TokenizerType::BPE;
    } else {
        tokenizer_type = TokenizerType::SPM_UNIGRAM;
    }

    // 7. Dynamic EOT / Stop Condition Detection (Model-Agnostic, Zero Hardcoding)
    eot_token_ids.clear();
    if (eos_token_id >= 0) {
        eot_token_ids.push_back(eos_token_id);
    }

    // Inspect token strings dynamically to discover all control/stop tokens
    if (tokens_key >= 0) {
        int64_t toktype_key = gguf_find_key(gctx, "tokenizer.ggml.token_type");
        const int32_t * tok_types = (toktype_key >= 0) ? (const int32_t *)gguf_get_arr_data(gctx, toktype_key) : nullptr;

        for (uint32_t i = 0; i < vocab_size; ++i) {
            // Check if token is marked as CONTROL (3) or USER_DEFINED (4)
            bool is_control = tok_types && (tok_types[i] == 3 || tok_types[i] == 4);

            const char * piece = gguf_get_arr_str(gctx, tokens_key, i);
            if (!piece) continue;

            std::string piece_str(piece);
            // Dynamic check for stop tokens (e.g. <|eot_id|>, <|im_end|>, <|end_of_text|>, </s>)
            if (is_control || piece_str.find("<|") != std::string::npos) {
                if (piece_str == "<|eot_id|>" || 
                    piece_str == "<|im_end|>" || 
                    piece_str == "<|end_of_text|>" || 
                    piece_str == "<|endoftext|>" ||
                    piece_str == "</s>") {
                    if (std::find(eot_token_ids.begin(), eot_token_ids.end(), (int32_t)i) == eot_token_ids.end()) {
                        eot_token_ids.push_back((int32_t)i);
                    }
                }
            }
        }
    }

    return true;
}

} // namespace perception
} // namespace opengp
