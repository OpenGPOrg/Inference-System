#pragma once

#include "perception/tokenizer_config.h"
#include "perception/prompt_template.h"
#include "perception/dynamic_regex.h"
#include "perception/bpe_engine.h"
#include "perception/embed_table.h"

#include <string>
#include <vector>
#include <memory>

namespace opengp {
namespace perception {

struct PerceptionOutput {
    std::vector<float> H_0;             // Continuous buffer of size [n * d_model]
    std::vector<int32_t> token_ids;     // Emitted token IDs
    std::vector<int32_t> pos_ids;       // Coordinate metadata [0, 1, ..., n-1]
    std::vector<std::string> chunks;    // Regex chunk breakdown
    
    size_t n_tokens = 0;
    size_t d_model = 0;

    const float * data() const { return H_0.data(); }
    size_t bytes() const { return H_0.size() * sizeof(float); }
};

class PerceptionPipeline {
public:
    PerceptionPipeline();
    ~PerceptionPipeline();

    // Initialize pipeline dynamically from any arbitrary GGUF model file
    bool initialize(const std::string & gguf_filepath);

    // Execute complete perception phase:
    // Prompt -> Templating -> Regex -> BPE -> Token IDs -> Embed Table -> H_0
    bool process(const std::string & user_prompt, 
                 PerceptionOutput & output, 
                 const std::string & system_prompt = "");

    // Single-token embedding helper for Phase 3 Autoregressive Decode loop
    bool embed_token(int32_t token_id, float * out_vec) const;

    const TokenizerConfig & get_config() const { return config_; }
    const BpeEngine &       get_bpe_engine() const { return bpe_engine_; }
    const EmbedTable &      get_embed_table() const { return embed_table_; }

private:
    std::string model_filepath_;
    TokenizerConfig config_;
    DynamicRegex regex_chunker_;
    BpeEngine bpe_engine_;
    EmbedTable embed_table_;
    bool is_initialized_ = false;
};

} // namespace perception
} // namespace opengp
