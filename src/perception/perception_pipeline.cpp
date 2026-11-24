#include "perception/perception_pipeline.h"
#include "gguf.h"
#include <cstdio>
#include <cmath>

namespace opengp {
namespace perception {

PerceptionPipeline::PerceptionPipeline() = default;
PerceptionPipeline::~PerceptionPipeline() = default;

bool PerceptionPipeline::initialize(const std::string & gguf_filepath) {
    model_filepath_ = gguf_filepath;

    struct gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    struct gguf_context * gctx = gguf_init_from_file(gguf_filepath.c_str(), params);
    if (!gctx) {
        std::fprintf(stderr, "[-] PerceptionPipeline: Failed to open '%s'\n", gguf_filepath.c_str());
        return false;
    }

    // 1. Dynamic Config Parsing
    if (!config_.parse_from_gguf(gctx)) {
        std::fprintf(stderr, "[-] PerceptionPipeline: Failed to parse tokenizer config\n");
        gguf_free(gctx);
        return false;
    }

    // 2. Configure Dynamic Regex Chunker from pre-tokenizer metadata
    regex_chunker_.configure_from_pre(config_.pre_tokenizer);

    // 3. Load BPE Engine
    if (!bpe_engine_.load_from_gguf(gctx, config_)) {
        std::fprintf(stderr, "[-] PerceptionPipeline: Failed to load BPE tables\n");
        gguf_free(gctx);
        return false;
    }

    // 4. Memory-map Embedding Table
    if (!embed_table_.init_from_gguf(gguf_filepath.c_str(), gctx)) {
        std::fprintf(stderr, "[-] PerceptionPipeline: Failed to initialize embedding table\n");
        gguf_free(gctx);
        return false;
    }

    gguf_free(gctx);
    is_initialized_ = true;
    return true;
}

bool PerceptionPipeline::process(const std::string & user_prompt, 
                                 PerceptionOutput & output, 
                                 const std::string & system_prompt) {
    if (!is_initialized_) {
        std::fprintf(stderr, "[-] PerceptionPipeline: Pipeline is not initialized!\n");
        return false;
    }

    // Stage 1: Prompt Templating & Role Injection
    std::string formatted_text = PromptTemplate::format(config_, user_prompt, system_prompt);

    // Stage 2: Pre-Tokenization Regex Chunker
    output.chunks = regex_chunker_.split(formatted_text);

    // Stage 3 & 4: Byte-Level BPE Merge Tournament
    output.token_ids = bpe_engine_.tokenize_chunks(output.chunks);

    // If model has a BOS token and it's not present at pos 0, inject it
    if (config_.bos_token_id >= 0 && (output.token_ids.empty() || output.token_ids[0] != config_.bos_token_id)) {
        // Only inject if model is raw completion or standard SPM
        if (config_.chat_template.empty()) {
            output.token_ids.insert(output.token_ids.begin(), config_.bos_token_id);
        }
    }

    output.n_tokens = output.token_ids.size();
    output.d_model = embed_table_.get_d_model();

    // Stage 5: Position IDs Generation (Deferred RoPE)
    output.pos_ids.resize(output.n_tokens);
    for (size_t i = 0; i < output.n_tokens; ++i) {
        output.pos_ids[i] = (int32_t)i;
    }

    // Stage 6, 7, 8: Host-RAM Row Gathering, Dequantization, and Staging Buffer Assembly
    output.H_0.resize(output.n_tokens * output.d_model);

    // Scaling check (Gemma uses sqrt(d_model), others use 1.0)
    float scale = 1.0f;
    if (config_.model_architecture.find("gemma") != std::string::npos) {
        scale = std::sqrt((float)output.d_model);
    }

    bool ok = embed_table_.gather_and_dequant(output.token_ids, output.H_0.data(), scale);
    return ok;
}

bool PerceptionPipeline::embed_token(int32_t token_id, float * out_vec) const {
    if (!is_initialized_ || !out_vec) return false;

    float scale = 1.0f;
    if (config_.model_architecture.find("gemma") != std::string::npos) {
        scale = std::sqrt(static_cast<float>(embed_table_.get_d_model()));
    }
    return embed_table_.gather_and_dequant({ token_id }, out_vec, scale);
}

} // namespace perception
} // namespace opengp
