#pragma once

#include "perception/perception_pipeline.h"
#include "responder/lm_head.h"
#include "responder/sampler.h"
#include "responder/detokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace opengp {
namespace responder {

struct RespondingOptions {
    LmHeadDevice  lm_head_device = LmHeadDevice::AUTO;
    int           gpu_device_id  = -1;
    int           n_threads      = 0;
    SamplerConfig sampler;
};

// Phase 3 Orchestrator: Final H_n vector -> Logits -> Sampler -> Stop check -> Streaming UTF-8 text
class RespondingEngine {
public:
    RespondingEngine() = default;

    bool initialize(const std::string &                    gguf_filepath,
                    const perception::PerceptionPipeline & perception,
                    const RespondingOptions &              options = RespondingOptions{});

    // Project hidden_vec [d_model] to logits, apply Sampler, and return chosen Token ID
    int32_t sample_next(const float * hidden_vec, const std::vector<int32_t> & recent_tokens);

    // Dynamic EOT check against TokenizerConfig::eot_token_ids (model-agnostic)
    bool is_stop_token(int32_t token_id) const;

    // Streaming UTF-8 detokenization helpers
    std::string decode_token(int32_t token_id);
    std::string flush();
    void        reset();

    const LmHead &        get_lm_head() const { return lm_head_; }
    const SamplerConfig & get_sampler_config() const { return sampler_.get_config(); }

private:
    LmHead               lm_head_;
    Sampler              sampler_;
    Detokenizer          detokenizer_;
    std::vector<int32_t> eot_token_ids_;
    std::vector<float>   logits_buf_;
    bool                 is_initialized_ = false;
};

} // namespace responder
} // namespace opengp
