#include "responder/responder_engine.h"

#include <algorithm>

namespace opengp {
namespace responder {

bool RespondingEngine::initialize(const std::string &                    gguf_filepath,
                                  const perception::PerceptionPipeline & perception,
                                  const RespondingOptions &              options) {
    is_initialized_ = false;

    if (!lm_head_.init(gguf_filepath, options.lm_head_device,
                       options.gpu_device_id, options.n_threads)) {
        return false;
    }

    sampler_.configure(options.sampler);
    detokenizer_.init(perception.get_config(), perception.get_bpe_engine());
    eot_token_ids_ = perception.get_config().eot_token_ids;
    logits_buf_.resize(lm_head_.get_vocab_size());

    is_initialized_ = true;
    return true;
}

int32_t RespondingEngine::sample_next(const float *                hidden_vec,
                                      const std::vector<int32_t> & recent_tokens) {
    if (!is_initialized_ || !hidden_vec) return -1;

    if (!lm_head_.project(hidden_vec, logits_buf_)) {
        return -1;
    }
    return sampler_.sample(logits_buf_, recent_tokens);
}

bool RespondingEngine::is_stop_token(int32_t token_id) const {
    return std::find(eot_token_ids_.begin(), eot_token_ids_.end(), token_id) != eot_token_ids_.end();
}

std::string RespondingEngine::decode_token(int32_t token_id) {
    return detokenizer_.push_token(token_id);
}

std::string RespondingEngine::flush() {
    return detokenizer_.flush();
}

void RespondingEngine::reset() {
    detokenizer_.reset();
}

} // namespace responder
} // namespace opengp
