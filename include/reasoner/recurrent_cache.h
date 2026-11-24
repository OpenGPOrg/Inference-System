#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "reasoner/model_config.h"
#include "reasoner/weight_store.h"

#include <cstddef>
#include <vector>

namespace opengp {
namespace reasoner {

// Fixed O(1) recurrent state storage (conv1d rolling state + Gated DeltaNet matrix) for SSM layers
class RecurrentStateCache {
public:
    RecurrentStateCache();
    ~RecurrentStateCache();

    RecurrentStateCache(const RecurrentStateCache &) = delete;
    RecurrentStateCache & operator=(const RecurrentStateCache &) = delete;

    bool init(const ModelConfig &            config,
              const std::vector<LayerKind> & layer_kinds,
              ggml_backend_t                 backend);

    // Zero-initializes all conv1d and DeltaNet recurrent states (e.g., on new turn/conversation)
    void reset();

    struct ggml_tensor * get_conv_state(int local_layer_idx) const;
    struct ggml_tensor * get_ssm_state(int local_layer_idx) const;

    int    get_n_ssm_layers() const { return n_ssm_layers_; }
    size_t total_bytes() const;

private:
    void cleanup();

    int                   n_ssm_layers_ = 0;
    struct ggml_context * ctx_          = nullptr;
    ggml_backend_buffer_t buf_          = nullptr;

    std::vector<struct ggml_tensor *> conv_states_; // [ssm_conv_kernel - 1, ssm_qkv_dim] (F32)
    std::vector<struct ggml_tensor *> ssm_states_;  // [ssm_state_size, ssm_state_size, ssm_time_step_rank, 1] (F32)
};

} // namespace reasoner
} // namespace opengp
