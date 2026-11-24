#include "reasoner/recurrent_cache.h"

#include <cstdio>

namespace opengp {
namespace reasoner {

RecurrentStateCache::RecurrentStateCache() = default;

RecurrentStateCache::~RecurrentStateCache() {
    cleanup();
}

void RecurrentStateCache::cleanup() {
    if (buf_) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    conv_states_.clear();
    ssm_states_.clear();
    n_ssm_layers_ = 0;
}

bool RecurrentStateCache::init(const ModelConfig &            config,
                               const std::vector<LayerKind> & layer_kinds,
                               ggml_backend_t                 backend) {
    cleanup();
    if (!backend || layer_kinds.empty()) return false;

    const size_t n_local = layer_kinds.size();
    conv_states_.assign(n_local, nullptr);
    ssm_states_.assign(n_local, nullptr);

    n_ssm_layers_ = 0;
    for (LayerKind k : layer_kinds) {
        if (k == LayerKind::GATED_DELTANET_SSM) n_ssm_layers_++;
    }

    if (n_ssm_layers_ == 0) {
        return true; // Pure Attention model — nothing to allocate
    }

    if (config.ssm_conv_kernel < 2 || config.ssm_state_size <= 0 ||
        config.ssm_group_count <= 0 || config.ssm_time_step_rank <= 0 || config.ssm_inner_size <= 0) {
        std::fprintf(stderr, "[-] RecurrentStateCache: Incomplete SSM hyperparameters in ModelConfig\n");
        return false;
    }

    const int64_t qkv_dim  = static_cast<int64_t>(2) * config.ssm_group_count * config.ssm_state_size
                           + config.ssm_inner_size;
    const int64_t d_conv   = config.ssm_conv_kernel - 1;
    const int64_t S_v      = config.ssm_state_size;
    const int64_t H_v      = config.ssm_time_step_rank;

    const size_t meta_bytes = ggml_tensor_overhead() * static_cast<size_t>(n_ssm_layers_) * 2 + 2048;
    ctx_ = ggml_init({ meta_bytes, nullptr, true });
    if (!ctx_) return false;

    for (size_t l = 0; l < n_local; ++l) {
        if (layer_kinds[l] != LayerKind::GATED_DELTANET_SSM) continue;

        conv_states_[l] = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, d_conv, qkv_dim);
        ssm_states_[l]  = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, S_v, S_v, H_v, 1);
        ggml_format_name(conv_states_[l], "ssm_conv_l%zu", l);
        ggml_format_name(ssm_states_[l],  "ssm_state_l%zu", l);
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buf_) {
        std::fprintf(stderr, "[-] RecurrentStateCache: Failed to allocate backend buffer for %d SSM layers\n",
                     n_ssm_layers_);
        cleanup();
        return false;
    }

    reset();
    return true;
}

void RecurrentStateCache::reset() {
    for (struct ggml_tensor * t : conv_states_) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    for (struct ggml_tensor * t : ssm_states_) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
}

struct ggml_tensor * RecurrentStateCache::get_conv_state(int local_layer_idx) const {
    if (local_layer_idx < 0 || local_layer_idx >= static_cast<int>(conv_states_.size())) return nullptr;
    return conv_states_[local_layer_idx];
}

struct ggml_tensor * RecurrentStateCache::get_ssm_state(int local_layer_idx) const {
    if (local_layer_idx < 0 || local_layer_idx >= static_cast<int>(ssm_states_.size())) return nullptr;
    return ssm_states_[local_layer_idx];
}

size_t RecurrentStateCache::total_bytes() const {
    return buf_ ? ggml_backend_buffer_get_size(buf_) : 0;
}

} // namespace reasoner
} // namespace opengp
