#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "reasoner/backend_selector.h"
#include "reasoner/model_config.h"
#include "reasoner/weight_store.h"
#include "reasoner/kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace opengp {
namespace reasoner {

struct ReasoningOutput {
    std::vector<float> H_L;             // Contiguous FP32 hidden states after owned layer range [n_tokens * d_model]
    size_t             n_tokens   = 0;
    size_t             d_model    = 0;
    double             elapsed_ms = 0.0;
    std::vector<float> layer_l2_norms;  // L2 norm of last token vector after each owned layer
};

// Portable activation payload passed between distributed volunteer pipeline stages
struct ActivationPacket {
    std::vector<float>   hidden_states;      // [n_tokens * d_model]
    std::vector<int32_t> pos_ids;            // [n_tokens]
    size_t               n_tokens           = 0;
    size_t               d_model            = 0;
    int                  next_layer         = 0; // Global layer index expected by the next stage
    bool                 final_norm_applied = false;
};

class ReasoningEngine {
public:
    ReasoningEngine();
    ~ReasoningEngine();

    ReasoningEngine(const ReasoningEngine &) = delete;
    ReasoningEngine & operator=(const ReasoningEngine &) = delete;

    // Initialize model config, owned layer slice, backend, mmap WeightStore, KVCache, and gallocr
    bool initialize(const std::string &      gguf_filepath,
                    const ReasoningOptions & options = ReasoningOptions{});

    // Execute a single owned Transformer layer (local_layer_idx in [0 .. layer_range.count()-1])
    bool forward_layer(int             local_layer_idx,
                       const float *   H_in,
                       const int32_t * pos_ids,
                       const float *   kq_mask,
                       size_t          n_tokens,
                       size_t          n_past,
                       float *         H_out);

    // Execute owned layer slice [layer_range.start .. layer_range.end) (+ final RMSNorm if tail stage)
    bool forward(const float *     H_0,
                 const int32_t *   pos_ids,
                 size_t            n_tokens,
                 ReasoningOutput & output);

    // Modular pipeline-stage handoff: validates incoming packet layer boundary and emits outgoing packet
    bool forward_packet(const ActivationPacket & in,
                        ActivationPacket &       out,
                        ReasoningOutput *        telemetry = nullptr);

    // Optional diagnostic helper: project a single hidden state vector [d_model] against lm_head
    bool compute_logits_for_vector(const float *        hidden_vec,
                                   std::vector<float> & out_logits) const;

    void reset_kv_cache();

    const ModelConfig &      get_config() const { return config_; }
    const ReasoningOptions & get_options() const { return options_; }
    const LayerRange &       get_layer_range() const { return layer_range_; }
    const WeightStore &      get_weight_store() const { return weights_; }
    const KVCache &          get_kv_cache() const { return kv_cache_; }

    bool               is_gpu_active() const { return is_gpu_; }
    bool               is_igpu_active() const { return is_igpu_; }
    GpuBackendKind     get_backend_kind() const { return backend_kind_; }
    const std::string & get_backend_name() const { return backend_name_; }
    int                get_active_threads() const { return active_threads_; }
    size_t             get_ctx_size() const { return active_ctx_size_; }
    size_t             get_scratch_bytes() const { return peak_scratch_bytes_; }

private:
    void cleanup();

    bool initialize_impl(const std::string &      gguf_filepath,
                         const ReasoningOptions & options,
                         bool                     allow_vulkan_fallback);
    bool init_backend();
    bool preflight_vulkan_ops(std::string & unsupported_desc);
    bool forward_vulkan_fused_prefix(int             n_fused,
                                     bool            apply_final_norm,
                                     const float *   H_in,
                                     const int32_t * pos_ids,
                                     const float *   kq_mask_f32,
                                     size_t          n_tokens,
                                     size_t          n_past,
                                     float *         H_out);

    ReasoningOptions options_;
    ModelConfig      config_;
    LayerRange       layer_range_;

    struct gguf_context * gctx_ = nullptr;

    ggml_backend_t backend_               = nullptr;
    ggml_gallocr_t gallocr_               = nullptr;
    GpuBackendKind backend_kind_          = GpuBackendKind::NONE;
    bool           is_gpu_                = false;
    bool           is_igpu_               = false;
    bool           vulkan_use_flash_attn_ = false;
    bool           vulkan_use_swiglu_     = false;
    int            active_threads_        = 1;
    std::string    backend_name_;

    WeightStore weights_;
    KVCache     kv_cache_;
    size_t      active_ctx_size_    = 0;
    size_t      peak_scratch_bytes_ = 0;
    bool        is_initialized_     = false;
};

} // namespace reasoner
} // namespace opengp
