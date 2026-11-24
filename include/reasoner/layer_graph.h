#pragma once

#include "ggml.h"
#include "reasoner/model_config.h"
#include "reasoner/weight_store.h"
#include "reasoner/kv_cache.h"

namespace opengp {
namespace reasoner {

struct LayerGraphIO {
    struct ggml_tensor * H_in    = nullptr; // [d_model, n_tokens] (F32)
    struct ggml_tensor * pos_ids = nullptr; // [n_tokens] (I32)
    struct ggml_tensor * kq_mask = nullptr; // [n_total, n_tokens] (F32)
    struct ggml_tensor * H_out   = nullptr; // [d_model, n_tokens] (F32)
    struct ggml_cgraph * graph   = nullptr;
};

// Build computation graph for one Transformer block (Pre-RMSNorm -> QKV/QK-Norm/RoPE -> KV write -> GQA/MHA -> SwiGLU)
LayerGraphIO build_layer_graph(struct ggml_context *      ctx,
                               const ModelConfig &        cfg,
                               const LayerWeights &       w,
                               const struct ggml_tensor * rope_freqs,
                               KVCache &                  kv,
                               int                        layer_idx,
                               int                        n_tokens,
                               int                        n_past);

// Build final RMSNorm graph (output_norm.weight) applied to residual stream H_{L}
struct FinalNormGraphIO {
    struct ggml_tensor * H_in  = nullptr; // [d_model, n_tokens] (F32)
    struct ggml_tensor * H_out = nullptr; // [d_model, n_tokens] (F32)
    struct ggml_cgraph * graph = nullptr;
};

FinalNormGraphIO build_final_norm_graph(struct ggml_context *      ctx,
                                        const ModelConfig &        cfg,
                                        const struct ggml_tensor * output_norm,
                                        int                        n_tokens);

} // namespace reasoner
} // namespace opengp
