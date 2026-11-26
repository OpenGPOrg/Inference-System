#pragma once

#include "ggml.h"
#include "reasoner/model_config.h"
#include "reasoner/weight_store.h"
#include "reasoner/kv_cache.h"

#include <vector>

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

// Options for multi-layer fused graph construction (used exclusively by Vulkan backend)
struct FusedGraphOptions {
    bool use_flash_attn   = false; // Use ggml_flash_attn_ext (eliminates V transpose + KQ buffer)
    bool use_swiglu       = false; // Use fused ggml_swiglu_split instead of silu + mul
    bool apply_final_norm = false; // Fuse output_norm RMSNorm at the tail of the graph
};

// Build a single computation graph chaining consecutive GPU-resident layers [0 .. layers.size()-1]
// (+ optional final RMSNorm). When n_tokens == 1, io.kq_mask is nullptr (no causal masking needed).
// When n_tokens > 1, io.kq_mask is GGML_TYPE_F16 if use_flash_attn else GGML_TYPE_F32.
LayerGraphIO build_fused_layers_graph(struct ggml_context *              ctx,
                                      const ModelConfig &                cfg,
                                      const std::vector<const LayerWeights *> & layers,
                                      const struct ggml_tensor *         rope_freqs,
                                      const struct ggml_tensor *         output_norm,
                                      KVCache &                          kv,
                                      int                                n_tokens,
                                      int                                n_past,
                                      const FusedGraphOptions &          opts);

} // namespace reasoner
} // namespace opengp
