#pragma once

#include "ggml.h"
#include "reasoner/model_config.h"
#include "reasoner/weight_store.h"
#include "reasoner/kv_cache.h"
#include "reasoner/recurrent_cache.h"

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

// Build computation graph for one block (Standard Attention, Gated Attention, or Gated DeltaNet SSM)
LayerGraphIO build_layer_graph(struct ggml_context *      ctx,
                               const ModelConfig &        cfg,
                               const LayerWeights &       w,
                               const struct ggml_tensor * rope_freqs,
                               KVCache &                  kv,
                               RecurrentStateCache *      recurrent_cache,
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

// Options for multi-layer fused graph construction (used by GPU backends: CUDA, Vulkan, Metal)
struct FusedGraphOptions {
    bool use_flash_attn   = false; // Use ggml_flash_attn_ext (eliminates V transpose + KQ buffer)
    bool use_swiglu       = false; // Use fused ggml_swiglu_split instead of silu + mul
    bool apply_final_norm = false; // Fuse output_norm RMSNorm at the tail of the graph
    int  pad_kv_stride    = 0;     // Pad FlashAttention KV read length to multiple (e.g. 256 for CUDA fast path)
};

// Build a single computation graph chaining consecutive GPU-resident layers [0 .. layers.size()-1]
// (+ optional final RMSNorm). When n_tokens == 1 and unpadded, io.kq_mask is nullptr.
// Otherwise io.kq_mask has width n_kv_read and type GGML_TYPE_F16 (if use_flash_attn) or F32.
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
