#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "reasoner/model_config.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace opengp {
namespace reasoner {

enum class LayerKind {
    STANDARD_ATTENTION,
    GATED_ATTENTION,
    GATED_DELTANET_SSM
};

// Tensor descriptors for a single Transformer / Hybrid SSM block (blk.<l>.*)
struct LayerWeights {
    LayerKind kind = LayerKind::STANDARD_ATTENTION;

    // Shared layer norms & SwiGLU FFN weights
    struct ggml_tensor * attn_norm = nullptr; // blk.l.attn_norm.weight
    struct ggml_tensor * ffn_norm  = nullptr; // blk.l.ffn_norm.weight (or post_attention_norm.weight)
    struct ggml_tensor * w_gate    = nullptr; // blk.l.ffn_gate.weight
    struct ggml_tensor * w_up      = nullptr; // blk.l.ffn_up.weight
    struct ggml_tensor * w_down    = nullptr; // blk.l.ffn_down.weight

    // Standard / Gated Attention weights
    struct ggml_tensor * wq          = nullptr; // blk.l.attn_q.weight
    struct ggml_tensor * wk          = nullptr; // blk.l.attn_k.weight
    struct ggml_tensor * wv          = nullptr; // blk.l.attn_v.weight
    struct ggml_tensor * wo          = nullptr; // blk.l.attn_output.weight
    struct ggml_tensor * attn_q_norm = nullptr; // blk.l.attn_q_norm.weight
    struct ggml_tensor * attn_k_norm = nullptr; // blk.l.attn_k_norm.weight
    struct ggml_tensor * bq          = nullptr; // blk.l.attn_q.bias
    struct ggml_tensor * bk          = nullptr; // blk.l.attn_k.bias
    struct ggml_tensor * bv          = nullptr; // blk.l.attn_v.bias
    struct ggml_tensor * bo          = nullptr; // blk.l.attn_output.bias

    // Gated DeltaNet SSM weights (used when kind == GATED_DELTANET_SSM)
    struct ggml_tensor * attn_qkv    = nullptr; // blk.l.attn_qkv.weight
    struct ggml_tensor * attn_gate   = nullptr; // blk.l.attn_gate.weight
    struct ggml_tensor * ssm_conv1d  = nullptr; // blk.l.ssm_conv1d.weight
    struct ggml_tensor * ssm_alpha   = nullptr; // blk.l.ssm_alpha.weight
    struct ggml_tensor * ssm_beta    = nullptr; // blk.l.ssm_beta.weight
    struct ggml_tensor * ssm_a       = nullptr; // blk.l.ssm_a
    struct ggml_tensor * ssm_dt_bias = nullptr; // blk.l.ssm_dt.bias
    struct ggml_tensor * ssm_norm    = nullptr; // blk.l.ssm_norm.weight
    struct ggml_tensor * ssm_out     = nullptr; // blk.l.ssm_out.weight

    size_t total_bytes     = 0;
    bool   is_gpu_resident = false;
};

// Zero-copy mmap GGUF weight loader with hybrid GPU resident + single-layer streaming
class WeightStore {
public:
    WeightStore();
    ~WeightStore();

    WeightStore(const WeightStore &) = delete;
    WeightStore & operator=(const WeightStore &) = delete;

    // Lightweight metadata-only scan of owned layers [range.start .. range.end)
    static std::vector<LayerKind> detect_layer_kinds(const struct gguf_context * gctx,
                                                     const ModelConfig &         config,
                                                     const LayerRange &          range);

    // Memory-map GGUF tensor blob, index owned [range.start .. range.end) blk.* weights & optional globals
    bool init(const std::string &   gguf_filepath,
              struct gguf_context * gctx,
              const ModelConfig &   config,
              const LayerRange &    range,
              ggml_backend_t        backend,
              bool                  is_gpu,
              size_t                available_vram_bytes);

    // Returns layer weights ready for compute on `backend` (local_layer_idx in [0 .. get_n_local_layers() - 1])
    const LayerWeights & prepare_layer(int local_layer_idx);
    void release() { cleanup(); }

    std::vector<LayerKind> get_layer_kinds() const;

    const struct ggml_tensor * get_output_norm() const { return output_norm_; }
    const struct ggml_tensor * get_rope_freqs() const { return rope_freqs_; }
    const struct ggml_tensor * get_lm_head() const { return lm_head_; }

    const LayerRange & get_layer_range() const { return range_; }
    int    get_n_local_layers() const { return n_local_layers_; }
    int    get_n_resident_layers() const { return n_resident_layers_; }
    int    get_n_streamed_layers() const { return n_streamed_layers_; }
    size_t get_max_layer_bytes() const { return max_layer_bytes_; }
    size_t get_total_weight_bytes() const { return total_weight_bytes_; }
    size_t get_vram_used_bytes() const { return vram_used_bytes_; }

private:
    void cleanup();

    int        fd_        = -1;
    void *     mmap_addr_ = nullptr;
    size_t     mmap_size_ = 0;

    bool       is_gpu_         = false;
    LayerRange range_;
    int        n_local_layers_ = 0;

    // Metadata-only context referencing mmap host pointers directly
    struct ggml_context * host_ctx_ = nullptr;
    std::vector<LayerWeights> host_layers_;

    struct ggml_tensor * host_output_norm_ = nullptr;
    struct ggml_tensor * host_rope_freqs_  = nullptr;
    struct ggml_tensor * host_lm_head_     = nullptr;

    // Compute-ready weights (pointing either to host_layers_ on CPU or GPU resident/staging buffers)
    std::vector<LayerWeights> active_layers_;

    // GPU resident storage (for layers 0 .. n_resident_layers_-1)
    struct ggml_context *  resident_ctx_ = nullptr;
    ggml_backend_buffer_t  resident_buf_ = nullptr;

    // Single-layer reusable GPU staging storage (when some layers must stream)
    struct ggml_context *  stage_ctx_    = nullptr;
    ggml_backend_buffer_t  stage_buf_    = nullptr;
    LayerWeights           stage_weights_;

    // Global backend weights (output_norm + optional rope_freqs)
    struct ggml_context *  global_ctx_   = nullptr;
    ggml_backend_buffer_t  global_buf_   = nullptr;
    struct ggml_tensor *   output_norm_  = nullptr;
    struct ggml_tensor *   rope_freqs_   = nullptr;
    struct ggml_tensor *   lm_head_      = nullptr;

    int    n_resident_layers_  = 0;
    int    n_streamed_layers_  = 0;
    size_t max_layer_bytes_    = 0;
    size_t total_weight_bytes_ = 0;
    size_t vram_used_bytes_    = 0;
};

} // namespace reasoner
} // namespace opengp
