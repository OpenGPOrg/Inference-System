#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "reasoning/model_config.h"

#include <cstddef>
#include <vector>

namespace opengp {
namespace reasoning {

// Dynamically growable per-layer KV Cache allocated on the active compute backend
class KVCache {
public:
    KVCache();
    ~KVCache();

    KVCache(const KVCache &) = delete;
    KVCache & operator=(const KVCache &) = delete;

    bool init(const ModelConfig & config,
              int                 n_local_layers,
              ggml_backend_t      backend,
              KVPrecision         precision,
              size_t              block_size);

    // Ensures storage can hold at least `required_tokens`, preserving [0 .. n_past_-1] if grown
    bool ensure_capacity(size_t required_tokens);

    // Graph views for writing current token slice [n_past .. n_past + n_tokens - 1]
    struct ggml_tensor * write_view_k(struct ggml_context * ctx, int local_layer_idx, size_t n_past, size_t n_tokens) const;
    struct ggml_tensor * write_view_v(struct ggml_context * ctx, int local_layer_idx, size_t n_past, size_t n_tokens) const;

    // Graph views for reading historical KV slice [0 .. n_total - 1] shaped as [d_head, n_kv_heads, n_total]
    struct ggml_tensor * read_view_k(struct ggml_context * ctx, int local_layer_idx, size_t n_total) const;
    struct ggml_tensor * read_view_v(struct ggml_context * ctx, int local_layer_idx, size_t n_total) const;

    void advance(size_t n_tokens) { n_past_ += n_tokens; }
    void reset() { n_past_ = 0; }

    int    get_n_local_layers() const { return n_local_layers_; }
    size_t get_n_past() const { return n_past_; }
    size_t get_capacity() const { return capacity_; }
    size_t total_bytes() const;
    size_t bytes_per_token() const;
    enum ggml_type get_dtype() const { return dtype_; }

private:
    void cleanup();

    ModelConfig    config_;
    int            n_local_layers_ = 0;
    ggml_backend_t backend_        = nullptr;
    enum ggml_type dtype_      = GGML_TYPE_F16;
    size_t         block_size_ = 256;

    size_t n_past_   = 0;
    size_t capacity_ = 0;

    struct ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;

    std::vector<struct ggml_tensor *> k_layers_; // [d_head_k * n_kv_heads, capacity]
    std::vector<struct ggml_tensor *> v_layers_; // [d_head_v * n_kv_heads, capacity]
};

} // namespace reasoning
} // namespace opengp
