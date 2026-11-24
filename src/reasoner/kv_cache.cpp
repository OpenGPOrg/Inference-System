#include "reasoner/kv_cache.h"

#include <algorithm>
#include <cstdio>

namespace opengp {
namespace reasoner {

KVCache::KVCache() = default;

KVCache::~KVCache() {
    cleanup();
}

void KVCache::cleanup() {
    if (buf_) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    k_layers_.clear();
    v_layers_.clear();
    capacity_ = 0;
    n_past_   = 0;
}

bool KVCache::init(const ModelConfig & config,
                   int                 n_local_layers,
                   ggml_backend_t      backend,
                   KVPrecision         precision) {
    cleanup();
    if (!backend || config.n_layers <= 0 || n_local_layers <= 0) return false;

    config_         = config;
    n_local_layers_ = n_local_layers;
    backend_        = backend;
    dtype_          = (precision == KVPrecision::F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    return true;
}

size_t KVCache::bytes_per_token() const {
    const size_t k_row = ggml_row_size(dtype_, static_cast<int64_t>(config_.d_head_k) * config_.n_kv_heads);
    const size_t v_row = ggml_row_size(dtype_, static_cast<int64_t>(config_.d_head_v) * config_.n_kv_heads);
    return static_cast<size_t>(n_local_layers_) * (k_row + v_row);
}

size_t KVCache::total_bytes() const {
    return buf_ ? ggml_backend_buffer_get_size(buf_) : 0;
}

bool KVCache::allocate(size_t capacity) {
    if (!backend_ || n_local_layers_ <= 0 || capacity == 0) {
        return false;
    }

    if (buf_) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    k_layers_.clear();
    v_layers_.clear();
    capacity_ = 0;
    n_past_   = 0;

    const int64_t k_dim = static_cast<int64_t>(config_.d_head_k) * config_.n_kv_heads;
    const int64_t v_dim = static_cast<int64_t>(config_.d_head_v) * config_.n_kv_heads;

    size_t meta_bytes = ggml_tensor_overhead() * static_cast<size_t>(n_local_layers_) * 2 + 2048;
    struct ggml_context * new_ctx = ggml_init({ meta_bytes, nullptr, true });
    if (!new_ctx) return false;

    std::vector<struct ggml_tensor *> new_k(n_local_layers_, nullptr);
    std::vector<struct ggml_tensor *> new_v(n_local_layers_, nullptr);

    for (int l = 0; l < n_local_layers_; ++l) {
        new_k[l] = ggml_new_tensor_2d(new_ctx, dtype_, k_dim, static_cast<int64_t>(capacity));
        new_v[l] = ggml_new_tensor_2d(new_ctx, dtype_, v_dim, static_cast<int64_t>(capacity));
        ggml_format_name(new_k[l], "kv_k_l%d", l);
        ggml_format_name(new_v[l], "kv_v_l%d", l);
    }

    ggml_backend_buffer_t new_buf = ggml_backend_alloc_ctx_tensors(new_ctx, backend_);
    if (!new_buf) {
        ggml_free(new_ctx);
        return false;
    }

    ctx_      = new_ctx;
    buf_      = new_buf;
    k_layers_ = std::move(new_k);
    v_layers_ = std::move(new_v);
    capacity_ = capacity;
    return true;
}

struct ggml_tensor * KVCache::write_view_k(struct ggml_context * ctx,
                                           int                   layer_idx,
                                           size_t                n_past,
                                           size_t                n_tokens) const {
    struct ggml_tensor * base = k_layers_[layer_idx];
    return ggml_view_2d(ctx, base, base->ne[0], static_cast<int64_t>(n_tokens),
                        base->nb[1], n_past * base->nb[1]);
}

struct ggml_tensor * KVCache::write_view_v(struct ggml_context * ctx,
                                           int                   layer_idx,
                                           size_t                n_past,
                                           size_t                n_tokens) const {
    struct ggml_tensor * base = v_layers_[layer_idx];
    return ggml_view_2d(ctx, base, base->ne[0], static_cast<int64_t>(n_tokens),
                        base->nb[1], n_past * base->nb[1]);
}

struct ggml_tensor * KVCache::read_view_k(struct ggml_context * ctx,
                                          int                   layer_idx,
                                          size_t                n_total) const {
    struct ggml_tensor * base = k_layers_[layer_idx];
    const size_t elem_size    = ggml_type_size(dtype_);
    return ggml_view_3d(ctx, base,
                        config_.d_head_k,
                        config_.n_kv_heads,
                        static_cast<int64_t>(n_total),
                        static_cast<size_t>(config_.d_head_k) * elem_size,
                        base->nb[1],
                        0);
}

struct ggml_tensor * KVCache::read_view_v(struct ggml_context * ctx,
                                          int                   layer_idx,
                                          size_t                n_total) const {
    struct ggml_tensor * base = v_layers_[layer_idx];
    const size_t elem_size    = ggml_type_size(dtype_);
    return ggml_view_3d(ctx, base,
                        config_.d_head_v,
                        config_.n_kv_heads,
                        static_cast<int64_t>(n_total),
                        static_cast<size_t>(config_.d_head_v) * elem_size,
                        base->nb[1],
                        0);
}

} // namespace reasoner
} // namespace opengp
