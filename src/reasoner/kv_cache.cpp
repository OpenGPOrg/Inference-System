#include "reasoner/kv_cache.h"

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
    layer_kinds_.clear();
    n_local_layers_ = 0;
    n_attn_layers_  = 0;
    capacity_       = 0;
    n_past_         = 0;
}

bool KVCache::init(const ModelConfig &            config,
                   const std::vector<LayerKind> & layer_kinds,
                   ggml_backend_t                 backend,
                   KVPrecision                    precision) {
    cleanup();
    if (!backend || config.n_layers <= 0 || layer_kinds.empty()) return false;

    config_         = config;
    layer_kinds_    = layer_kinds;
    n_local_layers_ = static_cast<int>(layer_kinds.size());
    n_attn_layers_  = 0;
    for (LayerKind k : layer_kinds_) {
        if (k != LayerKind::GATED_DELTANET_SSM) {
            n_attn_layers_++;
        }
    }
    backend_ = backend;
    switch (precision) {
        case KVPrecision::F32:  dtype_ = GGML_TYPE_F32;  break;
        case KVPrecision::Q8_0: dtype_ = GGML_TYPE_Q8_0; break;
        case KVPrecision::Q4_0: dtype_ = GGML_TYPE_Q4_0; break;
        case KVPrecision::F16:
        default:                dtype_ = GGML_TYPE_F16;  break;
    }
    if (ggml_is_quantized(dtype_) && ((config_.d_head_k % 32 != 0) || (config_.d_head_v % 32 != 0))) {
        std::fprintf(stderr,
                     "[!] KVCache: d_head (%d/%d) is not divisible by block size 32 for %s — falling back to f16\n",
                     config_.d_head_k, config_.d_head_v, ggml_type_name(dtype_));
        dtype_ = GGML_TYPE_F16;
    }
    return true;
}

size_t KVCache::bytes_per_token() const {
    const size_t k_row = ggml_row_size(dtype_, static_cast<int64_t>(config_.d_head_k) * config_.n_kv_heads);
    const size_t v_row = ggml_row_size(dtype_, static_cast<int64_t>(config_.d_head_v) * config_.n_kv_heads);
    return static_cast<size_t>(n_attn_layers_) * (k_row + v_row);
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
    k_layers_.assign(n_local_layers_, nullptr);
    v_layers_.assign(n_local_layers_, nullptr);
    capacity_ = 0;
    n_past_   = 0;

    if (n_attn_layers_ == 0) {
        capacity_ = capacity;
        return true; // Pure SSM slice: 0 KV tensors needed
    }

    const int64_t k_dim = static_cast<int64_t>(config_.d_head_k) * config_.n_kv_heads;
    const int64_t v_dim = static_cast<int64_t>(config_.d_head_v) * config_.n_kv_heads;

    size_t meta_bytes = ggml_tensor_overhead() * static_cast<size_t>(n_attn_layers_) * 2 + 2048;
    struct ggml_context * new_ctx = ggml_init({ meta_bytes, nullptr, true });
    if (!new_ctx) return false;

    std::vector<struct ggml_tensor *> new_k(n_local_layers_, nullptr);
    std::vector<struct ggml_tensor *> new_v(n_local_layers_, nullptr);

    for (int l = 0; l < n_local_layers_; ++l) {
        if (layer_kinds_[l] == LayerKind::GATED_DELTANET_SSM) continue;

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

    // Zero-initialize KV storage so padded FlashAttention reads beyond n_total never touch NaN/Inf
    for (int l = 0; l < n_local_layers_; ++l) {
        if (k_layers_[l]) ggml_backend_tensor_memset(k_layers_[l], 0, 0, ggml_nbytes(k_layers_[l]));
        if (v_layers_[l]) ggml_backend_tensor_memset(v_layers_[l], 0, 0, ggml_nbytes(v_layers_[l]));
    }
    return true;
}

struct ggml_tensor * KVCache::write_view_k(struct ggml_context * ctx,
                                           int                   local_layer_idx,
                                           size_t                n_past,
                                           size_t                n_tokens) const {
    struct ggml_tensor * base = k_layers_[local_layer_idx];
    return ggml_view_2d(ctx, base, base->ne[0], static_cast<int64_t>(n_tokens),
                        base->nb[1], n_past * base->nb[1]);
}

struct ggml_tensor * KVCache::write_view_v(struct ggml_context * ctx,
                                           int                   local_layer_idx,
                                           size_t                n_past,
                                           size_t                n_tokens) const {
    struct ggml_tensor * base = v_layers_[local_layer_idx];
    return ggml_view_2d(ctx, base, base->ne[0], static_cast<int64_t>(n_tokens),
                        base->nb[1], n_past * base->nb[1]);
}

struct ggml_tensor * KVCache::read_view_k(struct ggml_context * ctx,
                                          int                   local_layer_idx,
                                          size_t                n_total) const {
    struct ggml_tensor * base = k_layers_[local_layer_idx];
    return ggml_view_3d(ctx, base,
                        config_.d_head_k,
                        config_.n_kv_heads,
                        static_cast<int64_t>(n_total),
                        ggml_row_size(dtype_, config_.d_head_k),
                        base->nb[1],
                        0);
}

struct ggml_tensor * KVCache::read_view_v(struct ggml_context * ctx,
                                          int                   local_layer_idx,
                                          size_t                n_total) const {
    struct ggml_tensor * base = v_layers_[local_layer_idx];
    return ggml_view_3d(ctx, base,
                        config_.d_head_v,
                        config_.n_kv_heads,
                        static_cast<int64_t>(n_total),
                        ggml_row_size(dtype_, config_.d_head_v),
                        base->nb[1],
                        0);
}

} // namespace reasoner
} // namespace opengp
