#include "reasoner/weight_store.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace opengp {
namespace reasoner {

namespace {

size_t pad_up(size_t val, size_t align) {
    if (align == 0) return val;
    return ((val + align - 1) / align) * align;
}

// Uniform tensor visitor supporting Standard Attention, Gated Attention, and Gated DeltaNet SSM layers
template <typename Fn>
void visit_layer_tensors_impl(const LayerWeights & w, Fn && fn) {
    const bool is_ssm = (w.kind == LayerKind::GATED_DELTANET_SSM);

    // Core shared norm + SwiGLU FFN tensors
    fn("attn_norm",   w.attn_norm,   true);
    fn("ffn_norm",    w.ffn_norm,    true);
    fn("w_gate",      w.w_gate,      true);
    fn("w_up",        w.w_up,        true);
    fn("w_down",      w.w_down,      true);

    // Attention tensors (required when not SSM)
    fn("wq",          w.wq,          !is_ssm);
    fn("wk",          w.wk,          !is_ssm);
    fn("wv",          w.wv,          !is_ssm);
    fn("wo",          w.wo,          !is_ssm);
    fn("attn_q_norm", w.attn_q_norm, false);
    fn("attn_k_norm", w.attn_k_norm, false);
    fn("bq",          w.bq,          false);
    fn("bk",          w.bk,          false);
    fn("bv",          w.bv,          false);
    fn("bo",          w.bo,          false);

    // Gated DeltaNet SSM tensors (required when SSM)
    fn("attn_qkv",    w.attn_qkv,    is_ssm);
    fn("attn_gate",   w.attn_gate,   is_ssm);
    fn("ssm_conv1d",  w.ssm_conv1d,  is_ssm);
    fn("ssm_alpha",   w.ssm_alpha,   is_ssm);
    fn("ssm_beta",    w.ssm_beta,    is_ssm);
    fn("ssm_a",       w.ssm_a,       is_ssm);
    fn("ssm_dt_bias", w.ssm_dt_bias, is_ssm);
    fn("ssm_norm",    w.ssm_norm,    is_ssm);
    fn("ssm_out",     w.ssm_out,     is_ssm);
}

// Uniform pair-wise tensor mapping for cloning or staging a LayerWeights struct
template <typename CopyFn>
void transform_layer_tensors(const LayerWeights & src, LayerWeights & dst, CopyFn && copy_fn) {
    dst.kind        = src.kind;
    dst.total_bytes = src.total_bytes;

    copy_fn(dst.attn_norm,   src.attn_norm);
    copy_fn(dst.ffn_norm,    src.ffn_norm);
    copy_fn(dst.w_gate,      src.w_gate);
    copy_fn(dst.w_up,        src.w_up);
    copy_fn(dst.w_down,      src.w_down);
    copy_fn(dst.wq,          src.wq);
    copy_fn(dst.wk,          src.wk);
    copy_fn(dst.wv,          src.wv);
    copy_fn(dst.wo,          src.wo);
    copy_fn(dst.attn_q_norm, src.attn_q_norm);
    copy_fn(dst.attn_k_norm, src.attn_k_norm);
    copy_fn(dst.bq,          src.bq);
    copy_fn(dst.bk,          src.bk);
    copy_fn(dst.bv,          src.bv);
    copy_fn(dst.bo,          src.bo);
    copy_fn(dst.attn_qkv,    src.attn_qkv);
    copy_fn(dst.attn_gate,   src.attn_gate);
    copy_fn(dst.ssm_conv1d,  src.ssm_conv1d);
    copy_fn(dst.ssm_alpha,   src.ssm_alpha);
    copy_fn(dst.ssm_beta,    src.ssm_beta);
    copy_fn(dst.ssm_a,       src.ssm_a);
    copy_fn(dst.ssm_dt_bias, src.ssm_dt_bias);
    copy_fn(dst.ssm_norm,    src.ssm_norm);
    copy_fn(dst.ssm_out,     src.ssm_out);
}

struct ggml_tensor * clone_tensor_meta(struct ggml_context * ctx, const struct ggml_tensor * src) {
    if (!src) return nullptr;
    struct ggml_tensor * dst = ggml_dup_tensor(ctx, src);
    ggml_set_name(dst, ggml_get_name(src));
    return dst;
}

} // namespace

WeightStore::WeightStore() = default;

WeightStore::~WeightStore() {
    cleanup();
}

void WeightStore::cleanup() {
    if (stage_buf_) {
        ggml_backend_buffer_free(stage_buf_);
        stage_buf_ = nullptr;
    }
    if (stage_ctx_) {
        ggml_free(stage_ctx_);
        stage_ctx_ = nullptr;
    }
    if (resident_buf_) {
        ggml_backend_buffer_free(resident_buf_);
        resident_buf_ = nullptr;
    }
    if (resident_ctx_) {
        ggml_free(resident_ctx_);
        resident_ctx_ = nullptr;
    }
    if (global_buf_) {
        ggml_backend_buffer_free(global_buf_);
        global_buf_ = nullptr;
    }
    if (global_ctx_) {
        ggml_free(global_ctx_);
        global_ctx_ = nullptr;
    }
    if (host_ctx_) {
        ggml_free(host_ctx_);
        host_ctx_ = nullptr;
    }
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_size_);
        mmap_addr_ = nullptr;
        mmap_size_ = 0;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    host_output_norm_  = nullptr;
    host_rope_freqs_   = nullptr;
    host_lm_head_      = nullptr;
    output_norm_       = nullptr;
    rope_freqs_        = nullptr;
    lm_head_           = nullptr;
    n_local_layers_    = 0;
    n_resident_layers_ = 0;
    n_streamed_layers_ = 0;
    host_layers_.clear();
    active_layers_.clear();
}

std::vector<LayerKind> WeightStore::detect_layer_kinds(const struct gguf_context * gctx,
                                                         const ModelConfig &         config,
                                                         const LayerRange &          range) {
    const int n_local = range.count();
    std::vector<LayerKind> kinds(n_local, LayerKind::STANDARD_ATTENTION);
    if (!gctx || n_local <= 0) return kinds;

    const int64_t expected_q_dim = static_cast<int64_t>(config.n_heads) * config.d_head_k;
    for (int local_l = 0; local_l < n_local; ++local_l) {
        const int global_l = range.start + local_l;
        const std::string prefix = "blk." + std::to_string(global_l) + ".";

        if (gguf_find_tensor(gctx, (prefix + "ssm_out.weight").c_str()) >= 0) {
            kinds[local_l] = LayerKind::GATED_DELTANET_SSM;
        } else {
            int64_t q_idx = gguf_find_tensor(gctx, (prefix + "attn_q.weight").c_str());
            if (q_idx >= 0) {
                const int64_t * ne = gguf_get_tensor_ne(gctx, q_idx);
                if (ne[1] == expected_q_dim * 2) {
                    kinds[local_l] = LayerKind::GATED_ATTENTION;
                }
            }
        }
    }
    return kinds;
}

std::vector<LayerKind> WeightStore::get_layer_kinds() const {
    std::vector<LayerKind> kinds;
    kinds.reserve(host_layers_.size());
    for (const auto & lw : host_layers_) {
        kinds.push_back(lw.kind);
    }
    return kinds;
}

bool WeightStore::init(const std::string &   gguf_filepath,
                       struct gguf_context * gctx,
                       const ModelConfig &   config,
                       const LayerRange &    range,
                       ggml_backend_t        backend,
                       bool                  is_gpu,
                       size_t                available_vram_bytes) {
    cleanup();
    is_gpu_         = is_gpu;
    range_          = range;
    n_local_layers_ = range.count();

    if (!gctx || !backend || n_local_layers_ <= 0) return false;

    const bool need_output_norm = range_.should_apply_final_norm(config.n_layers);

    // 1. Parse tensor metadata via gguf_init_from_file(no_alloc=true) to build host ggml_tensors
    struct gguf_init_params params = { /*no_alloc=*/true, &host_ctx_ };
    struct gguf_context * tensor_gctx = gguf_init_from_file(gguf_filepath.c_str(), params);
    if (!tensor_gctx || !host_ctx_) {
        if (tensor_gctx) gguf_free(tensor_gctx);
        std::fprintf(stderr, "[-] WeightStore: Failed to initialize tensor metadata from '%s'\n",
                     gguf_filepath.c_str());
        return false;
    }

    // 2. mmap the GGUF file read-only for zero-copy host tensor data pointers
    fd_ = open(gguf_filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        gguf_free(tensor_gctx);
        cleanup();
        return false;
    }
    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        gguf_free(tensor_gctx);
        cleanup();
        return false;
    }
    mmap_size_ = static_cast<size_t>(sb.st_size);
    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        mmap_addr_ = nullptr;
        gguf_free(tensor_gctx);
        cleanup();
        return false;
    }

    uint8_t * data_base = static_cast<uint8_t *>(mmap_addr_) + gguf_get_data_offset(tensor_gctx);
    const int64_t n_tensors = gguf_get_n_tensors(tensor_gctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(tensor_gctx, i);
        struct ggml_tensor * t = ggml_get_tensor(host_ctx_, name);
        if (t) {
            t->data = data_base + gguf_get_tensor_offset(tensor_gctx, i);
        }
    }
    gguf_free(tensor_gctx);

    // 3. Resolve optional global tensors and owned blk.<global_l>.* slice
    if (need_output_norm) {
        host_output_norm_ = ggml_get_tensor(host_ctx_, "output_norm.weight");
        if (!host_output_norm_) {
            std::fprintf(stderr, "[-] WeightStore: Required tensor 'output_norm.weight' missing\n");
            cleanup();
            return false;
        }
    }
    host_rope_freqs_ = ggml_get_tensor(host_ctx_, "rope_freqs.weight");
    host_lm_head_    = ggml_get_tensor(host_ctx_, "output.weight");
    if (!host_lm_head_) {
        host_lm_head_ = ggml_get_tensor(host_ctx_, "token_embd.weight");
    }

    host_layers_.resize(n_local_layers_);
    total_weight_bytes_ = (host_output_norm_ ? ggml_nbytes(host_output_norm_) : 0) +
                          (host_rope_freqs_ ? ggml_nbytes(host_rope_freqs_) : 0);
    max_layer_bytes_ = 0;

    const int64_t expected_q_dim = static_cast<int64_t>(config.n_heads) * config.d_head_k;

    for (int local_l = 0; local_l < n_local_layers_; ++local_l) {
        const int    global_l = range_.start + local_l;
        std::string  prefix   = "blk." + std::to_string(global_l) + ".";
        LayerWeights & lw     = host_layers_[local_l];

        // Shared norms & FFN weights (with post_attention_norm fallback for hybrid architectures)
        lw.attn_norm = ggml_get_tensor(host_ctx_, (prefix + "attn_norm.weight").c_str());
        lw.ffn_norm  = ggml_get_tensor(host_ctx_, (prefix + "ffn_norm.weight").c_str());
        if (!lw.ffn_norm) {
            lw.ffn_norm = ggml_get_tensor(host_ctx_, (prefix + "post_attention_norm.weight").c_str());
        }
        lw.w_gate = ggml_get_tensor(host_ctx_, (prefix + "ffn_gate.weight").c_str());
        lw.w_up   = ggml_get_tensor(host_ctx_, (prefix + "ffn_up.weight").c_str());
        lw.w_down = ggml_get_tensor(host_ctx_, (prefix + "ffn_down.weight").c_str());

        // Data-driven block kind detection from present tensors
        struct ggml_tensor * ssm_out = ggml_get_tensor(host_ctx_, (prefix + "ssm_out.weight").c_str());
        if (ssm_out) {
            lw.kind        = LayerKind::GATED_DELTANET_SSM;
            lw.attn_qkv    = ggml_get_tensor(host_ctx_, (prefix + "attn_qkv.weight").c_str());
            lw.attn_gate   = ggml_get_tensor(host_ctx_, (prefix + "attn_gate.weight").c_str());
            lw.ssm_conv1d  = ggml_get_tensor(host_ctx_, (prefix + "ssm_conv1d.weight").c_str());
            lw.ssm_alpha   = ggml_get_tensor(host_ctx_, (prefix + "ssm_alpha.weight").c_str());
            lw.ssm_beta    = ggml_get_tensor(host_ctx_, (prefix + "ssm_beta.weight").c_str());
            lw.ssm_a       = ggml_get_tensor(host_ctx_, (prefix + "ssm_a").c_str());
            lw.ssm_dt_bias = ggml_get_tensor(host_ctx_, (prefix + "ssm_dt.bias").c_str());
            lw.ssm_norm    = ggml_get_tensor(host_ctx_, (prefix + "ssm_norm.weight").c_str());
            lw.ssm_out     = ssm_out;
        } else {
            lw.wq          = ggml_get_tensor(host_ctx_, (prefix + "attn_q.weight").c_str());
            lw.wk          = ggml_get_tensor(host_ctx_, (prefix + "attn_k.weight").c_str());
            lw.wv          = ggml_get_tensor(host_ctx_, (prefix + "attn_v.weight").c_str());
            lw.wo          = ggml_get_tensor(host_ctx_, (prefix + "attn_output.weight").c_str());
            lw.attn_q_norm = ggml_get_tensor(host_ctx_, (prefix + "attn_q_norm.weight").c_str());
            lw.attn_k_norm = ggml_get_tensor(host_ctx_, (prefix + "attn_k_norm.weight").c_str());
            lw.bq          = ggml_get_tensor(host_ctx_, (prefix + "attn_q.bias").c_str());
            lw.bk          = ggml_get_tensor(host_ctx_, (prefix + "attn_k.bias").c_str());
            lw.bv          = ggml_get_tensor(host_ctx_, (prefix + "attn_v.bias").c_str());
            lw.bo          = ggml_get_tensor(host_ctx_, (prefix + "attn_output.bias").c_str());

            lw.kind = (lw.wq && lw.wq->ne[1] == expected_q_dim * 2)
                          ? LayerKind::GATED_ATTENTION
                          : LayerKind::STANDARD_ATTENTION;
        }

        bool missing_req = false;
        lw.total_bytes = 0;
        visit_layer_tensors_impl(lw, [&](const char * field, const struct ggml_tensor * t, bool required) {
            if (required && !t) {
                std::fprintf(stderr, "[-] WeightStore: Missing required weight '%s' in global layer %d\n",
                             field, global_l);
                missing_req = true;
            }
            if (t) {
                lw.total_bytes += ggml_nbytes(t);
            }
        });
        if (missing_req) {
            cleanup();
            return false;
        }

        total_weight_bytes_ += lw.total_bytes;
        max_layer_bytes_ = std::max(max_layer_bytes_, lw.total_bytes);
    }

    // Keep lm_head pointing directly to mmap host memory (0 VRAM cost)
    lm_head_ = host_lm_head_;

    // 4. CPU Backend: zero-copy mmap pointers directly
    if (!is_gpu_) {
        active_layers_     = host_layers_;
        output_norm_       = host_output_norm_;
        rope_freqs_        = host_rope_freqs_;
        n_resident_layers_ = 0;
        n_streamed_layers_ = 0;
        vram_used_bytes_   = 0;
        return true;
    }

    // 5. GPU Backend: Global weights uploaded once + hybrid resident / streaming plan over owned slice
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t align = ggml_backend_buft_get_alignment(buft);

    // Compute exact padded backend alloc size per local layer
    std::vector<size_t> layer_alloc_bytes(n_local_layers_, 0);
    size_t max_layer_alloc = 0;
    constexpr size_t kMaxTensorsPerLayer = 24;
    for (int l = 0; l < n_local_layers_; ++l) {
        size_t alloc = align;
        visit_layer_tensors_impl(host_layers_[l], [&](const char *, const struct ggml_tensor * t, bool) {
            if (t) alloc += pad_up(ggml_backend_buft_get_alloc_size(buft, t), align);
        });
        layer_alloc_bytes[l] = alloc;
        max_layer_alloc = std::max(max_layer_alloc, alloc);
    }

    // Upload needed global weights (output_norm / rope_freqs) to GPU
    size_t global_alloc = 0;
    size_t n_global = (host_output_norm_ ? 1 : 0) + (host_rope_freqs_ ? 1 : 0);
    if (n_global > 0) {
        global_ctx_ = ggml_init({ ggml_tensor_overhead() * n_global + 1024, nullptr, true });
        if (host_output_norm_) output_norm_ = clone_tensor_meta(global_ctx_, host_output_norm_);
        if (host_rope_freqs_)  rope_freqs_  = clone_tensor_meta(global_ctx_, host_rope_freqs_);

        global_buf_ = ggml_backend_alloc_ctx_tensors(global_ctx_, backend);
        if (!global_buf_) {
            std::fprintf(stderr, "[-] WeightStore: Failed to allocate GPU global weights buffer\n");
            cleanup();
            return false;
        }
        ggml_backend_buffer_set_usage(global_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (output_norm_) ggml_backend_tensor_set(output_norm_, host_output_norm_->data, 0, ggml_nbytes(host_output_norm_));
        if (rope_freqs_)  ggml_backend_tensor_set(rope_freqs_,  host_rope_freqs_->data,  0, ggml_nbytes(host_rope_freqs_));
        global_alloc = ggml_backend_buffer_get_size(global_buf_);
    }

    // Plan how many leading owned layers fit permanently in available VRAM
    size_t budget = (available_vram_bytes > global_alloc) ? (available_vram_bytes - global_alloc) : 0;
    size_t total_resident_sum = 0;
    int resident_count = 0;

    size_t all_layers_alloc = 0;
    for (int l = 0; l < n_local_layers_; ++l) all_layers_alloc += layer_alloc_bytes[l];

    if (all_layers_alloc <= budget) {
        resident_count = n_local_layers_;
    } else {
        size_t res_budget = (budget > max_layer_alloc) ? (budget - max_layer_alloc) : 0;
        for (int l = 0; l < n_local_layers_; ++l) {
            if (total_resident_sum + layer_alloc_bytes[l] <= res_budget) {
                total_resident_sum += layer_alloc_bytes[l];
                resident_count++;
            } else {
                break;
            }
        }
    }

    n_resident_layers_ = resident_count;
    n_streamed_layers_ = n_local_layers_ - resident_count;
    active_layers_.resize(n_local_layers_);
    vram_used_bytes_ = global_alloc;

    // Allocate and upload GPU resident layers
    if (n_resident_layers_ > 0) {
        size_t meta_bytes = ggml_tensor_overhead() * kMaxTensorsPerLayer * n_resident_layers_ + 4096;
        resident_ctx_ = ggml_init({ meta_bytes, nullptr, true });

        std::vector<std::pair<struct ggml_tensor *, const struct ggml_tensor *>> copies;
        copies.reserve(kMaxTensorsPerLayer * n_resident_layers_);

        for (int l = 0; l < n_resident_layers_; ++l) {
            active_layers_[l].is_gpu_resident = true;
            transform_layer_tensors(host_layers_[l], active_layers_[l],
                                    [&](struct ggml_tensor *& out_ptr, const struct ggml_tensor * in_ptr) {
                                        if (in_ptr) {
                                            out_ptr = clone_tensor_meta(resident_ctx_, in_ptr);
                                            copies.push_back({ out_ptr, in_ptr });
                                        }
                                    });
        }

        resident_buf_ = ggml_backend_alloc_ctx_tensors(resident_ctx_, backend);
        if (!resident_buf_) {
            std::fprintf(stderr, "[-] WeightStore: Failed to allocate resident GPU buffer\n");
            cleanup();
            return false;
        }
        ggml_backend_buffer_set_usage(resident_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        vram_used_bytes_ += ggml_backend_buffer_get_size(resident_buf_);

        for (const auto & pair : copies) {
            ggml_backend_tensor_set(pair.first, pair.second->data, 0, ggml_nbytes(pair.second));
        }
    }

    // Allocate reusable single-layer GPU staging buffer if any layers stream
    if (n_streamed_layers_ > 0) {
        size_t stage_meta_bytes = ggml_tensor_overhead() * kMaxTensorsPerLayer + 2048;
        stage_ctx_ = ggml_init({ stage_meta_bytes, nullptr, true });
        stage_buf_ = ggml_backend_alloc_buffer(backend, max_layer_alloc);
        if (!stage_ctx_ || !stage_buf_) {
            std::fprintf(stderr, "[-] WeightStore: Failed to allocate single-layer GPU staging buffer (%zu bytes)\n",
                         max_layer_alloc);
            cleanup();
            return false;
        }
        ggml_backend_buffer_set_usage(stage_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        vram_used_bytes_ += ggml_backend_buffer_get_size(stage_buf_);
    }

    return true;
}

const LayerWeights & WeightStore::prepare_layer(int local_layer_idx) {
    if (!is_gpu_ || active_layers_[local_layer_idx].is_gpu_resident) {
        return active_layers_[local_layer_idx];
    }

    const LayerWeights & src = host_layers_[local_layer_idx];
    ggml_reset(stage_ctx_);
    ggml_backend_buffer_reset(stage_buf_);
    struct ggml_tallocr talloc = ggml_tallocr_new(stage_buf_);

    stage_weights_ = LayerWeights{};
    stage_weights_.is_gpu_resident = false;

    transform_layer_tensors(src, stage_weights_,
                            [&](struct ggml_tensor *& out_ptr, const struct ggml_tensor * in_ptr) {
                                if (!in_ptr) return;
                                out_ptr = clone_tensor_meta(stage_ctx_, in_ptr);
                                ggml_tallocr_alloc(&talloc, out_ptr);
                                ggml_backend_tensor_set(out_ptr, in_ptr->data, 0, ggml_nbytes(in_ptr));
                            });

    return stage_weights_;
}

} // namespace reasoner
} // namespace opengp
