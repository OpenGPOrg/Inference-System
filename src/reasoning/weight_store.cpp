#include "reasoning/weight_store.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace opengp {
namespace reasoning {

namespace {

size_t pad_up(size_t val, size_t align) {
    if (align == 0) return val;
    return ((val + align - 1) / align) * align;
}

// Uniform tensor visitor for LayerWeights (avoids duplicating code across inspection, resident, and staging copies)
template <typename Fn>
void visit_layer_tensors(LayerWeights & w, Fn && fn) {
    fn("attn_norm",   w.attn_norm,   true);
    fn("wq",          w.wq,          true);
    fn("wk",          w.wk,          true);
    fn("wv",          w.wv,          true);
    fn("wo",          w.wo,          true);
    fn("ffn_norm",    w.ffn_norm,    true);
    fn("w_gate",      w.w_gate,      true);
    fn("w_up",        w.w_up,        true);
    fn("w_down",      w.w_down,      true);
    fn("attn_q_norm", w.attn_q_norm, false);
    fn("attn_k_norm", w.attn_k_norm, false);
    fn("bq",          w.bq,          false);
    fn("bk",          w.bk,          false);
    fn("bv",          w.bv,          false);
    fn("bo",          w.bo,          false);
}

template <typename Fn>
void visit_layer_tensors_const(const LayerWeights & w, Fn && fn) {
    fn("attn_norm",   w.attn_norm,   true);
    fn("wq",          w.wq,          true);
    fn("wk",          w.wk,          true);
    fn("wv",          w.wv,          true);
    fn("wo",          w.wo,          true);
    fn("ffn_norm",    w.ffn_norm,    true);
    fn("w_gate",      w.w_gate,      true);
    fn("w_up",        w.w_up,        true);
    fn("w_down",      w.w_down,      true);
    fn("attn_q_norm", w.attn_q_norm, false);
    fn("attn_k_norm", w.attn_k_norm, false);
    fn("bq",          w.bq,          false);
    fn("bk",          w.bk,          false);
    fn("bv",          w.bv,          false);
    fn("bo",          w.bo,          false);
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
    host_output_norm_ = nullptr;
    host_rope_freqs_  = nullptr;
    host_lm_head_     = nullptr;
    output_norm_      = nullptr;
    rope_freqs_       = nullptr;
    lm_head_          = nullptr;
    n_local_layers_    = 0;
    n_resident_layers_ = 0;
    n_streamed_layers_ = 0;
    host_layers_.clear();
    active_layers_.clear();
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
        host_lm_head_ = ggml_get_tensor(host_ctx_, "token_embd.weight"); // Tied embeddings fallback
    }

    host_layers_.resize(n_local_layers_);
    total_weight_bytes_ = (host_output_norm_ ? ggml_nbytes(host_output_norm_) : 0) +
                          (host_rope_freqs_ ? ggml_nbytes(host_rope_freqs_) : 0);
    max_layer_bytes_ = 0;

    for (int local_l = 0; local_l < n_local_layers_; ++local_l) {
        const int    global_l = range_.start + local_l;
        std::string  prefix   = "blk." + std::to_string(global_l) + ".";
        LayerWeights & lw     = host_layers_[local_l];

        lw.attn_norm   = ggml_get_tensor(host_ctx_, (prefix + "attn_norm.weight").c_str());
        lw.wq          = ggml_get_tensor(host_ctx_, (prefix + "attn_q.weight").c_str());
        lw.wk          = ggml_get_tensor(host_ctx_, (prefix + "attn_k.weight").c_str());
        lw.wv          = ggml_get_tensor(host_ctx_, (prefix + "attn_v.weight").c_str());
        lw.wo          = ggml_get_tensor(host_ctx_, (prefix + "attn_output.weight").c_str());
        lw.ffn_norm    = ggml_get_tensor(host_ctx_, (prefix + "ffn_norm.weight").c_str());
        lw.w_gate      = ggml_get_tensor(host_ctx_, (prefix + "ffn_gate.weight").c_str());
        lw.w_up        = ggml_get_tensor(host_ctx_, (prefix + "ffn_up.weight").c_str());
        lw.w_down      = ggml_get_tensor(host_ctx_, (prefix + "ffn_down.weight").c_str());

        lw.attn_q_norm = ggml_get_tensor(host_ctx_, (prefix + "attn_q_norm.weight").c_str());
        lw.attn_k_norm = ggml_get_tensor(host_ctx_, (prefix + "attn_k_norm.weight").c_str());
        lw.bq          = ggml_get_tensor(host_ctx_, (prefix + "attn_q.bias").c_str());
        lw.bk          = ggml_get_tensor(host_ctx_, (prefix + "attn_k.bias").c_str());
        lw.bv          = ggml_get_tensor(host_ctx_, (prefix + "attn_v.bias").c_str());
        lw.bo          = ggml_get_tensor(host_ctx_, (prefix + "attn_output.bias").c_str());

        bool missing_req = false;
        lw.total_bytes = 0;
        visit_layer_tensors(lw, [&](const char * field, struct ggml_tensor * t, bool required) {
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
    size_t n_tensors_per_layer_max = 16;
    for (int l = 0; l < n_local_layers_; ++l) {
        size_t alloc = align; // initial offset alignment headroom
        visit_layer_tensors_const(host_layers_[l], [&](const char *, const struct ggml_tensor * t, bool) {
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

    // First check if ALL owned layers fit without needing a staging buffer
    size_t all_layers_alloc = 0;
    for (int l = 0; l < n_local_layers_; ++l) all_layers_alloc += layer_alloc_bytes[l];

    if (all_layers_alloc <= budget) {
        resident_count = n_local_layers_;
    } else {
        // Reserve one reusable staging buffer for streamed layers, fill remaining budget with resident layers
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
        size_t meta_bytes = ggml_tensor_overhead() * n_tensors_per_layer_max * n_resident_layers_ + 4096;
        resident_ctx_ = ggml_init({ meta_bytes, nullptr, true });

        std::vector<std::pair<struct ggml_tensor *, const struct ggml_tensor *>> copies;
        copies.reserve(n_tensors_per_layer_max * n_resident_layers_);

        for (int l = 0; l < n_resident_layers_; ++l) {
            const LayerWeights & src = host_layers_[l];
            LayerWeights & dst       = active_layers_[l];
            dst.total_bytes          = src.total_bytes;
            dst.is_gpu_resident      = true;

            auto clone_field = [&](struct ggml_tensor *& out_ptr, const struct ggml_tensor * in_ptr) {
                if (in_ptr) {
                    out_ptr = clone_tensor_meta(resident_ctx_, in_ptr);
                    copies.push_back({ out_ptr, in_ptr });
                }
            };
            clone_field(dst.attn_norm,   src.attn_norm);
            clone_field(dst.wq,          src.wq);
            clone_field(dst.wk,          src.wk);
            clone_field(dst.wv,          src.wv);
            clone_field(dst.wo,          src.wo);
            clone_field(dst.ffn_norm,    src.ffn_norm);
            clone_field(dst.w_gate,      src.w_gate);
            clone_field(dst.w_up,        src.w_up);
            clone_field(dst.w_down,      src.w_down);
            clone_field(dst.attn_q_norm, src.attn_q_norm);
            clone_field(dst.attn_k_norm, src.attn_k_norm);
            clone_field(dst.bq,          src.bq);
            clone_field(dst.bk,          src.bk);
            clone_field(dst.bv,          src.bv);
            clone_field(dst.bo,          src.bo);
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
        size_t stage_meta_bytes = ggml_tensor_overhead() * n_tensors_per_layer_max + 2048;
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

const LayerWeights & WeightStore::prepare_layer(int layer_idx) {
    if (!is_gpu_ || active_layers_[layer_idx].is_gpu_resident) {
        return active_layers_[layer_idx];
    }

    // Stream layer_idx from zero-copy mmap host RAM into reusable GPU staging buffer
    const LayerWeights & src = host_layers_[layer_idx];
    ggml_reset(stage_ctx_);
    ggml_backend_buffer_reset(stage_buf_);
    struct ggml_tallocr talloc = ggml_tallocr_new(stage_buf_);

    stage_weights_ = LayerWeights{};
    stage_weights_.total_bytes     = src.total_bytes;
    stage_weights_.is_gpu_resident = false;

    auto stage_tensor = [&](struct ggml_tensor *& out_ptr, const struct ggml_tensor * in_ptr) {
        if (!in_ptr) return;
        out_ptr = clone_tensor_meta(stage_ctx_, in_ptr);
        ggml_tallocr_alloc(&talloc, out_ptr);
        ggml_backend_tensor_set(out_ptr, in_ptr->data, 0, ggml_nbytes(in_ptr));
    };

    stage_tensor(stage_weights_.attn_norm,   src.attn_norm);
    stage_tensor(stage_weights_.wq,          src.wq);
    stage_tensor(stage_weights_.wk,          src.wk);
    stage_tensor(stage_weights_.wv,          src.wv);
    stage_tensor(stage_weights_.wo,          src.wo);
    stage_tensor(stage_weights_.ffn_norm,    src.ffn_norm);
    stage_tensor(stage_weights_.w_gate,      src.w_gate);
    stage_tensor(stage_weights_.w_up,        src.w_up);
    stage_tensor(stage_weights_.w_down,      src.w_down);
    stage_tensor(stage_weights_.attn_q_norm, src.attn_q_norm);
    stage_tensor(stage_weights_.attn_k_norm, src.attn_k_norm);
    stage_tensor(stage_weights_.bq,          src.bq);
    stage_tensor(stage_weights_.bk,          src.bk);
    stage_tensor(stage_weights_.bv,          src.bv);
    stage_tensor(stage_weights_.bo,          src.bo);

    return stage_weights_;
}

} // namespace reasoning
} // namespace opengp
