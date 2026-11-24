#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstddef>
#include <string>
#include <vector>

namespace opengp {
namespace responder {

enum class LmHeadDevice {
    AUTO, // Use CUDA GPU if weight fits in remaining free VRAM with headroom, else CPU mmap
    GPU,  // Force CUDA GPU
    CPU   // Force multi-threaded Host mmap CPU (0 VRAM)
};

// Persistent Unembedding / LM Head projector: h_last [d_model] -> logits [vocab_size]
class LmHead {
public:
    LmHead();
    ~LmHead();

    LmHead(const LmHead &) = delete;
    LmHead & operator=(const LmHead &) = delete;

    // Memory-map output.weight (fallback token_embd.weight for tied embeddings) and allocate compute graph once
    bool init(const std::string & gguf_filepath,
              LmHeadDevice        device_pref   = LmHeadDevice::AUTO,
              int                 gpu_device_id = 0,
              int                 n_threads     = 0);

    // Project a single hidden state vector [d_model] -> FP32 logits [vocab_size]
    bool project(const float * hidden_vec, std::vector<float> & out_logits);

    size_t get_vocab_size() const { return vocab_size_; }
    size_t get_d_model() const { return d_model_; }
    size_t get_weight_bytes() const { return weight_bytes_; }
    bool   is_gpu() const { return is_gpu_; }
    bool   is_tied() const { return is_tied_; }
    const std::string & get_backend_name() const { return backend_name_; }

private:
    void cleanup();

    int    fd_        = -1;
    void * mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;

    struct ggml_context * host_ctx_ = nullptr;
    struct ggml_tensor *  host_w_   = nullptr;

    ggml_backend_t        backend_  = nullptr;
    struct ggml_context * gpu_ctx_  = nullptr;
    ggml_backend_buffer_t gpu_buf_  = nullptr;
    struct ggml_tensor *  active_w_ = nullptr;

    // Reusable 1-vector projection graph + scratch allocator
    struct ggml_context * graph_ctx_ = nullptr;
    ggml_gallocr_t        gallocr_   = nullptr;
    struct ggml_tensor *  in_vec_    = nullptr;
    struct ggml_tensor *  out_vec_   = nullptr;
    struct ggml_cgraph *  graph_     = nullptr;

    size_t vocab_size_   = 0;
    size_t d_model_      = 0;
    size_t weight_bytes_ = 0;
    bool   is_gpu_       = false;
    bool   is_tied_      = false;
    std::string backend_name_;
};

} // namespace responder
} // namespace opengp
