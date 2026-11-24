#include "responder/lm_head.h"
#include "ggml-cpu.h"
#include "gguf.h"

#if OPENGP_HAS_CUDA
#include "ggml-cuda.h"
#endif

#if OPENGP_HAS_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace opengp {
namespace responder {

LmHead::LmHead() = default;

LmHead::~LmHead() {
    cleanup();
}

void LmHead::cleanup() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
    if (graph_ctx_) {
        ggml_free(graph_ctx_);
        graph_ctx_ = nullptr;
    }
    if (gpu_buf_) {
        ggml_backend_buffer_free(gpu_buf_);
        gpu_buf_ = nullptr;
    }
    if (gpu_ctx_) {
        ggml_free(gpu_ctx_);
        gpu_ctx_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
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
    host_w_   = nullptr;
    active_w_ = nullptr;
    in_vec_   = nullptr;
    out_vec_  = nullptr;
    graph_    = nullptr;
    is_gpu_   = false;
}

bool LmHead::init(const std::string & gguf_filepath,
                  LmHeadDevice        device_pref,
                  int                 gpu_device_id,
                  int                 n_threads) {
    cleanup();

    struct gguf_init_params params = { /*no_alloc=*/true, &host_ctx_ };
    struct gguf_context * gctx = gguf_init_from_file(gguf_filepath.c_str(), params);
    if (!gctx || !host_ctx_) {
        if (gctx) gguf_free(gctx);
        std::fprintf(stderr, "[-] LmHead: Failed to open GGUF '%s'\n", gguf_filepath.c_str());
        cleanup();
        return false;
    }

    const char * tensor_name = "output.weight";
    int64_t tensor_idx = gguf_find_tensor(gctx, tensor_name);
    if (tensor_idx < 0) {
        tensor_name = "token_embd.weight"; // Weight tying fallback
        tensor_idx = gguf_find_tensor(gctx, tensor_name);
        is_tied_ = true;
    } else {
        is_tied_ = false;
    }

    if (tensor_idx < 0) {
        std::fprintf(stderr, "[-] LmHead: Neither 'output.weight' nor 'token_embd.weight' found\n");
        gguf_free(gctx);
        cleanup();
        return false;
    }

    host_w_ = ggml_get_tensor(host_ctx_, tensor_name);
    if (!host_w_) {
        gguf_free(gctx);
        cleanup();
        return false;
    }

    d_model_      = static_cast<size_t>(host_w_->ne[0]);
    vocab_size_   = static_cast<size_t>(host_w_->ne[1]);
    weight_bytes_ = ggml_nbytes(host_w_);

    // mmap the GGUF file read-only for zero-copy weight access
    fd_ = open(gguf_filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        gguf_free(gctx);
        cleanup();
        return false;
    }
    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        gguf_free(gctx);
        cleanup();
        return false;
    }
    mmap_size_ = static_cast<size_t>(sb.st_size);
    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        mmap_addr_ = nullptr;
        gguf_free(gctx);
        cleanup();
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(gctx);
    const size_t tensor_off  = gguf_get_tensor_offset(gctx, tensor_idx);
    host_w_->data = static_cast<uint8_t *>(mmap_addr_) + data_offset + tensor_off;
    gguf_free(gctx);

    // Choose backend: GPU (CUDA or Apple Metal) if requested or AUTO with sufficient free VRAM headroom, else CPU mmap
    if (device_pref == LmHeadDevice::GPU || device_pref == LmHeadDevice::AUTO) {
        ggml_backend_t candidate_gpu = nullptr;
        std::string    gpu_label;

#if OPENGP_HAS_CUDA
        const int gpu_count = ggml_backend_cuda_get_device_count();
        if (gpu_count > 0 && gpu_device_id >= 0 && gpu_device_id < gpu_count) {
            candidate_gpu = ggml_backend_cuda_init(gpu_device_id);
            if (candidate_gpu) {
                char desc[256] = {};
                ggml_backend_cuda_get_device_description(gpu_device_id, desc, sizeof(desc));
                gpu_label = std::string("CUDA (") + desc + ")";
            }
        }
#endif

#if OPENGP_HAS_METAL
        if (!candidate_gpu) {
            candidate_gpu = ggml_backend_metal_init();
            if (candidate_gpu) {
                ggml_backend_dev_t dev = ggml_backend_get_device(candidate_gpu);
                const char * desc = dev ? ggml_backend_dev_description(dev) : "Apple GPU";
                gpu_label = std::string("Metal (") + (desc ? desc : "Apple GPU") + ")";
            }
        }
#endif

        if (candidate_gpu) {
            bool fits_budget = (device_pref == LmHeadDevice::GPU);
            if (device_pref == LmHeadDevice::AUTO) {
                size_t free_vram = 0, total_vram = 0;
                ggml_backend_dev_t dev = ggml_backend_get_device(candidate_gpu);
                if (dev) {
                    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
                }
                constexpr size_t kVramHeadroom = 256ULL * 1024ULL * 1024ULL;
                fits_budget = (free_vram > weight_bytes_ + kVramHeadroom);
            }

            if (fits_budget) {
                gpu_ctx_  = ggml_init({ ggml_tensor_overhead() * 4 + 1024, nullptr, true });
                active_w_ = ggml_dup_tensor(gpu_ctx_, host_w_);
                ggml_set_name(active_w_, ggml_get_name(host_w_));
                gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, candidate_gpu);
                if (gpu_buf_) {
                    ggml_backend_buffer_set_usage(gpu_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_tensor_set(active_w_, host_w_->data, 0, weight_bytes_);

                    backend_      = candidate_gpu;
                    backend_name_ = gpu_label;
                    is_gpu_       = true;
                } else {
                    ggml_free(gpu_ctx_);
                    gpu_ctx_ = nullptr;
                    ggml_backend_free(candidate_gpu);
                }
            } else {
                ggml_backend_free(candidate_gpu);
            }
        } else if (device_pref == LmHeadDevice::GPU) {
            std::fprintf(stderr, "[-] LmHead: GPU requested, but no CUDA or Metal device available\n");
            cleanup();
            return false;
        }
    }

    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) {
            std::fprintf(stderr, "[-] LmHead: ggml_backend_cpu_init failed\n");
            cleanup();
            return false;
        }
        int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        int threads    = (n_threads > 0) ? n_threads : std::max(1, hw_threads);
        ggml_backend_cpu_set_n_threads(backend_, threads);

        active_w_     = host_w_;
        is_gpu_       = false;
        backend_name_ = "CPU mmap (" + std::to_string(threads) + " threads)";
    }

    // Build reusable 1-vector unembedding graph once
    const size_t ctx_bytes = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    graph_ctx_ = ggml_init({ ctx_bytes, nullptr, true });
    if (!graph_ctx_) {
        cleanup();
        return false;
    }

    in_vec_ = ggml_new_tensor_1d(graph_ctx_, GGML_TYPE_F32, static_cast<int64_t>(d_model_));
    ggml_set_input(in_vec_);

    out_vec_ = ggml_mul_mat(graph_ctx_, active_w_, in_vec_);
    ggml_set_output(out_vec_);

    graph_ = ggml_new_graph(graph_ctx_);
    ggml_build_forward_expand(graph_, out_vec_);

    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!gallocr_ || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        std::fprintf(stderr, "[-] LmHead: Failed to allocate projection graph\n");
        cleanup();
        return false;
    }

    return true;
}

bool LmHead::project(const float * hidden_vec, std::vector<float> & out_logits) {
    if (!backend_ || !graph_ || !hidden_vec) return false;

    ggml_backend_tensor_set(in_vec_, hidden_vec, 0, d_model_ * sizeof(float));
    if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS) {
        return false;
    }

    out_logits.resize(vocab_size_);
    ggml_backend_tensor_get(out_vec_, out_logits.data(), 0, vocab_size_ * sizeof(float));
    return true;
}

} // namespace responder
} // namespace opengp
