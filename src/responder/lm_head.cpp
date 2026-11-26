#include "responder/lm_head.h"
#include "reasoner/backend_selector.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <exception>
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

    // Choose backend: GPU (CUDA -> Metal -> Vulkan) if requested or AUTO with sufficient free VRAM headroom, else CPU mmap
    if (device_pref == LmHeadDevice::GPU || device_pref == LmHeadDevice::AUTO) {
        using reasoner::GpuBackendKind;
        using reasoner::GpuBackendSelection;

        const bool allow_slow_igpu = (device_pref == LmHeadDevice::GPU);
        GpuBackendSelection sel = reasoner::select_gpu_backend(gpu_device_id, n_threads, allow_slow_igpu);
        if (sel.backend) {
            bool fits_budget = (device_pref == LmHeadDevice::GPU);
            if (device_pref == LmHeadDevice::AUTO) {
                size_t free_vram = 0, total_vram = 0;
                ggml_backend_dev_t dev = ggml_backend_get_device(sel.backend);
                if (dev) {
                    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
                }
                // On Vulkan iGPU (shared system RAM), cap usable budget to 50% of reported free memory
                size_t usable_mem = (sel.kind == GpuBackendKind::VULKAN && sel.is_igpu)
                                        ? (free_vram / 2)
                                        : free_vram;
                constexpr size_t kVramHeadroom = 256ULL * 1024ULL * 1024ULL;
                fits_budget = (usable_mem > weight_bytes_ + kVramHeadroom);
            }

            if (fits_budget) {
                bool uploaded = false;
                std::string upload_err;
                try {
                    gpu_ctx_  = ggml_init({ ggml_tensor_overhead() * 8 + 1024, nullptr, true });
                    active_w_ = ggml_dup_tensor(gpu_ctx_, host_w_);
                    ggml_set_name(active_w_, ggml_get_name(host_w_));

                    // Preflight op support for Vulkan (catches unsupported quant types or maxStorageBufferRange limits)
                    bool op_ok = true;
                    if (sel.kind == GpuBackendKind::VULKAN) {
                        struct ggml_tensor * probe_in  = ggml_new_tensor_1d(gpu_ctx_, GGML_TYPE_F32, static_cast<int64_t>(d_model_));
                        struct ggml_tensor * probe_out = ggml_mul_mat(gpu_ctx_, active_w_, probe_in);
                        op_ok = ggml_backend_supports_op(sel.backend, probe_out);
                        if (!op_ok) {
                            upload_err = std::string("Vulkan does not support LM Head MUL_MAT (") +
                                         ggml_type_name(active_w_->type) + ")";
                        }
                        // Rebuild clean gpu_ctx_ containing only active_w_ so only weights are allocated
                        ggml_free(gpu_ctx_);
                        gpu_ctx_  = nullptr;
                        active_w_ = nullptr;
                        if (op_ok) {
                            gpu_ctx_  = ggml_init({ ggml_tensor_overhead() * 4 + 1024, nullptr, true });
                            active_w_ = ggml_dup_tensor(gpu_ctx_, host_w_);
                            ggml_set_name(active_w_, ggml_get_name(host_w_));
                        }
                    }

                    if (op_ok) {
                        gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, sel.backend);
                        if (gpu_buf_) {
                            ggml_backend_buffer_set_usage(gpu_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                            ggml_backend_tensor_set(active_w_, host_w_->data, 0, weight_bytes_);

                            backend_      = sel.backend;
                            backend_name_ = sel.label;
                            is_gpu_       = true;
                            uploaded      = true;
                        } else {
                            upload_err = "GPU weight buffer allocation failed";
                        }
                    }
                } catch (const std::exception & e) {
                    upload_err = std::string("exception: ") + e.what();
                } catch (...) {
                    upload_err = "unknown exception";
                }

                if (!uploaded) {
                    if (gpu_buf_) {
                        ggml_backend_buffer_free(gpu_buf_);
                        gpu_buf_ = nullptr;
                    }
                    if (gpu_ctx_) {
                        ggml_free(gpu_ctx_);
                        gpu_ctx_ = nullptr;
                    }
                    active_w_ = nullptr;
                    ggml_backend_free(sel.backend);
                    if (device_pref == LmHeadDevice::GPU) {
                        std::fprintf(stderr, "[-] LmHead: GPU placement failed (%s)\n", upload_err.c_str());
                        cleanup();
                        return false;
                    }
                    if (!upload_err.empty()) {
                        std::fprintf(stderr, "[!] LmHead: %s — falling back to CPU mmap\n", upload_err.c_str());
                    }
                }
            } else {
                ggml_backend_free(sel.backend);
            }
        } else if (device_pref == LmHeadDevice::GPU) {
            const char * reason = sel.fail_reason.empty()
                                      ? "no CUDA, Metal, or Vulkan GPU device available"
                                      : sel.fail_reason.c_str();
            std::fprintf(stderr, "[-] LmHead: GPU requested, but no GPU backend available (%s)\n",
                         reason);
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

    try {
        ggml_backend_tensor_set(in_vec_, hidden_vec, 0, d_model_ * sizeof(float));
        if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS) {
            return false;
        }

        out_logits.resize(vocab_size_);
        ggml_backend_tensor_get(out_vec_, out_logits.data(), 0, vocab_size_ * sizeof(float));
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[-] LmHead: Backend exception during projection: %s\n", e.what());
        return false;
    } catch (...) {
        std::fprintf(stderr, "[-] LmHead: Unknown backend exception during projection\n");
        return false;
    }
}

} // namespace responder
} // namespace opengp
