#include "reasoner/reasoner_engine.h"
#include "reasoner/layer_graph.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>

namespace opengp {
namespace reasoner {

namespace {

float compute_l2_norm(const float * vec, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(vec[i]) * static_cast<double>(vec[i]);
    }
    return static_cast<float>(std::sqrt(sum));
}

} // namespace

ReasoningEngine::ReasoningEngine() = default;

ReasoningEngine::~ReasoningEngine() {
    cleanup();
}

void ReasoningEngine::cleanup() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
    // Release backend-owned buffers before freeing the backend itself
    kv_cache_.release();
    weights_.release();
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    if (gctx_) {
        gguf_free(gctx_);
        gctx_ = nullptr;
    }
    backend_kind_       = GpuBackendKind::NONE;
    is_gpu_             = false;
    is_igpu_            = false;
    is_initialized_     = false;
    active_ctx_size_    = 0;
    peak_scratch_bytes_ = 0;
}

bool ReasoningEngine::init_backend() {
    const bool want_gpu = (options_.device == DevicePreference::GPU || options_.device == DevicePreference::AUTO);

    if (want_gpu) {
        const bool allow_slow_igpu = (options_.device == DevicePreference::GPU);
        GpuBackendSelection sel = select_gpu_backend(options_.gpu_device_id, options_.n_threads, allow_slow_igpu);
        if (sel.backend) {
            backend_      = sel.backend;
            backend_kind_ = sel.kind;
            backend_name_ = sel.label;
            is_gpu_       = true;
            is_igpu_      = sel.is_igpu;
            return true;
        }

        if (options_.device == DevicePreference::GPU) {
            const char * reason = sel.fail_reason.empty()
                                      ? "no CUDA, Metal, or Vulkan GPU device available"
                                      : sel.fail_reason.c_str();
            std::fprintf(stderr, "[-] ReasoningEngine: GPU requested, but no GPU backend available (%s)\n",
                         reason);
            return false;
        }

        if (!sel.fail_reason.empty()) {
            std::fprintf(stderr, "[!] ReasoningEngine: %s — falling back to CPU\n",
                         sel.fail_reason.c_str());
        }
    }

    // Multi-threaded CPU backend (primary when CPU-only, automatic fallback when no GPU)
    backend_ = ggml_backend_cpu_init();
    if (!backend_) {
        std::fprintf(stderr, "[-] ReasoningEngine: ggml_backend_cpu_init failed\n");
        return false;
    }
    int hw_threads  = static_cast<int>(std::thread::hardware_concurrency());
    active_threads_ = (options_.n_threads > 0) ? options_.n_threads : std::max(1, hw_threads);
    ggml_backend_cpu_set_n_threads(backend_, active_threads_);

    backend_kind_ = GpuBackendKind::NONE;
    backend_name_ = "CPU (" + std::to_string(active_threads_) + " threads)";
    is_gpu_       = false;
    is_igpu_      = false;
    return true;
}

bool ReasoningEngine::preflight_vulkan_ops(std::string & unsupported_desc) {
    if (backend_kind_ != GpuBackendKind::VULKAN || !backend_) {
        return true;
    }

    auto check_graph = [&](struct ggml_cgraph * gf, const char * tag) -> bool {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (!node) continue;
            if (!ggml_backend_supports_op(backend_, node)) {
                std::string types;
                if (node->src[0]) {
                    types += ggml_type_name(node->src[0]->type);
                    if (node->src[1]) {
                        types += " x ";
                        types += ggml_type_name(node->src[1]->type);
                    }
                    types += " -> ";
                }
                types += ggml_type_name(node->type);
                unsupported_desc = std::string(tag) + ": op " + ggml_op_desc(node) + " (" + types + ") unsupported";
                return false;
            }
        }
        return true;
    };

    try {
        std::vector<int> layers_to_check;
        if (layer_range_.count() > 0) {
            layers_to_check.push_back(0);
            if (layer_range_.count() > 1) {
                layers_to_check.push_back(layer_range_.count() - 1);
            }
        }

        const int max_probe_tokens = (active_ctx_size_ >= 2) ? 2 : 1;
        for (int local_l : layers_to_check) {
            const LayerWeights & lw = weights_.prepare_layer(local_l);
            for (int n_tok = 1; n_tok <= max_probe_tokens; ++n_tok) {
                const size_t ctx_bytes = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
                struct ggml_context * gctx = ggml_init({ ctx_bytes, nullptr, true });
                if (!gctx) {
                    unsupported_desc = "preflight ggml_init failed";
                    return false;
                }
                LayerGraphIO io = build_layer_graph(
                    gctx, config_, lw, weights_.get_rope_freqs(), kv_cache_, local_l, n_tok, 0);
                const bool ok = check_graph(io.graph, "layer graph");
                ggml_free(gctx);
                if (!ok) return false;
            }
        }

        if (layer_range_.should_apply_final_norm(config_.n_layers)) {
            const size_t ctx_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            struct ggml_context * gctx = ggml_init({ ctx_bytes, nullptr, true });
            if (!gctx) {
                unsupported_desc = "preflight final_norm ggml_init failed";
                return false;
            }
            FinalNormGraphIO f_io = build_final_norm_graph(gctx, config_, weights_.get_output_norm(), 1);
            const bool ok = check_graph(f_io.graph, "final_norm graph");
            ggml_free(gctx);
            if (!ok) return false;
        }
        return true;
    } catch (const std::exception & e) {
        unsupported_desc = std::string("preflight exception: ") + e.what();
        return false;
    } catch (...) {
        unsupported_desc = "preflight unknown exception";
        return false;
    }
}

bool ReasoningEngine::initialize(const std::string &      gguf_filepath,
                                 const ReasoningOptions & options) {
    return initialize_impl(gguf_filepath, options, options.device == DevicePreference::AUTO);
}

bool ReasoningEngine::initialize_impl(const std::string &      gguf_filepath,
                                      const ReasoningOptions & options,
                                      bool                     allow_vulkan_fallback) {
    cleanup();
    options_ = options;

    struct gguf_init_params gparams = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gctx_ = gguf_init_from_file(gguf_filepath.c_str(), gparams);
    if (!gctx_) {
        std::fprintf(stderr, "[-] ReasoningEngine: Failed to open GGUF '%s'\n", gguf_filepath.c_str());
        return false;
    }

    if (!config_.load_from_gguf(gctx_, options_)) {
        cleanup();
        return false;
    }

    // Resolve and validate owned layer slice [start, end)
    layer_range_ = options_.layers;
    if (layer_range_.end < 0) {
        layer_range_.end = config_.n_layers;
    }
    if (layer_range_.start < 0 || layer_range_.end > config_.n_layers || layer_range_.start >= layer_range_.end) {
        std::fprintf(stderr, "[-] ReasoningEngine: Invalid layer range [%d, %d) for n_layers=%d\n",
                     layer_range_.start, layer_range_.end, config_.n_layers);
        cleanup();
        return false;
    }
    layer_range_.apply_final_norm = layer_range_.should_apply_final_norm(config_.n_layers) ? 1 : 0;

    if (!init_backend()) {
        cleanup();
        return false;
    }

    auto fallback_to_cpu = [&](const std::string & reason) -> bool {
        if (allow_vulkan_fallback && backend_kind_ == GpuBackendKind::VULKAN) {
            std::fprintf(stderr, "[!] ReasoningEngine: %s — re-initializing on CPU\n", reason.c_str());
            ReasoningOptions cpu_opts = options;
            cpu_opts.device = DevicePreference::CPU;
            return initialize_impl(gguf_filepath, cpu_opts, /*allow_vulkan_fallback=*/false);
        }
        cleanup();
        return false;
    };

    try {
        if (!kv_cache_.init(config_, layer_range_.count(), backend_, options_.kv_precision)) {
            return fallback_to_cpu("Vulkan KV cache init failed");
        }

        // Allocate fixed KV cache upfront (default 512 tokens), halving by 2 if memory is insufficient
        const size_t requested_ctx = (options_.ctx_size > 0) ? options_.ctx_size : 512;
        size_t       target_ctx    = requested_ctx;

        const float util = std::clamp(options_.vram_utilization, 0.10f, 0.95f);
        size_t max_kv_vram        = 0;
        size_t igpu_total_budget  = 0;
        const bool is_vulkan_igpu = (backend_kind_ == GpuBackendKind::VULKAN && is_igpu_);

        if (is_gpu_) {
            size_t free_vram = 0, total_vram = 0;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
            if (dev) {
                ggml_backend_dev_memory(dev, &free_vram, &total_vram);
            }
            size_t usable_vram = static_cast<size_t>(static_cast<double>(free_vram) * util);
            if (is_vulkan_igpu) {
                // On integrated GPUs, Vulkan heaps are shared system RAM already used by the OS and mmap
                igpu_total_budget = std::min(usable_vram, free_vram / 2);
                usable_vram       = igpu_total_budget;
                std::fprintf(stderr,
                             "[!] ReasoningEngine: Vulkan iGPU detected (shared system memory) — "
                             "capping GPU budget to %.1f MB (of %.1f MB reported free)\n",
                             usable_vram / (1024.0 * 1024.0),
                             free_vram / (1024.0 * 1024.0));
            }
            size_t min_non_kv = 512ULL * 1024ULL * 1024ULL; // Reserve headroom for staging layer + scratch
            max_kv_vram       = (usable_vram > min_non_kv) ? (usable_vram - min_non_kv) : (usable_vram / 2);
        }

        while (target_ctx >= 1) {
            if (is_gpu_ && kv_cache_.bytes_per_token() * target_ctx > max_kv_vram) {
                target_ctx /= 2;
                continue;
            }
            if (kv_cache_.allocate(target_ctx)) {
                break;
            }
            target_ctx /= 2;
        }

        if (target_ctx == 0) {
            std::fprintf(stderr, "[-] ReasoningEngine: Unable to allocate KV cache even for 1 token\n");
            return fallback_to_cpu("Vulkan KV cache allocation failed");
        }
        if (target_ctx < requested_ctx) {
            std::printf("[!] ReasoningEngine: Insufficient memory for ctx_size=%zu, halved to ctx_size=%zu (%.2f MB KV)\n",
                        requested_ctx, target_ctx, kv_cache_.total_bytes() / (1024.0 * 1024.0));
        }
        active_ctx_size_ = target_ctx;

        // Query remaining VRAM after KV allocation so WeightStore assigns resident vs streamed layers cleanly
        size_t weight_vram_budget = 0;
        if (is_gpu_) {
            size_t free_vram = 0, total_vram = 0;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
            if (dev) {
                ggml_backend_dev_memory(dev, &free_vram, &total_vram);
            }

            size_t usable_vram = static_cast<size_t>(static_cast<double>(free_vram) * util);
            if (is_vulkan_igpu) {
                size_t remaining_cap = (igpu_total_budget > kv_cache_.total_bytes())
                                           ? (igpu_total_budget - kv_cache_.total_bytes())
                                           : 0;
                usable_vram = std::min(usable_vram, remaining_cap);
            }
            size_t scratch_reserve = 256ULL * 1024ULL * 1024ULL;
            weight_vram_budget     = (usable_vram > scratch_reserve) ? (usable_vram - scratch_reserve) : (usable_vram / 2);
        }

        if (!weights_.init(gguf_filepath, gctx_, config_, layer_range_, backend_, is_gpu_, weight_vram_budget)) {
            return fallback_to_cpu("Vulkan weight allocation failed");
        }

        // Preflight check all ops on the actual model graph before committing to Vulkan
        std::string unsupported_desc;
        if (!preflight_vulkan_ops(unsupported_desc)) {
            if (backend_kind_ == GpuBackendKind::VULKAN && options_.device == DevicePreference::GPU) {
                std::fprintf(stderr, "[-] ReasoningEngine: Vulkan preflight failed (%s)\n",
                             unsupported_desc.c_str());
                cleanup();
                return false;
            }
            return fallback_to_cpu("Vulkan preflight failed (" + unsupported_desc + ")");
        }

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!gallocr_) {
            std::fprintf(stderr, "[-] ReasoningEngine: Failed to create ggml_gallocr\n");
            return fallback_to_cpu("Vulkan gallocr creation failed");
        }
    } catch (const std::exception & e) {
        return fallback_to_cpu(std::string("Vulkan init exception: ") + e.what());
    } catch (...) {
        return fallback_to_cpu("Vulkan init unknown exception");
    }

    is_initialized_ = true;
    return true;
}

void ReasoningEngine::reset_kv_cache() {
    kv_cache_.reset();
}

bool ReasoningEngine::forward_layer(int             local_layer_idx,
                                    const float *   H_in,
                                    const int32_t * pos_ids,
                                    const float *   kq_mask,
                                    size_t          n_tokens,
                                    size_t          n_past,
                                    float *         H_out) {
    if (!is_initialized_ || local_layer_idx < 0 || local_layer_idx >= layer_range_.count() ||
        !H_in || !pos_ids || !kq_mask || !H_out || n_tokens == 0) {
        return false;
    }

    const LayerWeights & lw = weights_.prepare_layer(local_layer_idx);
    const size_t layer_ctx_bytes = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
    struct ggml_context * gctx = ggml_init({ layer_ctx_bytes, nullptr, true });
    if (!gctx) return false;

    LayerGraphIO io = build_layer_graph(
        gctx, config_, lw, weights_.get_rope_freqs(), kv_cache_, local_layer_idx,
        static_cast<int>(n_tokens), static_cast<int>(n_past));

    if (!ggml_gallocr_alloc_graph(gallocr_, io.graph)) {
        std::fprintf(stderr, "[-] ReasoningEngine: ggml_gallocr_alloc_graph failed at local layer %d (global %d)\n",
                     local_layer_idx, layer_range_.start + local_layer_idx);
        ggml_free(gctx);
        return false;
    }
    peak_scratch_bytes_ = std::max(peak_scratch_bytes_, ggml_gallocr_get_buffer_size(gallocr_, 0));

    const size_t d_model = static_cast<size_t>(config_.d_model);
    const size_t n_total = n_past + n_tokens;

    try {
        ggml_backend_tensor_set(io.H_in,    H_in,    0, n_tokens * d_model * sizeof(float));
        ggml_backend_tensor_set(io.pos_ids, pos_ids, 0, n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set(io.kq_mask, kq_mask, 0, n_total * n_tokens * sizeof(float));

        if (ggml_backend_graph_compute(backend_, io.graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[-] ReasoningEngine: Graph execution failed at local layer %d (global %d)\n",
                         local_layer_idx, layer_range_.start + local_layer_idx);
            ggml_free(gctx);
            return false;
        }

        ggml_backend_tensor_get(io.H_out, H_out, 0, n_tokens * d_model * sizeof(float));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[-] ReasoningEngine: Backend exception at local layer %d (global %d): %s\n",
                     local_layer_idx, layer_range_.start + local_layer_idx, e.what());
        ggml_free(gctx);
        return false;
    } catch (...) {
        std::fprintf(stderr, "[-] ReasoningEngine: Unknown backend exception at local layer %d (global %d)\n",
                     local_layer_idx, layer_range_.start + local_layer_idx);
        ggml_free(gctx);
        return false;
    }
    ggml_free(gctx);
    return true;
}

bool ReasoningEngine::forward(const float *     H_0,
                              const int32_t *   pos_ids,
                              size_t            n_tokens,
                              ReasoningOutput & output) {
    if (!is_initialized_ || !H_0 || !pos_ids || n_tokens == 0) {
        return false;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    const size_t n_past  = kv_cache_.get_n_past();
    const size_t n_total = n_past + n_tokens;
    const size_t d_model = static_cast<size_t>(config_.d_model);
    const int    n_local = layer_range_.count();

    if (n_total > kv_cache_.get_capacity()) {
        std::fprintf(stderr, "[-] ReasoningEngine: Context limit reached (%zu > %zu tokens)\n",
                     n_total, kv_cache_.get_capacity());
        return false;
    }

    // Build causal attention mask [n_total, n_tokens]: j <= n_past + i -> 0.0f, else -INFINITY
    std::vector<float> kq_mask(n_total * n_tokens, 0.0f);
    for (size_t i = 0; i < n_tokens; ++i) {
        const size_t limit = n_past + i;
        float * row = kq_mask.data() + i * n_total;
        for (size_t j = limit + 1; j < n_total; ++j) {
            row[j] = -std::numeric_limits<float>::infinity();
        }
    }

    // Ping-pong FP32 activation buffers across owned layer slice
    std::vector<float> buf_a(n_tokens * d_model);
    std::vector<float> buf_b(n_tokens * d_model);
    std::memcpy(buf_a.data(), H_0, n_tokens * d_model * sizeof(float));

    std::vector<float> * H_curr = &buf_a;
    std::vector<float> * H_next = &buf_b;

    output.layer_l2_norms.resize(n_local);
    const size_t last_token_off = (n_tokens - 1) * d_model;

    for (int local_l = 0; local_l < n_local; ++local_l) {
        if (!forward_layer(local_l, H_curr->data(), pos_ids, kq_mask.data(),
                           n_tokens, n_past, H_next->data())) {
            return false;
        }
        output.layer_l2_norms[local_l] = compute_l2_norm(H_next->data() + last_token_off, d_model);
        std::swap(H_curr, H_next);
    }

    // Apply final output_norm only if this node owns the tail (or apply_final_norm == 1)
    if (layer_range_.should_apply_final_norm(config_.n_layers)) {
        output.H_L.resize(n_tokens * d_model);
        const size_t final_ctx_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
        struct ggml_context * gctx = ggml_init({ final_ctx_bytes, nullptr, true });
        if (!gctx) return false;

        FinalNormGraphIO f_io = build_final_norm_graph(
            gctx, config_, weights_.get_output_norm(), static_cast<int>(n_tokens));

        if (!ggml_gallocr_alloc_graph(gallocr_, f_io.graph)) {
            ggml_free(gctx);
            return false;
        }
        try {
            ggml_backend_tensor_set(f_io.H_in, H_curr->data(), 0, H_curr->size() * sizeof(float));
            if (ggml_backend_graph_compute(backend_, f_io.graph) != GGML_STATUS_SUCCESS) {
                ggml_free(gctx);
                return false;
            }
            ggml_backend_tensor_get(f_io.H_out, output.H_L.data(), 0, output.H_L.size() * sizeof(float));
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[-] ReasoningEngine: Backend exception in final RMSNorm: %s\n", e.what());
            ggml_free(gctx);
            return false;
        } catch (...) {
            std::fprintf(stderr, "[-] ReasoningEngine: Unknown backend exception in final RMSNorm\n");
            ggml_free(gctx);
            return false;
        }
        ggml_free(gctx);
    } else {
        output.H_L = std::move(*H_curr);
    }

    kv_cache_.advance(n_tokens);

    auto t_end        = std::chrono::high_resolution_clock::now();
    output.n_tokens   = n_tokens;
    output.d_model    = d_model;
    output.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return true;
}

bool ReasoningEngine::forward_packet(const ActivationPacket & in,
                                     ActivationPacket &       out,
                                     ReasoningOutput *        telemetry) {
    if (!is_initialized_ || in.final_norm_applied || in.n_tokens == 0) {
        std::fprintf(stderr, "[-] ReasoningEngine::forward_packet: invalid input packet\n");
        return false;
    }
    if (in.d_model != static_cast<size_t>(config_.d_model) ||
        in.hidden_states.size() != in.n_tokens * in.d_model ||
        in.pos_ids.size() != in.n_tokens) {
        std::fprintf(stderr, "[-] ReasoningEngine::forward_packet: packet dimension mismatch\n");
        return false;
    }
    if (in.next_layer != layer_range_.start) {
        std::fprintf(stderr, "[-] ReasoningEngine::forward_packet: expected next_layer=%d, got %d\n",
                     layer_range_.start, in.next_layer);
        return false;
    }

    ReasoningOutput r_out;
    if (!forward(in.hidden_states.data(), in.pos_ids.data(), in.n_tokens, r_out)) {
        return false;
    }

    out.pos_ids            = in.pos_ids;
    out.n_tokens           = in.n_tokens;
    out.d_model            = in.d_model;
    out.next_layer         = layer_range_.end;
    out.final_norm_applied = layer_range_.should_apply_final_norm(config_.n_layers);

    if (telemetry) {
        *telemetry = r_out;
    }
    out.hidden_states = std::move(r_out.H_L);
    return true;
}

bool ReasoningEngine::compute_logits_for_vector(const float *        hidden_vec,
                                                std::vector<float> & out_logits) const {
    const struct ggml_tensor * lm_head = weights_.get_lm_head();
    if (!is_initialized_ || !hidden_vec || !lm_head) return false;

    // Run lightweight 1-row CPU projection directly against mmap'd output.weight (0 VRAM cost)
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) return false;
    ggml_backend_cpu_set_n_threads(cpu, active_threads_ > 1 ? active_threads_ : 4);

    const size_t logits_ctx_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_context * ctx = ggml_init({ logits_ctx_bytes, nullptr, true });
    struct ggml_tensor * x    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.d_model);
    ggml_set_input(x);

    struct ggml_tensor * logits = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(lm_head), x);
    ggml_set_output(logits);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);
    if (ok) {
        ggml_backend_tensor_set(x, hidden_vec, 0, static_cast<size_t>(config_.d_model) * sizeof(float));
        ok = (ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);
        if (ok) {
            out_logits.resize(static_cast<size_t>(lm_head->ne[1]));
            ggml_backend_tensor_get(logits, out_logits.data(), 0, out_logits.size() * sizeof(float));
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return ok;
}

} // namespace reasoner
} // namespace opengp
