#include "reasoner/backend_selector.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#if OPENGP_HAS_CUDA
#include "ggml-cuda.h"
#endif

#if OPENGP_HAS_METAL
#include "ggml-metal.h"
#endif

#if OPENGP_HAS_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace opengp {
namespace reasoner {

namespace {

#if OPENGP_HAS_VULKAN

bool is_env_truthy(const char * name) {
    const char * val = std::getenv(name);
    if (!val || val[0] == '\0') return false;
    std::string s(val);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(s == "0" || s == "false" || s == "no" || s == "off");
}

bool is_software_renderer_desc(const char * desc) {
    if (!desc) return false;
    std::string s(desc);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s.find("llvmpipe") != std::string::npos ||
           s.find("lavapipe") != std::string::npos ||
           s.find("swiftshader") != std::string::npos ||
           s.find("microsoft basic render") != std::string::npos ||
           s.find("software") != std::string::npos;
}

struct SmokeTestInputs {
    static constexpr int D  = 32;
    static constexpr int N  = 2;
    static constexpr int H  = 2;
    static constexpr int Dh = 16;

    std::vector<float>   x;
    std::vector<float>   w_norm;
    std::vector<uint8_t> w_q4;
    std::vector<uint8_t> w_f16;
    std::vector<int32_t> pos_ids;
    std::vector<float>   kq_mask;

    SmokeTestInputs() {
        x.resize(D * N);
        w_norm.resize(D);
        std::vector<float> w1_f32(D * D);
        std::vector<float> w2_f32(D * D);

        for (int i = 0; i < D * N; ++i) x[i] = 0.25f * std::sin(static_cast<float>(i + 1) * 0.37f);
        for (int i = 0; i < D; ++i)     w_norm[i] = 0.8f + 0.01f * static_cast<float>(i);
        for (int i = 0; i < D * D; ++i) {
            w1_f32[i] = 0.15f * std::cos(static_cast<float>(i + 1) * 0.19f);
            w2_f32[i] = 0.15f * std::sin(static_cast<float>(i + 3) * 0.23f);
        }

        w_q4.resize(ggml_row_size(GGML_TYPE_Q4_0, D) * D);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w1_f32.data(), w_q4.data(), 0, D, D, nullptr);

        w_f16.resize(ggml_row_size(GGML_TYPE_F16, D) * D);
        ggml_fp32_to_fp16_row(w2_f32.data(), reinterpret_cast<ggml_fp16_t *>(w_f16.data()), D * D);

        pos_ids = { 0, 1 };
        kq_mask = { 0.0f, -std::numeric_limits<float>::infinity(), 0.0f, 0.0f };
    }
};

bool run_smoke_graph(ggml_backend_t       backend,
                     const SmokeTestInputs & inp,
                     std::vector<float> & out_sm,
                     std::vector<float> & out_ffn,
                     std::string &        err) {
    struct ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    struct ggml_context * gctx = nullptr;
    ggml_gallocr_t        alloc = nullptr;

    auto cleanup = [&]() {
        if (alloc) ggml_gallocr_free(alloc);
        if (gctx)  ggml_free(gctx);
        if (wbuf)  ggml_backend_buffer_free(wbuf);
        if (wctx)  ggml_free(wctx);
    };

    try {
        wctx = ggml_init({ ggml_tensor_overhead() * 8 + 1024, nullptr, true });
        if (!wctx) { err = "ggml_init(weights) failed"; cleanup(); return false; }

        struct ggml_tensor * w_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32,  inp.D);
        struct ggml_tensor * w_q4   = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, inp.D, inp.D);
        struct ggml_tensor * w_f16  = ggml_new_tensor_2d(wctx, GGML_TYPE_F16,  inp.D, inp.D);
        struct ggml_tensor * kv_k   = ggml_new_tensor_3d(wctx, GGML_TYPE_F16,  inp.Dh, inp.H, inp.N);

        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbuf) { err = "weight buffer allocation failed"; cleanup(); return false; }

        ggml_backend_tensor_set(w_norm, inp.w_norm.data(), 0, ggml_nbytes(w_norm));
        ggml_backend_tensor_set(w_q4,   inp.w_q4.data(),   0, ggml_nbytes(w_q4));
        ggml_backend_tensor_set(w_f16,  inp.w_f16.data(),  0, ggml_nbytes(w_f16));

        gctx = ggml_init({ ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true });
        if (!gctx) { err = "ggml_init(graph) failed"; cleanup(); return false; }

        struct ggml_tensor * x       = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, inp.D, inp.N);
        struct ggml_tensor * pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, inp.N);
        struct ggml_tensor * kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, inp.N, inp.N);
        ggml_set_input(x);
        ggml_set_input(pos_ids);
        ggml_set_input(kq_mask);

        struct ggml_tensor * normed = ggml_mul(gctx, ggml_rms_norm(gctx, x, 1e-5f), w_norm);
        struct ggml_tensor * q      = ggml_mul_mat(gctx, w_q4,  normed);
        struct ggml_tensor * k      = ggml_mul_mat(gctx, w_f16, normed);

        struct ggml_tensor * q3    = ggml_reshape_3d(gctx, q, inp.Dh, inp.H, inp.N);
        struct ggml_tensor * k3    = ggml_reshape_3d(gctx, k, inp.Dh, inp.H, inp.N);
        struct ggml_tensor * q_rot = ggml_rope_ext(gctx, q3, pos_ids, nullptr, inp.Dh, 2, 0,
                                                   10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * k_rot = ggml_rope_ext(gctx, k3, pos_ids, nullptr, inp.Dh, 2, 0,
                                                   10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * k_cpy = ggml_cpy(gctx, k_rot, kv_k);
        struct ggml_tensor * qp    = ggml_permute(gctx, q_rot, 0, 2, 1, 3);
        struct ggml_tensor * kp    = ggml_permute(gctx, kv_k,  0, 2, 1, 3);
        struct ggml_tensor * kq    = ggml_mul_mat(gctx, kp, qp);
        struct ggml_tensor * sm    = ggml_soft_max_ext(gctx, kq, kq_mask,
                                                       1.0f / std::sqrt(static_cast<float>(inp.Dh)), 0.0f);
        struct ggml_tensor * ffn   = ggml_add(gctx, ggml_silu(gctx, q), k);

        ggml_set_output(sm);
        ggml_set_output(ffn);

        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, k_cpy);
        ggml_build_forward_expand(gf, sm);
        ggml_build_forward_expand(gf, ffn);

        // Verify backend supports all smoke-test ops before executing
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, node)) {
                err = std::string("unsupported smoke-test op ") + ggml_op_desc(node);
                cleanup();
                return false;
            }
        }

        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            err = "ggml_gallocr_alloc_graph failed";
            cleanup();
            return false;
        }

        ggml_backend_tensor_set(x,       inp.x.data(),       0, inp.x.size() * sizeof(float));
        ggml_backend_tensor_set(pos_ids, inp.pos_ids.data(), 0, inp.pos_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(kq_mask, inp.kq_mask.data(), 0, inp.kq_mask.size() * sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            err = "ggml_backend_graph_compute returned failure";
            cleanup();
            return false;
        }

        out_sm.resize(ggml_nelements(sm));
        out_ffn.resize(ggml_nelements(ffn));
        ggml_backend_tensor_get(sm,  out_sm.data(),  0, out_sm.size()  * sizeof(float));
        ggml_backend_tensor_get(ffn, out_ffn.data(), 0, out_ffn.size() * sizeof(float));

        cleanup();
        return true;
    } catch (const std::exception & e) {
        err = std::string("exception: ") + e.what();
        cleanup();
        return false;
    } catch (...) {
        err = "unknown exception";
        cleanup();
        return false;
    }
}

bool check_vec_close(const std::vector<float> & got,
                     const std::vector<float> & ref,
                     const char *               tag,
                     std::string &              err) {
    if (got.size() != ref.size()) {
        err = std::string(tag) + " size mismatch";
        return false;
    }
    float max_ref = 0.0f;
    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            err = std::string(tag) + " produced non-finite value (NaN/Inf)";
            return false;
        }
        max_ref = std::max(max_ref, std::fabs(ref[i]));
        max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
    }
    const float tol = 1e-2f * std::max(1.0f, max_ref);
    if (max_err > tol) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s numerical mismatch vs CPU (max_err=%.5f > tol=%.5f)",
                      tag, max_err, tol);
        err = buf;
        return false;
    }
    return true;
}

bool run_vulkan_smoke_test_cached(int vk_dev_idx, ggml_backend_t vk_backend, std::string & err) {
    static std::mutex mu;
    static std::unordered_map<int, std::pair<bool, std::string>> cache;

    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(vk_dev_idx);
        if (it != cache.end()) {
            err = it->second.second;
            return it->second.first;
        }
    }

    SmokeTestInputs inp;
    std::vector<float> vk_sm, vk_ffn, cpu_sm, cpu_ffn;
    std::string vk_err, cpu_err;

    bool ok = run_smoke_graph(vk_backend, inp, vk_sm, vk_ffn, vk_err);
    if (!ok) {
        err = "Vulkan smoke test failed: " + vk_err;
    } else {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) {
            err = "CPU reference init failed during Vulkan smoke test";
            ok = false;
        } else {
            ggml_backend_cpu_set_n_threads(cpu, 1);
            if (!run_smoke_graph(cpu, inp, cpu_sm, cpu_ffn, cpu_err)) {
                err = "CPU reference smoke test failed: " + cpu_err;
                ok = false;
            } else {
                ok = check_vec_close(vk_sm,  cpu_sm,  "attention/softmax", err) &&
                     check_vec_close(vk_ffn, cpu_ffn, "q4_0/f16/silu",     err);
                if (!ok) {
                    err = "Vulkan smoke test failed: " + err;
                }
            }
            ggml_backend_free(cpu);
        }
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        cache[vk_dev_idx] = { ok, err };
    }
    return ok;
}

#endif // OPENGP_HAS_VULKAN

} // namespace

GpuBackendSelection select_gpu_backend(int gpu_device_id) {
    GpuBackendSelection result;

    // 1. NVIDIA CUDA (highest priority — untouched workflow when present)
#if OPENGP_HAS_CUDA
    {
        const int cuda_id   = (gpu_device_id < 0) ? 0 : gpu_device_id;
        const int gpu_count = ggml_backend_cuda_get_device_count();
        if (gpu_count > 0 && cuda_id >= 0 && cuda_id < gpu_count) {
            ggml_backend_t backend = ggml_backend_cuda_init(cuda_id);
            if (backend) {
                char desc[256] = {};
                ggml_backend_cuda_get_device_description(cuda_id, desc, sizeof(desc));
                result.backend = backend;
                result.kind    = GpuBackendKind::CUDA;
                result.label   = std::string("CUDA (") + desc + ")";
                result.is_igpu = false;
                return result;
            }
            result.fail_reason = "CUDA device " + std::to_string(cuda_id) + " initialization failed";
        } else if (gpu_count > 0 && gpu_device_id >= gpu_count) {
            result.fail_reason = "CUDA device index " + std::to_string(gpu_device_id) +
                                 " out of range (found " + std::to_string(gpu_count) + ")";
        }
    }
#endif

    // 2. Apple Metal (macOS native GPU)
#if OPENGP_HAS_METAL
    {
        ggml_backend_t backend = ggml_backend_metal_init();
        if (backend) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            const char * desc = dev ? ggml_backend_dev_description(dev) : "Apple GPU";
            result.backend = backend;
            result.kind    = GpuBackendKind::METAL;
            result.label   = std::string("Metal (") + (desc ? desc : "Apple GPU") + ")";
            result.is_igpu = dev && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
            return result;
        }
        if (result.fail_reason.empty()) {
            result.fail_reason = "Metal device initialization failed";
        }
    }
#endif

    // 3. Vulkan (AMD / Intel / integrated GPUs — only probed when CUDA & Metal did not provide a backend)
#if OPENGP_HAS_VULKAN
    if (is_env_truthy("OPENGP_DISABLE_VULKAN")) {
        result.fail_reason = "Vulkan disabled via OPENGP_DISABLE_VULKAN";
        return result;
    }

    try {
        ggml_backend_reg_t reg = ggml_backend_vk_reg();
        if (!reg) {
            result.fail_reason = "Vulkan unavailable (loader/ICD/driver init failed or Vulkan < 1.2)";
            return result;
        }

        const size_t dev_count = ggml_backend_reg_dev_count(reg);
        if (dev_count == 0) {
            result.fail_reason = "Vulkan initialized, but no supported GPU devices were found";
            return result;
        }

        struct VkCandidate {
            int         idx;
            std::string desc;
            bool        is_igpu;
            bool        is_software;
        };
        std::vector<VkCandidate> candidates;
        candidates.reserve(dev_count);

        for (size_t i = 0; i < dev_count; ++i) {
            ggml_backend_dev_t dev  = ggml_backend_reg_dev_get(reg, i);
            const char *       desc = dev ? ggml_backend_dev_description(dev) : "Vulkan GPU";
            const auto         type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_GPU;
            const bool is_sw = (type == GGML_BACKEND_DEVICE_TYPE_CPU) || is_software_renderer_desc(desc);
            candidates.push_back({
                static_cast<int>(i),
                desc ? desc : "Vulkan GPU",
                type == GGML_BACKEND_DEVICE_TYPE_IGPU,
                is_sw
            });
        }

        int chosen_idx = -1;
        if (gpu_device_id >= 0) {
            if (static_cast<size_t>(gpu_device_id) >= candidates.size()) {
                result.fail_reason = "Vulkan device index " + std::to_string(gpu_device_id) +
                                     " out of range (found " + std::to_string(candidates.size()) + " device(s))";
                return result;
            }
            if (candidates[gpu_device_id].is_software) {
                result.fail_reason = "Vulkan device " + std::to_string(gpu_device_id) + " ('" +
                                     candidates[gpu_device_id].desc + "') is a software renderer (skipped)";
                return result;
            }
            chosen_idx = gpu_device_id;
        } else {
            // Prefer first discrete GPU, otherwise first integrated GPU; always skip software renderers
            for (const auto & c : candidates) {
                if (!c.is_software && !c.is_igpu) {
                    chosen_idx = c.idx;
                    break;
                }
            }
            if (chosen_idx < 0) {
                for (const auto & c : candidates) {
                    if (!c.is_software && c.is_igpu) {
                        chosen_idx = c.idx;
                        break;
                    }
                }
            }
            if (chosen_idx < 0) {
                result.fail_reason = "Vulkan found only software renderer(s) (skipped)";
                return result;
            }
        }

        ggml_backend_t backend = ggml_backend_vk_init(static_cast<size_t>(chosen_idx));
        if (!backend) {
            result.fail_reason = "ggml_backend_vk_init(" + std::to_string(chosen_idx) + ") returned null";
            return result;
        }

        std::string smoke_err;
        if (!run_vulkan_smoke_test_cached(chosen_idx, backend, smoke_err)) {
            ggml_backend_free(backend);
            result.fail_reason = smoke_err;
            return result;
        }

        const auto & chosen = candidates[chosen_idx];
        result.backend = backend;
        result.kind    = GpuBackendKind::VULKAN;
        result.is_igpu = chosen.is_igpu;
        result.label   = std::string("Vulkan (") + chosen.desc + ")" + (chosen.is_igpu ? " [iGPU]" : "");
        return result;
    } catch (const std::exception & e) {
        result.fail_reason = std::string("Vulkan initialization threw exception: ") + e.what();
        return result;
    } catch (...) {
        result.fail_reason = "Vulkan initialization threw unknown exception";
        return result;
    }
#endif

    return result;
}

} // namespace reasoner
} // namespace opengp
