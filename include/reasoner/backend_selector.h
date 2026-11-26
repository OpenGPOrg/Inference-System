#pragma once

#include "ggml-backend.h"

#include <string>

namespace opengp {
namespace reasoner {

enum class GpuBackendKind {
    NONE,
    CUDA,
    METAL,
    VULKAN
};

struct GpuBackendSelection {
    ggml_backend_t backend     = nullptr;
    GpuBackendKind kind        = GpuBackendKind::NONE;
    std::string    label;       // e.g. "CUDA (...)", "Metal (...)", "Vulkan (...) [iGPU]"
    bool           is_igpu     = false;
    std::string    fail_reason; // Populated when backend == nullptr
};

// Select the highest-priority working GPU backend:
//   CUDA -> Metal -> Vulkan (skipped when CUDA/Metal succeeded or OPENGP_DISABLE_VULKAN=1).
//
// gpu_device_id < 0 selects automatically (CUDA: device 0; Vulkan: first discrete GPU,
// else first integrated GPU, skipping software renderers).
//
// For Vulkan integrated GPUs (iGPUs) in AUTO mode (allow_slow_igpu == false), runs a
// brief cached startup micro-benchmark against multi-threaded CPU (cpu_threads, 0 = auto)
// and falls back to CPU unless the iGPU is meaningfully faster. Passing allow_slow_igpu = true
// (e.g., explicit --device gpu) bypasses the speed gate while still verifying correctness.
//
// Never throws. Any Vulkan loader/driver/device/smoke-test/benchmark rejection is recorded
// in fail_reason so callers can log and seamlessly fall back to CPU.
GpuBackendSelection select_gpu_backend(int  gpu_device_id   = -1,
                                       int  cpu_threads     = 0,
                                       bool allow_slow_igpu = false);

} // namespace reasoner
} // namespace opengp
