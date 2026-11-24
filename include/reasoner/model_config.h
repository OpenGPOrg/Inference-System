#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct gguf_context;

namespace opengp {
namespace reasoner {

enum class DevicePreference {
    AUTO, // Prefer CUDA GPU if present at runtime, fall back to multi-threaded CPU
    GPU,  // Require CUDA GPU
    CPU   // Force multi-threaded CPU execution
};

enum class KVPrecision {
    F16,
    F32
};

enum class RopeTypeOverride {
    AUTO, // Infer from GGUF general.architecture (NEOX vs NORM)
    NORM, // Interleaved complex pairs (GGML_ROPE_TYPE_NORM = 0)
    NEOX  // Split-half pairs (GGML_ROPE_TYPE_NEOX = 2)
};

// Contiguous slice of Transformer blocks owned by this engine instance [start, end)
struct LayerRange {
    int start            = 0;  // Inclusive global layer index
    int end              = -1; // Exclusive global layer index (-1 resolves to n_layers)
    int apply_final_norm = -1; // -1 = auto (true iff end == n_layers), 0 = false, 1 = true

    int count() const { return (end > start) ? (end - start) : 0; }

    bool should_apply_final_norm(int total_layers) const {
        if (apply_final_norm == 0) return false;
        if (apply_final_norm == 1) return true;
        return end == total_layers;
    }
};

// Runtime-configurable options for the Reasoning Engine (zero hardcoded model limits)
struct ReasoningOptions {
    DevicePreference device           = DevicePreference::AUTO;
    int              gpu_device_id    = 0;
    int              n_threads        = 0;     // 0 = auto-detect hardware concurrency
    KVPrecision      kv_precision     = KVPrecision::F16;
    size_t           ctx_size         = 512;   // Fixed upfront context size (halved at init if memory insufficient)
    float            vram_utilization = 0.80f; // Fraction of free VRAM usable for resident weights + KV
    RopeTypeOverride rope_type        = RopeTypeOverride::AUTO;
    LayerRange       layers;                   // Subset of layers to load & execute (for modular/distributed nodes)
    bool             verbose          = true;
};

// Model hyperparameters extracted dynamically from GGUF <arch>.* metadata
struct ModelConfig {
    std::string arch;

    int32_t n_layers       = 0;
    int32_t d_model        = 0;
    int32_t d_ff           = 0;
    int32_t n_heads        = 0;
    int32_t n_kv_heads     = 0;
    int32_t d_head_k       = 0;
    int32_t d_head_v       = 0;
    int32_t context_length = 0; // Informational metadata from GGUF (not an engine cap)

    float   rms_norm_eps   = 1e-5f;
    int32_t rope_dim       = 0;
    float   rope_freq_base = 10000.0f;
    int32_t rope_mode      = 0; // GGML_ROPE_TYPE_NORM (0) or GGML_ROPE_TYPE_NEOX (2)

    bool load_from_gguf(const struct gguf_context * gctx, const ReasoningOptions & opts);
    void print_summary() const;
};

} // namespace reasoner
} // namespace opengp
