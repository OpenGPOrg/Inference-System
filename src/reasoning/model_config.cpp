#include "reasoning/model_config.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

namespace opengp {
namespace reasoning {

namespace {

constexpr int32_t kRopeTypeNorm = 0;
constexpr int32_t kRopeTypeNeox = 2;

int32_t read_gguf_int(const struct gguf_context * gctx, const std::string & key, int32_t fallback) {
    int64_t idx = gguf_find_key(gctx, key.c_str());
    if (idx < 0) return fallback;

    switch (gguf_get_kv_type(gctx, idx)) {
        case GGUF_TYPE_UINT32: return static_cast<int32_t>(gguf_get_val_u32(gctx, idx));
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gctx, idx);
        case GGUF_TYPE_UINT64: return static_cast<int32_t>(gguf_get_val_u64(gctx, idx));
        case GGUF_TYPE_INT64:  return static_cast<int32_t>(gguf_get_val_i64(gctx, idx));
        case GGUF_TYPE_UINT16: return static_cast<int32_t>(gguf_get_val_u16(gctx, idx));
        case GGUF_TYPE_INT16:  return static_cast<int32_t>(gguf_get_val_i16(gctx, idx));
        default:               return fallback;
    }
}

float read_gguf_float(const struct gguf_context * gctx, const std::string & key, float fallback) {
    int64_t idx = gguf_find_key(gctx, key.c_str());
    if (idx < 0) return fallback;

    switch (gguf_get_kv_type(gctx, idx)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gctx, idx);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(gctx, idx));
        case GGUF_TYPE_UINT32:  return static_cast<float>(gguf_get_val_u32(gctx, idx));
        case GGUF_TYPE_INT32:   return static_cast<float>(gguf_get_val_i32(gctx, idx));
        default:                return fallback;
    }
}

bool arch_uses_neox_rope(const std::string & arch) {
    std::string lower = arch;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower == "qwen2" || lower == "qwen3" || lower == "qwen3moe" ||
           lower == "gemma" || lower == "gemma2" || lower == "gemma3" ||
           lower == "phi3"  || lower == "falcon" || lower == "starcoder2" ||
           lower == "olmo2" || lower == "internlm2";
}

} // namespace

bool ModelConfig::load_from_gguf(const struct gguf_context * gctx, const ReasoningOptions & opts) {
    if (!gctx) return false;

    int64_t arch_key = gguf_find_key(gctx, "general.architecture");
    if (arch_key < 0 || gguf_get_kv_type(gctx, arch_key) != GGUF_TYPE_STRING) {
        std::fprintf(stderr, "[-] ModelConfig: 'general.architecture' missing in GGUF\n");
        return false;
    }
    arch = gguf_get_val_str(gctx, arch_key);

    n_layers       = read_gguf_int(gctx, arch + ".block_count", 0);
    d_model        = read_gguf_int(gctx, arch + ".embedding_length", 0);
    d_ff           = read_gguf_int(gctx, arch + ".feed_forward_length", 0);
    n_heads        = read_gguf_int(gctx, arch + ".attention.head_count", 0);
    n_kv_heads     = read_gguf_int(gctx, arch + ".attention.head_count_kv", n_heads);
    context_length = read_gguf_int(gctx, arch + ".context_length", 0);

    if (n_layers <= 0 || d_model <= 0 || n_heads <= 0 || n_kv_heads <= 0) {
        std::fprintf(stderr, "[-] ModelConfig: Invalid core dimensions for arch '%s'\n", arch.c_str());
        return false;
    }

    int32_t default_head_dim = d_model / n_heads;
    d_head_k = read_gguf_int(gctx, arch + ".attention.key_length", default_head_dim);
    d_head_v = read_gguf_int(gctx, arch + ".attention.value_length", d_head_k);

    rms_norm_eps   = read_gguf_float(gctx, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    rope_dim       = read_gguf_int(gctx, arch + ".rope.dimension_count", d_head_k);
    rope_freq_base = read_gguf_float(gctx, arch + ".rope.freq_base", 10000.0f);

    if (opts.rope_type == RopeTypeOverride::NEOX) {
        rope_mode = kRopeTypeNeox;
    } else if (opts.rope_type == RopeTypeOverride::NORM) {
        rope_mode = kRopeTypeNorm;
    } else {
        rope_mode = arch_uses_neox_rope(arch) ? kRopeTypeNeox : kRopeTypeNorm;
    }

    return true;
}

void ModelConfig::print_summary() const {
    const char * attn_type = (n_kv_heads == n_heads) ? "MHA" : (n_kv_heads == 1 ? "MQA" : "GQA");
    std::printf("[ModelConfig] Arch=%s | Layers=%d | d_model=%d | d_ff=%d\n",
                arch.c_str(), n_layers, d_model, d_ff);
    std::printf("[ModelConfig] Attn=%s (Q_heads=%d, KV_heads=%d, d_head=%d/%d) | rms_eps=%.1e\n",
                attn_type, n_heads, n_kv_heads, d_head_k, d_head_v, rms_norm_eps);
    std::printf("[ModelConfig] RoPE: dim=%d, freq_base=%.1f, mode=%s | GGUF train_ctx=%d (uncapped)\n",
                rope_dim, rope_freq_base, (rope_mode == kRopeTypeNeox ? "NEOX" : "NORM"), context_length);
}

} // namespace reasoning
} // namespace opengp
