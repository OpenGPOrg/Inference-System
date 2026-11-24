#include "reasoning/layer_graph.h"

#include <cmath>

namespace opengp {
namespace reasoning {

namespace {

struct ggml_tensor * rms_norm_weighted(struct ggml_context *      ctx,
                                       struct ggml_tensor *       x,
                                       const struct ggml_tensor * weight,
                                       float                      eps) {
    struct ggml_tensor * norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, const_cast<struct ggml_tensor *>(weight));
}

struct ggml_tensor * linear_proj(struct ggml_context *      ctx,
                                 struct ggml_tensor *       x,
                                 const struct ggml_tensor * w,
                                 const struct ggml_tensor * bias) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w), x);
    if (bias) {
        y = ggml_add(ctx, y, const_cast<struct ggml_tensor *>(bias));
    }
    return y;
}

} // namespace

LayerGraphIO build_layer_graph(struct ggml_context *      ctx,
                               const ModelConfig &        cfg,
                               const LayerWeights &       w,
                               const struct ggml_tensor * rope_freqs,
                               KVCache &                  kv,
                               int                        layer_idx,
                               int                        n_tokens,
                               int                        n_past) {
    LayerGraphIO io;
    const int n_total = n_past + n_tokens;

    io.H_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.d_model, n_tokens);
    ggml_set_name(io.H_in, "H_in");
    ggml_set_input(io.H_in);

    io.pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(io.pos_ids, "pos_ids");
    ggml_set_input(io.pos_ids);

    io.kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_total, n_tokens);
    ggml_set_name(io.kq_mask, "kq_mask");
    ggml_set_input(io.kq_mask);

    io.graph = ggml_new_graph(ctx);

    // 1. Pre-Attention RMSNorm
    struct ggml_tensor * cur = rms_norm_weighted(ctx, io.H_in, w.attn_norm, cfg.rms_norm_eps);

    // 2. Q, K, V linear projections
    struct ggml_tensor * Q = linear_proj(ctx, cur, w.wq, w.bq);
    struct ggml_tensor * K = linear_proj(ctx, cur, w.wk, w.bk);
    struct ggml_tensor * V = linear_proj(ctx, cur, w.wv, w.bv);

    // 3. Head slicing + optional per-head QK-Norm (e.g. Qwen3)
    Q = ggml_reshape_3d(ctx, Q, cfg.d_head_k, cfg.n_heads, n_tokens);
    K = ggml_reshape_3d(ctx, K, cfg.d_head_k, cfg.n_kv_heads, n_tokens);

    if (w.attn_q_norm) {
        Q = rms_norm_weighted(ctx, Q, w.attn_q_norm, cfg.rms_norm_eps);
    }
    if (w.attn_k_norm) {
        K = rms_norm_weighted(ctx, K, w.attn_k_norm, cfg.rms_norm_eps);
    }

    // 4. Rotary Position Embedding on Q and K (V stays unrotated as pure semantic payload)
    struct ggml_tensor * rf = const_cast<struct ggml_tensor *>(rope_freqs);
    Q = ggml_rope_ext(ctx, Q, io.pos_ids, rf, cfg.rope_dim, cfg.rope_mode, 0,
                      cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, io.pos_ids, rf, cfg.rope_dim, cfg.rope_mode, 0,
                      cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // 5. Append K and V into persistent KV Cache
    struct ggml_tensor * K_flat = ggml_reshape_2d(
        ctx, K, static_cast<int64_t>(cfg.d_head_k) * cfg.n_kv_heads, n_tokens);
    ggml_build_forward_expand(
        io.graph, ggml_cpy(ctx, K_flat, kv.write_view_k(ctx, layer_idx, n_past, n_tokens)));
    ggml_build_forward_expand(
        io.graph, ggml_cpy(ctx, V, kv.write_view_v(ctx, layer_idx, n_past, n_tokens)));

    // 6. Scaled Dot-Product Causal Attention (unified MHA / GQA / MQA)
    struct ggml_tensor * Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3); // [d_head_k, n_tokens, n_heads]
    struct ggml_tensor * K_all  = ggml_permute(
        ctx, kv.read_view_k(ctx, layer_idx, n_total), 0, 2, 1, 3); // [d_head_k, n_total, n_kv_heads]

    struct ggml_tensor * KQ      = ggml_mul_mat(ctx, K_all, Q_perm); // [n_total, n_tokens, n_heads]
    const float          scale   = 1.0f / std::sqrt(static_cast<float>(cfg.d_head_k));
    struct ggml_tensor * KQ_soft = ggml_soft_max_ext(ctx, KQ, io.kq_mask, scale, 0.0f);

    struct ggml_tensor * V_all = ggml_cont(
        ctx, ggml_permute(ctx, kv.read_view_v(ctx, layer_idx, n_total), 1, 2, 0, 3)); // [n_total, d_head_v, n_kv_heads]
    struct ggml_tensor * KQV        = ggml_mul_mat(ctx, V_all, KQ_soft);               // [d_head_v, n_tokens, n_heads]
    struct ggml_tensor * KQV_merged = ggml_cont_2d(
        ctx,
        ggml_permute(ctx, KQV, 0, 2, 1, 3),
        static_cast<int64_t>(cfg.d_head_v) * cfg.n_heads,
        n_tokens);

    // 7. Output projection + Residual 1
    struct ggml_tensor * attn_out = linear_proj(ctx, KQV_merged, w.wo, w.bo);
    struct ggml_tensor * H_mid    = ggml_add(ctx, io.H_in, attn_out);

    // 8. Post-Attention RMSNorm + SwiGLU FFN
    struct ggml_tensor * ffn_in = rms_norm_weighted(ctx, H_mid, w.ffn_norm, cfg.rms_norm_eps);
    struct ggml_tensor * gate   = ggml_silu(ctx, ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.w_gate), ffn_in));
    struct ggml_tensor * up     = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.w_up), ffn_in);
    struct ggml_tensor * ffn_out = ggml_mul_mat(
        ctx, const_cast<struct ggml_tensor *>(w.w_down), ggml_mul(ctx, gate, up));

    // 9. Residual 2
    io.H_out = ggml_add(ctx, H_mid, ffn_out);
    ggml_set_name(io.H_out, "H_out");
    ggml_set_output(io.H_out);
    ggml_build_forward_expand(io.graph, io.H_out);

    return io;
}

FinalNormGraphIO build_final_norm_graph(struct ggml_context *      ctx,
                                        const ModelConfig &        cfg,
                                        const struct ggml_tensor * output_norm,
                                        int                        n_tokens) {
    FinalNormGraphIO io;
    io.H_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.d_model, n_tokens);
    ggml_set_name(io.H_in, "H_final_in");
    ggml_set_input(io.H_in);

    io.H_out = rms_norm_weighted(ctx, io.H_in, output_norm, cfg.rms_norm_eps);
    ggml_set_name(io.H_out, "H_final_out");
    ggml_set_output(io.H_out);

    io.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(io.graph, io.H_out);
    return io;
}

} // namespace reasoning
} // namespace opengp
