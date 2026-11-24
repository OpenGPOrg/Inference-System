#include "reasoner/layer_graph.h"

#include <cmath>

namespace opengp {
namespace reasoner {

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

// Common Post-Attention/Post-SSM RMSNorm + SwiGLU FFN + Residual 2
struct ggml_tensor * build_swiglu_residual(struct ggml_context * ctx,
                                           struct ggml_tensor *  H_mid,
                                           const LayerWeights &  w,
                                           float                 eps) {
    struct ggml_tensor * ffn_in = rms_norm_weighted(ctx, H_mid, w.ffn_norm, eps);
    struct ggml_tensor * gate   = ggml_silu(ctx, ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.w_gate), ffn_in));
    struct ggml_tensor * up     = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.w_up), ffn_in);
    struct ggml_tensor * ffn_out = ggml_mul_mat(
        ctx, const_cast<struct ggml_tensor *>(w.w_down), ggml_mul(ctx, gate, up));
    return ggml_add(ctx, H_mid, ffn_out);
}

struct ggml_tensor * build_attention_block(struct ggml_context *      ctx,
                                           struct ggml_cgraph *       graph,
                                           const ModelConfig &        cfg,
                                           const LayerWeights &       w,
                                           const struct ggml_tensor * rope_freqs,
                                           KVCache &                  kv,
                                           struct ggml_tensor *       H_in,
                                           struct ggml_tensor *       pos_ids,
                                           struct ggml_tensor *       kq_mask,
                                           int                        layer_idx,
                                           int                        n_tokens,
                                           int                        n_past) {
    const int n_total = n_past + n_tokens;

    // 1. Pre-Attention RMSNorm
    struct ggml_tensor * cur = rms_norm_weighted(ctx, H_in, w.attn_norm, cfg.rms_norm_eps);

    // 2. Q (plus optional fused output gate), K, V projections
    struct ggml_tensor * Q    = nullptr;
    struct ggml_tensor * gate = nullptr;

    if (w.kind == LayerKind::GATED_ATTENTION) {
        struct ggml_tensor * Q_full = linear_proj(ctx, cur, w.wq, w.bq);
        struct ggml_tensor * Q_3d   = ggml_reshape_3d(ctx, Q_full, 2 * cfg.d_head_k, cfg.n_heads, n_tokens);
        Q = ggml_cont(ctx, ggml_view_3d(ctx, Q_3d, cfg.d_head_k, cfg.n_heads, n_tokens,
                                        Q_3d->nb[1], Q_3d->nb[2], 0));
        struct ggml_tensor * G_3d = ggml_cont(ctx, ggml_view_3d(ctx, Q_3d, cfg.d_head_k, cfg.n_heads, n_tokens,
                                        Q_3d->nb[1], Q_3d->nb[2], static_cast<size_t>(cfg.d_head_k) * sizeof(float)));
        gate = ggml_reshape_2d(ctx, G_3d, static_cast<int64_t>(cfg.d_head_k) * cfg.n_heads, n_tokens);
    } else {
        Q = linear_proj(ctx, cur, w.wq, w.bq);
        Q = ggml_reshape_3d(ctx, Q, cfg.d_head_k, cfg.n_heads, n_tokens);
    }

    struct ggml_tensor * K = ggml_reshape_3d(ctx, linear_proj(ctx, cur, w.wk, w.bk), cfg.d_head_k, cfg.n_kv_heads, n_tokens);
    struct ggml_tensor * V = linear_proj(ctx, cur, w.wv, w.bv);

    // 3. Optional per-head QK-Norm
    if (w.attn_q_norm) {
        Q = rms_norm_weighted(ctx, Q, w.attn_q_norm, cfg.rms_norm_eps);
    }
    if (w.attn_k_norm) {
        K = rms_norm_weighted(ctx, K, w.attn_k_norm, cfg.rms_norm_eps);
    }

    // 4. Rotary Position Embedding on Q and K (partial rope_dim supported natively)
    struct ggml_tensor * rf = const_cast<struct ggml_tensor *>(rope_freqs);
    Q = ggml_rope_ext(ctx, Q, pos_ids, rf, cfg.rope_dim, cfg.rope_mode, 0,
                      cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos_ids, rf, cfg.rope_dim, cfg.rope_mode, 0,
                      cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // 5. Append K and V into persistent KV Cache
    struct ggml_tensor * K_flat = ggml_reshape_2d(
        ctx, K, static_cast<int64_t>(cfg.d_head_k) * cfg.n_kv_heads, n_tokens);
    ggml_build_forward_expand(
        graph, ggml_cpy(ctx, K_flat, kv.write_view_k(ctx, layer_idx, n_past, n_tokens)));
    ggml_build_forward_expand(
        graph, ggml_cpy(ctx, V, kv.write_view_v(ctx, layer_idx, n_past, n_tokens)));

    // 6. Scaled Dot-Product Causal Attention (unified MHA / GQA / MQA)
    struct ggml_tensor * Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3);
    struct ggml_tensor * K_all  = ggml_permute(
        ctx, kv.read_view_k(ctx, layer_idx, n_total), 0, 2, 1, 3);

    struct ggml_tensor * KQ      = ggml_mul_mat(ctx, K_all, Q_perm);
    const float          scale   = 1.0f / std::sqrt(static_cast<float>(cfg.d_head_k));
    struct ggml_tensor * KQ_soft = ggml_soft_max_ext(ctx, KQ, kq_mask, scale, 0.0f);

    struct ggml_tensor * V_all = ggml_cont(
        ctx, ggml_permute(ctx, kv.read_view_v(ctx, layer_idx, n_total), 1, 2, 0, 3));
    struct ggml_tensor * KQV        = ggml_mul_mat(ctx, V_all, KQ_soft);
    struct ggml_tensor * KQV_merged = ggml_cont_2d(
        ctx,
        ggml_permute(ctx, KQV, 0, 2, 1, 3),
        static_cast<int64_t>(cfg.d_head_v) * cfg.n_heads,
        n_tokens);

    // 7. Optional Attention Output Gate (sigmoid(gate) * attn_out before Wo)
    if (gate) {
        KQV_merged = ggml_mul(ctx, KQV_merged, ggml_sigmoid(ctx, gate));
    }

    // 8. Output projection + Residual 1
    struct ggml_tensor * attn_out = linear_proj(ctx, KQV_merged, w.wo, w.bo);
    return ggml_add(ctx, H_in, attn_out);
}

struct ggml_tensor * build_gated_deltanet_block(struct ggml_context * ctx,
                                                struct ggml_cgraph *  graph,
                                                const ModelConfig &   cfg,
                                                const LayerWeights &  w,
                                                RecurrentStateCache * recurrent_cache,
                                                struct ggml_tensor *  H_in,
                                                int                   layer_idx,
                                                int                   n_tokens) {
    const int64_t S_v      = cfg.ssm_state_size;
    const int64_t n_groups = cfg.ssm_group_count;
    const int64_t H_v      = cfg.ssm_time_step_rank;
    const int64_t qk_dim   = n_groups * S_v;
    const int64_t v_dim    = cfg.ssm_inner_size;
    const int64_t qkv_dim  = 2 * qk_dim + v_dim;
    const int64_t d_conv   = cfg.ssm_conv_kernel;

    // 1. Pre-SSM RMSNorm
    struct ggml_tensor * cur = rms_norm_weighted(ctx, H_in, w.attn_norm, cfg.rms_norm_eps);

    // 2. Fused QKV & Output Gate Z projections
    struct ggml_tensor * qkv    = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.attn_qkv), cur);
    struct ggml_tensor * gate_z = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.attn_gate), cur);

    // 3. Causal Conv1D with persistent rolling state [d_conv - 1, qkv_dim]
    struct ggml_tensor * conv_state = recurrent_cache->get_conv_state(layer_idx);
    struct ggml_tensor * qkv_t      = ggml_cont(ctx, ggml_transpose(ctx, qkv)); // [n_tokens, qkv_dim]
    struct ggml_tensor * conv_in    = ggml_concat(ctx, conv_state, qkv_t, 0);  // [d_conv - 1 + n_tokens, qkv_dim]

    struct ggml_tensor * new_conv_state = ggml_view_2d(
        ctx, conv_in, d_conv - 1, qkv_dim, conv_in->nb[1], static_cast<size_t>(n_tokens) * sizeof(float));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, new_conv_state, conv_state));

    struct ggml_tensor * conv_in_3d = ggml_reshape_3d(ctx, conv_in, d_conv - 1 + n_tokens, qkv_dim, 1);
    struct ggml_tensor * qkv_conv   = ggml_silu(
        ctx, ggml_ssm_conv(ctx, conv_in_3d, const_cast<struct ggml_tensor *>(w.ssm_conv1d))); // [qkv_dim, n_tokens, 1]
    struct ggml_tensor * qkv_2d = ggml_reshape_2d(ctx, qkv_conv, qkv_dim, n_tokens);

    // 4. Split Q, K, V and L2-normalize Q and K (kernel internally multiplies output by 1 / sqrt(S_v))
    struct ggml_tensor * Q_raw = ggml_cont(
        ctx, ggml_view_2d(ctx, qkv_2d, qk_dim, n_tokens, qkv_2d->nb[1], 0));
    struct ggml_tensor * K_raw = ggml_cont(
        ctx, ggml_view_2d(ctx, qkv_2d, qk_dim, n_tokens, qkv_2d->nb[1], static_cast<size_t>(qk_dim) * sizeof(float)));
    struct ggml_tensor * V_raw = ggml_cont(
        ctx, ggml_view_2d(ctx, qkv_2d, v_dim, n_tokens, qkv_2d->nb[1], static_cast<size_t>(2 * qk_dim) * sizeof(float)));

    struct ggml_tensor * Q = ggml_l2_norm(ctx, ggml_reshape_4d(ctx, Q_raw, S_v, n_groups, n_tokens, 1), cfg.rms_norm_eps);
    struct ggml_tensor * K = ggml_l2_norm(ctx, ggml_reshape_4d(ctx, K_raw, S_v, n_groups, n_tokens, 1), cfg.rms_norm_eps);
    struct ggml_tensor * V = ggml_reshape_4d(ctx, V_raw, S_v, H_v, n_tokens, 1);

    // 5. Log decay `g` and delta update rate `beta`
    struct ggml_tensor * alpha = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.ssm_alpha), cur);
    struct ggml_tensor * g     = ggml_mul(
        ctx,
        ggml_softplus(ctx, ggml_add(ctx, alpha, const_cast<struct ggml_tensor *>(w.ssm_dt_bias))),
        const_cast<struct ggml_tensor *>(w.ssm_a));
    g = ggml_reshape_4d(ctx, g, 1, H_v, n_tokens, 1);

    struct ggml_tensor * beta = ggml_sigmoid(
        ctx, ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.ssm_beta), cur));
    beta = ggml_reshape_4d(ctx, beta, 1, H_v, n_tokens, 1);

    // 6. Gated DeltaNet recurrent kernel + persistent state write-back
    struct ggml_tensor * ssm_state = recurrent_cache->get_ssm_state(layer_idx);
    struct ggml_tensor * gdn       = ggml_gated_delta_net(ctx, Q, K, V, g, beta, ssm_state, 1);

    struct ggml_tensor * gdn_out   = ggml_view_2d(ctx, gdn, v_dim, n_tokens, gdn->nb[1], 0);
    struct ggml_tensor * new_state = ggml_view_4d(
        ctx, gdn, S_v, S_v, H_v, 1,
        static_cast<size_t>(S_v) * sizeof(float),
        static_cast<size_t>(S_v * S_v) * sizeof(float),
        static_cast<size_t>(S_v * S_v * H_v) * sizeof(float),
        static_cast<size_t>(n_tokens) * gdn->nb[1]);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, new_state, ssm_state));

    // 7. Per-head RMSNorm * SiLU(gate_z) -> ssm_out projection + Residual 1
    struct ggml_tensor * gdn_3d   = ggml_reshape_3d(ctx, ggml_cont(ctx, gdn_out), S_v, H_v, n_tokens);
    struct ggml_tensor * gdn_norm = ggml_reshape_2d(
        ctx, rms_norm_weighted(ctx, gdn_3d, w.ssm_norm, cfg.rms_norm_eps), v_dim, n_tokens);
    struct ggml_tensor * gated    = ggml_mul(ctx, gdn_norm, ggml_silu(ctx, gate_z));
    struct ggml_tensor * ssm_proj = ggml_mul_mat(ctx, const_cast<struct ggml_tensor *>(w.ssm_out), gated);

    return ggml_add(ctx, H_in, ssm_proj);
}

} // namespace

LayerGraphIO build_layer_graph(struct ggml_context *      ctx,
                               const ModelConfig &        cfg,
                               const LayerWeights &       w,
                               const struct ggml_tensor * rope_freqs,
                               KVCache &                  kv,
                               RecurrentStateCache *      recurrent_cache,
                               int                        layer_idx,
                               int                        n_tokens,
                               int                        n_past) {
    LayerGraphIO io;
    const int n_total = n_past + n_tokens;

    io.H_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.d_model, n_tokens);
    ggml_set_name(io.H_in, "H_in");
    ggml_set_input(io.H_in);

    io.graph = ggml_new_graph(ctx);

    struct ggml_tensor * H_mid = nullptr;
    if (w.kind == LayerKind::GATED_DELTANET_SSM) {
        H_mid = build_gated_deltanet_block(ctx, io.graph, cfg, w, recurrent_cache, io.H_in, layer_idx, n_tokens);
    } else {
        io.pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(io.pos_ids, "pos_ids");
        ggml_set_input(io.pos_ids);

        io.kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_total, n_tokens);
        ggml_set_name(io.kq_mask, "kq_mask");
        ggml_set_input(io.kq_mask);

        H_mid = build_attention_block(ctx, io.graph, cfg, w, rope_freqs, kv,
                                      io.H_in, io.pos_ids, io.kq_mask, layer_idx, n_tokens, n_past);
    }

    io.H_out = build_swiglu_residual(ctx, H_mid, w, cfg.rms_norm_eps);
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

} // namespace reasoner
} // namespace opengp
