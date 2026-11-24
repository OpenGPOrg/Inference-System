#include "perception/perception_pipeline.h"
#include "reasoner/reasoner_engine.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using opengp::perception::PerceptionOutput;
using opengp::perception::PerceptionPipeline;
using opengp::reasoner::DevicePreference;
using opengp::reasoner::KVPrecision;
using opengp::reasoner::ReasoningEngine;
using opengp::reasoner::ReasoningOptions;
using opengp::reasoner::ReasoningOutput;
using opengp::reasoner::RopeTypeOverride;

namespace {

void print_usage(const char * prog) {
    std::cout << "Usage: " << prog << " --model <model.gguf> [options]\n\n"
              << "Options:\n"
              << "  --model <path>           Path to GGUF model file (required)\n"
              << "  --prompt <text>          User prompt string\n"
              << "  --system <text>          Optional system prompt string\n"
              << "  --device auto|gpu|cpu    Execution device preference (default: auto)\n"
              << "  --gpu-id <id>            CUDA device index (default: 0)\n"
              << "  --threads <N>            CPU worker threads (default: 0 = auto)\n"
              << "  --kv-precision f16|f32   KV cache element precision (default: f16)\n"
              << "  --ctx-size <N>           Fixed context window size (default: 512; halved on init if VRAM low)\n"
              << "  --vram-util <0.1-0.95>   Max fraction of free VRAM to use (default: 0.80)\n"
              << "  --rope-type auto|norm|neox  Override RoPE layout (default: auto)\n"
              << "  --layer-start <N>        Inclusive global start layer (default: 0)\n"
              << "  --layer-end <N>          Exclusive global end layer (default: all)\n"
              << "  --no-final-norm          Skip final RMSNorm even if end == n_layers\n"
              << "  --verify-equivalence     Compare Prefill vs sequential Decode H_L\n"
              << "  --verify-pipeline <K>    Split model across K stages and verify ActivationPacket parity\n"
              << "  --preview-logits         Preview top-5 predicted next tokens via lm_head\n"
              << "  --help                   Show this help message\n";
}

bool check_finite(const std::vector<float> & v, size_t & nan_cnt, size_t & inf_cnt) {
    nan_cnt = 0;
    inf_cnt = 0;
    for (float x : v) {
        if (std::isnan(x)) nan_cnt++;
        else if (std::isinf(x)) inf_cnt++;
    }
    return nan_cnt == 0 && inf_cnt == 0;
}

double cosine_similarity(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 0.0) ? (dot / denom) : 0.0;
}

double max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return m;
}

void print_top5_tokens(const std::string & model_path, const std::vector<float> & logits) {
    struct gguf_init_params gp = { true, nullptr };
    struct gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gp);
    if (!gctx) return;

    int64_t tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    std::vector<int32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    size_t k = std::min<size_t>(5, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    std::cout << "\n>>> Top-5 Next-Token Logits Preview (Diagnostic):\n";
    for (size_t i = 0; i < k; ++i) {
        int32_t tid     = indices[i];
        const char * s  = (tok_key >= 0 && tid < static_cast<int32_t>(gguf_get_arr_n(gctx, tok_key)))
                              ? gguf_get_arr_str(gctx, tok_key, tid)
                              : "?";
        std::cout << "  #" << (i + 1) << "  ID=" << std::setw(6) << tid
                  << "  logit=" << std::fixed << std::setprecision(3) << logits[tid]
                  << "  piece=\"" << (s ? s : "") << "\"\n";
    }
    gguf_free(gctx);
}

} // namespace

int main(int argc, char ** argv) {
    std::string      model_path;
    std::string      prompt        = "Once upon a time";
    std::string      system_prompt;
    ReasoningOptions opts;
    bool             verify_equiv    = false;
    int              verify_pipeline = 0;
    bool             preview_logits  = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--model")            model_path    = need_val("--model");
        else if (arg == "--prompt")      prompt        = need_val("--prompt");
        else if (arg == "--system")      system_prompt = need_val("--system");
        else if (arg == "--gpu-id")      opts.gpu_device_id = std::atoi(need_val("--gpu-id"));
        else if (arg == "--threads")     opts.n_threads     = std::atoi(need_val("--threads"));
        else if (arg == "--ctx-size")    opts.ctx_size      = static_cast<size_t>(std::atoll(need_val("--ctx-size")));
        else if (arg == "--vram-util")   opts.vram_utilization = static_cast<float>(std::atof(need_val("--vram-util")));
        else if (arg == "--layer-start") opts.layers.start  = std::atoi(need_val("--layer-start"));
        else if (arg == "--layer-end")   opts.layers.end    = std::atoi(need_val("--layer-end"));
        else if (arg == "--no-final-norm") opts.layers.apply_final_norm = 0;
        else if (arg == "--verify-equivalence") verify_equiv    = true;
        else if (arg == "--verify-pipeline")    verify_pipeline = std::atoi(need_val("--verify-pipeline"));
        else if (arg == "--preview-logits")     preview_logits  = true;
        else if (arg == "--help")        { print_usage(argv[0]); return 0; }
        else if (arg == "--device") {
            std::string d = need_val("--device");
            if      (d == "auto") opts.device = DevicePreference::AUTO;
            else if (d == "gpu")  opts.device = DevicePreference::GPU;
            else if (d == "cpu")  opts.device = DevicePreference::CPU;
            else { std::fprintf(stderr, "Unknown --device '%s'\n", d.c_str()); return 1; }
        } else if (arg == "--kv-precision") {
            std::string p = need_val("--kv-precision");
            if      (p == "f16") opts.kv_precision = KVPrecision::F16;
            else if (p == "f32") opts.kv_precision = KVPrecision::F32;
            else { std::fprintf(stderr, "Unknown --kv-precision '%s'\n", p.c_str()); return 1; }
        } else if (arg == "--rope-type") {
            std::string r = need_val("--rope-type");
            if      (r == "auto") opts.rope_type = RopeTypeOverride::AUTO;
            else if (r == "norm") opts.rope_type = RopeTypeOverride::NORM;
            else if (r == "neox") opts.rope_type = RopeTypeOverride::NEOX;
            else { std::fprintf(stderr, "Unknown --rope-type '%s'\n", r.c_str()); return 1; }
        } else if (model_path.empty() && arg[0] != '-') {
            model_path = arg;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=================================================================\n";
    std::cout << "          OPENGP PHASE 2: MODULAR REASONING ENGINE               \n";
    std::cout << "=================================================================\n";
    std::cout << "Model  : " << model_path << "\n";
    std::cout << "Prompt : \"" << prompt << "\"\n";

    // Phase 1: Perception -> H_0
    PerceptionPipeline perception;
    if (!perception.initialize(model_path)) {
        std::cerr << "[-] Perception initialization failed\n";
        return 1;
    }
    PerceptionOutput p_out;
    if (!perception.process(prompt, p_out, system_prompt)) {
        std::cerr << "[-] Perception processing failed\n";
        return 1;
    }
    std::cout << "[Phase 1] Perception produced H_0: [" << p_out.n_tokens << " x " << p_out.d_model << "]\n";

    // Phase 2: Initialize Reasoning Engine
    ReasoningEngine engine;
    if (!engine.initialize(model_path, opts)) {
        std::cerr << "[-] ReasoningEngine initialization failed\n";
        return 1;
    }

    engine.get_config().print_summary();
    const auto & lr = engine.get_layer_range();
    const auto & ws = engine.get_weight_store();
    std::cout << "[Node Slice]    Layers  : [" << lr.start << " .. " << lr.end << ") ("
              << lr.count() << " / " << engine.get_config().n_layers << " layers, final_norm="
              << (lr.should_apply_final_norm(engine.get_config().n_layers) ? "ON" : "OFF") << ")\n";
    std::cout << "[Backend] Active Device : " << engine.get_backend_name() << "\n";
    if (engine.is_gpu_active()) {
        std::cout << "[Weights] Placement     : " << ws.get_n_resident_layers() << " GPU-resident, "
                  << ws.get_n_streamed_layers() << " mmap-streamed (Total weights: "
                  << std::fixed << std::setprecision(1) << (ws.get_total_weight_bytes() / (1024.0 * 1024.0))
                  << " MB, VRAM used by weights: "
                  << (ws.get_vram_used_bytes() / (1024.0 * 1024.0)) << " MB)\n";
    } else {
        std::cout << "[Weights] Placement     : 100% Zero-copy Host mmap ("
                  << std::fixed << std::setprecision(1) << (ws.get_total_weight_bytes() / (1024.0 * 1024.0))
                  << " MB)\n";
    }
    const auto & init_kv = engine.get_kv_cache();
    std::cout << "[KV Cache] Memory       : " << (engine.is_gpu_active() ? "CUDA VRAM" : "Host RAM")
              << " | allocated=" << std::setprecision(2) << (init_kv.total_bytes() / (1024.0 * 1024.0))
              << " MB (ctx=" << init_kv.get_capacity() << " tokens, "
              << std::setprecision(1) << (init_kv.bytes_per_token() / 1024.0) << " KB/tok, "
              << ggml_type_name(init_kv.get_dtype()) << ")\n";

    // Run Prefill forward pass
    ReasoningOutput r_out;
    if (!engine.forward(p_out.H_0.data(), p_out.pos_ids.data(), p_out.n_tokens, r_out)) {
        std::cerr << "[-] ReasoningEngine forward pass failed\n";
        return 1;
    }

    size_t nan_cnt = 0, inf_cnt = 0;
    bool finite = check_finite(r_out.H_L, nan_cnt, inf_cnt);

    const auto & kv = engine.get_kv_cache();
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "[Reasoning] Output H_n  : [" << r_out.n_tokens << " x " << r_out.d_model << "] ("
              << (r_out.H_L.size() * sizeof(float)) << " bytes, Contiguous FP32)\n";
    std::cout << "[Reasoning] Latency     : " << std::fixed << std::setprecision(2) << r_out.elapsed_ms << " ms\n";
    std::cout << "[Reasoning] KV Cache    : n_past=" << kv.get_n_past()
              << " / cap=" << kv.get_capacity()
              << " (" << std::setprecision(2) << (kv.total_bytes() / (1024.0 * 1024.0)) << " MB, "
              << ggml_type_name(kv.get_dtype()) << ")\n";
    std::cout << "[Reasoning] Peak Scratch: " << std::setprecision(2)
              << (engine.get_scratch_bytes() / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "[Integrity] Finite Check: " << (finite ? "PASS (0 NaN, 0 Inf)" : "FAIL")
              << " (NaN=" << nan_cnt << ", Inf=" << inf_cnt << ")\n";

    std::cout << "[Layer L2]  Norms (last token): ";
    for (size_t l = 0; l < r_out.layer_l2_norms.size(); ++l) {
        if (l == 0 || l + 1 == r_out.layer_l2_norms.size() || l % std::max<size_t>(1, r_out.layer_l2_norms.size() / 6) == 0) {
            std::cout << "L" << l << "=" << std::setprecision(2) << r_out.layer_l2_norms[l] << " ";
        }
    }
    std::cout << "\n";

    std::cout << "\n>>> HIDDEN STATE SAMPLES (H_in -> H_out after layers ["
              << lr.start << ".." << lr.end << ")"
              << (lr.should_apply_final_norm(engine.get_config().n_layers) ? " + RMSNorm" : "")
              << "):\n";
    auto print_token_row = [&](size_t row_idx) {
        const float * h0_row = p_out.H_0.data() + row_idx * p_out.d_model;
        const float * hn_row = r_out.H_L.data() + row_idx * r_out.d_model;
        double h0_norm = 0.0, hn_norm = 0.0;
        for (size_t d = 0; d < r_out.d_model; ++d) {
            h0_norm += static_cast<double>(h0_row[d]) * h0_row[d];
            hn_norm += static_cast<double>(hn_row[d]) * hn_row[d];
        }
        h0_norm = std::sqrt(h0_norm);
        hn_norm = std::sqrt(hn_norm);

        std::cout << "  Row " << std::setw(2) << row_idx
                  << " (ID " << std::setw(6) << p_out.token_ids[row_idx] << "): H_n=[ "
                  << std::fixed << std::setprecision(4);
        for (size_t d = 0; d < std::min<size_t>(4, r_out.d_model); ++d) {
            std::cout << hn_row[d] << (d + 1 < std::min<size_t>(4, r_out.d_model) ? ", " : " ");
        }
        std::cout << "... ] (Norm: H_0=" << std::setprecision(3) << h0_norm
                  << " -> H_n=" << hn_norm << ")\n";
    };

    const size_t sample_count = std::min<size_t>(r_out.n_tokens, 5);
    for (size_t t = 0; t < sample_count; ++t) {
        print_token_row(t);
    }
    if (r_out.n_tokens > sample_count) {
        if (r_out.n_tokens > sample_count + 1) {
            std::cout << "  ... (" << (r_out.n_tokens - sample_count - 1) << " middle tokens omitted)\n";
        }
        print_token_row(r_out.n_tokens - 1);
    }

    if (preview_logits) {
        const float * last_h = r_out.H_L.data() + (r_out.n_tokens - 1) * r_out.d_model;
        std::vector<float> logits;
        if (engine.compute_logits_for_vector(last_h, logits)) {
            print_top5_tokens(model_path, logits);
        }
    }

    // Optional Prefill vs Incremental Decode mathematical equivalence verification
    if (verify_equiv && p_out.n_tokens > 1) {
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << ">>> VERIFYING PREFILL vs. INCREMENTAL DECODE EQUIVALENCE ("
                  << p_out.n_tokens << " steps)...\n";

        engine.reset_kv_cache();
        ReasoningOutput step_out;
        for (size_t t = 0; t < p_out.n_tokens; ++t) {
            const float *   tok_vec = p_out.H_0.data() + t * p_out.d_model;
            const int32_t * tok_pos = &p_out.pos_ids[t];
            if (!engine.forward(tok_vec, tok_pos, 1, step_out)) {
                std::cerr << "[-] Decode step " << t << " failed\n";
                return 1;
            }
        }

        const float * prefill_last = r_out.H_L.data() + (r_out.n_tokens - 1) * r_out.d_model;
        const float * decode_last  = step_out.H_L.data();
        double cos_sim = cosine_similarity(prefill_last, decode_last, r_out.d_model);
        double max_err = max_abs_diff(prefill_last, decode_last, r_out.d_model);

        std::cout << "[Equiv] Cosine Similarity : " << std::fixed << std::setprecision(7) << cos_sim << "\n";
        std::cout << "[Equiv] Max Abs Diff      : " << std::scientific << std::setprecision(3) << max_err << "\n";
    }

    // Optional Distributed Multi-Stage ActivationPacket Pipeline Verification
    if (verify_pipeline >= 2) {
        const int L = engine.get_config().n_layers;
        const int K = std::min(verify_pipeline, L);
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << ">>> VERIFYING " << K << "-STAGE MODULAR PIPELINE PARITY (Layers [0.." << L << "))...\n";

        opengp::reasoner::ActivationPacket pkt;
        pkt.hidden_states      = p_out.H_0;
        pkt.pos_ids            = p_out.pos_ids;
        pkt.n_tokens           = p_out.n_tokens;
        pkt.d_model            = p_out.d_model;
        pkt.next_layer         = 0;
        pkt.final_norm_applied = false;

        for (int s = 0; s < K; ++s) {
            int l_start = (s * L) / K;
            int l_end   = ((s + 1) * L) / K;

            ReasoningOptions stage_opts = opts;
            stage_opts.layers.start            = l_start;
            stage_opts.layers.end              = l_end;
            stage_opts.layers.apply_final_norm = -1;

            ReasoningEngine stage;
            if (!stage.initialize(model_path, stage_opts)) {
                std::cerr << "[-] Pipeline stage " << s << " init failed\n";
                return 1;
            }

            opengp::reasoner::ActivationPacket next_pkt;
            ReasoningOutput                     stage_tel;
            if (!stage.forward_packet(pkt, next_pkt, &stage_tel)) {
                std::cerr << "[-] Pipeline stage " << s << " forward_packet failed\n";
                return 1;
            }

            std::cout << "  Stage " << s << " [" << l_start << ".." << l_end << "): "
                      << "KV=" << std::fixed << std::setprecision(2)
                      << (stage.get_kv_cache().total_bytes() / (1024.0 * 1024.0)) << " MB, "
                      << "Weights=" << std::setprecision(1)
                      << (stage.get_weight_store().get_total_weight_bytes() / (1024.0 * 1024.0)) << " MB, "
                      << "final_norm=" << (next_pkt.final_norm_applied ? "ON" : "OFF") << ", "
                      << "next_layer=" << next_pkt.next_layer << "\n";

            pkt = std::move(next_pkt);
        }

        const float * mono_last = r_out.H_L.data() + (r_out.n_tokens - 1) * r_out.d_model;
        const float * pipe_last = pkt.hidden_states.data() + (pkt.n_tokens - 1) * pkt.d_model;
        double cos_sim = cosine_similarity(mono_last, pipe_last, r_out.d_model);
        double max_err = max_abs_diff(mono_last, pipe_last, r_out.d_model);

        std::cout << "[Pipeline] Cosine Similarity : " << std::fixed << std::setprecision(7) << cos_sim << "\n";
        std::cout << "[Pipeline] Max Abs Diff      : " << std::scientific << std::setprecision(3) << max_err << "\n";
    }

    std::cout << "=================================================================\n";
    return finite ? 0 : 1;
}
