#include "perception/perception_pipeline.h"
#include "reasoner/reasoner_engine.h"
#include "responder/responder_engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
using opengp::responder::LmHeadDevice;
using opengp::responder::RespondingEngine;
using opengp::responder::RespondingOptions;

namespace {

void print_usage(const char * prog) {
    std::cout << "Usage: " << prog << " --model <model.gguf> [options]\n\n"
              << "End-to-End OpenGP Inference (Phase 1 Perception -> Phase 2 Reasoning -> Phase 3 Responding)\n\n"
              << "Prompt / Mode:\n"
              << "  --model <path>           Path to GGUF model file (required)\n"
              << "  --prompt <text>          Single-turn prompt\n"
              << "  --system <text>          Optional system prompt\n"
              << "  --interactive, -i        Interactive terminal chat loop\n"
              << "  --max-tokens <N>         Max tokens to generate per turn (0 = EOT only, default: 256)\n\n"
              << "Execution & Memory:\n"
              << "  --device auto|gpu|cpu    Reasoning device preference (default: auto)\n"
              << "  --lm-head auto|gpu|cpu   LM Head placement (default: auto)\n"
              << "  --gpu-id <id>            CUDA device index (default: 0)\n"
              << "  --threads <N>            CPU threads (default: 0 = auto)\n"
              << "  --kv-precision f16|f32   KV cache precision (default: f16)\n"
              << "  --ctx-size <N>           Fixed context window size (default: 512; halved on init if VRAM low)\n"
              << "  --vram-util <0.1-0.95>   Max fraction of free VRAM to use (default: 0.80)\n"
              << "  --rope-type auto|norm|neox  Override RoPE layout (default: auto)\n\n"
              << "Sampling:\n"
              << "  --temp <f>               Temperature (<= 0 = greedy argmax, default: 0.7)\n"
              << "  --top-k <N>              Top-K cutoff (0 = disabled, default: 40)\n"
              << "  --top-p <f>              Top-P nucleus cutoff (default: 0.90)\n"
              << "  --min-p <f>              Min-P cutoff (default: 0.05)\n"
              << "  --repeat-penalty <f>     Repetition penalty (1.0 = off, default: 1.10)\n"
              << "  --repeat-last-n <N>      Repetition lookback window (default: 64)\n"
              << "  --seed <N>               PRNG seed (0 = random, default: 0)\n"
              << "  --help                   Show this help message\n";
}

bool run_generation_turn(PerceptionPipeline &       perception,
                         ReasoningEngine &          reasoning,
                         RespondingEngine &         responding,
                         const std::string &        prompt,
                         const std::string &        system_prompt,
                         size_t                     max_tokens) {
    // 1. Phase 1: Perception -> H_0 [n_tokens x d_model]
    PerceptionOutput p_out;
    if (!perception.process(prompt, p_out, system_prompt)) {
        std::cerr << "[-] Perception failed\n";
        return false;
    }

    // Note: ReasoningEngine starts fresh per standalone turn unless conversation history is appended
    reasoning.reset_kv_cache();
    responding.reset();

    // 2. Phase 2: Prefill prompt through Reasoning layers -> H_n
    ReasoningOutput r_out;
    if (!reasoning.forward(p_out.H_0.data(), p_out.pos_ids.data(), p_out.n_tokens, r_out)) {
        std::cerr << "[-] Reasoning prefill failed\n";
        return false;
    }

    const double prefill_ms = r_out.elapsed_ms;
    std::vector<int32_t> recent_tokens = p_out.token_ids;
    std::vector<float>   next_emb(p_out.d_model);

    const float * h_last = r_out.H_L.data() + (r_out.n_tokens - 1) * r_out.d_model;
    ReasoningOutput step_out;

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    size_t generated_tokens = 0;
    const char * stop_reason = "max_tokens";

    // 3. Phase 3: Autoregressive Decode & Live Streaming Output
    for (size_t step = 0; max_tokens == 0 || step < max_tokens; ++step) {
        int32_t next_id = responding.sample_next(h_last, recent_tokens);
        if (next_id < 0) {
            std::cerr << "\n[-] Sampling failed\n";
            return false;
        }
        if (responding.is_stop_token(next_id)) {
            stop_reason = "EOT";
            break;
        }

        std::string piece = responding.decode_token(next_id);
        std::cout << piece << std::flush;

        recent_tokens.push_back(next_id);
        generated_tokens++;

        if (max_tokens > 0 && generated_tokens >= max_tokens) {
            stop_reason = "max_tokens";
            break;
        }
        if (reasoning.get_kv_cache().get_n_past() >= reasoning.get_kv_cache().get_capacity()) {
            stop_reason = "ctx_limit";
            break;
        }

        // Feed chosen token ID back through Phase 1 (single-row EmbedTable) -> Phase 2 (Decode)
        if (!perception.embed_token(next_id, next_emb.data())) {
            std::cerr << "\n[-] Single-token embedding failed for ID " << next_id << "\n";
            return false;
        }
        const int32_t next_pos = static_cast<int32_t>(reasoning.get_kv_cache().get_n_past());
        if (!reasoning.forward(next_emb.data(), &next_pos, 1, step_out)) {
            std::cerr << "\n[-] Reasoning decode step failed\n";
            return false;
        }
        h_last = step_out.H_L.data();
    }

    std::string tail = responding.flush();
    if (!tail.empty()) {
        std::cout << tail;
    }
    std::cout << std::endl;

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    double prefill_tps = (prefill_ms > 0.0) ? (p_out.n_tokens * 1000.0 / prefill_ms) : 0.0;
    double decode_tps  = (decode_ms > 0.0) ? (generated_tokens * 1000.0 / decode_ms) : 0.0;

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "[Stats] Prefill : " << p_out.n_tokens << " tokens in "
              << std::fixed << std::setprecision(1) << prefill_ms << " ms ("
              << std::setprecision(1) << prefill_tps << " tok/s)\n";
    std::cout << "[Stats] Decode  : " << generated_tokens << " tokens in "
              << std::setprecision(1) << decode_ms << " ms ("
              << std::setprecision(2) << decode_tps << " tok/s) | Stop: " << stop_reason << "\n";
    std::cout << "[Stats] KV Cache: " << reasoning.get_kv_cache().get_n_past() << " / "
              << reasoning.get_kv_cache().get_capacity() << " tokens ("
              << std::setprecision(2) << (reasoning.get_kv_cache().total_bytes() / (1024.0 * 1024.0)) << " MB)\n";
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string       model_path;
    std::string       prompt;
    std::string       system_prompt;
    bool              interactive = false;
    size_t            max_tokens  = 256;
    ReasoningOptions  r_opts;
    RespondingOptions p3_opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };

        if      (arg == "--model")          model_path    = need_val("--model");
        else if (arg == "--prompt")         prompt        = need_val("--prompt");
        else if (arg == "--system")         system_prompt = need_val("--system");
        else if (arg == "--interactive" || arg == "-i") interactive = true;
        else if (arg == "--max-tokens")     max_tokens    = static_cast<size_t>(std::atoll(need_val("--max-tokens")));
        else if (arg == "--gpu-id") {
            int id = std::atoi(need_val("--gpu-id"));
            r_opts.gpu_device_id  = id;
            p3_opts.gpu_device_id = id;
        } else if (arg == "--threads") {
            int t = std::atoi(need_val("--threads"));
            r_opts.n_threads  = t;
            p3_opts.n_threads = t;
        }
        else if (arg == "--ctx-size")       r_opts.ctx_size         = static_cast<size_t>(std::atoll(need_val("--ctx-size")));
        else if (arg == "--vram-util")      r_opts.vram_utilization = static_cast<float>(std::atof(need_val("--vram-util")));
        else if (arg == "--temp")           p3_opts.sampler.temperature    = static_cast<float>(std::atof(need_val("--temp")));
        else if (arg == "--top-k")          p3_opts.sampler.top_k          = std::atoi(need_val("--top-k"));
        else if (arg == "--top-p")          p3_opts.sampler.top_p          = static_cast<float>(std::atof(need_val("--top-p")));
        else if (arg == "--min-p")          p3_opts.sampler.min_p          = static_cast<float>(std::atof(need_val("--min-p")));
        else if (arg == "--repeat-penalty") p3_opts.sampler.repeat_penalty = static_cast<float>(std::atof(need_val("--repeat-penalty")));
        else if (arg == "--repeat-last-n")  p3_opts.sampler.repeat_last_n  = static_cast<size_t>(std::atoll(need_val("--repeat-last-n")));
        else if (arg == "--seed")           p3_opts.sampler.seed           = static_cast<uint32_t>(std::stoul(need_val("--seed")));
        else if (arg == "--help")           { print_usage(argv[0]); return 0; }
        else if (arg == "--device") {
            std::string d = need_val("--device");
            if      (d == "auto") r_opts.device = DevicePreference::AUTO;
            else if (d == "gpu")  r_opts.device = DevicePreference::GPU;
            else if (d == "cpu")  r_opts.device = DevicePreference::CPU;
            else { std::fprintf(stderr, "Unknown --device '%s'\n", d.c_str()); return 1; }
        } else if (arg == "--lm-head") {
            std::string d = need_val("--lm-head");
            if      (d == "auto") p3_opts.lm_head_device = LmHeadDevice::AUTO;
            else if (d == "gpu")  p3_opts.lm_head_device = LmHeadDevice::GPU;
            else if (d == "cpu")  p3_opts.lm_head_device = LmHeadDevice::CPU;
            else { std::fprintf(stderr, "Unknown --lm-head '%s'\n", d.c_str()); return 1; }
        } else if (arg == "--kv-precision") {
            std::string p = need_val("--kv-precision");
            if      (p == "f16") r_opts.kv_precision = KVPrecision::F16;
            else if (p == "f32") r_opts.kv_precision = KVPrecision::F32;
            else { std::fprintf(stderr, "Unknown --kv-precision '%s'\n", p.c_str()); return 1; }
        } else if (arg == "--rope-type") {
            std::string r = need_val("--rope-type");
            if      (r == "auto") r_opts.rope_type = RopeTypeOverride::AUTO;
            else if (r == "norm") r_opts.rope_type = RopeTypeOverride::NORM;
            else if (r == "neox") r_opts.rope_type = RopeTypeOverride::NEOX;
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
    if (!interactive && prompt.empty()) {
        prompt = "What is the capital of France? Answer concisely.";
    }

    std::cout << "=================================================================\n";
    std::cout << "        OPENGP END-TO-END INFERENCE ENGINE (PHASE 1 -> 3)        \n";
    std::cout << "=================================================================\n";
    std::cout << "Model : " << model_path << "\n";

    // Phase 1
    PerceptionPipeline perception;
    if (!perception.initialize(model_path)) {
        std::cerr << "[-] Phase 1 (Perception) initialization failed\n";
        return 1;
    }

    // Phase 2
    ReasoningEngine reasoning;
    if (!reasoning.initialize(model_path, r_opts)) {
        std::cerr << "[-] Phase 2 (Reasoning) initialization failed\n";
        return 1;
    }

    // Phase 3 (initialized after Reasoning so AUTO lm_head checks actual remaining free VRAM)
    RespondingEngine responding;
    if (!responding.initialize(model_path, perception, p3_opts)) {
        std::cerr << "[-] Phase 3 (Responding) initialization failed\n";
        return 1;
    }

    reasoning.get_config().print_summary();
    const auto & emb = perception.get_embed_table();
    const auto & ws  = reasoning.get_weight_store();
    const auto & kv  = reasoning.get_kv_cache();
    const auto & ssm = reasoning.get_recurrent_cache();
    const auto & lm  = responding.get_lm_head();
    std::cout << "[Perception] Device : Host RAM mmap (0.00 MB VRAM) | ["
              << emb.get_vocab_size() << " x " << emb.get_d_model() << "] ("
              << std::fixed << std::setprecision(1) << (emb.get_table_bytes() / (1024.0 * 1024.0))
              << " MB, " << ggml_type_name(emb.get_type()) << ")\n";
    std::cout << "[Reasoning]  Device : " << reasoning.get_backend_name();
    if (reasoning.is_gpu_active()) {
        std::cout << " (" << ws.get_n_resident_layers() << " resident, "
                  << ws.get_n_streamed_layers() << " streamed | "
                  << std::setprecision(1) << (ws.get_vram_used_bytes() / (1024.0 * 1024.0)) << " MB VRAM / "
                  << (ws.get_total_weight_bytes() / (1024.0 * 1024.0)) << " MB total)";
    } else {
        std::cout << " (100% Host mmap, "
                  << std::setprecision(1) << (ws.get_total_weight_bytes() / (1024.0 * 1024.0)) << " MB)";
    }
    std::cout << "\n[KV Cache]   Memory : " << (reasoning.is_gpu_active() ? "Device VRAM" : "Host RAM")
              << " | allocated=" << std::setprecision(2) << (kv.total_bytes() / (1024.0 * 1024.0))
              << " MB (ctx=" << kv.get_capacity() << " tokens, "
              << kv.get_n_attn_layers() << " attn layers, "
              << std::setprecision(1) << (kv.bytes_per_token() / 1024.0) << " KB/tok, "
              << ggml_type_name(kv.get_dtype()) << ")\n";
    if (ssm.get_n_ssm_layers() > 0) {
        std::cout << "[SSM State]  Memory : " << (reasoning.is_gpu_active() ? "Device VRAM" : "Host RAM")
                  << " | allocated=" << std::setprecision(2) << (ssm.total_bytes() / (1024.0 * 1024.0))
                  << " MB (" << ssm.get_n_ssm_layers() << " SSM layers, fixed O(1))\n";
    }
    std::cout << "[LM Head]    Device : " << lm.get_backend_name()
              << (lm.is_tied() ? " (tied token_embd)" : "")
              << " | [" << lm.get_d_model() << " x " << lm.get_vocab_size() << "] ("
              << std::fixed << std::setprecision(1) << (lm.get_weight_bytes() / (1024.0 * 1024.0))
              << " MB)\n";
    std::cout << "[Sampler]    "
              << std::defaultfloat << std::setprecision(6)
              << "temp=" << p3_opts.sampler.temperature
              << ", top_k=" << p3_opts.sampler.top_k
              << ", top_p=" << p3_opts.sampler.top_p
              << ", min_p=" << p3_opts.sampler.min_p
              << ", repeat_penalty=" << p3_opts.sampler.repeat_penalty << "\n";
    std::cout << "=================================================================\n";

    if (interactive) {
        std::cout << "Interactive mode ready. Type /quit to exit.\n\n";
        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line == "/quit" || line == "/exit") break;
            if (line.empty()) continue;

            std::cout << "\n";
            if (!run_generation_turn(perception, reasoning, responding, line, system_prompt, max_tokens)) {
                return 1;
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "Prompt: \"" << prompt << "\"\n\n";
        // For raw completion models (no chat template), echo prompt seamlessly before continuation
        if (perception.get_config().chat_template.empty()) {
            std::cout << prompt << std::flush;
        }
        if (!run_generation_turn(perception, reasoning, responding, prompt, system_prompt, max_tokens)) {
            return 1;
        }
    }

    return 0;
}
