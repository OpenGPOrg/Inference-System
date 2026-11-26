#include "perception/perception_pipeline.h"
#include "reasoner/reasoner_engine.h"
#include "responder/responder_engine.h"
#include "server/http_server.h"
#include "server/mini_json.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using opengp::perception::ChatMessage;
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
using opengp::responder::SamplerConfig;
using opengp::server::HttpConnection;
using opengp::server::HttpRequest;
using opengp::server::HttpServer;
using opengp::server::JsonValue;

namespace {

std::atomic<bool> g_stop_server{false};

void signal_handler(int) {
    g_stop_server.store(true);
}

std::string basename_of(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

int64_t unix_now() {
    return static_cast<int64_t>(std::time(nullptr));
}

std::string make_error_json(const std::string & message, const std::string & type = "invalid_request_error") {
    JsonValue err = JsonValue::object();
    err.set("message", JsonValue::from_string(message));
    err.set("type", JsonValue::from_string(type));
    JsonValue root = JsonValue::object();
    root.set("error", std::move(err));
    return root.dump();
}

std::string make_chunk_json(const std::string & id,
                            int64_t             created,
                            const std::string & model_id,
                            const std::string & delta_role,
                            const std::string & delta_content,
                            const std::string & finish_reason) {
    JsonValue delta = JsonValue::object();
    if (!delta_role.empty())    delta.set("role", JsonValue::from_string(delta_role));
    if (!delta_content.empty()) delta.set("content", JsonValue::from_string(delta_content));

    JsonValue choice = JsonValue::object();
    choice.set("index", JsonValue::from_int(0));
    choice.set("delta", std::move(delta));
    choice.set("finish_reason", finish_reason.empty() ? JsonValue::null() : JsonValue::from_string(finish_reason));

    JsonValue choices = JsonValue::array();
    choices.push_back(std::move(choice));

    JsonValue root = JsonValue::object();
    root.set("id", JsonValue::from_string(id));
    root.set("object", JsonValue::from_string("chat.completion.chunk"));
    root.set("created", JsonValue::from_int(created));
    root.set("model", JsonValue::from_string(model_id));
    root.set("choices", std::move(choices));
    return root.dump();
}

std::string make_completion_json(const std::string & id,
                                 int64_t             created,
                                 const std::string & model_id,
                                 const std::string & content,
                                 const std::string & finish_reason,
                                 int64_t             prompt_tokens,
                                 int64_t             completion_tokens) {
    JsonValue msg = JsonValue::object();
    msg.set("role", JsonValue::from_string("assistant"));
    msg.set("content", JsonValue::from_string(content));

    JsonValue choice = JsonValue::object();
    choice.set("index", JsonValue::from_int(0));
    choice.set("message", std::move(msg));
    choice.set("finish_reason", JsonValue::from_string(finish_reason));

    JsonValue choices = JsonValue::array();
    choices.push_back(std::move(choice));

    JsonValue usage = JsonValue::object();
    usage.set("prompt_tokens", JsonValue::from_int(prompt_tokens));
    usage.set("completion_tokens", JsonValue::from_int(completion_tokens));
    usage.set("total_tokens", JsonValue::from_int(prompt_tokens + completion_tokens));

    JsonValue root = JsonValue::object();
    root.set("id", JsonValue::from_string(id));
    root.set("object", JsonValue::from_string("chat.completion"));
    root.set("created", JsonValue::from_int(created));
    root.set("model", JsonValue::from_string(model_id));
    root.set("choices", std::move(choices));
    root.set("usage", std::move(usage));
    return root.dump();
}

bool extract_messages(const JsonValue & body, std::vector<ChatMessage> & out, std::string & err) {
    const JsonValue * msgs = body.find("messages");
    if (!msgs || !msgs->is_array() || msgs->as_array().empty()) {
        err = "'messages' must be a non-empty array";
        return false;
    }

    out.clear();
    for (const JsonValue & item : msgs->as_array()) {
        if (!item.is_object()) {
            err = "Each item in 'messages' must be an object";
            return false;
        }
        const JsonValue * role_v = item.find("role");
        const JsonValue * cont_v = item.find("content");
        if (!role_v || !role_v->is_string()) {
            err = "Each message must include string 'role'";
            return false;
        }
        std::string content;
        if (cont_v && cont_v->is_string()) {
            content = cont_v->as_string();
        } else if (cont_v && cont_v->is_array()) {
            for (const JsonValue & part : cont_v->as_array()) {
                if (part.is_object()) {
                    const JsonValue * txt = part.find("text");
                    if (txt && txt->is_string()) content += txt->as_string();
                }
            }
        } else if (cont_v && !cont_v->is_null()) {
            err = "Message 'content' must be a string, null, or text-part array";
            return false;
        }
        out.push_back({ role_v->as_string(), std::move(content) });
    }
    return true;
}

void print_usage(const char * prog) {
    std::cout << "Usage: " << prog << " --model <model.gguf> [options]\n\n"
              << "OpenGP OpenAI-Compatible Inference Server (/v1/chat/completions, /v1/models)\n\n"
              << "Server:\n"
              << "  --model <path>           Path to GGUF model file (required)\n"
              << "  --host <ip>              Bind address (default: 0.0.0.0)\n"
              << "  --port <N>               Listen port (default: 8080)\n"
              << "  --model-name <id>        Model ID advertised in /v1/models (default: filename)\n"
              << "  --max-tokens <N>         Default max completion tokens (default: 512)\n\n"
              << "Execution & Memory:\n"
              << "  --device auto|gpu|cpu    Reasoning device preference (default: auto)\n"
              << "  --lm-head auto|gpu|cpu   LM Head placement (default: auto)\n"
              << "  --gpu-id <id>            GPU device index (default: auto)\n"
              << "  --threads <N>            CPU threads (default: 0 = auto)\n"
              << "  --kv-precision f16|f32   KV cache precision (default: f16)\n"
              << "  --ctx-size <N>           Context window size (default: 4096)\n"
              << "  --vram-util <0.1-0.95>   Max fraction of free VRAM (default: 0.80)\n"
              << "  --rope-type auto|norm|neox  Override RoPE layout (default: auto)\n\n"
              << "Default Sampling:\n"
              << "  --temp <f>               Temperature (default: 0.7)\n"
              << "  --top-k <N>              Top-K cutoff (default: 40)\n"
              << "  --top-p <f>              Top-P nucleus cutoff (default: 0.90)\n"
              << "  --min-p <f>              Min-P cutoff (default: 0.05)\n"
              << "  --repeat-penalty <f>     Repetition penalty (default: 1.10)\n"
              << "  --repeat-last-n <N>      Repetition lookback window (default: 64)\n"
              << "  --seed <N>               PRNG seed (default: 0 = random)\n";
}

} // namespace

int main(int argc, char ** argv) {
    std::string       model_path;
    std::string       model_name;
    std::string       host        = "0.0.0.0";
    int               port        = 8080;
    size_t            default_max = 512;
    ReasoningOptions  r_opts;
    r_opts.ctx_size               = 4096;
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

        if      (arg == "--model")          model_path = need_val("--model");
        else if (arg == "--model-name")     model_name = need_val("--model-name");
        else if (arg == "--host")           host       = need_val("--host");
        else if (arg == "--port")           port       = std::atoi(need_val("--port"));
        else if (arg == "--max-tokens")     default_max = static_cast<size_t>(std::atoll(need_val("--max-tokens")));
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
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
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
    if (model_name.empty()) {
        model_name = basename_of(model_path);
    }

    std::cout << "=================================================================\n";
    std::cout << "         OPENGP OPENAI-COMPATIBLE SERVER (PHASE 1 -> 3)          \n";
    std::cout << "=================================================================\n";
    std::cout << "Model : " << model_path << " (id: " << model_name << ")\n";

    PerceptionPipeline perception;
    if (!perception.initialize(model_path)) {
        std::cerr << "[-] Phase 1 (Perception) initialization failed\n";
        return 1;
    }

    ReasoningEngine reasoning;
    if (!reasoning.initialize(model_path, r_opts)) {
        std::cerr << "[-] Phase 2 (Reasoning) initialization failed\n";
        return 1;
    }

    RespondingEngine responding;
    if (!responding.initialize(model_path, perception, p3_opts)) {
        std::cerr << "[-] Phase 3 (Responding) initialization failed\n";
        return 1;
    }

    reasoning.get_config().print_summary();
    const auto & ws = reasoning.get_weight_store();
    const auto & kv = reasoning.get_kv_cache();
    std::cout << "[Reasoning] Device : " << reasoning.get_backend_name()
              << " (" << ws.get_n_resident_layers() << " resident, "
              << ws.get_n_streamed_layers() << " streamed)"
              << " | KV=" << std::fixed << std::setprecision(2) << (kv.total_bytes() / (1024.0 * 1024.0))
              << " MB (ctx=" << reasoning.get_ctx_size() << ")\n";
    std::cout << "[LM Head]   Device : " << responding.get_lm_head().get_backend_name() << "\n";

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    const SamplerConfig default_sampler = p3_opts.sampler;
    std::vector<int32_t> cached_tokens;
    uint64_t             req_seq = 0;

    HttpServer server(host, port);
    std::cout << "[Server] Listening at http://" << host << ":" << port << "/v1\n";
    std::cout << "  - GET  /health\n";
    std::cout << "  - GET  /v1/models\n";
    std::cout << "  - POST /v1/chat/completions\n";
    std::cout << "=================================================================\n" << std::flush;

    auto handle_request = [&](const HttpRequest & req, HttpConnection & conn) {
        if (req.method == "OPTIONS") {
            conn.send_cors_preflight();
            return;
        }

        if (req.method == "GET" && req.path == "/health") {
            JsonValue h = JsonValue::object();
            h.set("status", JsonValue::from_string("ok"));
            h.set("model", JsonValue::from_string(model_name));
            h.set("ctx_size", JsonValue::from_int(static_cast<int64_t>(reasoning.get_ctx_size())));
            h.set("backend", JsonValue::from_string(reasoning.get_backend_name()));
            conn.send_json(200, "OK", h.dump());
            return;
        }

        if (req.method == "GET" && req.path == "/v1/models") {
            JsonValue m = JsonValue::object();
            m.set("id", JsonValue::from_string(model_name));
            m.set("object", JsonValue::from_string("model"));
            m.set("created", JsonValue::from_int(unix_now()));
            m.set("owned_by", JsonValue::from_string("opengp"));

            JsonValue data = JsonValue::array();
            data.push_back(std::move(m));

            JsonValue root = JsonValue::object();
            root.set("object", JsonValue::from_string("list"));
            root.set("data", std::move(data));
            conn.send_json(200, "OK", root.dump());
            return;
        }

        if (req.method != "POST" || req.path != "/v1/chat/completions") {
            conn.send_json(404, "Not Found", make_error_json("Unknown endpoint: " + req.path));
            return;
        }

        JsonValue body;
        std::string parse_err;
        if (!JsonValue::parse(req.body, body, parse_err) || !body.is_object()) {
            conn.send_json(400, "Bad Request", make_error_json("Invalid JSON body: " + parse_err));
            return;
        }

        std::vector<ChatMessage> messages;
        std::string msg_err;
        if (!extract_messages(body, messages, msg_err)) {
            conn.send_json(400, "Bad Request", make_error_json(msg_err));
            return;
        }

        // Per-request generation parameters
        SamplerConfig sc = default_sampler;
        if (const auto * v = body.find("temperature"); v && v->is_number())    sc.temperature    = static_cast<float>(v->as_double());
        if (const auto * v = body.find("top_p");       v && v->is_number())    sc.top_p          = static_cast<float>(v->as_double());
        if (const auto * v = body.find("top_k");       v && v->is_number())    sc.top_k          = static_cast<int>(v->as_int());
        if (const auto * v = body.find("min_p");       v && v->is_number())    sc.min_p          = static_cast<float>(v->as_double());
        if (const auto * v = body.find("repeat_penalty"); v && v->is_number()) sc.repeat_penalty = static_cast<float>(v->as_double());
        if (const auto * v = body.find("seed");        v && v->is_number())    sc.seed           = static_cast<uint32_t>(v->as_int());
        responding.set_sampler_config(sc);

        size_t max_tokens = default_max;
        if (const auto * v = body.find("max_completion_tokens"); v && v->is_number()) {
            max_tokens = static_cast<size_t>(std::max<int64_t>(1, v->as_int()));
        } else if (const auto * v = body.find("max_tokens"); v && v->is_number()) {
            max_tokens = static_cast<size_t>(std::max<int64_t>(1, v->as_int()));
        }

        bool stream = false;
        if (const auto * v = body.find("stream"); v && v->is_bool()) {
            stream = v->as_bool();
        }

        std::vector<std::string> stop_seqs;
        if (const auto * v = body.find("stop")) {
            if (v->is_string() && !v->as_string().empty()) {
                stop_seqs.push_back(v->as_string());
            } else if (v->is_array()) {
                for (const auto & s : v->as_array()) {
                    if (s.is_string() && !s.as_string().empty()) stop_seqs.push_back(s.as_string());
                }
            }
        }
        size_t max_stop_len = 0;
        for (const auto & s : stop_seqs) max_stop_len = std::max(max_stop_len, s.size());

        // Phase 1: Format full chat history -> BPE tokens + embeddings
        PerceptionOutput p_out;
        if (!perception.process_messages(messages, p_out) || p_out.n_tokens == 0) {
            conn.send_json(500, "Internal Server Error", make_error_json("Perception pipeline failed", "server_error"));
            return;
        }

        const size_t ctx_cap = reasoning.get_kv_cache().get_capacity();
        if (p_out.n_tokens >= ctx_cap) {
            conn.send_json(400, "Bad Request",
                           make_error_json("Prompt length (" + std::to_string(p_out.n_tokens) +
                                           ") exceeds context capacity (" + std::to_string(ctx_cap) + ")"));
            return;
        }

        // KV-cache prefix reuse across turns (disabled for recurrent Hybrid SSM models)
        size_t n_reuse = 0;
        if (!reasoning.get_config().is_hybrid_ssm()) {
            const size_t kv_past = reasoning.get_kv_cache().get_n_past();
            const size_t limit   = std::min({ cached_tokens.size(), p_out.token_ids.size(), kv_past });
            while (n_reuse < limit && cached_tokens[n_reuse] == p_out.token_ids[n_reuse]) {
                ++n_reuse;
            }
            if (n_reuse == p_out.n_tokens && n_reuse > 0) {
                --n_reuse; // Forward at least 1 token to produce h_last
            }
        }

        if (n_reuse > 0) {
            reasoning.truncate_kv_cache(n_reuse);
        } else {
            reasoning.reset_kv_cache();
        }
        cached_tokens.clear();
        responding.reset();

        const size_t suffix_len = p_out.n_tokens - n_reuse;
        const float *   h0_ptr  = p_out.H_0.data() + n_reuse * p_out.d_model;
        const int32_t * pos_ptr = p_out.pos_ids.data() + n_reuse;

        ReasoningOutput r_out;
        if (!reasoning.forward(h0_ptr, pos_ptr, suffix_len, r_out)) {
            conn.send_json(500, "Internal Server Error", make_error_json("Reasoning prefill failed", "server_error"));
            return;
        }
        cached_tokens.assign(p_out.token_ids.begin(), p_out.token_ids.end());

        const std::string comp_id = "chatcmpl-opengp-" + std::to_string(++req_seq);
        const int64_t     created = unix_now();

        if (stream && !conn.begin_sse()) {
            return;
        }
        if (stream) {
            if (!conn.send_sse_data(make_chunk_json(comp_id, created, model_name, "assistant", "", ""))) {
                return;
            }
        }

        std::vector<int32_t> recent_tokens = p_out.token_ids;
        std::vector<float>   next_emb(p_out.d_model);
        const float *        h_last = r_out.H_L.data() + (r_out.n_tokens - 1) * r_out.d_model;
        ReasoningOutput      step_out;

        std::string full_reply;
        std::string pending;
        std::string finish_reason = "length";
        size_t      gen_tokens    = 0;
        bool        client_alive  = true;

        auto emit_text = [&](const std::string & text) {
            if (text.empty()) return;
            full_reply += text;
            if (stream && client_alive) {
                if (!conn.send_sse_data(make_chunk_json(comp_id, created, model_name, "", text, ""))) {
                    client_alive = false;
                }
            }
        };

        auto t_dec_start = std::chrono::high_resolution_clock::now();

        for (size_t step = 0; step < max_tokens && client_alive && !g_stop_server.load(); ++step) {
            int32_t next_id = responding.sample_next(h_last, recent_tokens);
            if (next_id < 0 || responding.is_stop_token(next_id)) {
                finish_reason = "stop";
                break;
            }

            std::string piece = responding.decode_token(next_id);
            recent_tokens.push_back(next_id);
            ++gen_tokens;

            if (stop_seqs.empty()) {
                emit_text(piece);
            } else {
                pending += piece;
                size_t earliest = std::string::npos;
                for (const auto & s : stop_seqs) {
                    size_t p = pending.find(s);
                    if (p != std::string::npos && (earliest == std::string::npos || p < earliest)) {
                        earliest = p;
                    }
                }
                if (earliest != std::string::npos) {
                    emit_text(pending.substr(0, earliest));
                    pending.clear();
                    finish_reason = "stop";
                    break;
                }
                if (pending.size() >= max_stop_len) {
                    size_t safe = pending.size() - (max_stop_len - 1);
                    emit_text(pending.substr(0, safe));
                    pending.erase(0, safe);
                }
            }

            if (gen_tokens >= max_tokens) {
                finish_reason = "length";
                break;
            }
            if (reasoning.get_kv_cache().get_n_past() >= ctx_cap) {
                finish_reason = "length";
                break;
            }

            if (!perception.embed_token(next_id, next_emb.data())) {
                finish_reason = "stop";
                break;
            }
            const int32_t next_pos = static_cast<int32_t>(reasoning.get_kv_cache().get_n_past());
            if (!reasoning.forward(next_emb.data(), &next_pos, 1, step_out)) {
                finish_reason = "stop";
                break;
            }
            cached_tokens.push_back(next_id);
            h_last = step_out.H_L.data();
        }

        std::string tail = responding.flush();
        if (client_alive) {
            if (!stop_seqs.empty()) {
                pending += tail;
                size_t earliest = std::string::npos;
                for (const auto & s : stop_seqs) {
                    size_t p = pending.find(s);
                    if (p != std::string::npos && (earliest == std::string::npos || p < earliest)) {
                        earliest = p;
                    }
                }
                if (earliest != std::string::npos) {
                    emit_text(pending.substr(0, earliest));
                    finish_reason = "stop";
                } else {
                    emit_text(pending);
                }
            } else {
                emit_text(tail);
            }
        }

        auto   t_dec_end = std::chrono::high_resolution_clock::now();
        double dec_ms    = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();
        double pref_tps  = (r_out.elapsed_ms > 0.0) ? (suffix_len * 1000.0 / r_out.elapsed_ms) : 0.0;
        double dec_tps   = (dec_ms > 0.0) ? (gen_tokens * 1000.0 / dec_ms) : 0.0;

        std::cout << "[Req #" << req_seq << "] msgs=" << messages.size()
                  << " | prompt=" << p_out.n_tokens << " (reused=" << n_reuse
                  << ", prefill=" << suffix_len << " @ " << std::fixed << std::setprecision(1) << pref_tps << " tok/s)"
                  << " | gen=" << gen_tokens << " @ " << std::setprecision(2) << dec_tps << " tok/s"
                  << " | stop=" << finish_reason << "\n" << std::flush;

        if (stream) {
            if (client_alive) {
                conn.send_sse_data(make_chunk_json(comp_id, created, model_name, "", "", finish_reason));
                conn.send_sse_data("[DONE]");
            }
        } else {
            conn.send_json(200, "OK",
                           make_completion_json(comp_id, created, model_name, full_reply, finish_reason,
                                                static_cast<int64_t>(p_out.n_tokens),
                                                static_cast<int64_t>(gen_tokens)));
        }
    };

    server.listen_and_serve(handle_request, g_stop_server);
    std::cout << "\n[Server] Shut down cleanly.\n";
    return 0;
}
