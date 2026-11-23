#include "perception/perception_pipeline.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace opengp::perception;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_file.gguf> [\"user prompt string\"]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = (argc > 2) ? argv[2] : "What is phishing attack?";

    std::cout << "=================================================================\n";
    std::cout << "       OPENGP MODULAR PERCEPTION ENGINE CLI HARNESS               \n";
    std::cout << "=================================================================\n";
    std::cout << "Model Path   : " << model_path << "\n";
    std::cout << "Input Prompt : \"" << prompt << "\"\n";
    std::cout << "-----------------------------------------------------------------\n";

    PerceptionPipeline pipeline;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!pipeline.initialize(model_path)) {
        std::cerr << "[-] Error: Failed to initialize perception pipeline from " << model_path << std::endl;
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto & config = pipeline.get_config();
    const auto & embed = pipeline.get_embed_table();

    std::cout << "[Config] Model Arch   : " << config.model_architecture << "\n";
    std::cout << "[Config] Pre-tokenizer: " << config.pre_tokenizer << "\n";
    std::cout << "[Config] Vocab Size   : " << config.vocab_size << " tokens\n";
    std::cout << "[Config] Merges Rules : " << config.n_merges << " pairs\n";
    std::cout << "[Config] Special IDs  : BOS=" << config.bos_token_id 
              << ", EOS=" << config.eos_token_id 
              << ", UNK=" << config.unk_token_id << "\n";
    std::cout << "[Embed]  Tensor Shape : [" << embed.get_vocab_size() << " x " << embed.get_d_model() << "]\n";
    std::cout << "[Embed]  Quant Type   : " << ggml_type_name(embed.get_type()) 
              << " (" << embed.get_row_bytes() << " bytes/row, "
              << (embed.get_table_bytes() / 1024.0 / 1024.0) << " MB in Host RAM)\n";
    std::cout << "[Embed]  VRAM Used    : 0.00 MB (Preserved for Reasoning Layers & KV Cache)\n";
    std::cout << "-----------------------------------------------------------------\n";

    PerceptionOutput output;
    auto t2 = std::chrono::high_resolution_clock::now();
    if (!pipeline.process(prompt, output)) {
        std::cerr << "[-] Error: Failed to execute perception pipeline on prompt" << std::endl;
        return 1;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double proc_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Display Diagnostic Breakdown
    std::cout << "\n>>> STAGE 1 & 2: REGEX SEGMENTATION (" << output.chunks.size() << " chunks):\n[ ";
    for (size_t i = 0; i < std::min(output.chunks.size(), (size_t)12); ++i) {
        std::cout << "\"" << output.chunks[i] << "\"" << (i + 1 < output.chunks.size() ? ", " : " ");
    }
    if (output.chunks.size() > 12) std::cout << "... +" << (output.chunks.size() - 12) << " more";
    std::cout << "]\n";

    std::cout << "\n>>> STAGE 3 & 4: BPE TOKEN IDS (" << output.n_tokens << " tokens):\n[ ";
    for (size_t i = 0; i < output.token_ids.size(); ++i) {
        std::cout << output.token_ids[i] << (i + 1 < output.token_ids.size() ? ", " : " ");
    }
    std::cout << "]\n";

    std::cout << "\n>>> STAGE 5: POSITION IDS (Deferred RoPE):\n[ ";
    for (size_t i = 0; i < output.pos_ids.size(); ++i) {
        std::cout << output.pos_ids[i] << (i + 1 < output.pos_ids.size() ? ", " : " ");
    }
    std::cout << "]\n";

    std::cout << "\n>>> STAGE 6, 7 & 8: ASSEMBLED ACTIVATION MATRIX (H_0):\n";
    std::cout << "Shape        : [" << output.n_tokens << " x " << output.d_model << "]\n";
    std::cout << "Buffer Size  : " << output.bytes() << " bytes (Contiguous FP32, ready for DMA)\n";
    std::cout << "Memory Target: Layer 0 RMSNorm ('blk.0.attn_norm')\n\n";

    // Print first 4 latent dimensions and L2 norm of each token vector
    std::cout << "Token Vector Samples:\n";
    for (size_t i = 0; i < std::min(output.n_tokens, (size_t)5); ++i) {
        const float * row = output.H_0.data() + (i * output.d_model);
        double l2_norm = 0.0;
        for (size_t d = 0; d < output.d_model; ++d) {
            l2_norm += row[d] * row[d];
        }
        l2_norm = std::sqrt(l2_norm);

        std::cout << "  Row " << std::setw(2) << i 
                  << " (ID " << std::setw(6) << output.token_ids[i] << "): [ "
                  << std::fixed << std::setprecision(4)
                  << row[0] << ", " << row[1] << ", " << row[2] << ", " << row[3] 
                  << " ... ] (Norm: " << std::setprecision(3) << l2_norm << ")\n";
    }

    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "Perception Pipeline Initialized in : " << std::fixed << std::setprecision(2) << init_ms << " ms\n";
    std::cout << "Perception Pipeline Processed in   : " << std::fixed << std::setprecision(2) << proc_ms << " ms\n";
    std::cout << "PERCEPTION PHASE COMPLETE -> READY FOR LAYER 0 REASONING!\n";
    std::cout << "=================================================================\n";

    return 0;
}
