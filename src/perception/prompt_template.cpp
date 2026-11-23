#include "perception/prompt_template.h"

namespace opengp {
namespace perception {

std::string PromptTemplate::format(const TokenizerConfig & config, 
                                   const std::string & user_prompt, 
                                   const std::string & system_prompt) {
    // 1. Qwen / ChatML format
    if (config.pre_tokenizer == "qwen2" || config.pre_tokenizer == "qwen35" || 
        config.chat_template.find("<|im_start|>") != std::string::npos) {
        std::string out;
        if (!system_prompt.empty()) {
            out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        }
        out += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
        return out;
    }

    // 2. LLaMA-3 Header format
    if (config.pre_tokenizer == "llama3" || 
        config.chat_template.find("<|start_header_id|>") != std::string::npos) {
        std::string out = "<|begin_of_text|>";
        if (!system_prompt.empty()) {
            out += "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>";
        }
        out += "<|start_header_id|>user<|end_header_id|>\n\n" + user_prompt + "<|eot_id|>";
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    // 3. Raw / Fallback completion mode (e.g. Stories-15M or base foundation models)
    if (!system_prompt.empty()) {
        return system_prompt + "\n" + user_prompt;
    }
    return user_prompt;
}

} // namespace perception
} // namespace opengp
