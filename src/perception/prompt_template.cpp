#include "perception/prompt_template.h"

namespace opengp {
namespace perception {

std::string PromptTemplate::format(const TokenizerConfig & config,
                                   const std::string & user_prompt,
                                   const std::string & system_prompt) {
    std::vector<ChatMessage> messages;
    if (!system_prompt.empty()) {
        messages.push_back({ "system", system_prompt });
    }
    messages.push_back({ "user", user_prompt });
    return format_messages(config, messages);
}

std::string PromptTemplate::format_messages(const TokenizerConfig &          config,
                                            const std::vector<ChatMessage> & messages) {
    // 1. Qwen / ChatML format
    if (config.pre_tokenizer == "qwen2" || config.pre_tokenizer == "qwen35" ||
        config.chat_template.find("<|im_start|>") != std::string::npos) {
        std::string out;
        for (const auto & msg : messages) {
            out += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        }
        out += "<|im_start|>assistant\n";
        return out;
    }

    // 2. LLaMA-3 Header format
    if (config.pre_tokenizer == "llama3" ||
        config.chat_template.find("<|start_header_id|>") != std::string::npos) {
        std::string out = "<|begin_of_text|>";
        for (const auto & msg : messages) {
            out += "<|start_header_id|>" + msg.role + "<|end_header_id|>\n\n" + msg.content + "<|eot_id|>";
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    // 3. Raw / Fallback completion mode (e.g. Stories-15M or base foundation models)
    std::string out;
    for (size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) out += "\n";
        out += messages[i].content;
    }
    return out;
}

} // namespace perception
} // namespace opengp
