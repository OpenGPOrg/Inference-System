#pragma once

#include "perception/tokenizer_config.h"
#include <string>
#include <vector>

namespace opengp {
namespace perception {

struct ChatMessage {
    std::string role;
    std::string content;
};

class PromptTemplate {
public:
    // Format raw user prompt into model-appropriate conversation frame
    static std::string format(const TokenizerConfig & config, 
                              const std::string & user_prompt, 
                              const std::string & system_prompt = "");

    // Format multi-turn conversation history into model-appropriate prompt frame
    static std::string format_messages(const TokenizerConfig &          config,
                                       const std::vector<ChatMessage> & messages);
};

} // namespace perception
} // namespace opengp
