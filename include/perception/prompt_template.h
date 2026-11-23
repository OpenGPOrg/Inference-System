#pragma once

#include "perception/tokenizer_config.h"
#include <string>

namespace opengp {
namespace perception {

class PromptTemplate {
public:
    // Format raw user prompt into model-appropriate conversation frame
    static std::string format(const TokenizerConfig & config, 
                              const std::string & user_prompt, 
                              const std::string & system_prompt = "");
};

} // namespace perception
} // namespace opengp
