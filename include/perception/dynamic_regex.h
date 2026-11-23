#pragma once

#include "perception/tokenizer_config.h"
#include "perception/regex_patterns.h"
#include <string>
#include <vector>
#include <regex>

namespace opengp {
namespace perception {

class DynamicRegex {
public:
    DynamicRegex();
    explicit DynamicRegex(RegexPatternType pattern_type);

    // Configure dynamically from pre-tokenizer string (e.g. "qwen2", "llama3", "default")
    void configure_from_pre(const std::string & pre_tokenizer);

    // Splits input text into isolated semantic chunks (words, numbers, punctuation)
    std::vector<std::string> split(const std::string & text) const;

private:
    RegexPatternType pattern_type_ = RegexPatternType::CLASSIC_WHITESPACE;
    std::regex regex_pattern_;
    void compile_pattern();
};

} // namespace perception
} // namespace opengp
