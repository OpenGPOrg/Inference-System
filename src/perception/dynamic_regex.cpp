#include "perception/dynamic_regex.h"
#include <iostream>

namespace opengp {
namespace perception {

DynamicRegex::DynamicRegex() {
    compile_pattern();
}

DynamicRegex::DynamicRegex(RegexPatternType pattern_type) : pattern_type_(pattern_type) {
    compile_pattern();
}

void DynamicRegex::configure_from_pre(const std::string & pre_tokenizer) {
    // Model-Agnostic Pattern Matching:
    // Models with single-digit number tokens (like Qwen) use SINGLE_DIGIT_CONTRACTION
    // Models with multi-digit number tokens (like LLaMA-3) use MULTI_DIGIT_CONTRACTION
    if (pre_tokenizer == "llama3") {
        pattern_type_ = RegexPatternType::MULTI_DIGIT_CONTRACTION;
    } else if (pre_tokenizer == "qwen2" || pre_tokenizer == "qwen35") {
        pattern_type_ = RegexPatternType::SINGLE_DIGIT_CONTRACTION;
    } else {
        pattern_type_ = RegexPatternType::CLASSIC_WHITESPACE;
    }
    compile_pattern();
}

void DynamicRegex::compile_pattern() {
    try {
        switch (pattern_type_) {
            case RegexPatternType::MULTI_DIGIT_CONTRACTION:
                // Contractions, letters with optional prefix, 1-3 digits, symbols, whitespaces
                regex_pattern_ = std::regex(
                    "('s|'t|'re|'ve|'m|'ll|'d)|"
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}]+|"
                    "[0-9]{1,3}|"
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
                    "\\s*[\\r\\n]+|"
                    "\\s+",
                    std::regex::ECMAScript | std::regex::optimize
                );
                break;

            case RegexPatternType::SINGLE_DIGIT_CONTRACTION:
                // Contractions, letters with optional prefix, single digits, symbols, whitespaces
                regex_pattern_ = std::regex(
                    "('s|'t|'re|'ve|'m|'ll|'d)|"
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}]+|"
                    "[0-9]|"
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
                    "\\s*[\\r\\n]+|"
                    "\\s+",
                    std::regex::ECMAScript | std::regex::optimize
                );
                break;

            case RegexPatternType::CLASSIC_WHITESPACE:
            default:
                // Standard word, digit, and whitespace boundaries
                regex_pattern_ = std::regex(
                    "('s|'t|'re|'ve|'m|'ll|'d)|"
                    " ?[a-zA-Z]+|"
                    " ?[0-9]+|"
                    " ?[^\\sa-zA-Z0-9]+|"
                    "\\s+",
                    std::regex::ECMAScript | std::regex::optimize
                );
                break;
        }
    } catch (const std::exception & e) {
        // Fallback robust pattern
        regex_pattern_ = std::regex(
            "('s|'t|'re|'ve|'m|'ll|'d)|"
            " ?[a-zA-Z]+|"
            " ?[0-9]+|"
            " ?[^\\sa-zA-Z0-9]+|"
            "\\s+",
            std::regex::ECMAScript | std::regex::optimize
        );
    }
}

std::vector<std::string> DynamicRegex::split(const std::string & text) const {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;

    size_t pos = 0;
    while (pos < text.size()) {
        // Detect special tag delimiter '<|'
        if (text.compare(pos, 2, "<|") == 0) {
            size_t tag_end = text.find("|>", pos + 2);
            if (tag_end != std::string::npos) {
                size_t tag_len = (tag_end + 2) - pos;
                chunks.push_back(text.substr(pos, tag_len));
                pos = tag_end + 2;
                continue;
            }
        }

        size_t next_tag = text.find("<|", pos);
        size_t seg_len = (next_tag == std::string::npos) ? (text.size() - pos) : (next_tag - pos);
        std::string segment = text.substr(pos, seg_len);

        auto words_begin = std::sregex_iterator(segment.begin(), segment.end(), regex_pattern_);
        auto words_end = std::sregex_iterator();

        size_t matched_bytes = 0;
        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::smatch match = *it;
            chunks.push_back(match.str());
            matched_bytes += match.length();
        }

        if (matched_bytes < segment.size() && words_begin == words_end) {
            chunks.push_back(segment);
        }

        pos += seg_len;
    }

    return chunks;
}

} // namespace perception
} // namespace opengp
