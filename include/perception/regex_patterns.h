#pragma once

#include <string>

namespace opengp {
namespace perception {

enum class RegexPatternType {
    MULTI_DIGIT_CONTRACTION,  // e.g. LLaMA-3 style: contractions, words, 1-3 digits, symbols
    SINGLE_DIGIT_CONTRACTION, // e.g. Qwen style: contractions, words, single digits, symbols
    CLASSIC_WHITESPACE        // e.g. Classical GPT-2 / standard BPE regex
};

} // namespace perception
} // namespace opengp
