#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace opengp {
namespace responder {

struct SamplerConfig {
    float    temperature    = 0.7f;  // <= 0.0f selects greedy argmax
    int      top_k          = 40;    // 0 = disabled
    float    top_p          = 0.90f; // >= 1.0f = disabled (Nucleus sampling)
    float    min_p          = 0.05f; // <= 0.0f = disabled
    float    repeat_penalty = 1.10f; // 1.0f = disabled
    size_t   repeat_last_n  = 64;    // Sliding window of recent tokens to penalize
    uint32_t seed           = 0;     // 0 = random seed via std::random_device
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig & config = SamplerConfig{});

    void configure(const SamplerConfig & config);
    const SamplerConfig & get_config() const { return config_; }

    // Mutates `logits` in place (penalty, temperature, filtering) and samples next token ID
    int32_t sample(std::vector<float> & logits, const std::vector<int32_t> & recent_tokens);

private:
    SamplerConfig config_;
    std::mt19937  rng_;
};

} // namespace responder
} // namespace opengp
