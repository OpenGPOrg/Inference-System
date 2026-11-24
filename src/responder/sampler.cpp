#include "responder/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace opengp {
namespace responder {

Sampler::Sampler(const SamplerConfig & config) {
    configure(config);
}

void Sampler::configure(const SamplerConfig & config) {
    config_ = config;
    if (config_.seed != 0) {
        rng_.seed(config_.seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

int32_t Sampler::sample(std::vector<float> & logits, const std::vector<int32_t> & recent_tokens) {
    if (logits.empty()) return -1;

    const int32_t vocab_size = static_cast<int32_t>(logits.size());

    // 1. Repetition penalty over recent token window
    if (config_.repeat_penalty > 1.0f && !recent_tokens.empty() && config_.repeat_last_n > 0) {
        const size_t start = (recent_tokens.size() > config_.repeat_last_n)
                                 ? (recent_tokens.size() - config_.repeat_last_n)
                                 : 0;
        std::unordered_set<int32_t> seen;
        seen.reserve(recent_tokens.size() - start);
        for (size_t i = start; i < recent_tokens.size(); ++i) {
            int32_t tid = recent_tokens[i];
            if (tid >= 0 && tid < vocab_size && seen.insert(tid).second) {
                if (logits[tid] <= 0.0f) {
                    logits[tid] *= config_.repeat_penalty;
                } else {
                    logits[tid] /= config_.repeat_penalty;
                }
            }
        }
    }

    // 2. Greedy decoding fast path (temperature <= 0)
    if (config_.temperature <= 0.0f) {
        auto best_it = std::max_element(logits.begin(), logits.end());
        return static_cast<int32_t>(std::distance(logits.begin(), best_it));
    }

    // 3. Temperature scaling + Top-K candidate selection
    const float inv_temp = 1.0f / config_.temperature;
    std::vector<std::pair<float, int32_t>> candidates(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        candidates[i] = { logits[i] * inv_temp, i };
    }

    size_t n_cand = candidates.size();
    if (config_.top_k > 0 && static_cast<size_t>(config_.top_k) < n_cand) {
        n_cand = static_cast<size_t>(config_.top_k);
        std::partial_sort(candidates.begin(), candidates.begin() + n_cand, candidates.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        candidates.resize(n_cand);
    } else {
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
    }

    // 4. Numerically stable Softmax over candidates
    const float max_logit = candidates[0].first;
    std::vector<float> probs(n_cand);
    double sum_exp = 0.0;
    for (size_t i = 0; i < n_cand; ++i) {
        float w = std::exp(candidates[i].first - max_logit);
        probs[i] = w;
        sum_exp += w;
    }
    const float inv_sum = (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp) : 1.0f;
    for (size_t i = 0; i < n_cand; ++i) {
        probs[i] *= inv_sum;
    }

    // 5. Min-P filter: keep tokens >= min_p * p_max
    size_t keep = n_cand;
    if (config_.min_p > 0.0f) {
        const float threshold = config_.min_p * probs[0];
        keep = 1;
        while (keep < n_cand && probs[keep] >= threshold) {
            keep++;
        }
    }

    // 6. Top-P (Nucleus) filter
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        float cum = 0.0f;
        size_t nucleus = 0;
        for (size_t i = 0; i < keep; ++i) {
            cum += probs[i];
            nucleus = i + 1;
            if (cum >= config_.top_p) {
                break;
            }
        }
        keep = std::max<size_t>(1, nucleus);
    }

    // 7. Multinomial draw from surviving distribution
    double final_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        final_sum += probs[i];
    }

    std::uniform_real_distribution<double> dist(0.0, final_sum);
    const double u = dist(rng_);
    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[i];
        if (u <= acc) {
            return candidates[i].second;
        }
    }
    return candidates[keep - 1].second;
}

} // namespace responder
} // namespace opengp
