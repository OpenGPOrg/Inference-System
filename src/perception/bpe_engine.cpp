#include "perception/bpe_engine.h"
#include "gguf.h"
#include <queue>
#include <climits>
#include <algorithm>
#include <iostream>

namespace opengp {
namespace perception {

BpeEngine::BpeEngine() {
    init_byte_mappings();
}

BpeEngine::~BpeEngine() = default;

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

void BpeEngine::init_byte_mappings() {
    byte_to_unicode_.clear();
    unicode_to_byte_.clear();

    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n += 1;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        std::string u_str = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
        byte_to_unicode_[byte_val] = u_str;
        unicode_to_byte_[u_str] = byte_val;
    }
}

bool BpeEngine::load_from_gguf(const struct gguf_context * gctx, const TokenizerConfig & config) {
    if (!gctx) return false;

    type_ = config.tokenizer_type;
    vocab_map_.clear();
    id_to_vocab_.clear();
    merges_map_.clear();

    // 1. Load vocabulary entries
    int64_t tokens_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) return false;

    int n_tokens = (int)gguf_get_arr_n(gctx, tokens_key);
    id_to_vocab_.resize(n_tokens);

    for (int i = 0; i < n_tokens; ++i) {
        std::string piece = gguf_get_arr_str(gctx, tokens_key, i);
        id_to_vocab_[i] = piece;
        vocab_map_[piece] = i;
    }

    // 2. Load merge rules
    int64_t merges_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        int n_merges = (int)gguf_get_arr_n(gctx, merges_key);
        for (int i = 0; i < n_merges; ++i) {
            std::string rule = gguf_get_arr_str(gctx, merges_key, i);
            size_t space_pos = rule.find(' ');
            if (space_pos != std::string::npos) {
                std::string p1 = rule.substr(0, space_pos);
                std::string p2 = rule.substr(space_pos + 1);

                auto it1 = vocab_map_.find(p1);
                auto it2 = vocab_map_.find(p2);
                auto it_merged = vocab_map_.find(p1 + p2);

                if (it1 != vocab_map_.end() && it2 != vocab_map_.end() && it_merged != vocab_map_.end()) {
                    uint64_t key = ((uint64_t)(uint32_t)it1->second << 32) | (uint32_t)it2->second;
                    merges_map_[key] = { i, it_merged->second };
                }
            }
        }
    }

    return true;
}

int32_t BpeEngine::token_to_id(const std::string & token) const {
    auto it = vocab_map_.find(token);
    if (it != vocab_map_.end()) return it->second;
    return -1;
}

std::string BpeEngine::id_to_token(int32_t id) const {
    if (id >= 0 && id < (int32_t)id_to_vocab_.size()) {
        return id_to_vocab_[id];
    }
    return "<UNK>";
}

struct BpeNode {
    int32_t token_id;
    int index;
    BpeNode * prev;
    BpeNode * next;
    bool deleted;
};

struct BpeCandidate {
    int32_t rank;
    int32_t merged_id;
    BpeNode * left;
    BpeNode * right;

    bool operator>(const BpeCandidate & other) const {
        return rank > other.rank; // min-heap
    }
};

void BpeEngine::bpe_tournament(const std::string & chunk, std::vector<int32_t> & out_ids) const {
    // Check if the chunk as a whole is already in vocab (e.g. control tokens, special headers)
    auto it_direct = vocab_map_.find(chunk);
    if (it_direct != vocab_map_.end()) {
        out_ids.push_back(it_direct->second);
        return;
    }

    // Step A: Convert chunk bytes to initial base Unicode pieces and Token IDs
    std::vector<int32_t> initial_ids;
    for (unsigned char c : chunk) {
        auto it_u = byte_to_unicode_.find(c);
        std::string u_str = (it_u != byte_to_unicode_.end()) ? it_u->second : std::string(1, (char)c);
        auto it_v = vocab_map_.find(u_str);
        if (it_v != vocab_map_.end()) {
            initial_ids.push_back(it_v->second);
        } else {
            // Byte fallback token if directly formatted as <0x..>
            char hex_buf[16];
            std::snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", c);
            auto it_hex = vocab_map_.find(hex_buf);
            if (it_hex != vocab_map_.end()) {
                initial_ids.push_back(it_hex->second);
            }
        }
    }

    if (initial_ids.empty()) return;
    if (initial_ids.size() == 1) {
        out_ids.push_back(initial_ids[0]);
        return;
    }

    // Step B: Build Doubly Linked List
    std::vector<BpeNode> nodes(initial_ids.size());
    for (size_t i = 0; i < initial_ids.size(); ++i) {
        nodes[i].token_id = initial_ids[i];
        nodes[i].index = (int)i;
        nodes[i].prev = (i > 0) ? &nodes[i - 1] : nullptr;
        nodes[i].next = (i + 1 < initial_ids.size()) ? &nodes[i + 1] : nullptr;
        nodes[i].deleted = false;
    }

    // Step C: Min-Heap of merge candidates
    std::priority_queue<BpeCandidate, std::vector<BpeCandidate>, std::greater<BpeCandidate>> min_heap;

    for (size_t i = 0; i < initial_ids.size() - 1; ++i) {
        uint64_t key = ((uint64_t)(uint32_t)nodes[i].token_id << 32) | (uint32_t)nodes[i + 1].token_id;
        auto it = merges_map_.find(key);
        if (it != merges_map_.end()) {
            min_heap.push({ it->second.rank, it->second.merged_id, &nodes[i], &nodes[i + 1] });
        }
    }

    // Step D: Merge Tournament Loop
    while (!min_heap.empty()) {
        BpeCandidate top = min_heap.top();
        min_heap.pop();

        BpeNode * left = top.left;
        BpeNode * right = top.right;

        // Lazy deletion verification
        if (left->deleted || right->deleted || left->next != right) {
            continue;
        }

        // O(1) Splicing
        left->token_id = top.merged_id;
        right->deleted = true;
        left->next = right->next;
        if (right->next) {
            right->next->prev = left;
        }

        // O(log N) Neighbor updates
        if (left->prev && !left->prev->deleted) {
            uint64_t l_key = ((uint64_t)(uint32_t)left->prev->token_id << 32) | (uint32_t)left->token_id;
            auto it_l = merges_map_.find(l_key);
            if (it_l != merges_map_.end()) {
                min_heap.push({ it_l->second.rank, it_l->second.merged_id, left->prev, left });
            }
        }

        if (left->next && !left->next->deleted) {
            uint64_t r_key = ((uint64_t)(uint32_t)left->token_id << 32) | (uint32_t)left->next->token_id;
            auto it_r = merges_map_.find(r_key);
            if (it_r != merges_map_.end()) {
                min_heap.push({ it_r->second.rank, it_r->second.merged_id, left, left->next });
            }
        }
    }

    // Step E: Emit final tokens from linked list
    BpeNode * curr = &nodes[0];
    while (curr) {
        if (!curr->deleted) {
            out_ids.push_back(curr->token_id);
        }
        curr = curr->next;
    }
}

std::vector<int32_t> BpeEngine::tokenize_chunks(const std::vector<std::string> & chunks) const {
    std::vector<int32_t> all_ids;
    for (const auto & chunk : chunks) {
        if (chunk.empty()) continue;
        bpe_tournament(chunk, all_ids);
    }
    return all_ids;
}

} // namespace perception
} // namespace opengp
