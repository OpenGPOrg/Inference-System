#pragma once

#include "ggml.h"
#include <string>
#include <vector>
#include <cstdint>

struct gguf_context;

namespace opengp {
namespace perception {

class EmbedTable {
public:
    EmbedTable();
    ~EmbedTable();

    // Memory-map the GGUF file and inspect token_embd.weight
    bool init_from_gguf(const char * filepath, struct gguf_context * gctx);

    // Gather rows for given token IDs and dequantize them to contiguous FP32 buffer
    // output buffer must be allocated of size: n_tokens * d_model * sizeof(float)
    bool gather_and_dequant(const std::vector<int32_t> & token_ids, 
                            float * out_activations, 
                            float scale = 1.0f) const;

    size_t get_d_model() const { return d_model_; }
    size_t get_vocab_size() const { return vocab_size_; }
    enum ggml_type get_type() const { return type_; }
    size_t get_row_bytes() const { return row_bytes_; }
    size_t get_table_bytes() const { return table_bytes_; }

private:
    int fd_ = -1;
    void * mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;

    const void * tensor_data_ = nullptr;
    size_t d_model_ = 0;
    size_t vocab_size_ = 0;
    size_t row_bytes_ = 0;
    size_t table_bytes_ = 0;
    enum ggml_type type_ = GGML_TYPE_F32;
};

} // namespace perception
} // namespace opengp
