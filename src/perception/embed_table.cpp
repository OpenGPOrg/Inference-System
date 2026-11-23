#include "perception/embed_table.h"
#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace opengp {
namespace perception {

EmbedTable::EmbedTable() = default;

EmbedTable::~EmbedTable() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool EmbedTable::init_from_gguf(const char * filepath, struct gguf_context * gctx) {
    if (!filepath || !gctx) return false;

    // 1. Find embedding tensor descriptor
    int64_t tensor_idx = gguf_find_tensor(gctx, "token_embd.weight");
    if (tensor_idx < 0) {
        std::fprintf(stderr, "[-] EmbedTable: 'token_embd.weight' not found in GGUF\n");
        return false;
    }

    type_ = gguf_get_tensor_type(gctx, tensor_idx);
    const int64_t * ne = gguf_get_tensor_ne(gctx, tensor_idx);
    d_model_ = (size_t)ne[0];
    vocab_size_ = (size_t)ne[1];
    uint64_t tensor_offset = gguf_get_tensor_offset(gctx, tensor_idx);

    row_bytes_ = ggml_row_size(type_, d_model_);
    table_bytes_ = row_bytes_ * vocab_size_;

    // 2. Open and mmap file for zero-copy Host-RAM reading
    fd_ = open(filepath, O_RDONLY);
    if (fd_ < 0) {
        std::fprintf(stderr, "[-] EmbedTable: Failed to open '%s'\n", filepath);
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    mmap_size_ = sb.st_size;

    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        std::fprintf(stderr, "[-] EmbedTable: Failed to mmap '%s'\n", filepath);
        close(fd_);
        fd_ = -1;
        return false;
    }

    // GGUF tensor binary data begins after the metadata header (at alignment offset)
    size_t data_offset = gguf_get_data_offset(gctx);
    tensor_data_ = (const uint8_t *)mmap_addr_ + data_offset + tensor_offset;

    return true;
}

bool EmbedTable::gather_and_dequant(const std::vector<int32_t> & token_ids, 
                                    float * out_activations, 
                                    float scale) const {
    if (!tensor_data_ || !out_activations) return false;

    const ggml_type_traits * traits = ggml_get_type_traits(type_);
    const uint8_t * base_ptr = (const uint8_t *)tensor_data_;

    for (size_t i = 0; i < token_ids.size(); ++i) {
        int32_t tid = token_ids[i];
        if (tid < 0 || (size_t)tid >= vocab_size_) {
            // Out-of-bounds safeguard: zero-fill row
            std::memset(out_activations + (i * d_model_), 0, d_model_ * sizeof(float));
            continue;
        }

        // Direct O(1) pointer arithmetic: base + tid * row_bytes
        const void * src_row = base_ptr + ((size_t)tid * row_bytes_);
        float * dst_row = out_activations + (i * d_model_);

        if (type_ == GGML_TYPE_F32) {
            std::memcpy(dst_row, src_row, d_model_ * sizeof(float));
        } else if (traits && traits->to_float) {
            traits->to_float(src_row, dst_row, d_model_);
        } else {
            // Fallback: zero-fill
            std::memset(dst_row, 0, d_model_ * sizeof(float));
        }

        // Apply model-specific scaling (e.g. Gemma sqrt(d_model))
        if (scale != 1.0f) {
            for (size_t j = 0; j < d_model_; ++j) {
                dst_row[j] *= scale;
            }
        }
    }

    return true;
}

} // namespace perception
} // namespace opengp
