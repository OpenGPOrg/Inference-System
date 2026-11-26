cmake_minimum_required(VERSION 3.14)
project(opengp-engine LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1. Self-Contained Vendor: GGML & GGUF (In-Tree)
# -----------------------------------------------------------------------------
add_subdirectory(vendor/ggml)

# Include directories: local headers and vendor GGML headers
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/include"
)

# -----------------------------------------------------------------------------
# 2. Modular Perception Engine Core Library
# -----------------------------------------------------------------------------
add_library(opengp_perception_core STATIC
    src/perception/tokenizer_config.cpp
    src/perception/prompt_template.cpp
    src/perception/dynamic_regex.cpp
    src/perception/bpe_engine.cpp
    src/perception/embed_table.cpp
    src/perception/perception_pipeline.cpp
)

target_link_libraries(opengp_perception_core PUBLIC ggml)

# -----------------------------------------------------------------------------
# 3. Dedicated Perception Engine CLI Executable
# -----------------------------------------------------------------------------
add_executable(opengp_perception src/perception/main_perception.cpp)
target_link_libraries(opengp_perception PRIVATE opengp_perception_core)

# -----------------------------------------------------------------------------
# 4. Modular Reasoner Engine Core Library (Phase 2)
# -----------------------------------------------------------------------------
add_library(opengp_reasoner_core STATIC
    src/reasoner/backend_selector.cpp
    src/reasoner/model_config.cpp
    src/reasoner/weight_store.cpp
    src/reasoner/kv_cache.cpp
    src/reasoner/recurrent_cache.cpp
    src/reasoner/layer_graph.cpp
    src/reasoner/reasoner_engine.cpp
)

target_link_libraries(opengp_reasoner_core PUBLIC ggml opengp_perception_core)

# -----------------------------------------------------------------------------
# 5. Dedicated Reasoner Engine CLI Executable
# -----------------------------------------------------------------------------
add_executable(opengp_reasoner src/reasoner/main_reasoner.cpp)
target_link_libraries(opengp_reasoner PRIVATE opengp_reasoner_core)

# -----------------------------------------------------------------------------
# 6. Modular Responder Engine Core Library (Phase 3)
# -----------------------------------------------------------------------------
add_library(opengp_responder_core STATIC
    src/responder/lm_head.cpp
    src/responder/sampler.cpp
    src/responder/detokenizer.cpp
    src/responder/responder_engine.cpp
)

target_link_libraries(opengp_responder_core PUBLIC ggml opengp_perception_core opengp_reasoner_core)

# -----------------------------------------------------------------------------
# 7. Unified End-to-End Inference CLI Executable (Phase 1 -> 2 -> 3)
# -----------------------------------------------------------------------------
add_executable(opengp_infer src/responder/main_infer.cpp)
target_link_libraries(opengp_infer PRIVATE opengp_responder_core)

# -----------------------------------------------------------------------------
# 8. Standalone OpenAI-Compatible HTTP Server Executable (/v1/chat/completions)
# -----------------------------------------------------------------------------
find_package(Threads REQUIRED)
add_executable(opengp_server
    src/server/mini_json.cpp
    src/server/http_server.cpp
    src/server/main_server.cpp
)
target_link_libraries(opengp_server PRIVATE opengp_responder_core Threads::Threads)
