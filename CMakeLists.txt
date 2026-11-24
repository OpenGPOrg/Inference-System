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
# 4. Modular Reasoning Engine Core Library (Phase 2)
# -----------------------------------------------------------------------------
add_library(opengp_reasoning_core STATIC
    src/reasoning/model_config.cpp
    src/reasoning/weight_store.cpp
    src/reasoning/kv_cache.cpp
    src/reasoning/layer_graph.cpp
    src/reasoning/reasoning_engine.cpp
)

target_link_libraries(opengp_reasoning_core PUBLIC ggml opengp_perception_core)

# -----------------------------------------------------------------------------
# 5. Dedicated Reasoning Engine CLI Executable
# -----------------------------------------------------------------------------
add_executable(opengp_reasoning src/reasoning/main_reasoning.cpp)
target_link_libraries(opengp_reasoning PRIVATE opengp_reasoning_core)
