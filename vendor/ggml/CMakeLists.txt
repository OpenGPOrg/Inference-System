cmake_minimum_required(VERSION 3.14)
project(vendor_ggml C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# GNU extensions are required for CPU affinity / NUMA on Linux
add_definitions(-D_GNU_SOURCE)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-cpu
)

# Core GGML / GGUF implementation (including meta buffer)
add_library(ggml STATIC
    src/ggml.c
    src/ggml-alloc.c
    src/ggml-backend.cpp
    src/ggml-backend-reg.cpp
    src/ggml-backend-meta.cpp
    src/ggml-quants.c
    src/ggml-threading.cpp
    src/gguf.cpp
    src/ggml-cpu/ggml-cpu.c
    src/ggml-cpu/ggml-cpu.cpp
)

target_include_directories(ggml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Linux thread & math libraries
find_package(Threads REQUIRED)
target_link_libraries(ggml PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if (UNIX AND NOT APPLE)
    target_link_libraries(ggml PUBLIC m)
endif()
