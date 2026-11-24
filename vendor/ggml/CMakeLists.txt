cmake_minimum_required(VERSION 3.18)
project(vendor_ggml LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(OPENGP_CUDA "Enable CUDA GPU backend if CUDA Toolkit is available" ON)

# GNU extensions are required for CPU affinity / NUMA on Linux
add_definitions(-D_GNU_SOURCE)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-cpu
)

set(GGML_CPU_BACKEND_SOURCES
    src/ggml-cpu/ggml-cpu.c
    src/ggml-cpu/ggml-cpu.cpp
    src/ggml-cpu/repack.cpp
    src/ggml-cpu/hbm.cpp
    src/ggml-cpu/quants.c
    src/ggml-cpu/traits.cpp
    src/ggml-cpu/amx/amx.cpp
    src/ggml-cpu/amx/mmq.cpp
    src/ggml-cpu/tiled/tiled.cpp
    src/ggml-cpu/tiled/tiled-kernel.cpp
    src/ggml-cpu/binary-ops.cpp
    src/ggml-cpu/unary-ops.cpp
    src/ggml-cpu/vec.cpp
    src/ggml-cpu/ops.cpp
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i.86)$")
    list(APPEND GGML_CPU_BACKEND_SOURCES
        src/ggml-cpu/arch/x86/quants.c
        src/ggml-cpu/arch/x86/repack.cpp
    )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|arm.*)$")
    list(APPEND GGML_CPU_BACKEND_SOURCES
        src/ggml-cpu/arch/arm/quants.c
        src/ggml-cpu/arch/arm/repack.cpp
    )
endif()

# Core GGML / GGUF implementation (including meta buffer & full CPU backend)
add_library(ggml STATIC
    src/ggml.c
    src/ggml-alloc.c
    src/ggml-backend.cpp
    src/ggml-backend-reg.cpp
    src/ggml-backend-meta.cpp
    src/ggml-quants.c
    src/ggml-threading.cpp
    src/gguf.cpp
    ${GGML_CPU_BACKEND_SOURCES}
)

if (NOT MSVC)
    set_source_files_properties(${GGML_CPU_BACKEND_SOURCES} PROPERTIES COMPILE_FLAGS "-march=native")
endif()

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

# Optional CUDA Backend (auto-enabled when CUDA Toolkit is present)
if (OPENGP_CUDA)
    find_package(CUDAToolkit QUIET)
endif()

if (OPENGP_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "OpenGP: CUDA Toolkit ${CUDAToolkit_VERSION} found — enabling ggml-cuda backend")

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)
        endif()
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    file(GLOB GGML_SOURCES_CUDA "src/ggml-cuda/*.cu")
    file(GLOB SRCS_FATTN_TILE "src/ggml-cuda/template-instances/fattn-tile*.cu")
    file(GLOB SRCS_FATTN_MMA  "src/ggml-cuda/template-instances/fattn-mma*.cu")
    file(GLOB SRCS_MMQ        "src/ggml-cuda/template-instances/mmq*.cu")
    file(GLOB SRCS_MMF        "src/ggml-cuda/template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA
        ${SRCS_FATTN_TILE}
        ${SRCS_FATTN_MMA}
        ${SRCS_MMQ}
        ${SRCS_MMF}
        src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
    )

    add_library(ggml-cuda STATIC ${GGML_SOURCES_CUDA})

    target_include_directories(ggml-cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-cuda
    )

    # Configure FlashAttention KV combinations (default f16-f16; defines required by fattn.cu)
    set(FA_TYPES q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16 f8_e4m3)
    target_compile_definitions(ggml-cuda PRIVATE GGML_CUDA_FA_QUANTS="f16-f16")
    foreach (TYPE_V IN LISTS FA_TYPES)
        foreach (TYPE_K IN LISTS FA_TYPES)
            if ("${TYPE_K}-${TYPE_V}" STREQUAL "f16-f16")
                set(COMPILED 1)
            else()
                set(COMPILED 0)
            endif()
            string(TOUPPER "GGML_CUDA_FA_${TYPE_K}_${TYPE_V}" COMBINATION_DEF)
            target_compile_definitions(ggml-cuda PRIVATE ${COMBINATION_DEF}=${COMPILED})
        endforeach()
    endforeach()

    target_compile_options(ggml-cuda PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math;-extended-lambda;-Xcompiler;-Wno-pedantic>"
    )

    target_link_libraries(ggml-cuda PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cuda_driver
    )

    target_link_libraries(ggml PUBLIC ggml-cuda)
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_CUDA=1)
else()
    message(STATUS "OpenGP: CUDA Toolkit not available or disabled — building CPU-only backend")
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_CUDA=0)
endif()
