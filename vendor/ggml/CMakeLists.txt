cmake_minimum_required(VERSION 3.18)
project(vendor_ggml LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(OPENGP_CUDA  "Enable CUDA GPU backend if CUDA Toolkit is available" ON)
option(OPENGP_METAL "Enable Apple Metal GPU backend on macOS" ON)

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

# Optional Apple Metal Backend (auto-enabled on macOS with embedded .metal kernels)
if (APPLE AND OPENGP_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)

    message(STATUS "OpenGP: Apple Metal framework found — enabling ggml-metal backend")
    enable_language(OBJC ASM)

    set(METAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal")
    set(METALLIB_COMMON             "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-common.h")
    set(METALLIB_IMPL               "${METAL_DIR}/ggml-metal-impl.h")
    set(METALLIB_KERNELS_COMMON     "${METAL_DIR}/kernels/common.h")
    set(METALLIB_KERNELS_DEQUANTIZE "${METAL_DIR}/kernels/dequantize.h")
    set(METALLIB_KERNELS_QUANTIZE   "${METAL_DIR}/kernels/quantize.h")
    set(METALLIB_KERNELS_FA_COMMON     "${METAL_DIR}/kernels/fa_common.metal")
    set(METALLIB_KERNELS_FA_VEC_COMMON "${METAL_DIR}/kernels/fa_vec_common.metal")

    set(METALLIB_KERNEL_SOURCES
        kernels/fa_aux.metal kernels/fa_f16.metal kernels/fa_f32.metal
        kernels/fa_q4_0.metal kernels/fa_q4_1.metal kernels/fa_q5_0.metal kernels/fa_q5_1.metal kernels/fa_q8_0.metal
        kernels/fa_tensor.metal
        kernels/fa_vec_f16.metal kernels/fa_vec_f32.metal
        kernels/fa_vec_q4_0.metal kernels/fa_vec_q4_1.metal kernels/fa_vec_q5_0.metal kernels/fa_vec_q5_1.metal kernels/fa_vec_q8_0.metal
        kernels/mul_mv.metal kernels/mul_mv_mma.metal kernels/mul_mm.metal
        kernels/quantize.metal kernels/softmax.metal kernels/norm.metal kernels/unary.metal
        kernels/binbcast.metal kernels/reduce.metal kernels/tri.metal
        kernels/ssm.metal kernels/wkv.metal kernels/gated_delta_net.metal kernels/solve_tri.metal
        kernels/rope.metal kernels/conv.metal kernels/upscale.metal
        kernels/argsort.metal kernels/pool.metal kernels/misc.metal
    )

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metal_autogenerated")
    set(METALLIB_EMBED_ASM_FILES "")
    foreach(src ${METALLIB_KERNEL_SOURCES})
        get_filename_component(kind ${src} NAME_WE)
        string(REPLACE "-" "_" kind_sym ${kind})

        set(SRC   "${METAL_DIR}/kernels/${kind}.metal")
        set(EMBED "${CMAKE_CURRENT_BINARY_DIR}/metal_autogenerated/ggml-metal-embed-${kind}.metal")
        set(ASM   "${CMAKE_CURRENT_BINARY_DIR}/metal_autogenerated/ggml-metal-embed-${kind}.s")

        set(HEADERS_FOR_SRC ${METALLIB_KERNELS_COMMON})
        file(STRINGS ${SRC} _has_dequantize    REGEX "#include \"dequantize\\.h\"")
        file(STRINGS ${SRC} _has_quantize      REGEX "#include \"quantize\\.h\"")
        file(STRINGS ${SRC} _has_fa_common     REGEX "#include \"fa_common\\.metal\"")
        file(STRINGS ${SRC} _has_fa_vec_common REGEX "#include \"fa_vec_common\\.metal\"")
        if(_has_dequantize)
            list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_DEQUANTIZE})
        endif()
        if(_has_quantize)
            list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_QUANTIZE})
        endif()
        if(_has_fa_common)
            list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_FA_COMMON})
        endif()
        if(_has_fa_vec_common)
            list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_FA_VEC_COMMON})
        endif()

        add_custom_command(
            OUTPUT "${ASM}"
            COMMAND cat ${HEADERS_FOR_SRC} ${SRC} > "${EMBED}.tmp1"
            COMMAND sed -e "/\\#include \"common.h\"/d" -e "/\\#include \"dequantize.h\"/d" -e "/\\#include \"quantize.h\"/d" -e "/\\#include \"fa_common.metal\"/d" -e "/\\#include \"fa_vec_common.metal\"/d" -e "/\\#pragma once/d" < "${EMBED}.tmp1" > "${EMBED}.tmp2"
            COMMAND sed -e "/__embed_ggml-common.h__/r ${METALLIB_COMMON}" -e "/__embed_ggml-common.h__/d" < "${EMBED}.tmp2" > "${EMBED}.tmp3"
            COMMAND sed -e "/\\#include \"ggml-metal-impl.h\"/r ${METALLIB_IMPL}" -e "/\\#include \"ggml-metal-impl.h\"/d" < "${EMBED}.tmp3" > "${EMBED}"
            COMMAND echo ".section __DATA,__ggml_metallib"                       >  "${ASM}"
            COMMAND echo ".globl _ggml_metallib_${kind_sym}_start"               >> "${ASM}"
            COMMAND echo "_ggml_metallib_${kind_sym}_start:"                     >> "${ASM}"
            COMMAND echo .incbin "\"${EMBED}\""                                  >> "${ASM}"
            COMMAND echo ".globl _ggml_metallib_${kind_sym}_end"                 >> "${ASM}"
            COMMAND echo "_ggml_metallib_${kind_sym}_end:"                       >> "${ASM}"
            DEPENDS ${HEADERS_FOR_SRC} ${METALLIB_COMMON} ${METALLIB_IMPL} ${SRC}
            COMMENT "Generate embedded Metal library for ${kind}"
            VERBATIM
        )
        list(APPEND METALLIB_EMBED_ASM_FILES "${ASM}")
    endforeach()

    add_library(ggml-metal STATIC
        src/ggml-metal/ggml-metal.cpp
        src/ggml-metal/ggml-metal-device.m
        src/ggml-metal/ggml-metal-device.cpp
        src/ggml-metal/ggml-metal-common.cpp
        src/ggml-metal/ggml-metal-context.m
        src/ggml-metal/ggml-metal-fusion.cpp
        src/ggml-metal/ggml-metal-ops.cpp
        src/ggml-metal/ggml-metal-tuning.cpp
        ${METALLIB_EMBED_ASM_FILES}
    )

    target_include_directories(ggml-metal PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${METAL_DIR}
    )
    target_compile_definitions(ggml-metal PRIVATE GGML_METAL_EMBED_LIBRARY)
    target_link_libraries(ggml-metal PRIVATE ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})

    target_link_libraries(ggml PUBLIC ggml-metal)
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_METAL=1)
else()
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_METAL=0)
endif()
