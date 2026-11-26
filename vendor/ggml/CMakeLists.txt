cmake_minimum_required(VERSION 3.18)
project(vendor_ggml LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(OPENGP_CUDA  "Enable CUDA GPU backend if CUDA Toolkit is available" ON)
option(OPENGP_METAL "Enable Apple Metal GPU backend on macOS" ON)
set(OPENGP_VULKAN "AUTO" CACHE STRING "Vulkan GPU backend: AUTO (enabled when CUDA/Metal absent and Vulkan SDK + glslc found), ON, OFF")
set_property(CACHE OPENGP_VULKAN PROPERTY STRINGS AUTO ON OFF)

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
    set(_OPENGP_CUDA_ENABLED TRUE)
else()
    message(STATUS "OpenGP: CUDA Toolkit not available or disabled — ggml-cuda disabled")
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_CUDA=0)
    set(_OPENGP_CUDA_ENABLED FALSE)
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
    set(_OPENGP_METAL_ENABLED TRUE)
else()
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_METAL=0)
    set(_OPENGP_METAL_ENABLED FALSE)
endif()

# Optional Vulkan Backend (AMD / Intel / integrated GPUs; strictly soft-fail, skipped in AUTO when CUDA/Metal enabled)
string(TOUPPER "${OPENGP_VULKAN}" _OPENGP_VULKAN_MODE)
set(_OPENGP_VULKAN_ENABLED FALSE)
set(_OPENGP_VULKAN_TRY FALSE)
set(_OPENGP_VULKAN_EXPLICIT FALSE)

if (_OPENGP_VULKAN_MODE STREQUAL "OFF" OR _OPENGP_VULKAN_MODE STREQUAL "0" OR
    _OPENGP_VULKAN_MODE STREQUAL "FALSE" OR _OPENGP_VULKAN_MODE STREQUAL "NO")
    message(STATUS "OpenGP: Vulkan backend disabled (OPENGP_VULKAN=${OPENGP_VULKAN})")
elseif (_OPENGP_VULKAN_MODE STREQUAL "AUTO")
    if (APPLE)
        message(STATUS "OpenGP: macOS detected — skipping ggml-vulkan in AUTO mode (Apple Metal preferred)")
    elseif (_OPENGP_CUDA_ENABLED)
        message(STATUS "OpenGP: CUDA enabled — skipping ggml-vulkan in AUTO mode (pass -DOPENGP_VULKAN=ON to build both)")
    else()
        set(_OPENGP_VULKAN_TRY TRUE)
    endif()
elseif (_OPENGP_VULKAN_MODE STREQUAL "ON" OR _OPENGP_VULKAN_MODE STREQUAL "1" OR
        _OPENGP_VULKAN_MODE STREQUAL "TRUE" OR _OPENGP_VULKAN_MODE STREQUAL "YES")
    set(_OPENGP_VULKAN_TRY TRUE)
    set(_OPENGP_VULKAN_EXPLICIT TRUE)
else()
    message(WARNING "OpenGP: Unrecognized OPENGP_VULKAN='${OPENGP_VULKAN}' (expected AUTO, ON, or OFF) — disabling ggml-vulkan")
endif()

function(_opengp_vk_skip MSG)
    if (_OPENGP_VULKAN_EXPLICIT)
        message(WARNING "OpenGP: ${MSG} — continuing without ggml-vulkan")
    else()
        message(STATUS "OpenGP: ${MSG} — ggml-vulkan disabled")
    endif()
endfunction()

function(_opengp_vk_test_shader_ext GLSLC_BIN EXT_NAME TEST_FILE RESULT_VAR OUT_DEFS_VAR)
    execute_process(
        COMMAND "${GLSLC_BIN}" -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_FILE}"
        OUTPUT_QUIET
        ERROR_VARIABLE _glslc_err
        RESULT_VARIABLE _glslc_rc
    )
    if (_glslc_rc EQUAL 0 AND NOT _glslc_err MATCHES "extension not supported: ${EXT_NAME}")
        message(STATUS "OpenGP Vulkan: ${EXT_NAME} supported by glslc")
        set(_defs "${${OUT_DEFS_VAR}}")
        list(APPEND _defs "${RESULT_VAR}")
        set(${OUT_DEFS_VAR} "${_defs}" PARENT_SCOPE)
    else()
        message(STATUS "OpenGP Vulkan: ${EXT_NAME} not supported by glslc")
    endif()
endfunction()

if (_OPENGP_VULKAN_TRY)
    find_package(Vulkan QUIET)
    if (NOT Vulkan_FOUND)
        _opengp_vk_skip("Vulkan SDK/loader not found")
    else()
        # Locate glslc shader compiler (honor explicit -DGLSLC=... override if provided)
        if (DEFINED GLSLC)
            set(_OPENGP_GLSLC "${GLSLC}")
        elseif (Vulkan_GLSLC_EXECUTABLE AND EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
            set(_OPENGP_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
        else()
            find_program(_OPENGP_GLSLC_FOUND NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin")
            set(_OPENGP_GLSLC "${_OPENGP_GLSLC_FOUND}")
        endif()

        # Verify glslc exists and is runnable
        set(_OPENGP_GLSLC_OK FALSE)
        if (_OPENGP_GLSLC AND EXISTS "${_OPENGP_GLSLC}")
            execute_process(
                COMMAND "${_OPENGP_GLSLC}" --version
                OUTPUT_QUIET
                ERROR_QUIET
                RESULT_VARIABLE _glslc_ver_rc
            )
            if (_glslc_ver_rc EQUAL 0)
                set(_OPENGP_GLSLC_OK TRUE)
            endif()
        endif()

        # Verify Vulkan-Hpp C++ header (vulkan/vulkan.hpp) is available
        find_path(_OPENGP_VULKAN_HPP_DIR
            NAMES vulkan/vulkan.hpp
            HINTS ${Vulkan_INCLUDE_DIRS} "$ENV{VULKAN_SDK}/include"
        )

        if (NOT _OPENGP_GLSLC_OK)
            _opengp_vk_skip("glslc shader compiler not found or not runnable ('${_OPENGP_GLSLC}')")
        elseif (NOT _OPENGP_VULKAN_HPP_DIR)
            _opengp_vk_skip("vulkan/vulkan.hpp C++ header not found")
        else()
            set(_OPENGP_VULKAN_ENABLED TRUE)
        endif()
    endif()
endif()

if (_OPENGP_VULKAN_ENABLED)
    message(STATUS "OpenGP: Vulkan ${Vulkan_VERSION} + glslc (${_OPENGP_GLSLC}) found — enabling ggml-vulkan backend")

    set(VK_DIR         "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-vulkan")
    set(VK_SHADER_DIR  "${VK_DIR}/vulkan-shaders")
    set(VK_GEN_DIR     "${CMAKE_CURRENT_BINARY_DIR}/vulkan_autogenerated")
    set(VK_SPV_DIR     "${VK_GEN_DIR}/spv")
    file(MAKE_DIRECTORY "${VK_SPV_DIR}")

    set(VK_FEATURE_DEFS "")
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_KHR_cooperative_matrix"              "${VK_SHADER_DIR}/feature-tests/coopmat.comp"              "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"              VK_FEATURE_DEFS)
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_NV_cooperative_matrix2"              "${VK_SHADER_DIR}/feature-tests/coopmat2.comp"             "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"             VK_FEATURE_DEFS)
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_NV_cooperative_matrix_decode_vector" "${VK_SHADER_DIR}/feature-tests/coopmat2_decode_vector.comp" "GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT" VK_FEATURE_DEFS)
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_EXT_integer_dot_product"             "${VK_SHADER_DIR}/feature-tests/integer_dot.comp"          "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"          VK_FEATURE_DEFS)
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_EXT_bfloat16"                        "${VK_SHADER_DIR}/feature-tests/bfloat16.comp"             "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"             VK_FEATURE_DEFS)
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_EXT_float_e2m1"                      "${VK_SHADER_DIR}/feature-tests/float_e2m1.comp"           "GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT"           VK_FEATURE_DEFS)
    _opengp_vk_test_shader_ext("${_OPENGP_GLSLC}" "GL_EXT_float_e4m3"                      "${VK_SHADER_DIR}/feature-tests/float_e4m3.comp"           "GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT"           VK_FEATURE_DEFS)

    # Host tool that invokes glslc and embeds compiled SPIR-V shaders into C++ translation units
    add_executable(opengp-vulkan-shaders-gen
        "${VK_SHADER_DIR}/vulkan-shaders-gen.cpp"
    )
    target_compile_features(opengp-vulkan-shaders-gen PRIVATE cxx_std_17)
    target_link_libraries(opengp-vulkan-shaders-gen PRIVATE Threads::Threads)
    target_compile_definitions(opengp-vulkan-shaders-gen PRIVATE ${VK_FEATURE_DEFS})

    file(GLOB _VK_SHADER_COMP_FILES CONFIGURE_DEPENDS "${VK_SHADER_DIR}/*.comp")
    file(GLOB _VK_SHADER_GLSL_HEADERS CONFIGURE_DEPENDS "${VK_SHADER_DIR}/*.glsl")

    set(_VK_GEN_HPP "${VK_GEN_DIR}/ggml-vulkan-shaders.hpp")
    add_custom_command(
        OUTPUT "${_VK_GEN_HPP}"
        COMMAND opengp-vulkan-shaders-gen
            --output-dir "${VK_SPV_DIR}"
            --target-hpp "${_VK_GEN_HPP}"
        DEPENDS opengp-vulkan-shaders-gen
        COMMENT "Generating Vulkan shaders header"
        VERBATIM
    )

    set(_VK_GEN_CPP_FILES "")
    foreach (_shader_full ${_VK_SHADER_COMP_FILES})
        get_filename_component(_shader_name "${_shader_full}" NAME)
        set(_shader_cpp "${VK_GEN_DIR}/${_shader_name}.cpp")
        add_custom_command(
            OUTPUT "${_shader_cpp}"
            COMMAND opengp-vulkan-shaders-gen
                --glslc      "${_OPENGP_GLSLC}"
                --source     "${_shader_full}"
                --output-dir "${VK_SPV_DIR}"
                --target-hpp "${_VK_GEN_HPP}"
                --target-cpp "${_shader_cpp}"
            DEPENDS opengp-vulkan-shaders-gen "${_VK_GEN_HPP}" "${_shader_full}" ${_VK_SHADER_GLSL_HEADERS}
            COMMENT "Compiling Vulkan shaders for ${_shader_name}"
            VERBATIM
        )
        list(APPEND _VK_GEN_CPP_FILES "${_shader_cpp}")
    endforeach()

    add_library(ggml-vulkan STATIC
        "${VK_DIR}/ggml-vulkan.cpp"
        "${VK_DIR}/ggml-vulkan-buffers.cpp"
        "${VK_DIR}/ggml-vulkan-comm.cpp"
        "${VK_DIR}/ggml-vulkan-debug.cpp"
        "${_VK_GEN_HPP}"
        ${_VK_GEN_CPP_FILES}
    )

    target_include_directories(ggml-vulkan PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${VK_DIR}"
        "${VK_GEN_DIR}"
        "${VK_DIR}/third_party/spirv-headers/include"
        "${_OPENGP_VULKAN_HPP_DIR}"
    )

    target_compile_definitions(ggml-vulkan PRIVATE ${VK_FEATURE_DEFS})
    target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)

    target_link_libraries(ggml PUBLIC ggml-vulkan)
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_VULKAN=1)
else()
    target_compile_definitions(ggml PUBLIC OPENGP_HAS_VULKAN=0)
endif()

if (NOT _OPENGP_CUDA_ENABLED AND NOT _OPENGP_METAL_ENABLED AND NOT _OPENGP_VULKAN_ENABLED)
    message(STATUS "OpenGP: No GPU backend enabled — building CPU-only backend")
endif()
