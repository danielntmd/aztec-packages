barretenberg_module(ecc numeric crypto_keccak crypto_sha256)
set_target_properties(ecc_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DISABLE_ADX)
    message(STATUS "Disabling ADX assembly variant.")
    target_compile_definitions(ecc PRIVATE -DDISABLE_ADX)
endif()

# GPU MSM adapter (scalar_multiplication/gpu_msm.cpp) is picked up by the
# barretenberg_module glob of *.cpp. The file is guarded internally by
# `#ifdef BB_GPU_NATIVE` so it compiles to an empty translation unit when
# GPU_BACKEND=none (cost: one extra empty compile).
#
# When native CUDA kernel sources (.cu) are added, list them explicitly here
# since the barretenberg_module glob picks up .cpp only.
if(GPU_BACKEND STREQUAL "native")
    # Placeholder: uncomment and extend once .cu kernels land.
    # target_sources(ecc_objects PRIVATE
    #     scalar_multiplication/gpu_msm_kernels.cu)
endif()

target_precompile_headers(
    ecc
    PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:<random$<ANGLE-R>>"
    $<$<COMPILE_LANGUAGE:CXX>:"barretenberg/numeric/uint256/uint256.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"barretenberg/numeric/uint256/uint256_impl.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/asm_macros.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_declarations.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_impl.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_impl_generic.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_impl_x64.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field.hpp">
)
