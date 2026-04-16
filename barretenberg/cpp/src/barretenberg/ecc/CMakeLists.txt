barretenberg_module(ecc numeric crypto_keccak crypto_sha256)
set_target_properties(ecc_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DISABLE_ADX)
    message(STATUS "Disabling ADX assembly variant.")
    target_compile_definitions(ecc PRIVATE -DDISABLE_ADX)
endif()

# GPU MSM adapter (scalar_multiplication/gpu_msm.cpp) is picked up by the
# barretenberg_module glob of *.cpp. The file is guarded internally by
# `#ifdef BB_GPU_ICICLE` so it compiles to an empty translation unit when
# GPU_BACKEND=none (cost: one extra empty compile).
#
# When GPU_BACKEND=icicle, link against the actual ICICLE targets emitted by
# its CMake: the runtime plus the BN254 field/curve libraries.
if(GPU_BACKEND STREQUAL "icicle")
    foreach(icicle_target icicle_device icicle_field icicle_curve)
        if(NOT TARGET ${icicle_target})
            message(FATAL_ERROR "Expected ICICLE target '${icicle_target}' when GPU_BACKEND=icicle")
        endif()
    endforeach()
    target_link_libraries(ecc PUBLIC icicle_device icicle_field icicle_curve)

    # Bake the ICICLE backend library directory into the binary so the runtime
    # loader can find it without requiring ICICLE_BACKEND_INSTALL_DIR to be set.
    # The env var still takes precedence when present (checked first by ICICLE).
    get_target_property(_icicle_device_dir icicle_device BINARY_DIR)
    target_compile_definitions(ecc_objects PRIVATE
        BB_ICICLE_BUILD_BACKEND_DIR="${_icicle_device_dir}")
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
