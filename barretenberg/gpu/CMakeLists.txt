find_package(CUDAToolkit REQUIRED)

set(BB_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(BB_GPU_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src")

set(BB_GPU_HEADERS
    include/barretenberg/gpu/common/cuda_defines.cuh
    include/barretenberg/gpu/common/cuda_error.cuh
    include/barretenberg/gpu/common/cub_helpers.cuh
    include/barretenberg/gpu/common/device_buffer.hpp
    include/barretenberg/gpu/common/gpu_msm_context.hpp
    include/barretenberg/gpu/common/nvtx.hpp
    include/barretenberg/gpu/commitment_schemes/commitment_key_msm.hpp
    include/barretenberg/gpu/fields/field.cuh
    include/barretenberg/gpu/fields/bn254/params.cuh
    include/barretenberg/gpu/fields/bn254/fq32.cuh
    include/barretenberg/gpu/fields/bn254/fr32.cuh
    include/barretenberg/gpu/curves/bn254/bn254.cuh
    include/barretenberg/gpu/curves/bn254/fq32_g1.cuh
    include/barretenberg/gpu/msm/msm.hpp
    include/barretenberg/gpu/msm/msm_profile.cuh
    include/barretenberg/gpu/msm/msm_raw.cuh)

if(TARGET barretenberg_headers)
    target_sources(barretenberg_headers INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${BB_GPU_INCLUDE_DIR}"
        FILES ${BB_GPU_HEADERS})
endif()

add_library(gpu_objects OBJECT EXCLUDE_FROM_ALL
    src/common/gpu_msm_context.cu
    src/msm/msm.cpp
    src/msm/msm.cu)
set_target_properties(gpu_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gpu_objects
    PUBLIC "${BB_GPU_INCLUDE_DIR}"
    PRIVATE
        "${BB_GPU_CPP_SOURCE_DIR}"
        ${MSGPACK_INCLUDE}
        ${TRACY_INCLUDE}
        ${LMDB_INCLUDE}
        ${LIBDEFLATE_INCLUDE}
        ${HTTPLIB_INCLUDE}
        ${BACKWARD_INCLUDE}
        ${NLOHMANN_JSON_INCLUDE})

add_library(gpu STATIC $<TARGET_OBJECTS:gpu_objects>)
target_include_directories(gpu
    PUBLIC "${BB_GPU_INCLUDE_DIR}"
    PRIVATE
        "${BB_GPU_CPP_SOURCE_DIR}"
        ${MSGPACK_INCLUDE}
        ${TRACY_INCLUDE}
        ${LMDB_INCLUDE}
        ${LIBDEFLATE_INCLUDE}
        ${HTTPLIB_INCLUDE}
        ${BACKWARD_INCLUDE}
        ${NLOHMANN_JSON_INCLUDE})
find_library(BB_GPU_STDCXX_LIBRARY
    NAMES stdc++ libstdc++.so.6
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib/gcc/x86_64-linux-gnu/11
    NO_DEFAULT_PATH
    REQUIRED)

target_link_libraries(gpu
    PUBLIC
        common
        ecc
        polynomials
        CUDA::cudart
        ${BB_GPU_STDCXX_LIBRARY})

if(TARGET msgpack-c)
    add_dependencies(gpu_objects msgpack-c)
    add_dependencies(gpu msgpack-c)
endif()

add_library(gpu_test_objects OBJECT
    tests/bn254.test.cpp
    tests/bn254_test_kernels.cu)
target_include_directories(gpu_test_objects
    PRIVATE
        "${BB_GPU_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${BB_GPU_CPP_SOURCE_DIR}"
        ${MSGPACK_INCLUDE}
        ${TRACY_INCLUDE}
        ${LMDB_INCLUDE}
        ${LIBDEFLATE_INCLUDE}
        ${HTTPLIB_INCLUDE}
        ${BACKWARD_INCLUDE}
        ${NLOHMANN_JSON_INCLUDE})
target_link_libraries(gpu_test_objects
    PRIVATE
        gpu
        GTest::gtest
        GTest::gmock_main)

add_executable(gpu_tests $<TARGET_OBJECTS:gpu_test_objects>)
target_link_libraries(gpu_tests
    PRIVATE
        gpu
        GTest::gtest
        GTest::gtest_main
        GTest::gmock_main
        ${BB_GPU_STDCXX_LIBRARY})

if(TARGET msgpack-c)
    add_dependencies(gpu_test_objects msgpack-c)
    add_dependencies(gpu_tests msgpack-c)
endif()

if(NOT FUZZING)
    add_executable(gpu_msm_bench
        benchmark/msm_latency.bench.cpp)
    target_include_directories(gpu_msm_bench
        PRIVATE
            "${BB_GPU_INCLUDE_DIR}"
            "${BB_GPU_CPP_SOURCE_DIR}"
            ${MSGPACK_INCLUDE}
            ${TRACY_INCLUDE}
            ${LMDB_INCLUDE}
            ${LIBDEFLATE_INCLUDE}
            ${HTTPLIB_INCLUDE}
            ${BACKWARD_INCLUDE}
            ${NLOHMANN_JSON_INCLUDE})
    target_link_libraries(gpu_msm_bench
        PRIVATE
            gpu
            commitment_schemes
            benchmark::benchmark
            CUDA::cudart
            ${BB_GPU_STDCXX_LIBRARY})
    if(TARGET msgpack-c)
        add_dependencies(gpu_msm_bench msgpack-c)
    endif()
endif()

if(TARGET commitment_schemes)
    target_link_libraries(commitment_schemes PUBLIC gpu)
endif()

foreach(BB_GPU_ARCHIVE_TARGET barretenberg bb-external)
    if(TARGET ${BB_GPU_ARCHIVE_TARGET})
        target_sources(${BB_GPU_ARCHIVE_TARGET} PRIVATE $<TARGET_OBJECTS:gpu_objects>)
        target_link_libraries(${BB_GPU_ARCHIVE_TARGET} PUBLIC CUDA::cudart ${BB_GPU_STDCXX_LIBRARY})
        add_dependencies(${BB_GPU_ARCHIVE_TARGET} gpu_objects)
    endif()
endforeach()

if(NOT WASM)
    gtest_discover_tests(gpu_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TEST_FILTER -*_SKIP_CI* DISCOVERY_TIMEOUT 30)
endif()
