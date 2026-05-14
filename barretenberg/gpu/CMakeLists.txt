find_package(CUDAToolkit REQUIRED)

set(BB_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(BB_GPU_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src")
set(ICICLE_V28_ROOT "" CACHE PATH "Optional Icicle v2.8 install/build root for GPU MSM baseline benchmarks")
set(ICICLE_V28_LIBRARIES "" CACHE STRING "Optional semicolon-separated Icicle v2.8 libraries for GPU MSM baseline benchmarks")
set(ICICLE_V4_ROOT "" CACHE PATH "Optional Icicle v4 install/build root for GPU MSM baseline benchmarks")
set(ICICLE_V4_LIBRARIES "" CACHE STRING "Optional semicolon-separated Icicle v4 frontend libraries for GPU MSM baseline benchmarks")
set(ICICLE_V4_BACKEND_DIR "" CACHE PATH "Optional Icicle v4 CUDA backend directory loaded by the baseline benchmark")

set(BB_GPU_HEADERS
    include/barretenberg/gpu/common/cuda_defines.cuh
    include/barretenberg/gpu/common/cuda_error.cuh
    include/barretenberg/gpu/common/cub_helpers.cuh
    include/barretenberg/gpu/common/device_buffer.hpp
    include/barretenberg/gpu/common/device_context.hpp
    include/barretenberg/gpu/commitment_schemes/commitment_key_msm.hpp
    include/barretenberg/gpu/fields/field.cuh
    include/barretenberg/gpu/curves/bn254/params.cuh
    include/barretenberg/gpu/curves/bn254/fq.cuh
    include/barretenberg/gpu/curves/bn254/fr.cuh
    include/barretenberg/gpu/curves/bn254/g1.cuh
    include/barretenberg/gpu/curves/bn254/bn254.cuh
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
    src/common/device_context.cu
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

    set(BB_GPU_BENCHMARK_COMMON_SOURCES
        benchmark/msm_benchmark_common.cpp)

    function(bb_gpu_configure_benchmark_target target_name)
        target_include_directories(${target_name}
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmark"
                "${BB_GPU_INCLUDE_DIR}"
                "${BB_GPU_CPP_SOURCE_DIR}"
                ${MSGPACK_INCLUDE}
                ${TRACY_INCLUDE}
                ${LMDB_INCLUDE}
                ${LIBDEFLATE_INCLUDE}
                ${HTTPLIB_INCLUDE}
                ${BACKWARD_INCLUDE}
                ${NLOHMANN_JSON_INCLUDE})
        target_link_libraries(${target_name}
            PRIVATE
                commitment_schemes
                benchmark::benchmark
                CUDA::cudart
                ${BB_GPU_STDCXX_LIBRARY})
        if(TARGET msgpack-c)
            add_dependencies(${target_name} msgpack-c)
        endif()
    endfunction()

    function(bb_gpu_collect_icicle_libraries root cache_libraries out_libraries)
        if(cache_libraries)
            set(${out_libraries} ${cache_libraries} PARENT_SCOPE)
            return()
        endif()

        file(GLOB discovered_icicle_libraries CONFIGURE_DEPENDS
            "${root}/lib/*.so"
            "${root}/lib/*.a"
            "${root}/lib64/*.so"
            "${root}/lib64/*.a"
            "${root}/build/lib/*.so"
            "${root}/build/lib/*.a"
            "${root}/build/lib64/*.so"
            "${root}/build/lib64/*.a")
        set(${out_libraries} ${discovered_icicle_libraries} PARENT_SCOPE)
    endfunction()

    add_executable(gpu_msm_baseline_bench
        ${BB_GPU_BENCHMARK_COMMON_SOURCES}
        benchmark/msm_baseline_bb.bench.cpp)
    bb_gpu_configure_benchmark_target(gpu_msm_baseline_bench)
    target_link_libraries(gpu_msm_baseline_bench PRIVATE gpu)

    if(ICICLE_V28_ROOT)
        bb_gpu_collect_icicle_libraries("${ICICLE_V28_ROOT}" "${ICICLE_V28_LIBRARIES}" ICICLE_V28_LINK_LIBRARIES)
        if(ICICLE_V28_LINK_LIBRARIES)
            add_executable(gpu_msm_baseline_icicle_v28_bench
                ${BB_GPU_BENCHMARK_COMMON_SOURCES}
                benchmark/msm_baseline_icicle_v28.bench.cpp
                benchmark/msm_baseline_icicle_v28_adapter.cu)
            bb_gpu_configure_benchmark_target(gpu_msm_baseline_icicle_v28_bench)
            target_include_directories(gpu_msm_baseline_icicle_v28_bench
                PRIVATE
                    "${ICICLE_V28_ROOT}/include"
                    "${ICICLE_V28_ROOT}/icicle/include")
            target_link_libraries(gpu_msm_baseline_icicle_v28_bench PRIVATE ${ICICLE_V28_LINK_LIBRARIES})
            set_target_properties(gpu_msm_baseline_icicle_v28_bench PROPERTIES
                BUILD_RPATH "${ICICLE_V28_ROOT}/lib;${ICICLE_V28_ROOT}/lib64;${ICICLE_V28_ROOT}/build/lib;${ICICLE_V28_ROOT}/build/lib64")
        else()
            message(WARNING "ICICLE_V28_ROOT is set, but no Icicle v2.8 libraries were found. Set ICICLE_V28_LIBRARIES to build gpu_msm_baseline_icicle_v28_bench.")
        endif()
    endif()

    if(ICICLE_V4_ROOT)
        bb_gpu_collect_icicle_libraries("${ICICLE_V4_ROOT}" "${ICICLE_V4_LIBRARIES}" ICICLE_V4_LINK_LIBRARIES)
        if(ICICLE_V4_LINK_LIBRARIES)
            add_executable(gpu_msm_baseline_icicle_v4_bench
                ${BB_GPU_BENCHMARK_COMMON_SOURCES}
                benchmark/msm_baseline_icicle_v4.bench.cpp)
            bb_gpu_configure_benchmark_target(gpu_msm_baseline_icicle_v4_bench)
            target_include_directories(gpu_msm_baseline_icicle_v4_bench
                PRIVATE
                    "${ICICLE_V4_ROOT}/include")
            target_link_libraries(gpu_msm_baseline_icicle_v4_bench PRIVATE ${ICICLE_V4_LINK_LIBRARIES})
            set_target_properties(gpu_msm_baseline_icicle_v4_bench PROPERTIES
                BUILD_RPATH "${ICICLE_V4_ROOT}/lib;${ICICLE_V4_ROOT}/lib64")
            if(ICICLE_V4_BACKEND_DIR)
                target_compile_definitions(gpu_msm_baseline_icicle_v4_bench
                    PRIVATE
                        BB_GPU_ICICLE_V4_BACKEND_DIR="${ICICLE_V4_BACKEND_DIR}")
            endif()
        else()
            message(WARNING "ICICLE_V4_ROOT is set, but no Icicle v4 libraries were found. Set ICICLE_V4_LIBRARIES to build gpu_msm_baseline_icicle_v4_bench.")
        endif()
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
