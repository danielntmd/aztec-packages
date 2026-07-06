find_package(CUDAToolkit QUIET)

set(BB_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(BB_GPU_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(BB_GPU_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src")
set(BB_GPU_EXTERNAL_MSM_BENCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/external_msm")
option(BB_ENABLE_GPU_MSM_EXTERNAL_BENCH "Build optional external GPU MSM comparison benchmark runners" OFF)
set(BB_GPU_MSM_FIELD_BACKEND "bb" CACHE STRING "BN254 GPU MSM field arithmetic backend")
set_property(CACHE BB_GPU_MSM_FIELD_BACKEND PROPERTY STRINGS bb icicle)
set(BB_GPU_ICICLE_INCLUDE_DIRS "" CACHE STRING "Icicle include directories for the optional GPU MSM field backend")
find_library(BB_GPU_STDCXX_LIBRARY
    NAMES stdc++ libstdc++.so.6
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib/gcc/x86_64-linux-gnu/11
    NO_DEFAULT_PATH
    REQUIRED)
if(NOT BB_GPU_MSM_FIELD_BACKEND MATCHES "^(bb|icicle)$")
    message(FATAL_ERROR "BB_GPU_MSM_FIELD_BACKEND must be one of: bb, icicle")
endif()
if(BB_GPU_MSM_FIELD_BACKEND STREQUAL "icicle")
    if(NOT BB_GPU_ICICLE_INCLUDE_DIRS AND ICICLE_V4_INCLUDE_DIR)
        set(BB_GPU_ICICLE_INCLUDE_DIRS "${ICICLE_V4_INCLUDE_DIR}" CACHE STRING
            "Icicle include directories for the optional GPU MSM field backend" FORCE)
    endif()
    if(NOT BB_GPU_ICICLE_INCLUDE_DIRS AND ICICLE_V4_INSTALL_DIR)
        set(BB_GPU_ICICLE_INCLUDE_DIRS "${ICICLE_V4_INSTALL_DIR}/include" CACHE STRING
            "Icicle include directories for the optional GPU MSM field backend" FORCE)
    endif()
    if(NOT BB_GPU_ICICLE_INCLUDE_DIRS)
        message(FATAL_ERROR "Set BB_GPU_ICICLE_INCLUDE_DIRS or ICICLE_V4_INCLUDE_DIR when "
                            "BB_GPU_MSM_FIELD_BACKEND=icicle")
    endif()
endif()

function(bb_gpu_apply_msm_field_backend TARGET_NAME)
    if(BB_GPU_MSM_FIELD_BACKEND STREQUAL "icicle")
        target_compile_definitions(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:BB_GPU_USE_ICICLE_FIELD_KERNELS>)
        target_include_directories(${TARGET_NAME} PRIVATE ${BB_GPU_ICICLE_INCLUDE_DIRS})
    endif()
endfunction()

function(bb_gpu_add_external_cpu_msm_bench)
    add_executable(gpu_msm_external_cpu_bench
        benchmark/external_msm/cpu_msm_runner.cpp)
    target_include_directories(gpu_msm_external_cpu_bench
        PRIVATE
            "${BB_GPU_EXTERNAL_MSM_BENCH_INCLUDE_DIR}"
            "${BB_GPU_CPP_SOURCE_DIR}"
            ${MSGPACK_INCLUDE}
            ${TRACY_INCLUDE}
            ${LMDB_INCLUDE}
            ${LIBDEFLATE_INCLUDE}
            ${HTTPLIB_INCLUDE}
            ${BACKWARD_INCLUDE}
            ${NLOHMANN_JSON_INCLUDE})
    target_link_libraries(gpu_msm_external_cpu_bench
        PRIVATE
            ecc
            srs
            common
            ${BB_GPU_STDCXX_LIBRARY})
    if(TARGET msgpack-c)
        add_dependencies(gpu_msm_external_cpu_bench msgpack-c)
    endif()
endfunction()

set(BB_GPU_HEADERS
    include/barretenberg/gpu/common/cuda_defines.cuh
    include/barretenberg/gpu/common/cuda_error.hpp
    include/barretenberg/gpu/common/cub_helpers.cuh
    include/barretenberg/gpu/common/device_buffer_pool.hpp
    include/barretenberg/gpu/common/device_buffer.hpp
    include/barretenberg/gpu/common/device_span.hpp
    include/barretenberg/gpu/common/nvtx.hpp
    include/barretenberg/gpu/commitment_schemes/commitment_key_msm.hpp
    include/barretenberg/gpu/fields/field.cuh
    include/barretenberg/gpu/fields/bn254/params.cuh
    include/barretenberg/gpu/fields/bn254/fq32.cuh
    include/barretenberg/gpu/fields/bn254/icicle_fq32.cuh
    include/barretenberg/gpu/fields/bn254/fr32.cuh
    include/barretenberg/gpu/curves/bn254/bn254.cuh
    include/barretenberg/gpu/curves/bn254/fq32_g1.cuh
    include/barretenberg/gpu/backend.hpp)

if(TARGET barretenberg_headers)
    target_sources(barretenberg_headers INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${BB_GPU_INCLUDE_DIR}"
        FILES ${BB_GPU_HEADERS})
endif()

if(NOT CUDAToolkit_FOUND)
    if(NOT FUZZING AND BB_ENABLE_GPU_MSM_EXTERNAL_BENCH)
        bb_gpu_add_external_cpu_msm_bench()
    endif()
    message(STATUS "CUDAToolkit not found; skipping native CUDA GPU targets")
    return()
endif()

if(NOT GPU_BACKEND STREQUAL "native")
    if(NOT FUZZING AND BB_ENABLE_GPU_MSM_EXTERNAL_BENCH)
        bb_gpu_add_external_cpu_msm_bench()
    endif()
    message(STATUS "GPU_BACKEND is not native; skipping native CUDA GPU targets")
    return()
endif()

add_library(gpu_objects OBJECT EXCLUDE_FROM_ALL
    src/common/cuda_error.cpp
    src/common/cuda_stream.cu
    src/common/device_memory.cu
    src/common/gpu_msm_context.cu
    src/msm/bn254_msm.cpp
    src/msm/bn254_msm.cu)
set_target_properties(gpu_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gpu_objects
    PUBLIC "${BB_GPU_INCLUDE_DIR}"
    PRIVATE
        "${BB_GPU_SRC_DIR}"
        "${BB_GPU_CPP_SOURCE_DIR}"
        ${CUDAToolkit_INCLUDE_DIRS}
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
        "${BB_GPU_SRC_DIR}"
        "${BB_GPU_CPP_SOURCE_DIR}"
        ${MSGPACK_INCLUDE}
        ${TRACY_INCLUDE}
        ${LMDB_INCLUDE}
        ${LIBDEFLATE_INCLUDE}
        ${HTTPLIB_INCLUDE}
        ${BACKWARD_INCLUDE}
        ${NLOHMANN_JSON_INCLUDE})
target_link_libraries(gpu
    PUBLIC
        common
        ecc
        polynomials
        CUDA::cudart
        ${BB_GPU_STDCXX_LIBRARY})
bb_gpu_apply_msm_field_backend(gpu_objects)
bb_gpu_apply_msm_field_backend(gpu)

if(TARGET msgpack-c)
    add_dependencies(gpu_objects msgpack-c)
    add_dependencies(gpu msgpack-c)
endif()

add_library(gpu_test_objects OBJECT
    tests/bn254_field.test.cpp
    tests/bn254_g1.test.cpp
    tests/bn254_test_kernels.cu
    tests/msm.test.cpp
    tests/msm_perf_smoke.test.cpp
    tests/device_buffer.test.cpp
    tests/msm_heuristics.test.cpp)
target_include_directories(gpu_test_objects
    PRIVATE
        "${BB_GPU_INCLUDE_DIR}"
        "${BB_GPU_SRC_DIR}"
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
bb_gpu_apply_msm_field_backend(gpu_test_objects)

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
            "${BB_GPU_SRC_DIR}"
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
    bb_gpu_apply_msm_field_backend(gpu_msm_bench)
    if(TARGET msgpack-c)
        add_dependencies(gpu_msm_bench msgpack-c)
    endif()

    if(BB_ENABLE_GPU_MSM_EXTERNAL_BENCH)
        bb_gpu_add_external_cpu_msm_bench()

        add_executable(gpu_msm_external_bb_bench
            benchmark/external_msm/bb_msm_runner.cpp)
        target_include_directories(gpu_msm_external_bb_bench
            PRIVATE
                "${BB_GPU_EXTERNAL_MSM_BENCH_INCLUDE_DIR}"
                "${BB_GPU_INCLUDE_DIR}"
                "${BB_GPU_SRC_DIR}"
                "${BB_GPU_CPP_SOURCE_DIR}"
                ${MSGPACK_INCLUDE}
                ${TRACY_INCLUDE}
                ${LMDB_INCLUDE}
                ${LIBDEFLATE_INCLUDE}
                ${HTTPLIB_INCLUDE}
                ${BACKWARD_INCLUDE}
                ${NLOHMANN_JSON_INCLUDE})
        target_link_libraries(gpu_msm_external_bb_bench
            PRIVATE
                gpu
                commitment_schemes
                srs
                CUDA::cudart
                ${BB_GPU_STDCXX_LIBRARY})
        bb_gpu_apply_msm_field_backend(gpu_msm_external_bb_bench)
        if(TARGET msgpack-c)
            add_dependencies(gpu_msm_external_bb_bench msgpack-c)
        endif()

        set(ICICLE_V28_INSTALL_DIR "" CACHE PATH "Icicle v2.8.0 install directory")
        set(ICICLE_V28_INCLUDE_DIR "" CACHE PATH "Icicle v2.8.0 include directory")
        set(ICICLE_V28_LIBRARY_DIR "" CACHE PATH "Icicle v2.8.0 library directory")
        set(ICICLE_V28_LIBRARIES "" CACHE STRING "Icicle v2.8.0 libraries; overrides auto-discovery")
        if(NOT ICICLE_V28_INCLUDE_DIR AND ICICLE_V28_INSTALL_DIR)
            set(ICICLE_V28_INCLUDE_DIR "${ICICLE_V28_INSTALL_DIR}/include")
        endif()
        if(NOT ICICLE_V28_LIBRARY_DIR AND ICICLE_V28_INSTALL_DIR)
            set(ICICLE_V28_LIBRARY_DIR "${ICICLE_V28_INSTALL_DIR}/lib")
        endif()
        if(ICICLE_V28_INCLUDE_DIR)
            set(ICICLE_V28_INCLUDE_DIRS "${ICICLE_V28_INCLUDE_DIR}")
        endif()
        if(ICICLE_V28_LIBRARY_DIR AND NOT ICICLE_V28_LIBRARIES)
            find_library(ICICLE_V28_CURVE_LIBRARY
                NAMES ingo_curve_bn254
                PATHS "${ICICLE_V28_LIBRARY_DIR}"
                NO_DEFAULT_PATH)
            find_library(ICICLE_V28_FIELD_LIBRARY
                NAMES ingo_field_bn254_scalar ingo_field_bn254
                PATHS "${ICICLE_V28_LIBRARY_DIR}"
                NO_DEFAULT_PATH)
            if(ICICLE_V28_CURVE_LIBRARY AND ICICLE_V28_FIELD_LIBRARY)
                set(ICICLE_V28_LIBRARIES "${ICICLE_V28_CURVE_LIBRARY};${ICICLE_V28_FIELD_LIBRARY}")
            endif()
        endif()
        if(ICICLE_V28_INCLUDE_DIRS AND ICICLE_V28_LIBRARIES)
            add_executable(gpu_msm_external_icicle_v2_bench
                benchmark/external_msm/icicle_v2_msm_adapter.cu
                benchmark/external_msm/icicle_v2_msm_runner.cpp)
            target_include_directories(gpu_msm_external_icicle_v2_bench
                PRIVATE
                    "${BB_GPU_EXTERNAL_MSM_BENCH_INCLUDE_DIR}"
                    "${BB_GPU_CPP_SOURCE_DIR}"
                    ${MSGPACK_INCLUDE}
                    ${TRACY_INCLUDE}
                    ${LMDB_INCLUDE}
                    ${LIBDEFLATE_INCLUDE}
                    ${HTTPLIB_INCLUDE}
                    ${BACKWARD_INCLUDE}
                    ${NLOHMANN_JSON_INCLUDE}
                    ${ICICLE_V28_INCLUDE_DIRS})
            target_link_libraries(gpu_msm_external_icicle_v2_bench
                PRIVATE
                    ecc
                    srs
                    common
                    ${ICICLE_V28_LIBRARIES}
                    CUDA::cudart
                    ${BB_GPU_STDCXX_LIBRARY})
        else()
            message(STATUS "Skipping gpu_msm_external_icicle_v2_bench; set ICICLE_V28_INCLUDE_DIR and "
                           "ICICLE_V28_LIBRARIES or ICICLE_V28_INSTALL_DIR")
        endif()

        set(ICICLE_V4_INSTALL_DIR "" CACHE PATH "Icicle v4.0.0 install directory")
        set(ICICLE_V4_INCLUDE_DIR "" CACHE PATH "Icicle v4.0.0 include directory")
        set(ICICLE_V4_LIBRARY_DIR "" CACHE PATH "Icicle v4.0.0 library directory")
        set(ICICLE_V4_LIBRARIES "" CACHE STRING "Icicle v4.0.0 libraries; overrides auto-discovery")
        if(NOT ICICLE_V4_INCLUDE_DIR AND ICICLE_V4_INSTALL_DIR)
            set(ICICLE_V4_INCLUDE_DIR "${ICICLE_V4_INSTALL_DIR}/include")
        endif()
        if(NOT ICICLE_V4_LIBRARY_DIR AND ICICLE_V4_INSTALL_DIR)
            set(ICICLE_V4_LIBRARY_DIR "${ICICLE_V4_INSTALL_DIR}/lib")
        endif()
        if(ICICLE_V4_INCLUDE_DIR)
            set(ICICLE_V4_INCLUDE_DIRS "${ICICLE_V4_INCLUDE_DIR}")
        endif()
        if(ICICLE_V4_LIBRARY_DIR AND NOT ICICLE_V4_LIBRARIES)
            find_library(ICICLE_V4_CURVE_LIBRARY
                NAMES icicle_curve_bn254
                PATHS "${ICICLE_V4_LIBRARY_DIR}"
                NO_DEFAULT_PATH)
            find_library(ICICLE_V4_DEVICE_LIBRARY
                NAMES icicle_device
                PATHS "${ICICLE_V4_LIBRARY_DIR}"
                NO_DEFAULT_PATH)
            find_library(ICICLE_V4_FIELD_LIBRARY
                NAMES icicle_field_bn254_scalar icicle_field_bn254
                PATHS "${ICICLE_V4_LIBRARY_DIR}"
                NO_DEFAULT_PATH)
            if(ICICLE_V4_CURVE_LIBRARY AND ICICLE_V4_DEVICE_LIBRARY AND ICICLE_V4_FIELD_LIBRARY)
                set(ICICLE_V4_LIBRARIES
                    "${ICICLE_V4_CURVE_LIBRARY};${ICICLE_V4_FIELD_LIBRARY};${ICICLE_V4_DEVICE_LIBRARY}")
            endif()
        endif()
        if(ICICLE_V4_INCLUDE_DIRS AND ICICLE_V4_LIBRARIES)
            add_executable(gpu_msm_external_icicle_v4_bench
                benchmark/external_msm/icicle_v4_msm_runner.cpp)
            target_include_directories(gpu_msm_external_icicle_v4_bench
                PRIVATE
                    "${BB_GPU_EXTERNAL_MSM_BENCH_INCLUDE_DIR}"
                    "${BB_GPU_CPP_SOURCE_DIR}"
                    ${MSGPACK_INCLUDE}
                    ${TRACY_INCLUDE}
                    ${LMDB_INCLUDE}
                    ${LIBDEFLATE_INCLUDE}
                    ${HTTPLIB_INCLUDE}
                    ${BACKWARD_INCLUDE}
                    ${NLOHMANN_JSON_INCLUDE}
                    ${ICICLE_V4_INCLUDE_DIRS})
            target_link_libraries(gpu_msm_external_icicle_v4_bench
                PRIVATE
                    ecc
                    srs
                    common
                    ${ICICLE_V4_LIBRARIES}
                    ${BB_GPU_STDCXX_LIBRARY})
        else()
            message(STATUS "Skipping gpu_msm_external_icicle_v4_bench; set ICICLE_V4_INCLUDE_DIR and "
                           "ICICLE_V4_LIBRARIES or ICICLE_V4_INSTALL_DIR")
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
