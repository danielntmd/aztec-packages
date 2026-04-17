# ICICLE GPU acceleration backend.
#
# Activated by -DGPU_BACKEND=icicle on a native (non-WASM) build. The top-level
# CMakeLists.txt guards this include, calls enable_language(CUDA), and defines
# -DBB_GPU_ICICLE=1 so in-code `#ifdef BB_GPU_ICICLE` branches compile.
#
# ICICLE v3.x distributes its open-source C++/Rust API under Apache 2.0. The
# core MSM/field API is header-only and the runtime (icicle_device) plus
# per-curve libraries (icicle_field_bn254, icicle_curve_bn254) are built from
# source. The CUDA backend (libicicle_backend_cuda.so) ships separately from
# Ingonyama and is loaded at runtime via ICICLE_BACKEND_INSTALL_DIR.
#
# Without a CUDA backend binary installed, icicle_load_backend() falls back to
# ICICLE's built-in CPU backend — everything still works, just without GPU
# speedup. This is useful for integration testing on developer machines
# without a GPU.
#
# Reference:
#   https://github.com/ingonyama-zk/icicle
#   https://dev.ingonyama.com/

include(FetchContent)

# ICICLE reads `CURVE` (not namespaced) from its own CMakeLists.txt. Set BEFORE
# FetchContent_MakeAvailable so ICICLE picks it up during configure.
set(CURVE "bn254" CACHE STRING "ICICLE primary curve (BN254 used by Honk commitments)")

# Leave ICICLE features at their defaults (most ON). The precompiled CUDA
# backend binaries reference symbols from HASH/POSEIDON even when we only
# call MSM, so disabling them in the frontend breaks dynamic loading.
# We do skip ICICLE's own tests to save build time.
set(BUILD_TESTS OFF CACHE BOOL "")
# MSM stays ON (default).
# CPU_BACKEND stays ON (default) — provides fallback when CUDA_BACKEND is absent.
# CUDA_BACKEND stays OFF at build time; CUDA backend binary is loaded at runtime.

FetchContent_Declare(
    icicle
    GIT_REPOSITORY https://github.com/ingonyama-zk/icicle.git
    GIT_TAG v3.5.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR icicle
)

FetchContent_MakeAvailable(icicle)

if(icicle_SOURCE_DIR)
    set(ICICLE_INCLUDE "${icicle_SOURCE_DIR}/icicle/include" CACHE PATH "ICICLE include root")
    message(STATUS "ICICLE fetched at ${icicle_SOURCE_DIR}; include root: ${ICICLE_INCLUDE}")
else()
    message(WARNING "ICICLE FetchContent did not populate sources; GPU backend will not compile")
endif()
