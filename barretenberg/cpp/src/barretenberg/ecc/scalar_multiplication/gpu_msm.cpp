/**
 * @file gpu_msm.cpp
 * @brief Native CUDA GPU MSM adapter — stub implementation.
 *
 * Picked up automatically by the `barretenberg_module(ecc ...)` glob over
 * *.cpp in this directory. Everything is guarded by `#ifdef BB_GPU_NATIVE`,
 * so under `GPU_BACKEND=none` the file compiles to an empty translation
 * unit and has zero impact on default builds.
 *
 * Current state:
 *   - `init()`    — no-op (no device context / SRS cache yet).
 *   - `msm()`     — aborts at runtime; wire up once the CUDA kernels land.
 *   - `batch_msm()` — aborts at runtime.
 *   - `shutdown()` — no-op.
 *
 * The signatures exist so the dispatch in `commitment_key.hpp` links under
 * `GPU_BACKEND=native`. Run the CPU path (`GPU_BACKEND=none`) until the
 * native kernels are in place.
 */

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/scalar_multiplication/gpu_msm.hpp"

#include "barretenberg/common/throw_or_abort.hpp"

namespace bb::scalar_multiplication::gpu {

void init(std::span<const curve::BN254::AffineElement> /*srs_points*/)
{
    // No-op stub. Device context, SRS upload, and on-device cache land with
    // the memory submodule.
}

curve::BN254::AffineElement msm(PolynomialSpan<const curve::BN254::ScalarField> /*scalars*/,
                                std::span<const curve::BN254::AffineElement> /*points*/)
{
    throw_or_abort("bb::scalar_multiplication::gpu::msm: native CUDA MSM not yet implemented. "
                   "Build with GPU_BACKEND=none or wire up the native kernel before calling.");
}

std::vector<curve::BN254::AffineElement> batch_msm(std::span<std::span<const curve::BN254::AffineElement>> /*points*/,
                                                   std::span<std::span<curve::BN254::ScalarField>> /*scalars*/,
                                                   bool /*handle_edge_cases*/)
{
    throw_or_abort("bb::scalar_multiplication::gpu::batch_msm: native CUDA batch MSM not yet implemented. "
                   "Build with GPU_BACKEND=none or wire up the native kernel before calling.");
}

void shutdown()
{
    // No-op stub.
}

} // namespace bb::scalar_multiplication::gpu

#endif // BB_GPU_NATIVE
