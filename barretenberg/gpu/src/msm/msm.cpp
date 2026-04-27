/**
 * @file msm.cpp
 * @brief Native CUDA GPU MSM adapter — stub implementation.
 *
 * Compiled by the `barretenberg/gpu` module. Everything is guarded by
 * `#ifdef BB_GPU_NATIVE`, so under `GPU_BACKEND=none` the file compiles to an
 * empty translation unit and has zero impact on default builds.
 *
 * Current state:
 *   - `init()`    — uploads SRS points into the CUDA device context.
 *   - `msm()`     — aborts at runtime; wire up once the CUDA kernels land.
 *   - `batch_msm()` — aborts at runtime.
 *   - `shutdown()` — no-op.
 *
 * The signatures exist so the dispatch in `commitment_key.hpp` links under
 * `GPU_BACKEND=native`. Run the CPU path (`GPU_BACKEND=none`) until the
 * native kernels are in place.
 */

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm.hpp"

#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/gpu/common/device_context.hpp"

namespace bb::gpu::bn254 {

void init(std::span<const curve::BN254::AffineElement> srs_points)
{
    static_assert(sizeof(affine_g1_t) == sizeof(curve::BN254::AffineElement));
    static_assert(alignof(affine_g1_t) == alignof(curve::BN254::AffineElement));
    bb::gpu::default_context().ensure_srs_uploaded(
        { reinterpret_cast<const affine_g1_t*>(srs_points.data()), srs_points.size() });
}

curve::BN254::AffineElement msm(PolynomialSpan<const curve::BN254::ScalarField> /*scalars*/,
                                std::span<const curve::BN254::AffineElement> /*points*/)
{
    throw_or_abort("bb::gpu::bn254::msm: native CUDA MSM not yet implemented. "
                   "Build with GPU_BACKEND=none or wire up the native kernel before calling.");
}

std::vector<curve::BN254::AffineElement> batch_msm(std::span<std::span<const curve::BN254::AffineElement>> /*points*/,
                                                   std::span<std::span<curve::BN254::ScalarField>> /*scalars*/,
                                                   bool /*handle_edge_cases*/)
{
    throw_or_abort("bb::gpu::bn254::batch_msm: native CUDA batch MSM not yet implemented. "
                   "Build with GPU_BACKEND=none or wire up the native kernel before calling.");
}

void shutdown()
{
    // No-op stub.
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
