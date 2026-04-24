#pragma once

/**
 * @brief Native CUDA GPU MSM adapter for BN254 commitments.
 *
 * This header declares drop-in replacements for the CPU Pippenger routines in
 * scalar_multiplication.hpp, specialized for BN254. It is only compiled when
 * -DBB_GPU_NATIVE=1 is defined (via the GPU_BACKEND=native CMake option).
 *
 * Dispatch happens in CommitmentKey (commitment_schemes/commitment_key.hpp):
 *
 *     #ifdef BB_GPU_NATIVE
 *       if constexpr (std::is_same_v<Curve, curve::BN254>) {
 *         return scalar_multiplication::gpu::msm(polynomial, point_table);
 *       }
 *     #endif
 *     return scalar_multiplication::pippenger_unsafe<Curve>(polynomial, point_table);
 *
 * Grumpkin MSMs (IPA/ECCVM) remain on CPU for now. Batch and single MSM
 * signatures mirror the CPU public API (pippenger_unsafe<BN254> and
 * MSM<BN254>::batch_multi_scalar_mul).
 *
 * Current state: stub. Functions are declared so CommitmentKey dispatch links,
 * but the implementations in gpu_msm.cpp abort at runtime until the native
 * CUDA kernels land.
 */

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <vector>

namespace bb::scalar_multiplication::gpu {

/**
 * @brief One-time GPU initialization hook, called from `CommitmentKey<BN254>`.
 *
 * Intended to upload the SRS to device memory and cache it across MSMs.
 * Must be idempotent — on repeated `CommitmentKey` construction it should
 * detect the SRS is already uploaded (or a subset) and skip the upload.
 *
 * Stub: currently a no-op until the device context and SRS cache submodules
 * exist.
 */
void init(std::span<const curve::BN254::AffineElement> srs_points);

/**
 * @brief GPU equivalent of `scalar_multiplication::pippenger_unsafe<BN254>`.
 *
 * Returns affine to match what `CommitmentKey::commit` expects (the CPU path
 * returns a projective Element; the GPU path normalizes to affine on device).
 *
 * Below an internal size threshold, the implementation should fall back to
 * CPU Pippenger because PCIe transfer overhead dominates for small inputs.
 *
 * Stub: aborts at runtime.
 */
curve::BN254::AffineElement msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
                                std::span<const curve::BN254::AffineElement> points);

/**
 * @brief GPU equivalent of `MSM<BN254>::batch_multi_scalar_mul`.
 *
 * Each element of `points` / `scalars` is a separate MSM. `handle_edge_cases`
 * is retained for signature parity with the CPU API, but only `false` is
 * currently supported (matches the BN254 commitment flow, which dispatches
 * the unsafe CPU path).
 *
 * Unlike the CPU path, scalars are not required to be mutated by this call.
 * The parameter type keeps a non-const span for signature compatibility with
 * the CPU API.
 *
 * Stub: aborts at runtime.
 */
std::vector<curve::BN254::AffineElement> batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
                                                   std::span<std::span<curve::BN254::ScalarField>> scalars,
                                                   bool handle_edge_cases);

/**
 * @brief Release adapter-owned GPU resources. Stub: no-op.
 */
void shutdown();

} // namespace bb::scalar_multiplication::gpu

#endif // BB_GPU_NATIVE
