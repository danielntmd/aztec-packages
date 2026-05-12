#pragma once

/**
 * @brief Native CUDA GPU MSM adapter for BN254 commitments.
 *
 * This header declares drop-in replacements for the CPU Pippenger routines in
 * scalar_multiplication.hpp, specialized for BN254. It is only compiled when
 * -DBB_GPU_NATIVE=1 is defined (via the GPU_BACKEND=native CMake option).
 *
 * CommitmentKey integration is owned by
 * gpu/commitment_schemes/commitment_key_msm.hpp, keeping the C++ commitment
 * interface independent from BN254-specific GPU internals.
 *
 * Grumpkin MSMs (IPA/ECCVM) remain on CPU for now. Batch and single MSM
 * signatures mirror the CPU public API (pippenger_unsafe<BN254> and
 * MSM<BN254>::batch_multi_scalar_mul).
 *
 * The single-MSM path is implemented by the native CUDA adapter. Batch MSM is
 * currently a correctness-first serial wrapper over the single-MSM path.
 */

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <vector>

namespace bb::gpu::bn254 {

/**
 * @brief One-time GPU initialization hook, called from `CommitmentKey<BN254>`.
 *
 * Intended to upload the SRS to device memory and cache it across MSMs.
 * Must be idempotent — on repeated `CommitmentKey` construction it should
 * detect the SRS is already uploaded (or a subset) and skip the upload.
 *
 * Current implementation uploads the SRS into the CUDA device context.
 * Direct calls to `msm` must use points backed by the cached SRS. The
 * CommitmentKey integration calls this during construction.
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
 * @param bits_per_slice Optional Pippenger window size. `0` selects the
 *        adapter's current auto window heuristic.
 */
curve::BN254::AffineElement
msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    uint32_t bits_per_slice = 0);

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
 * Current implementation is a serial wrapper over `msm`.
 */
std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          bool handle_edge_cases);

/**
 * @brief Release adapter-owned GPU resources. Stub: no-op.
 */
void shutdown();

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
