#pragma once

/**
 * @brief GPU MSM adapter for ICICLE.
 *
 * This header declares drop-in replacements for the CPU Pippenger routines in
 * scalar_multiplication.hpp, specialized for BN254. It is only compiled when
 * -DBB_GPU_ICICLE=1 is defined (via the GPU_BACKEND=icicle CMake option).
 *
 * Dispatch happens in CommitmentKey (commitment_schemes/commitment_key.hpp):
 *     #ifdef BB_GPU_ICICLE
 *       if constexpr (std::is_same_v<Curve, curve::BN254>) {
 *         return scalar_multiplication::gpu::msm(polynomial, point_table);
 *       }
 *     #endif
 *     return scalar_multiplication::pippenger_unsafe<Curve>(polynomial, point_table);
 *
 * Grumpkin MSMs (used by IPA/ECCVM) remain on CPU for now; ICICLE does support
 * Grumpkin but that's a follow-up integration.
 */

#ifdef BB_GPU_ICICLE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <vector>

namespace bb::scalar_multiplication::gpu {

/**
 * @brief Upload (or expand) the SRS point table to GPU memory.
 *
 * Idempotent. If called repeatedly with a `points` span that is a prefix of or
 * equal to the currently-uploaded SRS, this is a no-op. If called with a
 * larger SRS, the existing GPU buffer is freed and replaced.
 *
 * Invoked from the `CommitmentKey<BN254>` constructor; user code does not
 * normally need to call this directly.
 */
void init(std::span<const curve::BN254::AffineElement> srs_points);

/**
 * @brief GPU equivalent of `scalar_multiplication::pippenger_unsafe<BN254>`.
 *
 * Returns `Curve::Element` (projective) to match `pippenger_unsafe` exactly,
 * so the dispatch in `CommitmentKey::commit()` is uniform between the CPU and
 * GPU paths and the existing Element→AffineElement conversion at commit's
 * return is reused.
 *
 * Below an internal size threshold, implementations should fall back to CPU
 * Pippenger because PCIe transfer latency dominates for small inputs.
 */
curve::BN254::Element msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
                          std::span<const curve::BN254::AffineElement> points);

/**
 * @brief GPU equivalent of `MSM<BN254>::batch_multi_scalar_mul`.
 *
 * Each element of `points` / `scalars` is a separate MSM. `handle_edge_cases`
 * is retained for signature parity; most callsites pass `false`. Returns
 * affine results to match the CPU path's return type (which itself normalizes
 * from projective internally).
 *
 * Unlike the CPU path, scalars are not mutated by this call (we copy them to
 * device memory and transform there). The parameter type keeps a non-const
 * span for signature compatibility.
 */
std::vector<curve::BN254::AffineElement> batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
                                                   std::span<std::span<curve::BN254::ScalarField>> scalars,
                                                   bool handle_edge_cases);

/**
 * @brief Release GPU resources held by the adapter.
 *
 * Safe to call multiple times. Typically invoked at process exit; the
 * CommitmentKey destructor does not call this because the GPU buffer is shared
 * across CommitmentKey instances.
 */
void shutdown();

} // namespace bb::scalar_multiplication::gpu

#endif // BB_GPU_ICICLE