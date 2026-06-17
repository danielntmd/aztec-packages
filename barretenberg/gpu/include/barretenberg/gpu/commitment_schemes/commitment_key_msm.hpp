#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/gpu/backend.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <vector>

namespace bb::gpu {

// `commitment_key.hpp` uses this to decide whether to route through the GPU
// backend or fall back to the CPU implementation.
template <class Curve>
inline constexpr bool commitment_key_msm_available = Backend<Curve>::available;

// No-op when `commitment_key_msm_available<Curve>` is `false`.
template <class Curve>
void init_commitment_key_srs(
    std::span<const typename Curve::AffineElement> srs_points) {
  if constexpr (commitment_key_msm_available<Curve>) {
    Backend<Curve>::init_srs(srs_points);
  }
}

template <class Curve>
typename Curve::AffineElement
commitment_key_msm(PolynomialSpan<const typename Curve::ScalarField> scalars,
                   std::span<const typename Curve::AffineElement> points) {
  static_assert(commitment_key_msm_available<Curve>,
                "GPU commitment MSM is not available for this curve");
  return Backend<Curve>::msm(scalars, points);
}

// The GPU XYZZ point addition formulas branch on `(x1 == x2)` inline, so
// duplicate or negated points within a bucket are handled natively.
template <class Curve>
std::vector<typename Curve::AffineElement> commitment_key_batch_msm(
    std::span<std::span<const typename Curve::AffineElement>> points,
    std::span<std::span<typename Curve::ScalarField>> scalars) {
  static_assert(commitment_key_msm_available<Curve>,
                "GPU commitment MSM is not available for this curve");
  return Backend<Curve>::batch_msm(points, scalars);
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
