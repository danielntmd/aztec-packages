#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/gpu/backend.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <vector>

namespace bb::gpu::bn254 {

inline void init(std::span<const curve::BN254::AffineElement> srs_points) {
  Backend<curve::BN254>::init_srs(srs_points);
}

inline curve::BN254::AffineElement
msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    const MsmConfig &cfg = {}) {
  return Backend<curve::BN254>::msm(scalars, points, cfg);
}

inline curve::BN254::AffineElement
msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    const uint32_t bits_per_slice) {
  return Backend<curve::BN254>::msm(
      scalars, points, MsmConfig{.bits_per_slice = bits_per_slice});
}

inline std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          const MsmConfig &cfg = {}) {
  return Backend<curve::BN254>::batch_msm(points, scalars, cfg);
}

inline std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          const uint32_t bits_per_slice) {
  return Backend<curve::BN254>::batch_msm(
      points, scalars, MsmConfig{.bits_per_slice = bits_per_slice});
}

inline void shutdown() { Backend<curve::BN254>::shutdown(); }

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
