#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <vector>

namespace bb::gpu::bn254 {

void init(std::span<const curve::BN254::AffineElement> srs_points);

curve::BN254::AffineElement
msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    uint32_t bits_per_slice = 0);

// Batch MSM preserves the CPU API shape. Each entry in points/scalars is a
// separate MSM, handle_edge_cases must be false, and the implementation runs
// the single-MSM path for each batch item.
std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          bool handle_edge_cases);

void shutdown();

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
