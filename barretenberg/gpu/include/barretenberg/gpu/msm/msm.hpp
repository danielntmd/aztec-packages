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

// Inputs are partitioned by (point_start_index, num_scalars) into uniform
// groups and each group is dispatched to the fused batched pipeline, capped at
// GPU_MSM_MAX_FUSED_BATCH_SIZE per launch. bits_per_slice=0 selects auto.
std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          bool handle_edge_cases, uint32_t bits_per_slice = 0);

void shutdown();

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
