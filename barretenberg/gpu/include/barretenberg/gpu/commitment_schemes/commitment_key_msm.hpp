#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/gpu/msm/msm.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <span>
#include <type_traits>
#include <vector>

namespace bb::gpu {

template <class Curve> inline constexpr bool commitment_key_msm_available = false;
template <> inline constexpr bool commitment_key_msm_available<curve::BN254> = true;

template <class Curve> void init_commitment_key_srs(std::span<const typename Curve::AffineElement> srs_points)
{
    if constexpr (std::is_same_v<Curve, curve::BN254>) {
        bn254::init(srs_points);
    }
}

template <class Curve>
typename Curve::AffineElement commitment_key_msm(PolynomialSpan<const typename Curve::ScalarField> scalars,
                                                 std::span<const typename Curve::AffineElement> points)
{
    static_assert(commitment_key_msm_available<Curve>, "GPU commitment MSM is not available for this curve");
    if constexpr (std::is_same_v<Curve, curve::BN254>) {
        return bn254::msm(scalars, points);
    }
}

template <class Curve>
std::vector<typename Curve::AffineElement> commitment_key_batch_msm(
    std::span<std::span<const typename Curve::AffineElement>> points,
    std::span<std::span<typename Curve::ScalarField>> scalars,
    bool handle_edge_cases)
{
    static_assert(commitment_key_msm_available<Curve>, "GPU commitment MSM is not available for this curve");
    if constexpr (std::is_same_v<Curve, curve::BN254>) {
        return bn254::batch_msm(points, scalars, handle_edge_cases);
    }
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
