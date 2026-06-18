#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/gpu/backend.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <cstdlib>
#include <span>
#include <string>
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

inline MsmConfig commitment_key_msm_config() {
  MsmConfig cfg{};
  if (const char* value = std::getenv("BB_GPU_MSM_PRECOMPUTE_FACTOR");
      value != nullptr && *value != '\0') {
    cfg.precompute_factor =
        static_cast<uint32_t>(std::stoul(std::string(value)));
  }
  return cfg;
}

inline size_t commitment_key_msm_max_batch_size() {
  if (const char* value = std::getenv("BB_GPU_MSM_MAX_BATCH_SIZE");
      value != nullptr && *value != '\0') {
    return std::stoul(std::string(value));
  }
  return 0;
}

template <class Curve>
typename Curve::AffineElement
commitment_key_msm(PolynomialSpan<const typename Curve::ScalarField> scalars,
                   std::span<const typename Curve::AffineElement> points) {
  static_assert(commitment_key_msm_available<Curve>,
                "GPU commitment MSM is not available for this curve");
  return Backend<Curve>::msm(scalars, points, commitment_key_msm_config());
}

// The GPU XYZZ point addition formulas branch on `(x1 == x2)` inline, so
// duplicate or negated points within a bucket are handled natively.
template <class Curve>
std::vector<typename Curve::AffineElement> commitment_key_batch_msm(
    std::span<std::span<const typename Curve::AffineElement>> points,
    std::span<std::span<typename Curve::ScalarField>> scalars) {
  static_assert(commitment_key_msm_available<Curve>,
                "GPU commitment MSM is not available for this curve");
  const MsmConfig cfg = commitment_key_msm_config();
  const size_t max_batch_size = commitment_key_msm_max_batch_size();
  if (max_batch_size == 0 || max_batch_size >= points.size()) {
    return Backend<Curve>::batch_msm(points, scalars, cfg);
  }

  std::vector<typename Curve::AffineElement> results;
  results.reserve(points.size());
  for (size_t offset = 0; offset < points.size(); offset += max_batch_size) {
    const size_t remaining = points.size() - offset;
    const size_t chunk_size =
        remaining < max_batch_size ? remaining : max_batch_size;
    auto chunk = Backend<Curve>::batch_msm(points.subspan(offset, chunk_size),
                                           scalars.subspan(offset, chunk_size),
                                           cfg);
    results.insert(results.end(), chunk.begin(), chunk.end());
  }
  return results;
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
