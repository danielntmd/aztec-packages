#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bb::gpu {

struct MsmConfig {
  // 0 selects the cost-minimising width via the internal heuristic; otherwise
  // must be in [1, 20].
  uint32_t bits_per_slice = 0;
  // Number of doubled SRS copies cached on device. Must be in [1, 16].
  uint32_t precompute_factor = 4;
  // Minimum point count covered when uploading the precomputed SRS. Larger
  // values keep the cache warm across MSMs spanning sub-ranges of one SRS;
  // smaller values trade reuse for device memory.
  size_t precompute_cache_min_length = size_t{1} << 24;
};

// Primary template exposes `available = false` so callers can `if constexpr`
// without including curve-specific headers.
template <class Curve> class Backend {
public:
  static constexpr bool available = false;
};

// All methods are thread-compatible but not thread-safe: callers must
// serialise access (the implementation routes through a single process-wide
// CUDA context and stream). `init_srs` must be called before any `msm` /
// `batch_msm`. Calls with matching `(point_start_index, num_scalars)` inside
// `batch_msm` are fused into batched Pippenger launches.
template <> class Backend<curve::BN254> {
public:
  static constexpr bool available = true;

  // `srs_points` must all be finite on-curve points; passing infinity or
  // off-curve points aborts. Re-uploading a different SRS invalidates the
  // cached precomputed (shifted) SRS.
  static void init_srs(std::span<const curve::BN254::AffineElement> srs_points);

  // `points` must alias a sub-span of the SRS uploaded via `init_srs`.
  static curve::BN254::AffineElement
  msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
      std::span<const curve::BN254::AffineElement> points,
      const MsmConfig &cfg = {});

  static std::vector<curve::BN254::AffineElement>
  batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
            std::span<std::span<curve::BN254::ScalarField>> scalars,
            const MsmConfig &cfg = {});

  // Safe to call multiple times.
  static void shutdown();
};

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
