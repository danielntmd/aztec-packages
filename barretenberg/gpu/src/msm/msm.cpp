#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm.hpp"

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/msm/msm_heuristics.hpp"
#include "barretenberg/gpu/msm/msm_raw.cuh"

#include <cstdint>
#include <limits>

namespace bb::gpu::bn254 {
namespace {

uint32_t resolve_bits_per_slice(const size_t num_points,
                                const uint32_t requested_bits) {
  const uint32_t bits =
      requested_bits == 0
          ? get_auto_bits_per_slice(num_points, get_msm_precompute_factor())
          : requested_bits;
  if (bits == 0 || bits > GPU_MSM_MAX_SLICE_BITS ||
      bits > GPU_MSM_NUM_BITS_IN_FIELD) {
    throw_or_abort("bb::gpu::bn254::msm: bits_per_slice must be in [1, 20]");
  }
  return bits;
}

curve::BN254::BaseField to_cpu_montgomery_field(const fq32_t &value) {
  curve::BN254::BaseField standard{
      static_cast<uint64_t>(value.limbs[0]) |
          (static_cast<uint64_t>(value.limbs[1]) << 32),
      static_cast<uint64_t>(value.limbs[2]) |
          (static_cast<uint64_t>(value.limbs[3]) << 32),
      static_cast<uint64_t>(value.limbs[4]) |
          (static_cast<uint64_t>(value.limbs[5]) << 32),
      static_cast<uint64_t>(value.limbs[6]) |
          (static_cast<uint64_t>(value.limbs[7]) << 32),
  };
  return standard.to_montgomery_form();
}

curve::BN254::AffineElement to_cpu_point(const fq32_affine_g1_t &point) {
  if (is_msb_set(point.x)) {
    return curve::BN254::AffineElement::infinity();
  }
  return {to_cpu_montgomery_field(point.x), to_cpu_montgomery_field(point.y)};
}

void validate_srs_points_are_finite(
    std::span<const curve::BN254::AffineElement> srs_points) {
  for (const auto &point : srs_points) {
    if (point.is_point_at_infinity() || !point.on_curve()) {
      throw_or_abort(
          "bb::gpu::bn254::init: SRS points must be finite on-curve points");
    }
  }
}

} // namespace

void init(std::span<const curve::BN254::AffineElement> srs_points) {
  static_assert(sizeof(host_affine_g1_montgomery_t) ==
                sizeof(curve::BN254::AffineElement));
  static_assert(alignof(host_affine_g1_montgomery_t) ==
                alignof(curve::BN254::AffineElement));
  validate_srs_points_are_finite(srs_points);
  bb::gpu::default_msm_context().ensure_srs_uploaded(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(srs_points.data()),
      srs_points.size());
}

curve::BN254::AffineElement
msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    const uint32_t requested_bits_per_slice) {
  if (scalars.size() == 0) {
    return curve::BN254::AffineElement::infinity();
  }
  if (points.size() < scalars.end_index()) {
    throw_or_abort(
        "bb::gpu::bn254::msm: point span is smaller than scalar end index");
  }
  if (scalars.end_index() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw_or_abort(
        "bb::gpu::bn254::msm: point indices exceed uint32 schedule range");
  }

  const uint32_t bits_per_slice =
      resolve_bits_per_slice(scalars.size(), requested_bits_per_slice);
  const size_t cached_point_start_index =
      bb::gpu::default_msm_context().get_srs_offset(
          reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
          points.size()) +
      scalars.start_index;
  fq32_affine_g1_t result{};
  msm_raw_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.span.data()),
      scalars.size(), cached_point_start_index, bits_per_slice, &result);
  return to_cpu_point(result);
}

// todo: danielntmd
std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          const bool handle_edge_cases) {
  BB_ASSERT_EQ(points.size(), scalars.size());
  if (handle_edge_cases) {
    throw_or_abort(
        "bb::gpu::bn254::batch_msm: handle_edge_cases is not supported");
  }
  std::vector<curve::BN254::AffineElement> results;
  results.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    results.emplace_back(msm({0, std::span<const curve::BN254::ScalarField>(
                                     scalars[i].data(), scalars[i].size())},
                             points[i]));
  }
  return results;
}

void shutdown() { bb::gpu::default_msm_context().reset(); }

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
