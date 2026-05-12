#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm.hpp"

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/gpu/common/device_context.hpp"
#include "barretenberg/gpu/msm/msm_raw.cuh"

#include <cstdint>
#include <cstring>
#include <limits>

namespace bb::gpu::bn254 {
namespace {

constexpr uint32_t NUM_BITS_IN_FIELD = 254;
constexpr uint32_t MAX_SLICE_BITS = 20;
constexpr size_t BUCKET_ACCUMULATION_COST = 5;

uint32_t get_auto_bits_per_slice(const size_t num_points) {
  auto compute_cost = [&](uint32_t bits) {
    const size_t rounds = (NUM_BITS_IN_FIELD + bits - 1) / bits;
    const size_t buckets = size_t{1} << bits;
    return rounds * (num_points + buckets * BUCKET_ACCUMULATION_COST);
  };

  uint32_t best_bits = 1;
  size_t best_cost = compute_cost(1);
  for (uint32_t bits = 2; bits < MAX_SLICE_BITS; ++bits) {
    const size_t cost = compute_cost(bits);
    if (cost < best_cost) {
      best_cost = cost;
      best_bits = bits;
    }
  }
  return best_bits;
}

uint32_t resolve_bits_per_slice(const size_t num_points,
                                const uint32_t requested_bits) {
  const uint32_t bits = requested_bits == 0
                            ? get_auto_bits_per_slice(num_points)
                            : requested_bits;
  if (bits == 0 || bits > MAX_SLICE_BITS || bits > NUM_BITS_IN_FIELD) {
    throw_or_abort("bb::gpu::bn254::msm: bits_per_slice must be in [1, 20]");
  }
  return bits;
}

curve::BN254::AffineElement to_cpu_point(const affine_g1_t &point) {
  curve::BN254::AffineElement out;
  std::memcpy(&out, &point, sizeof(out));
  return out.is_point_at_infinity() ? curve::BN254::AffineElement::infinity()
                                    : out;
}

} // namespace

void init(std::span<const curve::BN254::AffineElement> srs_points) {
  static_assert(sizeof(affine_g1_t) == sizeof(curve::BN254::AffineElement));
  static_assert(alignof(affine_g1_t) == alignof(curve::BN254::AffineElement));
  bb::gpu::default_context().ensure_srs_uploaded(
      {reinterpret_cast<const affine_g1_t *>(srs_points.data()),
       srs_points.size()});
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
      bb::gpu::default_context().get_srs_offset(
          reinterpret_cast<const affine_g1_t *>(points.data()), points.size()) +
      scalars.start_index;
  affine_g1_t result{};
  msm_raw(reinterpret_cast<const fr_t *>(scalars.span.data()), scalars.size(),
          cached_point_start_index, bits_per_slice, &result);
  return to_cpu_point(result);
}

std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          bool /*handle_edge_cases*/) {
  BB_ASSERT_EQ(points.size(), scalars.size());
  std::vector<curve::BN254::AffineElement> results;
  results.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    results.emplace_back(msm({0, std::span<const curve::BN254::ScalarField>(
                                     scalars[i].data(), scalars[i].size())},
                             points[i]));
  }
  return results;
}

void shutdown() { bb::gpu::default_context().reset(); }

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
