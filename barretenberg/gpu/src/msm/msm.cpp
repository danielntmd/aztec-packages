#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm.hpp"

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/curves/bn254/bn254_conversions.hpp"
#include "barretenberg/gpu/msm/msm_heuristics.hpp"
#include "barretenberg/gpu/msm/msm_raw.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

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

std::vector<curve::BN254::AffineElement>
batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
          std::span<std::span<curve::BN254::ScalarField>> scalars,
          const bool handle_edge_cases,
          const uint32_t requested_bits_per_slice) {
  BB_ASSERT_EQ(points.size(), scalars.size());
  if (handle_edge_cases) {
    throw_or_abort(
        "bb::gpu::bn254::batch_msm: handle_edge_cases is not supported");
  }
  const size_t num_msms = points.size();
  std::vector<curve::BN254::AffineElement> results(num_msms);
  if (num_msms == 0) {
    return results;
  }

  auto &context = bb::gpu::default_msm_context();

  struct EntryShape {
    size_t input_index;
    size_t num_scalars;
    size_t point_start_index;
  };
  std::vector<EntryShape> entries;
  entries.reserve(num_msms);
  for (size_t i = 0; i < num_msms; ++i) {
    const size_t n = scalars[i].size();
    if (n == 0) {
      results[i] = curve::BN254::AffineElement::infinity();
      continue;
    }
    if (points[i].size() < n) {
      throw_or_abort("bb::gpu::bn254::batch_msm: point span is smaller than "
                     "scalar count");
    }
    if (n > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      throw_or_abort("bb::gpu::bn254::batch_msm: point indices exceed uint32 "
                     "schedule range");
    }
    const size_t point_start_index = context.get_srs_offset(
        reinterpret_cast<const host_affine_g1_montgomery_t *>(points[i].data()),
        points[i].size());
    entries.push_back({i, n, point_start_index});
  }

  std::sort(entries.begin(), entries.end(),
            [](const EntryShape &a, const EntryShape &b) {
              if (a.num_scalars != b.num_scalars) {
                return a.num_scalars < b.num_scalars;
              }
              return a.point_start_index < b.point_start_index;
            });

  size_t group_begin = 0;
  while (group_begin < entries.size()) {
    size_t group_end = group_begin + 1;
    while (group_end < entries.size() &&
           entries[group_end].num_scalars == entries[group_begin].num_scalars &&
           entries[group_end].point_start_index ==
               entries[group_begin].point_start_index) {
      ++group_end;
    }

    const size_t group_size = group_end - group_begin;
    const size_t num_scalars = entries[group_begin].num_scalars;
    const size_t point_start_index = entries[group_begin].point_start_index;
    const uint32_t bits_per_slice =
        requested_bits_per_slice == 0
            ? get_auto_batched_bits_per_slice(
                  num_scalars,
                  static_cast<uint32_t>(std::min<size_t>(
                      group_size, GPU_MSM_MAX_FUSED_BATCH_SIZE)),
                  get_msm_precompute_factor())
            : requested_bits_per_slice;
    if (bits_per_slice == 0 || bits_per_slice > GPU_MSM_MAX_SLICE_BITS ||
        bits_per_slice > GPU_MSM_NUM_BITS_IN_FIELD) {
      throw_or_abort(
          "bb::gpu::bn254::batch_msm: bits_per_slice must be in [1, 20]");
    }

    size_t chunk_begin = group_begin;
    while (chunk_begin < group_end) {
      const size_t chunk_size = std::min<size_t>(GPU_MSM_MAX_FUSED_BATCH_SIZE,
                                                 group_end - chunk_begin);
      std::vector<const host_fr_montgomery_t *> chunk_scalar_pointers(
          chunk_size);
      std::vector<fq32_affine_g1_t> chunk_results(chunk_size);
      for (size_t k = 0; k < chunk_size; ++k) {
        const auto &entry = entries[chunk_begin + k];
        chunk_scalar_pointers[k] =
            reinterpret_cast<const host_fr_montgomery_t *>(
                scalars[entry.input_index].data());
      }
      msm_raw_batch_fq32(chunk_scalar_pointers.data(), num_scalars,
                         static_cast<uint32_t>(chunk_size), point_start_index,
                         bits_per_slice, chunk_results.data());
      for (size_t k = 0; k < chunk_size; ++k) {
        results[entries[chunk_begin + k].input_index] =
            to_cpu_point(chunk_results[k]);
      }
      chunk_begin += chunk_size;
    }
    group_begin = group_end;
  }

  return results;
}

void shutdown() { bb::gpu::default_msm_context().reset(); }

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
