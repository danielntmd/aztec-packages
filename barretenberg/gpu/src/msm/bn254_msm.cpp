#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/backend.hpp"

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/gpu/common/cuda_error.hpp"
#include "barretenberg/gpu/curves/bn254/bn254_conversions.hpp"
#include "common/gpu_msm_context.hpp"
#include "msm/internal/msm_heuristics.hpp"
#include "msm/internal/msm_raw.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace bb::gpu {
namespace {

using bn254::fq32_affine_g1_t;
using bn254::GPU_MSM_MAX_FUSED_BATCH_SIZE;
using bn254::GPU_MSM_MAX_SLICE_BITS;
using bn254::host_affine_g1_montgomery_t;
using bn254::host_fr_montgomery_t;
using bn254::MsmRawOptions;
using bn254::to_cpu_point;

// Resolve a public MsmConfig into already-validated raw options.
// `batch_size` selects between the single-MSM and batched auto-bits heuristic
// when `cfg.bits_per_slice == 0`; pass 1 for the non-batched path.
MsmRawOptions resolve_msm_options(const MsmConfig &cfg, const size_t num_points,
                                  const uint32_t batch_size = 1) {
  check_condition(bn254::is_valid_msm_precompute_factor(cfg.precompute_factor),
                  "Backend<BN254>: precompute factor must be in [1, 16]");
  const uint32_t bits =
      cfg.bits_per_slice != 0
          ? cfg.bits_per_slice
          : (batch_size == 1
                 ? bn254::get_auto_bits_per_slice(num_points,
                                                  cfg.precompute_factor)
                 : bn254::get_auto_batched_bits_per_slice(
                       num_points, batch_size, cfg.precompute_factor));
  check_condition(bits != 0 && bits <= GPU_MSM_MAX_SLICE_BITS,
                  "Backend<BN254>: bits_per_slice must be in [1, 20]");
  return MsmRawOptions{
      .bits_per_slice = bits,
      .precompute_factor = cfg.precompute_factor,
      .precompute_cache_min_length = cfg.precompute_cache_min_length,
  };
}

// One fused-batch dispatch: K MSMs (K <= GPU_MSM_MAX_FUSED_BATCH_SIZE) sharing
// `num_scalars_per_msm` scalars over the same SRS slice.
struct FusedMsmBatch {
  size_t num_scalars_per_msm;
  size_t point_start_index;
  std::vector<size_t> input_indices;
};

struct BatchPlan {
  std::vector<size_t> empty_input_indices;
  std::vector<FusedMsmBatch> fused_batches;
};

size_t select_fused_batch_size(const size_t num_scalars_per_msm,
                               const size_t max_batch_size,
                               const MsmConfig &cfg) {
  BB_BENCH_NAME("GPU::msm_memory_fit");
  for (size_t batch_size = max_batch_size; batch_size > 1; --batch_size) {
    const auto options = resolve_msm_options(
        cfg, num_scalars_per_msm, static_cast<uint32_t>(batch_size));
    if (bn254::msm_raw_batch_fq32_fits(num_scalars_per_msm,
                                       static_cast<uint32_t>(batch_size),
                                       options)) {
      return batch_size;
    }
  }
  return 1;
}

// Partition `(points, scalars)` into fused dispatches grouped by shared
// (num_scalars, point_start_index), then chunked at
// GPU_MSM_MAX_FUSED_BATCH_SIZE.
BatchPlan
plan_batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
               std::span<std::span<curve::BN254::ScalarField>> scalars,
               GpuMsmContext &context, const MsmConfig &cfg) {
  struct Entry {
    size_t input_index;
    size_t num_scalars;
    size_t point_start_index;
  };

  const size_t num_msms = points.size();
  BatchPlan plan;
  std::vector<Entry> entries;
  entries.reserve(num_msms);

  for (size_t i = 0; i < num_msms; ++i) {
    const size_t n = scalars[i].size();
    if (n == 0) {
      plan.empty_input_indices.push_back(i);
      continue;
    }
    check_condition(points[i].size() >= n,
                    "Backend<BN254>::batch_msm: point span is smaller than "
                    "scalar count");
    check_condition(
        n <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "Backend<BN254>::batch_msm: point indices exceed uint32 schedule "
        "range");
    const size_t point_start_index = context.get_srs_offset(
        reinterpret_cast<const host_affine_g1_montgomery_t *>(points[i].data()),
        points[i].size());
    entries.push_back({i, n, point_start_index});
  }

  std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
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

    size_t chunk_begin = group_begin;
    while (chunk_begin < group_end) {
      const size_t max_chunk_size =
          std::min<size_t>(GPU_MSM_MAX_FUSED_BATCH_SIZE,
                           group_end - chunk_begin);
      const size_t chunk_size = select_fused_batch_size(
          entries[group_begin].num_scalars, max_chunk_size, cfg);
      FusedMsmBatch fused{
          .num_scalars_per_msm = entries[group_begin].num_scalars,
          .point_start_index = entries[group_begin].point_start_index,
          .input_indices = {},
      };
      fused.input_indices.reserve(chunk_size);
      for (size_t k = 0; k < chunk_size; ++k) {
        fused.input_indices.push_back(entries[chunk_begin + k].input_index);
      }
      plan.fused_batches.push_back(std::move(fused));
      chunk_begin += chunk_size;
    }
    group_begin = group_end;
  }
  return plan;
}

void run_fused_batch(std::span<std::span<curve::BN254::ScalarField>> scalars,
                     const FusedMsmBatch &fused, const MsmConfig &cfg,
                     std::vector<curve::BN254::AffineElement> &results) {
  const auto batch_size = static_cast<uint32_t>(fused.input_indices.size());
  const auto options =
      resolve_msm_options(cfg, fused.num_scalars_per_msm, batch_size);

  std::vector<const host_fr_montgomery_t *> scalar_pointers;
  scalar_pointers.reserve(batch_size);
  for (const size_t idx : fused.input_indices) {
    scalar_pointers.push_back(
        reinterpret_cast<const host_fr_montgomery_t *>(scalars[idx].data()));
  }

  std::vector<fq32_affine_g1_t> chunk_results(batch_size);
  {
    BB_BENCH_NAME("GPU::msm_raw_batch");
    bn254::msm_raw_batch_fq32(scalar_pointers.data(),
                              fused.num_scalars_per_msm, batch_size,
                              fused.point_start_index, options,
                              chunk_results.data());
  }
  for (size_t k = 0; k < fused.input_indices.size(); ++k) {
    results[fused.input_indices[k]] = to_cpu_point(chunk_results[k]);
  }
}

} // namespace

void Backend<curve::BN254>::init_srs(
    std::span<const curve::BN254::AffineElement> srs_points) {
  static_assert(sizeof(host_affine_g1_montgomery_t) ==
                sizeof(curve::BN254::AffineElement));
  static_assert(alignof(host_affine_g1_montgomery_t) ==
                alignof(curve::BN254::AffineElement));
  BB_BENCH_NAME("GPU::srs_upload");
  default_msm_context().ensure_srs_uploaded(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(srs_points.data()),
      srs_points.size());
}

curve::BN254::AffineElement Backend<curve::BN254>::msm(
    PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points, const MsmConfig &cfg) {
  if (scalars.size() == 0) {
    return curve::BN254::AffineElement::infinity();
  }
  check_condition(points.size() >= scalars.end_index(),
                  "Backend<BN254>::msm: point span is smaller than scalar "
                  "end index");
  check_condition(scalars.end_index() <=
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                  "Backend<BN254>::msm: point indices exceed uint32 schedule "
                  "range");

  const auto options = resolve_msm_options(cfg, scalars.size());
  const size_t cached_point_start_index =
      default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size()) +
      scalars.start_index;
  fq32_affine_g1_t result{};
  {
    BB_BENCH_NAME("GPU::msm_raw");
    bn254::msm_raw_fq32(
        reinterpret_cast<const host_fr_montgomery_t *>(scalars.span.data()),
        scalars.size(), cached_point_start_index, options, &result);
  }
  return to_cpu_point(result);
}

std::vector<curve::BN254::AffineElement> Backend<curve::BN254>::batch_msm(
    std::span<std::span<const curve::BN254::AffineElement>> points,
    std::span<std::span<curve::BN254::ScalarField>> scalars,
    const MsmConfig &cfg) {
  BB_ASSERT_EQ(points.size(), scalars.size());

  std::vector<curve::BN254::AffineElement> results(points.size());
  if (points.empty()) {
    return results;
  }

  const auto plan = plan_batch_msm(points, scalars, default_msm_context(), cfg);
  for (const size_t idx : plan.empty_input_indices) {
    results[idx] = curve::BN254::AffineElement::infinity();
  }
  for (const auto &fused : plan.fused_batches) {
    run_fused_batch(scalars, fused, cfg, results);
  }
  return results;
}

void Backend<curve::BN254>::shutdown() { default_msm_context().reset(); }

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
