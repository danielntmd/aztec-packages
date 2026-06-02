#pragma once

#ifdef BB_GPU_NATIVE

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

constexpr uint32_t GPU_MSM_NUM_BITS_IN_FIELD = 254;
constexpr uint32_t GPU_MSM_MAX_SLICE_BITS = 20;
constexpr uint32_t GPU_MSM_MIN_PRECOMPUTE_FACTOR = 1;
constexpr uint32_t GPU_MSM_MAX_PRECOMPUTE_FACTOR = 16;
constexpr size_t GPU_MSM_BUCKET_ACCUMULATION_COST = 5;
constexpr size_t GPU_MSM_LARGE_BUCKET_MIN_THRESHOLD = 512;
constexpr size_t GPU_MSM_PRECOMPUTED_FOLDED_BUCKET_LIMIT =
    (GPU_MSM_LARGE_BUCKET_MIN_THRESHOLD * 3) / 4;

constexpr uint32_t GPU_MSM_MAX_FUSED_BATCH_SIZE = 16;
constexpr uint32_t GPU_MSM_BATCH_KEY_BITS = 4;
// Pinned against sizeof(fq32_xyzz_g1_t) by static_assert in msm.cu.
constexpr size_t GPU_MSM_BUCKET_ELEMENT_BYTES = 160;
constexpr size_t GPU_MSM_BATCHED_BUCKET_BUDGET_BYTES =
    size_t{4} * 1024 * 1024 * 1024;

inline size_t ceil_div_size_t(const size_t numerator,
                              const size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

inline bool is_valid_msm_precompute_factor(const uint32_t factor) {
  return factor >= GPU_MSM_MIN_PRECOMPUTE_FACTOR &&
         factor <= GPU_MSM_MAX_PRECOMPUTE_FACTOR;
}

inline uint32_t
get_effective_msm_precompute_factor(const uint32_t original_num_windows,
                                    const uint32_t requested_factor) {
  const uint32_t valid_requested_factor =
      requested_factor < GPU_MSM_MIN_PRECOMPUTE_FACTOR
          ? GPU_MSM_MIN_PRECOMPUTE_FACTOR
      : requested_factor > GPU_MSM_MAX_PRECOMPUTE_FACTOR
          ? GPU_MSM_MAX_PRECOMPUTE_FACTOR
          : requested_factor;
  return valid_requested_factor < original_num_windows ? valid_requested_factor
                                                       : original_num_windows;
}

inline size_t
estimate_folded_bucket_pressure(const size_t num_points,
                                const uint32_t bits_per_slice,
                                const uint32_t precompute_factor) {
  const uint32_t original_windows =
      (GPU_MSM_NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t effective_precompute_factor =
      get_effective_msm_precompute_factor(original_windows, precompute_factor);
  const uint32_t folded_windows =
      effective_precompute_factor == 1
          ? original_windows
          : static_cast<uint32_t>(
                ceil_div_size_t(original_windows, effective_precompute_factor));
  size_t max_pressure = 0;

  for (uint32_t target_window = 0; target_window < folded_windows;
       ++target_window) {
    size_t pressure = 0;
    for (uint32_t low_window = 0; low_window < original_windows; ++low_window) {
      const uint32_t window =
          effective_precompute_factor == 1
              ? low_window
              : folded_windows - 1 - (low_window % folded_windows);
      if (window != target_window) {
        continue;
      }
      const uint32_t lo_bit = low_window * bits_per_slice;
      if (lo_bit >= GPU_MSM_NUM_BITS_IN_FIELD) {
        continue;
      }
      const uint32_t remaining_bits = GPU_MSM_NUM_BITS_IN_FIELD - lo_bit;
      const uint32_t actual_slice_bits =
          remaining_bits < bits_per_slice ? remaining_bits : bits_per_slice;
      const size_t buckets = size_t{1} << actual_slice_bits;
      pressure += ceil_div_size_t(num_points, buckets);
    }
    if (pressure > max_pressure) {
      max_pressure = pressure;
    }
  }
  return max_pressure;
}

inline bool has_pathological_bucket_pressure(const size_t num_points,
                                             const uint32_t bits_per_slice,
                                             const uint32_t precompute_factor) {
  const uint32_t original_windows =
      (GPU_MSM_NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t effective_precompute_factor =
      get_effective_msm_precompute_factor(original_windows, precompute_factor);
  const size_t limit = effective_precompute_factor == 1
                           ? GPU_MSM_LARGE_BUCKET_MIN_THRESHOLD
                           : GPU_MSM_PRECOMPUTED_FOLDED_BUCKET_LIMIT;
  return estimate_folded_bucket_pressure(num_points, bits_per_slice,
                                         precompute_factor) >= limit;
}

inline size_t estimate_msm_cost(const size_t num_points,
                                const uint32_t bits_per_slice) {
  const size_t rounds =
      (GPU_MSM_NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const size_t buckets = size_t{1} << bits_per_slice;
  return rounds * (num_points + buckets * GPU_MSM_BUCKET_ACCUMULATION_COST);
}

inline uint32_t get_auto_bits_per_slice(const size_t num_points,
                                        const uint32_t precompute_factor) {
  uint32_t best_bits = 1;
  size_t best_cost = estimate_msm_cost(num_points, 1);
  uint32_t best_non_pathological_bits = 0;
  size_t best_non_pathological_cost = 0;

  for (uint32_t bits = 2; bits < GPU_MSM_MAX_SLICE_BITS; ++bits) {
    const size_t cost = estimate_msm_cost(num_points, bits);
    if (cost < best_cost) {
      best_cost = cost;
      best_bits = bits;
    }
    if (!has_pathological_bucket_pressure(num_points, bits,
                                          precompute_factor) &&
        (best_non_pathological_bits == 0 ||
         cost < best_non_pathological_cost)) {
      best_non_pathological_bits = bits;
      best_non_pathological_cost = cost;
    }
  }

  return best_non_pathological_bits != 0 ? best_non_pathological_bits
                                         : best_bits;
}

inline uint32_t get_auto_bits_per_slice(const size_t num_points) {
  return get_auto_bits_per_slice(num_points, 1);
}

inline bool is_valid_fused_batch_size(const uint32_t batch_size) {
  return batch_size >= 1 && batch_size <= GPU_MSM_MAX_FUSED_BATCH_SIZE;
}

// The cost-minimizing c is K-independent (the K factor cancels in argmin); the
// batch_size parameter is accepted for symmetry and future tuning.
inline size_t estimate_batched_msm_cost(const size_t num_points,
                                        const uint32_t batch_size,
                                        const uint32_t bits_per_slice) {
  (void)batch_size;
  return estimate_msm_cost(num_points, bits_per_slice);
}

inline size_t estimate_batched_bucket_bytes(const size_t num_points,
                                            const uint32_t batch_size,
                                            const uint32_t bits_per_slice,
                                            const uint32_t precompute_factor) {
  (void)num_points;
  if (bits_per_slice == 0 || batch_size == 0) {
    return 0;
  }
  const uint32_t original_windows =
      (GPU_MSM_NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t effective_precompute_factor =
      get_effective_msm_precompute_factor(original_windows, precompute_factor);
  const uint32_t active_num_windows =
      effective_precompute_factor == 1
          ? original_windows
          : static_cast<uint32_t>(
                ceil_div_size_t(original_windows, effective_precompute_factor));
  const size_t buckets_per_window = size_t{1} << bits_per_slice;
  return static_cast<size_t>(batch_size) *
         static_cast<size_t>(active_num_windows) * buckets_per_window *
         GPU_MSM_BUCKET_ELEMENT_BYTES;
}

inline bool has_pathological_batched_bucket_pressure(
    const size_t num_points, const uint32_t batch_size,
    const uint32_t bits_per_slice, const uint32_t precompute_factor) {
  if (has_pathological_bucket_pressure(num_points, bits_per_slice,
                                       precompute_factor)) {
    return true;
  }
  return estimate_batched_bucket_bytes(num_points, batch_size, bits_per_slice,
                                       precompute_factor) >
         GPU_MSM_BATCHED_BUCKET_BUDGET_BYTES;
}

inline uint32_t
get_auto_batched_bits_per_slice(const size_t num_points,
                                const uint32_t batch_size,
                                const uint32_t precompute_factor) {
  uint32_t best_bits = 1;
  size_t best_cost = estimate_batched_msm_cost(num_points, batch_size, 1);
  uint32_t best_non_pathological_bits = 0;
  size_t best_non_pathological_cost = 0;

  for (uint32_t bits = 2; bits < GPU_MSM_MAX_SLICE_BITS; ++bits) {
    const size_t cost = estimate_batched_msm_cost(num_points, batch_size, bits);
    if (cost < best_cost) {
      best_cost = cost;
      best_bits = bits;
    }
    if (!has_pathological_batched_bucket_pressure(num_points, batch_size, bits,
                                                  precompute_factor) &&
        (best_non_pathological_bits == 0 ||
         cost < best_non_pathological_cost)) {
      best_non_pathological_bits = bits;
      best_non_pathological_cost = cost;
    }
  }

  return best_non_pathological_bits != 0 ? best_non_pathological_bits
                                         : best_bits;
}

inline uint32_t get_auto_batched_bits_per_slice(const size_t num_points,
                                                const uint32_t batch_size) {
  return get_auto_batched_bits_per_slice(num_points, batch_size, 1);
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
