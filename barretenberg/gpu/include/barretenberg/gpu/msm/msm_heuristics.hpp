#pragma once

#ifdef BB_GPU_NATIVE

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

constexpr uint32_t GPU_MSM_NUM_BITS_IN_FIELD = 254;
constexpr uint32_t GPU_MSM_MAX_SLICE_BITS = 20;
constexpr size_t GPU_MSM_BUCKET_ACCUMULATION_COST = 5;
constexpr size_t GPU_MSM_LARGE_BUCKET_MIN_THRESHOLD = 512;
constexpr size_t GPU_MSM_PRECOMPUTED_FOLDED_BUCKET_LIMIT =
    (GPU_MSM_LARGE_BUCKET_MIN_THRESHOLD * 3) / 4;

inline size_t ceil_div_size_t(const size_t numerator,
                              const size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

inline size_t
estimate_folded_bucket_pressure(const size_t num_points,
                                const uint32_t bits_per_slice,
                                const uint32_t precompute_factor) {
  const uint32_t original_windows =
      (GPU_MSM_NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t folded_windows =
      precompute_factor == 1 ? original_windows
                             : static_cast<uint32_t>(ceil_div_size_t(
                                   original_windows, precompute_factor));
  size_t max_pressure = 0;

  for (uint32_t target_window = 0; target_window < folded_windows;
       ++target_window) {
    size_t pressure = 0;
    for (uint32_t low_window = 0; low_window < original_windows; ++low_window) {
      const uint32_t window =
          precompute_factor == 1
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
  const size_t limit = precompute_factor == 1
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

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
