#include "msm/internal/msm_heuristics.hpp"

#ifdef BB_GPU_NATIVE

#include <gtest/gtest.h>

#include <cstddef>

namespace {

TEST(GpuMsmHeuristics, AutoWindowAvoidsPathologicalHighSlice) {
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16), 15U);
}

TEST(GpuMsmHeuristics, PrecomputeFactorRangeAndEffectiveClamp) {
  EXPECT_FALSE(bb::gpu::bn254::is_valid_msm_precompute_factor(0));
  EXPECT_TRUE(bb::gpu::bn254::is_valid_msm_precompute_factor(1));
  EXPECT_TRUE(bb::gpu::bn254::is_valid_msm_precompute_factor(16));
  EXPECT_FALSE(bb::gpu::bn254::is_valid_msm_precompute_factor(17));

  EXPECT_EQ(bb::gpu::bn254::get_effective_msm_precompute_factor(13, 0), 1U);
  EXPECT_EQ(bb::gpu::bn254::get_effective_msm_precompute_factor(13, 5), 5U);
  EXPECT_EQ(bb::gpu::bn254::get_effective_msm_precompute_factor(13, 16), 13U);
}

TEST(GpuMsmHeuristics, AutoWindowAccountsForPrecomputeFolding) {
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 1), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 2), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 3), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 4), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 5), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 8), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 16), 8U);

  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 1), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 2), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 3), 13U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 4), 13U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 5), 13U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 8), 13U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 16), 13U);

  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 1), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 2), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 3), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 4), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 5), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 8), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 16), 15U);
}

TEST(GpuMsmHeuristics, LargePrecomputedMsmUsesGeneralHeuristic) {
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 24, 4), 17U);
}

TEST(GpuMsmHeuristics, FusedBatchSizeValidity) {
  EXPECT_FALSE(bb::gpu::bn254::is_valid_fused_batch_size(0));
  EXPECT_TRUE(bb::gpu::bn254::is_valid_fused_batch_size(1));
  EXPECT_TRUE(bb::gpu::bn254::is_valid_fused_batch_size(
      bb::gpu::bn254::GPU_MSM_MAX_FUSED_BATCH_SIZE));
  EXPECT_FALSE(bb::gpu::bn254::is_valid_fused_batch_size(
      bb::gpu::bn254::GPU_MSM_MAX_FUSED_BATCH_SIZE + 1));
}

TEST(GpuMsmHeuristics, BatchedCostMatchesSingleMsmCost) {
  for (const size_t n : {size_t{1} << 12, size_t{1} << 16, size_t{1} << 20}) {
    for (const uint32_t k : {1U, 2U, 4U, 8U, 16U}) {
      for (uint32_t bits = 1; bits < bb::gpu::bn254::GPU_MSM_MAX_SLICE_BITS;
           ++bits) {
        EXPECT_EQ(bb::gpu::bn254::estimate_batched_msm_cost(n, k, bits),
                  bb::gpu::bn254::estimate_msm_cost(n, bits))
            << "n=" << n << " k=" << k << " bits=" << bits;
      }
    }
  }
}

TEST(GpuMsmHeuristics, BatchedBucketBytesKnownConfigurations) {
  EXPECT_EQ(bb::gpu::bn254::estimate_batched_bucket_bytes(size_t{1} << 16, 1U,
                                                          15U, 1U),
            size_t{1} * 17 * 32768 * 160);
  EXPECT_EQ(bb::gpu::bn254::estimate_batched_bucket_bytes(size_t{1} << 16, 16U,
                                                          15U, 1U),
            size_t{16} * 17 * 32768 * 160);
  EXPECT_EQ(bb::gpu::bn254::estimate_batched_bucket_bytes(size_t{1} << 16, 4U,
                                                          15U, 4U),
            size_t{4} * 5 * 32768 * 160);
  EXPECT_EQ(bb::gpu::bn254::estimate_batched_bucket_bytes(0U, 4U, 15U, 1U) > 0,
            true);
  EXPECT_EQ(bb::gpu::bn254::estimate_batched_bucket_bytes(size_t{1} << 16, 0U,
                                                          15U, 1U),
            0U);
  EXPECT_EQ(bb::gpu::bn254::estimate_batched_bucket_bytes(size_t{1} << 16, 4U,
                                                          0U, 1U),
            0U);
}

TEST(GpuMsmHeuristics, BatchedAutoMatchesSingleAtTypicalSizes) {
  for (const size_t n : {size_t{1} << 12, size_t{1} << 14, size_t{1} << 16,
                         size_t{1} << 18, size_t{1} << 20}) {
    for (const uint32_t factor : {1U, 4U, 8U, 16U}) {
      const uint32_t single =
          bb::gpu::bn254::get_auto_bits_per_slice(n, factor);
      for (const uint32_t k : {1U, 2U, 4U, 8U, 16U}) {
        EXPECT_EQ(bb::gpu::bn254::get_auto_batched_bits_per_slice(n, k, factor),
                  single)
            << "n=" << n << " k=" << k << " factor=" << factor;
      }
    }
  }
}

TEST(GpuMsmHeuristics, BatchedAutoReturnsValidBitsPerSlice) {
  for (const size_t n : {size_t{1} << 10, size_t{1} << 16, size_t{1} << 24}) {
    for (const uint32_t k : {1U, 2U, 8U, 16U}) {
      const uint32_t bits =
          bb::gpu::bn254::get_auto_batched_bits_per_slice(n, k);
      EXPECT_GE(bits, 1U) << "n=" << n << " k=" << k;
      EXPECT_LT(bits, bb::gpu::bn254::GPU_MSM_MAX_SLICE_BITS)
          << "n=" << n << " k=" << k;
    }
  }
}

TEST(GpuMsmHeuristics, BatchedPathologicalPressureMatchesSingleAtK1) {
  for (const size_t n : {size_t{1} << 12, size_t{1} << 16, size_t{1} << 20}) {
    for (const uint32_t factor : {1U, 4U, 8U, 16U}) {
      for (uint32_t bits = 1; bits < bb::gpu::bn254::GPU_MSM_MAX_SLICE_BITS;
           ++bits) {
        EXPECT_EQ(
            bb::gpu::bn254::has_pathological_batched_bucket_pressure(
                n, 1U, bits, factor),
            bb::gpu::bn254::has_pathological_bucket_pressure(n, bits, factor))
            << "n=" << n << " bits=" << bits << " factor=" << factor;
      }
    }
  }
}

} // namespace

#endif // BB_GPU_NATIVE
