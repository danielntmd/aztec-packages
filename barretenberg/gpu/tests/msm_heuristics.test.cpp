#include "barretenberg/gpu/msm/msm_heuristics.hpp"

#ifdef BB_GPU_NATIVE

#include <gtest/gtest.h>

#include <cstddef>

namespace {

TEST(GpuMsmHeuristics, AutoWindowAvoidsPathologicalHighSlice) {
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16), 15U);
}

TEST(GpuMsmHeuristics, AutoWindowAccountsForPrecomputeFolding) {
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 1), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 2), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 4), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 12, 8), 8U);

  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 1), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 2), 8U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 4), 13U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 14, 8), 13U);

  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 1), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 2), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 4), 15U);
  EXPECT_EQ(bb::gpu::bn254::get_auto_bits_per_slice(size_t{1} << 16, 8), 15U);
}

} // namespace

#endif // BB_GPU_NATIVE
