#include "bn254_test_utils.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/numeric/random/engine.hpp"

#include <array>

namespace {

using namespace bb;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

TEST(GpuBn254, Fq32OpsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  const std::array<fq, 8> lhs_values = {
      fq::zero(),
      fq::one(),
      fq(2),
      -fq::one(),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
  };
  const std::array<fq, 8> rhs_values = {
      fq::one(),
      fq(3),
      -fq::one(),
      fq(5),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
  };

  for (size_t i = 0; i < lhs_values.size(); ++i) {
    gpu_testing::fq32_ops_output output{};
    gpu_testing::run_fq32_ops(gpu_testing::to_fq32_standard(lhs_values[i]),
                              gpu_testing::to_fq32_standard(rhs_values[i]),
                              output);

    gpu_testing::expect_same_standard_field(output.add,
                                            lhs_values[i] + rhs_values[i]);
    gpu_testing::expect_same_standard_field(output.sub,
                                            lhs_values[i] - rhs_values[i]);
    gpu_testing::expect_same_standard_field(output.neg, -lhs_values[i]);
    gpu_testing::expect_same_standard_field(output.dbl,
                                            lhs_values[i] + lhs_values[i]);
    gpu_testing::expect_same_standard_field(output.mul,
                                            lhs_values[i] * rhs_values[i]);
    gpu_testing::expect_same_standard_field(output.sqr, lhs_values[i].sqr());
    if (lhs_values[i].is_zero()) {
      gpu_testing::expect_fq32_zero(output.inv);
      gpu_testing::expect_fq32_zero(output.inv_product);
    } else {
      gpu_testing::expect_same_standard_field(output.inv,
                                              lhs_values[i].invert());
      gpu_testing::expect_same_standard_field(output.inv_product, fq::one());
    }
    gpu_testing::expect_same_standard_field(output.normalized_lhs,
                                            lhs_values[i]);
    gpu_testing::expect_same_standard_field(
        output.chain,
        gpu_testing::fq32_chain_reference(lhs_values[i], rhs_values[i]));
  }

  fq32_t modulus{};
  for (size_t i = 0; i < 8; ++i) {
    modulus.limbs[i] = modulus_limb(static_cast<int>(i));
  }
  const fq32_t one = fq32_t::from_u32(1);

  gpu_testing::fq32_ops_output output{};
  gpu_testing::run_fq32_ops(modulus, one, output);

  gpu_testing::expect_fq32_zero(output.normalized_lhs);
  gpu_testing::expect_same_standard_field(output.add, fq::one());
  gpu_testing::expect_same_standard_field(output.sub, -fq::one());
  gpu_testing::expect_fq32_zero(output.mul);
}

TEST(GpuBn254, FrMontgomeryAndScalarSliceMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  fr scalar = fr::random_element(&engine);
  constexpr size_t round = 3;
  constexpr size_t slice_size = 13;

  gpu_testing::fr_ops_output output{};
  gpu_testing::run_fr_ops(gpu_testing::to_host_fr_montgomery(scalar), round,
                          slice_size, output);

  fr scalar_standard = scalar.from_montgomery_form_reduced();
  gpu_testing::expect_same_standard_scalar(output.from_montgomery, scalar);
  EXPECT_EQ(output.slice,
            scalar_multiplication::legacy::MSM<curve::BN254>::get_scalar_slice(
                scalar_standard, round, slice_size));
}

} // namespace

#endif // BB_GPU_NATIVE
