#include "barretenberg/gpu/msm/msm.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/device_context.hpp"
#include "bn254_test_kernels.hpp"
#include "barretenberg/numeric/random/engine.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

#define BB_REQUIRE_CUDA_DEVICE()                                                                                      \
    do {                                                                                                              \
        if (const char* device_status = gpu_testing::cuda_device_status()) {                                           \
            GTEST_SKIP() << "No CUDA-capable device is available: " << device_status;                                 \
        }                                                                                                             \
    } while (false)

fq_t to_gpu(const fq& value)
{
    return fq_t::raw(value.data[0], value.data[1], value.data[2], value.data[3]);
}

fr_t to_gpu(const fr& value)
{
    return fr_t::raw(value.data[0], value.data[1], value.data[2], value.data[3]);
}

affine_g1_t to_gpu(const curve::BN254::AffineElement& value)
{
    return { to_gpu(value.x), to_gpu(value.y) };
}

fq to_cpu(const fq_t& value)
{
    return { value.data[0], value.data[1], value.data[2], value.data[3] };
}

curve::BN254::AffineElement to_cpu(const affine_g1_t& value)
{
    curve::BN254::AffineElement out{ to_cpu(value.x), to_cpu(value.y) };
    return out.is_point_at_infinity() ? curve::BN254::AffineElement::infinity() : out;
}

void expect_same_raw(const fq_t& actual, const fq& expected)
{
    EXPECT_EQ(actual.data[0], expected.data[0]);
    EXPECT_EQ(actual.data[1], expected.data[1]);
    EXPECT_EQ(actual.data[2], expected.data[2]);
    EXPECT_EQ(actual.data[3], expected.data[3]);
}

void expect_same_raw(const fr_t& actual, const fr& expected)
{
    EXPECT_EQ(actual.data[0], expected.data[0]);
    EXPECT_EQ(actual.data[1], expected.data[1]);
    EXPECT_EQ(actual.data[2], expected.data[2]);
    EXPECT_EQ(actual.data[3], expected.data[3]);
}

void expect_same_field(const fq_t& actual, const fq& expected)
{
    EXPECT_EQ(to_cpu(actual), expected);
}

void expect_same_point(const affine_g1_t& actual, const curve::BN254::AffineElement& expected)
{
    EXPECT_EQ(to_cpu(actual), expected);
}

curve::BN254::Element reduce_window_buckets_reference(std::vector<curve::BN254::Element>& buckets,
                                                      const std::vector<bool>& bucket_exists)
{
    curve::BN254::Element running_sum = curve::BN254::Group::point_at_infinity;
    curve::BN254::Element sum = curve::BN254::Group::point_at_infinity;
    for (size_t i = buckets.size() - 1; i > 0; --i) {
        if (bucket_exists[i]) {
            running_sum += buckets[i];
        }
        sum += running_sum;
    }
    return sum;
}

curve::BN254::AffineElement reference_msm_with_explicit_window(
    std::span<const curve::BN254::AffineElement> points,
    PolynomialSpan<const curve::BN254::ScalarField> scalars,
    const uint32_t bits_per_slice)
{
    constexpr size_t NUM_BITS_IN_FIELD = scalar_multiplication::MSM<curve::BN254>::NUM_BITS_IN_FIELD;
    const size_t num_windows = (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
    const size_t num_buckets = size_t{ 1 } << bits_per_slice;
    const size_t remainder = NUM_BITS_IN_FIELD % bits_per_slice;

    std::vector<fr> standard_scalars;
    standard_scalars.reserve(scalars.size());
    for (size_t i = 0; i < scalars.size(); ++i) {
        standard_scalars.emplace_back(scalars.span[i].from_montgomery_form_reduced());
    }

    curve::BN254::Element result = curve::BN254::Group::point_at_infinity;
    std::vector<curve::BN254::Element> buckets(num_buckets);
    std::vector<bool> bucket_exists(num_buckets);
    for (size_t round = 0; round < num_windows; ++round) {
        std::fill(bucket_exists.begin(), bucket_exists.end(), false);
        for (size_t i = 0; i < standard_scalars.size(); ++i) {
            const uint32_t bucket =
                scalar_multiplication::MSM<curve::BN254>::get_scalar_slice(standard_scalars[i], round, bits_per_slice);
            if (bucket == 0) {
                continue;
            }
            const auto& point = points[scalars.start_index + i];
            if (bucket_exists[bucket]) {
                buckets[bucket] += point;
            } else {
                buckets[bucket] = point;
                bucket_exists[bucket] = true;
            }
        }

        const curve::BN254::Element window_result = reduce_window_buckets_reference(buckets, bucket_exists);
        const size_t num_doublings = (round == num_windows - 1 && remainder != 0) ? remainder : bits_per_slice;
        for (size_t i = 0; i < num_doublings; ++i) {
            result.self_dbl();
        }
        result += window_result;
    }
    return curve::BN254::AffineElement(result);
}

} // namespace

TEST(GpuBn254, FqOpsMatchCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    std::array<fq, 5> lhs_values = {
        fq::zero(),
        fq::one(),
        fq(2),
        fq::random_element(&engine),
        fq::random_element(&engine),
    };
    std::array<fq, 5> rhs_values = {
        fq::one(),
        fq(3),
        fq::random_element(&engine),
        fq::random_element(&engine),
        fq::random_element(&engine),
    };

    for (size_t i = 0; i < lhs_values.size(); ++i) {
        gpu_testing::fq_ops_output output{};
        gpu_testing::run_fq_ops(to_gpu(lhs_values[i]), to_gpu(rhs_values[i]), output);

        expect_same_field(output.add, lhs_values[i] + rhs_values[i]);
        expect_same_field(output.sub, lhs_values[i] - rhs_values[i]);
        expect_same_field(output.neg, -lhs_values[i]);
        expect_same_field(output.dbl, lhs_values[i] + lhs_values[i]);
        expect_same_field(output.mul, lhs_values[i] * rhs_values[i]);
        expect_same_field(output.sqr, lhs_values[i].sqr());
        expect_same_raw(output.from_montgomery, lhs_values[i].from_montgomery_form_reduced());
        EXPECT_EQ(output.eq, lhs_values[i] == rhs_values[i]);
        EXPECT_EQ(output.is_zero, lhs_values[i].is_zero());
        if (!lhs_values[i].is_zero()) {
            expect_same_field(output.inv, lhs_values[i].invert());
        }
    }
}

TEST(GpuBn254, FrMontgomeryAndScalarSliceMatchCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    fr scalar = fr::random_element(&engine);
    constexpr size_t round = 3;
    constexpr size_t slice_size = 13;

    gpu_testing::fr_ops_output output{};
    gpu_testing::run_fr_ops(to_gpu(scalar), round, slice_size, output);

    fr scalar_standard = scalar.from_montgomery_form_reduced();
    expect_same_raw(output.from_montgomery, scalar_standard);
    EXPECT_EQ(output.slice, scalar_multiplication::MSM<curve::BN254>::get_scalar_slice(scalar_standard, round, slice_size));
}

TEST(GpuBn254, G1OpsMatchCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    curve::BN254::AffineElement lhs = curve::BN254::AffineElement::random_element(&engine);
    curve::BN254::AffineElement rhs = curve::BN254::AffineElement::random_element(&engine);

    gpu_testing::g1_ops_output output{};
    gpu_testing::run_g1_ops(to_gpu(lhs), to_gpu(rhs), output);

    curve::BN254::Element lhs_element(lhs);
    curve::BN254::Element rhs_element(rhs);

    expect_same_point(output.mixed_add, curve::BN254::AffineElement(lhs_element + rhs));
    expect_same_point(output.jacobian_add, curve::BN254::AffineElement(lhs_element + rhs_element));
    expect_same_point(output.dbl, curve::BN254::AffineElement(lhs_element.dbl()));
    expect_same_point(output.neg, -lhs);
    EXPECT_TRUE(output.on_curve_lhs);
    EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, G1EdgeCasesMatchCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    curve::BN254::AffineElement generator = curve::BN254::Group::affine_one;
    curve::BN254::AffineElement infinity = curve::BN254::AffineElement::infinity();

    gpu_testing::g1_ops_output output{};
    gpu_testing::run_g1_ops(to_gpu(generator), to_gpu(generator), output);
    expect_same_point(output.mixed_add, curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
    expect_same_point(output.jacobian_add, curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));

    gpu_testing::run_g1_ops(to_gpu(generator), to_gpu(-generator), output);
    expect_same_point(output.mixed_add, infinity);
    expect_same_point(output.jacobian_add, infinity);

    gpu_testing::run_g1_ops(to_gpu(infinity), to_gpu(generator), output);
    expect_same_point(output.mixed_add, generator);
    expect_same_point(output.jacobian_add, generator);
    EXPECT_TRUE(output.on_curve_lhs);
    EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, G1ChainedMixedAddMatchesCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    const curve::BN254::AffineElement lhs = curve::BN254::AffineElement::random_element(&engine);
    const curve::BN254::AffineElement rhs = curve::BN254::AffineElement::random_element(&engine);
    const curve::BN254::AffineElement tail = curve::BN254::AffineElement::random_element(&engine);
    const curve::BN254::AffineElement infinity = curve::BN254::AffineElement::infinity();

    const std::vector<std::vector<curve::BN254::AffineElement>> cases = {
        { lhs, rhs, tail, curve::BN254::AffineElement::random_element(&engine) },
        { lhs, lhs, tail },
        { lhs, -lhs, tail },
        { infinity, lhs, rhs, tail },
    };

    for (const auto& points : cases) {
        std::vector<affine_g1_t> gpu_points;
        gpu_points.reserve(points.size());
        for (const auto& point : points) {
            gpu_points.emplace_back(to_gpu(point));
        }

        affine_g1_t output{};
        gpu_testing::run_g1_chained_mixed_add(gpu_points.data(), gpu_points.size(), output);

        curve::BN254::Element expected = curve::BN254::Group::point_at_infinity;
        for (const auto& point : points) {
            expected += point;
        }
        expect_same_point(output, curve::BN254::AffineElement(expected));
    }
}

TEST(GpuBn254, MsmAllZeroAndEmptyReturnInfinity)
{
    BB_REQUIRE_CUDA_DEVICE();

    std::vector<curve::BN254::AffineElement> points(4, curve::BN254::Group::affine_one);
    std::vector<fr> empty_scalars;
    EXPECT_EQ(bb::gpu::bn254::msm({ 0, std::span<const fr>(empty_scalars.data(), empty_scalars.size()) }, points, 4),
              curve::BN254::AffineElement::infinity());

    std::vector<fr> zero_scalars(4, fr::zero());
    EXPECT_EQ(bb::gpu::bn254::msm({ 0, std::span<const fr>(zero_scalars.data(), zero_scalars.size()) }, points, 4),
              curve::BN254::AffineElement::infinity());
}

TEST(GpuBn254, MsmExplicitWindowsMatchCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    std::vector<curve::BN254::AffineElement> points;
    std::vector<fr> scalars;
    for (size_t i = 0; i < 18; ++i) {
        points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
        scalars.emplace_back(i % 5 == 0 ? fr::zero() : fr::random_element(&engine));
    }

    for (uint32_t bits_per_slice : { 1U, 4U, 8U, 13U }) {
        auto scalar_span = PolynomialSpan<const fr>{ 0, std::span<const fr>(scalars.data(), scalars.size()) };
        const auto expected = reference_msm_with_explicit_window(points, scalar_span, bits_per_slice);
        const auto actual = bb::gpu::bn254::msm(scalar_span, points, bits_per_slice);
        EXPECT_EQ(actual, expected) << "bits_per_slice=" << bits_per_slice;
    }
}

TEST(GpuBn254, MsmStartIndexMatchesCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    std::vector<curve::BN254::AffineElement> points;
    for (size_t i = 0; i < 10; ++i) {
        points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    }
    std::vector<fr> scalars = { fr::random_element(&engine), fr::zero(), fr::random_element(&engine), fr(17) };
    auto scalar_span = PolynomialSpan<const fr>{ 3, std::span<const fr>(scalars.data(), scalars.size()) };

    const auto expected = reference_msm_with_explicit_window(points, scalar_span, 5);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 5);
    EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, MsmSameBucketNormalAccumulationAndReductionMatchCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    std::vector<curve::BN254::AffineElement> points;
    std::vector<fr> scalars;
    for (size_t i = 0; i < 6; ++i) {
        points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
        scalars.emplace_back(fr(5));
    }
    auto scalar_span = PolynomialSpan<const fr>{ 0, std::span<const fr>(scalars.data(), scalars.size()) };

    const auto expected = reference_msm_with_explicit_window(points, scalar_span, 4);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
    EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, MsmLargeBucketAccumulationMatchesCpu)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    std::vector<curve::BN254::AffineElement> points;
    std::vector<fr> scalars;
    for (size_t i = 0; i < 600; ++i) {
        points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
        scalars.emplace_back(fr(5));
    }
    auto scalar_span = PolynomialSpan<const fr>{ 0, std::span<const fr>(scalars.data(), scalars.size()) };

    const auto expected = reference_msm_with_explicit_window(points, scalar_span, 4);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
    EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, MsmAutoWindowMatchesCpuSafePippenger)
{
    BB_REQUIRE_CUDA_DEVICE();

    auto& engine = numeric::get_debug_randomness();
    std::vector<curve::BN254::AffineElement> points;
    std::vector<fr> scalars;
    for (size_t i = 0; i < 32; ++i) {
        points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
        scalars.emplace_back(i % 7 == 0 ? fr::zero() : fr::random_element(&engine));
    }

    auto scalar_span = PolynomialSpan<const fr>{ 0, std::span<const fr>(scalars.data(), scalars.size()) };
    const auto expected = curve::BN254::AffineElement(
        scalar_multiplication::pippenger<curve::BN254>(scalar_span, points, /*handle_edge_cases=*/true));
    const auto actual = bb::gpu::bn254::msm(scalar_span, points);
    EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, DeviceBufferCopiesRoundTrip)
{
    BB_REQUIRE_CUDA_DEVICE();

    std::vector<fq_t> input = { to_gpu(fq::zero()), to_gpu(fq::one()), to_gpu(fq(17)) };
    std::vector<fq_t> output(input.size());
    DeviceBuffer<fq_t> buffer;

    copy_to_device(buffer, std::span<const fq_t>(input.data(), input.size()), default_context().stream());
    copy_to_host(std::span<fq_t>(output.data(), output.size()), buffer, default_context().stream());
    default_context().sync();

    EXPECT_EQ(std::memcmp(input.data(), output.data(), sizeof(fq_t) * input.size()), 0);
}

#endif // BB_GPU_NATIVE
