/**
 * @brief CPU-vs-GPU MSM differential smoke tests.
 *
 * These tests exercise the ICICLE GPU adapter in gpu_msm.hpp/cpp by comparing
 * its results against the reference CPU Pippenger path. They are only compiled
 * when BB_GPU_ICICLE is defined (GPU_BACKEND=icicle).
 *
 * Key coverage:
 *   - Single MSM above and below the GPU_MSM_FALLBACK_THRESHOLD
 *   - PolynomialSpan start_index handling
 *   - Batch MSM
 *   - Scalar preservation (GPU must not mutate caller's scalars)
 *   - Empty / single-element edge cases
 */

#ifdef BB_GPU_ICICLE

#include "barretenberg/ecc/scalar_multiplication/gpu_msm.hpp"
#include "barretenberg/commitment_schemes/commitment_key.hpp"
#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "barretenberg/srs/global_crs.hpp"

#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>

namespace bb {
namespace {

auto& engine = numeric::get_randomness();

using BN254 = curve::BN254;
using Fr = BN254::ScalarField;
using AffineElement = BN254::AffineElement;
using Element = BN254::Element;
using Polynomial = bb::Polynomial<Fr>;

// Mirror the threshold from gpu_msm.cpp so tests can straddle it.
constexpr size_t GPU_MSM_FALLBACK_THRESHOLD = 1UL << 14;

class GpuMsmTest : public ::testing::Test {
  public:
    static void SetUpTestSuite() { srs::init_file_crs_factory(srs::bb_crs_path()); }

    static std::vector<Fr> random_scalars(size_t n)
    {
        std::vector<Fr> s(n);
        for (auto& v : s) {
            v = Fr::random_element(&engine);
        }
        return s;
    }

    /// CPU reference: compute commitment via pippenger_unsafe.
    static Element cpu_msm(PolynomialSpan<const Fr> scalars, std::span<const AffineElement> points)
    {
        return scalar_multiplication::pippenger_unsafe<BN254>(scalars, points);
    }

    /// CPU reference for batch: MSM<BN254>::batch_multi_scalar_mul.
    static std::vector<AffineElement> cpu_batch_msm(std::span<std::span<const AffineElement>> points,
                                                    std::span<std::span<Fr>> scalars)
    {
        return scalar_multiplication::MSM<BN254>::batch_multi_scalar_mul(points, scalars, false);
    }
};

// ---------------------------------------------------------------------------
// Single MSM: below threshold (should fall back to CPU internally)
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, SingleMsmBelowThreshold)
{
    constexpr size_t n = GPU_MSM_FALLBACK_THRESHOLD - 1;
    CommitmentKey<BN254> ck(n);
    auto scalars = random_scalars(n);
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(0, scalars);
    Element gpu_result = scalar_multiplication::gpu::msm(poly, points);
    Element cpu_result = cpu_msm(poly, points);

    EXPECT_EQ(AffineElement(gpu_result), AffineElement(cpu_result));
}

// ---------------------------------------------------------------------------
// Single MSM: at threshold (first size that hits the ICICLE path)
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, SingleMsmAtThreshold)
{
    constexpr size_t n = GPU_MSM_FALLBACK_THRESHOLD;
    CommitmentKey<BN254> ck(n);
    auto scalars = random_scalars(n);
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(0, scalars);
    Element gpu_result = scalar_multiplication::gpu::msm(poly, points);
    Element cpu_result = cpu_msm(poly, points);

    EXPECT_EQ(AffineElement(gpu_result), AffineElement(cpu_result));
}

// ---------------------------------------------------------------------------
// Single MSM: substantially above threshold
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, SingleMsmLarge)
{
    constexpr size_t n = 1UL << 16; // 65536
    CommitmentKey<BN254> ck(n);
    auto scalars = random_scalars(n);
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(0, scalars);
    Element gpu_result = scalar_multiplication::gpu::msm(poly, points);
    Element cpu_result = cpu_msm(poly, points);

    EXPECT_EQ(AffineElement(gpu_result), AffineElement(cpu_result));
}

// ---------------------------------------------------------------------------
// Single MSM with non-zero start_index
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, SingleMsmWithStartIndex)
{
    constexpr size_t start_index = 100;
    constexpr size_t poly_size = GPU_MSM_FALLBACK_THRESHOLD; // above threshold
    constexpr size_t total_srs = start_index + poly_size;
    CommitmentKey<BN254> ck(total_srs);
    auto scalars = random_scalars(poly_size);
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(start_index, scalars);
    Element gpu_result = scalar_multiplication::gpu::msm(poly, points);
    Element cpu_result = cpu_msm(poly, points);

    EXPECT_EQ(AffineElement(gpu_result), AffineElement(cpu_result));
}

// ---------------------------------------------------------------------------
// Batch MSM: mix of sizes (some above threshold, some below)
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, BatchMsmMixedSizes)
{
    // Sizes chosen so some hit ICICLE, some hit CPU fallback.
    std::vector<size_t> sizes = { 100, GPU_MSM_FALLBACK_THRESHOLD + 1, 50, GPU_MSM_FALLBACK_THRESHOLD * 2 };
    size_t max_size = *std::max_element(sizes.begin(), sizes.end());
    CommitmentKey<BN254> ck(max_size);
    std::span<const AffineElement> srs = ck.get_monomial_points();

    // Build scalar + point spans
    std::vector<std::vector<Fr>> scalar_vecs;
    std::vector<std::span<const AffineElement>> point_spans;
    std::vector<std::span<Fr>> scalar_spans;
    for (size_t n : sizes) {
        scalar_vecs.push_back(random_scalars(n));
        point_spans.push_back(srs.first(n));
        scalar_spans.push_back(scalar_vecs.back());
    }

    // GPU batch
    auto gpu_results = scalar_multiplication::gpu::batch_msm(point_spans, scalar_spans, false);

    // CPU reference (need copies because CPU mutates scalars then restores)
    std::vector<std::vector<Fr>> scalar_copies;
    std::vector<std::span<Fr>> cpu_scalar_spans;
    for (auto& v : scalar_vecs) {
        scalar_copies.push_back(v); // copy
        cpu_scalar_spans.push_back(scalar_copies.back());
    }
    auto cpu_results = cpu_batch_msm(point_spans, cpu_scalar_spans);

    ASSERT_EQ(gpu_results.size(), cpu_results.size());
    for (size_t i = 0; i < gpu_results.size(); ++i) {
        EXPECT_EQ(gpu_results[i], cpu_results[i]) << "Mismatch in MSM #" << i << " (size=" << sizes[i] << ")";
    }
}

// ---------------------------------------------------------------------------
// Scalar preservation: GPU adapter must not mutate the input scalars
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, ScalarPreservation)
{
    constexpr size_t n = GPU_MSM_FALLBACK_THRESHOLD; // above threshold to hit ICICLE
    CommitmentKey<BN254> ck(n);
    auto scalars = random_scalars(n);
    auto scalars_copy = scalars; // snapshot before
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(0, scalars);
    scalar_multiplication::gpu::msm(poly, points);

    EXPECT_EQ(scalars, scalars_copy) << "GPU msm() mutated the caller's scalar buffer";
}

// ---------------------------------------------------------------------------
// Empty input: should return point at infinity
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, EmptyInput)
{
    CommitmentKey<BN254> ck(16);
    std::vector<Fr> empty;
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(0, empty);
    Element result = scalar_multiplication::gpu::msm(poly, points);

    EXPECT_TRUE(result.is_point_at_infinity());
}

// ---------------------------------------------------------------------------
// Single element: simplest non-trivial case
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, SingleElement)
{
    CommitmentKey<BN254> ck(16);
    auto scalars = random_scalars(1);
    std::span<const AffineElement> points = ck.get_monomial_points();

    PolynomialSpan<const Fr> poly(0, scalars);
    Element gpu_result = scalar_multiplication::gpu::msm(poly, points);
    Element cpu_result = cpu_msm(poly, points);

    EXPECT_EQ(AffineElement(gpu_result), AffineElement(cpu_result));
}

// ---------------------------------------------------------------------------
// End-to-end commitment equality through CommitmentKey::commit
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, CommitmentKeyCommitLarge)
{
    constexpr size_t n = GPU_MSM_FALLBACK_THRESHOLD * 2; // well above threshold
    CommitmentKey<BN254> ck(n);
    auto poly = Polynomial::random(n);

    // GPU path (via CommitmentKey dispatch)
    AffineElement gpu_commitment = ck.commit(poly);

    // CPU reference: manual pippenger_unsafe call
    std::span<const AffineElement> points = ck.get_monomial_points();
    PolynomialSpan<const Fr> poly_span(poly);
    Element cpu_result = cpu_msm(poly_span, points);

    EXPECT_EQ(gpu_commitment, AffineElement(cpu_result));
}

// ---------------------------------------------------------------------------
// End-to-end batch commit through CommitmentKey::batch_commit
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, CommitmentKeyBatchCommitLarge)
{
    constexpr size_t n = GPU_MSM_FALLBACK_THRESHOLD * 2;
    constexpr size_t num_polys = 4;
    CommitmentKey<BN254> ck(n);

    std::vector<Polynomial> polys;
    for (size_t i = 0; i < num_polys; ++i) {
        polys.emplace_back(Polynomial::random(n));
    }
    RefVector<Polynomial> poly_refs(polys);
    auto gpu_commitments = ck.batch_commit(poly_refs);

    // CPU reference: individual commits via pippenger
    std::span<const AffineElement> points = ck.get_monomial_points();
    for (size_t i = 0; i < num_polys; ++i) {
        PolynomialSpan<const Fr> poly_span(polys[i]);
        Element cpu_result = cpu_msm(poly_span, points);
        EXPECT_EQ(gpu_commitments[i], AffineElement(cpu_result)) << "Mismatch in poly #" << i;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: wall-clock CPU vs GPU at multiple sizes, with BB_BENCH breakdown
//
// Gated behind a DISABLED_ prefix so it does not run in CI. Invoke with:
//   BB_BENCH=1 ./bin/commitment_schemes_tests \
//       --gtest_also_run_disabled_tests \
//       --gtest_filter=GpuMsmTest.DISABLED_Benchmark
//
// The BB_BENCH hierarchical breakdown (printed at process exit) attributes
// time to copy_scalars / copy_points / bn254_msm / result_to_bb — so you can
// see how much of the GPU wall time is host-side marshalling vs the kernel.
// ---------------------------------------------------------------------------
TEST_F(GpuMsmTest, DISABLED_Benchmark)
{
    constexpr std::array<size_t, 4> size_exponents = { 14, 16, 18, 20 };
    constexpr size_t warmup = 2;
    constexpr size_t iterations = 5;

    // Enable BB_BENCH collection so the hierarchical breakdown prints on exit.
    bb::detail::use_bb_bench = true;

    std::printf("\n%6s | %10s | %12s | %12s | %8s\n", "size", "points", "GPU ms/iter", "CPU ms/iter", "speedup");
    std::printf("-------+------------+--------------+--------------+--------\n");

    for (size_t exp : size_exponents) {
        const size_t n = 1UL << exp;
        CommitmentKey<BN254> ck(n);
        auto scalars = random_scalars(n);
        std::span<const AffineElement> points = ck.get_monomial_points();
        PolynomialSpan<const Fr> poly(0, scalars);

        // Warmup (pays one-time GPU / backend init cost so it doesn't skew timing).
        for (size_t i = 0; i < warmup; ++i) {
            volatile auto r1 = scalar_multiplication::gpu::msm(poly, points);
            volatile auto r2 = cpu_msm(poly, points);
            (void)r1;
            (void)r2;
        }

        // GPU timing
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            auto r = scalar_multiplication::gpu::msm(poly, points);
            (void)r;
        }
        auto t1 = std::chrono::steady_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        // CPU timing
        t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            auto r = cpu_msm(poly, points);
            (void)r;
        }
        t1 = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        char size_label[16];
        std::snprintf(size_label, sizeof(size_label), "2^%zu", exp);
        std::printf("%6s | %10zu | %12.2f | %12.2f | %7.2fx\n", size_label, n, gpu_ms, cpu_ms, cpu_ms / gpu_ms);
    }
    std::printf("\n(Detailed GPU component timing follows below from BB_BENCH.)\n\n");
}

} // namespace
} // namespace bb

#endif // BB_GPU_ICICLE
