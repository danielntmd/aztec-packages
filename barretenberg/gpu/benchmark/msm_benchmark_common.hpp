#pragma once

#include "barretenberg/commitment_schemes/commitment_key.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace bb::gpu::msm_bench {

using Curve = bb::curve::BN254;
using Fr = Curve::ScalarField;
using Commitment = Curve::AffineElement;

constexpr int MIN_LOG_NUM_POINTS = 10;
constexpr int MAX_LOG_NUM_POINTS = 24;
constexpr size_t MAX_BENCH_NUM_POINTS = size_t{1} << MAX_LOG_NUM_POINTS;

struct BenchInput {
  explicit BenchInput(size_t num_points);

  bb::CommitmentKey<Curve> commitment_key;
  bb::Polynomial<Fr> polynomial;
};

void ensure_benchmark_crs();
std::unique_ptr<BenchInput> make_input(size_t num_points);

uint32_t auto_bits_per_slice(size_t num_points);
Commitment cpu_msm(const BenchInput &input);
void assert_equal(std::string_view backend, int log_num_points,
                  const Commitment &expected, const Commitment &actual);
bool skip_correctness_checks();

} // namespace bb::gpu::msm_bench
