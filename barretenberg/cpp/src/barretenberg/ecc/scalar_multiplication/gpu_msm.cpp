// === AUDIT STATUS ===
// internal:    { status: not started, auditors: [], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

// GPU MSM adapter — ICICLE binding stubs.
//
// This file is ONLY compiled when the build is configured with
// -DGPU_BACKEND=icicle. It provides the out-of-line definitions for the
// functions declared in gpu_msm.hpp.
//
// FILE EXTENSION NOTE:
// This file is `.cpp` (compiled by the host C++ compiler) rather than `.cu`
// because the stub has no actual CUDA kernels. Including BB's template-heavy
// headers (bn254.hpp, polynomial.hpp) through nvcc + gcc-12 does not work in
// practice — complex C++20 template metaprogramming triggers parse failures
// in cudafe++. When real CUDA kernels are added for the ICICLE integration,
// the recommended architecture is:
//   - Keep this .cpp file as the BB-facing adapter layer (full BB types,
//     compiled by clang).
//   - Add a sibling .cu file exposing a narrow C-like interface (raw
//     pointers + sizes + opaque scalar/point structs) that this .cpp invokes.
// This layering avoids nvcc ever having to parse BB's C++20 headers.
//
// IMPLEMENTATION STATUS: scaffolding only.
//
// The function bodies below are placeholders that abort at runtime with a
// clear message. They exist so that:
//   1. The CMake build with GPU_BACKEND=icicle links cleanly (symbols resolve)
//   2. CommitmentKey's #ifdef BB_GPU_ICICLE dispatch paths compile
//   3. An operator attempting to actually run a proof with the GPU backend
//      gets an explicit error rather than a silent miscomputation
//
// The real implementation needs to:
//   - Convert BB's `curve::BN254::AffineElement` (two bb::fq coordinates, each
//     4x uint64_t in Montgomery form) to ICICLE's `affine_t<bn254::fq>`.
//     If memory layout is identical this may be a reinterpret_cast; otherwise
//     a format conversion is required.
//   - Upload SRS points to device memory in `init()` (idempotent; cache by
//     srs size).
//   - Convert BB scalars from Montgomery form to standard form before handoff
//     to ICICLE (the CPU path does this inside
//     `transform_scalar_and_get_nonzero_scalar_indices`).
//   - Call `icicle::msm::msm` / its batched variant via a .cu sibling file.
//   - Convert projective results back to affine (BN254::AffineElement).
//   - Handle the `PolynomialSpan::start_index` offset (offset into both scalar
//     and point arrays before the ICICLE call).
//   - Implement a size threshold: fall back to CPU `pippenger_unsafe` below
//     ~2^14 points to avoid PCIe-transfer overhead dominating.

#ifdef BB_GPU_ICICLE

#include "barretenberg/ecc/scalar_multiplication/gpu_msm.hpp"
#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include "icicle/api/bn254.h"
#include "icicle/curves/affine.h"
#include "icicle/curves/projective.h"
#include "icicle/device.h"
#include "icicle/fields/snark_fields/bn254_base.h"
#include "icicle/fields/snark_fields/bn254_scalar.h"
#include "icicle/runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <vector>

namespace bb::scalar_multiplication::gpu {

namespace {
using BbScalar = curve::BN254::ScalarField;
using BbBaseField = curve::BN254::BaseField;
using BbAffine = curve::BN254::AffineElement;
using BbProjective = curve::BN254::Element;

using IcicleScalar = ::bn254::scalar_t;
using IciclePointField = ::bn254::point_field_t;
using IcicleAffine = ::bn254::affine_t;
using IcicleProjective = ::bn254::projective_t;

constexpr size_t GPU_MSM_FALLBACK_THRESHOLD = 1UL << 14;

static_assert(std::is_standard_layout_v<BbScalar>);
static_assert(std::is_standard_layout_v<BbAffine>);
static_assert(std::is_standard_layout_v<BbProjective>);
static_assert(std::is_trivially_copyable_v<BbScalar>);
static_assert(std::is_trivially_copyable_v<BbAffine>);

static_assert(sizeof(BbScalar) == sizeof(IcicleScalar), "BB/ICICLE BN254 scalar size mismatch");
static_assert(sizeof(BbAffine) == sizeof(IcicleAffine), "BB/ICICLE BN254 affine size mismatch");
static_assert(sizeof(BbProjective) == sizeof(IcicleProjective), "BB/ICICLE BN254 projective size mismatch");

static_assert(alignof(BbScalar) >= alignof(IcicleScalar), "BB BN254 scalar alignment is weaker than ICICLE's");
static_assert(alignof(BbAffine) >= alignof(IcicleAffine), "BB BN254 affine alignment is weaker than ICICLE's");
static_assert(alignof(BbProjective) >= alignof(IcicleProjective),
              "BB BN254 projective alignment is weaker than ICICLE's");

static_assert(offsetof(BbAffine, x) == offsetof(IcicleAffine, x), "BB/ICICLE affine x offset mismatch");
static_assert(offsetof(BbAffine, y) == offsetof(IcicleAffine, y), "BB/ICICLE affine y offset mismatch");
static_assert(offsetof(BbProjective, x) == offsetof(IcicleProjective, x), "BB/ICICLE projective x offset mismatch");
static_assert(offsetof(BbProjective, y) == offsetof(IcicleProjective, y), "BB/ICICLE projective y offset mismatch");
static_assert(offsetof(BbProjective, z) == offsetof(IcicleProjective, z), "BB/ICICLE projective z offset mismatch");
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

eIcicleError& backend_load_status()
{
    static eIcicleError status = eIcicleError::SUCCESS;
    return status;
}

void ensure_backend_loaded()
{
    static std::once_flag once;
    std::call_once(once, []() {
        // Try env var / default (/opt/icicle/backend) first — this is what
        // production operators will set.
        auto err = icicle_load_backend_from_env_or_default();
#ifdef BB_ICICLE_BUILD_BACKEND_DIR
        // Fall back to the build-tree location baked in by CMake. This lets
        // tests and benchmarks work out-of-the-box without setting env vars.
        if (err != eIcicleError::SUCCESS) {
            err = icicle_load_backend(BB_ICICLE_BUILD_BACKEND_DIR, /*is_recursive=*/true);
        }
#endif
        backend_load_status() = err;
    });

    if (backend_load_status() != eIcicleError::SUCCESS) {
        throw_or_abort(format("ICICLE backend load failed: ", get_error_string(backend_load_status())));
    }
}

void ensure_cuda_device_selected()
{
    ensure_backend_loaded();

    const auto status = icicle_set_device(icicle::Device("CUDA", 0));
    if (status != eIcicleError::SUCCESS) {
        throw_or_abort(format("Failed to select ICICLE CUDA device: ", get_error_string(status)));
    }
}

BbProjective icicle_to_bb(const IcicleProjective& point)
{
    if (IcicleProjective::is_zero(point)) {
        return BbProjective::infinity();
    }

    // Two format mismatches to cross between ICICLE and BB:
    //
    // 1. Field representation. ICICLE stores field elements in STANDARD
    //    (non-Montgomery) form internally — see e.g. bn254::G1::gen_x = {1,0,...}
    //    rather than {R mod p, 0, ...}. BB stores them in Montgomery form. We
    //    convert each coordinate from standard → Montgomery on the way back.
    //
    // 2. Projective coordinate system. ICICLE's Projective is STANDARD
    //    projective: (x, y, z) represents affine (x/z, y/z). BB's Element is
    //    JACOBIAN: (x, y, z) represents affine (x/z^2, y/z^3). We can't just
    //    reinterpret the triple — we normalize to affine in ICICLE first, then
    //    build a BB Jacobian with z = 1.
    const IcicleAffine icicle_affine = IcicleProjective::to_affine(point);

    const auto to_bb_field = [](const IciclePointField& field) {
        BbBaseField out(static_cast<uint64_t>(field.limbs_storage.limbs64[0]),
                        static_cast<uint64_t>(field.limbs_storage.limbs64[1]),
                        static_cast<uint64_t>(field.limbs_storage.limbs64[2]),
                        static_cast<uint64_t>(field.limbs_storage.limbs64[3]));
        out.self_to_montgomery_form();
        return out;
    };

    // Construct the BB Jacobian equivalent of the affine point: (x, y, 1).
    return BbProjective(to_bb_field(icicle_affine.x), to_bb_field(icicle_affine.y), BbBaseField::one());
}

template <typename T, typename U> std::vector<T> copy_as_vector(std::span<const U> input)
{
    static_assert(sizeof(T) == sizeof(U), "layout copy requires equal-sized elements");
    std::vector<T> out(input.size());
    if (!input.empty()) {
        std::memcpy(out.data(), input.data(), input.size_bytes());
    }
    return out;
}

BbProjective run_icicle_msm(std::span<const BbScalar> scalars, std::span<const BbAffine> points)
{
    BB_BENCH_NAME("gpu::run_icicle_msm");
    BB_ASSERT_EQ(points.size(), scalars.size());
    ensure_cuda_device_selected();

    std::vector<IcicleScalar> icicle_scalars;
    std::vector<IcicleAffine> icicle_points;
    {
        BB_BENCH_NAME("gpu::run_icicle_msm/copy_scalars");
        icicle_scalars = copy_as_vector<IcicleScalar>(scalars);
    }
    {
        BB_BENCH_NAME("gpu::run_icicle_msm/copy_points");
        icicle_points = copy_as_vector<IcicleAffine>(points);
    }
    IcicleProjective icicle_result{};

    auto config = default_msm_config();
    config.batch_size = 1;
    config.are_points_shared_in_batch = true;
    config.are_scalars_on_device = false;
    config.are_scalars_montgomery_form = true;
    config.are_points_on_device = false;
    config.are_points_montgomery_form = true;
    config.are_results_on_device = false;
    config.is_async = false;

    {
        BB_BENCH_NAME("gpu::run_icicle_msm/bn254_msm");
        const auto status = bn254_msm(
            icicle_scalars.data(), icicle_points.data(), static_cast<int>(scalars.size()), &config, &icicle_result);
        if (status != eIcicleError::SUCCESS) {
            throw_or_abort(format("ICICLE BN254 MSM failed: ", get_error_string(status)));
        }
    }

    {
        BB_BENCH_NAME("gpu::run_icicle_msm/result_to_bb");
        return icicle_to_bb(icicle_result);
    }
}

} // namespace

void init(std::span<const curve::BN254::AffineElement> srs_points)
{
    ensure_backend_loaded();

    // ICICLE's affine zero is (0, 0); BB's G1 infinity sentinel is distinct.
    // Catch accidental sentinel propagation before any future reinterpret-cast
    // or upload path can hand malformed points to ICICLE.
    for (size_t i = 0; i < srs_points.size(); ++i) {
        BB_ASSERT_DEBUG(!srs_points[i].is_point_at_infinity(),
                        "Commitment SRS contains a point at infinity at index " << i);
    }

    // No-op while the GPU path is stubbed. A real implementation would upload
    // the SRS to device memory here and cache the upload size.
}

curve::BN254::Element msm(PolynomialSpan<const curve::BN254::ScalarField> scalars,
                          std::span<const curve::BN254::AffineElement> points)
{
    if (scalars.size() == 0) {
        return curve::BN254::Group::point_at_infinity;
    }

    const size_t num_scalars = scalars.size();
    BB_ASSERT_GTE(points.size(), scalars.start_index + num_scalars);

    if (num_scalars < GPU_MSM_FALLBACK_THRESHOLD) {
        return scalar_multiplication::pippenger_unsafe<curve::BN254>(scalars, points);
    }

    const auto scalar_slice = std::span<const curve::BN254::ScalarField>(scalars.span.data(), num_scalars);
    const auto point_slice = points.subspan(scalars.start_index, num_scalars);
    return run_icicle_msm(scalar_slice, point_slice);
}

std::vector<curve::BN254::AffineElement> batch_msm(std::span<std::span<const curve::BN254::AffineElement>> points,
                                                   std::span<std::span<curve::BN254::ScalarField>> scalars,
                                                   bool handle_edge_cases)
{
    BB_ASSERT(!handle_edge_cases, "GPU batch_msm currently supports only handle_edge_cases=false");

    std::vector<curve::BN254::AffineElement> results;
    results.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        const auto point_span = points[i];
        const auto scalar_span = std::span<const curve::BN254::ScalarField>(scalars[i].data(), scalars[i].size());
        BB_ASSERT_GTE(point_span.size(), scalar_span.size());

        if (scalar_span.empty()) {
            results.emplace_back(curve::BN254::Group::affine_point_at_infinity);
            continue;
        }

        const auto trimmed_points = point_span.first(scalar_span.size());
        if (scalar_span.size() < GPU_MSM_FALLBACK_THRESHOLD) {
            const PolynomialSpan<const curve::BN254::ScalarField> poly(0, scalar_span);
            results.emplace_back(scalar_multiplication::pippenger_unsafe<curve::BN254>(poly, trimmed_points));
            continue;
        }

        results.emplace_back(run_icicle_msm(scalar_span, trimmed_points));
    }

    return results;
}

void shutdown()
{
    // No-op while the GPU path is stubbed.
}

} // namespace bb::scalar_multiplication::gpu

#endif // BB_GPU_ICICLE