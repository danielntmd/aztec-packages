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
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include "icicle/curves/affine.h"
#include "icicle/curves/projective.h"
#include "icicle/fields/snark_fields/bn254_base.h"
#include "icicle/fields/snark_fields/bn254_scalar.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace bb::scalar_multiplication::gpu {

namespace {
using BbScalar = curve::BN254::ScalarField;
using BbAffine = curve::BN254::AffineElement;
using BbProjective = curve::BN254::Element;

struct IcicleBn254G1Tag;
using IciclePointField = ::Field<::bn254::fq_config>;
using IcicleScalar = ::bn254::scalar_t;
using IcicleAffine = ::Affine<IciclePointField>;
using IcicleProjective = ::Projective<IciclePointField, IcicleScalar, IcicleBn254G1Tag>;

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

[[noreturn]] void not_implemented(const char* fn)
{
    // Intentional abort: the GPU backend was compiled in but the ICICLE
    // integration has not been wired up yet. This matches the CPU fallback
    // contract — the caller must not silently accept a wrong commitment.
    std::fprintf(stderr,
                 "bb::scalar_multiplication::gpu::%s is not implemented yet. "
                 "Build with -DGPU_BACKEND=none, or finish the ICICLE integration.\n",
                 fn);
    std::abort();
}
} // namespace

void init(std::span<const curve::BN254::AffineElement> srs_points)
{
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

curve::BN254::Element msm(PolynomialSpan<const curve::BN254::ScalarField> /*scalars*/,
                          std::span<const curve::BN254::AffineElement> /*points*/)
{
    not_implemented("msm");
}

std::vector<curve::BN254::AffineElement> batch_msm(std::span<std::span<const curve::BN254::AffineElement>> /*points*/,
                                                   std::span<std::span<curve::BN254::ScalarField>> /*scalars*/,
                                                   bool /*handle_edge_cases*/)
{
    not_implemented("batch_msm");
}

void shutdown()
{
    // No-op while the GPU path is stubbed.
}

} // namespace bb::scalar_multiplication::gpu

#endif // BB_GPU_ICICLE