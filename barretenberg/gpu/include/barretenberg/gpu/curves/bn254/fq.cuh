#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/params.cuh"
#include "barretenberg/gpu/fields/field.cuh"

namespace bb::gpu::bn254 {

using fq_t = bb::gpu::detail::field_t<detail::Bn254FqParams>;

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
