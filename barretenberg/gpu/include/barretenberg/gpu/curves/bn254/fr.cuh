#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/params.cuh"
#include "barretenberg/gpu/fields/field.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

using fr_t = bb::gpu::detail::field_t<detail::Bn254FrParams>;

BB_GPU_HD inline uint32_t get_scalar_slice(const fr_t& scalar, const size_t round, const size_t slice_size)
{
    constexpr size_t NUM_BITS_IN_FIELD = 254;
    constexpr size_t LIMB_BITS = 64;
    const size_t hi_bit = NUM_BITS_IN_FIELD - (round * slice_size);
    const size_t lo_bit = (hi_bit < slice_size) ? 0 : hi_bit - slice_size;
    const size_t start_limb = lo_bit / LIMB_BITS;
    const size_t end_limb = hi_bit / LIMB_BITS;
    const size_t lo_slice_offset = lo_bit & (LIMB_BITS - 1);
    const size_t actual_slice_size = hi_bit - lo_bit;
    const size_t lo_slice_bits =
        (LIMB_BITS - lo_slice_offset < actual_slice_size) ? (LIMB_BITS - lo_slice_offset) : actual_slice_size;
    const size_t hi_slice_bits = actual_slice_size - lo_slice_bits;
    const uint64_t lo_mask = (uint64_t{ 1 } << lo_slice_bits) - 1;
    const uint64_t hi_mask = hi_slice_bits == 0 ? 0 : ((uint64_t{ 1 } << hi_slice_bits) - 1);
    const uint64_t lo_slice = (scalar.data[start_limb] >> lo_slice_offset) & lo_mask;
    const uint64_t hi_slice = (start_limb != end_limb) ? (scalar.data[end_limb] & hi_mask) : 0;
    return static_cast<uint32_t>(lo_slice | (hi_slice << lo_slice_bits));
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
