#include "barretenberg/gpu/curves/bn254/fq.cuh"
#include "barretenberg/gpu/curves/bn254/fq32.cuh"
#include "barretenberg/gpu/curves/bn254/g1.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef BB_GPU_HAVE_ICICLE_V28
#include "curves/params/bn254.cuh"
#endif

namespace {

constexpr int THREADS_PER_BLOCK = 256;

using bb_fq_t = bb::gpu::bn254::fq_t;
using bb_affine_t = bb::gpu::bn254::affine_g1_t;
using bb_jacobian_t = bb::gpu::bn254::jacobian_g1_t;
using bb_xyzz_t = bb::gpu::bn254::xyzz_g1_t;
using exp_fq32_t = bb::gpu::bn254::experimental::fq32_t;

enum class bench_case : int {
  BB_ADD = 0,
  BB_MUL = 1,
  BB_SQR = 2,
  BB_XYZZ_MIXED_ADD = 3,
  ICICLE_ADD = 4,
  ICICLE_MUL = 5,
  ICICLE_SQR = 6,
  ICICLE_PROJECTIVE_MIXED_ADD = 7,
  BB_ADD_NO_PREREDUCE = 8,
  BB_MUL_NO_PREREDUCE = 9,
  BB32_ADD = 10,
  BB32_MUL = 11,
  BB_XYZZ_MIXED_ADD_UNCHECKED = 12,
  BB_JACOBIAN_MIXED_ADD_UNCHECKED = 13,
  ICICLE_XYZZ_MIXED_ADD_UNCHECKED = 14,
  ICICLE_XYZZ_MIXED_ADD_CHECKED = 15,
  BB32_PTX_MUL = 16,
  BB32_BARRETT_MUL = 17,
  BB32_BARRETT_TRUNC_MUL = 18,
  BB32_BARRETT_TRUNC_PTX_MUL = 19,
  FQ32_STRAIGHTLINE_MUL = 20,
  FQ32_STRAIGHTLINE_SQR = 21,
  FQ32_KARATSUBA_MUL = 22,
  FQ32_WIDE_PRODUCT = 23,
  FQ32_STRAIGHTLINE_WIDE_PRODUCT = 24,
  FQ32_KARATSUBA_WIDE_PRODUCT = 25,
  BB_SQR_NO_PREREDUCE = 26,
  BB_SQR_DEDICATED_NO_PREREDUCE = 27,
  BB_XYZZ_MIXED_ADD_ASSUME_FINITE = 28,
  FQ32_CALLABLE_MUL = 29,
  FQ32_CALLABLE_SQR = 30,
  FQ32_DEDICATED_SQR = 31,
  FQ32_REDUCE_ONLY_BARRETT = 32,
  FQ32_CURVE_SHAPE_XYZZ_MIXED_ADD = 33,
  FQ32_CURVE_SHAPE_XYZZ_ADD = 34,
  FQ32_CURVE_SHAPE_XYZZ_DOUBLE = 35,
  BB_CURVE_SHAPE_XYZZ_ADD = 36,
  BB_CURVE_SHAPE_XYZZ_DOUBLE = 37,
  BB_SUB = 38,
  BB_SUB_NO_PREREDUCE = 39,
  BB_NEG = 40,
  BB_IS_ZERO = 41,
  BB_EQUAL = 42,
  ICICLE_SUB = 43,
  ICICLE_NEG = 44,
  ICICLE_IS_ZERO = 45,
  ICICLE_EQUAL = 46,
  ICICLE_WIDE_PRODUCT = 47,
  ICICLE_REDUCE_ONLY = 48,
  FQ32_HALF_PRODUCT_MUL = 49,
  FQ32_HALF_PRODUCT_WIDE_PRODUCT = 50,
  FQ32_ADD = 51,
  FQ32_SUB = 52,
  FQ32_NEG = 53,
  FQ32_IS_ZERO = 54,
  FQ32_EQUAL = 55,
  FQ32_REDUCE_ONLY_BARRETT_REPRESENTATIVE = 56,
  FQ32_KARATSUBA_FUSED_MUL = 57,
  FQ32_KARATSUBA_FUSED_WIDE_PRODUCT = 58,
};

void check_cuda(const cudaError_t error) {
  if (error != cudaSuccess) {
    std::abort();
  }
}

template <typename Launch> float time_cuda_launch(Launch &&launch) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));
  check_cuda(cudaEventRecord(start));
  launch();
  check_cuda(cudaEventRecord(stop));
  check_cuda(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop));
  check_cuda(cudaEventDestroy(start));
  check_cuda(cudaEventDestroy(stop));
  return elapsed_ms;
}

__device__ bb_fq_t make_bb_fq(const uint32_t seed) {
  return bb_fq_t::from_u64(static_cast<uint64_t>(seed) + 1);
}

struct alignas(32) bb32_fq_t {
  uint32_t limbs[8];
};

struct alignas(64) exp_fq32_wide_t {
  uint32_t limbs[16];
};

struct alignas(32) exp_fq32_xyzz_t {
  exp_fq32_t x;
  exp_fq32_t y;
  exp_fq32_t zz;
  exp_fq32_t zzz;
  bool infinity;
};

__device__ __forceinline__ uint32_t mix_u32(uint32_t value) {
  value += 0x9e3779b9U;
  value = (value ^ (value >> 16)) * 0x85ebca6bU;
  value = (value ^ (value >> 13)) * 0xc2b2ae35U;
  return value ^ (value >> 16);
}

__device__ __forceinline__ exp_fq32_t make_random_fq32(const uint32_t index,
                                                       const uint32_t stream) {
  constexpr uint32_t seed = 0x6d2b79f5U;
  exp_fq32_t out{};
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    out.limbs[i] = mix_u32(seed ^ (index * 0x9e3779b1U) ^
                           (stream * 0x85ebca77U) ^
                           (static_cast<uint32_t>(i) * 0xc2b2ae3dU));
  }
  out.limbs[7] =
      mix_u32(seed ^ (index * 0x27d4eb2dU) ^ (stream * 0x165667b1U)) %
      (bb::gpu::bn254::experimental::modulus_limb(7) + 1U);
  if (bb::gpu::bn254::experimental::ge_modulus(out)) {
    bb::gpu::bn254::experimental::sub_modulus_in_place(out);
  }
  return out;
}

__device__ __forceinline__ bool fq32_equal(const exp_fq32_t &lhs,
                                           const exp_fq32_t &rhs) {
  uint32_t diff = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    diff |= lhs.limbs[i] ^ rhs.limbs[i];
  }
  return diff == 0;
}

__device__ __forceinline__ exp_fq32_t fq32_modulus_minus_one() {
  exp_fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = bb::gpu::bn254::experimental::modulus_limb(i);
  }
  uint32_t borrow = 0;
  borrow = bb::gpu::bn254::experimental::sub_u32_with_borrow_in(out.limbs[0],
                                                                1U, borrow);
#pragma unroll
  for (int i = 1; i < 8; ++i) {
    borrow = bb::gpu::bn254::experimental::sub_u32_with_borrow_in(out.limbs[i],
                                                                  0U, borrow);
  }
  return out;
}

__device__ __forceinline__ exp_fq32_t make_validation_fq32(const uint32_t index,
                                                           const uint32_t stream) {
  switch (index) {
  case 0:
    return exp_fq32_t::zero();
  case 1:
    return exp_fq32_t::one();
  case 2:
    return fq32_modulus_minus_one();
  case 3:
    return stream == 0x101U ? fq32_modulus_minus_one() : exp_fq32_t::one();
  case 4:
    return stream == 0x101U ? exp_fq32_t::zero() : exp_fq32_t::one();
  default:
    return make_random_fq32(index, stream);
  }
}

__global__ void fq32_seed_inputs_kernel(exp_fq32_t *lhs, exp_fq32_t *rhs,
                                        const size_t count,
                                        const uint32_t lhs_stream,
                                        const uint32_t rhs_stream) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  lhs[idx] = make_random_fq32(static_cast<uint32_t>(idx), lhs_stream);
  rhs[idx] = make_random_fq32(static_cast<uint32_t>(idx), rhs_stream);
}

__device__ constexpr uint32_t BB32_MODULUS[8] = {
    0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
};

__device__ constexpr uint32_t BB32_R2[8] = {
    0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911,
    0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71,
};

__device__ constexpr uint32_t BB32_BARRETT_M[8] = {
    0x19bf90e5, 0x6f3aed8a, 0x67cd4c08, 0xae965e17,
    0x68073013, 0xab074a58, 0x623a04a7, 0x54a47462,
};

__device__ constexpr uint32_t BB32_NEG_MODULUS[8] = {
    0x278302b9, 0xc3df73e9, 0x978e3572, 0x687e956e,
    0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0xcf9bb18d,
};

constexpr uint32_t BB32_R_INV = 0xe4866389;

__device__ bool bb32_ge(const bb32_fq_t &lhs, const uint32_t *rhs) {
  for (int i = 7; i >= 0; --i) {
    if (lhs.limbs[i] > rhs[i]) {
      return true;
    }
    if (lhs.limbs[i] < rhs[i]) {
      return false;
    }
  }
  return true;
}

__device__ uint32_t bb32_sub_modulus_in_place(bb32_fq_t &value) {
  uint64_t borrow = 0;
  for (int i = 0; i < 8; ++i) {
    const uint64_t subtrahend = static_cast<uint64_t>(BB32_MODULUS[i]) + borrow;
    const uint64_t limb = value.limbs[i];
    value.limbs[i] = static_cast<uint32_t>(limb - subtrahend);
    borrow = limb < subtrahend ? 1 : 0;
  }
  return static_cast<uint32_t>(borrow);
}

__device__ bb32_fq_t bb32_add(const bb32_fq_t &lhs, const bb32_fq_t &rhs) {
  bb32_fq_t out{};
  uint64_t carry = 0;
  for (int i = 0; i < 8; ++i) {
    const uint64_t sum =
        static_cast<uint64_t>(lhs.limbs[i]) + rhs.limbs[i] + carry;
    out.limbs[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }
  if (carry != 0 || bb32_ge(out, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(out);
  }
  return out;
}

__device__ void bb32_mul_wide(const bb32_fq_t &lhs, const bb32_fq_t &rhs,
                              uint32_t out[16]) {
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(lhs.limbs[i]) * rhs.limbs[j] + out[i + j] +
          carry;
      out[i + j] = static_cast<uint32_t>(product);
      carry = product >> 32;
    }
    for (int limb = i + 8; carry != 0 && limb < 16; ++limb) {
      const uint64_t sum = static_cast<uint64_t>(out[limb]) + carry;
      out[limb] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
  }
}

__device__ bb32_fq_t bb32_high_with_slack(const uint32_t wide[16]) {
  bb32_fq_t out{};
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = (wide[i + 8] << 4) | (wide[i + 7] >> 28);
  }
  return out;
}

__device__ bb32_fq_t bb32_mul_high(const bb32_fq_t &lhs,
                                   const uint32_t rhs[8]) {
  uint32_t wide[16] = {};
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(lhs.limbs[i]) * rhs[j] + wide[i + j] + carry;
      wide[i + j] = static_cast<uint32_t>(product);
      carry = product >> 32;
    }
    for (int limb = i + 8; carry != 0 && limb < 16; ++limb) {
      const uint64_t sum = static_cast<uint64_t>(wide[limb]) + carry;
      wide[limb] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
  }

  bb32_fq_t out{};
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = wide[i + 8];
  }
  return out;
}

__device__ bb32_fq_t bb32_mul_high_truncated(const bb32_fq_t &lhs,
                                             const uint32_t rhs[8]) {
  uint32_t wide[16] = {};
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    const int start = i < 6 ? 6 - i : 0;
    for (int j = start; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(lhs.limbs[i]) * rhs[j] + wide[i + j] + carry;
      wide[i + j] = static_cast<uint32_t>(product);
      carry = product >> 32;
    }
    for (int limb = i + 8; carry != 0 && limb < 16; ++limb) {
      const uint64_t sum = static_cast<uint64_t>(wide[limb]) + carry;
      wide[limb] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
  }

  bb32_fq_t out{};
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = wide[i + 8];
  }
  return out;
}

__device__ bb32_fq_t bb32_low_mul_add_neg_modulus(const bb32_fq_t &quotient,
                                                  const uint32_t low[8]) {
  bb32_fq_t out{};
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = low[i];
  }
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; i + j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(quotient.limbs[i]) * BB32_NEG_MODULUS[j] +
          out.limbs[i + j] + carry;
      out.limbs[i + j] = static_cast<uint32_t>(product);
      carry = product >> 32;
    }
  }
  return out;
}

__device__ bb32_fq_t bb32_reduce_barrett(const uint32_t wide[16]) {
  const bb32_fq_t high = bb32_high_with_slack(wide);
  const bb32_fq_t quotient = bb32_mul_high(high, BB32_BARRETT_M);
  bb32_fq_t reduced = bb32_low_mul_add_neg_modulus(quotient, wide);
  if (bb32_ge(reduced, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(reduced);
  }
  if (bb32_ge(reduced, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(reduced);
  }
  return reduced;
}

__device__ bb32_fq_t bb32_reduce_barrett_truncated(const uint32_t wide[16]) {
  const bb32_fq_t high = bb32_high_with_slack(wide);
  const bb32_fq_t quotient = bb32_mul_high_truncated(high, BB32_BARRETT_M);
  bb32_fq_t reduced = bb32_low_mul_add_neg_modulus(quotient, wide);
  if (bb32_ge(reduced, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(reduced);
  }
  if (bb32_ge(reduced, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(reduced);
  }
  return reduced;
}

__device__ bb32_fq_t bb32_barrett_mul(const bb32_fq_t &lhs,
                                      const bb32_fq_t &rhs) {
  uint32_t wide[16] = {};
  bb32_mul_wide(lhs, rhs, wide);
  return bb32_reduce_barrett(wide);
}

__device__ bb32_fq_t bb32_barrett_truncated_mul(const bb32_fq_t &lhs,
                                                const bb32_fq_t &rhs) {
  uint32_t wide[16] = {};
  bb32_mul_wide(lhs, rhs, wide);
  return bb32_reduce_barrett_truncated(wide);
}

__device__ bb32_fq_t bb32_mont_mul(const bb32_fq_t &lhs, const bb32_fq_t &rhs) {
  uint32_t t[9] = {};
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(lhs.limbs[j]) * rhs.limbs[i] + t[j] + carry;
      t[j] = static_cast<uint32_t>(product);
      carry = product >> 32;
    }
    t[8] = static_cast<uint32_t>(carry);

    const uint32_t m = t[0] * BB32_R_INV;
    carry = 0;
    for (int j = 0; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(m) * BB32_MODULUS[j] + t[j] + carry;
      if (j > 0) {
        t[j - 1] = static_cast<uint32_t>(product);
      }
      carry = product >> 32;
    }
    const uint64_t high = static_cast<uint64_t>(t[8]) + carry;
    t[7] = static_cast<uint32_t>(high);
    t[8] = static_cast<uint32_t>(high >> 32);
  }

  bb32_fq_t out{};
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = t[i];
  }
  if (t[8] != 0 || bb32_ge(out, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(out);
  }
  return out;
}

__device__ __forceinline__ void
mad_u32_with_carry(uint32_t &low, uint32_t &carry_out, const uint32_t a,
                   const uint32_t b, const uint32_t addend,
                   const uint32_t carry_in) {
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
               "madc.hi.u32 %1, %2, %3, 0;\n\t"
               "add.cc.u32 %0, %0, %5;\n\t"
               "addc.u32 %1, %1, 0;\n\t"
               : "=r"(low), "=r"(carry_out)
               : "r"(a), "r"(b), "r"(addend), "r"(carry_in));
}

__device__ bb32_fq_t bb32_ptx_mont_mul(const bb32_fq_t &lhs,
                                       const bb32_fq_t &rhs) {
  uint32_t t[9] = {};
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      uint32_t low = 0;
      uint32_t next_carry = 0;
      mad_u32_with_carry(low, next_carry, lhs.limbs[j], rhs.limbs[i], t[j],
                         carry);
      t[j] = low;
      carry = next_carry;
    }
    t[8] = carry;

    const uint32_t m = t[0] * BB32_R_INV;
    carry = 0;
    for (int j = 0; j < 8; ++j) {
      uint32_t low = 0;
      uint32_t next_carry = 0;
      mad_u32_with_carry(low, next_carry, m, BB32_MODULUS[j], t[j], carry);
      if (j > 0) {
        t[j - 1] = low;
      }
      carry = next_carry;
    }
    const uint64_t high = static_cast<uint64_t>(t[8]) + carry;
    t[7] = static_cast<uint32_t>(high);
    t[8] = static_cast<uint32_t>(high >> 32);
  }

  bb32_fq_t out{};
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = t[i];
  }
  if (t[8] != 0 || bb32_ge(out, BB32_MODULUS)) {
    bb32_sub_modulus_in_place(out);
  }
  return out;
}

__device__ bb32_fq_t bb32_from_u32(const uint32_t seed) {
  bb32_fq_t value{};
  value.limbs[0] = seed + 1;
  bb32_fq_t r2{};
  for (int i = 0; i < 8; ++i) {
    r2.limbs[i] = BB32_R2[i];
  }
  return bb32_mont_mul(value, r2);
}

__device__ bb32_fq_t bb32_ptx_from_u32(const uint32_t seed) {
  bb32_fq_t value{};
  value.limbs[0] = seed + 1;
  bb32_fq_t r2{};
  for (int i = 0; i < 8; ++i) {
    r2.limbs[i] = BB32_R2[i];
  }
  return bb32_ptx_mont_mul(value, r2);
}

__device__ bb32_fq_t bb32_barrett_from_u32(const uint32_t seed) {
  bb32_fq_t value{};
  value.limbs[0] = seed + 1;
  return value;
}

__device__ bb_fq_t bb_add_no_prereduce(const bb_fq_t &lhs, const bb_fq_t &rhs) {
  uint64_t carry = 0;
  uint64_t next_carry = 0;
  const uint64_t r0 =
      bb::gpu::detail::addc(lhs.data[0], rhs.data[0], 0, next_carry);
  carry = next_carry;
  const uint64_t r1 =
      bb::gpu::detail::addc(lhs.data[1], rhs.data[1], carry, next_carry);
  carry = next_carry;
  const uint64_t r2 =
      bb::gpu::detail::addc(lhs.data[2], rhs.data[2], carry, next_carry);
  carry = next_carry;
  const uint64_t r3 =
      bb::gpu::detail::addc(lhs.data[3], rhs.data[3], carry, next_carry);
  carry = next_carry;
  bb_fq_t out = bb_fq_t::raw(r0, r1, r2, r3);
  if (carry != 0 || bb_fq_t::ge(out, bb_fq_t::modulus())) {
    out = out.reduce_once();
  }
  return out;
}

__device__ bb_fq_t bb_sub_no_prereduce(const bb_fq_t &lhs, const bb_fq_t &rhs) {
  uint64_t borrow = 0;
  uint64_t next_borrow = 0;
  const uint64_t r0 =
      bb::gpu::detail::sbb(lhs.data[0], rhs.data[0], borrow, next_borrow);
  borrow = next_borrow;
  const uint64_t r1 =
      bb::gpu::detail::sbb(lhs.data[1], rhs.data[1], borrow, next_borrow);
  borrow = next_borrow;
  const uint64_t r2 =
      bb::gpu::detail::sbb(lhs.data[2], rhs.data[2], borrow, next_borrow);
  borrow = next_borrow;
  const uint64_t r3 =
      bb::gpu::detail::sbb(lhs.data[3], rhs.data[3], borrow, next_borrow);
  borrow = next_borrow;

  bb_fq_t out = bb_fq_t::raw(r0, r1, r2, r3);
  if (borrow != 0) {
    const bb_fq_t p = bb_fq_t::modulus();
    uint64_t carry = 0;
    uint64_t next_carry = 0;
    const uint64_t s0 =
        bb::gpu::detail::addc(out.data[0], p.data[0], 0, next_carry);
    carry = next_carry;
    const uint64_t s1 =
        bb::gpu::detail::addc(out.data[1], p.data[1], carry, next_carry);
    carry = next_carry;
    const uint64_t s2 =
        bb::gpu::detail::addc(out.data[2], p.data[2], carry, next_carry);
    carry = next_carry;
    const uint64_t s3 =
        bb::gpu::detail::addc(out.data[3], p.data[3], carry, next_carry);
    out = bb_fq_t::raw(s0, s1, s2, s3);
  }
  return out;
}

__device__ bb_fq_t bb_mul_no_prereduce(const bb_fq_t &lhs, const bb_fq_t &rhs) {
  uint64_t t[9] = {};

  for (size_t i = 0; i < 4; ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < 4; ++j) {
      t[i + j] = bb::gpu::detail::mac(t[i + j], lhs.data[i], rhs.data[j], carry,
                                      carry);
    }
    bb::gpu::detail::add_to_limb(t, i + 4, carry);
  }

  const bb_fq_t p = bb_fq_t::modulus();
  for (size_t i = 0; i < 4; ++i) {
    const uint64_t m = t[i] * bb::gpu::bn254::detail::Bn254FqParams::r_inv;
    uint64_t carry = 0;
    for (size_t j = 0; j < 4; ++j) {
      t[i + j] = bb::gpu::detail::mac(t[i + j], m, p.data[j], carry, carry);
    }
    bb::gpu::detail::add_to_limb(t, i + 4, carry);
  }

  return bb_fq_t::raw(t[4], t[5], t[6], t[7]).reduce_full();
}

__global__ void bb_field_add_kernel(bb_fq_t *out, const size_t count,
                                    const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 11);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 29);
  for (int i = 0; i < inner_iters; ++i) {
    a = a + b;
    b = b + a;
  }
  out[idx] = a + b;
}

__global__ void bb_field_add_no_prereduce_kernel(bb_fq_t *out,
                                                 const size_t count,
                                                 const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 11);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 29);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb_add_no_prereduce(a, b);
    b = bb_add_no_prereduce(b, a);
  }
  out[idx] = bb_add_no_prereduce(a, b);
}

__global__ void bb_field_sub_kernel(bb_fq_t *out, const size_t count,
                                    const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 41);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 7);
  for (int i = 0; i < inner_iters; ++i) {
    a = a - b;
    b = b - a;
  }
  out[idx] = a - b;
}

__global__ void bb_field_sub_no_prereduce_kernel(bb_fq_t *out,
                                                 const size_t count,
                                                 const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 41);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 7);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb_sub_no_prereduce(a, b);
    b = bb_sub_no_prereduce(b, a);
  }
  out[idx] = bb_sub_no_prereduce(a, b);
}

__global__ void bb_field_neg_kernel(bb_fq_t *out, const size_t count,
                                    const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 43);
  for (int i = 0; i < inner_iters; ++i) {
    a = -a;
  }
  out[idx] = a;
}

__global__ void bb_field_is_zero_kernel(uint32_t *out, const size_t count,
                                        const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 47);
  uint32_t hits = 0;
  for (int i = 0; i < inner_iters; ++i) {
    hits += a.is_zero() ? 1U : 0U;
    a.data[0] ^= static_cast<uint64_t>(i + 1);
  }
  out[idx] = hits ^ static_cast<uint32_t>(a.data[0]);
}

__global__ void bb_field_equal_kernel(uint32_t *out, const size_t count,
                                      const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 53);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 59);
  uint32_t hits = 0;
  for (int i = 0; i < inner_iters; ++i) {
    hits += (a == b) ? 1U : 0U;
    b.data[0] ^= static_cast<uint64_t>(i + 1);
  }
  out[idx] = hits ^ static_cast<uint32_t>(b.data[0]);
}

__global__ void bb_field_mul_kernel(bb_fq_t *out, const size_t count,
                                    const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 13);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = a * b;
    b = b + make_bb_fq(static_cast<uint32_t>(i) + 1);
  }
  out[idx] = a;
}

__global__ void bb_field_mul_no_prereduce_kernel(bb_fq_t *out,
                                                 const size_t count,
                                                 const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 13);
  bb_fq_t b = make_bb_fq(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb_mul_no_prereduce(a, b);
    b = bb_add_no_prereduce(b, make_bb_fq(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void bb32_field_add_kernel(bb32_fq_t *out, const size_t count,
                                      const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb32_fq_t a = bb32_from_u32(static_cast<uint32_t>(idx) + 11);
  bb32_fq_t b = bb32_from_u32(static_cast<uint32_t>(idx) + 29);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb32_add(a, b);
    b = bb32_add(b, a);
  }
  out[idx] = bb32_add(a, b);
}

__global__ void bb32_field_mul_kernel(bb32_fq_t *out, const size_t count,
                                      const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb32_fq_t a = bb32_from_u32(static_cast<uint32_t>(idx) + 13);
  bb32_fq_t b = bb32_from_u32(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb32_mont_mul(a, b);
    b = bb32_add(b, bb32_from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void bb32_ptx_field_mul_kernel(bb32_fq_t *out, const size_t count,
                                          const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb32_fq_t a = bb32_ptx_from_u32(static_cast<uint32_t>(idx) + 13);
  bb32_fq_t b = bb32_ptx_from_u32(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb32_ptx_mont_mul(a, b);
    b = bb32_add(b, bb32_ptx_from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void bb32_barrett_field_mul_kernel(bb32_fq_t *out,
                                              const size_t count,
                                              const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb32_fq_t a = bb32_barrett_from_u32(static_cast<uint32_t>(idx) + 13);
  bb32_fq_t b = bb32_barrett_from_u32(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb32_barrett_mul(a, b);
    b = bb32_add(b, bb32_barrett_from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void bb32_barrett_truncated_field_mul_kernel(bb32_fq_t *out,
                                                        const size_t count,
                                                        const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb32_fq_t a = bb32_barrett_from_u32(static_cast<uint32_t>(idx) + 13);
  bb32_fq_t b = bb32_barrett_from_u32(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb32_barrett_truncated_mul(a, b);
    b = bb32_add(b, bb32_barrett_from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void
bb32_barrett_truncated_ptx_field_mul_kernel(exp_fq32_t *out, const size_t count,
                                            const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_input_field_mul_kernel(exp_fq32_t *out,
                                            const exp_fq32_t *lhs,
                                            const exp_fq32_t *rhs,
                                            const size_t count,
                                            const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  const exp_fq32_t b = rhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul_straightline(a, b);
  }
  out[idx] = a;
}

__global__ void fq32_input_field_sqr_kernel(exp_fq32_t *out,
                                            const exp_fq32_t *lhs,
                                            const exp_fq32_t *rhs,
                                            const size_t count,
                                            const int inner_iters) {
  (void)rhs;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::sqr_straightline(a);
  }
  out[idx] = a;
}

__global__ void fq32_input_field_add_kernel(exp_fq32_t *out,
                                            const exp_fq32_t *lhs,
                                            const exp_fq32_t *rhs,
                                            const size_t count,
                                            const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  exp_fq32_t b = rhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::add(a, b);
    b = bb::gpu::bn254::experimental::add(b, a);
  }
  out[idx] = bb::gpu::bn254::experimental::add(a, b);
}

__global__ void fq32_input_field_sub_kernel(exp_fq32_t *out,
                                            const exp_fq32_t *lhs,
                                            const exp_fq32_t *rhs,
                                            const size_t count,
                                            const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  exp_fq32_t b = rhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::sub(a, b);
    b = bb::gpu::bn254::experimental::sub(b, a);
  }
  out[idx] = bb::gpu::bn254::experimental::sub(a, b);
}

__global__ void fq32_input_field_neg_kernel(exp_fq32_t *out,
                                            const exp_fq32_t *lhs,
                                            const exp_fq32_t *rhs,
                                            const size_t count,
                                            const int inner_iters) {
  (void)rhs;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::neg(a);
  }
  out[idx] = a;
}

__global__ void fq32_input_field_is_zero_kernel(uint32_t *out,
                                                const exp_fq32_t *lhs,
                                                const exp_fq32_t *rhs,
                                                const size_t count,
                                                const int inner_iters) {
  (void)rhs;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  uint32_t hits = 0;
  for (int i = 0; i < inner_iters; ++i) {
    hits += bb::gpu::bn254::experimental::is_zero(a) ? 1U : 0U;
    a.limbs[0] ^= static_cast<uint32_t>(i) + 1U;
  }
  out[idx] = hits ^ a.limbs[0];
}

__global__ void fq32_input_field_equal_kernel(uint32_t *out,
                                              const exp_fq32_t *lhs,
                                              const exp_fq32_t *rhs,
                                              const size_t count,
                                              const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = lhs[idx];
  const exp_fq32_t b = rhs[idx];
  uint32_t hits = 0;
  for (int i = 0; i < inner_iters; ++i) {
    hits += fq32_equal(a, b) ? 1U : 0U;
    a.limbs[0] ^= static_cast<uint32_t>(i) + 1U;
  }
  out[idx] = hits ^ a.limbs[0];
}

__device__ __noinline__ exp_fq32_t fq32_callable_mul(const exp_fq32_t &lhs,
                                                     const exp_fq32_t &rhs) {
  return bb::gpu::bn254::experimental::mul_straightline(lhs, rhs);
}

__device__ __noinline__ exp_fq32_t fq32_callable_sqr(const exp_fq32_t &value) {
  return bb::gpu::bn254::experimental::sqr_straightline(value);
}

__device__ __forceinline__ exp_fq32_t
fq32_dedicated_sqr(const exp_fq32_t &value) {
  uint32_t wide[16] = {};
  bb::gpu::bn254::experimental::sqr_wide_straightline(value, wide);
  return bb::gpu::bn254::experimental::reduce_straightline(wide);
}

__global__ void fq32_callable_field_mul_kernel(exp_fq32_t *out,
                                               const size_t count,
                                               const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
  for (int i = 0; i < inner_iters; ++i) {
    a = fq32_callable_mul(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_callable_field_sqr_kernel(exp_fq32_t *out,
                                               const size_t count,
                                               const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 18U);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::add(
        fq32_callable_sqr(a),
        exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 3));
  }
  out[idx] = a;
}

__global__ void fq32_dedicated_field_sqr_kernel(exp_fq32_t *out,
                                                const size_t count,
                                                const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 18U);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::add(
        fq32_dedicated_sqr(a),
        exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 3));
  }
  out[idx] = a;
}

__global__ void fq32_reduce_only_barrett_kernel(exp_fq32_t *out,
                                                const size_t count,
                                                const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  uint32_t wide[16] = {};
  bb::gpu::bn254::experimental::mul_wide_straightline(a, b, wide);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::reduce_straightline(wide);
    wide[static_cast<uint32_t>(i) & 15U] ^=
        a.limbs[static_cast<uint32_t>(i) & 7U];
  }
  out[idx] = a;
}

__global__ void fq32_reduce_only_barrett_representative_kernel(
    exp_fq32_t *out, const size_t count, const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  uint32_t wide[16] = {};
  bb::gpu::bn254::experimental::mul_wide_straightline(a, b, wide);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::reduce_straightline(wide);
    wide[0] ^= a.limbs[0] + static_cast<uint32_t>(i);
    wide[8] ^= a.limbs[7] ^ static_cast<uint32_t>(i << 1);
  }
  out[idx] = a;
}

__global__ void fq32_karatsuba_field_mul_kernel(exp_fq32_t *out,
                                                const size_t count,
                                                const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul_karatsuba(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_karatsuba_fused_field_mul_kernel(exp_fq32_t *out,
                                                      const size_t count,
                                                      const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul_karatsuba_fused(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_half_product_field_mul_kernel(exp_fq32_t *out,
                                                   const size_t count,
                                                   const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul_half_product_direct(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_wide_product_kernel(exp_fq32_wide_t *out,
                                         const size_t count,
                                         const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  exp_fq32_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::experimental::mul_wide(a, b, wide.limbs);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = wide;
}

__global__ void fq32_straightline_wide_product_kernel(exp_fq32_wide_t *out,
                                                      const size_t count,
                                                      const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  exp_fq32_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::experimental::mul_wide_straightline(a, b, wide.limbs);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = wide;
}

__global__ void fq32_karatsuba_wide_product_kernel(exp_fq32_wide_t *out,
                                                   const size_t count,
                                                   const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  exp_fq32_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::experimental::mul_wide_karatsuba(a, b, wide.limbs);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = wide;
}

__global__ void fq32_karatsuba_fused_wide_product_kernel(
    exp_fq32_wide_t *out, const size_t count, const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  exp_fq32_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::experimental::mul_wide_karatsuba_fused(a, b, wide.limbs);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = wide;
}

__global__ void fq32_half_product_wide_product_kernel(
    exp_fq32_wide_t *out, const size_t count, const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = make_random_fq32(static_cast<uint32_t>(idx), 14U);
  exp_fq32_t b = make_random_fq32(static_cast<uint32_t>(idx), 32U);
  exp_fq32_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::experimental::mul_wide_half_product_direct(a, b,
                                                               wide.limbs);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = wide;
}

__global__ void bb_field_sqr_kernel(bb_fq_t *out, const size_t count,
                                    const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 17);
  for (int i = 0; i < inner_iters; ++i) {
    a = a.sqr() + make_bb_fq(static_cast<uint32_t>(i) + 3);
  }
  out[idx] = a;
}

__global__ void bb_field_sqr_no_prereduce_kernel(bb_fq_t *out,
                                                 const size_t count,
                                                 const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 17);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb_add_no_prereduce(a.sqr_assume_canonical(),
                            make_bb_fq(static_cast<uint32_t>(i) + 3));
  }
  out[idx] = a;
}

__global__ void
bb_field_sqr_dedicated_no_prereduce_kernel(bb_fq_t *out, const size_t count,
                                           const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_fq_t a = make_bb_fq(static_cast<uint32_t>(idx) + 17);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb_add_no_prereduce(a.sqr_dedicated_assume_canonical(),
                            make_bb_fq(static_cast<uint32_t>(i) + 3));
  }
  out[idx] = a;
}

__global__ void bb_xyzz_mixed_add_kernel(bb_xyzz_t *out, const size_t count,
                                         const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_xyzz_t accumulator{
      make_bb_fq(static_cast<uint32_t>(idx) + 101),
      make_bb_fq(static_cast<uint32_t>(idx) + 103),
      bb_fq_t::one(),
      bb_fq_t::one(),
      false,
  };
  bb_affine_t point{
      make_bb_fq(static_cast<uint32_t>(idx) + 107),
      make_bb_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::xyzz_mixed_add(accumulator, point);
  }
  out[idx] = accumulator;
}

__device__ __forceinline__ void
bb_jacobian_mixed_add_unchecked(bb_jacobian_t &lhs, const bb_affine_t &rhs) {
  bb_fq_t t0 = lhs.z.sqr();
  bb_fq_t t1 = rhs.x * t0;
  t1 = t1 - lhs.x;
  bb_fq_t t2 = lhs.z * t0;
  t2 = t2 * rhs.y;
  t2 = t2 - lhs.y;
  t2 = t2 + t2;
  lhs.z = lhs.z + t1;
  bb_fq_t t3 = t1.sqr();
  t0 = t0 + t3;
  lhs.z = lhs.z.sqr();
  lhs.z = lhs.z - t0;
  t3 = t3 + t3;
  t3 = t3 + t3;
  t1 = t1 * t3;
  t3 = t3 * lhs.x;
  t0 = t3 + t3;
  t0 = t0 + t1;
  lhs.x = t2.sqr();
  lhs.x = lhs.x - t0;
  t3 = t3 - lhs.x;
  t1 = t1 * lhs.y;
  t1 = t1 + t1;
  t3 = t2 * t3;
  lhs.y = t3 - t1;
}

__device__ __forceinline__ void
bb_xyzz_mixed_add_unchecked(bb_xyzz_t &lhs, const bb_affine_t &rhs) {
  bb::gpu::bn254::xyzz_mixed_add_unchecked(lhs, rhs);
}

__global__ void bb_xyzz_mixed_add_unchecked_kernel(bb_xyzz_t *out,
                                                   const size_t count,
                                                   const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_xyzz_t accumulator{
      make_bb_fq(static_cast<uint32_t>(idx) + 101),
      make_bb_fq(static_cast<uint32_t>(idx) + 103),
      bb_fq_t::one(),
      bb_fq_t::one(),
      false,
  };
  bb_affine_t point{
      make_bb_fq(static_cast<uint32_t>(idx) + 107),
      make_bb_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    bb_xyzz_mixed_add_unchecked(accumulator, point);
  }
  out[idx] = accumulator;
}

__global__ void bb_xyzz_mixed_add_assume_finite_kernel(bb_xyzz_t *out,
                                                       const size_t count,
                                                       const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_xyzz_t accumulator{
      make_bb_fq(static_cast<uint32_t>(idx) + 101),
      make_bb_fq(static_cast<uint32_t>(idx) + 103),
      bb_fq_t::one(),
      bb_fq_t::one(),
      false,
  };
  bb_affine_t point{
      make_bb_fq(static_cast<uint32_t>(idx) + 107),
      make_bb_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::xyzz_mixed_add_assume_finite(accumulator, point);
  }
  out[idx] = accumulator;
}

__global__ void bb_jacobian_mixed_add_unchecked_kernel(bb_jacobian_t *out,
                                                       const size_t count,
                                                       const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_jacobian_t accumulator{
      make_bb_fq(static_cast<uint32_t>(idx) + 101),
      make_bb_fq(static_cast<uint32_t>(idx) + 103),
      bb_fq_t::one(),
      false,
  };
  bb_affine_t point{
      make_bb_fq(static_cast<uint32_t>(idx) + 107),
      make_bb_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    bb_jacobian_mixed_add_unchecked(accumulator, point);
  }
  out[idx] = accumulator;
}

__device__ __forceinline__ exp_fq32_t fq32_add(const exp_fq32_t &lhs,
                                               const exp_fq32_t &rhs) {
  return bb::gpu::bn254::experimental::add(lhs, rhs);
}

__device__ __forceinline__ exp_fq32_t fq32_sub(const exp_fq32_t &lhs,
                                               const exp_fq32_t &rhs) {
  return bb::gpu::bn254::experimental::sub(lhs, rhs);
}

__device__ __forceinline__ exp_fq32_t fq32_mul(const exp_fq32_t &lhs,
                                               const exp_fq32_t &rhs) {
  return bb::gpu::bn254::experimental::mul_straightline(lhs, rhs);
}

__device__ __forceinline__ exp_fq32_t fq32_sqr(const exp_fq32_t &value) {
  return bb::gpu::bn254::experimental::sqr_straightline(value);
}

__device__ __forceinline__ exp_fq32_xyzz_t fq32_xyzz_infinity() {
  return {exp_fq32_t::zero(), exp_fq32_t::zero(), exp_fq32_t::zero(),
          exp_fq32_t::zero(), true};
}

__device__ __forceinline__ void fq32_xyzz_double(exp_fq32_xyzz_t &point) {
  if (point.infinity) {
    return;
  }

  exp_fq32_t u = fq32_add(point.y, point.y);
  if (bb::gpu::bn254::experimental::is_zero(u)) {
    point = fq32_xyzz_infinity();
    return;
  }

  const exp_fq32_t v = fq32_sqr(u);
  const exp_fq32_t w = fq32_mul(u, v);
  const exp_fq32_t s = fq32_mul(point.x, v);
  exp_fq32_t m = fq32_sqr(point.x);
  m = fq32_add(fq32_add(m, m), m);
  const exp_fq32_t x3 = fq32_sub(fq32_sqr(m), fq32_add(s, s));
  point.y = fq32_sub(fq32_mul(m, fq32_sub(s, x3)), fq32_mul(w, point.y));
  point.x = x3;
  point.zz = fq32_mul(v, point.zz);
  point.zzz = fq32_mul(w, point.zzz);
}

__device__ __forceinline__ void
fq32_xyzz_mixed_add_shape(exp_fq32_xyzz_t &lhs, const exp_fq32_t &rhs_x,
                          const exp_fq32_t &rhs_y) {
  exp_fq32_t p = fq32_sub(fq32_mul(rhs_x, lhs.zz), lhs.x);
  exp_fq32_t r = fq32_sub(fq32_mul(rhs_y, lhs.zzz), lhs.y);
  exp_fq32_t pp = fq32_sqr(p);
  exp_fq32_t q = fq32_mul(lhs.x, pp);
  exp_fq32_t ppp = fq32_mul(p, pp);
  lhs.zz = fq32_mul(lhs.zz, pp);
  exp_fq32_t x3 = fq32_sqr(r);
  x3 = fq32_sub(x3, ppp);
  pp = fq32_add(q, q);
  x3 = fq32_sub(x3, pp);
  lhs.zzz = fq32_mul(lhs.zzz, ppp);
  q = fq32_sub(q, x3);
  q = fq32_mul(r, q);
  ppp = fq32_mul(lhs.y, ppp);
  lhs.y = fq32_sub(q, ppp);
  lhs.x = x3;
}

__device__ __forceinline__ void
fq32_xyzz_add_shape(exp_fq32_xyzz_t &lhs, const exp_fq32_xyzz_t &rhs) {
  if (lhs.infinity) {
    lhs = rhs;
    return;
  }
  if (rhs.infinity) {
    return;
  }

  exp_fq32_t u1 = fq32_mul(lhs.x, rhs.zz);
  exp_fq32_t u2 = fq32_mul(rhs.x, lhs.zz);
  exp_fq32_t s1 = fq32_mul(lhs.y, rhs.zzz);
  exp_fq32_t s2 = fq32_mul(rhs.y, lhs.zzz);
  exp_fq32_t p = fq32_sub(u2, u1);
  exp_fq32_t r = fq32_sub(s2, s1);

  if (bb::gpu::bn254::experimental::is_zero(p)) {
    if (bb::gpu::bn254::experimental::is_zero(r)) {
      fq32_xyzz_double(lhs);
    } else {
      lhs = fq32_xyzz_infinity();
    }
    return;
  }

  exp_fq32_t pp = fq32_sqr(p);
  p = fq32_mul(p, pp);
  u1 = fq32_mul(u1, pp);
  exp_fq32_t x3 = fq32_sub(fq32_sqr(r), p);
  x3 = fq32_sub(x3, fq32_add(u1, u1));
  s1 = fq32_mul(s1, p);
  u1 = fq32_mul(r, fq32_sub(u1, x3));
  lhs.y = fq32_sub(u1, s1);
  lhs.x = x3;
  lhs.zz = fq32_mul(fq32_mul(lhs.zz, rhs.zz), pp);
  lhs.zzz = fq32_mul(fq32_mul(lhs.zzz, rhs.zzz), p);
}

__global__ void fq32_curve_shape_xyzz_mixed_add_kernel(exp_fq32_xyzz_t *out,
                                                       const size_t count,
                                                       const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_xyzz_t accumulator{
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 101),
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 103),
      exp_fq32_t::one(),
      exp_fq32_t::one(),
      false,
  };
  const exp_fq32_t point_x =
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 107);
  const exp_fq32_t point_y =
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 109);
  for (int i = 0; i < inner_iters; ++i) {
    fq32_xyzz_mixed_add_shape(accumulator, point_x, point_y);
  }
  out[idx] = accumulator;
}

__global__ void fq32_curve_shape_xyzz_add_kernel(exp_fq32_xyzz_t *out,
                                                 const size_t count,
                                                 const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_xyzz_t accumulator{
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 101),
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 103),
      exp_fq32_t::one(),
      exp_fq32_t::one(),
      false,
  };
  const exp_fq32_xyzz_t point{
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 107),
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 109),
      exp_fq32_t::one(),
      exp_fq32_t::one(),
      false,
  };
  for (int i = 0; i < inner_iters; ++i) {
    fq32_xyzz_add_shape(accumulator, point);
  }
  out[idx] = accumulator;
}

__global__ void fq32_curve_shape_xyzz_double_kernel(exp_fq32_xyzz_t *out,
                                                    const size_t count,
                                                    const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_xyzz_t accumulator{
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 101),
      exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 103),
      exp_fq32_t::one(),
      exp_fq32_t::one(),
      false,
  };
  for (int i = 0; i < inner_iters; ++i) {
    fq32_xyzz_double(accumulator);
  }
  out[idx] = accumulator;
}

__global__ void bb_curve_shape_xyzz_add_kernel(bb_xyzz_t *out,
                                               const size_t count,
                                               const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_xyzz_t accumulator{
      make_bb_fq(static_cast<uint32_t>(idx) + 101),
      make_bb_fq(static_cast<uint32_t>(idx) + 103),
      bb_fq_t::one(),
      bb_fq_t::one(),
      false,
  };
  const bb_xyzz_t point{
      make_bb_fq(static_cast<uint32_t>(idx) + 107),
      make_bb_fq(static_cast<uint32_t>(idx) + 109),
      bb_fq_t::one(),
      bb_fq_t::one(),
      false,
  };
  for (int i = 0; i < inner_iters; ++i) {
    accumulator = bb::gpu::bn254::xyzz_add(accumulator, point);
  }
  out[idx] = accumulator;
}

__global__ void bb_curve_shape_xyzz_double_kernel(bb_xyzz_t *out,
                                                  const size_t count,
                                                  const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  bb_xyzz_t accumulator{
      make_bb_fq(static_cast<uint32_t>(idx) + 101),
      make_bb_fq(static_cast<uint32_t>(idx) + 103),
      bb_fq_t::one(),
      bb_fq_t::one(),
      false,
  };
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::self_double(accumulator);
  }
  out[idx] = accumulator;
}

#ifdef BB_GPU_HAVE_ICICLE_V28

using icicle_fq_t = bn254::point_field_t;
using icicle_wide_t = icicle_fq_t::Wide;
using icicle_affine_t = bn254::affine_t;
using icicle_projective_t = bn254::projective_t;

struct alignas(32) icicle_xyzz_t {
  icicle_fq_t x;
  icicle_fq_t y;
  icicle_fq_t zz;
  icicle_fq_t zzz;
  bool infinity;
};

__device__ icicle_fq_t make_icicle_fq(const uint32_t seed) {
  return icicle_fq_t::from(seed + 1);
}

__device__ __forceinline__ icicle_fq_t fq32_to_icicle(const exp_fq32_t &value) {
  icicle_fq_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs_storage.limbs[i] = value.limbs[i];
  }
  return out;
}

__device__ __forceinline__ exp_fq32_t icicle_to_fq32(const icicle_fq_t &value) {
  exp_fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = value.limbs_storage.limbs[i];
  }
  return bb::gpu::bn254::experimental::normalize(out);
}

__device__ __forceinline__ icicle_fq_t make_random_icicle_fq(
    const uint32_t index, const uint32_t stream) {
  return fq32_to_icicle(make_random_fq32(index, stream));
}

__global__ void icicle_seed_inputs_kernel(icicle_fq_t *lhs, icicle_fq_t *rhs,
                                          const size_t count,
                                          const uint32_t lhs_stream,
                                          const uint32_t rhs_stream) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  lhs[idx] = make_random_icicle_fq(static_cast<uint32_t>(idx), lhs_stream);
  rhs[idx] = make_random_icicle_fq(static_cast<uint32_t>(idx), rhs_stream);
}

__global__ void icicle_wide_product_kernel(icicle_wide_t *out,
                                           const size_t count,
                                           const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = make_random_icicle_fq(static_cast<uint32_t>(idx), 13U);
  icicle_fq_t b = make_random_icicle_fq(static_cast<uint32_t>(idx), 31U);
  icicle_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    wide = icicle_fq_t::mul_wide(a, b);
    b = b + make_icicle_fq(static_cast<uint32_t>(i) + 1);
  }
  out[idx] = wide;
}

__global__ void icicle_reduce_only_kernel(icicle_fq_t *out, const size_t count,
                                          const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const icicle_fq_t a = make_random_icicle_fq(static_cast<uint32_t>(idx), 13U);
  const icicle_fq_t b = make_random_icicle_fq(static_cast<uint32_t>(idx), 31U);
  icicle_wide_t wide = icicle_fq_t::mul_wide(a, b);
  icicle_fq_t reduced{};
  for (int i = 0; i < inner_iters; ++i) {
    reduced = icicle_fq_t::reduce(wide);
    wide.limbs_storage.limbs[static_cast<uint32_t>(i) & 15U] ^=
        reduced.limbs_storage.limbs[static_cast<uint32_t>(i) & 7U];
  }
  out[idx] = reduced;
}

__global__ void icicle_input_field_add_kernel(icicle_fq_t *out,
                                              const icicle_fq_t *lhs,
                                              const icicle_fq_t *rhs,
                                              const size_t count,
                                              const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  icicle_fq_t b = rhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = a + b;
    b = b + a;
  }
  out[idx] = a + b;
}

__global__ void icicle_input_field_sub_kernel(icicle_fq_t *out,
                                              const icicle_fq_t *lhs,
                                              const icicle_fq_t *rhs,
                                              const size_t count,
                                              const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  icicle_fq_t b = rhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = a - b;
    b = b - a;
  }
  out[idx] = a - b;
}

__global__ void icicle_input_field_neg_kernel(icicle_fq_t *out,
                                              const icicle_fq_t *lhs,
                                              const icicle_fq_t *rhs,
                                              const size_t count,
                                              const int inner_iters) {
  (void)rhs;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = icicle_fq_t::neg(a);
  }
  out[idx] = a;
}

__global__ void icicle_input_field_is_zero_kernel(uint32_t *out,
                                                  const icicle_fq_t *lhs,
                                                  const icicle_fq_t *rhs,
                                                  const size_t count,
                                                  const int inner_iters) {
  (void)rhs;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  uint32_t hits = 0;
  for (int i = 0; i < inner_iters; ++i) {
    hits += (a == icicle_fq_t::zero()) ? 1U : 0U;
    a.limbs_storage.limbs[0] ^= static_cast<uint32_t>(i) + 1U;
  }
  out[idx] = hits ^ a.limbs_storage.limbs[0];
}

__global__ void icicle_input_field_equal_kernel(uint32_t *out,
                                                const icicle_fq_t *lhs,
                                                const icicle_fq_t *rhs,
                                                const size_t count,
                                                const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  const icicle_fq_t b = rhs[idx];
  uint32_t hits = 0;
  for (int i = 0; i < inner_iters; ++i) {
    hits += (a == b) ? 1U : 0U;
    a.limbs_storage.limbs[0] ^= static_cast<uint32_t>(i) + 1U;
  }
  out[idx] = hits ^ a.limbs_storage.limbs[0];
}

__global__ void icicle_input_field_mul_kernel(icicle_fq_t *out,
                                              const icicle_fq_t *lhs,
                                              const icicle_fq_t *rhs,
                                              const size_t count,
                                              const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  const icicle_fq_t b = rhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = a * b;
  }
  out[idx] = a;
}

__global__ void icicle_input_field_sqr_kernel(icicle_fq_t *out,
                                              const icicle_fq_t *lhs,
                                              const icicle_fq_t *rhs,
                                              const size_t count,
                                              const int inner_iters) {
  (void)rhs;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = lhs[idx];
  for (int i = 0; i < inner_iters; ++i) {
    a = icicle_fq_t::sqr(a);
  }
  out[idx] = a;
}

__global__ void icicle_projective_mixed_add_kernel(icicle_projective_t *out,
                                                   const size_t count,
                                                   const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_projective_t accumulator{
      make_icicle_fq(static_cast<uint32_t>(idx) + 101),
      make_icicle_fq(static_cast<uint32_t>(idx) + 103),
      icicle_fq_t::one(),
  };
  icicle_affine_t point{
      make_icicle_fq(static_cast<uint32_t>(idx) + 107),
      make_icicle_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    accumulator = accumulator + point;
  }
  out[idx] = accumulator;
}

__device__ __forceinline__ void
icicle_xyzz_mixed_add_unchecked(icicle_xyzz_t &lhs,
                                const icicle_affine_t &rhs) {
  icicle_fq_t p = rhs.x * lhs.zz;
  p = p - lhs.x;
  icicle_fq_t r = rhs.y * lhs.zzz;
  r = r - lhs.y;
  icicle_fq_t pp = icicle_fq_t::sqr(p);
  icicle_fq_t ppp = p * pp;
  lhs.zz = lhs.zz * pp;
  lhs.zzz = lhs.zzz * ppp;
  icicle_fq_t q = lhs.x * pp;
  icicle_fq_t x3 = icicle_fq_t::sqr(r);
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

__device__ __forceinline__ bool
icicle_is_infinity(const icicle_affine_t &point) {
  return point.x == icicle_fq_t::zero() && point.y == icicle_fq_t::zero();
}

__device__ __forceinline__ void icicle_xyzz_double(icicle_xyzz_t &point) {
  if (point.infinity) {
    return;
  }

  icicle_fq_t u = point.y + point.y;
  if (u == icicle_fq_t::zero()) {
    point.infinity = true;
    return;
  }

  const icicle_fq_t v = icicle_fq_t::sqr(u);
  const icicle_fq_t w = u * v;
  const icicle_fq_t s = point.x * v;
  icicle_fq_t m = icicle_fq_t::sqr(point.x);
  m = m + m + m;
  const icicle_fq_t two_s = s + s;
  const icicle_fq_t x3 = icicle_fq_t::sqr(m) - two_s;
  point.y = (m * (s - x3)) - (w * point.y);
  point.x = x3;
  point.zz = v * point.zz;
  point.zzz = w * point.zzz;
}

__device__ __forceinline__ void
icicle_xyzz_mixed_add_checked(icicle_xyzz_t &lhs, const icicle_affine_t &rhs) {
  if (icicle_is_infinity(rhs)) {
    return;
  }
  if (lhs.infinity) {
    lhs.x = rhs.x;
    lhs.y = rhs.y;
    lhs.zz = icicle_fq_t::one();
    lhs.zzz = icicle_fq_t::one();
    lhs.infinity = false;
    return;
  }

  icicle_fq_t p = rhs.x * lhs.zz;
  p = p - lhs.x;
  icicle_fq_t r = rhs.y * lhs.zzz;
  r = r - lhs.y;
  if (p == icicle_fq_t::zero()) {
    if (r == icicle_fq_t::zero()) {
      icicle_xyzz_double(lhs);
    } else {
      lhs.infinity = true;
    }
    return;
  }

  icicle_fq_t pp = icicle_fq_t::sqr(p);
  icicle_fq_t ppp = p * pp;
  lhs.zz = lhs.zz * pp;
  lhs.zzz = lhs.zzz * ppp;
  icicle_fq_t q = lhs.x * pp;
  icicle_fq_t x3 = icicle_fq_t::sqr(r);
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

__global__ void icicle_xyzz_mixed_add_unchecked_kernel(icicle_xyzz_t *out,
                                                       const size_t count,
                                                       const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_xyzz_t accumulator{
      make_icicle_fq(static_cast<uint32_t>(idx) + 101),
      make_icicle_fq(static_cast<uint32_t>(idx) + 103),
      icicle_fq_t::one(),
      icicle_fq_t::one(),
      false,
  };
  icicle_affine_t point{
      make_icicle_fq(static_cast<uint32_t>(idx) + 107),
      make_icicle_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    icicle_xyzz_mixed_add_unchecked(accumulator, point);
  }
  out[idx] = accumulator;
}

__global__ void icicle_xyzz_mixed_add_checked_kernel(icicle_xyzz_t *out,
                                                     const size_t count,
                                                     const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_xyzz_t accumulator{
      make_icicle_fq(static_cast<uint32_t>(idx) + 101),
      make_icicle_fq(static_cast<uint32_t>(idx) + 103),
      icicle_fq_t::one(),
      icicle_fq_t::one(),
      false,
  };
  icicle_affine_t point{
      make_icicle_fq(static_cast<uint32_t>(idx) + 107),
      make_icicle_fq(static_cast<uint32_t>(idx) + 109),
  };
  for (int i = 0; i < inner_iters; ++i) {
    icicle_xyzz_mixed_add_checked(accumulator, point);
  }
  out[idx] = accumulator;
}

__global__ void fq32_icicle_validation_kernel(uint32_t *mismatches,
                                              const size_t count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const exp_fq32_t lhs =
      make_validation_fq32(static_cast<uint32_t>(idx), 0x101U);
  const exp_fq32_t rhs =
      make_validation_fq32(static_cast<uint32_t>(idx), 0x202U);
  const icicle_fq_t icicle_lhs = fq32_to_icicle(lhs);
  const icicle_fq_t icicle_rhs = fq32_to_icicle(rhs);

  uint32_t mismatch = 0;
  mismatch |= fq32_equal(bb::gpu::bn254::experimental::add(lhs, rhs),
                         icicle_to_fq32(icicle_lhs + icicle_rhs))
                    ? 0U
                    : 1U;
  mismatch |= fq32_equal(bb::gpu::bn254::experimental::sub(lhs, rhs),
                         icicle_to_fq32(icicle_lhs - icicle_rhs))
                    ? 0U
                    : 2U;
  mismatch |= fq32_equal(bb::gpu::bn254::experimental::neg(lhs),
                         icicle_to_fq32(icicle_fq_t::neg(icicle_lhs)))
                    ? 0U
                    : 4U;
  mismatch |= fq32_equal(bb::gpu::bn254::experimental::mul_straightline(lhs, rhs),
                         icicle_to_fq32(icicle_lhs * icicle_rhs))
                    ? 0U
                    : 8U;
  mismatch |= fq32_equal(bb::gpu::bn254::experimental::sqr_straightline(lhs),
                         icicle_to_fq32(icicle_fq_t::sqr(icicle_lhs)))
                    ? 0U
                    : 16U;
  mismatch |=
      (bb::gpu::bn254::experimental::is_zero(lhs) ==
       (icicle_lhs == icicle_fq_t::zero()))
          ? 0U
          : 32U;
  mismatch |= (fq32_equal(lhs, rhs) == (icicle_lhs == icicle_rhs)) ? 0U : 64U;

  if (mismatch != 0) {
    atomicAdd(&mismatches[0], 1U);
    atomicOr(&mismatches[1], mismatch);
  }
}

#endif

template <typename T, typename Kernel>
float run_kernel(const size_t count, const int inner_iters, Kernel kernel) {
  T *out = nullptr;
  check_cuda(cudaMalloc(&out, sizeof(T) * count));
  const int blocks =
      static_cast<int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const float elapsed_ms = time_cuda_launch([&]() {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(out, count, inner_iters);
  });
  check_cuda(cudaGetLastError());
  check_cuda(cudaFree(out));
  return elapsed_ms;
}

template <typename Out, typename Input, typename SeedKernel, typename Kernel>
float run_seeded_input_kernel(const size_t count, const int inner_iters,
                              const uint32_t lhs_stream,
                              const uint32_t rhs_stream,
                              SeedKernel seed_kernel, Kernel kernel) {
  Out *out = nullptr;
  Input *lhs = nullptr;
  Input *rhs = nullptr;
  check_cuda(cudaMalloc(&out, sizeof(Out) * count));
  check_cuda(cudaMalloc(&lhs, sizeof(Input) * count));
  check_cuda(cudaMalloc(&rhs, sizeof(Input) * count));
  const int blocks =
      static_cast<int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  seed_kernel<<<blocks, THREADS_PER_BLOCK>>>(lhs, rhs, count, lhs_stream,
                                             rhs_stream);
  check_cuda(cudaGetLastError());
  const float elapsed_ms = time_cuda_launch([&]() {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(out, lhs, rhs, count, inner_iters);
  });
  check_cuda(cudaGetLastError());
  check_cuda(cudaFree(rhs));
  check_cuda(cudaFree(lhs));
  check_cuda(cudaFree(out));
  return elapsed_ms;
}

template <typename Out, typename Kernel>
float run_fq32_seeded_kernel(const size_t count, const int inner_iters,
                             const uint32_t lhs_stream,
                             const uint32_t rhs_stream, Kernel kernel) {
  return run_seeded_input_kernel<Out, exp_fq32_t>(
      count, inner_iters, lhs_stream, rhs_stream, fq32_seed_inputs_kernel,
      kernel);
}

#ifdef BB_GPU_HAVE_ICICLE_V28
template <typename Out, typename Kernel>
float run_icicle_seeded_kernel(const size_t count, const int inner_iters,
                               const uint32_t lhs_stream,
                               const uint32_t rhs_stream, Kernel kernel) {
  return run_seeded_input_kernel<Out, icicle_fq_t>(
      count, inner_iters, lhs_stream, rhs_stream, icicle_seed_inputs_kernel,
      kernel);
}
#endif

} // namespace

extern "C" int bb_gpu_field_bench_has_icicle_v28() {
#ifdef BB_GPU_HAVE_ICICLE_V28
  return 1;
#else
  return 0;
#endif
}

extern "C" int bb_gpu_field_bench_cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

#ifdef BB_GPU_HAVE_ICICLE_V28
template <typename Fq32Out, typename IcicleOut, typename Fq32Kernel,
          typename IcicleKernel>
int run_field_pair_impl(const size_t count, const int inner_iters,
                        const uint32_t lhs_stream, const uint32_t rhs_stream,
                        const int reverse_order, float *fq32_ms,
                        float *icicle_ms, Fq32Kernel fq32_kernel,
                        IcicleKernel icicle_kernel) {
  if (fq32_ms == nullptr || icicle_ms == nullptr) {
    return -1;
  }
  const int blocks =
      static_cast<int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  Fq32Out *fq32_out = nullptr;
  exp_fq32_t *fq32_lhs = nullptr;
  exp_fq32_t *fq32_rhs = nullptr;
  IcicleOut *icicle_out = nullptr;
  icicle_fq_t *icicle_lhs = nullptr;
  icicle_fq_t *icicle_rhs = nullptr;

  check_cuda(cudaMalloc(&fq32_out, sizeof(Fq32Out) * count));
  check_cuda(cudaMalloc(&fq32_lhs, sizeof(exp_fq32_t) * count));
  check_cuda(cudaMalloc(&fq32_rhs, sizeof(exp_fq32_t) * count));
  check_cuda(cudaMalloc(&icicle_out, sizeof(IcicleOut) * count));
  check_cuda(cudaMalloc(&icicle_lhs, sizeof(icicle_fq_t) * count));
  check_cuda(cudaMalloc(&icicle_rhs, sizeof(icicle_fq_t) * count));

  fq32_seed_inputs_kernel<<<blocks, THREADS_PER_BLOCK>>>(fq32_lhs, fq32_rhs,
                                                         count, lhs_stream,
                                                         rhs_stream);
  check_cuda(cudaGetLastError());
  icicle_seed_inputs_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      icicle_lhs, icicle_rhs, count, lhs_stream, rhs_stream);
  check_cuda(cudaGetLastError());

  auto run_fq32 = [&]() {
    return time_cuda_launch([&]() {
      fq32_kernel<<<blocks, THREADS_PER_BLOCK>>>(fq32_out, fq32_lhs, fq32_rhs,
                                                 count, inner_iters);
    });
  };
  auto run_icicle = [&]() {
    return time_cuda_launch([&]() {
      icicle_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          icicle_out, icicle_lhs, icicle_rhs, count, inner_iters);
    });
  };

  if (reverse_order != 0) {
    *icicle_ms = run_icicle();
    check_cuda(cudaGetLastError());
    *fq32_ms = run_fq32();
    check_cuda(cudaGetLastError());
  } else {
    *fq32_ms = run_fq32();
    check_cuda(cudaGetLastError());
    *icicle_ms = run_icicle();
    check_cuda(cudaGetLastError());
  }

  check_cuda(cudaFree(icicle_rhs));
  check_cuda(cudaFree(icicle_lhs));
  check_cuda(cudaFree(icicle_out));
  check_cuda(cudaFree(fq32_rhs));
  check_cuda(cudaFree(fq32_lhs));
  check_cuda(cudaFree(fq32_out));
  return 0;
}
#endif

extern "C" int
bb_gpu_field_bench_run_field_pair(const int case_id, const int log_elements,
                                  const int inner_iters,
                                  const int reverse_order, float *fq32_ms,
                                  float *icicle_ms) {
#ifdef BB_GPU_HAVE_ICICLE_V28
  const size_t count = size_t{1} << log_elements;
  switch (static_cast<bench_case>(case_id)) {
  case bench_case::FQ32_ADD:
    return run_field_pair_impl<exp_fq32_t, icicle_fq_t>(
        count, inner_iters, 11U, 29U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_add_kernel, icicle_input_field_add_kernel);
  case bench_case::FQ32_SUB:
    return run_field_pair_impl<exp_fq32_t, icicle_fq_t>(
        count, inner_iters, 41U, 7U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_sub_kernel, icicle_input_field_sub_kernel);
  case bench_case::FQ32_NEG:
    return run_field_pair_impl<exp_fq32_t, icicle_fq_t>(
        count, inner_iters, 43U, 0U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_neg_kernel, icicle_input_field_neg_kernel);
  case bench_case::FQ32_IS_ZERO:
    return run_field_pair_impl<uint32_t, uint32_t>(
        count, inner_iters, 47U, 0U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_is_zero_kernel, icicle_input_field_is_zero_kernel);
  case bench_case::FQ32_EQUAL:
    return run_field_pair_impl<uint32_t, uint32_t>(
        count, inner_iters, 53U, 59U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_equal_kernel, icicle_input_field_equal_kernel);
  case bench_case::FQ32_STRAIGHTLINE_MUL:
    return run_field_pair_impl<exp_fq32_t, icicle_fq_t>(
        count, inner_iters, 14U, 32U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_mul_kernel, icicle_input_field_mul_kernel);
  case bench_case::FQ32_STRAIGHTLINE_SQR:
    return run_field_pair_impl<exp_fq32_t, icicle_fq_t>(
        count, inner_iters, 18U, 0U, reverse_order, fq32_ms, icicle_ms,
        fq32_input_field_sqr_kernel, icicle_input_field_sqr_kernel);
  default:
    return -1;
  }
#else
  (void)case_id;
  (void)log_elements;
  (void)inner_iters;
  (void)reverse_order;
  (void)fq32_ms;
  (void)icicle_ms;
  return -1;
#endif
}

extern "C" int
bb_gpu_field_bench_validate_fq32_vs_icicle(const int log_elements) {
#ifdef BB_GPU_HAVE_ICICLE_V28
  const size_t count = size_t{1} << log_elements;
  uint32_t *device_mismatches = nullptr;
  check_cuda(cudaMalloc(&device_mismatches, sizeof(uint32_t) * 2));
  check_cuda(cudaMemset(device_mismatches, 0, sizeof(uint32_t) * 2));
  const int blocks =
      static_cast<int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  fq32_icicle_validation_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      device_mismatches, count);
  check_cuda(cudaGetLastError());
  uint32_t mismatches[2] = {};
  check_cuda(cudaMemcpy(mismatches, device_mismatches, sizeof(uint32_t) * 2,
                        cudaMemcpyDeviceToHost));
  check_cuda(cudaFree(device_mismatches));
  if (mismatches[0] != 0) {
    std::fprintf(stderr, "fq32-vs-Icicle validation failed: count=%u mask=0x%x\n",
                 mismatches[0], mismatches[1]);
  }
  return static_cast<int>(mismatches[1]);
#else
  (void)log_elements;
  return -1;
#endif
}

extern "C" float bb_gpu_field_bench_run(const int case_id,
                                        const int log_elements,
                                        const int inner_iters) {
  const size_t count = size_t{1} << log_elements;
  switch (static_cast<bench_case>(case_id)) {
  case bench_case::BB_ADD:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_add_kernel);
  case bench_case::BB_SUB:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_sub_kernel);
  case bench_case::BB_NEG:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_neg_kernel);
  case bench_case::BB_IS_ZERO:
    return run_kernel<uint32_t>(count, inner_iters, bb_field_is_zero_kernel);
  case bench_case::BB_EQUAL:
    return run_kernel<uint32_t>(count, inner_iters, bb_field_equal_kernel);
  case bench_case::BB_MUL:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_mul_kernel);
  case bench_case::BB_SQR:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_sqr_kernel);
  case bench_case::BB_SQR_NO_PREREDUCE:
    return run_kernel<bb_fq_t>(count, inner_iters,
                               bb_field_sqr_no_prereduce_kernel);
  case bench_case::BB_SQR_DEDICATED_NO_PREREDUCE:
    return run_kernel<bb_fq_t>(count, inner_iters,
                               bb_field_sqr_dedicated_no_prereduce_kernel);
  case bench_case::BB_XYZZ_MIXED_ADD:
    return run_kernel<bb_xyzz_t>(count, inner_iters, bb_xyzz_mixed_add_kernel);
  case bench_case::BB_ADD_NO_PREREDUCE:
    return run_kernel<bb_fq_t>(count, inner_iters,
                               bb_field_add_no_prereduce_kernel);
  case bench_case::BB_SUB_NO_PREREDUCE:
    return run_kernel<bb_fq_t>(count, inner_iters,
                               bb_field_sub_no_prereduce_kernel);
  case bench_case::BB_MUL_NO_PREREDUCE:
    return run_kernel<bb_fq_t>(count, inner_iters,
                               bb_field_mul_no_prereduce_kernel);
  case bench_case::BB32_ADD:
    return run_kernel<bb32_fq_t>(count, inner_iters, bb32_field_add_kernel);
  case bench_case::BB32_MUL:
    return run_kernel<bb32_fq_t>(count, inner_iters, bb32_field_mul_kernel);
  case bench_case::BB32_PTX_MUL:
    return run_kernel<bb32_fq_t>(count, inner_iters, bb32_ptx_field_mul_kernel);
  case bench_case::BB32_BARRETT_MUL:
    return run_kernel<bb32_fq_t>(count, inner_iters,
                                 bb32_barrett_field_mul_kernel);
  case bench_case::BB32_BARRETT_TRUNC_MUL:
    return run_kernel<bb32_fq_t>(count, inner_iters,
                                 bb32_barrett_truncated_field_mul_kernel);
  case bench_case::BB32_BARRETT_TRUNC_PTX_MUL:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  bb32_barrett_truncated_ptx_field_mul_kernel);
  case bench_case::FQ32_STRAIGHTLINE_MUL:
    return run_fq32_seeded_kernel<exp_fq32_t>(
        count, inner_iters, 14U, 32U, fq32_input_field_mul_kernel);
  case bench_case::FQ32_STRAIGHTLINE_SQR:
    return run_fq32_seeded_kernel<exp_fq32_t>(
        count, inner_iters, 18U, 0U, fq32_input_field_sqr_kernel);
  case bench_case::FQ32_ADD:
    return run_fq32_seeded_kernel<exp_fq32_t>(
        count, inner_iters, 11U, 29U, fq32_input_field_add_kernel);
  case bench_case::FQ32_SUB:
    return run_fq32_seeded_kernel<exp_fq32_t>(
        count, inner_iters, 41U, 7U, fq32_input_field_sub_kernel);
  case bench_case::FQ32_NEG:
    return run_fq32_seeded_kernel<exp_fq32_t>(
        count, inner_iters, 43U, 0U, fq32_input_field_neg_kernel);
  case bench_case::FQ32_IS_ZERO:
    return run_fq32_seeded_kernel<uint32_t>(
        count, inner_iters, 47U, 0U, fq32_input_field_is_zero_kernel);
  case bench_case::FQ32_EQUAL:
    return run_fq32_seeded_kernel<uint32_t>(
        count, inner_iters, 53U, 59U, fq32_input_field_equal_kernel);
  case bench_case::FQ32_CALLABLE_MUL:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_callable_field_mul_kernel);
  case bench_case::FQ32_CALLABLE_SQR:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_callable_field_sqr_kernel);
  case bench_case::FQ32_DEDICATED_SQR:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_dedicated_field_sqr_kernel);
  case bench_case::FQ32_REDUCE_ONLY_BARRETT:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_reduce_only_barrett_kernel);
  case bench_case::FQ32_REDUCE_ONLY_BARRETT_REPRESENTATIVE:
    return run_kernel<exp_fq32_t>(
        count, inner_iters, fq32_reduce_only_barrett_representative_kernel);
  case bench_case::FQ32_KARATSUBA_MUL:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_karatsuba_field_mul_kernel);
  case bench_case::FQ32_KARATSUBA_FUSED_MUL:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_karatsuba_fused_field_mul_kernel);
  case bench_case::FQ32_HALF_PRODUCT_MUL:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_half_product_field_mul_kernel);
  case bench_case::FQ32_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(count, inner_iters,
                                       fq32_wide_product_kernel);
  case bench_case::FQ32_STRAIGHTLINE_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(count, inner_iters,
                                       fq32_straightline_wide_product_kernel);
  case bench_case::FQ32_KARATSUBA_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(count, inner_iters,
                                       fq32_karatsuba_wide_product_kernel);
  case bench_case::FQ32_KARATSUBA_FUSED_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(
        count, inner_iters, fq32_karatsuba_fused_wide_product_kernel);
  case bench_case::FQ32_HALF_PRODUCT_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(
        count, inner_iters, fq32_half_product_wide_product_kernel);
  case bench_case::BB_XYZZ_MIXED_ADD_UNCHECKED:
    return run_kernel<bb_xyzz_t>(count, inner_iters,
                                 bb_xyzz_mixed_add_unchecked_kernel);
  case bench_case::BB_XYZZ_MIXED_ADD_ASSUME_FINITE:
    return run_kernel<bb_xyzz_t>(count, inner_iters,
                                 bb_xyzz_mixed_add_assume_finite_kernel);
  case bench_case::BB_JACOBIAN_MIXED_ADD_UNCHECKED:
    return run_kernel<bb_jacobian_t>(count, inner_iters,
                                     bb_jacobian_mixed_add_unchecked_kernel);
  case bench_case::FQ32_CURVE_SHAPE_XYZZ_MIXED_ADD:
    return run_kernel<exp_fq32_xyzz_t>(count, inner_iters,
                                       fq32_curve_shape_xyzz_mixed_add_kernel);
  case bench_case::FQ32_CURVE_SHAPE_XYZZ_ADD:
    return run_kernel<exp_fq32_xyzz_t>(count, inner_iters,
                                       fq32_curve_shape_xyzz_add_kernel);
  case bench_case::FQ32_CURVE_SHAPE_XYZZ_DOUBLE:
    return run_kernel<exp_fq32_xyzz_t>(count, inner_iters,
                                       fq32_curve_shape_xyzz_double_kernel);
  case bench_case::BB_CURVE_SHAPE_XYZZ_ADD:
    return run_kernel<bb_xyzz_t>(count, inner_iters,
                                 bb_curve_shape_xyzz_add_kernel);
  case bench_case::BB_CURVE_SHAPE_XYZZ_DOUBLE:
    return run_kernel<bb_xyzz_t>(count, inner_iters,
                                 bb_curve_shape_xyzz_double_kernel);
#ifdef BB_GPU_HAVE_ICICLE_V28
  case bench_case::ICICLE_ADD:
    return run_icicle_seeded_kernel<icicle_fq_t>(
        count, inner_iters, 11U, 29U, icicle_input_field_add_kernel);
  case bench_case::ICICLE_SUB:
    return run_icicle_seeded_kernel<icicle_fq_t>(
        count, inner_iters, 41U, 7U, icicle_input_field_sub_kernel);
  case bench_case::ICICLE_NEG:
    return run_icicle_seeded_kernel<icicle_fq_t>(
        count, inner_iters, 43U, 0U, icicle_input_field_neg_kernel);
  case bench_case::ICICLE_IS_ZERO:
    return run_icicle_seeded_kernel<uint32_t>(
        count, inner_iters, 47U, 0U, icicle_input_field_is_zero_kernel);
  case bench_case::ICICLE_EQUAL:
    return run_icicle_seeded_kernel<uint32_t>(
        count, inner_iters, 53U, 59U, icicle_input_field_equal_kernel);
  case bench_case::ICICLE_WIDE_PRODUCT:
    return run_kernel<icicle_wide_t>(count, inner_iters,
                                     icicle_wide_product_kernel);
  case bench_case::ICICLE_REDUCE_ONLY:
    return run_kernel<icicle_fq_t>(count, inner_iters,
                                   icicle_reduce_only_kernel);
  case bench_case::ICICLE_MUL:
    return run_icicle_seeded_kernel<icicle_fq_t>(
        count, inner_iters, 14U, 32U, icicle_input_field_mul_kernel);
  case bench_case::ICICLE_SQR:
    return run_icicle_seeded_kernel<icicle_fq_t>(
        count, inner_iters, 18U, 0U, icicle_input_field_sqr_kernel);
  case bench_case::ICICLE_PROJECTIVE_MIXED_ADD:
    return run_kernel<icicle_projective_t>(count, inner_iters,
                                           icicle_projective_mixed_add_kernel);
  case bench_case::ICICLE_XYZZ_MIXED_ADD_UNCHECKED:
    return run_kernel<icicle_xyzz_t>(count, inner_iters,
                                     icicle_xyzz_mixed_add_unchecked_kernel);
  case bench_case::ICICLE_XYZZ_MIXED_ADD_CHECKED:
    return run_kernel<icicle_xyzz_t>(count, inner_iters,
                                     icicle_xyzz_mixed_add_checked_kernel);
#else
  case bench_case::ICICLE_ADD:
  case bench_case::ICICLE_SUB:
  case bench_case::ICICLE_NEG:
  case bench_case::ICICLE_IS_ZERO:
  case bench_case::ICICLE_EQUAL:
  case bench_case::ICICLE_WIDE_PRODUCT:
  case bench_case::ICICLE_REDUCE_ONLY:
  case bench_case::ICICLE_MUL:
  case bench_case::ICICLE_SQR:
  case bench_case::ICICLE_PROJECTIVE_MIXED_ADD:
  case bench_case::ICICLE_XYZZ_MIXED_ADD_UNCHECKED:
  case bench_case::ICICLE_XYZZ_MIXED_ADD_CHECKED:
    return -1.0F;
#endif
  }
  return -1.0F;
}
