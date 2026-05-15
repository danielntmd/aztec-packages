#include "barretenberg/gpu/curves/bn254/fq.cuh"
#include "barretenberg/gpu/curves/bn254/fq32.cuh"
#include "barretenberg/gpu/curves/bn254/g1.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_straightline_field_mul_kernel(exp_fq32_t *out,
                                                   const size_t count,
                                                   const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul_straightline(a, b);
    b = bb::gpu::bn254::experimental::add(
        b, exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 1));
  }
  out[idx] = a;
}

__global__ void fq32_straightline_field_sqr_kernel(exp_fq32_t *out,
                                                   const size_t count,
                                                   const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 18);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::add(
        bb::gpu::bn254::experimental::sqr_straightline(a),
        exp_fq32_t::from_u32(static_cast<uint32_t>(i) + 3));
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

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
  for (int i = 0; i < inner_iters; ++i) {
    a = bb::gpu::bn254::experimental::mul_karatsuba(a, b);
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

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
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

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
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

  exp_fq32_t a = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 14);
  exp_fq32_t b = exp_fq32_t::from_u32(static_cast<uint32_t>(idx) + 32);
  exp_fq32_wide_t wide{};
  for (int i = 0; i < inner_iters; ++i) {
    bb::gpu::bn254::experimental::mul_wide_karatsuba(a, b, wide.limbs);
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
    point.x = point.x + make_bb_fq(static_cast<uint32_t>(i) + 1);
    point.y = point.y + make_bb_fq(static_cast<uint32_t>(i) + 7);
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
  bb_fq_t p = rhs.x * lhs.zz;
  p = p - lhs.x;
  bb_fq_t r = rhs.y * lhs.zzz;
  r = r - lhs.y;
  bb_fq_t pp = p.sqr();
  bb_fq_t ppp = p * pp;
  lhs.zz = lhs.zz * pp;
  lhs.zzz = lhs.zzz * ppp;
  bb_fq_t q = lhs.x * pp;
  bb_fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
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
    point.x = point.x + make_bb_fq(static_cast<uint32_t>(i) + 1);
    point.y = point.y + make_bb_fq(static_cast<uint32_t>(i) + 7);
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
    point.x = point.x + make_bb_fq(static_cast<uint32_t>(i) + 1);
    point.y = point.y + make_bb_fq(static_cast<uint32_t>(i) + 7);
  }
  out[idx] = accumulator;
}

#ifdef BB_GPU_HAVE_ICICLE_V28

using icicle_fq_t = bn254::point_field_t;
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

__global__ void icicle_field_add_kernel(icicle_fq_t *out, const size_t count,
                                        const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = make_icicle_fq(static_cast<uint32_t>(idx) + 11);
  icicle_fq_t b = make_icicle_fq(static_cast<uint32_t>(idx) + 29);
  for (int i = 0; i < inner_iters; ++i) {
    a = a + b;
    b = b + a;
  }
  out[idx] = a + b;
}

__global__ void icicle_field_mul_kernel(icicle_fq_t *out, const size_t count,
                                        const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = make_icicle_fq(static_cast<uint32_t>(idx) + 13);
  icicle_fq_t b = make_icicle_fq(static_cast<uint32_t>(idx) + 31);
  for (int i = 0; i < inner_iters; ++i) {
    a = a * b;
    b = b + make_icicle_fq(static_cast<uint32_t>(i) + 1);
  }
  out[idx] = a;
}

__global__ void icicle_field_sqr_kernel(icicle_fq_t *out, const size_t count,
                                        const int inner_iters) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  icicle_fq_t a = make_icicle_fq(static_cast<uint32_t>(idx) + 17);
  for (int i = 0; i < inner_iters; ++i) {
    a = icicle_fq_t::sqr(a) + make_icicle_fq(static_cast<uint32_t>(i) + 3);
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
    point.x = point.x + make_icicle_fq(static_cast<uint32_t>(i) + 1);
    point.y = point.y + make_icicle_fq(static_cast<uint32_t>(i) + 7);
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
    point.x = point.x + make_icicle_fq(static_cast<uint32_t>(i) + 1);
    point.y = point.y + make_icicle_fq(static_cast<uint32_t>(i) + 7);
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
    point.x = point.x + make_icicle_fq(static_cast<uint32_t>(i) + 1);
    point.y = point.y + make_icicle_fq(static_cast<uint32_t>(i) + 7);
  }
  out[idx] = accumulator;
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

extern "C" float bb_gpu_field_bench_run(const int case_id,
                                        const int log_elements,
                                        const int inner_iters) {
  const size_t count = size_t{1} << log_elements;
  switch (static_cast<bench_case>(case_id)) {
  case bench_case::BB_ADD:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_add_kernel);
  case bench_case::BB_MUL:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_mul_kernel);
  case bench_case::BB_SQR:
    return run_kernel<bb_fq_t>(count, inner_iters, bb_field_sqr_kernel);
  case bench_case::BB_XYZZ_MIXED_ADD:
    return run_kernel<bb_xyzz_t>(count, inner_iters, bb_xyzz_mixed_add_kernel);
  case bench_case::BB_ADD_NO_PREREDUCE:
    return run_kernel<bb_fq_t>(count, inner_iters,
                               bb_field_add_no_prereduce_kernel);
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
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_straightline_field_mul_kernel);
  case bench_case::FQ32_STRAIGHTLINE_SQR:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_straightline_field_sqr_kernel);
  case bench_case::FQ32_KARATSUBA_MUL:
    return run_kernel<exp_fq32_t>(count, inner_iters,
                                  fq32_karatsuba_field_mul_kernel);
  case bench_case::FQ32_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(count, inner_iters,
                                       fq32_wide_product_kernel);
  case bench_case::FQ32_STRAIGHTLINE_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(
        count, inner_iters, fq32_straightline_wide_product_kernel);
  case bench_case::FQ32_KARATSUBA_WIDE_PRODUCT:
    return run_kernel<exp_fq32_wide_t>(count, inner_iters,
                                       fq32_karatsuba_wide_product_kernel);
  case bench_case::BB_XYZZ_MIXED_ADD_UNCHECKED:
    return run_kernel<bb_xyzz_t>(count, inner_iters,
                                 bb_xyzz_mixed_add_unchecked_kernel);
  case bench_case::BB_JACOBIAN_MIXED_ADD_UNCHECKED:
    return run_kernel<bb_jacobian_t>(count, inner_iters,
                                     bb_jacobian_mixed_add_unchecked_kernel);
#ifdef BB_GPU_HAVE_ICICLE_V28
  case bench_case::ICICLE_ADD:
    return run_kernel<icicle_fq_t>(count, inner_iters, icicle_field_add_kernel);
  case bench_case::ICICLE_MUL:
    return run_kernel<icicle_fq_t>(count, inner_iters, icicle_field_mul_kernel);
  case bench_case::ICICLE_SQR:
    return run_kernel<icicle_fq_t>(count, inner_iters, icicle_field_sqr_kernel);
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
