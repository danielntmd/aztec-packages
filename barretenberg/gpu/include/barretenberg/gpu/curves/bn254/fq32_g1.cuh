#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_defines.cuh"
#include "barretenberg/gpu/fields/bn254/fq32.cuh"
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS
#include "barretenberg/gpu/fields/bn254/icicle_fq32.cuh"
#endif

namespace bb::gpu::bn254 {

struct alignas(64) fq32_affine_g1_t {
  fq32_t x;
  fq32_t y;
};

// EFD short-Weierstrass XYZZ: affine x = X / ZZ, y = Y / ZZZ.
struct alignas(32) fq32_xyzz_g1_t {
  fq32_t x;
  fq32_t y;
  fq32_t zz;
  fq32_t zzz;
  bool infinity;
};

BB_GPU_HD_FORCEINLINE fq32_t fq32_add(const fq32_t &lhs, const fq32_t &rhs) {
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS
  return detail::icicle_fq32_add(lhs, rhs);
#else
  return add(lhs, rhs);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t fq32_sub(const fq32_t &lhs, const fq32_t &rhs) {
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS
  return detail::icicle_fq32_sub(lhs, rhs);
#else
  return sub(lhs, rhs);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t fq32_mul(const fq32_t &lhs, const fq32_t &rhs) {
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS
  return detail::icicle_fq32_mul(lhs, rhs);
#else
  return mul_straightline(lhs, rhs);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t fq32_sqr(const fq32_t &value) {
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS
  return detail::icicle_fq32_sqr(value);
#else
  return fq32_mul(value, value);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t fq32_inv(const fq32_t &value) {
  return inv(value);
}

BB_GPU_HD_FORCEINLINE bool fq32_eq(const fq32_t &lhs, const fq32_t &rhs) {
  const fq32_t lhs_normalized = normalize(lhs);
  const fq32_t rhs_normalized = normalize(rhs);
  uint32_t diff = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    diff |= lhs_normalized.limbs[i] ^ rhs_normalized.limbs[i];
  }
  return diff == 0;
}

BB_GPU_HD_FORCEINLINE fq32_affine_g1_t fq32_affine_infinity() {
  fq32_affine_g1_t out{fq32_t::zero(), fq32_t::zero()};
  self_set_msb(out.x);
  return out;
}

BB_GPU_HD_FORCEINLINE bool is_infinity(const fq32_affine_g1_t &point) {
  return is_msb_set(point.x);
}

BB_GPU_HD_FORCEINLINE fq32_xyzz_g1_t fq32_xyzz_infinity() {
  return {fq32_t::zero(), fq32_t::zero(), fq32_t::zero(), fq32_t::zero(), true};
}

BB_GPU_HD_FORCEINLINE bool is_infinity(const fq32_xyzz_g1_t &point) {
  return point.infinity || is_zero(point.zz);
}

BB_GPU_HD_FORCEINLINE fq32_xyzz_g1_t
fq32_to_xyzz(const fq32_affine_g1_t &point) {
  return is_infinity(point) ? fq32_xyzz_infinity()
                            : fq32_xyzz_g1_t{point.x, point.y, fq32_t::one(),
                                             fq32_t::one(), false};
}

BB_GPU_HD_FORCEINLINE fq32_affine_g1_t
fq32_xyzz_to_affine(const fq32_xyzz_g1_t &point) {
  if (is_infinity(point)) {
    return fq32_affine_infinity();
  }
  // EFD XYZZ scaling: A = 1 / ZZZ, B = (ZZ * A)^2.
  const fq32_t zzz_inv = fq32_inv(point.zzz);
  const fq32_t z_inv = fq32_mul(point.zz, zzz_inv);
  const fq32_t zz_inv = fq32_sqr(z_inv);
  return {fq32_mul(point.x, zz_inv), fq32_mul(point.y, zzz_inv)};
}

BB_GPU_HD_FORCEINLINE fq32_affine_g1_t
fq32_affine_neg(const fq32_affine_g1_t &point) {
  // EFD affine negation on short Weierstrass curves: -(x, y) = (x, -y).
  return is_infinity(point) ? point : fq32_affine_g1_t{point.x, neg(point.y)};
}

BB_GPU_HD_FORCEINLINE bool fq32_on_curve(const fq32_affine_g1_t &point) {
  if (is_infinity(point)) {
    return true;
  }
  // BN254 G1 is the short-Weierstrass curve y^2 = x^3 + 3.
  const fq32_t x2 = fq32_sqr(point.x);
  const fq32_t x3 = fq32_mul(x2, point.x);
  const fq32_t rhs = fq32_add(x3, fq32_t::from_u32(3));
  const fq32_t lhs = fq32_sqr(point.y);
  return fq32_eq(lhs, rhs);
}

BB_GPU_HD_FORCEINLINE void self_double(fq32_xyzz_g1_t &point) {
  if (is_infinity(point)) {
    return;
  }

  // EFD XYZZ dbl-2008-s-1 with a = 0 for BN254.
  fq32_t u = fq32_add(point.y, point.y);
  if (is_zero(u)) {
    point = fq32_xyzz_infinity();
    return;
  }

  const fq32_t v = fq32_sqr(u);
  const fq32_t w = fq32_mul(u, v);
  const fq32_t s = fq32_mul(point.x, v);
  fq32_t m = fq32_sqr(point.x);
  m = fq32_add(fq32_add(m, m), m);
  const fq32_t two_s = fq32_add(s, s);
  const fq32_t x3 = fq32_sub(fq32_sqr(m), two_s);
  point.y = fq32_sub(fq32_mul(m, fq32_sub(s, x3)), fq32_mul(w, point.y));
  point.x = x3;
  point.zz = fq32_mul(v, point.zz);
  point.zzz = fq32_mul(w, point.zzz);
}

BB_GPU_HD_FORCEINLINE void
fq32_xyzz_mixed_add_assume_finite(fq32_xyzz_g1_t &lhs,
                                  const fq32_affine_g1_t &rhs) {
  // EFD XYZZ madd-2008-s: rhs is affine, so ZZ2 = ZZZ2 = 1.
  fq32_t p = fq32_sub(fq32_mul(rhs.x, lhs.zz), lhs.x);
  fq32_t r = fq32_sub(fq32_mul(rhs.y, lhs.zzz), lhs.y);

  if (is_zero(p)) {
    if (is_zero(r)) {
      self_double(lhs);
    } else {
      lhs = fq32_xyzz_infinity();
    }
    return;
  }

  fq32_t pp = fq32_sqr(p);
  fq32_t q = fq32_mul(lhs.x, pp);
  fq32_t ppp = fq32_mul(p, pp);
  lhs.zz = fq32_mul(lhs.zz, pp);
  fq32_t x3 = fq32_sqr(r);
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

BB_GPU_HD_FORCEINLINE void
fq32_xyzz_mixed_add_zz1_equals_one_assume_finite(fq32_xyzz_g1_t &lhs,
                                                 const fq32_affine_g1_t &rhs) {
  // EFD XYZZ mmadd-2008-s: both inputs start with ZZ = ZZZ = 1.
  fq32_t p = fq32_sub(rhs.x, lhs.x);
  fq32_t r = fq32_sub(rhs.y, lhs.y);
  if (is_zero(p)) {
    if (is_zero(r)) {
      self_double(lhs);
    } else {
      lhs = fq32_xyzz_infinity();
    }
    return;
  }

  fq32_t pp = fq32_sqr(p);
  fq32_t q = fq32_mul(lhs.x, pp);
  fq32_t ppp = fq32_mul(p, pp);
  lhs.zz = pp;
  fq32_t x3 = fq32_sqr(r);
  x3 = fq32_sub(x3, ppp);
  pp = fq32_add(q, q);
  x3 = fq32_sub(x3, pp);
  lhs.zzz = ppp;
  q = fq32_sub(q, x3);
  q = fq32_mul(r, q);
  ppp = fq32_mul(lhs.y, ppp);
  lhs.y = fq32_sub(q, ppp);
  lhs.x = x3;
}

BB_GPU_HD_FORCEINLINE void fq32_xyzz_mixed_add(fq32_xyzz_g1_t &lhs,
                                               const fq32_affine_g1_t &rhs) {
  if (is_infinity(rhs)) {
    return;
  }
  if (is_infinity(lhs)) {
    lhs = fq32_to_xyzz(rhs);
    return;
  }
  fq32_xyzz_mixed_add_assume_finite(lhs, rhs);
}

BB_GPU_HD_FORCEINLINE void fq32_xyzz_add_assign(fq32_xyzz_g1_t &lhs,
                                                const fq32_xyzz_g1_t &rhs) {
  if (is_infinity(lhs)) {
    lhs = rhs;
    return;
  }
  if (is_infinity(rhs)) {
    return;
  }

  // EFD XYZZ add-2008-s for two arbitrary XYZZ points.
  fq32_t u1 = fq32_mul(lhs.x, rhs.zz);
  fq32_t u2 = fq32_mul(rhs.x, lhs.zz);
  fq32_t s1 = fq32_mul(lhs.y, rhs.zzz);
  fq32_t s2 = fq32_mul(rhs.y, lhs.zzz);
  fq32_t p = fq32_sub(u2, u1);
  fq32_t r = fq32_sub(s2, s1);

  if (is_zero(p)) {
    if (is_zero(r)) {
      self_double(lhs);
    } else {
      lhs = fq32_xyzz_infinity();
    }
    return;
  }

  fq32_t pp = fq32_sqr(p);
  p = fq32_mul(p, pp);
  u1 = fq32_mul(u1, pp);
  fq32_t x3 = fq32_sub(fq32_sqr(r), p);
  x3 = fq32_sub(x3, fq32_add(u1, u1));
  s1 = fq32_mul(s1, p);
  u1 = fq32_mul(r, fq32_sub(u1, x3));
  lhs.y = fq32_sub(u1, s1);
  lhs.x = x3;
  lhs.zz = fq32_mul(fq32_mul(lhs.zz, rhs.zz), pp);
  lhs.zzz = fq32_mul(fq32_mul(lhs.zzz, rhs.zzz), p);
}

BB_GPU_HD_FORCEINLINE void
fq32_xyzz_add_assign_rhs_finite(fq32_xyzz_g1_t &lhs,
                                const fq32_xyzz_g1_t &rhs) {
  if (is_infinity(lhs)) {
    lhs = rhs;
    return;
  }

  fq32_t u1 = fq32_mul(lhs.x, rhs.zz);
  fq32_t u2 = fq32_mul(rhs.x, lhs.zz);
  fq32_t s1 = fq32_mul(lhs.y, rhs.zzz);
  fq32_t s2 = fq32_mul(rhs.y, lhs.zzz);
  fq32_t p = fq32_sub(u2, u1);
  fq32_t r = fq32_sub(s2, s1);

  if (is_zero(p)) {
    if (is_zero(r)) {
      self_double(lhs);
    } else {
      lhs = fq32_xyzz_infinity();
    }
    return;
  }

  fq32_t pp = fq32_sqr(p);
  p = fq32_mul(p, pp);
  u1 = fq32_mul(u1, pp);
  fq32_t x3 = fq32_sub(fq32_sqr(r), p);
  x3 = fq32_sub(x3, fq32_add(u1, u1));
  s1 = fq32_mul(s1, p);
  u1 = fq32_mul(r, fq32_sub(u1, x3));
  lhs.y = fq32_sub(u1, s1);
  lhs.x = x3;
  lhs.zz = fq32_mul(fq32_mul(lhs.zz, rhs.zz), pp);
  lhs.zzz = fq32_mul(fq32_mul(lhs.zzz, rhs.zzz), p);
}

BB_GPU_HD_FORCEINLINE void fq32_chained_xyzz_mixed_add_indexed_nonzero(
    fq32_xyzz_g1_t &accumulator, const fq32_affine_g1_t *points,
    const uint32_t *point_indices, const int start, const int count) {
  if (count <= 0) {
    accumulator = fq32_xyzz_infinity();
    return;
  }

  int offset = 0;
  fq32_affine_g1_t point = points[point_indices[start + offset]];
  accumulator = {point.x, point.y, fq32_t::one(), fq32_t::one(), false};
  ++offset;
  if (offset < count) {
    fq32_xyzz_mixed_add_zz1_equals_one_assume_finite(
        accumulator, points[point_indices[start + offset]]);
    ++offset;
  }
  for (; offset < count; ++offset) {
    point = points[point_indices[start + offset]];
    if (accumulator.infinity) {
      accumulator = {point.x, point.y, fq32_t::one(), fq32_t::one(), false};
    } else {
      fq32_xyzz_mixed_add_assume_finite(accumulator, point);
    }
  }
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
