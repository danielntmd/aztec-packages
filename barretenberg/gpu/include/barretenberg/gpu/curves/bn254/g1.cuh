#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/fq.cuh"
#include "barretenberg/gpu/curves/bn254/fr.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

struct alignas(64) affine_g1_t {
  fq_t x;
  fq_t y;
};

struct alignas(32) jacobian_g1_t {
  fq_t x;
  fq_t y;
  fq_t z;
  bool infinity;
};

struct alignas(32) xyzz_g1_t {
  fq_t x;
  fq_t y;
  fq_t zz;
  fq_t zzz;
  bool infinity;
};

BB_GPU_HD inline bool is_infinity(const affine_g1_t &point) {
  return point.x.is_msb_set();
}

BB_GPU_HD inline affine_g1_t affine_infinity() {
  affine_g1_t out{fq_t::zero(), fq_t::zero()};
  out.x.self_set_msb();
  return out;
}

BB_GPU_HD inline jacobian_g1_t jacobian_infinity() {
  return {fq_t::zero(), fq_t::zero(), fq_t::zero(), true};
}

BB_GPU_HD inline xyzz_g1_t xyzz_infinity() {
  return {fq_t::zero(), fq_t::zero(), fq_t::zero(), fq_t::zero(), true};
}

BB_GPU_HD inline affine_g1_t affine_neg(const affine_g1_t &point) {
  if (is_infinity(point)) {
    return point;
  }
  return {point.x, -point.y};
}

BB_GPU_HD inline jacobian_g1_t jacobian_neg(const jacobian_g1_t &point) {
  if (point.infinity) {
    return point;
  }
  return {point.x, -point.y, point.z, false};
}

BB_GPU_HD inline jacobian_g1_t to_jacobian(const affine_g1_t &point) {
  if (is_infinity(point)) {
    return jacobian_infinity();
  }
  return {point.x, point.y, fq_t::one(), false};
}

BB_GPU_HD inline xyzz_g1_t to_xyzz(const affine_g1_t &point) {
  if (is_infinity(point)) {
    return xyzz_infinity();
  }
  return {point.x, point.y, fq_t::one(), fq_t::one(), false};
}

BB_GPU_HD inline affine_g1_t to_affine(const xyzz_g1_t &point) {
  if (point.infinity) {
    return affine_infinity();
  }
  const fq_t zzz_inv = point.zzz.inv();
  const fq_t zz_inv = (point.zz * zzz_inv).sqr();
  return {point.x * zz_inv, point.y * zzz_inv};
}

BB_GPU_HD inline void self_double(jacobian_g1_t &point) {
  if (point.infinity) {
    return;
  }

  fq_t t0 = point.x.sqr();
  fq_t t1 = point.y.sqr();
  fq_t t2 = t1.sqr();
  t1 = t1 + point.x;
  t1 = t1.sqr();
  fq_t t3 = t0 + t2;
  t1 = t1 - t3;
  t1 = t1 + t1;
  t3 = t0 + t0;
  t3 = t3 + t0;
  point.z = point.z + point.z;
  point.z = point.z * point.y;
  t0 = t1 + t1;
  point.x = t3.sqr();
  point.x = point.x - t0;
  t2 = t2 + t2;
  t2 = t2 + t2;
  t2 = t2 + t2;
  point.y = t1 - point.x;
  point.y = point.y * t3;
  point.y = point.y - t2;
}

BB_GPU_HD inline jacobian_g1_t jacobian_double(jacobian_g1_t point) {
  self_double(point);
  return point;
}

BB_GPU_HD inline void self_double(xyzz_g1_t &point) {
  if (point.infinity) {
    return;
  }

  fq_t u = point.y + point.y;
  if (u.is_zero()) {
    point = xyzz_infinity();
    return;
  }

  const fq_t v = u.sqr();
  const fq_t w = u * v;
  const fq_t s = point.x * v;
  fq_t m = point.x.sqr();
  m = m + m + m;
  const fq_t two_s = s + s;
  const fq_t x3 = m.sqr() - two_s;
  point.y = (m * (s - x3)) - (w * point.y);
  point.x = x3;
  point.zz = v * point.zz;
  point.zzz = w * point.zzz;
}

BB_GPU_HD inline xyzz_g1_t xyzz_double(xyzz_g1_t point) {
  self_double(point);
  return point;
}

BB_GPU_HD inline void mixed_add(jacobian_g1_t &lhs, const affine_g1_t &rhs) {
  if (is_infinity(rhs)) {
    return;
  }
  if (lhs.infinity) {
    lhs = to_jacobian(rhs);
    return;
  }

  fq_t t0 = lhs.z.sqr();
  fq_t t1 = rhs.x * t0;
  t1 = t1 - lhs.x;
  fq_t t2 = lhs.z * t0;
  t2 = t2 * rhs.y;
  t2 = t2 - lhs.y;

  if (t1.is_zero()) {
    if (t2.is_zero()) {
      self_double(lhs);
    } else {
      lhs = jacobian_infinity();
    }
    return;
  }

  t2 = t2 + t2;
  lhs.z = lhs.z + t1;
  fq_t t3 = t1.sqr();
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

BB_GPU_HD inline void mixed_add_z1_equals_one(jacobian_g1_t &lhs,
                                              const affine_g1_t &rhs) {
  if (is_infinity(rhs)) {
    return;
  }
  if (lhs.infinity) {
    lhs = to_jacobian(rhs);
    return;
  }

  const fq_t h = rhs.x - lhs.x;
  fq_t r = rhs.y - lhs.y;
  if (h.is_zero()) {
    if (r.is_zero()) {
      self_double(lhs);
    } else {
      lhs = jacobian_infinity();
    }
    return;
  }

  const fq_t hh = h.sqr();
  const fq_t two_hh = hh + hh;
  const fq_t i = two_hh + two_hh;
  const fq_t j = h * i;
  r = r + r;
  const fq_t v = lhs.x * i;
  const fq_t two_v = v + v;
  lhs.x = r.sqr() - j - two_v;
  fq_t two_y1_j = lhs.y * j;
  two_y1_j = two_y1_j + two_y1_j;
  lhs.y = (r * (v - lhs.x)) - two_y1_j;
  lhs.z = h + h;
}

BB_GPU_HD inline void xyzz_mixed_add(xyzz_g1_t &lhs, const affine_g1_t &rhs) {
  if (is_infinity(rhs)) {
    return;
  }
  if (lhs.infinity) {
    lhs = to_xyzz(rhs);
    return;
  }

  fq_t p = rhs.x * lhs.zz;
  p = p - lhs.x;
  fq_t r = rhs.y * lhs.zzz;
  r = r - lhs.y;

  if (p.is_zero()) {
    if (r.is_zero()) {
      self_double(lhs);
    } else {
      lhs = xyzz_infinity();
    }
    return;
  }

  fq_t pp = p.sqr();
  fq_t ppp = p * pp;
  lhs.zz = lhs.zz * pp;
  lhs.zzz = lhs.zzz * ppp;
  fq_t q = lhs.x * pp;
  fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

BB_GPU_HD inline void xyzz_mixed_add_zz1_equals_one(xyzz_g1_t &lhs,
                                                    const affine_g1_t &rhs) {
  if (is_infinity(rhs)) {
    return;
  }
  if (lhs.infinity) {
    lhs = to_xyzz(rhs);
    return;
  }

  fq_t p = rhs.x - lhs.x;
  fq_t r = rhs.y - lhs.y;
  if (p.is_zero()) {
    if (r.is_zero()) {
      self_double(lhs);
    } else {
      lhs = xyzz_infinity();
    }
    return;
  }

  fq_t pp = p.sqr();
  fq_t ppp = p * pp;
  lhs.zz = pp;
  lhs.zzz = ppp;
  fq_t q = lhs.x * pp;
  fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

BB_GPU_HD inline void xyzz_mixed_add_assume_finite(xyzz_g1_t &lhs,
                                                   const affine_g1_t &rhs) {
  fq_t p = rhs.x * lhs.zz;
  p = p - lhs.x;
  fq_t r = rhs.y * lhs.zzz;
  r = r - lhs.y;

  if (p.is_zero()) {
    if (r.is_zero()) {
      self_double(lhs);
    } else {
      lhs = xyzz_infinity();
    }
    return;
  }

  fq_t pp = p.sqr();
  fq_t ppp = p * pp;
  lhs.zz = lhs.zz * pp;
  lhs.zzz = lhs.zzz * ppp;
  fq_t q = lhs.x * pp;
  fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

BB_GPU_HD inline void
xyzz_mixed_add_zz1_equals_one_assume_finite(xyzz_g1_t &lhs,
                                            const affine_g1_t &rhs) {
  fq_t p = rhs.x - lhs.x;
  fq_t r = rhs.y - lhs.y;
  if (p.is_zero()) {
    if (r.is_zero()) {
      self_double(lhs);
    } else {
      lhs = xyzz_infinity();
    }
    return;
  }

  fq_t pp = p.sqr();
  fq_t ppp = p * pp;
  lhs.zz = pp;
  lhs.zzz = ppp;
  fq_t q = lhs.x * pp;
  fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

BB_GPU_HD inline void xyzz_mixed_add_unchecked(xyzz_g1_t &lhs,
                                               const affine_g1_t &rhs) {
  fq_t p = rhs.x * lhs.zz;
  p = p - lhs.x;
  fq_t r = rhs.y * lhs.zzz;
  r = r - lhs.y;
  fq_t pp = p.sqr();
  fq_t ppp = p * pp;
  lhs.zz = lhs.zz * pp;
  lhs.zzz = lhs.zzz * ppp;
  fq_t q = lhs.x * pp;
  fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

BB_GPU_HD inline void
xyzz_mixed_add_zz1_equals_one_unchecked(xyzz_g1_t &lhs,
                                        const affine_g1_t &rhs) {
  fq_t p = rhs.x - lhs.x;
  fq_t r = rhs.y - lhs.y;
  fq_t pp = p.sqr();
  fq_t ppp = p * pp;
  lhs.zz = pp;
  lhs.zzz = ppp;
  fq_t q = lhs.x * pp;
  fq_t x3 = r.sqr();
  x3 = x3 - ppp;
  pp = q + q;
  x3 = x3 - pp;
  q = q - x3;
  q = r * q;
  ppp = lhs.y * ppp;
  lhs.y = q - ppp;
  lhs.x = x3;
}

BB_GPU_HD inline jacobian_g1_t chained_mixed_add(const affine_g1_t *points,
                                                 const size_t num_points) {
  size_t offset = 0;
  while (offset < num_points && is_infinity(points[offset])) {
    ++offset;
  }
  if (offset == num_points) {
    return jacobian_infinity();
  }

  jacobian_g1_t accumulator = to_jacobian(points[offset]);
  ++offset;
  while (offset < num_points && is_infinity(points[offset])) {
    ++offset;
  }
  if (offset < num_points) {
    mixed_add_z1_equals_one(accumulator, points[offset]);
    ++offset;
  }
  for (; offset < num_points; ++offset) {
    mixed_add(accumulator, points[offset]);
  }
  return accumulator;
}

BB_GPU_HD inline xyzz_g1_t chained_xyzz_mixed_add(const affine_g1_t *points,
                                                  const size_t num_points) {
  size_t offset = 0;
  while (offset < num_points && is_infinity(points[offset])) {
    ++offset;
  }
  if (offset == num_points) {
    return xyzz_infinity();
  }

  xyzz_g1_t accumulator = to_xyzz(points[offset]);
  ++offset;
  while (offset < num_points && is_infinity(points[offset])) {
    ++offset;
  }
  if (offset < num_points) {
    xyzz_mixed_add_zz1_equals_one(accumulator, points[offset]);
    ++offset;
  }
  for (; offset < num_points; ++offset) {
    xyzz_mixed_add(accumulator, points[offset]);
  }
  return accumulator;
}

BB_GPU_HD inline jacobian_g1_t chained_mixed_add_indexed(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  int offset = first_offset;
  while (offset < count && is_infinity(points[point_indices[start + offset]])) {
    offset += step;
  }
  if (offset >= count) {
    return jacobian_infinity();
  }

  jacobian_g1_t accumulator =
      to_jacobian(points[point_indices[start + offset]]);
  offset += step;
  while (offset < count && is_infinity(points[point_indices[start + offset]])) {
    offset += step;
  }
  if (offset < count) {
    mixed_add_z1_equals_one(accumulator, points[point_indices[start + offset]]);
    offset += step;
  }
  for (; offset < count; offset += step) {
    mixed_add(accumulator, points[point_indices[start + offset]]);
  }
  return accumulator;
}

BB_GPU_HD inline jacobian_g1_t chained_mixed_add_indexed_nonzero(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  if (first_offset >= count) {
    return jacobian_infinity();
  }

  int offset = first_offset;
  jacobian_g1_t accumulator =
      to_jacobian(points[point_indices[start + offset]]);
  offset += step;
  if (offset < count) {
    mixed_add_z1_equals_one(accumulator, points[point_indices[start + offset]]);
    offset += step;
  }
  for (; offset < count; offset += step) {
    mixed_add(accumulator, points[point_indices[start + offset]]);
  }
  return accumulator;
}

BB_GPU_HD inline void chained_xyzz_mixed_add_indexed_nonzero(
    xyzz_g1_t &accumulator, const affine_g1_t *points,
    const uint32_t *point_indices, const int start, const int count,
    const int first_offset = 0, const int step = 1) {
  if (first_offset >= count) {
    accumulator = xyzz_infinity();
    return;
  }

  int offset = first_offset;
  accumulator = to_xyzz(points[point_indices[start + offset]]);
  offset += step;
  if (offset < count) {
    xyzz_mixed_add_zz1_equals_one(accumulator,
                                  points[point_indices[start + offset]]);
    offset += step;
  }
  for (; offset < count; offset += step) {
    xyzz_mixed_add(accumulator, points[point_indices[start + offset]]);
  }
}

BB_GPU_HD inline xyzz_g1_t chained_xyzz_mixed_add_indexed_nonzero(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  xyzz_g1_t accumulator;
  chained_xyzz_mixed_add_indexed_nonzero(accumulator, points, point_indices,
                                         start, count, first_offset, step);
  return accumulator;
}

BB_GPU_HD inline xyzz_g1_t
chained_xyzz_mixed_add_indexed_nonzero_assume_finite(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  if (first_offset >= count) {
    return xyzz_infinity();
  }

  int offset = first_offset;
  xyzz_g1_t accumulator = to_xyzz(points[point_indices[start + offset]]);
  offset += step;
  if (offset < count) {
    xyzz_mixed_add_zz1_equals_one_assume_finite(
        accumulator, points[point_indices[start + offset]]);
    offset += step;
  }
  for (; offset < count; offset += step) {
    xyzz_mixed_add_assume_finite(accumulator,
                                 points[point_indices[start + offset]]);
  }
  return accumulator;
}

BB_GPU_HD inline xyzz_g1_t chained_xyzz_mixed_add_indexed_nonzero_unchecked(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  int offset = first_offset;
  xyzz_g1_t accumulator = to_xyzz(points[point_indices[start + offset]]);
  offset += step;
  if (offset < count) {
    xyzz_mixed_add_zz1_equals_one_unchecked(
        accumulator, points[point_indices[start + offset]]);
    offset += step;
  }
  for (; offset < count; offset += step) {
    xyzz_mixed_add_unchecked(accumulator,
                             points[point_indices[start + offset]]);
  }
  return accumulator;
}

BB_GPU_HD inline jacobian_g1_t jacobian_add(jacobian_g1_t lhs,
                                            const jacobian_g1_t &rhs) {
  if (lhs.infinity) {
    return rhs;
  }
  if (rhs.infinity) {
    return lhs;
  }

  fq_t z1z1 = lhs.z.sqr();
  fq_t z2z2 = rhs.z.sqr();
  fq_t s2 = z1z1 * lhs.z;
  fq_t u2 = z1z1 * rhs.x;
  s2 = s2 * rhs.y;
  fq_t u1 = z2z2 * lhs.x;
  fq_t s1 = z2z2 * rhs.z;
  s1 = s1 * lhs.y;
  fq_t f = s2 - s1;
  fq_t h = u2 - u1;

  if (h.is_zero()) {
    if (f.is_zero()) {
      self_double(lhs);
    } else {
      lhs = jacobian_infinity();
    }
    return lhs;
  }

  f = f + f;
  fq_t i = h + h;
  i = i.sqr();
  fq_t j = h * i;
  u1 = u1 * i;
  u2 = u1 + u1;
  u2 = u2 + j;
  lhs.x = f.sqr();
  lhs.x = lhs.x - u2;
  j = j * s1;
  j = j + j;
  lhs.y = u1 - lhs.x;
  lhs.y = lhs.y * f;
  lhs.y = lhs.y - j;
  lhs.z = lhs.z + rhs.z;
  z1z1 = z1z1 + z2z2;
  lhs.z = lhs.z.sqr();
  lhs.z = lhs.z - z1z1;
  lhs.z = lhs.z * h;
  return lhs;
}

BB_GPU_HD inline xyzz_g1_t xyzz_add(xyzz_g1_t lhs, const xyzz_g1_t &rhs) {
  if (lhs.infinity) {
    return rhs;
  }
  if (rhs.infinity) {
    return lhs;
  }

  const fq_t u1 = lhs.x * rhs.zz;
  const fq_t u2 = rhs.x * lhs.zz;
  const fq_t s1 = lhs.y * rhs.zzz;
  const fq_t s2 = rhs.y * lhs.zzz;
  const fq_t p = u2 - u1;
  const fq_t r = s2 - s1;

  if (p.is_zero()) {
    if (r.is_zero()) {
      self_double(lhs);
    } else {
      lhs = xyzz_infinity();
    }
    return lhs;
  }

  const fq_t pp = p.sqr();
  const fq_t ppp = p * pp;
  const fq_t q = u1 * pp;
  const fq_t two_q = q + q;
  const fq_t x3 = r.sqr() - ppp - two_q;
  lhs.y = (r * (q - x3)) - (s1 * ppp);
  lhs.x = x3;
  lhs.zz = lhs.zz * rhs.zz * pp;
  lhs.zzz = lhs.zzz * rhs.zzz * ppp;
  return lhs;
}

BB_GPU_HD inline affine_g1_t to_affine(const jacobian_g1_t &point) {
  if (point.infinity) {
    return affine_infinity();
  }
  const fq_t z_inv = point.z.inv();
  const fq_t zz_inv = z_inv.sqr();
  const fq_t zzz_inv = zz_inv * z_inv;
  return {point.x * zz_inv, point.y * zzz_inv};
}

BB_GPU_HD inline void batch_normalize(jacobian_g1_t *points,
                                      const size_t num_points) {
  // Minimal correctness implementation for phase 1. MSM can replace this with
  // a parallel prefix/batch inversion strategy once the primitive layer is
  // stable.
  for (size_t i = 0; i < num_points; ++i) {
    if (!points[i].infinity) {
      points[i] = to_jacobian(to_affine(points[i]));
    }
  }
}

BB_GPU_HD inline bool on_curve(const affine_g1_t &point) {
  if (is_infinity(point)) {
    return true;
  }
  const fq_t x2 = point.x.sqr();
  const fq_t x3 = x2 * point.x;
  const fq_t rhs = x3 + fq_t::from_u64(3);
  const fq_t lhs = point.y.sqr();
  return lhs == rhs;
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
