#include "barretenberg/gpu/curves/bn254/g1.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_THREADS = 32;
constexpr int BUCKET_THREADS = 256;
constexpr int BUCKET_WARPS_PER_BLOCK = BUCKET_THREADS / WARP_THREADS;

using affine_t = bb::gpu::bn254::affine_g1_t;
using fq_t = bb::gpu::bn254::fq_t;
using jacobian_t = bb::gpu::bn254::jacobian_g1_t;
using xyzz_t = bb::gpu::bn254::xyzz_g1_t;

enum class bench_case : int {
  JACOBIAN_NORMAL = 0,
  XYZZ_NORMAL = 1,
  JACOBIAN_LARGE = 2,
  XYZZ_LARGE = 3,
  JACOBIAN_UNCHECKED_NORMAL = 4,
  XYZZ_UNCHECKED_NORMAL = 5,
  PROJECTIVE_RCB_NORMAL = 6,
  JACOBIAN_UNCHECKED_LARGE = 7,
  XYZZ_UNCHECKED_LARGE = 8,
  PROJECTIVE_RCB_LARGE = 9,
  XYZZ_ASSUME_FINITE_NORMAL = 10,
  XYZZ_ASSUME_FINITE_LARGE = 11,
};

enum class segmented_bench_case : int {
  ACCUMULATE_KERNEL_ONLY = 0,
  SUBPIPELINE = 1,
  TREE_ACCUMULATE_KERNEL_ONLY = 2,
  TREE_SUBPIPELINE = 3,
};

struct projective_t {
  fq_t x;
  fq_t y;
  fq_t z;
};

void check_cuda(const cudaError_t error) {
  if (error != cudaSuccess) {
    std::abort();
  }
}

uint32_t ceil_div_u32(const size_t value, const uint32_t divisor) {
  return static_cast<uint32_t>((value + divisor - 1) / divisor);
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

__device__ fq_t make_fq(const uint32_t seed) {
  return fq_t::from_u64(static_cast<uint64_t>(seed) + 1);
}

__device__ fq_t fq_b3() {
  return fq_t::raw(0xf60647ce410d7ff7, 0x2f3d6f4dd31bd011, 0x2943337e3940c6d1,
                   0x1d9598e8a7e39857);
}

__device__ __forceinline__ void
jacobian_mixed_add_z1_equals_one_unchecked(jacobian_t &lhs,
                                           const affine_t &rhs) {
  const fq_t h = rhs.x - lhs.x;
  fq_t r = rhs.y - lhs.y;
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

__device__ __forceinline__ void
jacobian_mixed_add_unchecked(jacobian_t &lhs, const affine_t &rhs) {
  fq_t t0 = lhs.z.sqr();
  fq_t t1 = rhs.x * t0;
  t1 = t1 - lhs.x;
  fq_t t2 = lhs.z * t0;
  t2 = t2 * rhs.y;
  t2 = t2 - lhs.y;
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

__device__ __forceinline__ void projective_rcb_mixed_add(projective_t &lhs,
                                                         const affine_t &rhs) {
  const fq_t x1 = lhs.x;
  const fq_t y1 = lhs.y;
  const fq_t z1 = lhs.z;
  const fq_t x2 = rhs.x;
  const fq_t y2 = rhs.y;
  const fq_t b3 = fq_b3();
  const fq_t t00 = x1 * x2;
  const fq_t t01 = y1 * y2;
  const fq_t t03 = x1 + y1;
  const fq_t t04 = x2 + y2;
  const fq_t t05 = t03 * t04;
  const fq_t t06 = t00 + t01;
  const fq_t t07 = t05 - t06;
  const fq_t t08 = y1 + z1;
  const fq_t t09 = y2 + fq_t::one();
  const fq_t t10 = t08 * t09;
  const fq_t t11 = t01 + z1;
  const fq_t t12 = t10 - t11;
  const fq_t t13 = x1 + z1;
  const fq_t t14 = x2 + fq_t::one();
  const fq_t t15 = t13 * t14;
  const fq_t t16 = t00 + z1;
  const fq_t t17 = t15 - t16;
  const fq_t t18 = t00 + t00;
  const fq_t t19 = t18 + t00;
  const fq_t t20 = b3 * z1;
  const fq_t t21 = t01 + t20;
  const fq_t t22 = t01 - t20;
  const fq_t t23 = b3 * t17;
  lhs.x = (t07 * t22) - (t12 * t23);
  lhs.y = (t22 * t21) + (t23 * t19);
  lhs.z = (t21 * t12) + (t19 * t07);
}

__device__ __forceinline__ projective_t
projective_rcb_add(const projective_t &lhs, const projective_t &rhs) {
  const fq_t b3 = fq_b3();
  const fq_t t00 = lhs.x * rhs.x;
  const fq_t t01 = lhs.y * rhs.y;
  const fq_t t02 = lhs.z * rhs.z;
  const fq_t t03 = lhs.x + lhs.y;
  const fq_t t04 = rhs.x + rhs.y;
  const fq_t t05 = t03 * t04;
  const fq_t t06 = t00 + t01;
  const fq_t t07 = t05 - t06;
  const fq_t t08 = lhs.y + lhs.z;
  const fq_t t09 = rhs.y + rhs.z;
  const fq_t t10 = t08 * t09;
  const fq_t t11 = t01 + t02;
  const fq_t t12 = t10 - t11;
  const fq_t t13 = lhs.x + lhs.z;
  const fq_t t14 = rhs.x + rhs.z;
  const fq_t t15 = t13 * t14;
  const fq_t t16 = t00 + t02;
  const fq_t t17 = t15 - t16;
  const fq_t t18 = t00 + t00;
  const fq_t t19 = t18 + t00;
  const fq_t t20 = b3 * t02;
  const fq_t t21 = t01 + t20;
  const fq_t t22 = t01 - t20;
  const fq_t t23 = b3 * t17;
  return {
      (t07 * t22) - (t12 * t23),
      (t22 * t21) + (t23 * t19),
      (t21 * t12) + (t19 * t07),
  };
}

__device__ jacobian_t chained_jacobian_mixed_add_unchecked(
    const affine_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  int offset = first_offset;
  jacobian_t accumulator =
      bb::gpu::bn254::to_jacobian(points[point_indices[start + offset]]);
  offset += step;
  if (offset < count) {
    jacobian_mixed_add_z1_equals_one_unchecked(
        accumulator, points[point_indices[start + offset]]);
    offset += step;
  }
  for (; offset < count; offset += step) {
    jacobian_mixed_add_unchecked(accumulator,
                                 points[point_indices[start + offset]]);
  }
  return accumulator;
}

__device__ xyzz_t chained_xyzz_mixed_add_unchecked(
    const affine_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  return bb::gpu::bn254::chained_xyzz_mixed_add_indexed_nonzero_unchecked(
      points, point_indices, start, count, first_offset, step);
}

__device__ xyzz_t chained_xyzz_mixed_add_assume_finite(
    const affine_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  using bb::gpu::bn254::chained_xyzz_mixed_add_indexed_nonzero_assume_finite;
  return chained_xyzz_mixed_add_indexed_nonzero_assume_finite(
      points, point_indices, start, count, first_offset, step);
}

__device__ projective_t chained_projective_rcb_mixed_add(
    const affine_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  int offset = first_offset;
  const affine_t first = points[point_indices[start + offset]];
  projective_t accumulator{first.x, first.y, fq_t::one()};
  offset += step;
  for (; offset < count; offset += step) {
    projective_rcb_mixed_add(accumulator,
                             points[point_indices[start + offset]]);
  }
  return accumulator;
}

__global__ void setup_accumulation_inputs_kernel(
    affine_t *points, uint32_t *point_indices, int *bucket_run_indices,
    uint32_t *unique_bucket_indices, int *bucket_sizes, int *bucket_offsets,
    const int num_buckets, const int bucket_size) {
  const size_t point_idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t num_points = static_cast<size_t>(num_buckets) * bucket_size;
  if (point_idx < num_points) {
    points[point_idx] = {
        make_fq(static_cast<uint32_t>(point_idx) * 2 + 101),
        make_fq(static_cast<uint32_t>(point_idx) * 2 + 103),
    };
    point_indices[point_idx] = static_cast<uint32_t>(point_idx);
  }

  if (point_idx < static_cast<size_t>(num_buckets)) {
    const int bucket_idx = static_cast<int>(point_idx);
    bucket_run_indices[bucket_idx] = bucket_idx;
    unique_bucket_indices[bucket_idx] = static_cast<uint32_t>(bucket_idx);
    bucket_sizes[bucket_idx] = bucket_size;
    bucket_offsets[bucket_idx] = bucket_idx * bucket_size;
  }
}

__global__ void accumulate_normal_jacobian_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, jacobian_t *buckets,
    const int num_buckets) {
  const int job_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx]] =
      bb::gpu::bn254::chained_mixed_add_indexed_nonzero(points, point_indices,
                                                        start, count);
}

__global__ void accumulate_normal_xyzz_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, xyzz_t *buckets,
    const int num_buckets) {
  const int job_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  xyzz_t accumulator;
  bb::gpu::bn254::chained_xyzz_mixed_add_indexed_nonzero(
      accumulator, points, point_indices, start, count);
  buckets[unique_bucket_indices[run_idx]] = accumulator;
}

__global__ void accumulate_normal_jacobian_unchecked_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, jacobian_t *buckets,
    const int num_buckets) {
  const int job_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx]] =
      chained_jacobian_mixed_add_unchecked(points, point_indices, start, count);
}

__global__ void accumulate_normal_xyzz_unchecked_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, xyzz_t *buckets,
    const int num_buckets) {
  const int job_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx]] =
      chained_xyzz_mixed_add_unchecked(points, point_indices, start, count);
}

__global__ void accumulate_normal_xyzz_assume_finite_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, xyzz_t *buckets,
    const int num_buckets) {
  const int job_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx]] =
      chained_xyzz_mixed_add_assume_finite(points, point_indices, start, count);
}

__global__ void accumulate_normal_projective_rcb_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points,
    projective_t *buckets, const int num_buckets) {
  const int job_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx]] =
      chained_projective_rcb_mixed_add(points, point_indices, start, count);
}

__global__ void accumulate_large_jacobian_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, jacobian_t *buckets,
    const int num_buckets) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>(blockIdx.x * BUCKET_WARPS_PER_BLOCK + warp_idx);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  jacobian_t local = bb::gpu::bn254::chained_mixed_add_indexed_nonzero(
      points, point_indices, start, count, static_cast<int>(lane_idx),
      WARP_THREADS);

  __shared__ jacobian_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] = bb::gpu::bn254::jacobian_add(
          partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_jacobian_unchecked_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, jacobian_t *buckets,
    const int num_buckets) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>(blockIdx.x * BUCKET_WARPS_PER_BLOCK + warp_idx);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  jacobian_t local = chained_jacobian_mixed_add_unchecked(
      points, point_indices, start, count, static_cast<int>(lane_idx),
      WARP_THREADS);

  __shared__ jacobian_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] = bb::gpu::bn254::jacobian_add(
          partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_xyzz_unchecked_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, xyzz_t *buckets,
    const int num_buckets) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>(blockIdx.x * BUCKET_WARPS_PER_BLOCK + warp_idx);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  xyzz_t local = chained_xyzz_mixed_add_unchecked(
      points, point_indices, start, count, static_cast<int>(lane_idx),
      WARP_THREADS);

  __shared__ xyzz_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] = bb::gpu::bn254::xyzz_add(
          partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_xyzz_assume_finite_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, xyzz_t *buckets,
    const int num_buckets) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>(blockIdx.x * BUCKET_WARPS_PER_BLOCK + warp_idx);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  xyzz_t local = chained_xyzz_mixed_add_assume_finite(
      points, point_indices, start, count, static_cast<int>(lane_idx),
      WARP_THREADS);

  __shared__ xyzz_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] = bb::gpu::bn254::xyzz_add(
          partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_projective_rcb_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points,
    projective_t *buckets, const int num_buckets) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>(blockIdx.x * BUCKET_WARPS_PER_BLOCK + warp_idx);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  projective_t local = chained_projective_rcb_mixed_add(
      points, point_indices, start, count, static_cast<int>(lane_idx),
      WARP_THREADS);

  __shared__ projective_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] = projective_rcb_add(
          partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_xyzz_kernel(
    const int *bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *point_indices, const affine_t *points, xyzz_t *buckets,
    const int num_buckets) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>(blockIdx.x * BUCKET_WARPS_PER_BLOCK + warp_idx);
  if (job_idx >= num_buckets) {
    return;
  }

  const int run_idx = bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int start = bucket_offsets[run_idx];
  xyzz_t local = bb::gpu::bn254::chained_xyzz_mixed_add_indexed_nonzero(
      points, point_indices, start, count, static_cast<int>(lane_idx),
      WARP_THREADS);

  __shared__ xyzz_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] = bb::gpu::bn254::xyzz_add(
          partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void setup_segment_jobs_kernel(int *segment_point_offsets,
                                          int *segment_point_counts,
                                          const int num_buckets,
                                          const int bucket_size,
                                          const int segment_size) {
  const int segment_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int segments_per_bucket =
      ((bucket_size + segment_size - 1) / segment_size) + 1;
  const int num_segments = num_buckets * segments_per_bucket;
  if (segment_idx >= num_segments) {
    return;
  }

  const int bucket_idx = segment_idx / segments_per_bucket;
  const int local_segment_idx =
      segment_idx - (bucket_idx * segments_per_bucket);
  const int local_offset = local_segment_idx * segment_size;
  const int remaining = bucket_size - local_offset;
  segment_point_offsets[segment_idx] =
      (bucket_idx * bucket_size) + local_offset;
  segment_point_counts[segment_idx] =
      remaining <= 0 ? 0
                     : (remaining < segment_size ? remaining : segment_size);
}

__global__ void accumulate_large_xyzz_segments_kernel(
    const int *segment_point_offsets, const int *segment_point_counts,
    const uint32_t *point_indices, const affine_t *points,
    xyzz_t *segment_partials, const int num_segments) {
  const int segment_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (segment_idx >= num_segments) {
    return;
  }

  segment_partials[segment_idx] =
      bb::gpu::bn254::chained_xyzz_mixed_add_indexed_nonzero(
          points, point_indices, segment_point_offsets[segment_idx],
          segment_point_counts[segment_idx]);
}

__global__ void reduce_large_xyzz_segment_partials_kernel(
    const xyzz_t *segment_partials, xyzz_t *buckets, const int num_buckets,
    const int segments_per_bucket) {
  const int bucket_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bucket_idx >= num_buckets) {
    return;
  }

  xyzz_t accumulator = bb::gpu::bn254::xyzz_infinity();
  const int segment_offset = bucket_idx * segments_per_bucket;
  for (int i = 0; i < segments_per_bucket; ++i) {
    accumulator = bb::gpu::bn254::xyzz_add(
        accumulator, segment_partials[segment_offset + i]);
  }
  buckets[bucket_idx] = accumulator;
}

__global__ void setup_segment_counts_kernel(int *segment_counts,
                                            const int num_buckets,
                                            const int segments_per_bucket) {
  const int bucket_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bucket_idx < num_buckets) {
    segment_counts[bucket_idx] = segments_per_bucket;
  }
}

__global__ void reduce_large_xyzz_segment_partials_tree_kernel(
    int *segment_counts, xyzz_t *segment_partials, const int num_segments,
    const int segments_per_bucket) {
  const int segment_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (segment_idx >= num_segments) {
    return;
  }

  const int bucket_idx = segment_idx / segments_per_bucket;
  const int count = segment_counts[bucket_idx];
  if (count <= 1) {
    return;
  }

  const int segment_offset = bucket_idx * segments_per_bucket;
  const int local_idx = segment_idx - segment_offset;
  if (local_idx >= count) {
    return;
  }

  const int upper_offset = (count + 1) >> 1;
  if (local_idx < (count >> 1)) {
    segment_partials[segment_idx] = bb::gpu::bn254::xyzz_add(
        segment_partials[segment_idx],
        segment_partials[segment_offset + upper_offset + local_idx]);
  }
}

__global__ void update_segment_counts_kernel(int *segment_counts,
                                             const int num_buckets) {
  const int bucket_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bucket_idx >= num_buckets) {
    return;
  }

  const int count = segment_counts[bucket_idx];
  if (count > 1) {
    segment_counts[bucket_idx] = (count + 1) >> 1;
  }
}

__global__ void scatter_large_xyzz_segment_partials_kernel(
    const int *segment_counts, const xyzz_t *segment_partials, xyzz_t *buckets,
    const int num_buckets, const int segments_per_bucket) {
  const int bucket_idx =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bucket_idx >= num_buckets || segment_counts[bucket_idx] == 0) {
    return;
  }

  buckets[bucket_idx] = segment_partials[bucket_idx * segments_per_bucket];
}

float run_accumulation_benchmark(const bench_case accumulation_case,
                                 const int log_buckets, const int bucket_size) {
  const int num_buckets = 1 << log_buckets;
  const size_t num_points = static_cast<size_t>(num_buckets) * bucket_size;

  affine_t *points = nullptr;
  uint32_t *point_indices = nullptr;
  int *bucket_run_indices = nullptr;
  uint32_t *unique_bucket_indices = nullptr;
  int *bucket_sizes = nullptr;
  int *bucket_offsets = nullptr;
  void *buckets = nullptr;

  check_cuda(cudaMalloc(&points, sizeof(affine_t) * num_points));
  check_cuda(cudaMalloc(&point_indices, sizeof(uint32_t) * num_points));
  check_cuda(cudaMalloc(&bucket_run_indices, sizeof(int) * num_buckets));
  check_cuda(
      cudaMalloc(&unique_bucket_indices, sizeof(uint32_t) * num_buckets));
  check_cuda(cudaMalloc(&bucket_sizes, sizeof(int) * num_buckets));
  check_cuda(cudaMalloc(&bucket_offsets, sizeof(int) * num_buckets));
  size_t bucket_element_size = sizeof(jacobian_t);
  if (accumulation_case == bench_case::XYZZ_NORMAL ||
      accumulation_case == bench_case::XYZZ_LARGE ||
      accumulation_case == bench_case::XYZZ_UNCHECKED_NORMAL ||
      accumulation_case == bench_case::XYZZ_UNCHECKED_LARGE ||
      accumulation_case == bench_case::XYZZ_ASSUME_FINITE_NORMAL ||
      accumulation_case == bench_case::XYZZ_ASSUME_FINITE_LARGE) {
    bucket_element_size = sizeof(xyzz_t);
  } else if (accumulation_case == bench_case::PROJECTIVE_RCB_NORMAL ||
             accumulation_case == bench_case::PROJECTIVE_RCB_LARGE) {
    bucket_element_size = sizeof(projective_t);
  }
  check_cuda(cudaMalloc(&buckets, bucket_element_size *
                                      static_cast<size_t>(num_buckets)));

  const uint32_t setup_blocks = ceil_div_u32(num_points, THREADS_PER_BLOCK);
  setup_accumulation_inputs_kernel<<<setup_blocks, THREADS_PER_BLOCK>>>(
      points, point_indices, bucket_run_indices, unique_bucket_indices,
      bucket_sizes, bucket_offsets, num_buckets, bucket_size);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());

  const uint32_t normal_blocks =
      ceil_div_u32(static_cast<size_t>(num_buckets), THREADS_PER_BLOCK);
  const uint32_t large_blocks =
      ceil_div_u32(static_cast<size_t>(num_buckets), BUCKET_WARPS_PER_BLOCK);

  auto launch = [&]() {
    switch (accumulation_case) {
    case bench_case::JACOBIAN_NORMAL:
      accumulate_normal_jacobian_kernel<<<normal_blocks, THREADS_PER_BLOCK>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points,
          static_cast<jacobian_t *>(buckets), num_buckets);
      break;
    case bench_case::XYZZ_NORMAL:
      accumulate_normal_xyzz_kernel<<<normal_blocks, THREADS_PER_BLOCK>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points, static_cast<xyzz_t *>(buckets),
          num_buckets);
      break;
    case bench_case::JACOBIAN_UNCHECKED_NORMAL:
      accumulate_normal_jacobian_unchecked_kernel<<<normal_blocks,
                                                    THREADS_PER_BLOCK>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points,
          static_cast<jacobian_t *>(buckets), num_buckets);
      break;
    case bench_case::XYZZ_UNCHECKED_NORMAL:
      accumulate_normal_xyzz_unchecked_kernel<<<normal_blocks,
                                                THREADS_PER_BLOCK>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points, static_cast<xyzz_t *>(buckets),
          num_buckets);
      break;
    case bench_case::XYZZ_ASSUME_FINITE_NORMAL:
      accumulate_normal_xyzz_assume_finite_kernel<<<normal_blocks,
                                                    THREADS_PER_BLOCK>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points, static_cast<xyzz_t *>(buckets),
          num_buckets);
      break;
    case bench_case::PROJECTIVE_RCB_NORMAL:
      accumulate_normal_projective_rcb_kernel<<<normal_blocks,
                                                THREADS_PER_BLOCK>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points,
          static_cast<projective_t *>(buckets), num_buckets);
      break;
    case bench_case::JACOBIAN_LARGE:
      accumulate_large_jacobian_kernel<<<large_blocks, BUCKET_THREADS>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points,
          static_cast<jacobian_t *>(buckets), num_buckets);
      break;
    case bench_case::XYZZ_LARGE:
      accumulate_large_xyzz_kernel<<<large_blocks, BUCKET_THREADS>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points, static_cast<xyzz_t *>(buckets),
          num_buckets);
      break;
    case bench_case::JACOBIAN_UNCHECKED_LARGE:
      accumulate_large_jacobian_unchecked_kernel<<<large_blocks,
                                                   BUCKET_THREADS>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points,
          static_cast<jacobian_t *>(buckets), num_buckets);
      break;
    case bench_case::XYZZ_UNCHECKED_LARGE:
      accumulate_large_xyzz_unchecked_kernel<<<large_blocks, BUCKET_THREADS>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points, static_cast<xyzz_t *>(buckets),
          num_buckets);
      break;
    case bench_case::XYZZ_ASSUME_FINITE_LARGE:
      accumulate_large_xyzz_assume_finite_kernel<<<large_blocks,
                                                   BUCKET_THREADS>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points, static_cast<xyzz_t *>(buckets),
          num_buckets);
      break;
    case bench_case::PROJECTIVE_RCB_LARGE:
      accumulate_large_projective_rcb_kernel<<<large_blocks, BUCKET_THREADS>>>(
          bucket_run_indices, unique_bucket_indices, bucket_sizes,
          bucket_offsets, point_indices, points,
          static_cast<projective_t *>(buckets), num_buckets);
      break;
    }
    check_cuda(cudaGetLastError());
  };

  launch();
  check_cuda(cudaDeviceSynchronize());
  const float elapsed_ms = time_cuda_launch(launch);

  check_cuda(cudaFree(buckets));
  check_cuda(cudaFree(bucket_offsets));
  check_cuda(cudaFree(bucket_sizes));
  check_cuda(cudaFree(unique_bucket_indices));
  check_cuda(cudaFree(bucket_run_indices));
  check_cuda(cudaFree(point_indices));
  check_cuda(cudaFree(points));
  return elapsed_ms;
}

float run_segmented_accumulation_benchmark(
    const segmented_bench_case benchmark_case, const int log_buckets,
    const int bucket_size, const int segment_size) {
  const int num_buckets = 1 << log_buckets;
  const int segments_per_bucket =
      ((bucket_size + segment_size - 1) / segment_size) + 1;
  const int num_segments = num_buckets * segments_per_bucket;
  const size_t num_points = static_cast<size_t>(num_buckets) * bucket_size;

  affine_t *points = nullptr;
  uint32_t *point_indices = nullptr;
  int *bucket_run_indices = nullptr;
  uint32_t *unique_bucket_indices = nullptr;
  int *bucket_sizes = nullptr;
  int *bucket_offsets = nullptr;
  int *segment_point_offsets = nullptr;
  int *segment_point_counts = nullptr;
  int *segment_counts = nullptr;
  xyzz_t *segment_partials = nullptr;
  xyzz_t *buckets = nullptr;

  check_cuda(cudaMalloc(&points, sizeof(affine_t) * num_points));
  check_cuda(cudaMalloc(&point_indices, sizeof(uint32_t) * num_points));
  check_cuda(cudaMalloc(&bucket_run_indices, sizeof(int) * num_buckets));
  check_cuda(
      cudaMalloc(&unique_bucket_indices, sizeof(uint32_t) * num_buckets));
  check_cuda(cudaMalloc(&bucket_sizes, sizeof(int) * num_buckets));
  check_cuda(cudaMalloc(&bucket_offsets, sizeof(int) * num_buckets));
  check_cuda(cudaMalloc(&segment_point_offsets, sizeof(int) * num_segments));
  check_cuda(cudaMalloc(&segment_point_counts, sizeof(int) * num_segments));
  check_cuda(cudaMalloc(&segment_counts, sizeof(int) * num_buckets));
  check_cuda(cudaMalloc(&segment_partials, sizeof(xyzz_t) * num_segments));
  check_cuda(cudaMalloc(&buckets, sizeof(xyzz_t) * num_buckets));

  const uint32_t setup_blocks = ceil_div_u32(num_points, THREADS_PER_BLOCK);
  setup_accumulation_inputs_kernel<<<setup_blocks, THREADS_PER_BLOCK>>>(
      points, point_indices, bucket_run_indices, unique_bucket_indices,
      bucket_sizes, bucket_offsets, num_buckets, bucket_size);
  check_cuda(cudaGetLastError());

  const uint32_t segment_blocks =
      ceil_div_u32(static_cast<size_t>(num_segments), THREADS_PER_BLOCK);
  const uint32_t bucket_blocks =
      ceil_div_u32(static_cast<size_t>(num_buckets), THREADS_PER_BLOCK);

  auto setup_segments = [&]() {
    setup_segment_jobs_kernel<<<segment_blocks, THREADS_PER_BLOCK>>>(
        segment_point_offsets, segment_point_counts, num_buckets, bucket_size,
        segment_size);
    check_cuda(cudaGetLastError());
    setup_segment_counts_kernel<<<bucket_blocks, THREADS_PER_BLOCK>>>(
        segment_counts, num_buckets, segments_per_bucket);
    check_cuda(cudaGetLastError());
  };

  auto launch_segments = [&]() {
    accumulate_large_xyzz_segments_kernel<<<segment_blocks,
                                            THREADS_PER_BLOCK>>>(
        segment_point_offsets, segment_point_counts, point_indices, points,
        segment_partials, num_segments);
    check_cuda(cudaGetLastError());
  };

  auto reduce_segments = [&]() {
    reduce_large_xyzz_segment_partials_kernel<<<bucket_blocks,
                                                THREADS_PER_BLOCK>>>(
        segment_partials, buckets, num_buckets, segments_per_bucket);
    check_cuda(cudaGetLastError());
  };

  auto reduce_segments_tree = [&]() {
    for (int active_segments = segments_per_bucket; active_segments > 1;
         active_segments = (active_segments + 1) >> 1) {
      reduce_large_xyzz_segment_partials_tree_kernel<<<segment_blocks,
                                                       THREADS_PER_BLOCK>>>(
          segment_counts, segment_partials, num_segments, segments_per_bucket);
      check_cuda(cudaGetLastError());
      update_segment_counts_kernel<<<bucket_blocks, THREADS_PER_BLOCK>>>(
          segment_counts, num_buckets);
      check_cuda(cudaGetLastError());
    }
    scatter_large_xyzz_segment_partials_kernel<<<bucket_blocks,
                                                 THREADS_PER_BLOCK>>>(
        segment_counts, segment_partials, buckets, num_buckets,
        segments_per_bucket);
    check_cuda(cudaGetLastError());
  };

  auto launch = [&]() {
    switch (benchmark_case) {
    case segmented_bench_case::ACCUMULATE_KERNEL_ONLY:
    case segmented_bench_case::TREE_ACCUMULATE_KERNEL_ONLY:
      launch_segments();
      break;
    case segmented_bench_case::SUBPIPELINE:
      setup_segments();
      launch_segments();
      reduce_segments();
      break;
    case segmented_bench_case::TREE_SUBPIPELINE:
      setup_segments();
      launch_segments();
      reduce_segments_tree();
      break;
    }
  };

  setup_segments();
  check_cuda(cudaDeviceSynchronize());
  launch();
  check_cuda(cudaDeviceSynchronize());
  setup_segments();
  check_cuda(cudaDeviceSynchronize());
  const float elapsed_ms = time_cuda_launch(launch);

  check_cuda(cudaFree(buckets));
  check_cuda(cudaFree(segment_partials));
  check_cuda(cudaFree(segment_counts));
  check_cuda(cudaFree(segment_point_counts));
  check_cuda(cudaFree(segment_point_offsets));
  check_cuda(cudaFree(bucket_offsets));
  check_cuda(cudaFree(bucket_sizes));
  check_cuda(cudaFree(unique_bucket_indices));
  check_cuda(cudaFree(bucket_run_indices));
  check_cuda(cudaFree(point_indices));
  check_cuda(cudaFree(points));
  return elapsed_ms;
}

} // namespace

extern "C" int bb_gpu_msm_accum_bench_cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

extern "C" float bb_gpu_msm_accum_bench_run(const int case_id,
                                            const int log_buckets,
                                            const int bucket_size) {
  return run_accumulation_benchmark(static_cast<bench_case>(case_id),
                                    log_buckets, bucket_size);
}

extern "C" float bb_gpu_msm_segmented_accum_bench_run(const int case_id,
                                                      const int log_buckets,
                                                      const int bucket_size,
                                                      const int segment_size) {
  return run_segmented_accumulation_benchmark(
      static_cast<segmented_bench_case>(case_id), log_buckets, bucket_size,
      segment_size);
}
