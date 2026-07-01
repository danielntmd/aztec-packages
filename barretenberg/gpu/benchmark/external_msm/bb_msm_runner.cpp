#include "barretenberg/common/log.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/backend.hpp"
#include "barretenberg/gpu/curves/bn254/bn254_conversions.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "common/gpu_msm_context.hpp"
#include "msm/internal/msm_heuristics.hpp"
#include "msm/internal/msm_profile.hpp"
#include "msm/internal/msm_raw.hpp"

#include "msm_benchmark_common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace {

bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void check_cuda_benchmark(const cudaError_t err, const std::string &context) {
  if (err != cudaSuccess) {
    bb::gpu::benchmark_msm::fail(context + ": " + cudaGetErrorString(err));
  }
}

template <typename T> class CudaAllocation {
public:
  explicit CudaAllocation(const size_t count) : count_(count) {
    if (count_ != 0) {
      check_cuda_benchmark(
          cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count_),
          "cudaMalloc");
    }
  }

  ~CudaAllocation() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  CudaAllocation(const CudaAllocation &) = delete;
  CudaAllocation &operator=(const CudaAllocation &) = delete;

  T *data() const { return ptr_; }
  size_t size() const { return count_; }

private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

template <typename T>
void copy_to_device(CudaAllocation<T> &dst, const T *src, const size_t count) {
  check_cuda_benchmark(
      cudaMemcpy(dst.data(), src, sizeof(T) * count, cudaMemcpyHostToDevice),
      "cudaMemcpy host to device");
}

template <typename T>
void copy_to_host(T *dst, const CudaAllocation<T> &src, const size_t count) {
  check_cuda_benchmark(
      cudaMemcpy(dst, src.data(), sizeof(T) * count, cudaMemcpyDeviceToHost),
      "cudaMemcpy device to host");
}

size_t resolve_point_start_index(
    std::span<const bb::gpu::benchmark_msm::Commitment> points) {
  return bb::gpu::default_msm_context().get_srs_offset(
      reinterpret_cast<const bb::gpu::bn254::host_affine_g1_montgomery_t *>(
          points.data()),
      points.size());
}

bb::gpu::benchmark_msm::Commitment
cpu_msm(const bb::gpu::benchmark_msm::CpuInput &input,
        const size_t scalar_offset = 0) {
  const auto *scalar_start = input.scalars.data() + scalar_offset;
  const auto scalar_span = bb::PolynomialSpan<const bb::gpu::benchmark_msm::Fr>{
      0, {scalar_start, input.points.size()}};
  return bb::scalar_multiplication::pippenger<bb::gpu::benchmark_msm::Curve>(
      scalar_span, input.points);
}

void add_optional(std::optional<double> &target, const double value) {
  target = target.value_or(0.0) + value;
}

void apply_profile(bb::gpu::benchmark_msm::TimedRun &run,
                   const bb::gpu::bn254::msm_profile &profile) {
  add_optional(run.precompute_device_ms, profile.precompute_bases_ms);
  add_optional(run.device_ms, profile.total_profiled_ms);
  run.backend_host_preamble_ms += profile.backend_host_preamble_ms;
  run.backend_host_cleanup_ms += profile.backend_host_cleanup_ms;
  run.backend_host_total_ms += profile.backend_host_total_ms;
  run.c = profile.bits_per_slice;
  run.large_bucket_count += profile.large_bucket_count;
  run.large_bucket_point_count += profile.large_bucket_point_count;
  run.large_bucket_chunk_count += profile.large_bucket_chunk_count;
  run.max_bucket_size = std::max(run.max_bucket_size, profile.max_bucket_size);
  run.large_bucket_threshold = profile.large_bucket_threshold;
  run.large_bucket_mode = profile.large_bucket_mode;
}

bool add_memory_requirement(size_t &total, const size_t count,
                            const size_t element_size) {
  if (count != 0 && element_size > std::numeric_limits<size_t>::max() / count) {
    return false;
  }
  const size_t bytes = count * element_size;
  if (bytes > std::numeric_limits<size_t>::max() - total) {
    return false;
  }
  total += bytes;
  return true;
}

bool has_enough_transient_memory(const size_t num_points,
                                 const uint32_t batch_size, const uint32_t c,
                                 const uint32_t precompute_factor,
                                 const size_t persistent_bytes,
                                 std::string &reason) {
  constexpr size_t BUCKET_ELEMENT_BYTES = 160;
  constexpr uint32_t CHUNKED_REDUCTION_CHUNK_SIZE = 256;
  constexpr uint32_t WINDOW_KEY_BITS = 8;

  const uint32_t original_windows =
      (bb::gpu::bn254::GPU_MSM_NUM_BITS_IN_FIELD + c - 1) / c;
  const uint32_t effective_precompute_factor =
      bb::gpu::bn254::get_effective_msm_precompute_factor(original_windows,
                                                          precompute_factor);
  const uint32_t active_windows =
      effective_precompute_factor == 1
          ? original_windows
          : static_cast<uint32_t>(bb::gpu::bn254::ceil_div_size_t(
                original_windows, effective_precompute_factor));
  const uint32_t flat_windows = batch_size * active_windows;
  const size_t total_entries = static_cast<size_t>(batch_size) * num_points *
                               static_cast<size_t>(original_windows);
  if (total_entries > static_cast<size_t>(std::numeric_limits<int>::max())) {
    reason = "schedule exceeds CUB int range";
    return false;
  }

  const size_t total_dense_buckets =
      static_cast<size_t>(flat_windows) * (size_t{1} << c);
  const size_t max_encoded_buckets =
      std::min(total_entries, total_dense_buckets);
  const size_t max_reduction_chunks_per_window =
      bb::gpu::bn254::ceil_div_size_t(size_t{1} << (c - 1),
                                      CHUNKED_REDUCTION_CHUNK_SIZE);

  size_t required = 0;
  bool valid =
      add_memory_requirement(required, 1, persistent_bytes) &&
      add_memory_requirement(required,
                             static_cast<size_t>(batch_size) * num_points,
                             sizeof(bb::gpu::bn254::host_fr_montgomery_t)) &&
      add_memory_requirement(required, total_entries, 6 * sizeof(uint32_t)) &&
      add_memory_requirement(required, max_encoded_buckets, sizeof(int)) &&
      add_memory_requirement(required, max_encoded_buckets,
                             2 * sizeof(uint32_t)) &&
      add_memory_requirement(required, max_encoded_buckets, 2 * sizeof(int)) &&
      add_memory_requirement(required, total_dense_buckets,
                             BUCKET_ELEMENT_BYTES) &&
      add_memory_requirement(required, static_cast<size_t>(flat_windows) * c,
                             BUCKET_ELEMENT_BYTES) &&
      add_memory_requirement(required, flat_windows, BUCKET_ELEMENT_BYTES) &&
      add_memory_requirement(required,
                             static_cast<size_t>(flat_windows) *
                                 max_reduction_chunks_per_window,
                             BUCKET_ELEMENT_BYTES) &&
      add_memory_requirement(required, batch_size,
                             sizeof(bb::gpu::bn254::fq32_affine_g1_t)) &&
      add_memory_requirement(required, 1, sizeof(int)) &&
      add_memory_requirement(required, total_entries, 2 * sizeof(uint32_t));
  if (!valid) {
    reason = "estimated transient allocation exceeds size_t range";
    return false;
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    reason = "cudaMemGetInfo failed";
    return false;
  }
  if (required > free_bytes) {
    reason = "estimated transient allocation requires " +
             std::to_string(required) + " bytes, but only " +
             std::to_string(free_bytes) + " bytes are free";
    return false;
  }
  return true;
}

void log_skip(const std::string &implementation, const std::string &mode,
              const int log_num_points, const uint32_t batch_size,
              const uint32_t precompute_factor, const std::string &reason) {
  info(implementation, " ", mode, " n=2^", log_num_points,
       " batch=", batch_size, " precompute_factor=", precompute_factor,
       " skipped: ", reason);
}

bb::gpu::benchmark_msm::TimedRun
run_single(const bb::gpu::benchmark_msm::CpuInput &input,
           const uint32_t precompute_factor, const uint32_t c,
           const double setup_ms) {
  bb::gpu::bn254::fq32_affine_g1_t raw_result{};
  bb::gpu::bn254::msm_profile profile{};
  const bb::gpu::bn254::MsmRawOptions raw_options{
      .bits_per_slice = c,
      .precompute_factor = precompute_factor,
      .precompute_cache_min_length =
          bb::gpu::MsmConfig{}.precompute_cache_min_length,
      .max_fused_batch_size = bb::gpu::bn254::GPU_MSM_MAX_FUSED_BATCH_SIZE,
  };

  bb::gpu::benchmark_msm::HostTimer timer;
  bb::gpu::bn254::msm_raw_profiled_fq32(
      reinterpret_cast<const bb::gpu::bn254::host_fr_montgomery_t *>(
          input.scalars.data()),
      input.points.size(), resolve_point_start_index(input.points), raw_options,
      &raw_result, &profile);

  bb::gpu::benchmark_msm::TimedRun run;
  run.memory_placement = "host";
  run.setup_wall_ms = setup_ms;
  run.outer_wall_ms = timer.elapsed_ms();
  run.backend_wall_ms = run.outer_wall_ms;
  apply_profile(run, profile);
  run.result = bb::gpu::benchmark_msm::result_id(
      bb::gpu::bn254::to_cpu_point(raw_result));
  (void)precompute_factor;
  return run;
}

bb::gpu::benchmark_msm::TimedRun
run_batch(const bb::gpu::benchmark_msm::CpuInput &input,
          const uint32_t batch_size, const uint32_t precompute_factor,
          const uint32_t c, const uint32_t max_fused_batch_size,
          const double setup_ms) {
  const bb::gpu::MsmConfig cfg{
      .bits_per_slice = c,
      .precompute_factor = precompute_factor,
      .precompute_cache_min_length =
          bb::gpu::MsmConfig{}.precompute_cache_min_length,
  };
  std::vector<std::span<const bb::gpu::benchmark_msm::Commitment>> point_spans(
      batch_size, input.points);
  std::vector<std::span<bb::gpu::benchmark_msm::Fr>> scalar_spans;
  scalar_spans.reserve(batch_size);
  auto *scalars =
      const_cast<bb::gpu::benchmark_msm::Fr *>(input.scalars.data());
  for (uint32_t batch = 0; batch < batch_size; ++batch) {
    scalar_spans.emplace_back(scalars + static_cast<size_t>(batch) *
                                            input.points.size(),
                              input.points.size());
  }

  bb::gpu::benchmark_msm::TimedRun run;
  run.memory_placement = "host";
  run.setup_wall_ms = setup_ms;
  bb::gpu::benchmark_msm::HostTimer timer;
  std::vector<bb::curve::BN254::AffineElement> results;
  results.reserve(batch_size);
  for (uint32_t offset = 0; offset < batch_size;
       offset += max_fused_batch_size) {
    const uint32_t chunk_size =
        std::min<uint32_t>(batch_size - offset, max_fused_batch_size);
    auto chunk = bb::gpu::Backend<bb::curve::BN254>::batch_msm(
        std::span(point_spans).subspan(offset, chunk_size),
        std::span(scalar_spans).subspan(offset, chunk_size), cfg);
    results.insert(results.end(), chunk.begin(), chunk.end());
  }

  run.outer_wall_ms = timer.elapsed_ms();
  run.backend_wall_ms = run.outer_wall_ms;
  run.c = c;
  std::vector<std::string> result_ids;
  result_ids.reserve(results.size());
  for (const auto &result : results) {
    result_ids.push_back(bb::gpu::benchmark_msm::result_id(result));
  }
  run.result = bb::gpu::benchmark_msm::join_result_ids(result_ids);
  (void)precompute_factor;
  return run;
}

bb::gpu::benchmark_msm::TimedRun
run_single_device(const bb::gpu::benchmark_msm::CpuInput &input,
                  const uint32_t precompute_factor, const uint32_t c,
                  const double setup_ms) {
  const size_t num_points = input.points.size();
  CudaAllocation<bb::gpu::bn254::host_fr_montgomery_t> device_scalars(
      num_points);
  copy_to_device(device_scalars,
                 reinterpret_cast<const bb::gpu::bn254::host_fr_montgomery_t *>(
                     input.scalars.data()),
                 num_points);
  CudaAllocation<bb::gpu::bn254::fq32_affine_g1_t> device_result(1);

  const bb::gpu::bn254::MsmRawOptions raw_options{
      .bits_per_slice = c,
      .precompute_factor = precompute_factor,
      .precompute_cache_min_length =
          bb::gpu::MsmConfig{}.precompute_cache_min_length,
      .max_fused_batch_size = bb::gpu::bn254::GPU_MSM_MAX_FUSED_BATCH_SIZE,
  };

  bb::gpu::bn254::msm_profile profile{};
  bb::gpu::benchmark_msm::HostTimer timer;
  bb::gpu::bn254::msm_raw_batch_device_profiled_fq32(
      device_scalars.data(), num_points, 1,
      resolve_point_start_index(input.points), raw_options,
      device_result.data(), &profile);

  bb::gpu::benchmark_msm::TimedRun run;
  run.memory_placement = "device";
  run.setup_wall_ms = setup_ms;
  run.outer_wall_ms = timer.elapsed_ms();
  run.backend_wall_ms = run.outer_wall_ms;
  apply_profile(run, profile);

  bb::gpu::bn254::fq32_affine_g1_t raw_result{};
  copy_to_host(&raw_result, device_result, 1);
  run.result = bb::gpu::benchmark_msm::result_id(
      bb::gpu::bn254::to_cpu_point(raw_result));
  return run;
}

bb::gpu::benchmark_msm::TimedRun
run_batch_device(const bb::gpu::benchmark_msm::CpuInput &input,
                 const uint32_t batch_size, const uint32_t precompute_factor,
                 const uint32_t c, const uint32_t max_fused_batch_size,
                 const double setup_ms) {
  const size_t num_points = input.points.size();
  const size_t total_scalars = static_cast<size_t>(batch_size) * num_points;
  CudaAllocation<bb::gpu::bn254::host_fr_montgomery_t> device_scalars(
      total_scalars);
  copy_to_device(device_scalars,
                 reinterpret_cast<const bb::gpu::bn254::host_fr_montgomery_t *>(
                     input.scalars.data()),
                 total_scalars);
  CudaAllocation<bb::gpu::bn254::fq32_affine_g1_t> device_results(batch_size);

  const bb::gpu::bn254::MsmRawOptions raw_options{
      .bits_per_slice = c,
      .precompute_factor = precompute_factor,
      .precompute_cache_min_length =
          bb::gpu::MsmConfig{}.precompute_cache_min_length,
      .max_fused_batch_size = max_fused_batch_size,
  };

  bb::gpu::benchmark_msm::TimedRun run;
  run.memory_placement = "device";
  run.setup_wall_ms = setup_ms;
  bb::gpu::benchmark_msm::HostTimer timer;
  for (uint32_t batch_offset = 0; batch_offset < batch_size;
       batch_offset += max_fused_batch_size) {
    const uint32_t chunk_size =
        std::min<uint32_t>(max_fused_batch_size, batch_size - batch_offset);
    bb::gpu::bn254::msm_profile profile{};
    bb::gpu::bn254::msm_raw_batch_device_profiled_fq32(
        device_scalars.data() + static_cast<size_t>(batch_offset) * num_points,
        num_points, chunk_size, resolve_point_start_index(input.points),
        raw_options, device_results.data() + batch_offset, &profile);
    apply_profile(run, profile);
  }

  run.outer_wall_ms = timer.elapsed_ms();
  run.backend_wall_ms = run.outer_wall_ms;

  std::vector<bb::gpu::bn254::fq32_affine_g1_t> raw_results(batch_size);
  copy_to_host(raw_results.data(), device_results, batch_size);
  std::vector<std::string> result_ids;
  result_ids.reserve(raw_results.size());
  for (const auto &raw_result : raw_results) {
    result_ids.push_back(bb::gpu::benchmark_msm::result_id(
        bb::gpu::bn254::to_cpu_point(raw_result)));
  }
  run.result = bb::gpu::benchmark_msm::join_result_ids(result_ids);
  return run;
}

void verify_single(const bb::gpu::benchmark_msm::CpuInput &input,
                   const bb::gpu::benchmark_msm::TimedRun &run) {
  const auto expected = bb::gpu::benchmark_msm::result_id(cpu_msm(input));
  if (run.result != expected) {
    bb::gpu::benchmark_msm::fail(
        "BB GPU MSM external benchmark correctness check failed");
  }
}

void verify_batch_first_result(const bb::gpu::benchmark_msm::CpuInput &input,
                               const bb::gpu::benchmark_msm::TimedRun &run) {
  const auto expected = bb::gpu::benchmark_msm::result_id(cpu_msm(input, 0));
  if (!run.result.starts_with(expected)) {
    bb::gpu::benchmark_msm::fail(
        "BB GPU batch MSM external benchmark correctness check failed");
  }
}

double upload_srs(std::span<const bb::gpu::benchmark_msm::Commitment> points) {
  bb::gpu::benchmark_msm::HostTimer timer;
  bb::gpu::default_msm_context().ensure_srs_uploaded(
      reinterpret_cast<const bb::gpu::bn254::host_affine_g1_montgomery_t *>(
          points.data()),
      points.size());
  return timer.elapsed_ms();
}

void run_single_sweep(const bb::gpu::benchmark_msm::Options &options,
                      std::ofstream &out) {
  for (int log_num_points = options.min_log; log_num_points <= options.max_log;
       log_num_points += options.log_step) {
    bb::gpu::default_msm_context().reset();
    const size_t num_points = size_t{1} << log_num_points;
    std::optional<std::vector<bb::gpu::benchmark_msm::Commitment>> points;

    for (const uint32_t precompute_factor : options.precompute_factors) {
      bb::gpu::default_msm_context().release_shifted_srs();
      bb::gpu::default_msm_context().release_msm_buffers();
      const uint32_t c = options.c == 0
                             ? bb::gpu::bn254::get_auto_bits_per_slice(
                                   num_points, precompute_factor)
                             : static_cast<uint32_t>(options.c);
      const uint32_t original_windows =
          (bb::gpu::bn254::GPU_MSM_NUM_BITS_IN_FIELD + c - 1) / c;
      const uint32_t effective_precompute_factor =
          bb::gpu::bn254::get_effective_msm_precompute_factor(
              original_windows, precompute_factor);
      size_t persistent_bytes =
          (points.has_value()
               ? 0
               : num_points * sizeof(bb::gpu::bn254::fq32_affine_g1_t)) +
          (effective_precompute_factor > 1
               ? num_points * static_cast<size_t>(effective_precompute_factor) *
                     sizeof(bb::gpu::bn254::fq32_affine_g1_t)
               : 0);
      if (options.memory_placement == "device") {
        persistent_bytes +=
            num_points * sizeof(bb::gpu::bn254::host_fr_montgomery_t) +
            sizeof(bb::gpu::bn254::fq32_affine_g1_t);
      }
      std::string skip_reason;
      if (!has_enough_transient_memory(num_points, 1, c, precompute_factor,
                                       persistent_bytes, skip_reason)) {
        log_skip("bb", "single", log_num_points, 1, precompute_factor,
                 skip_reason);
        continue;
      }
      if (!points.has_value()) {
        points.emplace(bb::gpu::benchmark_msm::make_points(
            num_points,
            bb::gpu::benchmark_msm::points_seed(options, log_num_points)));
      }
      const double setup_ms = upload_srs(*points);
      {
        auto scalars = bb::gpu::benchmark_msm::make_scalars(
            points->size(), bb::gpu::benchmark_msm::scalars_seed(
                                options, log_num_points, -1, 1));
        const bb::gpu::benchmark_msm::CpuInput input{*points, scalars};
        if (options.memory_placement == "device") {
          (void)run_single_device(input, precompute_factor, c, setup_ms);
        } else {
          (void)run_single(input, precompute_factor, c, setup_ms);
        }
      }

      for (int repeat = 0; repeat < options.repeats; ++repeat) {
        auto scalars = bb::gpu::benchmark_msm::make_scalars(
            points->size(), bb::gpu::benchmark_msm::scalars_seed(
                                options, log_num_points, repeat, 1));
        const bb::gpu::benchmark_msm::CpuInput input{*points, scalars};
        auto run =
            options.memory_placement == "device"
                ? run_single_device(input, precompute_factor, c, setup_ms)
                : run_single(input, precompute_factor, c, setup_ms);
        if (log_num_points <= 16) {
          verify_single(input, run);
        }
        write_record(out, "bb", "single", log_num_points, 1, precompute_factor,
                     repeat,
                     bb::gpu::benchmark_msm::scalars_seed(
                         options, log_num_points, repeat, 1),
                     run);
      }
    }
  }
}

void run_batch_sweep(const bb::gpu::benchmark_msm::Options &options,
                     std::ofstream &out) {
  bb::gpu::default_msm_context().reset();
  const size_t num_points = size_t{1} << options.batch_log;
  std::optional<std::vector<bb::gpu::benchmark_msm::Commitment>> points;

  for (const uint32_t precompute_factor : options.precompute_factors) {
    bb::gpu::default_msm_context().release_shifted_srs();
    bb::gpu::default_msm_context().release_msm_buffers();
    const uint32_t c =
        options.c == 0 ? bb::gpu::bn254::get_auto_batched_bits_per_slice(
                             num_points, options.batch_size, precompute_factor)
                       : static_cast<uint32_t>(options.c);
    const uint32_t preflight_batch_size =
        std::min<uint32_t>(options.bb_max_fused_batch_size, options.batch_size);
    const uint32_t original_windows =
        (bb::gpu::bn254::GPU_MSM_NUM_BITS_IN_FIELD + c - 1) / c;
    const uint32_t effective_precompute_factor =
        bb::gpu::bn254::get_effective_msm_precompute_factor(original_windows,
                                                            precompute_factor);
    size_t persistent_bytes =
        (points.has_value()
             ? 0
             : num_points * sizeof(bb::gpu::bn254::fq32_affine_g1_t)) +
        (effective_precompute_factor > 1
             ? num_points * static_cast<size_t>(effective_precompute_factor) *
                   sizeof(bb::gpu::bn254::fq32_affine_g1_t)
             : 0);
    if (options.memory_placement == "device") {
      persistent_bytes += num_points * static_cast<size_t>(options.batch_size) *
                              sizeof(bb::gpu::bn254::host_fr_montgomery_t) +
                          static_cast<size_t>(options.batch_size) *
                              sizeof(bb::gpu::bn254::fq32_affine_g1_t);
    }
    std::string skip_reason;
    if (!has_enough_transient_memory(num_points, preflight_batch_size, c,
                                     precompute_factor, persistent_bytes,
                                     skip_reason)) {
      log_skip("bb", "batch", options.batch_log, options.batch_size,
               precompute_factor, skip_reason);
      continue;
    }
    if (!points.has_value()) {
      points.emplace(bb::gpu::benchmark_msm::make_points(
          num_points,
          bb::gpu::benchmark_msm::points_seed(options, options.batch_log)));
    }
    const double setup_ms = upload_srs(*points);
    {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points->size() * static_cast<size_t>(options.batch_size),
          bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log, -1,
                                               options.batch_size));
      const bb::gpu::benchmark_msm::CpuInput input{*points, scalars};
      if (options.memory_placement == "device") {
        (void)run_batch_device(input, options.batch_size, precompute_factor, c,
                               options.bb_max_fused_batch_size, setup_ms);
      } else {
        (void)run_batch(input, options.batch_size, precompute_factor, c,
                        options.bb_max_fused_batch_size, setup_ms);
      }
    }

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points->size() * static_cast<size_t>(options.batch_size),
          bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log,
                                               repeat, options.batch_size));
      const bb::gpu::benchmark_msm::CpuInput input{*points, scalars};
      auto run =
          options.memory_placement == "device"
              ? run_batch_device(input, options.batch_size, precompute_factor,
                                 c, options.bb_max_fused_batch_size, setup_ms)
              : run_batch(input, options.batch_size, precompute_factor, c,
                          options.bb_max_fused_batch_size, setup_ms);
      if (options.batch_log <= 16) {
        verify_batch_first_result(input, run);
      }
      write_record(out, "bb", "batch", options.batch_log, options.batch_size,
                   precompute_factor, repeat,
                   bb::gpu::benchmark_msm::scalars_seed(
                       options, options.batch_log, repeat, options.batch_size),
                   run);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (!cuda_available()) {
      bb::gpu::benchmark_msm::fail("No CUDA-capable device is available");
    }

    const auto options = bb::gpu::benchmark_msm::parse_options(argc, argv);
    std::ofstream out;
    bb::gpu::benchmark_msm::open_output(out, options.output_path);

    if (options.mode == "single" || options.mode == "all") {
      run_single_sweep(options, out);
    }
    if (options.mode == "batch" || options.mode == "all") {
      run_batch_sweep(options, out);
    }
  } catch (const std::exception &err) {
    bb::gpu::benchmark_msm::fail(err.what());
  }
  return 0;
}
