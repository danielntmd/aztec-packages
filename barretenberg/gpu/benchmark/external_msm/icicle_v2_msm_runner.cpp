#include "icicle_v2_msm_adapter.hpp"
#include "msm_benchmark_common.hpp"

#include "barretenberg/common/log.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <limits>
#include <memory>
#include <optional>

namespace {

template <typename BbField>
void copy_field(uint32_t limbs[8], const BbField &in) {
  const BbField canonical = in.from_montgomery_form().reduce_once();
  for (size_t i = 0; i < 4; ++i) {
    limbs[2 * i] = static_cast<uint32_t>(canonical.data[i]);
    limbs[2 * i + 1] = static_cast<uint32_t>(canonical.data[i] >> 32U);
  }
}

std::vector<icicle_v2_affine_t>
convert_points(std::span<const bb::gpu::benchmark_msm::Commitment> points) {
  std::vector<icicle_v2_affine_t> out(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].is_point_at_infinity()) {
      out[i].infinity = 1;
      continue;
    }
    copy_field(out[i].x, points[i].x);
    copy_field(out[i].y, points[i].y);
  }
  return out;
}

std::vector<icicle_v2_scalar_t>
convert_scalars(std::span<const bb::gpu::benchmark_msm::Fr> scalars) {
  std::vector<icicle_v2_scalar_t> out(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {
    copy_field(out[i].limbs, scalars[i]);
  }
  return out;
}

std::string hex_field(const uint32_t limbs[8]) {
  std::string out;
  for (int i = 7; i >= 0; --i) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8) << limbs[i];
    out += stream.str();
  }
  return out;
}

std::string result_id(const icicle_v2_affine_t &result) {
  if (result.infinity != 0) {
    return "infinity";
  }
  return hex_field(result.x) + ":" + hex_field(result.y);
}

[[noreturn]] void fail_adapter(const std::string &context) {
  bb::gpu::benchmark_msm::fail(context + ": " + icicle_v2_last_error());
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

bool has_enough_memory(const size_t num_points, const uint32_t batch_size,
                       const uint32_t precompute_factor, const int c,
                       std::string &reason) {
  const uint32_t effective_c = c > 0 ? static_cast<uint32_t>(c) : 16;
  const size_t num_windows = (254 + effective_c - 1) / effective_c;
  const size_t total_scalars = num_points * static_cast<size_t>(batch_size);
  const size_t total_entries = total_scalars * num_windows;

  size_t required = 0;
  const bool valid =
      add_memory_requirement(required, num_points * precompute_factor,
                             sizeof(icicle_v2_affine_t)) &&
      add_memory_requirement(required, total_scalars,
                             sizeof(icicle_v2_scalar_t)) &&
      add_memory_requirement(required, total_entries, 6 * sizeof(uint32_t));
  if (!valid) {
    reason = "estimated allocation exceeds size_t range";
    return false;
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    reason = "cudaMemGetInfo failed";
    return false;
  }
  constexpr size_t HEADROOM_BYTES = 512ULL << 20U;
  const size_t available =
      free_bytes > HEADROOM_BYTES ? free_bytes - HEADROOM_BYTES : 0;
  if (required > available) {
    reason = "estimated allocation requires " + std::to_string(required) +
             " bytes, but only " + std::to_string(available) +
             " bytes are available after headroom";
    return false;
  }
  return true;
}

void log_skip(const std::string &mode, const int log_num_points,
              const uint32_t batch_size, const uint32_t precompute_factor,
              const std::string &reason) {
  info("icicle-v2.8.0 ", mode, " n=2^", log_num_points, " batch=", batch_size,
       " precompute_factor=", precompute_factor, " skipped: ", reason);
}

class PreparedBases {
public:
  PreparedBases(std::span<const bb::gpu::benchmark_msm::Commitment> points,
                const uint32_t precompute_factor, const int c) {
    bb::gpu::benchmark_msm::HostTimer setup_timer;
    const auto converted_points = convert_points(points);
    prepared_ = icicle_v2_prepare_bases(
        converted_points.data(), converted_points.size(), precompute_factor, c,
        nullptr, &precompute_wall_ms, &precompute_device_ms);
    setup_wall_ms = setup_timer.elapsed_ms();
    if (prepared_ == nullptr) {
      fail_adapter("icicle_v2_prepare_bases");
    }
  }

  PreparedBases(const PreparedBases &) = delete;
  PreparedBases &operator=(const PreparedBases &) = delete;

  ~PreparedBases() { icicle_v2_free_prepared_bases(prepared_); }

  icicle_v2_prepared_bases_t *get() const { return prepared_; }

  double setup_wall_ms = 0.0;
  double precompute_wall_ms = 0.0;
  double precompute_device_ms = 0.0;

private:
  icicle_v2_prepared_bases_t *prepared_ = nullptr;
};

bb::gpu::benchmark_msm::TimedRun
run_msm(std::span<const bb::gpu::benchmark_msm::Fr> scalars,
        const size_t num_points, const uint32_t batch_size,
        const PreparedBases &prepared, const uint32_t precompute_factor,
        const int c) {
  bb::gpu::benchmark_msm::HostTimer outer_timer;
  const auto converted_scalars = convert_scalars(scalars);
  std::vector<icicle_v2_affine_t> results(batch_size);

  double backend_wall_ms = 0.0;
  double device_ms = 0.0;
  if (icicle_v2_run_msm(converted_scalars.data(), num_points, batch_size,
                        prepared.get(), precompute_factor, c, results.data(),
                        &backend_wall_ms, &device_ms) != 0) {
    fail_adapter("icicle_v2_run_msm");
  }

  bb::gpu::benchmark_msm::TimedRun run;
  run.outer_wall_ms = outer_timer.elapsed_ms();
  run.backend_wall_ms = backend_wall_ms;
  run.device_ms = device_ms;
  run.setup_wall_ms = prepared.setup_wall_ms;
  run.precompute_wall_ms = prepared.precompute_wall_ms;
  run.precompute_device_ms = prepared.precompute_device_ms;
  run.c = static_cast<uint32_t>(c);

  std::vector<std::string> result_ids;
  result_ids.reserve(results.size());
  for (const auto &result : results) {
    result_ids.push_back(result_id(result));
  }
  run.result = bb::gpu::benchmark_msm::join_result_ids(result_ids);
  return run;
}

void run_single_sweep(const bb::gpu::benchmark_msm::Options &options,
                      std::ofstream &out) {
  for (int log_num_points = options.min_log; log_num_points <= options.max_log;
       log_num_points += options.log_step) {
    const size_t num_points = size_t{1} << log_num_points;
    std::optional<std::vector<bb::gpu::benchmark_msm::Commitment>> points;
    for (const uint32_t precompute_factor : options.precompute_factors) {
      std::string skip_reason;
      if (!has_enough_memory(num_points, 1, precompute_factor, options.c,
                             skip_reason)) {
        log_skip("single", log_num_points, 1, precompute_factor, skip_reason);
        continue;
      }
      if (!points.has_value()) {
        points.emplace(bb::gpu::benchmark_msm::make_points(
            num_points,
            bb::gpu::benchmark_msm::points_seed(options, log_num_points)));
      }
      const PreparedBases prepared(*points, precompute_factor, options.c);
      {
        auto scalars = bb::gpu::benchmark_msm::make_scalars(
            points->size(), bb::gpu::benchmark_msm::scalars_seed(
                                options, log_num_points, -1, 1));
        (void)run_msm(scalars, points->size(), 1, prepared, precompute_factor,
                      options.c);
      }
      for (int repeat = 0; repeat < options.repeats; ++repeat) {
        auto scalars = bb::gpu::benchmark_msm::make_scalars(
            points->size(), bb::gpu::benchmark_msm::scalars_seed(
                                options, log_num_points, repeat, 1));
        auto run = run_msm(scalars, points->size(), 1, prepared,
                           precompute_factor, options.c);
        write_record(out, "icicle-v2.8.0", "single", log_num_points, 1,
                     precompute_factor, repeat,
                     bb::gpu::benchmark_msm::scalars_seed(
                         options, log_num_points, repeat, 1),
                     run);
      }
    }
  }
}

void run_batch_sweep(const bb::gpu::benchmark_msm::Options &options,
                     std::ofstream &out) {
  const size_t num_points = size_t{1} << options.batch_log;
  std::optional<std::vector<bb::gpu::benchmark_msm::Commitment>> points;
  for (const uint32_t precompute_factor : options.precompute_factors) {
    std::string skip_reason;
    if (!has_enough_memory(num_points,
                           static_cast<uint32_t>(options.batch_size),
                           precompute_factor, options.c, skip_reason)) {
      log_skip("batch", options.batch_log, options.batch_size,
               precompute_factor, skip_reason);
      continue;
    }
    if (!points.has_value()) {
      points.emplace(bb::gpu::benchmark_msm::make_points(
          num_points,
          bb::gpu::benchmark_msm::points_seed(options, options.batch_log)));
    }
    const PreparedBases prepared(*points, precompute_factor, options.c);
    {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points->size() * static_cast<size_t>(options.batch_size),
          bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log, -1,
                                               options.batch_size));
      (void)run_msm(scalars, points->size(),
                    static_cast<uint32_t>(options.batch_size), prepared,
                    precompute_factor, options.c);
    }
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points->size() * static_cast<size_t>(options.batch_size),
          bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log,
                                               repeat, options.batch_size));
      auto run = run_msm(scalars, points->size(),
                         static_cast<uint32_t>(options.batch_size), prepared,
                         precompute_factor, options.c);
      write_record(out, "icicle-v2.8.0", "batch", options.batch_log,
                   options.batch_size, precompute_factor, repeat,
                   bb::gpu::benchmark_msm::scalars_seed(
                       options, options.batch_log, repeat, options.batch_size),
                   run);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = bb::gpu::benchmark_msm::parse_options(argc, argv);
    if (options.memory_placement != "host") {
      bb::gpu::benchmark_msm::fail(
          "Icicle v2.8.0 benchmark only supports host memory placement");
    }
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
