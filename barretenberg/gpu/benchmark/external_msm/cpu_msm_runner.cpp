#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "msm_benchmark_common.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

using bb::gpu::benchmark_msm::Commitment;
using bb::gpu::benchmark_msm::Fr;
using bb::gpu::benchmark_msm::Options;
using bb::gpu::benchmark_msm::TimedRun;

TimedRun run_cpu_single(std::span<const Commitment> points,
                        std::span<const Fr> scalars, const Options &options) {
  const auto scalar_span = bb::PolynomialSpan<const Fr>{0, scalars};

  bb::gpu::benchmark_msm::HostTimer timer;
  const auto result =
      bb::scalar_multiplication::pippenger_unsafe<
          bb::gpu::benchmark_msm::Curve>(scalar_span, points);
  const double msm_wall_ms = timer.elapsed_ms();

  TimedRun run;
  run.outer_wall_ms = msm_wall_ms;
  run.backend_wall_ms = msm_wall_ms;
  run.backend_host_total_ms = msm_wall_ms;
  run.memory_placement = "host";
  run.c = static_cast<uint32_t>(options.c);
  run.cpu_threads = static_cast<uint32_t>(bb::get_num_cpus());
  run.result = bb::gpu::benchmark_msm::result_id(result);
  return run;
}

TimedRun run_cpu_batch(std::span<const Commitment> points,
                       std::span<Fr> scalars, const uint32_t batch_size,
                       const Options &options) {
  std::vector<std::span<const Commitment>> point_spans;
  std::vector<std::span<Fr>> scalar_spans;
  point_spans.reserve(batch_size);
  scalar_spans.reserve(batch_size);

  for (uint32_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    point_spans.emplace_back(points);
    scalar_spans.emplace_back(
        scalars.subspan(static_cast<size_t>(batch_index) * points.size(),
                        points.size()));
  }

  bb::gpu::benchmark_msm::HostTimer timer;
  const auto results =
      bb::scalar_multiplication::MSM<bb::gpu::benchmark_msm::Curve>::
          batch_multi_scalar_mul(point_spans, scalar_spans, false);
  const double msm_wall_ms = timer.elapsed_ms();

  std::vector<std::string> result_ids;
  result_ids.reserve(results.size());
  for (const auto &result : results) {
    result_ids.push_back(bb::gpu::benchmark_msm::result_id(result));
  }

  TimedRun run;
  run.outer_wall_ms = msm_wall_ms;
  run.backend_wall_ms = msm_wall_ms;
  run.backend_host_total_ms = msm_wall_ms;
  run.memory_placement = "host";
  run.c = static_cast<uint32_t>(options.c);
  run.cpu_threads = static_cast<uint32_t>(bb::get_num_cpus());
  run.result = bb::gpu::benchmark_msm::join_result_ids(result_ids);
  return run;
}

void write_for_factors(std::ofstream &out, const Options &options,
                       const std::string &mode, const int log_num_points,
                       const int batch_size, const int repeat,
                       const TimedRun &run) {
  for (const uint32_t precompute_factor : options.precompute_factors) {
    bb::gpu::benchmark_msm::write_record(
        out, "cpu", mode, log_num_points, batch_size, precompute_factor, repeat,
        bb::gpu::benchmark_msm::scalars_seed(options, log_num_points, repeat,
                                             batch_size),
        run);
  }
}

void run_single_sweep(const Options &options, std::ofstream &out) {
  for (int log_num_points = options.min_log; log_num_points <= options.max_log;
       log_num_points += options.log_step) {
    const auto points = bb::gpu::benchmark_msm::make_points(
        size_t{1} << log_num_points,
        bb::gpu::benchmark_msm::points_seed(options, log_num_points));

    {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points.size(), bb::gpu::benchmark_msm::scalars_seed(
                             options, log_num_points, -1, 1));
      (void)run_cpu_single(points, scalars, options);
    }

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points.size(), bb::gpu::benchmark_msm::scalars_seed(
                             options, log_num_points, repeat, 1));
      write_for_factors(out, options, "single", log_num_points, 1, repeat,
                        run_cpu_single(points, scalars, options));
    }
  }
}

void run_batch_sweep(const Options &options, std::ofstream &out) {
  const auto points = bb::gpu::benchmark_msm::make_points(
      size_t{1} << options.batch_log,
      bb::gpu::benchmark_msm::points_seed(options, options.batch_log));

  {
    auto scalars = bb::gpu::benchmark_msm::make_scalars(
        points.size() * static_cast<size_t>(options.batch_size),
        bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log, -1,
                                             options.batch_size));
    (void)run_cpu_batch(points, scalars,
                        static_cast<uint32_t>(options.batch_size), options);
  }

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    auto scalars = bb::gpu::benchmark_msm::make_scalars(
        points.size() * static_cast<size_t>(options.batch_size),
        bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log, repeat,
                                             options.batch_size));
    write_for_factors(
        out, options, "batch", options.batch_log, options.batch_size, repeat,
        run_cpu_batch(points, scalars, static_cast<uint32_t>(options.batch_size),
                      options));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = bb::gpu::benchmark_msm::parse_options(argc, argv);
    if (options.memory_placement != "host") {
      throw std::runtime_error(
          "cpu runner only supports --memory-placement host");
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
}
