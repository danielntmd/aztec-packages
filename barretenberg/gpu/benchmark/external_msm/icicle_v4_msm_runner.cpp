#include "icicle/curves/params/bn254.h"
#include "icicle/errors.h"
#include "icicle/msm.h"
#include "icicle/runtime.h"

#include "msm_benchmark_common.hpp"

#include <exception>
#include <memory>

namespace {

template <typename IcicleField, typename BbField>
void copy_field(IcicleField &out, const BbField &in) {
  const BbField canonical = in.from_montgomery_form().reduce_once();
  for (size_t i = 0; i < 4; ++i) {
    out.limbs_storage.limbs[2 * i] = static_cast<uint32_t>(canonical.data[i]);
    out.limbs_storage.limbs[2 * i + 1] =
        static_cast<uint32_t>(canonical.data[i] >> 32U);
  }
}

std::vector<bn254::affine_t>
convert_points(std::span<const bb::gpu::benchmark_msm::Commitment> points) {
  std::vector<bn254::affine_t> out(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    copy_field(out[i].x, points[i].x);
    copy_field(out[i].y, points[i].y);
  }
  return out;
}

std::vector<bn254::scalar_t>
convert_scalars(std::span<const bb::gpu::benchmark_msm::Fr> scalars) {
  std::vector<bn254::scalar_t> out(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {
    copy_field(out[i], scalars[i]);
  }
  return out;
}

std::string hex_field(const bn254::point_field_t &value) {
  std::string out;
  for (int i = 7; i >= 0; --i) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8)
           << value.limbs_storage.limbs[i];
    out += stream.str();
  }
  return out;
}

std::string result_id(bn254::projective_t result) {
  if (result.is_zero()) {
    return "infinity";
  }
  const bn254::affine_t affine = result.to_affine();
  return hex_field(affine.x) + ":" + hex_field(affine.y);
}

void check_icicle(const icicle::eIcicleError err, const std::string &context) {
  if (err != icicle::eIcicleError::SUCCESS) {
    bb::gpu::benchmark_msm::fail(context + ": " +
                                 icicle::get_error_string(err));
  }
}

template <typename T> class DeviceAllocation {
public:
  explicit DeviceAllocation(const size_t count) : count_(count) {
    check_icicle(
        icicle_malloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count),
        "icicle_malloc");
  }

  ~DeviceAllocation() {
    if (ptr_ != nullptr) {
      icicle_free(ptr_);
    }
  }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  T *data() { return ptr_; }

private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

struct PreparedBases {
  std::unique_ptr<DeviceAllocation<bn254::affine_t>> device_points;
  double precompute_ms = 0.0;
  double setup_ms = 0.0;
};

icicle::MSMConfig make_config(const uint32_t precompute_factor, const int c) {
  auto config = icicle::default_msm_config();
  config.precompute_factor = static_cast<int>(precompute_factor);
  config.c = c;
  config.are_scalars_montgomery_form = false;
  config.are_points_montgomery_form = false;
  config.are_points_shared_in_batch = true;
  return config;
}

PreparedBases
prepare_bases(std::span<const bb::gpu::benchmark_msm::Commitment> points,
              const uint32_t precompute_factor, icicle::MSMConfig config) {
  bb::gpu::benchmark_msm::HostTimer setup_timer;
  auto converted_points = convert_points(points);
  std::vector<bn254::affine_t> precomputed_points(points.size() *
                                                  precompute_factor);
  PreparedBases prepared;

  if (precompute_factor == 1) {
    precomputed_points = std::move(converted_points);
  } else {
    bb::gpu::benchmark_msm::HostTimer precompute_timer;
    check_icicle(icicle::msm_precompute_bases(
                     converted_points.data(), static_cast<int>(points.size()),
                     config, precomputed_points.data()),
                 "icicle::msm_precompute_bases");
    prepared.precompute_ms = precompute_timer.elapsed_ms();
  }

  prepared.device_points = std::make_unique<DeviceAllocation<bn254::affine_t>>(
      precomputed_points.size());
  check_icicle(icicle_copy(prepared.device_points->data(),
                           precomputed_points.data(),
                           sizeof(bn254::affine_t) * precomputed_points.size()),
               "icicle_copy points");
  prepared.setup_ms = setup_timer.elapsed_ms();
  return prepared;
}

bb::gpu::benchmark_msm::TimedRun
run_msm(std::span<const bb::gpu::benchmark_msm::Fr> scalars,
        const size_t num_points, const uint32_t batch_size,
        const PreparedBases &prepared, icicle::MSMConfig config) {
  auto converted_scalars = convert_scalars(scalars);
  std::vector<bn254::projective_t> results(batch_size);

  config.batch_size = static_cast<int>(batch_size);
  config.are_points_on_device = true;
  config.are_scalars_on_device = false;
  config.are_results_on_device = false;

  bb::gpu::benchmark_msm::HostTimer timer;
  check_icicle(
      icicle::msm(converted_scalars.data(), prepared.device_points->data(),
                  static_cast<int>(num_points), config, results.data()),
      "icicle::msm");

  bb::gpu::benchmark_msm::TimedRun run;
  run.msm_e2e_ms = timer.elapsed_ms();
  run.backend_call_ms = run.msm_e2e_ms;
  run.gpu_total_ms = run.msm_e2e_ms;
  run.setup_ms = prepared.setup_ms;
  run.precompute_ms = prepared.precompute_ms;
  run.c = static_cast<uint32_t>(config.c);
  std::vector<std::string> result_ids;
  result_ids.reserve(results.size());
  for (const auto &result : results) {
    result_ids.push_back(result_id(result));
  }
  run.result = bb::gpu::benchmark_msm::join_result_ids(result_ids);
  return run;
}

void initialize_backend(int argc, char **argv) {
  const std::string backend_path =
      bb::gpu::benchmark_msm::read_arg(argc, argv, "--icicle-backend-dir", "");
  if (!backend_path.empty()) {
    check_icicle(icicle_load_backend(backend_path.c_str(), true),
                 "icicle_load_backend");
  } else {
    check_icicle(icicle_load_backend_from_env_or_default(),
                 "icicle_load_backend_from_env_or_default; set "
                 "ICICLE_BACKEND_INSTALL_DIR or pass "
                 "--icicle-backend-dir");
  }

  check_icicle(icicle_set_device(icicle::Device{"CUDA", 0}),
               "icicle_set_device(CUDA,0)");
  int device_count = 0;
  check_icicle(icicle_get_device_count(device_count),
               "icicle_get_device_count");
  if (device_count <= 0) {
    bb::gpu::benchmark_msm::fail("Icicle CUDA backend reported no devices");
  }
}

void run_single_sweep(const bb::gpu::benchmark_msm::Options &options,
                      std::ofstream &out) {
  for (int log_num_points = options.min_log; log_num_points <= options.max_log;
       log_num_points += options.log_step) {
    const auto points = bb::gpu::benchmark_msm::make_points(
        size_t{1} << log_num_points,
        bb::gpu::benchmark_msm::points_seed(options, log_num_points));
    for (const uint32_t precompute_factor : options.precompute_factors) {
      auto config = make_config(precompute_factor, options.c);
      auto prepared = prepare_bases(points, precompute_factor, config);
      {
        auto scalars = bb::gpu::benchmark_msm::make_scalars(
            points.size(), bb::gpu::benchmark_msm::scalars_seed(
                               options, log_num_points, -1, 1));
        (void)run_msm(scalars, points.size(), 1, prepared, config);
      }
      for (int repeat = 0; repeat < options.repeats; ++repeat) {
        auto scalars = bb::gpu::benchmark_msm::make_scalars(
            points.size(), bb::gpu::benchmark_msm::scalars_seed(
                               options, log_num_points, repeat, 1));
        auto run = run_msm(scalars, points.size(), 1, prepared, config);
        write_record(out, "icicle-v4.0.0", "single", log_num_points, 1,
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
  const auto points = bb::gpu::benchmark_msm::make_points(
      size_t{1} << options.batch_log,
      bb::gpu::benchmark_msm::points_seed(options, options.batch_log));
  for (const uint32_t precompute_factor : options.precompute_factors) {
    auto config = make_config(precompute_factor, options.c);
    auto prepared = prepare_bases(points, precompute_factor, config);
    {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points.size() * static_cast<size_t>(options.batch_size),
          bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log, -1,
                                               options.batch_size));
      (void)run_msm(scalars, points.size(),
                    static_cast<uint32_t>(options.batch_size), prepared,
                    config);
    }
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      auto scalars = bb::gpu::benchmark_msm::make_scalars(
          points.size() * static_cast<size_t>(options.batch_size),
          bb::gpu::benchmark_msm::scalars_seed(options, options.batch_log,
                                               repeat, options.batch_size));
      auto run =
          run_msm(scalars, points.size(),
                  static_cast<uint32_t>(options.batch_size), prepared, config);
      write_record(out, "icicle-v4.0.0", "batch", options.batch_log,
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
    initialize_backend(argc, argv);
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
