#pragma once

#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/numeric/random/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace bb::gpu::benchmark_msm {

using Curve = bb::curve::BN254;
using Fq = Curve::BaseField;
using Fr = Curve::ScalarField;
using Commitment = Curve::AffineElement;

struct Options {
  std::string mode = "all";
  std::string output_path = "/tmp/bb_gpu_msm_external_bench.jsonl";
  int min_log = 10;
  int max_log = 24;
  int log_step = 2;
  int batch_log = 21;
  int batch_size = 32;
  int repeats = 5;
  int c = 0;
  uint64_t seed = 0x8b1f2a77d3c45e91ULL;
  std::vector<uint32_t> precompute_factors = {1, 4, 8};
};

struct CpuInput {
  std::span<const Commitment> points;
  std::span<const Fr> scalars;
};

struct OwnedCpuInput {
  std::vector<Commitment> points;
  std::vector<Fr> scalars;

  CpuInput view() const { return {points, scalars}; }
};

struct TimedRun {
  double setup_ms = 0.0;
  double precompute_ms = 0.0;
  double msm_e2e_ms = 0.0;
  double backend_call_ms = 0.0;
  double device_event_ms = 0.0;
  double gpu_total_ms = 0.0;
  double backend_host_preamble_ms = 0.0;
  double backend_host_cleanup_ms = 0.0;
  double backend_host_total_ms = 0.0;
  uint32_t c = 0;
  uint64_t large_bucket_count = 0;
  uint64_t large_bucket_point_count = 0;
  uint64_t large_bucket_chunk_count = 0;
  uint64_t max_bucket_size = 0;
  uint32_t large_bucket_threshold = 0;
  uint32_t large_bucket_mode = 0;
  std::string result;
};

class HostTimer {
public:
  HostTimer() : start_(Clock::now()) {}

  double elapsed_ms() const {
    const auto elapsed = Clock::now() - start_;
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                   .count()) /
           1'000'000.0;
  }

private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

[[noreturn]] inline void fail(const std::string &message) {
  throw_or_abort(message);
}

inline bool has_arg(const int argc, char **argv, const std::string_view name) {
  return std::any_of(argv + 1, argv + argc,
                     [&](const char *arg) { return arg == name; });
}

inline std::string read_arg(const int argc, char **argv,
                            const std::string_view name,
                            const std::string &fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

inline int read_int_arg(const int argc, char **argv,
                        const std::string_view name, const int fallback) {
  const std::string value = read_arg(argc, argv, name, "");
  if (value.empty()) {
    return fallback;
  }
  return std::stoi(value);
}

inline uint64_t read_u64_arg(const int argc, char **argv,
                             const std::string_view name,
                             const uint64_t fallback) {
  const std::string value = read_arg(argc, argv, name, "");
  if (value.empty()) {
    return fallback;
  }
  return std::stoull(value);
}

inline std::vector<uint32_t> parse_factors(const std::string &value) {
  std::vector<uint32_t> factors;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      factors.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
  }
  if (factors.empty()) {
    fail("--factors must contain at least one integer");
  }
  return factors;
}

inline Options parse_options(const int argc, char **argv) {
  if (has_arg(argc, argv, "--help")) {
    fail("usage: runner [--mode single|batch|all] [--output file] [--min-log "
         "N] [--max-log N] "
         "[--log-step N] [--factors 1,4,8] [--repeats N] [--batch-log N] "
         "[--batch-size N] [--c N] "
         "[--seed N]");
  }

  Options options;
  options.mode = read_arg(argc, argv, "--mode", options.mode);
  options.output_path = read_arg(argc, argv, "--output", options.output_path);
  options.min_log = read_int_arg(argc, argv, "--min-log", options.min_log);
  options.max_log = read_int_arg(argc, argv, "--max-log", options.max_log);
  options.log_step = read_int_arg(argc, argv, "--log-step", options.log_step);
  options.batch_log =
      read_int_arg(argc, argv, "--batch-log", options.batch_log);
  options.batch_size =
      read_int_arg(argc, argv, "--batch-size", options.batch_size);
  options.repeats = read_int_arg(argc, argv, "--repeats", options.repeats);
  options.c = read_int_arg(argc, argv, "--c", options.c);
  options.seed = read_u64_arg(argc, argv, "--seed", options.seed);
  options.precompute_factors =
      parse_factors(read_arg(argc, argv, "--factors", "1,4,8"));

  if (options.mode != "single" && options.mode != "batch" &&
      options.mode != "all") {
    fail("--mode must be one of single, batch, all");
  }
  if (options.min_log < 0 || options.max_log < options.min_log ||
      options.log_step <= 0 || options.repeats <= 0) {
    fail("invalid benchmark range or repeat count");
  }
  if (options.batch_size <= 0) {
    fail("--batch-size must be positive");
  }
  return options;
}

inline uint64_t mix_seed(uint64_t seed, const uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
  return seed;
}

inline uint64_t points_seed(const Options &options, const int log_num_points) {
  return mix_seed(mix_seed(options.seed, static_cast<uint64_t>(log_num_points)),
                  0x706f696e7473ULL);
}

inline uint64_t scalars_seed(const Options &options, const int log_num_points,
                             const int repeat, const int batch_size) {
  uint64_t seed = mix_seed(options.seed, static_cast<uint64_t>(log_num_points));
  seed = mix_seed(seed, static_cast<uint64_t>(repeat));
  seed = mix_seed(seed, static_cast<uint64_t>(batch_size));
  return mix_seed(seed, 0x7363616c617273ULL);
}

inline std::vector<Commitment> make_points(const size_t num_points,
                                           const uint64_t seed) {
  std::vector<Commitment> points;
  points.resize(num_points);

  using Element = Curve::Element;
  constexpr size_t CHUNK_SIZE = 1 << 16;
  std::vector<Element> chunk;
  chunk.resize(std::min(CHUNK_SIZE, num_points));

  const Element generator = Element::one();
  Element current = generator * Fr(seed == 0 ? 1ULL : seed);
  size_t offset = 0;
  while (offset < num_points) {
    const size_t chunk_size = std::min(CHUNK_SIZE, num_points - offset);
    if (chunk.size() != chunk_size) {
      chunk.resize(chunk_size);
    }
    for (size_t i = 0; i < chunk_size; ++i) {
      chunk[i] = current;
      current += generator;
    }
    Element::batch_normalize(chunk.data(), chunk_size);
    for (size_t i = 0; i < chunk_size; ++i) {
      points[offset + i] = static_cast<Commitment>(chunk[i]);
    }
    offset += chunk_size;
  }
  return points;
}

inline uint64_t splitmix64(uint64_t &state) {
  uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

inline std::vector<Fr> make_scalars(const size_t num_scalars,
                                    const uint64_t seed) {
  std::vector<Fr> scalars;
  scalars.reserve(num_scalars);
  uint64_t state = seed;
  constexpr Fr POW_2_256 = Fr(uint256_t(1) << 128).sqr();
  for (size_t i = 0; i < num_scalars; ++i) {
    const Fr lo(uint256_t(splitmix64(state), splitmix64(state),
                          splitmix64(state), splitmix64(state)));
    const Fr hi(uint256_t(splitmix64(state), splitmix64(state),
                          splitmix64(state), splitmix64(state)));
    scalars.emplace_back(lo + (POW_2_256 * hi));
  }
  return scalars;
}

inline OwnedCpuInput make_input(const Options &options,
                                const int log_num_points, const int repeat,
                                const int batch_size) {
  const size_t num_points = size_t{1} << log_num_points;
  return {
      make_points(num_points, points_seed(options, log_num_points)),
      make_scalars(num_points * static_cast<size_t>(batch_size),
                   scalars_seed(options, log_num_points, repeat, batch_size)),
  };
}

inline std::string hex_u64(const uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}

inline std::string hex_field(const Fq &value) {
  const Fq canonical = value.from_montgomery_form().reduce_once();
  return hex_u64(canonical.data[3]) + hex_u64(canonical.data[2]) +
         hex_u64(canonical.data[1]) + hex_u64(canonical.data[0]);
}

inline std::string result_id(const Commitment &point) {
  if (point.is_point_at_infinity()) {
    return "infinity";
  }
  return hex_field(point.x) + ":" + hex_field(point.y);
}

inline std::string join_result_ids(const std::vector<std::string> &results) {
  std::string joined;
  for (size_t i = 0; i < results.size(); ++i) {
    if (i != 0) {
      joined += "|";
    }
    joined += results[i];
  }
  return joined;
}

inline std::string json_escape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char c : value) {
    switch (c) {
    case '"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\n':
      escaped += "\\n";
      break;
    default:
      escaped += c;
      break;
    }
  }
  return escaped;
}

inline void write_record(std::ofstream &out, const std::string &implementation,
                         const std::string &mode, const int log_num_points,
                         const int batch_size, const uint32_t precompute_factor,
                         const int repeat, const uint64_t seed,
                         const TimedRun &run) {
  const size_t num_points = size_t{1} << log_num_points;
  out << "{"
      << "\"implementation\":\"" << json_escape(implementation) << "\","
      << "\"mode\":\"" << mode << "\","
      << "\"log_num_points\":" << log_num_points << ","
      << "\"num_points\":" << num_points << ","
      << "\"batch_size\":" << batch_size << ","
      << "\"precompute_factor\":" << precompute_factor << ","
      << "\"repeat\":" << repeat << ","
      << "\"seed\":" << seed << ","
      << "\"c\":" << run.c << ","
      << "\"setup_ms\":" << run.setup_ms << ","
      << "\"precompute_ms\":" << run.precompute_ms << ","
      << "\"msm_e2e_ms\":" << run.msm_e2e_ms << ","
      << "\"backend_call_ms\":" << run.backend_call_ms << ","
      << "\"device_event_ms\":" << run.device_event_ms << ","
      << "\"gpu_total_ms\":" << run.gpu_total_ms << ","
      << "\"backend_host_preamble_ms\":" << run.backend_host_preamble_ms << ","
      << "\"backend_host_cleanup_ms\":" << run.backend_host_cleanup_ms << ","
      << "\"backend_host_total_ms\":" << run.backend_host_total_ms << ","
      << "\"per_msm_ms\":" << run.gpu_total_ms / static_cast<double>(batch_size)
      << ","
      << "\"large_bucket_count\":" << run.large_bucket_count << ","
      << "\"large_bucket_point_count\":" << run.large_bucket_point_count << ","
      << "\"large_bucket_chunks\":" << run.large_bucket_chunk_count << ","
      << "\"max_bucket_size\":" << run.max_bucket_size << ","
      << "\"large_bucket_threshold\":" << run.large_bucket_threshold << ","
      << "\"large_bucket_mode\":" << run.large_bucket_mode << ","
      << "\"result\":\"" << json_escape(run.result) << "\""
      << "}\n";
}

inline void open_output(std::ofstream &out, const std::string &path) {
  out.open(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    fail("failed to open benchmark output file: " + path);
  }
}

} // namespace bb::gpu::benchmark_msm
