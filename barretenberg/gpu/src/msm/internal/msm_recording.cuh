// Recorder adapters let the same MSM pipeline run with or without profiling.
enum class msm_stage {
  h2d_points,
  h2d_scalars,
  split_scalars,
  precompute_bases,
  sort_records,
  encode_buckets,
  scan_bucket_offsets,
  build_bucket_jobs,
  sort_bucket_jobs,
  bucket_distribution,
  init_buckets,
  accumulate_normal_buckets,
  accumulate_large_buckets,
  reduce_buckets,
  compose_windows,
  final_accumulation,
  d2h_result,
};

const char *stage_nvtx_name(const msm_stage stage_name) {
  switch (stage_name) {
  case msm_stage::h2d_points:
    return "bb.msm.h2d_points";
  case msm_stage::h2d_scalars:
    return "bb.msm.h2d_scalars";
  case msm_stage::split_scalars:
    return "bb.msm.split_scalars";
  case msm_stage::precompute_bases:
    return "bb.msm.precompute_bases";
  case msm_stage::sort_records:
    return "bb.msm.sort_records";
  case msm_stage::encode_buckets:
    return "bb.msm.encode_buckets";
  case msm_stage::scan_bucket_offsets:
    return "bb.msm.scan_bucket_offsets";
  case msm_stage::build_bucket_jobs:
    return "bb.msm.build_bucket_jobs";
  case msm_stage::sort_bucket_jobs:
    return "bb.msm.sort_bucket_jobs";
  case msm_stage::bucket_distribution:
    return "bb.msm.bucket_distribution";
  case msm_stage::init_buckets:
    return "bb.msm.init_buckets";
  case msm_stage::accumulate_normal_buckets:
    return "bb.msm.accumulate_normal_buckets";
  case msm_stage::accumulate_large_buckets:
    return "bb.msm.accumulate_large_buckets";
  case msm_stage::reduce_buckets:
    return "bb.msm.reduce_buckets";
  case msm_stage::compose_windows:
    return "bb.msm.compose_windows";
  case msm_stage::final_accumulation:
    return "bb.msm.final_accumulation";
  case msm_stage::d2h_result:
    return "bb.msm.d2h_result";
  }
  return "bb.msm.unknown_stage";
}

class NoopMsmRecorder {
public:
  void start(cudaStream_t, uint32_t) {}
  void set_total_entries(uint32_t) {}
  void set_encoded_buckets(uint32_t, uint32_t, uint32_t) {}
  void set_large_bucket_threshold(uint32_t) {}
  void stop() {}
  bool enabled() const { return false; }
  void add_h2d_scalars_ms(float) {}
  void add_split_scalars_ms(float) {}
  void set_scalar_copy_split_pipeline_ms(float) {}
  void set_scalar_copy_split_overlap_ms(float) {}
  void set_scalar_chunk_profile(size_t, float, float) {}
  void set_precompute_config(uint32_t, uint32_t, uint64_t) {}
  void set_large_bucket_config(uint32_t, uint32_t, uint64_t) {}
  void set_bucket_distribution(const uint64_t *) {}
  const char *scalar_copy_split_range_name() const {
    return "bb.msm.scalar_copy_split_pipeline";
  }

  template <typename Stage> void time(msm_stage, Stage &&stage) {
    std::forward<Stage>(stage)();
  }
};

class ProfileMsmRecorder {
public:
  explicit ProfileMsmRecorder(msm_profile *profile) : profile_(profile) {
    if (profile_ != nullptr) {
      *profile_ = {};
    }
  }

  ProfileMsmRecorder(const ProfileMsmRecorder &) = delete;
  ProfileMsmRecorder &operator=(const ProfileMsmRecorder &) = delete;

  ~ProfileMsmRecorder() {
    if (profile_ != nullptr && !stopped_) {
      stop();
    }
  }

  void start(cudaStream_t stream, const uint32_t bits_per_slice) {
    stream_ = stream;
    if (profile_ == nullptr) {
      return;
    }
    profile_->bits_per_slice = bits_per_slice;
    total_range_.emplace("bb.msm");
    check_cuda(cudaEventCreate(&total_start_), "cudaEventCreate total start");
    check_cuda(cudaEventCreate(&total_stop_), "cudaEventCreate total stop");
    check_cuda(cudaEventRecord(total_start_, stream_),
               "cudaEventRecord total start");
  }

  void set_total_entries(const uint32_t total_entries) {
    if (profile_ != nullptr) {
      profile_->total_entries = total_entries;
    }
  }

  void set_encoded_buckets(const uint32_t encoded_buckets,
                           const uint32_t active_buckets,
                           const uint32_t zero_bucket_offset) {
    if (profile_ != nullptr) {
      profile_->encoded_buckets = encoded_buckets;
      profile_->active_buckets = active_buckets;
      profile_->zero_bucket_offset = zero_bucket_offset;
    }
  }

  void set_large_bucket_threshold(const uint32_t large_bucket_threshold) {
    if (profile_ != nullptr) {
      profile_->large_bucket_threshold = large_bucket_threshold;
    }
  }

  bool enabled() const { return profile_ != nullptr; }

  void add_h2d_scalars_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->h2d_scalars_ms += elapsed_ms;
    }
  }

  void add_split_scalars_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->split_scalars_ms += elapsed_ms;
    }
  }

  void set_scalar_copy_split_pipeline_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->scalar_copy_split_pipeline_ms = elapsed_ms;
    }
  }

  void set_scalar_copy_split_overlap_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->scalar_copy_split_overlap_ms = elapsed_ms;
    }
  }

  void set_scalar_chunk_profile(const size_t chunk_index, const float copy_ms,
                                const float split_ms) {
    if (profile_ == nullptr) {
      return;
    }
    if (chunk_index == 0) {
      profile_->scalar_chunk0_copy_ms = copy_ms;
      profile_->scalar_chunk0_split_ms = split_ms;
    } else {
      profile_->scalar_chunk1_copy_ms = copy_ms;
      profile_->scalar_chunk1_split_ms = split_ms;
    }
  }

  void set_precompute_config(const uint32_t factor,
                             const uint32_t folded_windows,
                             const uint64_t precomputed_srs_bytes) {
    if (profile_ != nullptr) {
      profile_->precompute_factor = factor;
      profile_->folded_windows = folded_windows;
      profile_->precomputed_srs_bytes = precomputed_srs_bytes;
    }
  }

  void set_large_bucket_config(const uint32_t mode, const uint32_t chunk_size,
                               const uint64_t chunk_count) {
    if (profile_ != nullptr) {
      profile_->large_bucket_mode = mode;
      profile_->large_bucket_chunk_size = chunk_size;
      profile_->large_bucket_chunk_count = chunk_count;
    }
  }

  void set_bucket_distribution(const uint64_t *stats) {
    if (profile_ == nullptr) {
      return;
    }
    profile_->normal_bucket_count = stats[BUCKET_STAT_NORMAL_JOBS];
    profile_->large_bucket_count = stats[BUCKET_STAT_LARGE_JOBS];
    profile_->normal_bucket_point_count = stats[BUCKET_STAT_NORMAL_POINTS];
    profile_->large_bucket_point_count = stats[BUCKET_STAT_LARGE_POINTS];
    profile_->max_bucket_size = stats[BUCKET_STAT_MAX_SIZE];
    profile_->large_bucket_chunk_count = stats[BUCKET_STAT_LARGE_CHUNKS];
    for (size_t i = 0; i < MSM_BUCKET_HISTOGRAM_BINS; ++i) {
      profile_->bucket_size_histogram[i] =
          stats[BUCKET_STAT_HISTOGRAM_START + i];
    }
  }

  const char *scalar_copy_split_range_name() const {
    return "bb.msm.scalar_copy_split_pipeline";
  }

  template <typename Stage>
  void time(const msm_stage stage_name, Stage &&stage) {
    float *elapsed_ms = stage_slot(stage_name);
    if (elapsed_ms == nullptr) {
      std::forward<Stage>(stage)();
      return;
    }

    bb::gpu::ScopedNvtxRange nvtx_range(stage_nvtx_name(stage_name));
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start_event, stream_), "cudaEventRecord start");
    std::forward<Stage>(stage)();
    check_cuda(cudaEventRecord(stop_event, stream_), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
    float stage_elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&stage_elapsed_ms, start_event, stop_event),
               "cudaEventElapsedTime");
    *elapsed_ms += stage_elapsed_ms;
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
  }

  void stop() {
    if (profile_ == nullptr || stopped_) {
      return;
    }
    check_cuda(cudaEventRecord(total_stop_, stream_),
               "cudaEventRecord total stop");
    check_cuda(cudaEventSynchronize(total_stop_),
               "cudaEventSynchronize total stop");
    check_cuda(cudaEventElapsedTime(&profile_->total_profiled_ms, total_start_,
                                    total_stop_),
               "cudaEventElapsedTime total");
    check_cuda(cudaEventDestroy(total_stop_), "cudaEventDestroy total stop");
    check_cuda(cudaEventDestroy(total_start_), "cudaEventDestroy total start");
    total_range_.reset();
    stopped_ = true;
  }

private:
  float *stage_slot(const msm_stage stage_name) {
    if (profile_ == nullptr) {
      return nullptr;
    }
    switch (stage_name) {
    case msm_stage::h2d_points:
      return &profile_->h2d_points_ms;
    case msm_stage::h2d_scalars:
      return &profile_->h2d_scalars_ms;
    case msm_stage::split_scalars:
      return &profile_->split_scalars_ms;
    case msm_stage::precompute_bases:
      return &profile_->precompute_bases_ms;
    case msm_stage::sort_records:
      return &profile_->sort_records_ms;
    case msm_stage::encode_buckets:
      return &profile_->encode_buckets_ms;
    case msm_stage::scan_bucket_offsets:
      return &profile_->scan_bucket_offsets_ms;
    case msm_stage::build_bucket_jobs:
      return &profile_->build_bucket_jobs_ms;
    case msm_stage::sort_bucket_jobs:
      return &profile_->sort_bucket_jobs_ms;
    case msm_stage::bucket_distribution:
      return &profile_->bucket_distribution_ms;
    case msm_stage::init_buckets:
      return &profile_->init_buckets_ms;
    case msm_stage::accumulate_normal_buckets:
      return &profile_->accumulate_normal_buckets_ms;
    case msm_stage::accumulate_large_buckets:
      return &profile_->accumulate_large_buckets_ms;
    case msm_stage::reduce_buckets:
      return &profile_->reduce_buckets_ms;
    case msm_stage::compose_windows:
      return &profile_->compose_windows_ms;
    case msm_stage::final_accumulation:
      return &profile_->final_accumulation_ms;
    case msm_stage::d2h_result:
      return &profile_->d2h_result_ms;
    }
    return nullptr;
  }

  msm_profile *profile_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t total_start_ = nullptr;
  cudaEvent_t total_stop_ = nullptr;
  std::optional<bb::gpu::ScopedNvtxRange> total_range_;
  bool stopped_ = false;
};
