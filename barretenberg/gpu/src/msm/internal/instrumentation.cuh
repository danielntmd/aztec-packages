// CUDA event helpers and runtime MSM knobs shared by all raw MSM paths.
float elapsed_ms(cudaEvent_t start_event, cudaEvent_t stop_event) {
  float elapsed = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed, start_event, stop_event),
             "cudaEventElapsedTime");
  return elapsed;
}

void destroy_event(cudaEvent_t &event) {
  if (event != nullptr) {
    check_cuda(cudaEventDestroy(event), "cudaEventDestroy");
    event = nullptr;
  }
}

void create_timing_event(cudaEvent_t &event) {
  check_cuda(cudaEventCreate(&event), "cudaEventCreate timing");
}

void create_dependency_event(cudaEvent_t &event) {
  check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
             "cudaEventCreateWithFlags dependency");
}

void record_event(cudaEvent_t event, cudaStream_t stream,
                  const char *operation) {
  check_cuda(cudaEventRecord(event, stream), operation);
}

void wait_event(cudaStream_t stream, cudaEvent_t event, const char *operation) {
  check_cuda(cudaStreamWaitEvent(stream, event, 0), operation);
}

// RAII cudaEvent_t. `enabled=false` makes `record`/`get` no-ops, so the same
// call sites work whether profiling is on or off.
class OptionalTimingEvent {
public:
  explicit OptionalTimingEvent(const bool enabled) {
    if (enabled) {
      create_timing_event(event_);
    }
  }
  OptionalTimingEvent(const OptionalTimingEvent &) = delete;
  OptionalTimingEvent &operator=(const OptionalTimingEvent &) = delete;
  ~OptionalTimingEvent() { destroy_event(event_); }

  void record(const cudaStream_t stream, const char *operation) const {
    if (event_ != nullptr) {
      record_event(event_, stream, operation);
    }
  }

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

uint32_t scalar_split_first_chunk_percent() {
  return DEFAULT_SCALAR_SPLIT_FIRST_CHUNK_PERCENT;
}
// Drives the `msm_stage` enum, the NVTX-range name lookup, and the per-stage
// timing slot. Each entry corresponds to an `<name>_ms` field on `msm_profile`.
#define BB_GPU_MSM_STAGES(STAGE)                                               \
  STAGE(precompute_bases)                                                      \
  STAGE(sort_records)                                                          \
  STAGE(encode_buckets)                                                        \
  STAGE(scan_bucket_offsets)                                                   \
  STAGE(build_bucket_jobs)                                                     \
  STAGE(sort_bucket_jobs)                                                      \
  STAGE(accumulate_normal_buckets)                                             \
  STAGE(accumulate_large_buckets)                                              \
  STAGE(reduce_buckets)                                                        \
  STAGE(compose_windows)                                                       \
  STAGE(final_accumulation)                                                    \
  STAGE(d2h_result)

enum class msm_stage {
#define BB_GPU_DECLARE_MSM_STAGE(NAME) NAME,
  BB_GPU_MSM_STAGES(BB_GPU_DECLARE_MSM_STAGE)
#undef BB_GPU_DECLARE_MSM_STAGE
};

const char *stage_nvtx_name(const msm_stage stage_name) {
  switch (stage_name) {
#define BB_GPU_MSM_STAGE_NVTX(NAME)                                            \
  case msm_stage::NAME:                                                        \
    return "bb.msm." #NAME;
    BB_GPU_MSM_STAGES(BB_GPU_MSM_STAGE_NVTX)
#undef BB_GPU_MSM_STAGE_NVTX
  }
  return "bb.msm.unknown_stage";
}

class NoopMsmRecorder {
public:
  void start(cudaStream_t, uint32_t) {}
  void set_active_buckets(uint32_t) {}
  void set_large_bucket_threshold(uint32_t) {}
  void stop() {}
  bool enabled() const { return false; }
  void add_h2d_scalars_ms(float) {}
  void add_split_scalars_ms(float) {}
  void set_scalar_copy_split_pipeline_ms(float) {}
  void set_precompute_config(uint32_t, uint32_t, uint64_t) {}
  void set_large_bucket_config(bool, uint64_t) {}
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

  void set_active_buckets(const uint32_t active_buckets) {
    if (profile_ != nullptr) {
      profile_->active_buckets = active_buckets;
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

  void set_precompute_config(const uint32_t factor,
                             const uint32_t folded_windows,
                             const uint64_t precomputed_srs_bytes) {
    if (profile_ != nullptr) {
      profile_->precompute_factor = factor;
      profile_->folded_windows = folded_windows;
      profile_->precomputed_srs_bytes = precomputed_srs_bytes;
    }
  }

  void set_large_bucket_config(const bool has_large_buckets,
                               const uint64_t chunk_count) {
    if (profile_ != nullptr) {
      profile_->has_large_buckets = has_large_buckets;
      profile_->large_bucket_chunk_count = chunk_count;
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
#define BB_GPU_MSM_STAGE_SLOT(NAME)                                            \
  case msm_stage::NAME:                                                        \
    return &profile_->NAME##_ms;
      BB_GPU_MSM_STAGES(BB_GPU_MSM_STAGE_SLOT)
#undef BB_GPU_MSM_STAGE_SLOT
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
