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

uint32_t scalar_split_first_chunk_percent() {
  return DEFAULT_SCALAR_SPLIT_FIRST_CHUNK_PERCENT;
}

uint32_t &msm_precompute_factor_ref() {
  static uint32_t factor = 4;
  return factor;
}

uint32_t current_msm_precompute_factor() { return msm_precompute_factor_ref(); }
