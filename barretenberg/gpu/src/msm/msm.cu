#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm_raw.cuh"

#include "barretenberg/gpu/common/cub_helpers.cuh"
#include "barretenberg/gpu/common/cuda_error.cuh"
#include "barretenberg/gpu/common/device_buffer.hpp"
#include "barretenberg/gpu/common/device_context.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace bb::gpu::bn254 {
namespace {

constexpr uint32_t NUM_BITS_IN_FIELD = 254;
constexpr uint32_t SPLIT_THREADS = 1024;
constexpr uint32_t BUCKET_THREADS = 256;
constexpr uint32_t REDUCTION_THREADS = 256;
constexpr int LARGE_BUCKET_MIN_THRESHOLD = 512;
constexpr uint32_t WINDOW_KEY_BITS = 8;
constexpr bool USE_SERIAL_RUNNING_SUM_REDUCTION_FALLBACK = false;

enum class msm_stage {
    h2d_points,
    h2d_scalars,
    split_scalars,
    sort_records,
    encode_buckets,
    scan_bucket_offsets,
    build_bucket_jobs,
    sort_bucket_jobs,
    init_buckets,
    accumulate_normal_buckets,
    accumulate_large_buckets,
    reduce_buckets,
    compose_windows,
    final_accumulation,
    d2h_result,
};

class NoopMsmRecorder {
  public:
    void start(cudaStream_t, uint32_t) {}
    void set_total_entries(uint32_t) {}
    void set_encoded_buckets(uint32_t, uint32_t, uint32_t) {}
    void set_large_bucket_threshold(uint32_t) {}
    void stop() {}

    template <typename Stage> void time(msm_stage, Stage&& stage) { std::forward<Stage>(stage)(); }
};

class ProfileMsmRecorder {
  public:
    explicit ProfileMsmRecorder(msm_profile* profile)
        : profile_(profile)
    {
        if (profile_ != nullptr) {
            *profile_ = {};
        }
    }

    ProfileMsmRecorder(const ProfileMsmRecorder&) = delete;
    ProfileMsmRecorder& operator=(const ProfileMsmRecorder&) = delete;

    ~ProfileMsmRecorder()
    {
        if (profile_ != nullptr && !stopped_) {
            stop();
        }
    }

    void start(cudaStream_t stream, const uint32_t bits_per_slice)
    {
        stream_ = stream;
        if (profile_ == nullptr) {
            return;
        }
        profile_->bits_per_slice = bits_per_slice;
        check_cuda(cudaEventCreate(&total_start_), "cudaEventCreate total start");
        check_cuda(cudaEventCreate(&total_stop_), "cudaEventCreate total stop");
        check_cuda(cudaEventRecord(total_start_, stream_), "cudaEventRecord total start");
    }

    void set_total_entries(const uint32_t total_entries)
    {
        if (profile_ != nullptr) {
            profile_->total_entries = total_entries;
        }
    }

    void set_encoded_buckets(const uint32_t encoded_buckets,
                             const uint32_t active_buckets,
                             const uint32_t zero_bucket_offset)
    {
        if (profile_ != nullptr) {
            profile_->encoded_buckets = encoded_buckets;
            profile_->active_buckets = active_buckets;
            profile_->zero_bucket_offset = zero_bucket_offset;
        }
    }

    void set_large_bucket_threshold(const uint32_t large_bucket_threshold)
    {
        if (profile_ != nullptr) {
            profile_->large_bucket_threshold = large_bucket_threshold;
        }
    }

    template <typename Stage> void time(const msm_stage stage_name, Stage&& stage)
    {
        float* elapsed_ms = stage_slot(stage_name);
        if (elapsed_ms == nullptr) {
            std::forward<Stage>(stage)();
            return;
        }

        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event, stream_), "cudaEventRecord start");
        std::forward<Stage>(stage)();
        check_cuda(cudaEventRecord(stop_event, stream_), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        check_cuda(cudaEventElapsedTime(elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
    }

    void stop()
    {
        if (profile_ == nullptr || stopped_) {
            return;
        }
        check_cuda(cudaEventRecord(total_stop_, stream_), "cudaEventRecord total stop");
        check_cuda(cudaEventSynchronize(total_stop_), "cudaEventSynchronize total stop");
        check_cuda(cudaEventElapsedTime(&profile_->total_profiled_ms, total_start_, total_stop_), "cudaEventElapsedTime total");
        check_cuda(cudaEventDestroy(total_stop_), "cudaEventDestroy total stop");
        check_cuda(cudaEventDestroy(total_start_), "cudaEventDestroy total start");
        stopped_ = true;
    }

  private:
    float* stage_slot(const msm_stage stage_name)
    {
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

    msm_profile* profile_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t total_start_ = nullptr;
    cudaEvent_t total_stop_ = nullptr;
    bool stopped_ = false;
};

__global__ void split_scalars_kernel(const fr_t* scalars_montgomery,
                                     fr_t* scalars_standard,
                                     uint64_t* bucket_indices,
                                     uint32_t* point_indices,
                                     const size_t num_scalars,
                                     const uint32_t point_start_index,
                                     const uint32_t bits_per_slice,
                                     const uint32_t num_windows)
{
    const size_t scalar_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (scalar_idx >= num_scalars) {
        return;
    }

    const fr_t scalar = scalars_montgomery[scalar_idx].from_montgomery_form_reduced();
    scalars_standard[scalar_idx] = scalar;

    const size_t output_base = scalar_idx * num_windows;
    const uint32_t point_index = point_start_index + static_cast<uint32_t>(scalar_idx);
    for (uint32_t window = 0; window < num_windows; ++window) {
        const uint32_t digit = scalar.is_zero() ? 0 : get_scalar_slice(scalar, window, bits_per_slice);
        bucket_indices[output_base + window] = digit == 0 ? 0 : ((static_cast<uint64_t>(window) << bits_per_slice) | digit);
        point_indices[output_base + window] = point_index;
    }
}

__global__ void build_bucket_jobs_kernel(const int* bucket_sizes,
                                         uint32_t* bucket_size_sort_keys,
                                         int* bucket_run_indices,
                                         const int zero_bucket_offset,
                                         const int num_active_buckets)
{
    const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (job_idx >= num_active_buckets) {
        return;
    }

    const int run_idx = job_idx + zero_bucket_offset;
    const uint32_t bucket_size = static_cast<uint32_t>(bucket_sizes[run_idx]);
    bucket_size_sort_keys[job_idx] = ~bucket_size;
    bucket_run_indices[job_idx] = run_idx;
}

__global__ void init_bucket_storage_kernel(jacobian_g1_t* buckets, const size_t num_buckets)
{
    const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < num_buckets) {
        buckets[idx] = jacobian_infinity();
    }
}

__global__ void accumulate_normal_buckets_kernel(const int* sorted_bucket_run_indices,
                                                 const uint64_t* unique_bucket_indices,
                                                 const int* bucket_sizes,
                                                 const int* bucket_offsets,
                                                 const uint32_t* sorted_point_indices,
                                                 const affine_g1_t* points,
                                                 jacobian_g1_t* buckets,
                                                 const int num_active_buckets,
                                                 const int large_bucket_threshold)
{
    const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (job_idx >= num_active_buckets) {
        return;
    }

    const int run_idx = sorted_bucket_run_indices[job_idx];
    const int count = bucket_sizes[run_idx];
    if (count > large_bucket_threshold) {
        return;
    }

    const int start = bucket_offsets[run_idx];
    buckets[unique_bucket_indices[run_idx]] = chained_mixed_add_indexed_nonzero(points, sorted_point_indices, start, count);
}

__global__ void accumulate_large_buckets_kernel(const int* sorted_bucket_run_indices,
                                                const uint64_t* unique_bucket_indices,
                                                const int* bucket_sizes,
                                                const int* bucket_offsets,
                                                const uint32_t* sorted_point_indices,
                                                const affine_g1_t* points,
                                                jacobian_g1_t* buckets,
                                                const int num_active_buckets,
                                                const int large_bucket_threshold)
{
    const int job_idx = static_cast<int>(blockIdx.x);
    if (job_idx >= num_active_buckets) {
        return;
    }

    const int run_idx = sorted_bucket_run_indices[job_idx];
    const int count = bucket_sizes[run_idx];
    if (count <= large_bucket_threshold) {
        return;
    }

    const int start = bucket_offsets[run_idx];
    jacobian_g1_t local = chained_mixed_add_indexed_nonzero(
        points, sorted_point_indices, start, count, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));

    __shared__ jacobian_g1_t partials[BUCKET_THREADS];
    partials[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] = jacobian_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        buckets[unique_bucket_indices[run_idx]] = partials[0];
    }
}

__global__ void reduce_bucket_bit_kernel(jacobian_g1_t* buckets,
                                         jacobian_g1_t* bit_sums,
                                         const uint32_t bit,
                                         const uint32_t bits_per_slice,
                                         const uint32_t num_windows)
{
    const uint32_t window = blockIdx.x;
    if (window >= num_windows) {
        return;
    }

    const uint32_t bucket_stride = uint32_t{ 1 } << bits_per_slice;
    const uint32_t half = uint32_t{ 1 } << bit;
    const uint32_t base = window * bucket_stride;

    jacobian_g1_t local = jacobian_infinity();
    for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        local = jacobian_add(local, buckets[base + half + i]);
    }

    __shared__ jacobian_g1_t partials[REDUCTION_THREADS];
    partials[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] = jacobian_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bit_sums[(window * bits_per_slice) + bit] = partials[0];
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        buckets[base + i] = jacobian_add(buckets[base + i], buckets[base + half + i]);
    }
}

__global__ void compose_window_sums_kernel(const jacobian_g1_t* bit_sums,
                                           jacobian_g1_t* window_sums,
                                           const uint32_t bits_per_slice,
                                           const uint32_t num_windows)
{
    const uint32_t window = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (window >= num_windows) {
        return;
    }

    jacobian_g1_t accumulator = jacobian_infinity();
    for (int bit = static_cast<int>(bits_per_slice) - 1; bit >= 0; --bit) {
        self_double(accumulator);
        accumulator = jacobian_add(accumulator, bit_sums[(window * bits_per_slice) + static_cast<uint32_t>(bit)]);
    }
    window_sums[window] = accumulator;
}

__global__ void reduce_windows_running_sum_kernel(const jacobian_g1_t* buckets,
                                                  jacobian_g1_t* window_sums,
                                                  const uint32_t bits_per_slice,
                                                  const uint32_t num_windows)
{
    const uint32_t window = blockIdx.x;
    if (window >= num_windows || threadIdx.x != 0) {
        return;
    }

    const uint32_t bucket_stride = uint32_t{ 1 } << bits_per_slice;
    const uint32_t base = window * bucket_stride;
    jacobian_g1_t running_sum = jacobian_infinity();
    jacobian_g1_t window_sum = jacobian_infinity();

    for (uint32_t bucket = bucket_stride - 1; bucket > 0; --bucket) {
        running_sum = jacobian_add(running_sum, buckets[base + bucket]);
        window_sum = jacobian_add(window_sum, running_sum);
    }
    window_sums[window] = window_sum;
}

__global__ void final_accumulation_kernel(const jacobian_g1_t* window_sums,
                                          affine_g1_t* result,
                                          const uint32_t bits_per_slice,
                                          const uint32_t num_windows,
                                          const uint32_t remainder)
{
    const uint32_t msm_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (msm_index > 0) {
        return;
    }

    jacobian_g1_t accumulator = jacobian_infinity();
    for (uint32_t window = 0; window < num_windows; ++window) {
        const uint32_t num_doublings = (window == num_windows - 1 && remainder != 0) ? remainder : bits_per_slice;
        for (uint32_t i = 0; i < num_doublings; ++i) {
            self_double(accumulator);
        }
        accumulator = jacobian_add(accumulator, window_sums[window]);
    }
    *result = to_affine(accumulator);
}

affine_g1_t affine_infinity_host()
{
    affine_g1_t out{ fq_t::zero(), fq_t::zero() };
    out.x.self_set_msb();
    return out;
}

template <typename Recorder>
void bucket_pippenger_msm_impl(const fr_t* scalars,
                               const size_t num_scalars,
                               const affine_g1_t* points,
                               const size_t num_points,
                               const size_t point_start_index_size,
                               const uint32_t bits_per_slice,
                               affine_g1_t* result_host,
                               Recorder& recorder)
{
    auto& context = bb::gpu::default_context();
    void* stream = context.stream();
    cudaStream_t cuda_stream = as_cuda_stream(stream);
    recorder.start(cuda_stream, bits_per_slice);
    recorder.time(msm_stage::h2d_points, [&]() {
        context.ensure_srs_uploaded({ points, num_points });
    });

    const uint32_t num_windows = (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
    const uint32_t remainder = NUM_BITS_IN_FIELD % bits_per_slice;
    const uint32_t point_start_index = static_cast<uint32_t>(point_start_index_size);
    const size_t total_entries_size = num_scalars * static_cast<size_t>(num_windows);
    check_condition(total_entries_size <= static_cast<size_t>(INT32_MAX), "bb::gpu::bn254::msm: schedule exceeds CUB int range");
    const int total_entries = static_cast<int>(total_entries_size);
    recorder.set_total_entries(static_cast<uint32_t>(total_entries));

    DeviceBuffer<fr_t> scalars_montgomery;
    DeviceBuffer<fr_t> scalars_standard;
    recorder.time(msm_stage::h2d_scalars, [&]() {
        copy_to_device(scalars_montgomery, { scalars, num_scalars }, stream);
    });
    scalars_standard.resize(num_scalars);

    DeviceBuffer<uint64_t> bucket_indices;
    DeviceBuffer<uint64_t> sorted_bucket_indices;
    DeviceBuffer<uint32_t> point_indices;
    DeviceBuffer<uint32_t> sorted_point_indices;
    bucket_indices.resize(total_entries_size);
    sorted_bucket_indices.resize(total_entries_size);
    point_indices.resize(total_entries_size);
    sorted_point_indices.resize(total_entries_size);

    const uint32_t split_blocks = ceil_div_u32(num_scalars, SPLIT_THREADS);
    recorder.time(msm_stage::split_scalars, [&]() {
        split_scalars_kernel<<<split_blocks, SPLIT_THREADS, 0, cuda_stream>>>(scalars_montgomery.data(),
                                                                              scalars_standard.data(),
                                                                              bucket_indices.data(),
                                                                              point_indices.data(),
                                                                              num_scalars,
                                                                              point_start_index,
                                                                              bits_per_slice,
                                                                              num_windows);
        check_cuda(cudaGetLastError(), "split_scalars_kernel launch");
    });

    DeviceBuffer<std::byte> temp_storage;
    recorder.time(msm_stage::sort_records, [&]() {
        cub_sort_pairs(temp_storage,
                       bucket_indices.data(),
                       sorted_bucket_indices.data(),
                       point_indices.data(),
                       sorted_point_indices.data(),
                       total_entries,
                       0,
                       bits_per_slice + WINDOW_KEY_BITS,
                       stream);
    });

    DeviceBuffer<uint64_t> single_bucket_indices;
    DeviceBuffer<int> bucket_sizes;
    DeviceBuffer<int> num_encoded_buckets_device;
    single_bucket_indices.resize(total_entries_size);
    bucket_sizes.resize(total_entries_size);
    num_encoded_buckets_device.resize(1);
    recorder.time(msm_stage::encode_buckets, [&]() {
        cub_run_length_encode(temp_storage,
                              sorted_bucket_indices.data(),
                              single_bucket_indices.data(),
                              bucket_sizes.data(),
                              num_encoded_buckets_device.data(),
                              total_entries,
                              stream);
    });

    int num_encoded_buckets = 0;
    uint64_t first_bucket_index = 0;
    copy_device_to_host(&num_encoded_buckets, num_encoded_buckets_device.data(), sizeof(int), stream);
    copy_device_to_host(&first_bucket_index, single_bucket_indices.data(), sizeof(uint64_t), stream);
    context.sync();
    const int zero_bucket_offset = (num_encoded_buckets > 0 && first_bucket_index == 0) ? 1 : 0;
    const int num_active_buckets = num_encoded_buckets - zero_bucket_offset;
    recorder.set_encoded_buckets(static_cast<uint32_t>(num_encoded_buckets),
                                 static_cast<uint32_t>(num_active_buckets),
                                 static_cast<uint32_t>(zero_bucket_offset));
    if (num_active_buckets == 0) {
        *result_host = affine_infinity_host();
        recorder.stop();
        return;
    }

    DeviceBuffer<int> bucket_offsets;
    bucket_offsets.resize(static_cast<size_t>(num_encoded_buckets));
    recorder.time(msm_stage::scan_bucket_offsets, [&]() {
        cub_exclusive_sum(temp_storage, bucket_sizes.data(), bucket_offsets.data(), num_encoded_buckets, stream);
    });

    DeviceBuffer<uint32_t> bucket_size_sort_keys;
    DeviceBuffer<uint32_t> sorted_bucket_size_sort_keys;
    DeviceBuffer<int> bucket_run_indices;
    DeviceBuffer<int> sorted_bucket_run_indices;
    bucket_size_sort_keys.resize(static_cast<size_t>(num_active_buckets));
    sorted_bucket_size_sort_keys.resize(static_cast<size_t>(num_active_buckets));
    bucket_run_indices.resize(static_cast<size_t>(num_active_buckets));
    sorted_bucket_run_indices.resize(static_cast<size_t>(num_active_buckets));

    const uint32_t bucket_job_blocks = ceil_div_u32(static_cast<size_t>(num_active_buckets), BUCKET_THREADS);
    recorder.time(msm_stage::build_bucket_jobs, [&]() {
        build_bucket_jobs_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(bucket_sizes.data(),
                                                                                       bucket_size_sort_keys.data(),
                                                                                       bucket_run_indices.data(),
                                                                                       zero_bucket_offset,
                                                                                       num_active_buckets);
        check_cuda(cudaGetLastError(), "build_bucket_jobs_kernel launch");
    });

    recorder.time(msm_stage::sort_bucket_jobs, [&]() {
        cub_sort_pairs(temp_storage,
                       bucket_size_sort_keys.data(),
                       sorted_bucket_size_sort_keys.data(),
                       bucket_run_indices.data(),
                       sorted_bucket_run_indices.data(),
                       num_active_buckets,
                       0,
                       32,
                       stream);
    });

    const uint32_t bucket_stride = uint32_t{ 1 } << bits_per_slice;
    const size_t total_dense_buckets = static_cast<size_t>(num_windows) * bucket_stride;
    DeviceBuffer<jacobian_g1_t> dense_buckets;
    DeviceBuffer<jacobian_g1_t> bit_sums;
    DeviceBuffer<jacobian_g1_t> window_sums;
    DeviceBuffer<affine_g1_t> result_device;
    dense_buckets.resize(total_dense_buckets);
    bit_sums.resize(static_cast<size_t>(num_windows) * bits_per_slice);
    window_sums.resize(num_windows);
    result_device.resize(1);

    const uint32_t init_blocks = ceil_div_u32(total_dense_buckets, BUCKET_THREADS);
    recorder.time(msm_stage::init_buckets, [&]() {
        init_bucket_storage_kernel<<<init_blocks, BUCKET_THREADS, 0, cuda_stream>>>(dense_buckets.data(), total_dense_buckets);
        check_cuda(cudaGetLastError(), "init_bucket_storage_kernel launch");
    });

    const int average_bucket_size =
        static_cast<int>((num_scalars + static_cast<size_t>(bucket_stride) - 1) / static_cast<size_t>(bucket_stride));
    const int threshold_candidate = 4 * average_bucket_size;
    const int large_bucket_threshold =
        threshold_candidate > LARGE_BUCKET_MIN_THRESHOLD ? threshold_candidate : LARGE_BUCKET_MIN_THRESHOLD;
    recorder.set_large_bucket_threshold(static_cast<uint32_t>(large_bucket_threshold));

    recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
        accumulate_normal_buckets_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(sorted_bucket_run_indices.data(),
                                                                                               single_bucket_indices.data(),
                                                                                               bucket_sizes.data(),
                                                                                               bucket_offsets.data(),
                                                                                               sorted_point_indices.data(),
                                                                                               context.srs_points().data(),
                                                                                               dense_buckets.data(),
                                                                                               num_active_buckets,
                                                                                               large_bucket_threshold);
        check_cuda(cudaGetLastError(), "accumulate_normal_buckets_kernel launch");
    });

    recorder.time(msm_stage::accumulate_large_buckets, [&]() {
        accumulate_large_buckets_kernel<<<static_cast<uint32_t>(num_active_buckets), BUCKET_THREADS, 0, cuda_stream>>>(
            sorted_bucket_run_indices.data(),
            single_bucket_indices.data(),
            bucket_sizes.data(),
            bucket_offsets.data(),
            sorted_point_indices.data(),
            context.srs_points().data(),
            dense_buckets.data(),
            num_active_buckets,
            large_bucket_threshold);
        check_cuda(cudaGetLastError(), "accumulate_large_buckets_kernel launch");
    });

    if (USE_SERIAL_RUNNING_SUM_REDUCTION_FALLBACK) {
        recorder.time(msm_stage::reduce_buckets, [&]() {
            reduce_windows_running_sum_kernel<<<num_windows, 1, 0, cuda_stream>>>(
                dense_buckets.data(), window_sums.data(), bits_per_slice, num_windows);
            check_cuda(cudaGetLastError(), "reduce_windows_running_sum_kernel launch");
        });
    } else {
        recorder.time(msm_stage::reduce_buckets, [&]() {
            for (int bit = static_cast<int>(bits_per_slice) - 1; bit >= 0; --bit) {
                reduce_bucket_bit_kernel<<<num_windows, REDUCTION_THREADS, 0, cuda_stream>>>(
                    dense_buckets.data(), bit_sums.data(), static_cast<uint32_t>(bit), bits_per_slice, num_windows);
                check_cuda(cudaGetLastError(), "reduce_bucket_bit_kernel launch");
            }
        });

        const uint32_t window_blocks = ceil_div_u32(num_windows, BUCKET_THREADS);
        recorder.time(msm_stage::compose_windows, [&]() {
            compose_window_sums_kernel<<<window_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
                bit_sums.data(), window_sums.data(), bits_per_slice, num_windows);
            check_cuda(cudaGetLastError(), "compose_window_sums_kernel launch");
        });
    }

    recorder.time(msm_stage::final_accumulation, [&]() {
        final_accumulation_kernel<<<1, 32, 0, cuda_stream>>>(
            window_sums.data(), result_device.data(), bits_per_slice, num_windows, remainder);
        check_cuda(cudaGetLastError(), "final_accumulation_kernel launch");
    });

    recorder.time(msm_stage::d2h_result, [&]() {
        copy_device_to_host(result_host, result_device.data(), sizeof(affine_g1_t), stream);
    });
    recorder.stop();
    context.sync();
}

void bucket_pippenger_msm(const fr_t* scalars,
                          const size_t num_scalars,
                          const affine_g1_t* points,
                          const size_t num_points,
                          const size_t point_start_index_size,
                          const uint32_t bits_per_slice,
                          affine_g1_t* result_host)
{
    NoopMsmRecorder recorder;
    bucket_pippenger_msm_impl(
        scalars, num_scalars, points, num_points, point_start_index_size, bits_per_slice, result_host, recorder);
}

void bucket_pippenger_msm_profiled(const fr_t* scalars,
                                   const size_t num_scalars,
                                   const affine_g1_t* points,
                                   const size_t num_points,
                                   const size_t point_start_index_size,
                                   const uint32_t bits_per_slice,
                                   affine_g1_t* result_host,
                                   msm_profile* profile)
{
    ProfileMsmRecorder recorder(profile);
    bucket_pippenger_msm_impl(
        scalars, num_scalars, points, num_points, point_start_index_size, bits_per_slice, result_host, recorder);
}

} // namespace

void msm_raw(const fr_t* scalars,
             const size_t num_scalars,
             const affine_g1_t* points,
             const size_t num_points,
             const size_t point_start_index_size,
             const uint32_t bits_per_slice,
             affine_g1_t* result_host)
{
    if (num_scalars == 0) {
        *result_host = affine_infinity_host();
        return;
    }

    bucket_pippenger_msm(scalars, num_scalars, points, num_points, point_start_index_size, bits_per_slice, result_host);
}

void msm_raw_profiled(const fr_t* scalars,
                      const size_t num_scalars,
                      const affine_g1_t* points,
                      const size_t num_points,
                      const size_t point_start_index_size,
                      const uint32_t bits_per_slice,
                      affine_g1_t* result_host,
                      msm_profile* profile)
{
    if (num_scalars == 0) {
        *result_host = affine_infinity_host();
        if (profile != nullptr) {
            *profile = {};
        }
        return;
    }

    bucket_pippenger_msm_profiled(
        scalars, num_scalars, points, num_points, point_start_index_size, bits_per_slice, result_host, profile);
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
