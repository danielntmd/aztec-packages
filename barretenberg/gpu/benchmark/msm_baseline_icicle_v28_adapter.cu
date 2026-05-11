#include "api/bn254.h"

#include <cuda_runtime.h>

#include <cstring>

extern "C" cudaError_t bb_gpu_icicle_v28_msm_projective(const void* scalars,
                                                        const void* points,
                                                        const int msm_size,
                                                        const int points_size,
                                                        const int bits_per_slice,
                                                        void* projective_result)
{
    static_assert(sizeof(bn254::scalar_t) == 32);
    static_assert(sizeof(bn254::affine_t) == 64);
    static_assert(sizeof(bn254::projective_t) == 96);

    msm::MSMConfig config = msm::default_msm_config();
    config.points_size = points_size == 0 ? msm_size : points_size;
    config.precompute_factor = 1;
    config.c = bits_per_slice;
    config.bitsize = 254;
    config.large_bucket_factor = 10;
    config.batch_size = 1;
    config.are_scalars_on_device = false;
    config.are_scalars_montgomery_form = false;
    config.are_points_on_device = false;
    config.are_points_montgomery_form = false;
    config.are_results_on_device = false;
    config.is_big_triangle = false;
    config.is_async = false;

    cudaError_t error = bn254_msm_cuda(static_cast<const bn254::scalar_t*>(scalars),
                                       static_cast<const bn254::affine_t*>(points),
                                       msm_size,
                                       config,
                                       static_cast<bn254::projective_t*>(projective_result));
    return error;
}

extern "C" cudaError_t bb_gpu_icicle_v28_msm_projective_with_options(const void* scalars,
                                                                     const void* points,
                                                                     const int msm_size,
                                                                     const int points_size,
                                                                     const int bits_per_slice,
                                                                     const bool scalars_on_device,
                                                                     const bool scalars_montgomery_form,
                                                                     const bool points_on_device,
                                                                     const bool points_montgomery_form,
                                                                     const bool results_on_device,
                                                                     const bool is_big_triangle,
                                                                     const int large_bucket_factor,
                                                                     void* projective_result)
{
    static_assert(sizeof(bn254::scalar_t) == 32);
    static_assert(sizeof(bn254::affine_t) == 64);
    static_assert(sizeof(bn254::projective_t) == 96);

    msm::MSMConfig config = msm::default_msm_config();
    config.points_size = points_size == 0 ? msm_size : points_size;
    config.precompute_factor = 1;
    config.c = bits_per_slice;
    config.bitsize = 254;
    config.large_bucket_factor = large_bucket_factor;
    config.batch_size = 1;
    config.are_scalars_on_device = scalars_on_device;
    config.are_scalars_montgomery_form = scalars_montgomery_form;
    config.are_points_on_device = points_on_device;
    config.are_points_montgomery_form = points_montgomery_form;
    config.are_results_on_device = results_on_device;
    config.is_big_triangle = is_big_triangle;
    config.is_async = false;

    return bn254_msm_cuda(static_cast<const bn254::scalar_t*>(scalars),
                          static_cast<const bn254::affine_t*>(points),
                          msm_size,
                          config,
                          static_cast<bn254::projective_t*>(projective_result));
}

extern "C" void bb_gpu_icicle_v28_projective_to_affine(const void* projective_result, void* affine_result)
{
    bn254::affine_t affine{};
    auto projective = *static_cast<const bn254::projective_t*>(projective_result);
    bn254_to_affine(&projective, &affine);
    std::memcpy(affine_result, &affine, sizeof(affine));
}

extern "C" void bb_gpu_icicle_v28_scalar_to_montgomery(const void* regular, void* montgomery)
{
    const auto input = *static_cast<const bn254::scalar_t*>(regular);
    const auto output = bn254::scalar_t::to_montgomery(input);
    std::memcpy(montgomery, &output, sizeof(output));
}

extern "C" void bb_gpu_icicle_v28_affine_to_montgomery(const void* regular, void* montgomery)
{
    const auto input = *static_cast<const bn254::affine_t*>(regular);
    const auto output = bn254::affine_t::to_montgomery(input);
    std::memcpy(montgomery, &output, sizeof(output));
}

extern "C" cudaError_t bb_gpu_icicle_v28_msm(const void* scalars,
                                             const void* points,
                                             const int msm_size,
                                             const int points_size,
                                             const int bits_per_slice,
                                             void* affine_result)
{
    bn254::projective_t projective_result{};
    cudaError_t error =
        bb_gpu_icicle_v28_msm_projective(scalars, points, msm_size, points_size, bits_per_slice, &projective_result);
    if (error != cudaSuccess) {
        return error;
    }
    bb_gpu_icicle_v28_projective_to_affine(&projective_result, affine_result);
    return cudaSuccess;
}
