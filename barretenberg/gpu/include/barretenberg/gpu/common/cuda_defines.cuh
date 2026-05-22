#pragma once

#ifdef BB_GPU_NATIVE

#if defined(__CUDACC__)
#define BB_GPU_HD __host__ __device__
#define BB_GPU_D __device__
#define BB_GPU_HD_FORCEINLINE __host__ __device__ __forceinline__
#define BB_GPU_D_FORCEINLINE __device__ __forceinline__
#else
#define BB_GPU_HD
#define BB_GPU_D
#define BB_GPU_HD_FORCEINLINE inline
#define BB_GPU_D_FORCEINLINE inline
#endif

#endif // BB_GPU_NATIVE
