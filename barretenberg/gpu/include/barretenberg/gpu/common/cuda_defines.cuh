#pragma once

#ifdef BB_GPU_NATIVE

#if defined(__CUDACC__)
#define BB_GPU_HD __host__ __device__
#define BB_GPU_D __device__
#else
#define BB_GPU_HD
#define BB_GPU_D
#endif

#endif // BB_GPU_NATIVE
