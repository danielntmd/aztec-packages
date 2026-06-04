#pragma once

#include <cstddef>
#include <cstdint>

struct icicle_v2_scalar_t {
  uint32_t limbs[8];
};

struct icicle_v2_affine_t {
  uint32_t x[8];
  uint32_t y[8];
  int infinity;
};

struct icicle_v2_prepared_bases_t;

extern "C" icicle_v2_prepared_bases_t *
icicle_v2_prepare_bases(const icicle_v2_affine_t *points, size_t num_points,
                        uint32_t precompute_factor, int c, double *setup_ms,
                        double *precompute_ms);

extern "C" void
icicle_v2_free_prepared_bases(icicle_v2_prepared_bases_t *prepared);

extern "C" int icicle_v2_run_msm(const icicle_v2_scalar_t *scalars,
                                 size_t num_points, uint32_t batch_size,
                                 const icicle_v2_prepared_bases_t *prepared,
                                 uint32_t precompute_factor, int c,
                                 icicle_v2_affine_t *results,
                                 double *backend_call_ms,
                                 double *device_event_ms);

extern "C" const char *icicle_v2_last_error();
