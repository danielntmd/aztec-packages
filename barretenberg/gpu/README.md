# barretenberg/gpu

CUDA backend for barretenberg's cryptographic primitives. The v1 surface is
the BN254 bucket-method Pippenger MSM used by `CommitmentKey<BN254>`.

For build conventions, file-layout rules, and how to add a new curve, see
[`CLAUDE.md`](./CLAUDE.md).

## Public API

```cpp
#include "barretenberg/gpu/backend.hpp"

bb::gpu::Backend<bb::curve::BN254>::init_srs(srs_points);
auto C = bb::gpu::Backend<bb::curve::BN254>::msm(
    scalars, points,
    bb::gpu::MsmConfig{.bits_per_slice = 0, .precompute_factor = 4});
bb::gpu::Backend<bb::curve::BN254>::shutdown();
```

The `commitment_key.hpp` integration uses the thin templated wrappers in
`barretenberg/gpu/commitment_schemes/commitment_key_msm.hpp`, so direct
callers usually go through `bb::CommitmentKey<Curve>` rather than touching
`Backend<Curve>` themselves.

## Pipeline

The BN254 MSM follows the bucket-method Pippenger algorithm with three
significant additions:

1. **Two-chunk H2D / split overlap.** Scalars are uploaded in two host-to-
   device transfers; the first chunk's split-into-windows kernel runs in
   parallel with the second chunk's H2D copy.
   (`src/msm/internal/scalar_split_pipeline.cuh`)

2. **Precomputed shifted SRS.** Each `MsmConfig::precompute_factor` doubles
   the SRS on device and folds `precompute_factor` Pippenger windows into one.
   The shifted SRS is cached on the `GpuMsmContext` and reused across calls
   with matching `(shift_bits, precompute_factor, span)`.
   (`src/common/gpu_msm_context.cu`)

3. **Chunked large-bucket accumulation.** Buckets whose sizes exceed an
   adaptive threshold (`4 × average size`, floored at 512) are split into
   fixed-size chunks, each chunk is summed by its own thread, and the chunk
   partials are folded into the final bucket value.
   (`src/msm/internal/{large_bucket_accumulation,chunked_large_buckets}.cuh`)

The remaining stages — record sort, run-length encode, bucket-offset scan,
normal accumulation, bit-sum reduction, window composition, final
accumulation — are stock Pippenger; per-stage timing is captured by the
profiling recorder when callers opt in via `msm_raw_profiled_fq32`.

## Targets

| Target | Source | Purpose |
|---|---|---|
| `gpu` | `src/` | Static library; linked into `barretenberg` and `bb-external` |
| `gpu_tests` | `tests/` | gtest suite (47 tests on BN254) |
| `gpu_msm_bench` | `benchmark/msm_latency.bench.cpp` | Google Benchmark latency runner |

Run a single bench case:

```bash
./build/gpu/gpu_msm_bench \
    --benchmark_filter='BN254/GPU/msm_profiled/24/4$' \
    --benchmark_min_warmup_time=0
```

Run all tests:

```bash
./build/gpu/gpu_tests
```
