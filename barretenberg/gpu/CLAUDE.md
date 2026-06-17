# barretenberg/gpu

CUDA backend for barretenberg primitives. Currently exposes a BN254 MSM
implementation; the design accommodates additional curves and primitives via
`Backend<Curve>` specialisations.

## Public surface

`include/barretenberg/gpu/` is what the rest of the project may include.

| Header | Purpose |
|---|---|
| `backend.hpp` | `bb::gpu::Backend<Curve>` template + `MsmConfig`. The only entry point for new code. |
| `commitment_schemes/commitment_key_msm.hpp` | Thin templated wrappers routed through `Backend<Curve>`; consumed by `barretenberg/cpp/src/barretenberg/commitment_schemes/commitment_key.hpp`. |
| `common/{cuda_error,device_buffer,device_buffer_pool,device_span,cuda_defines,cub_helpers,nvtx}.{hpp,cuh}` | Reusable host-side CUDA helpers. |
| `curves/bn254/{bn254,fq32_g1}.cuh`, `fields/{field.cuh,bn254/*}` | Device-side field/curve types used by kernels and tests. |

Everything else lives under `src/` and is implementation detail — including
`src/common/gpu_msm_context.hpp` and `src/msm/internal/*`. Do not include
internal headers from outside this directory.

## Directory layout

```
gpu/
├── benchmark/msm_latency.bench.cpp     # gpu_msm_bench: latency benchmark
├── include/barretenberg/gpu/           # PUBLIC surface (see table above)
├── src/
│   ├── common/                         # device_memory.cu, cuda_stream.cu, gpu_msm_context.{hpp,cu}
│   └── msm/
│       ├── bn254_msm.{cpp,cu}          # Backend<curve::BN254> implementation
│       └── internal/                   # Pippenger pipeline stages, kernels, instrumentation
└── tests/                              # gpu_tests (gtest)
```

`src/msm/internal/*.cuh` files are TU-local source fragments concatenated by
`bn254_msm.cu` inside one anonymous namespace. They do not use `#pragma once`
and are not intended to be includable independently.

## File suffix rule

- `.cuh` — uses CUDA intrinsics (`__device__`, `__global__`, `<<<>>>`, CUB).
- `.hpp` — host-only declarations / templates / PODs over CUDA-runtime types.

Renaming a file from `.cuh` to `.hpp` (or vice versa) is a deliberate signal
to readers about what the contents need to compile. Keep this convention when
adding new files.

## Build

Driven by `CMakeLists.txt` in this directory. Targets:

- `gpu` — static library linked into `barretenberg` and `bb-external`.
- `gpu_tests` — gtest executable (run via `ctest -L gpu` or directly).
- `gpu_msm_bench` — Google Benchmark executable.

The build requires CUDA Toolkit (the project uses 12.8 locally) and a host
toolchain new enough for libstdc++ ≥ 12 / libc++. See the parent project's
README for the supported configurations; on the development box the zig
wrapper at `.codex/toolchains/zigwrap/c++` is the known-good host compiler.

## Adding a new curve

1. Add `curves/<curve>/` and `fields/<curve>/` headers in `include/`.
2. Add a `Backend<curve::Foo>` specialisation declaration in `backend.hpp`.
3. Implement the specialisation in `src/msm/foo_msm.cpp` (mirror
   `bn254_msm.cpp`) and add it to `CMakeLists.txt`.
4. Internal pipeline headers live in `src/msm/internal/`. Reuse what makes
   sense (sort/encode/scan stages, buffer pool, instrumentation) and add new
   curve-specific kernels alongside.
5. `commitment_key_msm.hpp` picks the new backend up automatically through
   `Backend<Curve>::available`.

## Testing

`gpu_tests` covers field, curve, MSM correctness, and buffer-pool reuse. The
`GpuBn254PerfSmoke.Msm2p16MatchesCpu` test runs a 2^16 MSM and emits
`gpu_total_ms`; treat its output as a quick health check, not a regression
gate (no fixed threshold).

For real perf comparison, run `gpu_msm_bench` and diff the JSON output
(`--benchmark_out_format=json`) against a previous run on the same hardware.
