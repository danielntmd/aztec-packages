# External GPU MSM Benchmarks

This directory contains the benchmark harness for comparing the native BB GPU
BN254 MSM implementation against Icicle v2.8.0 and Icicle v4.x.

The harness is intended for production-machine performance runs where the SRS
or point table is already prepared and scalar inputs are fresh per MSM. It is
not a full prover benchmark.

## What Gets Built

The CMake targets are:

| Target | Backend |
|---|---|
| `gpu_msm_external_bb_bench` | Native BB GPU MSM |
| `gpu_msm_external_icicle_v2_bench` | Icicle v2.8.0 adapter |
| `gpu_msm_external_icicle_v4_bench` | Icicle v4.x adapter |

The wrapper script is:

```bash
barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py
```

It runs the selected binaries, validates that all requested rows were produced,
checks that all implementations produce matching MSM results for each seed and
shape, and writes raw records plus summary tables.

The wrapper supports two scalar/result memory placements:

| Placement | Meaning |
|---|---|
| `--memory-placement host` | Points are resident, scalars are passed from host memory, and results are returned to host inside the backend call. |
| `--memory-placement device` | Points, scalars, and results are device-resident for the timed backend call. Scalar upload and result copy-back for validation happen outside `backend_wall_ms`. |

Icicle v2.8.0 only supports `host` placement in this harness. BB and Icicle v4.x
support both placements.

## Production Suite Overview

For the current GPU MSM cost-analysis run, collect these outputs on the
production machine:

| Suite | Purpose | Command section |
|---|---|---|
| Main MSM sweep | Compare BB, Icicle v2.8.0, and Icicle v4.x across `2^10..2^24`, factors `1,4,8`, host scalars. | [Main MSM Sweep](#main-msm-sweep) |
| Batch-100 host-scalar run | Compare the previous cost-report shape: `2^20 x 100`, `pf=1`, scalar transfer included in backend call. | [Batch-100 Cost-Analysis Shape](#batch-100-cost-analysis-shape) |
| Batch-100 device-resident run | Compare the same `2^20 x 100` shape after scalars are already on device and results stay on device until after timing. | [Device-Resident Batch-100 Shape](#device-resident-batch-100-shape) |
| Optional c-value sweep | Diagnose whether auto `c` is responsible for a performance shift. | [C-Value Sweep](#c-value-sweep) |
| Root rollup / Chonk IVC proof | Measure proof generation using captured proof inputs or a stored rollup proof job. | [Proof Generation](#proof-generation) |

The MSM wrapper writes metadata automatically. The proof-generation scripts
write run records and summaries, but do not currently capture the full machine
metadata; save `nvidia-smi`, git SHA/status, build configuration, and the exact
command next to proof benchmark outputs.

## Build Checklist

Configure BB with the native GPU backend and a CUDA architecture matching the
production GPU. Do not rely on CMake's default CUDA architecture when collecting
numbers for a report.

Example:

```bash
PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake \
  -S barretenberg/cpp \
  -B /tmp/aztec-bb-gpu-msm-bench \
  -G "Unix Makefiles" \
  -DMOBILE=ON \
  -DAVM=OFF \
  -DENABLE_HEAVY_TESTS=OFF \
  -DGPU_BACKEND=native \
  -DCUDAToolkit_ROOT=/path/to/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<compute-capability-without-dot> \
  -DCMAKE_C_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-cc.sh \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-c++.sh \
  -DCMAKE_AR=/absolute/path/to/barretenberg/cpp/scripts/zig-ar.sh \
  -DCMAKE_RANLIB=/absolute/path/to/barretenberg/cpp/scripts/zig-ranlib.sh
```

For example, use `-DCMAKE_CUDA_ARCHITECTURES=120` for compute capability 12.0.

Build the benchmark targets:

```bash
PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake --build /tmp/aztec-bb-gpu-msm-bench \
  --target gpu_msm_external_bb_bench \
           gpu_msm_external_icicle_v2_bench \
           gpu_msm_external_icicle_v4_bench -- -j16
```

## Runtime Inputs

The runner needs CUDA runtime libraries and the Icicle libraries visible through
`LD_LIBRARY_PATH`.

Example:

```bash
export LD_LIBRARY_PATH=/path/to/icicle-v2/lib:/path/to/icicle-v4/lib:/path/to/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

For Icicle v4, either set `ICICLE_BACKEND_INSTALL_DIR` or pass
`--icicle-backend-dir` to the wrapper.

Before a production run, record:

```bash
nvidia-smi
```

The wrapper also stores `nvidia-smi`, selected environment variables, git SHA,
git status, runner mtimes, and CMake cache fields in `metadata.json`.

## Main MSM Sweep

This is the broad comparison sweep used for the cost-analysis baseline:

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench \
  --output-dir /tmp/gpu-msm-external-main \
  --implementations bb,icicle-v2.8.0,icicle-v4.0.0 \
  --mode single \
  --memory-placement host \
  --min-log 10 \
  --max-log 24 \
  --log-step 2 \
  --factors 1,4,8 \
  --repeats 5 \
  --c 0 \
  --icicle-backend-dir /path/to/icicle-v4/backend
```

Use `--allow-skips` only when a row is expected to exceed memory. Without that
flag, missing rows fail the run.

## Batch-100 Cost-Analysis Shape

This run matches the "100 random columns of length 2^20" shape discussed for
the cost report. It uses precompute factor 1 and auto `c`. The host-placement
version includes scalar host-to-device transfer inside the backend call.

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench \
  --output-dir /tmp/gpu-msm-external-batch-2p20x100 \
  --implementations bb,icicle-v4.0.0 \
  --mode batch \
  --memory-placement host \
  --batch-log 20 \
  --batch-size 100 \
  --factors 1 \
  --repeats 5 \
  --c 0 \
  --icicle-backend-dir /path/to/icicle-v4/backend
```

The BB runner defaults to a maximum fused batch size of 16, so this shape runs
as chunks `16+16+16+16+16+16+4`. Override this only for experiments:

```bash
--bb-max-fused-batch-size 100
```

On a 16 GB local GPU, the flat `2^20 x 100` BB fused path did not fit; its
preflight estimated about 70.5 GB of transient allocation. Icicle v2.8.0 also
did not fit locally for this shape under its preflight estimate. Icicle v4.x did
complete locally.

## Device-Resident Batch-100 Shape

Use this run to measure backend execution after scalar inputs have already been
uploaded and results can stay on device until after timing.

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench \
  --output-dir /tmp/gpu-msm-external-batch-2p20x100-device \
  --implementations bb,icicle-v4.0.0 \
  --mode batch \
  --memory-placement device \
  --batch-log 20 \
  --batch-size 100 \
  --factors 1 \
  --repeats 5 \
  --c 0 \
  --icicle-backend-dir /path/to/icicle-v4/backend
```

This still uses BB's default fused chunk cap of 16, so BB executes the 100
columns as `16+16+16+16+16+16+4` while pointing into one pre-uploaded scalar
buffer. On the local 16 GB RTX 5060 Ti, a repeat-1 smoke completed with:

| Backend | Placement | Shape | comparison avg ms | per MSM ms |
|---|---|---:|---:|---:|
| BB | device | `2^20 x 100` | 1468.150 | 14.682 |
| Icicle v4.x | device | `2^20 x 100` | 1641.950 | 16.420 |

Use production hardware numbers for reporting; the local row is only a sanity
check that the path runs and validates.

## Proof Generation

There are two proof-generation benchmark entry points. Use the one that matches
the production artifact available on the machine.

Proof generation needs a full `bb` binary, not just the MSM benchmark runners.
The MSM build above uses `MOBILE=ON`, which is appropriate for the external MSM
suite but is not the right build for the rollup prover path. Build a separate
GPU-enabled proof binary with `MOBILE=OFF`:

```bash
PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake \
  -S barretenberg/cpp \
  -B /tmp/aztec-bb-gpu-proof \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOBILE=OFF \
  -DAVM=OFF \
  -DENABLE_HEAVY_TESTS=OFF \
  -DGPU_BACKEND=native \
  -DCUDAToolkit_ROOT=/path/to/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<compute-capability-without-dot> \
  -DCMAKE_C_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-cc.sh \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-c++.sh \
  -DCMAKE_AR=/absolute/path/to/barretenberg/cpp/scripts/zig-ar.sh \
  -DCMAKE_RANLIB=/absolute/path/to/barretenberg/cpp/scripts/zig-ranlib.sh

PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake --build /tmp/aztec-bb-gpu-proof --target bb -- -j16
```

The e2e proof-input generation path depends on the normal generated TypeScript,
Noir, and L1 artifacts. From a fresh rebase, run the normal bootstrap/build
flow before attempting the e2e command:

```bash
PATH=/path/to/cmake/bin:/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
./bootstrap.sh build yarn-project
```

On the local 16 GB RTX 5060 Ti setup, the GPU proof `bb` target previously
built successfully with `GPU_BACKEND=native`, `MOBILE=OFF`, `AVM=OFF`, and
`CMAKE_CUDA_ARCHITECTURES=120`. After rebasing this benchmark branch onto
`upstream/v4`, a clean focused `cmake --build --preset wasm-threads --target bb`
check on `upstream/v4` also passed. Re-run the normal bootstrap flow on the
target production machine before relying on e2e proof-generation timings.

### Proof Input Reproducibility

For CPU/GPU comparisons, keep the proof inputs fixed. The e2e flow below
generates real proof jobs, but the complete proof-store directory should be
treated as a captured fixture once it is generated. Replaying the same stored
input URI through different `bb` binaries is deterministic enough for benchmark
comparison; regenerating the e2e flow may produce a different proof-store layout
or different job identifiers because the surrounding test runtime creates fresh
state.

Record these fields next to every proof benchmark result:

| Field | Purpose |
|---|---|
| Proof input URI | Identifies the exact proof job being replayed. |
| Proof store URI | Lets the prover resolve sibling artifacts and outputs. |
| SHA256 of the input file | Confirms CPU and GPU replay used identical inputs. |
| SHA256 of the output proof | Confirms repeated replays produced the same proof bytes. |
| Git SHA/status and `bb` binary path | Ties the replay to the exact implementation. |
| GPU env vars | Captures settings such as `BB_GPU_MSM_PRECOMPUTE_FACTOR` and `BB_GPU_MSM_MAX_BATCH_SIZE`. |

Do not commit large generated proof stores into the repository. For production
benchmarking, copy the captured proof-store directory to the target machine or
store it in an external artifact location, then replay that same fixture for
CPU and GPU runs.

### Captured Chonk IVC Inputs

Use this when you have an `ivc-inputs.msgpack` file for the root rollup or other
target flow:

```bash
python3 barretenberg/gpu/benchmark/proof_generation/run_chonk_ivc_proof_bench.py \
  --bb-bin /path/to/bb \
  --ivc-inputs /path/to/root-rollup/ivc-inputs.msgpack \
  --output-dir /tmp/gpu-proof-chonk-ivc-root-rollup \
  --warmups 1 \
  --repeats 5 \
  --label root-rollup
```

The script runs:

```bash
bb prove --scheme chonk --ivc_inputs_path <ivc-inputs.msgpack>
```

and writes `raw.jsonl`, `summary.json`, `summary.md`, per-run `bb.log`,
`benchmark_breakdown.json`, `memory_profile.json`, and proof-size fields.

### Stored Rollup Proof Job

Use this when the production artifact is a stored prover-client proof input URI.
This path invokes `BBNativeRollupProver` and is closer to the rollup proof-job
orchestration path than direct `bb prove`:

```bash
node yarn-project/scripts/run_rollup_proof_job_bench.mjs \
  --proof-uri file:///path/to/proof-input \
  --bb-bin /path/to/bb \
  --acvm-bin /path/to/acvm \
  --output-dir /tmp/gpu-rollup-proof-job-root-rollup \
  --expected-type ROOT_ROLLUP \
  --warmups 1 \
  --repeats 5
```

For non-file proof stores, pass `--proof-store` explicitly if it cannot be
inferred from `--proof-uri`.

To generate a stored root-rollup proof input locally, run the real-proof e2e
prover flow with a file proof store. The first transfer test is the shortest
targeted path that should advance into epoch proving and write rollup proof jobs:

```bash
env -u FAKE_PROOFS \
  LD_LIBRARY_PATH=/path/to/cuda-12.8/lib64:$LD_LIBRARY_PATH \
  BB_BINARY_PATH=/tmp/aztec-bb-gpu-proof/bin/bb \
  ACVM_BINARY_PATH=/absolute/path/to/noir/noir-repo/target/release/acvm \
  BB_WORKING_DIRECTORY=/tmp/aztec-gpu-e2e-bb-work \
  ACVM_WORKING_DIRECTORY=/tmp/aztec-gpu-e2e-acvm-work \
  BB_SKIP_CLEANUP=1 \
  PROVER_REAL_PROOFS=1 \
  PROVER_PROOF_STORE=file:///tmp/aztec-gpu-e2e-proof-store \
  PROVER_AGENT_COUNT=1 \
  LOG_LEVEL=info \
  JEST_CACHE_DIR=/tmp/aztec-gpu-e2e-jest-cache \
  yarn-project/end-to-end/scripts/run_test.sh simple e2e_prover/full \
    "makes both public and private transfers"
```

After the e2e run succeeds, choose the generated root-rollup input:

```bash
find /tmp/aztec-gpu-e2e-proof-store/inputs/ROOT_ROLLUP -type f
```

Replay that proof job with the benchmark script:

```bash
node yarn-project/scripts/run_rollup_proof_job_bench.mjs \
  --proof-uri file:///tmp/aztec-gpu-e2e-proof-store/inputs/ROOT_ROLLUP/<job-id> \
  --proof-store file:///tmp/aztec-gpu-e2e-proof-store \
  --bb-bin /tmp/aztec-bb-gpu-proof/bin/bb \
  --acvm-bin /absolute/path/to/noir/noir-repo/target/release/acvm \
  --output-dir /tmp/gpu-rollup-proof-job-root-rollup \
  --expected-type ROOT_ROLLUP \
  --warmups 1 \
  --repeats 5
```

Report proof generation separately from MSM microbenchmarks. Proof timings
include orchestration, witness/proving work, BB process behavior, and any
configured cleanup policy; they are not isolated MSM timings.

## C-Value Sweep

Use `--c-values` to run explicit `c` values. This is useful when checking
whether auto `c` is responsible for a performance shift.

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench \
  --output-dir /tmp/gpu-msm-external-c-sweep \
  --implementations bb \
  --mode single \
  --memory-placement host \
  --min-log 24 \
  --max-log 24 \
  --factors 4 \
  --repeats 5 \
  --c-values 13,14,15,16,17,18
```

## Timing Fields

Each raw JSONL record includes:

| Field | Meaning |
|---|---|
| `setup_wall_ms` | Point/SRS setup for that factor, including host conversion and upload where applicable. Excluded from `comparison_ms`. |
| `memory_placement` | `host` or `device`, matching `--memory-placement`. |
| `precompute_wall_ms` | Host wall time for explicit precomputation, if the backend exposes it. |
| `precompute_device_ms` | CUDA event time for precomputation, when available. |
| `outer_wall_ms` | Runner-level call region. For Icicle runners this includes host-side scalar format conversion before the backend call. |
| `backend_wall_ms` | Backend call wall time. This is the closest cross-backend comparison field. |
| `device_ms` | CUDA-event/profiled device time when available. BB and Icicle v2 expose this; Icicle v4 does not. |
| `comparison_ms` | `device_ms` when present, otherwise `backend_wall_ms`. |
| `per_msm_ms` | `comparison_ms / batch_size`. |
| `c` | Backend-reported or resolved c value. `requested_c` is added by the wrapper. |

For the resident-points backend comparison:

| Portion | Included in `backend_wall_ms` |
|---|---|
| Point/SRS setup and upload | No |
| Scalar generation | No |
| BB-to-Icicle host scalar format conversion | No |
| Scalar host-to-device transfer | Yes |
| MSM GPU work | Yes |
| Result copy back to host | Yes |

Do not label `backend_wall_ms` as pure kernel time. It is the backend-call wall
time with resident points and host scalars.

For `--memory-placement device`:

| Portion | Included in `backend_wall_ms` |
|---|---|
| Point/SRS setup and upload | No |
| Scalar generation | No |
| Scalar host-to-device upload | No |
| MSM GPU work | Yes |
| Result copy back to host for validation | No |

Do not compare `host` and `device` placement without naming the placement. They
answer different questions.

## Outputs

The wrapper writes:

| File | Contents |
|---|---|
| `<implementation>.jsonl` | Raw records from each backend runner. |
| `raw.jsonl` | Merged raw records with `requested_c`. |
| `summary.json` | Aggregated summary rows. |
| `summary.md` | Human-readable summary table. |
| `metadata.json` | Reproduction metadata, commands, completeness result, git state, GPU info, and output paths. |

For production reporting, include:

- GPU model, driver, CUDA toolkit, and `CMAKE_CUDA_ARCHITECTURES`.
- Git SHA and dirty status from `metadata.json`.
- The exact command from `metadata.json`.
- `summary.md`.
- Whether the run passed completeness and result matching.
- The memory placement used for each table.
- Any skipped rows and the skip reason.

## Sanity Checks

Treat the run as suspect if:

- `metadata.json` has `"status": "failed"`.
- Completeness reports missing rows without an expected memory skip.
- Result validation fails.
- Icicle v4 is dramatically slower than expected on the same production GPU.
- `outer_wall_ms` is used as the comparison metric without explaining host
  scalar conversion overhead.
- `host` and `device` placement rows are mixed without explanation.
- The CMake cache shows an unexpected CUDA architecture.
