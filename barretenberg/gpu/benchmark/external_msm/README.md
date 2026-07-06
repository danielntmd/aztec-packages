# External GPU MSM Benchmarks

This directory contains the benchmark harness for comparing CPU BN254 MSM, the
native BB GPU BN254 MSM implementation, and Icicle v4.x.

The harness is intended for production-machine performance runs where the SRS
or point table is already prepared and scalar inputs are fresh per MSM. It is
not a full prover benchmark.

## What Gets Built

The CMake targets are:

| Target | Backend |
|---|---|
| `gpu_msm_external_cpu_bench` | CPU Pippenger |
| `gpu_msm_external_bb_bench` | Native BB GPU MSM |
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

The CPU runner only supports `host` placement in this harness. BB and Icicle
v4.x support both placements.

## Production Suite Overview

For the current GPU MSM cost-analysis run, collect these outputs on the
production machine:

| Suite | Purpose | Command section |
|---|---|---|
| Main MSM sweep | Compare BB and Icicle v4.x across `2^10..2^24`, factors `1,4,8`, host scalars. | [Main MSM Sweep](#main-msm-sweep) |
| CPU/GPU single MSM sweep | Add CPU Pippenger rows to the same single-MSM sweep schema. | [CPU/GPU Single MSM Sweep](#cpugpu-single-msm-sweep) |
| BB+Icicle field single MSM sweep | Run the BB MSM runner from a separate build that uses Icicle's inline BN254 base-field arithmetic. | [BB+Icicle Field Single MSM Sweep](#bbicicle-field-single-msm-sweep) |
| Batch-100 host-scalar run | Compare the previous cost-report shape: `2^20 x 100`, `pf=1`, scalar transfer included in backend call. | [Batch-100 Cost-Analysis Shape](#batch-100-cost-analysis-shape) |
| Batch-100 device-resident run | Compare the same `2^20 x 100` shape after scalars are already on device and results stay on device until after timing. | [Device-Resident Batch-100 Shape](#device-resident-batch-100-shape) |
| BB+Icicle field batch-100 runs | Run the `2^20 x 100` host and device batch shapes from the BB+Icicle field build. | [BB+Icicle Field Batch-100 Shapes](#bbicicle-field-batch-100-shapes) |
| Optional c-value sweep | Diagnose whether auto `c` is responsible for a performance shift. | [C-Value Sweep](#c-value-sweep) |
| Root rollup / Chonk IVC proof | Measure proof generation using captured proof inputs or a stored rollup proof job. | [Proof Generation](#proof-generation) |
| Proof-store replay | Replay every captured proof job from a real-proof e2e store through the native prover path. | [Proof-Store Replay](#proof-store-replay) |

The MSM wrapper writes metadata automatically. The proof-generation scripts
write run records and summaries, but do not currently capture the full machine
metadata; save `nvidia-smi`, git SHA/status, build configuration, and the exact
command next to proof benchmark outputs.

## Build Checklist

Configure BB with the native GPU backend and a CUDA architecture matching the
production GPU. Do not rely on CMake's default CUDA architecture when collecting
numbers for a report.

The external benchmark runners are optional CMake targets; enable them with
`BB_ENABLE_GPU_MSM_EXTERNAL_BENCH=ON`.

The external Icicle v4 runner needs Icicle v4 headers, shared libraries, and
the CUDA backend. Point CMake at them with `ICICLE_V4_INSTALL_DIR` or the
explicit include/library cache entries:

```bash
-DICICLE_V4_INSTALL_DIR=/path/to/icicle-v4
```

Some Icicle v4 release installs do not copy every curve parameter header used
by the external runner. In that case, use the source include directory and the
installed library directory explicitly:

```bash
-DICICLE_V4_INCLUDE_DIR=/path/to/open-icicle/icicle/include \
-DICICLE_V4_LIBRARY_DIR=/path/to/icicle-v4/lib
```

If the release artifact keeps the CUDA backend outside the library prefix, pass
that backend directory at runtime with `--icicle-backend-dir` or set
`ICICLE_BACKEND_INSTALL_DIR`.

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
  -DBB_ENABLE_GPU_MSM_EXTERNAL_BENCH=ON \
  -DCUDAToolkit_ROOT=/path/to/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<compute-capability-without-dot> \
  -DICICLE_V4_INSTALL_DIR=/path/to/icicle-v4 \
  -DCMAKE_C_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-cc.sh \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-c++.sh \
  -DCMAKE_AR=/absolute/path/to/barretenberg/cpp/scripts/zig-ar.sh \
  -DCMAKE_RANLIB=/absolute/path/to/barretenberg/cpp/scripts/zig-ranlib.sh
```

For example, use `-DCMAKE_CUDA_ARCHITECTURES=120` for compute capability 12.0.
For CPU baselines collected on the target host, consider adding
`-DTARGET_ARCH=native`; record that setting with the benchmark output because it
can change CPU timings.

Build the benchmark targets:

```bash
PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake --build /tmp/aztec-bb-gpu-msm-bench \
  --target gpu_msm_external_cpu_bench \
           gpu_msm_external_bb_bench \
           gpu_msm_external_icicle_v4_bench -- -j16
```

On a CPU-only machine, build only the CPU MSM runner without enabling the native
GPU backend or requiring CUDA:

```bash
PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake \
  -S barretenberg/cpp \
  -B /tmp/aztec-bb-cpu-msm-bench \
  -G "Unix Makefiles" \
  -DMOBILE=ON \
  -DAVM=OFF \
  -DENABLE_HEAVY_TESTS=OFF \
  -DGPU_BACKEND=none \
  -DBB_ENABLE_GPU_MSM_EXTERNAL_BENCH=ON \
  -DCMAKE_C_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-cc.sh \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/barretenberg/cpp/scripts/zig-c++.sh \
  -DCMAKE_AR=/absolute/path/to/barretenberg/cpp/scripts/zig-ar.sh \
  -DCMAKE_RANLIB=/absolute/path/to/barretenberg/cpp/scripts/zig-ranlib.sh

PATH=/path/to/zig:$PATH \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
/path/to/cmake --build /tmp/aztec-bb-cpu-msm-bench \
  --target gpu_msm_external_cpu_bench -- -j16
```

To compile the BB runner with Icicle's inline BN254 Fq arithmetic in the MSM
curve formulas, configure a separate build directory with:

```bash
-DBB_GPU_MSM_FIELD_BACKEND=icicle \
-DBB_ENABLE_GPU_MSM_EXTERNAL_BENCH=ON \
-DBB_GPU_ICICLE_INCLUDE_DIRS="/path/to/icicle/include;/path/to/icicle/backend/cuda/include"
```

For the proof replay and BB+Icicle field-kernel benchmarks, the full Icicle
source tree is not required. A field-only header artifact is sufficient:

```bash
tar -C /tmp -xzf /path/to/icicle-v4-field-only-headers.tar.gz

-DBB_GPU_MSM_FIELD_BACKEND=icicle \
-DBB_ENABLE_GPU_MSM_EXTERNAL_BENCH=ON \
-DBB_GPU_ICICLE_INCLUDE_DIRS=/tmp/icicle-v4-field-only-headers/include
```

This artifact should contain Icicle field, math, utility, and CUDA helper
headers only. It should not contain `icicle/curves/**`; the BB integration uses
`Field<bn254::fq_config>` directly for BN254 base-field arithmetic.

The default is `-DBB_GPU_MSM_FIELD_BACKEND=bb`.

## Runtime Inputs

The runner needs CUDA runtime libraries and the Icicle libraries visible through
`LD_LIBRARY_PATH`.

Example:

```bash
export LD_LIBRARY_PATH=/path/to/icicle-v4/lib:/path/to/cuda-12.8/lib64:$LD_LIBRARY_PATH
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
  --implementations bb,icicle-v4.0.0 \
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

## CPU/GPU Single MSM Sweep

Add `cpu` to the selected implementations when you want CPU Pippenger rows in
the same raw JSONL files and summary table as the GPU backends:

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench \
  --output-dir /tmp/gpu-msm-external-main-with-cpu \
  --implementations cpu,bb,icicle-v4.0.0 \
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

CPU rows use the same CRS monomial points and generated scalars as the GPU rows.
CPU timing ignores `c` and precompute factor, but emits rows for each requested
value so the wrapper can validate completeness and compare results across
backends.

The CPU runner uses Barretenberg's normal parallel-for thread count:
`HARDWARE_CONCURRENCY` when set, otherwise `min(32, hardware_concurrency)`.
Raw records include `cpu_threads`, and the wrapper summary reports the average.
Both single and batched CPU MSM rows use the unsafe SRS-point path: single calls
`pippenger_unsafe`, and batch calls `batch_multi_scalar_mul(...,
handle_edge_cases=false)`.

## BB+Icicle Field Single MSM Sweep

The BB+Icicle field-kernel variant is still the BB runner binary. The wrapper
therefore emits `implementation=bb`; keep it in a separate output directory and
record that the build directory was configured with
`BB_GPU_MSM_FIELD_BACKEND=icicle`.

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench-icicle-field \
  --output-dir /tmp/gpu-msm-external-main-bb-icicle-field \
  --implementations bb \
  --mode single \
  --memory-placement host \
  --min-log 10 \
  --max-log 24 \
  --log-step 2 \
  --factors 1,4,8 \
  --repeats 5 \
  --c 0
```

## Batch-100 Cost-Analysis Shape

This run matches the "100 random columns of length 2^20" shape discussed for
the cost report. It uses precompute factor 1 and auto `c`. The host-placement
version includes scalar host-to-device transfer inside the backend call.
Pass `--batch-log 20` explicitly; the wrapper default is not the cost-report
shape.

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

The BB API defaults to a maximum fused batch size of 16, so it chunks this
shape internally as `16+16+16+16+16+16+4`. Lower this only for experiments:

```bash
--bb-max-fused-batch-size 8
```

On a 16 GB local GPU, the flat `2^20 x 100` BB fused path did not fit; its
preflight estimated about 70.5 GB of transient allocation. Icicle v4.x did
complete locally.

## Device-Resident Batch-100 Shape

Use this run to measure backend execution after scalar inputs have already been
uploaded and results can stay on device until after timing.
Pass `--batch-log 20` explicitly here as well so the host and device rows use
the same `2^20 x 100` shape.

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

## BB+Icicle Field Batch-100 Shapes

Run the same batch shapes from the BB+Icicle field build directory. As with the
single-MSM sweep, the wrapper records these rows as `implementation=bb`, so keep
the output directories distinct.

Host scalars:

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench-icicle-field \
  --output-dir /tmp/gpu-msm-external-batch-2p20x100-bb-icicle-field \
  --implementations bb \
  --mode batch \
  --memory-placement host \
  --batch-log 20 \
  --batch-size 100 \
  --factors 1 \
  --repeats 5 \
  --c 0
```

Device-resident scalars:

```bash
python3 barretenberg/gpu/benchmark/external_msm/run_external_msm_comparison.py \
  --build-dir /tmp/aztec-bb-gpu-msm-bench-icicle-field \
  --output-dir /tmp/gpu-msm-external-batch-2p20x100-device-bb-icicle-field \
  --implementations bb \
  --mode batch \
  --memory-placement device \
  --batch-log 20 \
  --batch-size 100 \
  --factors 1 \
  --repeats 5 \
  --c 0
```

## MSM Cost Report Tables

Use `summary.json` as the source for cost-report tables. The timing value is
`comparison_ms_avg`, which is `device_ms` when the backend reports CUDA event
time and otherwise `backend_wall_ms`. Point/SRS setup and precompute setup are
not included.

For the single-MSM sweep, report one row per backend and MSM size. If several
precompute factors were run for the same backend and size, use the row with the
lowest `comparison_ms_avg` and report that row's resolved `c` and precompute
factor.

| Backend | c | Precompute Factor | MSM Size | Best Avg (ms) | MSMs/$ |
|---|---:|---:|---:|---:|---:|

Single-MSM cost formula:

```text
MSMs/$ = (3600 / time_s) / cost_per_h
time_s = Best Avg (ms) / 1000
```

For the batch-100 runs, report one row per backend and scalar location. Scalar
location is the `memory_placement` value from `summary.json`: `host` or
`device`. The `Avg (s)` column is the average time for the whole 100-column
batch, not the per-MSM time.

| Backend | Scalar Location | c | Avg (s) | MSMs/$ |
|---|---|---:|---:|---:|

Batch-100 cost formula:

```text
MSMs/$ = 100 * (3600 / time_s) / cost_per_h
time_s = Avg (s)
```

This helper prints either layout from one or more output directories. Pass the
BB+Icicle field build output with a label containing `ICICLE_FIELD` so the `bb`
rows are displayed as `BB + Icicle`.

```bash
COST_PER_H=<instance-cost-per-hour> \
REPORT=single \
node - MAIN=/tmp/gpu-msm-external-main-with-cpu \
       BB_ICICLE_FIELD=/tmp/gpu-msm-external-main-bb-icicle-field <<'NODE'
const fs = require('fs');
const path = require('path');

const costPerHour = Number(process.env.COST_PER_H);
const report = process.env.REPORT || 'single';
if (!Number.isFinite(costPerHour) || costPerHour <= 0) {
  throw new Error('set COST_PER_H to the instance cost per hour');
}

function backendName(label, implementation) {
  if (implementation === 'cpu') return 'BB CPU';
  if (implementation === 'bb' && /ICICLE_FIELD/i.test(label)) return 'BB + Icicle';
  if (implementation === 'bb') return 'BB GPU';
  if (implementation === 'icicle-v4.0.0') return 'Icicle v4.0.0';
  return implementation;
}

function loadRows() {
  return process.argv.slice(2).flatMap(arg => {
    const [label, dir] = arg.split('=');
    if (!label || !dir) throw new Error(`expected LABEL=/path, got ${arg}`);
    return JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8')).map(row => ({
      ...row,
      backend: backendName(label, row.implementation),
    }));
  });
}

function cValue(row) {
  return row.backend === 'BB CPU' ? 'N/A' : String(Math.round(row.reported_c_avg));
}

function fmtNumber(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function fmtCost(value) {
  return Math.round(value).toLocaleString('en-US');
}

function bestBy(rows, keyOf) {
  const grouped = new Map();
  for (const row of rows) {
    const key = keyOf(row);
    const current = grouped.get(key);
    if (!current || row.comparison_ms_avg < current.comparison_ms_avg) {
      grouped.set(key, row);
    }
  }
  return [...grouped.values()];
}

if (report === 'single') {
  const rows = bestBy(
    loadRows().filter(row => row.mode === 'single'),
    row => `${row.backend}:${row.log_num_points}`,
  ).sort((a, b) => a.backend.localeCompare(b.backend) || a.log_num_points - b.log_num_points);

  console.log('| Backend | c | Precompute Factor | MSM Size | Best Avg (ms) | MSMs/$ |');
  console.log('|---|---:|---:|---:|---:|---:|');
  for (const row of rows) {
    const msmsPerDollar = (3600 / (row.comparison_ms_avg / 1000)) / costPerHour;
    const factor = row.backend === 'BB CPU' ? 'N/A' : row.precompute_factor;
    console.log(`| ${row.backend} | ${cValue(row)} | ${factor} | 2^${row.log_num_points} | ${fmtNumber(row.comparison_ms_avg)} | ${fmtCost(msmsPerDollar)} |`);
  }
} else if (report === 'batch') {
  const batchSize = Number(process.env.BATCH_SIZE || 100);
  const rows = bestBy(
    loadRows().filter(row => row.mode === 'batch' && row.batch_size === batchSize),
    row => `${row.backend}:${row.memory_placement}`,
  ).sort((a, b) => a.backend.localeCompare(b.backend) || a.memory_placement.localeCompare(b.memory_placement));

  console.log('| Backend | Scalar Location | c | Avg (s) | MSMs/$ |');
  console.log('|---|---|---:|---:|---:|');
  for (const row of rows) {
    const avgSeconds = row.comparison_ms_avg / 1000;
    const msmsPerDollar = batchSize * (3600 / avgSeconds) / costPerHour;
    console.log(`| ${row.backend} | ${row.memory_placement} | ${cValue(row)} | ${fmtNumber(avgSeconds)} | ${fmtCost(msmsPerDollar)} |`);
  }
} else {
  throw new Error('REPORT must be single or batch');
}
NODE
```

For the batch table, rerun the same helper with `REPORT=batch`,
`BATCH_SIZE=100`, and the host/device batch output directories:
`HOST=/tmp/gpu-msm-external-batch-2p20x100`,
`DEVICE=/tmp/gpu-msm-external-batch-2p20x100-device`,
`HOST_ICICLE_FIELD=/tmp/gpu-msm-external-batch-2p20x100-bb-icicle-field`, and
`DEVICE_ICICLE_FIELD=/tmp/gpu-msm-external-batch-2p20x100-device-bb-icicle-field`.

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

Do not commit large ad-hoc generated proof stores into the repository. For
production benchmarking, replay a fixed fixture for CPU and GPU runs. This
benchmark includes a small committed proof-store fixture so production machines
can reproduce the same proof inputs directly from git.

The committed sanity fixture is:

```text
barretenberg/gpu/benchmark/proof_generation/fixtures/aztec-gpu-e2e-proof-store-sanity-full.tar.gz
```

with SHA256:

```text
333740f7da89537a2ca21df67a8754f59585298b000b36606707005220a5880e
```

Extract it on the target machine with:

```bash
tar -C /tmp -xzf barretenberg/gpu/benchmark/proof_generation/fixtures/aztec-gpu-e2e-proof-store-sanity-full.tar.gz
```

Then use `file:///tmp/aztec-gpu-e2e-proof-store-sanity-full` as the
`--proof-store` value.

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

### Proof-Store Replay

Use this when you want a broader production-shaped proof benchmark without
timing the full e2e node/sequencer/test harness. The script discovers every
captured proof input under a local file proof store, replays each job
sequentially through `BBNativeRollupProver`, and writes total, per-type, and
per-job timings. Each prover method verifies the generated proof before
returning, so a successful row means the proof was generated and verified.

First inspect the captured fixture:

```bash
node yarn-project/scripts/run_proof_store_replay_bench.mjs \
  --proof-store file:///tmp/aztec-gpu-e2e-proof-store \
  --output-dir /tmp/gpu-proof-store-replay-list \
  --list
```

Then run CPU and GPU replays against the same proof-store fixture:

```bash
node yarn-project/scripts/run_proof_store_replay_bench.mjs \
  --proof-store file:///tmp/aztec-gpu-e2e-proof-store \
  --bb-bin /path/to/cpu/bb \
  --acvm-bin /absolute/path/to/noir/noir-repo/target/release/acvm \
  --output-dir /tmp/cpu-proof-store-replay \
  --warmups 0 \
  --repeats 1
```

```bash
BB_GPU_MSM_PRECOMPUTE_FACTOR=1 \
BB_GPU_MSM_MAX_BATCH_SIZE=1 \
node yarn-project/scripts/run_proof_store_replay_bench.mjs \
  --proof-store file:///tmp/aztec-gpu-e2e-proof-store \
  --bb-bin /path/to/gpu/bb \
  --acvm-bin /absolute/path/to/noir/noir-repo/target/release/acvm \
  --output-dir /tmp/gpu-proof-store-replay \
  --warmups 0 \
  --repeats 1
```

`BB_GPU_MSM_MAX_BATCH_SIZE=1` is only needed on smaller GPUs when the fused
root-rollup batch exceeds available memory. On production hardware, prefer the
default fused path unless memory preflight fails.

If the captured store includes `PUBLIC_VM`, the replay uses `bb avm_prove` for
that job and therefore needs a `bb` binary built with AVM support.
`PUBLIC_TX_BASE_ROLLUP` can also require AVM recursion support. For a
rollup/MSM-focused replay using an `AVM=OFF` proof binary, exclude those jobs:

```bash
--exclude-types PUBLIC_VM,PUBLIC_TX_BASE_ROLLUP
```

The replay script writes:

| File | Contents |
|---|---|
| `metadata.json` | Command, git state, GPU env vars, `nvidia-smi`, discovered proof inputs, and input SHA256 hashes. |
| `raw.jsonl` | Per-job records plus one suite-total record per repeat, including setup, input-load, proof-generation, output, stage-total, and residual-overhead timings. |
| `summary.json` | Aggregated total, per-type, and per-job records with timing splits that sum with overhead to the end-to-end elapsed time. |
| `summary.md` | Human-readable tables for reporting the timing splits. |

This is not a full epoch wall-clock benchmark. It intentionally excludes node
startup, transaction submission, sequencer timing, publication, and prover-agent
scheduling noise. Use the original `e2e_prover/full` command as the full-system
smoke benchmark and the proof-store replay as the production-shaped proving
benchmark.

### CRS-Adjusted Proof Breakdown

Use this reporting view when comparing steady-state CPU and GPU proof time
after CRS/SRS setup. The raw replay timings remain in `raw.jsonl` and
`summary.md`; the adjusted view is derived from those files.

Run CPU and GPU with proof profiling and a persistent BB worker so setup and
GPU SRS prewarm happen before the timed proof loop:

```bash
BB_PROOF_BENCH=1 \
node yarn-project/scripts/run_proof_store_replay_bench.mjs \
  --proof-store file:///tmp/aztec-gpu-e2e-proof-store-sanity-full \
  --bb-bin /path/to/cpu/bb \
  --acvm-bin /absolute/path/to/noir/noir-repo/target/release/acvm \
  --output-dir /tmp/proof-replay-cpu \
  --include-types PARITY_BASE,ROOT_ROLLUP \
  --repeats 3 \
  --persistent-bb-worker
```

```bash
BB_PROOF_BENCH=1 \
BB_GPU_MSM_PRECOMPUTE_FACTOR=1 \
BB_GPU_MSM_MAX_BATCH_SIZE=16 \
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/path/to/cuda-12.8/lib64:$LD_LIBRARY_PATH \
node yarn-project/scripts/run_proof_store_replay_bench.mjs \
  --proof-store file:///tmp/aztec-gpu-e2e-proof-store-sanity-full \
  --bb-bin /path/to/gpu/bb \
  --acvm-bin /absolute/path/to/noir/noir-repo/target/release/acvm \
  --output-dir /tmp/proof-replay-gpu \
  --include-types PARITY_BASE,ROOT_ROLLUP \
  --repeats 3 \
  --persistent-bb-worker
```

For the full non-AVM proof replay set used in the GPU MSM proof tables, use:

```bash
--include-types PUBLIC_CHONK_VERIFIER,PARITY_BASE,PARITY_ROOT,PRIVATE_TX_BASE_ROLLUP,BLOCK_ROOT_SINGLE_TX_FIRST_ROLLUP,CHECKPOINT_ROOT_SINGLE_BLOCK_ROLLUP,ROOT_ROLLUP
```

The adjusted proof time is:

```text
adjusted proof time =
  elapsedMs
  - sum(CRS::* native timers)
  - sum(GPU::srs_upload native timers)
```

The compact additive columns are derived as:

| Column | Source |
|---|---|
| `adjusted proof ms` | `elapsedMs - CRS::* - GPU::srs_upload` |
| `witgen ms` | `witnessGenerationMs` |
| `commitments/MSM ms` | Sum of all `CommitmentKey::commit` and `CommitmentKey::batch_commit` entries in `bb-bench-hierarchical.json` |
| `sumcheck ms` | `bbAdditiveStagesMs.sumcheckMs` |
| `pcs ms` | `bbAdditiveStagesMs.pcsMs` |
| `prover core ms` | Adjusted `UltraHonkAPI::prove` time minus `commitments/MSM`, `sumcheck`, and `pcs` |
| `verify/harness/overhead ms` | Remainder needed for the compact columns to sum to adjusted proof time |

Use the saved hierarchical profiles rather than `summary.md`'s
`direct commitments ms` field for `commitments/MSM`; some commitment timers are
nested under Oink and circuit-construction parents and must be counted from the
native profile directly.

Generate the compact adjusted table from two replay output directories:

```bash
node - CPU=/tmp/proof-replay-cpu GPU=/tmp/proof-replay-gpu <<'NODE'
const fs = require('node:fs');
const path = require('node:path');

const inputs = process.argv.slice(2).map(arg => {
  const [label, dir] = arg.split('=');
  if (!label || !dir) {
    throw new Error(`Expected LABEL=/path/to/output, got ${arg}`);
  }
  return { label, dir };
});

function ms(ns) {
  return (ns ?? 0) / 1_000_000;
}

function benchEntryMs(entry) {
  return ms(entry.time_max ?? entry.time);
}

function readRecords(dir) {
  return fs
    .readFileSync(path.join(dir, 'raw.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(line => JSON.parse(line))
    .filter(record => record.benchmark === 'proof-store-replay-job' && !record.warmup);
}

function readBench(record) {
  const profile = record.nativeProofProfiles?.[0];
  if (!profile?.bbBenchPath) {
    throw new Error(`Missing bbBenchPath for ${record.proofType}`);
  }
  return JSON.parse(fs.readFileSync(profile.bbBenchPath, 'utf8'));
}

function sumBench(bench, pred) {
  let total = 0;
  for (const [name, entries] of Object.entries(bench)) {
    if (!pred(name)) {
      continue;
    }
    for (const entry of entries) {
      total += benchEntryMs(entry);
    }
  }
  return total;
}

function summarizeRecord(label, record) {
  const bench = readBench(record);
  const crs = sumBench(bench, name => name.startsWith('CRS::'));
  const gpuSrs = sumBench(bench, name => name === 'GPU::srs_upload');
  const commitments = sumBench(bench, name => name === 'CommitmentKey::commit' || name === 'CommitmentKey::batch_commit');
  const adjustedApi = record.bbAdditiveStagesMs.ultraHonkApiProveMs - crs - gpuSrs;
  const sumcheck = record.bbAdditiveStagesMs.sumcheckMs;
  const pcs = record.bbAdditiveStagesMs.pcsMs;
  const proverCore = adjustedApi - commitments - sumcheck - pcs;
  const adjustedProof = record.elapsedMs - crs - gpuSrs;
  const verifyHarnessOverhead = adjustedProof - record.witnessGenerationMs - proverCore - commitments - sumcheck - pcs;
  return {
    proofType: record.proofType,
    backend: label,
    adjustedProof,
    witgen: record.witnessGenerationMs,
    proverCore,
    commitments,
    sumcheck,
    pcs,
    verifyHarnessOverhead,
  };
}

function average(rows) {
  const result = { proofType: rows[0].proofType, backend: rows[0].backend, samples: rows.length };
  for (const key of ['adjustedProof', 'witgen', 'proverCore', 'commitments', 'sumcheck', 'pcs', 'verifyHarnessOverhead']) {
    result[key] = rows.reduce((sum, row) => sum + row[key], 0) / rows.length;
  }
  result.additiveTotal =
    result.witgen + result.proverCore + result.commitments + result.sumcheck + result.pcs + result.verifyHarnessOverhead;
  return result;
}

const grouped = new Map();
for (const input of inputs) {
  for (const record of readRecords(input.dir)) {
    const row = summarizeRecord(input.label, record);
    const key = `${row.proofType}:${row.backend}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  }
}

const rows = [...grouped.values()].map(average).sort((a, b) => a.proofType.localeCompare(b.proofType) || a.backend.localeCompare(b.backend));
const fmt = value => value.toFixed(3);

console.log('| proof | backend | samples | adjusted proof ms | witgen ms | prover core ms | commitments/MSM ms | sumcheck ms | pcs ms | verify/harness/overhead ms | additive total ms |');
console.log('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|');
for (const row of rows) {
  console.log(
    `| ${row.proofType} | ${row.backend} | ${row.samples} | ${fmt(row.adjustedProof)} | ${fmt(row.witgen)} | ${fmt(row.proverCore)} | ${fmt(row.commitments)} | ${fmt(row.sumcheck)} | ${fmt(row.pcs)} | ${fmt(row.verifyHarnessOverhead)} | ${fmt(row.additiveTotal)} |`,
  );
}
NODE
```

The `additive total ms` column should match `adjusted proof ms` up to rounding.
If it does not, inspect the run's `raw.jsonl` and referenced
`bb-bench-hierarchical.json` files before reporting the table.

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
| `device_ms` | CUDA-event/profiled device time when available. BB exposes this; Icicle v4 does not. |
| `comparison_ms` | `device_ms` when present, otherwise `backend_wall_ms`. |
| `per_msm_ms` | `comparison_ms / batch_size`. |
| `c` | Backend-reported or resolved c value. `requested_c` is added by the wrapper. |
| `cpu_threads` | Effective Barretenberg CPU thread count for CPU rows. Zero for GPU rows. |

For CPU rows, `backend_wall_ms` is the timed CPU Pippenger region,
`comparison_ms` equals `backend_wall_ms`, and CUDA/precompute fields are empty
or zero.

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
