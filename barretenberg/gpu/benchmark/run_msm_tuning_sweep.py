#!/usr/bin/env python3

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path


EXECUTABLES = {
    "bb": ("gpu_msm_baseline_bench", "BN254/Baseline/BB_GPU/E2E"),
    "v28": ("gpu_msm_baseline_icicle_v28_bench", "BN254/Baseline/IcicleV28/DeviceResident"),
    "v4": ("gpu_msm_baseline_icicle_v4_bench", "BN254/Baseline/IcicleV4/DeviceResident"),
}


def find_executable(build_dir: Path, name: str) -> Path | None:
    for candidate in (build_dir / "gpu" / name, build_dir / "bin" / name, build_dir / name):
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = list(build_dir.rglob(name))
    return matches[0] if matches else None


def to_ms(value: float, unit: str) -> float:
    if unit == "ms":
        return value
    if unit == "us":
        return value / 1_000.0
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "s":
        return value * 1_000.0
    raise ValueError(f"unsupported benchmark time unit: {unit}")


def run_one(
    executable: Path,
    benchmark_name: str,
    log_size: int,
    env_overrides: dict[str, str],
    args: argparse.Namespace,
) -> tuple[float, float | None]:
    env = os.environ.copy()
    env.update(env_overrides)
    command = [
        str(executable),
        f"--benchmark_filter=^{benchmark_name}/{log_size}$",
        "--benchmark_format=json",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup}",
    ]
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"{executable.name} failed for {benchmark_name}/{log_size} with {env_overrides}")

    payload = json.loads(completed.stdout)
    benchmarks = payload.get("benchmarks", [])
    if not benchmarks:
        raise RuntimeError(f"{executable.name} produced no benchmark rows for {benchmark_name}/{log_size}")
    entry = benchmarks[0]
    real_ms = to_ms(float(entry["real_time"]), entry.get("time_unit", "ns"))
    backend_ms = float(entry["backend_ms"]) if "backend_ms" in entry else None
    return real_ms, backend_ms


def candidate_cs(auto_c: int, radius: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted(set(explicit))
    return [c for c in range(max(1, auto_c - radius), auto_c + radius + 1)]


def auto_c(log_size: int) -> int:
    num_points = 1 << log_size
    if num_points <= (1 << 10):
        return 6
    if num_points <= (1 << 12):
        return 8
    if num_points <= (1 << 14):
        return 9
    if num_points <= (1 << 16):
        return 11
    if num_points <= (1 << 18):
        return 13
    return 15


def configs_for_backend(backend: str, c_values: list[int], args: argparse.Namespace) -> list[dict[str, int]]:
    if backend == "bb":
        return [{ "c": c } for c in c_values]
    if backend == "v28":
        return [
            { "c": c, "big_triangle": big, "large_bucket_factor": large }
            for c, big, large in itertools.product(c_values, args.big_triangle, args.large_bucket_factor)
        ]
    if backend == "v4":
        return [
            { "c": c, "big_triangle": big, "large_bucket_factor": large, "nof_chunks": chunks }
            for c, big, large, chunks in itertools.product(
                c_values, args.big_triangle, args.large_bucket_factor, args.nof_chunks
            )
        ]
    raise ValueError(f"unknown backend: {backend}")


def env_for_config(config: dict[str, int]) -> dict[str, str]:
    env = { "MSM_BENCH_C_OVERRIDE": str(config["c"]) }
    if "big_triangle" in config:
        env["ICICLE_MSM_BIG_TRIANGLE"] = str(config["big_triangle"])
    if "large_bucket_factor" in config:
        env["ICICLE_MSM_LARGE_BUCKET_FACTOR"] = str(config["large_bucket_factor"])
    if "nof_chunks" in config:
        env["ICICLE_MSM_NOF_CHUNKS"] = str(config["nof_chunks"])
    return env


def config_label(config: dict[str, int]) -> str:
    fields = [f"c={config['c']}"]
    if "big_triangle" in config:
        fields.append(f"triangle={config['big_triangle']}")
    if "large_bucket_factor" in config:
        fields.append(f"large={config['large_bucket_factor']}")
    if "nof_chunks" in config:
        fields.append(f"chunks={config['nof_chunks']}")
    return ",".join(fields)


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep BN254 MSM benchmark tuning knobs.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=sorted(EXECUTABLES), default=sorted(EXECUTABLES))
    parser.add_argument("--log-sizes", nargs="+", type=int, default=[20])
    parser.add_argument("--c-values", nargs="+", type=int)
    parser.add_argument("--c-radius", type=int, default=2)
    parser.add_argument("--big-triangle", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--large-bucket-factor", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--nof-chunks", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--min-time", default="1x")
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    for backend in args.backends:
        executable_name, benchmark_name = EXECUTABLES[backend]
        executable = find_executable(args.build_dir, executable_name)
        if executable is None:
            print(f"missing executable for {backend}: {executable_name}", file=sys.stderr)
            return 1

        print(f"\n{backend}: {executable}")
        for log_size in args.log_sizes:
            c_values = candidate_cs(auto_c(log_size), args.c_radius, args.c_values)
            rows = []
            for config in configs_for_backend(backend, c_values, args):
                real_ms, backend_ms = run_one(executable, benchmark_name, log_size, env_for_config(config), args)
                rows.append({ "real_ms": real_ms, "backend_ms": backend_ms, **config })

            rows.sort(key=lambda row: row["backend_ms"] if row["backend_ms"] is not None else row["real_ms"])
            print(f"  n=2^{log_size}, tried={len(rows)}")
            print(f"  {'rank':>4} {'real_ms':>10} {'backend_ms':>10} config")
            for rank, row in enumerate(rows[: args.top], start=1):
                print(f"  {rank:>4} {fmt(row['real_ms']):>10} {fmt(row['backend_ms']):>10} {config_label(row)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
