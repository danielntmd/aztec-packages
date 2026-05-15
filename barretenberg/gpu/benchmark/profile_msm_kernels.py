#!/usr/bin/env python3

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path


BACKENDS = {
    "bb-jacobian": {
        "executable": "gpu_msm_baseline_bench",
        "benchmark": "BN254/Baseline/BB_GPU/Jacobian/E2E",
        "ncu_kernel": "regex:.*(accumulate_normal_buckets_kernel|reduce_bucket_bit_kernel|compose_window_sums_kernel|final_accumulation_kernel).*",
    },
    "bb-xyzz": {
        "executable": "gpu_msm_baseline_bench",
        "benchmark": "BN254/Baseline/BB_GPU/XYZZ/E2E",
        "ncu_kernel": "regex:.*(accumulate_normal_buckets_xyzz_kernel|reduce_xyzz_bucket_bit_kernel|compose_xyzz_window_sums_kernel|final_xyzz_accumulation_kernel).*",
    },
    "icicle-v28": {
        "executable": "gpu_msm_baseline_icicle_v28_bench",
        "benchmark": "BN254/Baseline/IcicleV28/DeviceResident",
        "ncu_kernel": "regex:.*",
    },
    "icicle-v4": {
        "executable": "gpu_msm_baseline_icicle_v4_bench",
        "benchmark": "BN254/Baseline/IcicleV4/DeviceResident",
        "ncu_kernel": "regex:.*",
    },
}

DEFAULT_ICICLE_LIBRARY_PATHS = [
    "/tmp/cuda-12.8/lib64",
    "/tmp/cuda-12.8/targets/x86_64-linux/lib",
    "/tmp/cuda-12.8/extras/CUPTI/lib64",
    "/tmp/icicle-v4.0.0/icicle/lib",
    "/tmp/icicle-v4.0.0/icicle/lib/backend/cuda",
    "/tmp/icicle-v4.0.0/icicle/lib/backend/bn254/cuda",
]


def find_executable(build_dir: Path, name: str) -> Path:
    for candidate in (
        build_dir / "gpu" / name,
        build_dir / "bin" / name,
        build_dir / name,
    ):
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = list(build_dir.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {build_dir}")


def find_tool(explicit: str | None, names: list[str], fallbacks: list[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"Requested tool does not exist: {path}")
    for fallback in fallbacks:
        path = Path(fallback)
        if path.exists():
            return path
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    raise FileNotFoundError(f"Could not find any of: {', '.join(names)}")


def run(command: list[str], env: dict[str, str], cwd: Path, stdout_path: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    if stdout_path is None:
        completed = subprocess.run(command, env=env, cwd=cwd, text=True, check=False)
    else:
        with stdout_path.open("w", encoding="utf-8") as stdout_file:
            completed = subprocess.run(command, env=env, cwd=cwd, text=True, stdout=stdout_file, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def benchmark_command(executable: Path, benchmark_name: str, size: int, args: argparse.Namespace) -> list[str]:
    return [
        str(executable),
        f"--benchmark_filter=^{benchmark_name}/{size}$",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup}",
    ]


def profiling_env(skip_correctness: bool) -> dict[str, str]:
    env = os.environ.copy()
    if skip_correctness:
        env["MSM_BENCH_SKIP_CORRECTNESS"] = "1"

    library_paths = [path for path in DEFAULT_ICICLE_LIBRARY_PATHS if Path(path).exists()]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    if library_paths:
        env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    return env


def validate_backend(backend: str, executable: Path, size: int, args: argparse.Namespace, output_dir: Path) -> None:
    benchmark_name = BACKENDS[backend]["benchmark"]
    command = benchmark_command(executable, benchmark_name, size, args)
    command.append("--benchmark_format=json")
    run(command, profiling_env(skip_correctness=False), Path.cwd(), output_dir / f"validate_{backend}_2p{size}.json")


def run_nsys(backend: str, executable: Path, size: int, args: argparse.Namespace, output_dir: Path, nsys: Path) -> None:
    benchmark_name = BACKENDS[backend]["benchmark"]
    output_base = output_dir / f"nsys_{backend}_2p{size}"
    command = [
        str(nsys),
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cuda-memory-usage=true",
        "--output",
        str(output_base),
        *benchmark_command(executable, benchmark_name, size, args),
    ]
    run(command, profiling_env(skip_correctness=True), Path.cwd(), output_dir / f"{output_base.name}.log")

    report = output_base.with_suffix(".nsys-rep")
    if report.exists():
        stats_command = [
            str(nsys),
            "stats",
            "--force-overwrite=true",
            "--report",
            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,nvtx_kern_sum",
            "--format",
            "csv",
            str(report),
        ]
        run(stats_command, profiling_env(skip_correctness=True), Path.cwd(), output_dir / f"{output_base.name}_stats.csv")


def run_ncu(backend: str, executable: Path, size: int, args: argparse.Namespace, output_dir: Path, ncu: Path) -> None:
    benchmark_name = BACKENDS[backend]["benchmark"]
    output_base = output_dir / f"ncu_{backend}_2p{size}"
    kernel_name = args.ncu_kernel_name or BACKENDS[backend]["ncu_kernel"]
    command = [
        str(ncu),
        "--target-processes",
        "all",
        "--set",
        args.ncu_set,
        "--kernel-name",
        kernel_name,
        "--launch-count",
        str(args.ncu_launch_count),
        "--force-overwrite",
        "--export",
        str(output_base),
        *benchmark_command(executable, benchmark_name, size, args),
    ]
    run(command, profiling_env(skip_correctness=True), Path.cwd(), output_dir / f"{output_base.name}.log")


def write_manifest(args: argparse.Namespace, output_dir: Path, backends: list[str], nsys: Path | None, ncu: Path | None) -> None:
    nsys_version = command_output([str(nsys), "--version"]) if nsys else ""
    ncu_version = command_output([str(ncu), "--version"]) if ncu else ""
    manifest = output_dir / "README.txt"
    lines = [
        "GPU MSM profiling run",
        f"created_utc={dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"git_branch={command_output(['git', 'branch', '--show-current'])}",
        f"git_commit={command_output(['git', 'rev-parse', '--short', 'HEAD'])}",
        f"gpu={command_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])}",
        f"build_dir={args.build_dir}",
        f"backends={','.join(backends)}",
        f"sizes={','.join(str(size) for size in args.sizes)}",
        f"tool={args.tool}",
        f"min_time={args.min_time}",
        f"warmup={args.warmup}",
        f"validate={not args.no_validate}",
        f"nsys={nsys if nsys else ''}",
        f"nsys_version={nsys_version}",
        f"ncu={ncu if ncu else ''}",
        f"ncu_version={ncu_version}",
        "",
        "Traced benchmark runs set MSM_BENCH_SKIP_CORRECTNESS=1 after the validation pass.",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile BN254 GPU MSM baseline kernels with Nsight tools.")
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/aztec-bb-gpu-build-zig-128b"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backends", nargs="+", choices=[*BACKENDS.keys(), "all"], default=["all"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 18, 20, 24])
    parser.add_argument("--tool", choices=("nsys", "ncu", "both", "benchmark"), default="nsys")
    parser.add_argument("--min-time", default="1x")
    parser.add_argument("--warmup", default="0")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--nsys", help="Path to nsys")
    parser.add_argument("--ncu", help="Path to ncu")
    parser.add_argument("--ncu-set", default="basic")
    parser.add_argument("--ncu-kernel-name", help="Override the Nsight Compute kernel-name selector.")
    parser.add_argument("--ncu-launch-count", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_backends = list(BACKENDS) if "all" in args.backends else args.backends
    output_dir = args.output_dir or Path("/tmp/aztec-msm-profiles") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    nsys = None
    ncu = None
    if args.tool in ("nsys", "both"):
        nsys = find_tool(args.nsys, ["nsys"], ["/tmp/cuda-12.8/bin/nsys"])
    if args.tool in ("ncu", "both"):
        ncu = find_tool(
            args.ncu,
            ["ncu", "nv-nsight-cu-cli"],
            [
                "/tmp/nsight-compute-2025.3.1/opt/nvidia/nsight-compute/2025.3.1/ncu",
                "/tmp/cuda-12.8/bin/ncu",
                "/usr/bin/ncu",
            ],
        )

    write_manifest(args, output_dir, selected_backends, nsys, ncu)

    executables = {
        backend: find_executable(args.build_dir, BACKENDS[backend]["executable"]) for backend in selected_backends
    }

    if not args.no_validate:
        validation_size = max(args.sizes)
        for backend, executable in executables.items():
            validate_backend(backend, executable, validation_size, args, output_dir)

    for backend, executable in executables.items():
        for size in args.sizes:
            if args.tool == "benchmark":
                command = benchmark_command(executable, BACKENDS[backend]["benchmark"], size, args)
                command.append("--benchmark_format=json")
                run(command, profiling_env(skip_correctness=True), Path.cwd(), output_dir / f"benchmark_{backend}_2p{size}.json")
            if args.tool in ("nsys", "both"):
                assert nsys is not None
                run_nsys(backend, executable, size, args, output_dir, nsys)
            if args.tool in ("ncu", "both"):
                assert ncu is not None
                run_ncu(backend, executable, size, args, output_dir, ncu)

    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
