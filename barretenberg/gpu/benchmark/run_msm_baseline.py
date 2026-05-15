#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


EXECUTABLES = {
    "bb": "gpu_msm_baseline_bench",
    "icicle_v28": "gpu_msm_baseline_icicle_v28_bench",
    "icicle_v4": "gpu_msm_baseline_icicle_v4_bench",
}

BENCHMARKS = {
    "bb": [
        "BN254/Baseline/CPU",
        "BN254/Baseline/BB_GPU/Jacobian/E2E",
        "BN254/Baseline/BB_GPU/XYZZ/E2E",
    ],
    "icicle_v28": [
        "BN254/Baseline/IcicleV28/E2E",
        "BN254/Baseline/IcicleV28/PreparedHost",
        "BN254/Baseline/IcicleV28/DeviceResident",
    ],
    "icicle_v4": [
        "BN254/Baseline/IcicleV4/E2E",
        "BN254/Baseline/IcicleV4/PreparedHost",
        "BN254/Baseline/IcicleV4/DeviceResident",
    ],
}

BACKENDS = {
    "CPU": "cpu_ms",
    "BB_GPU_Jacobian": "bb_jac",
    "BB_GPU_XYZZ": "bb_xyzz",
    "IcicleV28": "v28",
    "IcicleV4": "v4",
}

PHASE_PREFIXES = {
    "BB_GPU_Jacobian": "bb_jac",
    "BB_GPU_XYZZ": "bb_xyzz",
    "IcicleV28": "v28",
    "IcicleV4": "v4",
}

MODES = {
    "E2E": "e2e",
    "PreparedHost": "prepared_host",
    "DeviceResident": "device",
}


def find_executable(build_dir: Path, name: str) -> Path | None:
    for candidate in (
        build_dir / "gpu" / name,
        build_dir / "bin" / name,
        build_dir / name,
    ):
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


def backend_from_name(name: str) -> str | None:
    if "/BB_GPU/Jacobian/" in name:
        return "BB_GPU_Jacobian"
    if "/BB_GPU/XYZZ/" in name:
        return "BB_GPU_XYZZ"
    for backend in BACKENDS:
        if f"/{backend}" in name:
            return backend
    return None


def mode_from_name(name: str, backend: str) -> str:
    if backend == "CPU":
        return "cpu"
    for mode, key in MODES.items():
        if f"/{mode}/" in name:
            return key
    return "e2e"


def log_size_from_name(name: str) -> int | None:
    suffix = name.rsplit("/", 1)[-1]
    if suffix.endswith("_median") or suffix.endswith("_mean") or suffix.endswith("_stddev"):
        suffix = suffix.rsplit("_", 1)[0]
    return int(suffix) if suffix.isdigit() else None


def selected_benchmarks(benchmarks: list[dict], stat: str) -> list[dict]:
    suffix = f"_{stat}"
    selected = [entry for entry in benchmarks if entry["name"].endswith(suffix)]
    if selected:
        return selected
    medians = [entry for entry in benchmarks if entry["name"].endswith("_median")]
    if medians:
        return medians
    return [
        entry
        for entry in benchmarks
        if not entry["name"].endswith("_mean")
        and not entry["name"].endswith("_median")
        and not entry["name"].endswith("_stddev")
        and not entry["name"].endswith("_cv")
    ]


def run_benchmark(
    executable: Path,
    args: argparse.Namespace,
    env_overrides: dict[str, str] | None = None,
    benchmark_filter: str = "BN254/Baseline",
) -> dict[int, dict[str, float]]:
    command = [
        str(executable),
        f"--benchmark_filter={benchmark_filter}",
        "--benchmark_format=json",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup}",
    ]
    if args.repetitions > 1:
        command.append(f"--benchmark_repetitions={args.repetitions}")

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"{executable.name} failed with exit code {completed.returncode}")

    payload = json.loads(completed.stdout)
    rows: dict[int, dict[str, float]] = {}
    for entry in selected_benchmarks(payload.get("benchmarks", []), args.stat):
        backend = backend_from_name(entry["name"])
        log_size = log_size_from_name(entry["name"])
        if backend is None or log_size is None:
            continue
        mode = mode_from_name(entry["name"], backend)
        row = rows.setdefault(log_size, {})
        if backend == "CPU":
            row[BACKENDS[backend]] = to_ms(float(entry["real_time"]), entry.get("time_unit", "ns"))
        else:
            row[f"{BACKENDS[backend]}_{mode}_ms"] = to_ms(float(entry["real_time"]), entry.get("time_unit", "ns"))
        if "c" in entry and backend != "CPU":
            row[f"{BACKENDS[backend]}_c"] = int(round(float(entry["c"])))
        if backend in PHASE_PREFIXES:
            prefix = PHASE_PREFIXES[backend]
            for phase in ("preprocess_ms", "backend_ms", "postprocess_ms", "setup_h2d_ms"):
                if phase in entry:
                    row[f"{prefix}_{mode}_{phase}"] = float(entry[phase])
    return rows


def run_isolated_benchmarks(
    executables: dict[str, Path], args: argparse.Namespace, env_by_label: dict[str, dict[str, str] | None]
) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for label, executable in executables.items():
        for benchmark_name in BENCHMARKS[label]:
            for log_size in args.log_sizes:
                benchmark_filter = f"^{benchmark_name}/{log_size}$"
                result = run_benchmark(executable, args, env_by_label[label], benchmark_filter)
                merge_rows(rows, result)
    return rows


def merge_rows(target: dict[int, dict[str, float]], source: dict[int, dict[str, float]]) -> None:
    for log_size, source_row in source.items():
        target.setdefault(log_size, {}).update(source_row)


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:9.3f}"


def fmt_ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:7.2f}x"


def print_table(rows: dict[int, dict[str, float]], enabled: list[str]) -> None:
    print("Enabled benchmark binaries: " + ", ".join(enabled))
    print()
    headers = [
        "n",
        "bb_c",
        "v28_c",
        "v4_c",
        "cpu_ms",
        "bb_jac_e2e",
        "bb_jac_backend",
        "bb_xyzz_e2e",
        "bb_xyzz_backend",
        "v28_e2e",
        "v28_host",
        "v28_device",
        "v4_e2e",
        "v4_host",
        "v4_device",
        "bbjac/cpu",
        "bbxyzz/cpu",
        "v28dev/bbjac",
        "v28dev/bbxyzz",
        "v4dev/bbjac",
        "v4dev/bbxyzz",
    ]
    print(" ".join(f"{header:>14}" for header in headers))
    print(" ".join("-" * 14 for _ in headers))
    for log_size in sorted(rows):
        row = rows[log_size]
        cpu = row.get("cpu_ms")
        bb_jac = row.get("bb_jac_e2e_ms")
        bb_jac_backend = row.get("bb_jac_e2e_backend_ms")
        bb_xyzz = row.get("bb_xyzz_e2e_ms")
        bb_xyzz_backend = row.get("bb_xyzz_e2e_backend_ms")
        v28_device = row.get("v28_device_ms")
        v4_device = row.get("v4_device_ms")
        values = [
            f"2^{log_size}",
            str(int(row["bb_jac_c"])) if "bb_jac_c" in row else "-",
            str(int(row["v28_c"])) if "v28_c" in row else "-",
            str(int(row["v4_c"])) if "v4_c" in row else "-",
            fmt_ms(cpu),
            fmt_ms(bb_jac),
            fmt_ms(bb_jac_backend),
            fmt_ms(bb_xyzz),
            fmt_ms(bb_xyzz_backend),
            fmt_ms(row.get("v28_e2e_ms")),
            fmt_ms(row.get("v28_prepared_host_ms")),
            fmt_ms(v28_device),
            fmt_ms(row.get("v4_e2e_ms")),
            fmt_ms(row.get("v4_prepared_host_ms")),
            fmt_ms(v4_device),
            fmt_ratio(cpu / bb_jac if cpu and bb_jac else None),
            fmt_ratio(cpu / bb_xyzz if cpu and bb_xyzz else None),
            fmt_ratio(bb_jac_backend / v28_device if bb_jac_backend and v28_device else None),
            fmt_ratio(bb_xyzz_backend / v28_device if bb_xyzz_backend and v28_device else None),
            fmt_ratio(bb_jac_backend / v4_device if bb_jac_backend and v4_device else None),
            fmt_ratio(bb_xyzz_backend / v4_device if bb_xyzz_backend and v4_device else None),
        ]
        print(" ".join(f"{value:>14}" for value in values))

    print()
    phase_headers = [
        "n",
        "bb_jac_pre",
        "bb_jac_backend",
        "bb_xyzz_pre",
        "bb_xyzz_backend",
        "v28_e2e_pre",
        "v28_e2e_api",
        "v28_host_api",
        "v28_device_api",
        "v4_e2e_pre",
        "v4_e2e_api",
        "v4_host_api",
        "v4_device_api",
    ]
    print(" ".join(f"{header:>14}" for header in phase_headers))
    print(" ".join("-" * 14 for _ in phase_headers))
    for log_size in sorted(rows):
        row = rows[log_size]
        values = [
            f"2^{log_size}",
            fmt_ms(row.get("bb_jac_e2e_preprocess_ms")),
            fmt_ms(row.get("bb_jac_e2e_backend_ms")),
            fmt_ms(row.get("bb_xyzz_e2e_preprocess_ms")),
            fmt_ms(row.get("bb_xyzz_e2e_backend_ms")),
            fmt_ms(row.get("v28_e2e_preprocess_ms")),
            fmt_ms(row.get("v28_e2e_backend_ms")),
            fmt_ms(row.get("v28_prepared_host_backend_ms")),
            fmt_ms(row.get("v28_device_backend_ms")),
            fmt_ms(row.get("v4_e2e_preprocess_ms")),
            fmt_ms(row.get("v4_e2e_backend_ms")),
            fmt_ms(row.get("v4_prepared_host_backend_ms")),
            fmt_ms(row.get("v4_device_backend_ms")),
        ]
        print(" ".join(f"{value:>14}" for value in values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BN254 MSM baseline benchmark suite.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--min-time", default="1x")
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--stat", choices=("mean", "median"), default="median")
    parser.add_argument("--icicle-auto-c", action="store_true", help="Run Icicle backends with c=0 auto selection.")
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run each benchmark mode and size in a fresh subprocess before aggregating results.",
    )
    parser.add_argument("--log-sizes", nargs="+", type=int, default=list(range(10, 25, 2)))
    args = parser.parse_args()

    rows: dict[int, dict[str, float]] = {}
    enabled: list[str] = []
    executables: dict[str, Path] = {}
    env_by_label: dict[str, dict[str, str] | None] = {}
    for label, executable_name in EXECUTABLES.items():
        executable = find_executable(args.build_dir, executable_name)
        if executable is None:
            continue
        env_overrides = { "MSM_BENCH_C_OVERRIDE": "0" } if args.icicle_auto_c and label.startswith("icicle") else None
        enabled.append(f"{label}:{executable}" + ("(c=auto)" if env_overrides else ""))
        executables[label] = executable
        env_by_label[label] = env_overrides

    if args.isolated:
        rows = run_isolated_benchmarks(executables, args, env_by_label)
    else:
        for label, executable in executables.items():
            merge_rows(rows, run_benchmark(executable, args, env_by_label[label]))

    if not rows:
        print("No MSM baseline benchmark binaries were found.", file=sys.stderr)
        return 1

    print_table(rows, enabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
