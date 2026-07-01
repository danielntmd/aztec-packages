#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

RUNNERS = {
    "cpu": "gpu_msm_external_cpu_bench",
    "bb": "gpu_msm_external_bb_bench",
    "icicle-v2.8.0": "gpu_msm_external_icicle_v2_bench",
    "icicle-v4.0.0": "gpu_msm_external_icicle_v4_bench",
}

METADATA_ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "ICICLE_BACKEND_INSTALL_DIR",
    "LD_LIBRARY_PATH",
    "PATH",
]

CMAKE_CACHE_KEYS = [
    "CMAKE_BUILD_TYPE",
    "CMAKE_CUDA_ARCHITECTURES",
    "CMAKE_CUDA_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CUDAToolkit_ROOT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run BB/Icicle GPU MSM comparison benchmarks.")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--implementations", default="bb,icicle-v2.8.0,icicle-v4.0.0")
    parser.add_argument("--mode", default="all", choices=["single", "batch", "all"])
    parser.add_argument("--memory-placement", default="host", choices=["host", "device"])
    parser.add_argument("--min-log", default=10, type=int)
    parser.add_argument("--max-log", default=24, type=int)
    parser.add_argument("--log-step", default=2, type=int)
    parser.add_argument("--factors", default="1,4,8")
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--batch-log", default=21, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--c", default=0, type=int)
    parser.add_argument("--c-values", default="")
    parser.add_argument("--seed", default="10023601464742706833")
    parser.add_argument("--icicle-backend-dir", default="")
    parser.add_argument("--bb-max-fused-batch-size", default=16, type=int)
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="Allow missing benchmark rows, recording them in summary metadata instead of failing.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def c_values(args):
    values = parse_csv_ints(args.c_values) if args.c_values else [args.c]
    if not values:
        raise RuntimeError("at least one c value is required")
    if any(value < 0 for value in values):
        raise RuntimeError("c values must be non-negative")
    return values


def factors(args):
    values = parse_csv_ints(args.factors)
    if not values:
        raise RuntimeError("at least one precompute factor is required")
    if any(value <= 0 for value in values):
        raise RuntimeError("precompute factors must be positive")
    return values


def log_values(args):
    return list(range(args.min_log, args.max_log + 1, args.log_step))


def jsonable_args(args):
    out = vars(args).copy()
    out["build_dir"] = str(out["build_dir"])
    out["output_dir"] = str(out["output_dir"])
    return out


def run_capture(cmd, cwd=None):
    try:
        completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
        return {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except OSError as err:
        return {"command": cmd, "error": str(err)}


def read_cmake_cache(build_dir: Path):
    cache_path = build_dir / "CMakeCache.txt"
    values = {"path": str(cache_path), "exists": cache_path.exists()}
    if not cache_path.exists():
        return values
    for line in cache_path.read_text(errors="replace").splitlines():
        if line.startswith("//") or ":" not in line or "=" not in line:
            continue
        name_type, value = line.split("=", 1)
        name = name_type.split(":", 1)[0]
        if name in CMAKE_CACHE_KEYS:
            values[name] = value
    return values


def runner_path(build_dir: Path, target: str) -> Path:
    candidates = [build_dir / "bin" / target, build_dir / "gpu" / target, build_dir / target]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def runner_metadata(build_dir: Path, implementations):
    metadata = {}
    for implementation in implementations:
        path = runner_path(build_dir, RUNNERS[implementation])
        entry = {"path": str(path), "exists": path.exists()}
        if path.exists():
            stat = path.stat()
            entry.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime": datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(),
                }
            )
        metadata[implementation] = entry
    return metadata


def initial_metadata(args, implementations, c_values_list):
    return {
        "status": "started",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": jsonable_args(args),
        "implementations": implementations,
        "requested_c_values": c_values_list,
        "requested_precompute_factors": factors(args),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": str(Path.cwd()),
        },
        "environment": {key: os.environ.get(key, "") for key in METADATA_ENV_KEYS},
        "git": {
            "rev_parse_head": run_capture(["git", "rev-parse", "HEAD"]),
            "status_short": run_capture(["git", "status", "--short", "--branch"]),
        },
        "gpu": {
            "nvidia_smi_query": run_capture(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,driver_version,memory.total,memory.free",
                    "--format=csv,noheader",
                ]
            ),
            "nvidia_smi_full": run_capture(["nvidia-smi"]),
        },
        "build": {
            "cmake_cache": read_cmake_cache(args.build_dir),
            "runners": runner_metadata(args.build_dir, implementations),
        },
        "commands": [],
    }


def write_metadata(output_dir: Path, metadata):
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def output_stem(implementation: str, c_value: int, c_count: int):
    return implementation if c_count == 1 else f"{implementation}.c{c_value}"


def run_one(args, implementation: str, c_value: int, output_path: Path):
    target = RUNNERS[implementation]
    binary = runner_path(args.build_dir, target)
    if not binary.exists():
        raise RuntimeError(f"{implementation} runner is missing: {binary}")

    cmd = [
        str(binary),
        "--mode",
        args.mode,
        "--output",
        str(output_path),
        "--memory-placement",
        args.memory_placement,
        "--min-log",
        str(args.min_log),
        "--max-log",
        str(args.max_log),
        "--log-step",
        str(args.log_step),
        "--factors",
        args.factors,
        "--repeats",
        str(args.repeats),
        "--batch-log",
        str(args.batch_log),
        "--batch-size",
        str(args.batch_size),
        "--c",
        str(c_value),
        "--seed",
        args.seed,
    ]
    if implementation == "bb":
        cmd += ["--bb-max-fused-batch-size", str(args.bb_max_fused_batch_size)]
    if implementation == "icicle-v4.0.0" and args.icicle_backend_dir:
        cmd += ["--icicle-backend-dir", args.icicle_backend_dir]
    subprocess.run(cmd, check=True)
    return cmd


def load_records(runs):
    records = []
    for run in runs:
        with run["path"].open() as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    record["requested_c"] = run["requested_c"]
                    record.setdefault("memory_placement", "host")
                    records.append(record)
    return records


def avg(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def fmt(value):
    return "" if value is None else f"{value:.3f}"


def record_key(record):
    return (
        record["implementation"],
        record["requested_c"],
        record["mode"],
        record["memory_placement"],
        record["log_num_points"],
        record["batch_size"],
        record["precompute_factor"],
        record["repeat"],
    )


def expected_record_keys(args, implementations, c_values_list):
    expected = set()
    factor_values = factors(args)
    if args.mode in ("single", "all"):
        for implementation in implementations:
            for c_value in c_values_list:
                for log_num_points in log_values(args):
                    for precompute_factor in factor_values:
                        for repeat in range(args.repeats):
                            expected.add(
                                (
                                    implementation,
                                    c_value,
                                    "single",
                                    args.memory_placement,
                                    log_num_points,
                                    1,
                                    precompute_factor,
                                    repeat,
                                )
                            )
    if args.mode in ("batch", "all"):
        for implementation in implementations:
            for c_value in c_values_list:
                for precompute_factor in factor_values:
                    for repeat in range(args.repeats):
                        expected.add(
                            (
                                implementation,
                                c_value,
                                "batch",
                                args.memory_placement,
                                args.batch_log,
                                args.batch_size,
                                precompute_factor,
                                repeat,
                            )
                        )
    return expected


def validate_completeness(records, args, implementations, c_values_list, allow_skips):
    expected = expected_record_keys(args, implementations, c_values_list)
    counts = Counter(record_key(record) for record in records)
    present = set(counts)
    missing = sorted(expected - present)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    result = {
        "expected_rows": len(expected),
        "actual_rows": len(records),
        "missing_rows": len(missing),
        "duplicate_rows": len(duplicates),
        "missing_examples": [list(key) for key in missing[:20]],
        "duplicate_examples": [{"key": list(key), "count": counts[key]} for key in duplicates[:20]],
    }
    if (missing or duplicates) and not allow_skips:
        details = []
        if missing:
            details.append(f"missing {len(missing)} expected benchmark rows; first examples: {result['missing_examples']}")
        if duplicates:
            details.append(f"found {len(duplicates)} duplicate benchmark rows; first examples: {result['duplicate_examples']}")
        raise RuntimeError("; ".join(details))
    return result


def validate_results(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["requested_c"],
            record["mode"],
            record["memory_placement"],
            record["log_num_points"],
            record["batch_size"],
            record["precompute_factor"],
            record["repeat"],
            record["seed"],
        )
        grouped[key].append(record)

    mismatches = []
    for key, values in grouped.items():
        expected = values[0]["result"]
        for value in values[1:]:
            if value["result"] != expected:
                mismatches.append((key, values))
                break
    if mismatches:
        key, values = mismatches[0]
        details = ", ".join(f"{value['implementation']}={value['result']}" for value in values)
        raise RuntimeError(f"result mismatch for {key}: {details}")


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["implementation"],
            record["requested_c"],
            record["mode"],
            record["memory_placement"],
            record["log_num_points"],
            record["batch_size"],
            record["precompute_factor"],
        )
        grouped[key].append(record)

    summaries = []
    for key, values in sorted(grouped.items()):
        implementation, requested_c, mode, memory_placement, log_num_points, batch_size, factor = key
        comparison = [value["comparison_ms"] for value in values]
        mean_comparison = statistics.fmean(comparison)
        stdev_comparison = statistics.pstdev(comparison) if len(comparison) > 1 else 0.0
        max_z = 0.0
        if stdev_comparison > 0.0:
            max_z = max(abs(value - mean_comparison) / stdev_comparison for value in comparison)
        summaries.append(
            {
                "implementation": implementation,
                "requested_c": requested_c,
                "mode": mode,
                "memory_placement": memory_placement,
                "log_num_points": log_num_points,
                "num_points": 1 << log_num_points,
                "batch_size": batch_size,
                "precompute_factor": factor,
                "samples": len(values),
                "comparison_ms_avg": mean_comparison,
                "comparison_ms_min": min(comparison),
                "comparison_ms_max": max(comparison),
                "comparison_ms_stdev": stdev_comparison,
                "outer_wall_ms_avg": avg(value["outer_wall_ms"] for value in values),
                "backend_wall_ms_avg": avg(value["backend_wall_ms"] for value in values),
                "device_ms_avg": avg(value.get("device_ms") for value in values),
                "setup_wall_ms_avg": avg(value["setup_wall_ms"] for value in values),
                "precompute_wall_ms_avg": avg(value["precompute_wall_ms"] for value in values),
                "precompute_device_ms_avg": avg(value.get("precompute_device_ms") for value in values),
                "backend_host_preamble_ms_avg": avg(value.get("backend_host_preamble_ms", 0.0) for value in values),
                "backend_host_cleanup_ms_avg": avg(value.get("backend_host_cleanup_ms", 0.0) for value in values),
                "backend_host_total_ms_avg": avg(value.get("backend_host_total_ms", 0.0) for value in values),
                "reported_c_avg": statistics.fmean(value["c"] for value in values),
                "large_bucket_count_avg": statistics.fmean(value["large_bucket_count"] for value in values),
                "max_bucket_size_avg": statistics.fmean(value["max_bucket_size"] for value in values),
                "extreme_sample": bool(max_z > 3.5 and len(values) >= 5),
            }
        )
    return summaries


def write_markdown(path: Path, summaries):
    with path.open("w") as out:
        out.write(
            "| implementation | requested c | reported c avg | mode | memory | log_n | batch | factor | samples | "
            "comparison avg ms | backend wall avg ms | device avg ms | outer wall avg ms | setup wall avg ms | "
            "precompute wall avg ms | extreme |\n"
        )
        out.write("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in summaries:
            out.write(
                f"| {row['implementation']} | {row['requested_c']} | {row['reported_c_avg']:.2f} | "
                f"{row['mode']} | {row['memory_placement']} | {row['log_num_points']} | {row['batch_size']} | "
                f"{row['precompute_factor']} | {row['samples']} | {fmt(row['comparison_ms_avg'])} | "
                f"{fmt(row['backend_wall_ms_avg'])} | {fmt(row['device_ms_avg'])} | "
                f"{fmt(row['outer_wall_ms_avg'])} | {fmt(row['setup_wall_ms_avg'])} | "
                f"{fmt(row['precompute_wall_ms_avg'])} | {row['extreme_sample']} |\n"
            )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    implementations = parse_csv_strings(args.implementations)
    unknown = sorted(set(implementations) - set(RUNNERS))
    if unknown:
        raise RuntimeError(f"unknown implementations: {', '.join(unknown)}")
    if args.memory_placement == "device" and "cpu" in implementations:
        raise RuntimeError("CPU runner only supports --memory-placement host")
    if args.memory_placement == "device" and "icicle-v2.8.0" in implementations:
        raise RuntimeError("Icicle v2.8.0 runner does not support --memory-placement device")
    c_values_list = c_values(args)

    metadata = initial_metadata(args, implementations, c_values_list)
    write_metadata(args.output_dir, metadata)
    try:
        runs = []
        c_count = len(c_values_list)
        for c_value in c_values_list:
            for implementation in implementations:
                raw_path = args.output_dir / f"{output_stem(implementation, c_value, c_count)}.jsonl"
                cmd = run_one(args, implementation, c_value, raw_path)
                command_record = {
                    "implementation": implementation,
                    "requested_c": c_value,
                    "output": str(raw_path),
                    "command": cmd,
                }
                metadata["commands"].append(command_record)
                runs.append({"path": raw_path, "implementation": implementation, "requested_c": c_value})
                write_metadata(args.output_dir, metadata)

        records = load_records(runs)
        completeness = validate_completeness(records, args, implementations, c_values_list, args.allow_skips)
        validate_results(records)
        merged_path = args.output_dir / "raw.jsonl"
        with merged_path.open("w") as out:
            for record in records:
                out.write(json.dumps(record, separators=(",", ":")) + "\n")

        summaries = summarize(records)
        (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        write_markdown(args.output_dir / "summary.md", summaries)

        metadata["status"] = "complete"
        metadata["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata["completeness"] = completeness
        metadata["outputs"] = {
            "raw": str(merged_path),
            "summary_json": str(args.output_dir / "summary.json"),
            "summary_markdown": str(args.output_dir / "summary.md"),
        }
        write_metadata(args.output_dir, metadata)
    except Exception as err:
        metadata["status"] = "failed"
        metadata["failed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata["error"] = str(err)
        write_metadata(args.output_dir, metadata)
        raise


if __name__ == "__main__":
    main()
