#!/usr/bin/env python3
import argparse
import gzip
import json
import statistics
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a BB Chonk IVC proof benchmark from captured ivc-inputs.msgpack.")
    parser.add_argument("--bb-bin", required=True, type=Path)
    parser.add_argument("--ivc-inputs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--warmups", default=0, type=int)
    parser.add_argument("--label", default="")
    parser.add_argument("--extra-bb-arg", action="append", default=[])
    return parser.parse_args()


def proof_size(path: Path):
    if not path.exists():
        return None, None
    proof = path.read_bytes()
    return len(proof), len(gzip.compress(proof))


def run_one(args, run_dir: Path, repeat: int, warmup: bool):
    run_dir.mkdir(parents=True, exist_ok=True)
    breakdown_path = run_dir / "benchmark_breakdown.json"
    memory_path = run_dir / "memory_profile.json"
    log_path = run_dir / "bb.log"
    cmd = [
        str(args.bb_bin),
        "prove",
        "-o",
        str(run_dir),
        "--ivc_inputs_path",
        str(args.ivc_inputs),
        "--scheme",
        "chonk",
        "-v",
        "--print_bench",
        "--bench_out_hierarchical",
        str(breakdown_path),
        "--memory_profile_out",
        str(memory_path),
        *args.extra_bb_arg,
    ]

    start = time.perf_counter()
    with log_path.open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    raw_size, compressed_size = proof_size(run_dir / "proof")
    record = {
        "benchmark": "chonk-ivc-proof",
        "label": args.label or args.ivc_inputs.parent.name,
        "repeat": repeat,
        "warmup": warmup,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_ms": elapsed_ms,
        "proof_size_bytes": raw_size,
        "proof_gzip_size_bytes": compressed_size,
        "run_dir": str(run_dir),
        "bb_bin": str(args.bb_bin),
        "ivc_inputs": str(args.ivc_inputs),
        "benchmark_breakdown": str(breakdown_path) if breakdown_path.exists() else None,
        "memory_profile": str(memory_path) if memory_path.exists() else None,
        "log": str(log_path),
        "command": cmd,
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"bb prove failed with exit code {result.returncode}; see {log_path}")
    return record


def write_summary(output_dir: Path, records):
    measured = [record for record in records if not record["warmup"]]
    elapsed = [record["elapsed_ms"] for record in measured]
    summary = {
        "benchmark": "chonk-ivc-proof",
        "samples": len(measured),
        "elapsed_ms_avg": statistics.fmean(elapsed) if elapsed else None,
        "elapsed_ms_min": min(elapsed) if elapsed else None,
        "elapsed_ms_max": max(elapsed) if elapsed else None,
        "elapsed_ms_stdev": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
        "records": measured,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "summary.md").open("w") as out:
        out.write("| repeat | elapsed ms | proof bytes | proof gzip bytes | run dir |\n")
        out.write("|---:|---:|---:|---:|---|\n")
        for record in measured:
            out.write(
                f"| {record['repeat']} | {record['elapsed_ms']:.3f} | "
                f"{record['proof_size_bytes'] or ''} | {record['proof_gzip_size_bytes'] or ''} | "
                f"{record['run_dir']} |\n"
            )
    return summary


def main():
    args = parse_args()
    if not args.bb_bin.exists():
        raise RuntimeError(f"missing bb binary: {args.bb_bin}")
    if not args.ivc_inputs.exists():
        raise RuntimeError(f"missing ivc inputs: {args.ivc_inputs}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    raw_path = args.output_dir / "raw.jsonl"
    with raw_path.open("w") as raw:
        for index in range(args.warmups):
            record = run_one(args, args.output_dir / f"warmup_{index:02d}", index, True)
            raw.write(json.dumps(record, separators=(",", ":")) + "\n")
            raw.flush()
            records.append(record)
        for index in range(args.repeats):
            record = run_one(args, args.output_dir / f"run_{index:02d}", index, False)
            raw.write(json.dumps(record, separators=(",", ":")) + "\n")
            raw.flush()
            records.append(record)

    summary = write_summary(args.output_dir, records)
    if summary["elapsed_ms_avg"] is not None:
        print(f"elapsed avg: {summary['elapsed_ms_avg']:.3f} ms over {summary['samples']} sample(s)")
    print(f"summary: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
