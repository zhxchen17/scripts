#!/usr/bin/env python3
"""Benchmark torch.compile cold-start and warm-start times for vLLM offline inference."""

import argparse
import fnmatch
import getpass
import io
import os
import random
import re
import shutil
import string
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

# Regex to capture the compile time from vllm's monitor log line:
#   "torch.compile takes 123.45 s in total"
_COMPILE_TIME_RE = re.compile(r"torch\.compile takes ([\d.]+) s")

# Minimal inline script executed as a subprocess for each inference run.
# It reads a TOML config, constructs an LLM, and runs a single generate call
# (batch size 1).  The compile-time log line is emitted by vllm automatically.
_INFERENCE_SCRIPT = textwrap.dedent(r"""
    import sys, tomllib
    from vllm import LLM, SamplingParams

    config_path = sys.argv[1]
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    model = cfg.pop("model")
    llm = LLM(model=model, **cfg)
    llm.generate(["Hello"], SamplingParams(max_tokens=1))
""")


_RUN_ID = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
_TRACE_PREFIX = "/tmp/compile_bench"

_CACHE_DIRS = [
    Path.home() / ".cache" / "vllm",
    Path(f"/tmp/torchinductor_{getpass.getuser()}"),
]


def _clear_caches() -> None:
    """Remove vllm and torchinductor caches to ensure a true cold start."""
    for d in _CACHE_DIRS:
        if d.exists():
            print(f"  🗑  removing cache: {d}", flush=True)
            shutil.rmtree(d)


def _tee_stream(stream: io.TextIOWrapper, sink, captured: list[str]) -> None:
    """Read *stream* line-by-line, forward each line to *sink*, and collect."""
    for line in stream:
        captured.append(line)
        sink.write(line)
        sink.flush()
    stream.close()


def bench_compile_time(
    config_path: str | Path,
    num_warm_runs: int = 1,
    debug: bool = False,
) -> dict:
    """Run N+1 vllm offline-inference invocations and collect compile times.

    Args:
        config_path: Path to a TOML config file containing at minimum ``model``
            and any extra kwargs forwarded to ``vllm.LLM``.
        num_warm_runs: Number of warm-start iterations (default 1).
            Total runs = 1 (cold) + num_warm_runs.

    Returns:
        A dict with keys ``model``, ``cold_start``, ``warm_start_avg``.
    """
    config_path = Path(config_path)
    model_name = config_path.stem
    total_runs = 1 + num_warm_runs
    compile_times: list[float] = []
    trace_dirs: list[str] = []

    # Clear all caches that affect compile times so cold start is truly cold.
    _clear_caches()

    for i in range(total_runs):
        label = "cold" if i == 0 else f"warm_{i}"
        print(f"[{model_name}] run {i + 1}/{total_runs} ({label}) …", flush=True)

        trace_dir = f"{_TRACE_PREFIX}_{model_name}_{label}_{_RUN_ID}"
        env = {**os.environ, "TORCH_TRACE": trace_dir}

        if debug:
            proc = subprocess.Popen(
                [sys.executable, "-c", _INFERENCE_SCRIPT, str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            t_out = threading.Thread(
                target=_tee_stream,
                args=(proc.stdout, sys.stdout, stdout_lines),
                daemon=True,
            )
            t_err = threading.Thread(
                target=_tee_stream,
                args=(proc.stderr, sys.stderr, stderr_lines),
                daemon=True,
            )
            t_out.start()
            t_err.start()
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
            t_out.join()
            t_err.join()
            returncode = proc.returncode
            stdout_text = "".join(stdout_lines)
            stderr_text = "".join(stderr_lines)
        else:
            proc = subprocess.Popen(
                [sys.executable, "-c", _INFERENCE_SCRIPT, str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                stdout_text, stderr_text = proc.communicate()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
            returncode = proc.returncode

        if returncode != 0:
            print(
                f"\n✗ [{model_name}] run {i + 1}/{total_runs} ({label}) "
                f"failed with exit code {returncode}",
                file=sys.stderr,
                flush=True,
            )
            if not debug:
                tail = "\n".join(stderr_text.splitlines()[-50:])
                if tail:
                    print(tail, file=sys.stderr, flush=True)
            sys.exit(returncode)

        combined_output = stdout_text + "\n" + stderr_text
        match = _COMPILE_TIME_RE.search(combined_output)

        if match:
            t = float(match.group(1))
            compile_times.append(t)
            print(f"  → torch.compile took {t:.2f} s", flush=True)
        else:
            print(
                f"  ⚠ could not find compile-time log line in output",
                file=sys.stderr,
                flush=True,
            )
            compile_times.append(float("nan"))

        trace_dirs.append(trace_dir)

    cold_start = compile_times[0]
    warm_times = compile_times[1:]
    warm_avg = (
        sum(t for t in warm_times if t == t) / max(sum(1 for t in warm_times if t == t), 1)
    )

    return {
        "model": model_name,
        "cold_start": cold_start,
        "warm_start_avg": warm_avg,
        "cold_trace": trace_dirs[0],
        "warm_trace": trace_dirs[-1] if num_warm_runs > 0 else "",
    }


def discover_configs(configs_dir: str | Path, pattern: str) -> list[Path]:
    """Return sorted TOML config paths whose stem matches *pattern*."""
    configs_dir = Path(configs_dir)
    matched = [
        p
        for p in sorted(configs_dir.glob("*.toml"))
        if fnmatch.fnmatch(p.stem, pattern)
    ]
    return matched


def print_results_table(results: list[dict]) -> None:
    """Print a formatted summary table."""
    header = (
        "Model", "Cold Start (s)", "Warm Start Avg (s)",
        "Cold Trace", "Warm Trace",
    )
    rows = [
        (
            r["model"],
            f"{r['cold_start']:.2f}",
            f"{r['warm_start_avg']:.2f}",
            r["cold_trace"],
            r["warm_trace"],
        )
        for r in results
    ]

    ncols = len(header)
    col_widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(ncols)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    print()
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile times for vLLM models."
    )
    parser.add_argument(
        "--filter",
        default="*",
        help=(
            "Glob pattern matched against config file stems. "
            'Use "*" to run all models, or e.g. "llama" for a specific one.'
        ),
    )
    parser.add_argument(
        "--num-warm-runs",
        type=int,
        default=1,
        help="Number of warm-start iterations per model (default: 1).",
    )
    parser.add_argument(
        "--configs-dir",
        default=os.path.join(os.path.dirname(__file__), "configs"),
        help="Directory containing TOML config files (default: ./configs).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Forward all vLLM subprocess stdout/stderr to the terminal.",
    )
    args = parser.parse_args()

    configs = discover_configs(args.configs_dir, args.filter)
    if not configs:
        print(
            f'No configs matched pattern "{args.filter}" in {args.configs_dir}',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Will benchmark {len(configs)} model(s): "
          f"{', '.join(c.stem for c in configs)}")
    print(f"Runs per model: 1 cold + {args.num_warm_runs} warm")
    print()

    results: list[dict] = []
    for cfg in configs:
        result = bench_compile_time(
            cfg, num_warm_runs=args.num_warm_runs, debug=args.debug
        )
        results.append(result)

    print_results_table(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
