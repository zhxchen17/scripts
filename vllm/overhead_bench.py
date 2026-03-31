"""Offline vLLM inference benchmark with torch profiler.

Example usage:
python ./vllm/overhead_bench.py --model meta-llama/meta-llama-3-8b --input-len 512 \
     --output-len 4 --num-prompts 8 --profile --with-stack --trace-dir ./trace
"""

import argparse
import os
import random
import time

import torch
from vllm import LLM, SamplingParams


def make_random_prompts(num_prompts: int, input_len: int, vocab_size: int = 32000):
    """Generate random token-id prompts (mirrors --dataset-name random)."""
    prompts = []
    for _ in range(num_prompts):
        token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        prompts.append({"prompt_token_ids": token_ids})
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Offline vLLM benchmark with profiling")
    parser.add_argument("--model", type=str, default="meta-llama/meta-llama-3-8b")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--trace-dir", type=str, default="./trace")
    parser.add_argument("--with-stack", action="store_true", default=False)
    args = parser.parse_args()

    # Disable V1 multiprocessing so the torch profiler captures
    # the full Python stack in the same process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Matches: -cc.cudagraph_mode=none
    llm = LLM(
        model=args.model,
        compilation_config={"cudagraph_mode": "none"},
    )

    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    prompts = make_random_prompts(args.num_prompts, args.input_len)

    # Warmup run (outside profiler)
    print("Warmup ...")
    llm.generate(prompts, sampling_params)

    # Profiled run
    if args.profile:
        os.makedirs(args.trace_dir, exist_ok=True)
        existing = set(os.listdir(args.trace_dir))
        print(f"Running with torch profiler (trace dir: {args.trace_dir}) ...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=args.with_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.trace_dir),
        ) as prof:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0
        new_files = set(os.listdir(args.trace_dir)) - existing
        for f in sorted(new_files):
            print(f"Trace file: {os.path.join(args.trace_dir, f)}")
    else:
        print("Running (no profiler) ...")
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - t0

    # Print summary
    total_input = args.input_len * args.num_prompts
    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n{'='*50}")
    print(f"Model:            {args.model}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Input len:        {args.input_len}")
    print(f"Output len:       {args.output_len}")
    print(f"Total input tok:  {total_input}")
    print(f"Total output tok: {total_output}")
    print(f"Elapsed:          {elapsed:.3f}s")
    print(f"Throughput:        {(total_input + total_output) / elapsed:.1f} tok/s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
