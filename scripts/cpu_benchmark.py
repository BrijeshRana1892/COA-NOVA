"""
Phase 5b: CPU baseline (float16) for cross-platform comparison.

Same harness as benchmark_harness.py but device='cpu' to enable an
MPS-vs-CPU latency comparison. Float16 is preserved (matches the MPS
baseline) although CPU does not get hardware acceleration for fp16 ops
on Apple Silicon (operations are widened to fp32 internally).

Output: data/cpu_benchmark.csv
"""

import time
import statistics
import csv
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "Explain how a computer processor works in simple terms:"
MAX_NEW_TOKENS = 128
WARMUP_RUNS    = 3
TIMED_TRIALS   = 5
DATA_DIR       = "data"


def run_trial(tokenizer, model, device, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    token_times = []
    e2e_start = time.perf_counter()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    ttft = time.perf_counter() - t0

    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    for _ in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        token_times.append(time.perf_counter() - t0)

        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

    e2e = time.perf_counter() - e2e_start
    return {
        "ttft_ms": ttft * 1000,
        "token_times_ms": [t * 1000 for t in token_times],
        "e2e_ms": e2e * 1000,
        "n_tokens": len(generated),
    }


def percentile(data, p):
    return float(np.percentile(data, p))


def summarize(token_times_ms):
    return {
        "median_ms":  statistics.median(token_times_ms),
        "mean_ms":    statistics.mean(token_times_ms),
        "p95_ms":     percentile(token_times_ms, 95),
        "p99_ms":     percentile(token_times_ms, 99),
        "min_ms":     min(token_times_ms),
        "max_ms":     max(token_times_ms),
        "stdev_ms":   statistics.stdev(token_times_ms) if len(token_times_ms) > 1 else 0,
    }


def main():
    device = torch.device("cpu")
    print(f"CPU Baseline Benchmark — TinyLlama-1.1B float16 on {device}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()

    print(f"Running {WARMUP_RUNS} warm-up trials...")
    for i in range(WARMUP_RUNS):
        run_trial(tokenizer, model, device, PROMPT, MAX_NEW_TOKENS)
        print(f"  warm-up {i+1}/{WARMUP_RUNS} done")

    print(f"\nRunning {TIMED_TRIALS} timed trials...")
    all_trials = []
    for i in range(TIMED_TRIALS):
        r = run_trial(tokenizer, model, device, PROMPT, MAX_NEW_TOKENS)
        all_trials.append(r)
        print(f"  trial {i+1:02d}/{TIMED_TRIALS}  TTFT={r['ttft_ms']:8.1f} ms  "
              f"med-tok={statistics.median(r['token_times_ms']):7.2f} ms  "
              f"E2E={r['e2e_ms']:9.1f} ms")

    all_tok = []
    for t in all_trials:
        all_tok.extend(t["token_times_ms"])
    cross = summarize(all_tok)
    cross_ttft = summarize([t["ttft_ms"] for t in all_trials])
    throughput = 1000.0 / cross["median_ms"]

    print(f"\n{'=' * 60}")
    print(f"  CPU SUMMARY")
    print(f"{'=' * 60}")
    print(f"  TTFT median: {cross_ttft['median_ms']:.2f} ms")
    print(f"  Per-token median: {cross['median_ms']:.2f} ms "
          f"(p95={cross['p95_ms']:.2f}, p99={cross['p99_ms']:.2f})")
    print(f"  Throughput median: {throughput:.2f} tokens/sec")

    out = os.path.join(DATA_DIR, "cpu_benchmark.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["backend", "torch-cpu-float16"])
        w.writerow(["device", "cpu"])
        w.writerow(["n_trials", len(all_trials)])
        w.writerow(["max_new_tokens", MAX_NEW_TOKENS])
        w.writerow(["ttft_median_ms", f"{cross_ttft['median_ms']:.4f}"])
        w.writerow(["ttft_p95_ms", f"{cross_ttft['p95_ms']:.4f}"])
        w.writerow(["per_token_median_ms", f"{cross['median_ms']:.4f}"])
        w.writerow(["per_token_mean_ms", f"{cross['mean_ms']:.4f}"])
        w.writerow(["per_token_p95_ms", f"{cross['p95_ms']:.4f}"])
        w.writerow(["per_token_p99_ms", f"{cross['p99_ms']:.4f}"])
        w.writerow(["per_token_stdev_ms", f"{cross['stdev_ms']:.4f}"])
        w.writerow(["throughput_tokens_per_sec", f"{throughput:.4f}"])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
