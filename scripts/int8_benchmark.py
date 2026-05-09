"""
Phase 5a: INT8 Weight Quantization Benchmark.

Goal: measure per-token decode latency for TinyLlama-1.1B in INT8 vs the
float16 baseline.

NOTE on the MPS backend: bitsandbytes 0.49 still relies on CUDA-only
LLM.int8() kernels. `load_in_8bit=True` therefore cannot run on Apple
Silicon's MPS backend at the time of writing. We fall back gracefully:

  1. Try BitsAndBytesConfig(load_in_8bit=True) on MPS.
  2. If unsupported, run INT8 on CPU using torch.ao dynamic INT8 weight
     quantization (per-channel symmetric on nn.Linear), which is the
     comparable INT8 weight-only path that PyTorch ships natively.

Both paths produce real measured latencies. The paper reports the path
that actually executed.

Output: data/int8_benchmark.csv with summary statistics matching the
columns of data/benchmark_summary.csv.
"""

import time
import statistics
import csv
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "Explain how a computer processor works in simple terms:"
MAX_NEW_TOKENS = 128
WARMUP_RUNS    = 3
TIMED_TRIALS   = 10
DATA_DIR       = "data"


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_trial(tokenizer, model, device, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    token_times = []
    e2e_start = time.perf_counter()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    sync(device)
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
        sync(device)
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


def try_bnb_mps():
    """Attempt to load INT8 model on MPS via bitsandbytes."""
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": "mps"}
    ).eval()
    return tok, mdl, torch.device("mps"), "bitsandbytes-int8-mps"


def torch_int8_cpu():
    """Fallback: torch native dynamic INT8 weight quantization on CPU.

    On Apple Silicon (arm64) the only supported torch quantized backend is
    qnnpack; the default x86 fbgemm engine is unavailable.
    """
    torch.backends.quantized.engine = "qnnpack"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True
    ).to("cpu").eval()
    qmdl = torch.quantization.quantize_dynamic(
        mdl, {torch.nn.Linear}, dtype=torch.qint8
    )
    return tok, qmdl, torch.device("cpu"), "torch-dynamic-int8-cpu (qnnpack)"


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
    print("INT8 Quantization Benchmark — TinyLlama-1.1B")
    print("=" * 60)

    # bitsandbytes 0.49 ships only a CUDA-accelerated INT8 GEMM. On MPS it
    # silently falls back to a CPU dequantize-then-fp16-matmul path that
    # runs ~30x slower than the float16 baseline (we measured >5 minutes
    # for a single warm-up trial). We therefore use torch's native dynamic
    # INT8 weight quantization on CPU as the comparable INT8 measurement
    # path. The paper documents this design choice explicitly.
    print("Loading INT8 model via torch.quantization.quantize_dynamic on CPU...")
    print("(bitsandbytes-MPS path is unsupported in 0.49: INT8 GEMM is")
    print(" CUDA-only; the MPS fallback is several minutes per trial.)")
    tokenizer, model, device, backend = torch_int8_cpu()
    print(f"  Loaded INT8 model on CPU via torch.quantization")

    print(f"\nBackend: {backend} | Device: {device}")
    print(f"Running {WARMUP_RUNS} warm-up trials...")
    for i in range(WARMUP_RUNS):
        run_trial(tokenizer, model, device, PROMPT, MAX_NEW_TOKENS)
        print(f"  warm-up {i+1}/{WARMUP_RUNS} done")

    print(f"\nRunning {TIMED_TRIALS} timed trials...")
    all_trials = []
    for i in range(TIMED_TRIALS):
        r = run_trial(tokenizer, model, device, PROMPT, MAX_NEW_TOKENS)
        all_trials.append(r)
        print(f"  trial {i+1:02d}/{TIMED_TRIALS}  TTFT={r['ttft_ms']:7.1f} ms  "
              f"med-tok={statistics.median(r['token_times_ms']):6.2f} ms  "
              f"E2E={r['e2e_ms']:8.1f} ms")

    # Aggregate
    all_tok = []
    for t in all_trials:
        all_tok.extend(t["token_times_ms"])
    cross = summarize(all_tok)
    cross_ttft = summarize([t["ttft_ms"] for t in all_trials])
    throughput = 1000.0 / cross["median_ms"]

    print(f"\n{'=' * 60}")
    print(f"  INT8 SUMMARY  ({backend})")
    print(f"{'=' * 60}")
    print(f"  TTFT median: {cross_ttft['median_ms']:.2f} ms")
    print(f"  Per-token median: {cross['median_ms']:.2f} ms "
          f"(p95={cross['p95_ms']:.2f}, p99={cross['p99_ms']:.2f})")
    print(f"  Throughput median: {throughput:.2f} tokens/sec")

    # Write CSV
    out = os.path.join(DATA_DIR, "int8_benchmark.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["backend", backend])
        w.writerow(["device", str(device)])
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
