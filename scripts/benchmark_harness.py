"""
Phase 2: Repeatable Benchmark Harness for TinyLlama on Apple Silicon (MPS).

Measures:
  - TTFT  (Time to First Token) — prefill phase latency
  - Per-token steady-state latency (median, p95, p99)
  - End-to-end generation time

Protocol:
  - 3 warm-up runs  (discarded — lets the GPU JIT-compile and cache kernels)
  - 10 timed trials (recorded)
  - Outlier filtering via IQR method on per-token median

Outputs:
  - data/benchmark_raw.csv     — every token latency from every trial
  - data/benchmark_summary.csv — per-trial aggregates + cross-trial stats
  - figures/timing_diagram.png — visual methodology diagram
"""

import time
import statistics
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "Explain how a computer processor works in simple terms:"
MAX_NEW_TOKENS = 128
WARMUP_RUNS    = 3
TIMED_TRIALS   = 10
DATA_DIR       = "data"
FIGURES_DIR    = "figures"


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Force GPU to finish all pending ops before reading the clock.

    CONCEPT: Why synchronize?
    GPU operations are submitted to a command queue and execute asynchronously.
    If you read time.perf_counter() immediately after launching a GPU op,
    you measure how fast the CPU *submitted* the work, not how fast the GPU
    *finished* it. Synchronizing blocks the CPU until the GPU is done.
    """
    if device.type == "mps":
        torch.mps.synchronize()


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_id, device):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded {params:.2f}B parameters on {device}\n")
    return tokenizer, model


# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(tokenizer, model, device, prompt, max_new_tokens):
    """
    Run one complete generation and return timing measurements.

    Returns dict with:
      ttft_ms          : Time to First Token in milliseconds
      token_times_ms   : list of per-token decode latencies (ms)
      e2e_ms           : total wall-clock time (prefill + all decode steps)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    token_times = []
    e2e_start = time.perf_counter()

    # ── Prefill (TTFT measurement) ──────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    sync(device)
    ttft = time.perf_counter() - t0

    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    # ── Decode loop (per-token latency) ────────────────────────────────────
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


# ── Outlier filtering ──────────────────────────────────────────────────────────
def iqr_filter(values):
    """
    Remove outliers using the IQR (Interquartile Range) method.

    CONCEPT: IQR Outlier Filtering
    Sort the data → find Q1 (25th percentile) and Q3 (75th percentile).
    IQR = Q3 - Q1 (the middle 50% spread).
    Any value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is flagged as an outlier.
    This is robust to extreme values and is the standard box-plot method.

    In benchmarking, outliers arise from OS scheduling interrupts, thermal
    throttling, or one-time JIT compilation events.
    """
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (arr >= lo) & (arr <= hi)
    return arr[mask].tolist(), (~mask).sum()


# ── Statistics helpers ─────────────────────────────────────────────────────────
def percentile(data, p):
    return float(np.percentile(data, p))


def summarize(token_times_ms):
    """Compute aggregate stats for a list of per-token latencies."""
    return {
        "median_ms":  statistics.median(token_times_ms),
        "mean_ms":    statistics.mean(token_times_ms),
        "p95_ms":     percentile(token_times_ms, 95),
        "p99_ms":     percentile(token_times_ms, 99),
        "min_ms":     min(token_times_ms),
        "max_ms":     max(token_times_ms),
        "stdev_ms":   statistics.stdev(token_times_ms) if len(token_times_ms) > 1 else 0,
    }


# ── CSV writers ────────────────────────────────────────────────────────────────
def save_raw_csv(all_trials, path):
    """
    Save every individual token latency.
    Columns: trial, token_index, latency_ms
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "token_index", "latency_ms"])
        for trial_idx, trial in enumerate(all_trials):
            for tok_idx, lat in enumerate(trial["token_times_ms"]):
                writer.writerow([trial_idx + 1, tok_idx + 1, f"{lat:.4f}"])
    print(f"  Raw data → {path}")


def save_summary_csv(trial_summaries, cross_stats, path):
    """
    Save per-trial summary stats and cross-trial aggregate at the bottom.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["trial", "ttft_ms", "e2e_ms", "n_tokens",
              "median_ms", "mean_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "stdev_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trial_summaries:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                             for k, v in row.items()})
        # blank separator row
        writer.writerow({k: "" for k in fields})
        # cross-trial aggregate
        agg = {"trial": "AGGREGATE"}
        agg.update({k: f"{v:.4f}" if isinstance(v, float) else v
                    for k, v in cross_stats.items()})
        writer.writerow(agg)
    print(f"  Summary  → {path}")


# ── Timing diagram ─────────────────────────────────────────────────────────────
def plot_timing_diagram(trial_summaries, all_trials, out_path):
    """
    Two-panel figure:
      Top:    Timeline bar showing TTFT vs decode phase for one representative trial
      Bottom: Per-token latency over decode steps (all trials overlaid, median highlighted)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("LLaMA Benchmark — Timing Measurement Methodology\n"
                 f"Model: TinyLlama-1.1B | Prompt tokens: 12 | Max new tokens: {MAX_NEW_TOKENS}",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Timeline bar ──────────────────────────────────────────────
    # Use median trial (closest to median TTFT)
    medians = [t["ttft_ms"] for t in trial_summaries]
    med_ttft = statistics.median(medians)
    rep_idx = min(range(len(medians)), key=lambda i: abs(medians[i] - med_ttft))
    rep = trial_summaries[rep_idx]

    ttft = rep["ttft_ms"]
    decode = rep["e2e_ms"] - ttft

    ax1.barh(0, ttft, color="#e74c3c", height=0.4, label=f"Prefill / TTFT  ({ttft:.0f} ms)")
    ax1.barh(0, decode, left=ttft, color="#3498db", height=0.4,
             label=f"Decode phase  ({decode:.0f} ms, {rep['n_tokens']} tokens)")

    # Annotate TTFT arrow
    ax1.annotate("", xy=(ttft, 0.35), xytext=(0, 0.35),
                 arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=1.5))
    ax1.text(ttft / 2, 0.42, f"TTFT\n{ttft:.0f} ms", ha="center", va="bottom",
             color="#e74c3c", fontsize=9, fontweight="bold")

    # Annotate total
    total = rep["e2e_ms"]
    ax1.annotate("", xy=(total, -0.35), xytext=(0, -0.35),
                 arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.5))
    ax1.text(total / 2, -0.43, f"End-to-end: {total:.0f} ms", ha="center", va="top",
             color="#2c3e50", fontsize=9, fontweight="bold")

    ax1.set_xlim(0, total * 1.05)
    ax1.set_ylim(-0.7, 0.7)
    ax1.set_xlabel("Wall-clock time (ms)")
    ax1.set_title("Panel A — Generation Timeline (representative trial)", fontsize=10)
    ax1.set_yticks([])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.axvline(ttft, color="#e74c3c", linestyle="--", alpha=0.5)

    # ── Panel 2: Per-token latency over steps ──────────────────────────────
    all_token_lats = [t["token_times_ms"] for t in all_trials]
    max_len = max(len(x) for x in all_token_lats)

    # Pad shorter trials with NaN so we can stack into a matrix
    mat = np.full((len(all_token_lats), max_len), np.nan)
    for i, row in enumerate(all_token_lats):
        mat[i, :len(row)] = row

    col_median = np.nanmedian(mat, axis=0)
    col_p95    = np.nanpercentile(mat, 95, axis=0)
    steps      = np.arange(1, max_len + 1)

    # Plot each trial lightly
    for i, row in enumerate(all_token_lats):
        ax2.plot(range(1, len(row) + 1), row, color="#aab7c4", alpha=0.35,
                 linewidth=0.8, label="Individual trials" if i == 0 else "")

    # Median + p95 envelope
    ax2.plot(steps, col_median, color="#2980b9", linewidth=2, label="Median latency")
    ax2.plot(steps, col_p95, color="#e67e22", linewidth=1.5,
             linestyle="--", label="p95 latency")

    # Shade the "steady-state" region (skip first 5 tokens which may be noisy)
    ax2.axvspan(5, max_len, alpha=0.06, color="green",
                label="Steady-state region (tokens 5+)")

    # Mark measurement stats
    ss_median = statistics.median(col_median[4:].tolist())
    ax2.axhline(ss_median, color="#27ae60", linestyle=":", linewidth=1.5,
                label=f"Steady-state median: {ss_median:.1f} ms")

    ax2.set_xlabel("Decode step (token index)")
    ax2.set_ylabel("Per-token latency (ms)")
    ax2.set_title("Panel B — Per-Token Latency Across All Trials", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(1, max_len)
    ax2.set_ylim(bottom=0)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Diagram  → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}\n")

    tokenizer, model = load_model(MODEL_ID, device)

    # ── Warm-up runs (discarded) ────────────────────────────────────────────
    # CONCEPT: Why warm up?
    # On first run, PyTorch compiles Metal shaders, allocates memory pools,
    # and the OS brings model weights into physical RAM. These one-time costs
    # would contaminate our measurements. Warm-up runs let all of that happen
    # before we start the clock.
    print(f"Running {WARMUP_RUNS} warm-up trials (discarded)...")
    for i in range(WARMUP_RUNS):
        run_trial(tokenizer, model, device, PROMPT, MAX_NEW_TOKENS)
        print(f"  Warm-up {i+1}/{WARMUP_RUNS} done")

    # ── Timed trials ────────────────────────────────────────────────────────
    print(f"\nRunning {TIMED_TRIALS} timed trials...")
    all_trials = []
    for i in range(TIMED_TRIALS):
        result = run_trial(tokenizer, model, device, PROMPT, MAX_NEW_TOKENS)
        all_trials.append(result)
        print(f"  Trial {i+1:02d}/{TIMED_TRIALS} | "
              f"TTFT: {result['ttft_ms']:6.1f} ms | "
              f"Median tok: {statistics.median(result['token_times_ms']):5.2f} ms | "
              f"E2E: {result['e2e_ms']:7.1f} ms")

    # ── Outlier filtering on per-trial median ───────────────────────────────
    trial_medians = [statistics.median(t["token_times_ms"]) for t in all_trials]
    filtered_medians, n_outliers = iqr_filter(trial_medians)
    print(f"\nOutlier filtering: removed {n_outliers} trial(s) by IQR on per-trial median.")

    # Keep only non-outlier trials
    kept = [t for t, m in zip(all_trials, trial_medians) if m in filtered_medians]
    if len(kept) == 0:
        kept = all_trials  # fallback: don't discard everything

    # ── Per-trial summary rows ──────────────────────────────────────────────
    trial_summaries = []
    for i, trial in enumerate(kept):
        stats = summarize(trial["token_times_ms"])
        row = {
            "trial":    i + 1,
            "ttft_ms":  trial["ttft_ms"],
            "e2e_ms":   trial["e2e_ms"],
            "n_tokens": trial["n_tokens"],
        }
        row.update(stats)
        trial_summaries.append(row)

    # ── Cross-trial aggregate (pool all token latencies across kept trials) ─
    all_token_lats = []
    for t in kept:
        all_token_lats.extend(t["token_times_ms"])

    cross = summarize(all_token_lats)
    cross_ttft = summarize([t["ttft_ms"] for t in kept])

    # ── Print final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY  ({len(kept)} trials after filtering)")
    print(f"{'='*60}")
    print(f"  TTFT        — median: {cross_ttft['median_ms']:7.2f} ms  "
          f"| p95: {cross_ttft['p95_ms']:7.2f} ms  | p99: {cross_ttft['p99_ms']:7.2f} ms")
    print(f"  Per-token   — median: {cross['median_ms']:7.2f} ms  "
          f"| p95: {cross['p95_ms']:7.2f} ms  | p99: {cross['p99_ms']:7.2f} ms")
    print(f"  Throughput  — {1000 / cross['median_ms']:.1f} tokens/sec (median)")
    print(f"{'='*60}\n")

    # ── Save outputs ────────────────────────────────────────────────────────
    print("Saving outputs...")
    save_raw_csv(kept, os.path.join(DATA_DIR, "benchmark_raw.csv"))
    save_summary_csv(trial_summaries, cross, os.path.join(DATA_DIR, "benchmark_summary.csv"))
    plot_timing_diagram(trial_summaries, kept, os.path.join(FIGURES_DIR, "timing_diagram.png"))
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
