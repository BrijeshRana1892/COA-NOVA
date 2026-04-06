"""
Phase 4: Scaling Analysis

Measures how per-token decode latency changes across three dimensions:
  1. Context length  : [64, 128, 256, 512, 1024] tokens in KV-cache
  2. Model size      : TinyLlama-1.1B vs OpenLLaMA-3B (same LLaMA architecture)
  3. Precision       : float16 vs float32 (TinyLlama-1.1B only)

METHODOLOGY — Context length measurement:
  Instead of measuring TTFT (which scales with prompt length), we measure
  *decode* latency after the KV-cache has been filled to the target size.
  Steps:
    1. Prefill with a short prompt (~12 tokens)
    2. Autoregressively generate tokens until the KV-cache reaches the
       target context length (no timing)
    3. Measure the next MEASURE_STEPS decode steps → this is our signal

  This isolates the cost of reading a KV-cache of a given size, which is
  the bottleneck we care about for long-context inference.

CONCEPT — Why context length matters:
  Each decode step reads the ENTIRE KV-cache (all past K and V tensors) to
  compute attention scores. Memory bandwidth required = 2 × num_layers ×
  seq_len × hidden_dim × bytes_per_element. As context grows, this read
  dominates latency — a classic memory-bandwidth bottleneck.

Outputs:
  data/scaling_context_length.csv
  data/scaling_model_size.csv
  data/scaling_precision.csv
  figures/scaling_context_length.png
  figures/scaling_model_size.png
  figures/scaling_precision.png
"""

import time
import csv
import os
import gc
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ─────────────────────────────────────────────────────────────────────
CONTEXT_LENGTHS  = [64, 128, 256, 512, 1024]
MEASURE_STEPS    = 15      # decode steps to time at each context length
WARMUP_FILLS     = 2       # full context-fill warm-ups before measuring
BASE_PROMPT      = "Explain how a computer processor works in simple terms:"

MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "OpenLLaMA-3B":   "openlm-research/open_llama_3b_v2",
}

PRECISIONS = {
    "float16": torch.float16,
    "float32": torch.float32,
}

DATA_DIR    = "data"
FIGURES_DIR = "figures"


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_model(model_id, device, dtype=torch.float16):
    print(f"  Loading {model_id} ({dtype}) ...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"{n_params:.2f}B params loaded.")
    return tokenizer, model


def unload_model(model):
    """Free GPU and CPU memory before loading the next model."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Core measurement ───────────────────────────────────────────────────────────
def fill_to_context_length(tokenizer, model, device, target_len):
    """
    Fill the KV-cache to `target_len` tokens by autoregressively generating
    tokens from the base prompt. Returns (past_key_values, next_token) ready
    for the first measured decode step.
    """
    inputs = tokenizer(BASE_PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    current_len = input_ids.shape[1]

    # Prefill
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    past_kv    = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    current_len += 1

    # Generate tokens silently until KV-cache reaches target_len
    while current_len < target_len:
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv    = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_len += 1

    return past_kv, next_token


def measure_decode_latency(model, device, past_kv, next_token, n_steps):
    """
    Run `n_steps` decode steps and return per-step latencies in ms.
    Each step reads the full KV-cache and generates one new token.
    """
    latencies = []
    for _ in range(n_steps):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        sync(device)
        latencies.append((time.perf_counter() - t0) * 1000)
        past_kv    = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return latencies


def run_context_sweep(tokenizer, model, device, context_lengths, n_steps, n_warmup):
    """
    For each context length: warm up, then measure decode latency.
    Returns dict: {ctx_len: {"median": ..., "p95": ..., "p99": ..., "samples": [...]}}
    """
    results = {}
    for ctx in context_lengths:
        print(f"    context={ctx:4d} tokens ", end="", flush=True)

        # Warm-up fills (not timed)
        for _ in range(n_warmup):
            past_kv, next_token = fill_to_context_length(tokenizer, model, device, ctx)
            sync(device)

        # Timed measurement
        past_kv, next_token = fill_to_context_length(tokenizer, model, device, ctx)
        lats = measure_decode_latency(model, device, past_kv, next_token, n_steps)

        med = statistics.median(lats)
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        results[ctx] = {"median": med, "p95": p95, "p99": p99, "samples": lats}
        print(f"→ median={med:.2f} ms  p95={p95:.2f} ms")

    return results


# ── CSV writers ────────────────────────────────────────────────────────────────
def save_csv(rows, fieldnames, path):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  Saved → {path}")


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_context_scaling(all_results, out_path):
    """
    Line plot: per-token latency vs context length, one line per model.
    Shaded band shows p25–p75 range across MEASURE_STEPS samples.
    Annotates inflection points where latency growth accelerates.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"TinyLlama-1.1B": "#3498db", "OpenLLaMA-3B": "#e74c3c"}
    markers = {"TinyLlama-1.1B": "o", "OpenLLaMA-3B": "s"}

    for model_name, ctx_results in all_results.items():
        if not ctx_results:
            continue
        ctxs    = sorted(ctx_results.keys())
        medians = [ctx_results[c]["median"] for c in ctxs]
        p25s    = [float(np.percentile(ctx_results[c]["samples"], 25)) for c in ctxs]
        p75s    = [float(np.percentile(ctx_results[c]["samples"], 75)) for c in ctxs]
        color   = colors.get(model_name, "#2c3e50")

        ax.plot(ctxs, medians, color=color, marker=markers.get(model_name, "o"),
                linewidth=2, markersize=7, label=f"{model_name} (median)")
        ax.fill_between(ctxs, p25s, p75s, alpha=0.15, color=color)

        # Annotate each point with its value
        for c, m in zip(ctxs, medians):
            ax.annotate(f"{m:.1f}", (c, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5, color=color)

        # Mark inflection: largest absolute jump between consecutive points
        deltas = [medians[i+1] - medians[i] for i in range(len(medians)-1)]
        if deltas:
            max_jump_idx = int(np.argmax(deltas))
            inflect_ctx = ctxs[max_jump_idx + 1]
            inflect_val = medians[max_jump_idx + 1]
            ax.axvline(inflect_ctx, color=color, linestyle=":", alpha=0.5, linewidth=1)
            ax.annotate(f"Inflection\n@{inflect_ctx}",
                        xy=(inflect_ctx, inflect_val),
                        xytext=(inflect_ctx + 30, inflect_val * 1.05),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xscale("log", base=2)
    ax.set_xticks(CONTEXT_LENGTHS)
    ax.set_xticklabels([str(c) for c in CONTEXT_LENGTHS])
    ax.set_xlabel("Context length (tokens in KV-cache)", fontsize=11)
    ax.set_ylabel("Per-token decode latency (ms)", fontsize=11)
    ax.set_title("Scaling with Context Length — Per-Token Decode Latency\n"
                 "(shaded band = p25–p75; dashed line = steepest inflection point)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path}")


def plot_model_size(all_results, out_path):
    """
    Grouped bar chart: side-by-side comparison of 1.1B vs 3B at each context length.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model_names = [m for m in all_results if all_results[m]]
    if len(model_names) < 2:
        print("  Skipping model-size plot (only one model available).")
        return

    ctxs   = sorted(all_results[model_names[0]].keys())
    x      = np.arange(len(ctxs))
    width  = 0.35
    colors = ["#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        medians = [all_results[model_name][c]["median"] for c in ctxs]
        p95s    = [all_results[model_name][c]["p95"]    for c in ctxs]
        yerr    = [p - m for p, m in zip(p95s, medians)]
        bars = ax.bar(x + i * width - width/2, medians, width,
                      label=model_name, color=color, alpha=0.85,
                      yerr=yerr, capsize=4, error_kw={"elinewidth": 1.2})
        for bar, val in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in ctxs])
    ax.set_xlabel("Context length (tokens)", fontsize=11)
    ax.set_ylabel("Per-token decode latency (ms)", fontsize=11)
    ax.set_title("Model Size Scaling — 1.1B vs 3B at Each Context Length\n"
                 "(bars = median; error bars = p95)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path}")


def plot_precision(precision_results, out_path):
    """
    Line plot overlay: float16 vs float32 latency across context lengths.
    Includes a ratio subplot showing the speedup factor.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    colors = {"float16": "#27ae60", "float32": "#e67e22"}
    ctxs   = sorted(next(iter(precision_results.values())).keys())

    medians_by_prec = {}
    for prec, ctx_res in precision_results.items():
        medians = [ctx_res[c]["median"] for c in ctxs]
        medians_by_prec[prec] = medians
        ax1.plot(ctxs, medians, color=colors[prec], marker="o", linewidth=2,
                 markersize=7, label=prec)
        for c, m in zip(ctxs, medians):
            ax1.annotate(f"{m:.1f}", (c, m), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=colors[prec])

    ax1.set_ylabel("Per-token decode latency (ms)", fontsize=11)
    ax1.set_title("Precision Scaling — float16 vs float32\n(TinyLlama-1.1B on MPS)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Ratio subplot: float32 / float16  (speedup of float16 over float32)
    if "float16" in medians_by_prec and "float32" in medians_by_prec:
        ratios = [f32 / f16 for f32, f16 in
                  zip(medians_by_prec["float32"], medians_by_prec["float16"])]
        ax2.bar(ctxs, ratios, color="#8e44ad", alpha=0.75, width=30)
        ax2.axhline(1.0, color="black", linewidth=1, linestyle="--")
        for c, r in zip(ctxs, ratios):
            ax2.text(c, r + 0.02, f"{r:.2f}×", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("float32 / float16\n(speedup ratio)", fontsize=10)
        ax2.set_xlabel("Context length (tokens)", fontsize=11)
        ax2.set_xticks(ctxs)
        ax2.set_xticklabels([str(c) for c in ctxs])
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}  |  Context lengths: {CONTEXT_LENGTHS}\n")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Context length + model size sweep ───────────────────────────────
    # Run both models across all context lengths.
    all_ctx_results = {}    # {model_name: {ctx: {median, p95, p99, samples}}}
    ctx_csv_rows    = []

    for model_name, model_id in MODELS.items():
        print(f"\n[{model_name}] Context length sweep")
        try:
            tokenizer, model = load_model(model_id, device, dtype=torch.float16)
        except Exception as e:
            print(f"  Could not load {model_name}: {e}")
            all_ctx_results[model_name] = {}
            continue

        ctx_results = run_context_sweep(
            tokenizer, model, device,
            CONTEXT_LENGTHS, MEASURE_STEPS, WARMUP_FILLS,
        )
        all_ctx_results[model_name] = ctx_results

        for ctx, res in ctx_results.items():
            ctx_csv_rows.append({
                "model":      model_name,
                "context":    ctx,
                "median_ms":  f"{res['median']:.4f}",
                "p95_ms":     f"{res['p95']:.4f}",
                "p99_ms":     f"{res['p99']:.4f}",
            })

        unload_model(model)
        print(f"  [{model_name}] unloaded.\n")

    save_csv(ctx_csv_rows,
             ["model", "context", "median_ms", "p95_ms", "p99_ms"],
             os.path.join(DATA_DIR, "scaling_context_length.csv"))

    # ── 2. Precision sweep (TinyLlama-1.1B only) ────────────────────────────
    print("\n[Precision sweep]  TinyLlama-1.1B: float16 vs float32")
    precision_results = {}
    prec_csv_rows     = []

    for prec_name, dtype in PRECISIONS.items():
        print(f"\n  Precision: {prec_name}")
        try:
            tokenizer, model = load_model(MODELS["TinyLlama-1.1B"], device, dtype=dtype)
        except Exception as e:
            print(f"  Could not load with {prec_name}: {e}")
            continue

        ctx_results = run_context_sweep(
            tokenizer, model, device,
            CONTEXT_LENGTHS, MEASURE_STEPS, WARMUP_FILLS,
        )
        precision_results[prec_name] = ctx_results

        for ctx, res in ctx_results.items():
            prec_csv_rows.append({
                "precision":  prec_name,
                "context":    ctx,
                "median_ms":  f"{res['median']:.4f}",
                "p95_ms":     f"{res['p95']:.4f}",
                "p99_ms":     f"{res['p99']:.4f}",
            })

        unload_model(model)
        print(f"  [{prec_name}] unloaded.")

    save_csv(prec_csv_rows,
             ["precision", "context", "median_ms", "p95_ms", "p99_ms"],
             os.path.join(DATA_DIR, "scaling_precision.csv"))

    # ── 3. Print summary tables ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  CONTEXT LENGTH SCALING SUMMARY  (median per-token ms)")
    print(f"{'='*62}")
    header = f"  {'Context':>8}" + "".join(f"  {m:>18}" for m in all_ctx_results)
    print(header)
    print(f"  {'-'*58}")
    for ctx in CONTEXT_LENGTHS:
        row = f"  {ctx:>8}"
        for res in all_ctx_results.values():
            val = res.get(ctx, {}).get("median", float("nan"))
            row += f"  {val:>18.2f}"
        print(row)

    print(f"\n{'='*62}")
    print("  PRECISION SCALING SUMMARY  (median per-token ms)")
    print(f"{'='*62}")
    print(f"  {'Context':>8}  {'float16':>10}  {'float32':>10}  {'ratio f32/f16':>14}")
    print(f"  {'-'*50}")
    for ctx in CONTEXT_LENGTHS:
        f16  = precision_results.get("float16", {}).get(ctx, {}).get("median", float("nan"))
        f32  = precision_results.get("float32", {}).get(ctx, {}).get("median", float("nan"))
        ratio = f32 / f16 if f16 and f16 > 0 else float("nan")
        print(f"  {ctx:>8}  {f16:>10.2f}  {f32:>10.2f}  {ratio:>13.2f}x")

    # ── 4. Generate plots ───────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_context_scaling(all_ctx_results,
                         os.path.join(FIGURES_DIR, "scaling_context_length.png"))
    plot_model_size(all_ctx_results,
                    os.path.join(FIGURES_DIR, "scaling_model_size.png"))
    if precision_results:
        plot_precision(precision_results,
                       os.path.join(FIGURES_DIR, "scaling_precision.png"))

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
