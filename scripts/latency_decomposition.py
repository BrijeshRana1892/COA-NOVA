"""
Phase 3: Latency Decomposition via PyTorch Forward Hooks.

Instruments each architectural component of TinyLlama using pre/post hooks
to measure time spent in:
  - Embedding lookup
  - QKV projection (Q, K, V linear layers)
  - Attention core  (KV-cache read/write + scaled dot-product + softmax)
  - Output projection (O linear layer)
  - MLP feed-forward (gate + up + down projections)
  - LayerNorm (RMSNorm, pre- and post-attention)
  - LM head (final vocabulary projection)
  - Sampling (argmax / token selection)
  - Framework overhead (everything else)

METHODOLOGY NOTE:
  MPS (Apple Silicon GPU) executes ops asynchronously. To get accurate
  per-component times we call torch.mps.synchronize() before and after
  each hook boundary. This serializes what would normally be pipelined,
  so the *absolute* times are inflated vs real inference. However, the
  *relative* breakdown (what fraction each component contributes) is
  accurate and is what we care about for bottleneck analysis.

  We measure 30 decode tokens (not the full 128) to keep the hook
  overhead manageable. Results are averaged across all 22 layers and
  all 30 tokens.

Outputs:
  data/decomposition_per_token.csv  — per-component time for each token
  data/decomposition_summary.csv    — mean/median/stdev per component
  figures/decomposition_bar.png     — stacked bar chart per token
  figures/decomposition_pie.png     — pie chart of average breakdown
"""

import time
import csv
import os
import statistics
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "Explain how a computer processor works in simple terms:"
DECODE_TOKENS  = 30          # tokens to profile (hook overhead makes full 128 slow)
DATA_DIR       = "data"
FIGURES_DIR    = "figures"

# Colour palette for components (consistent across bar + pie charts)
COMPONENT_COLORS = {
    "Embedding":        "#2ecc71",
    "LayerNorm":        "#95a5a6",
    "QKV Projection":   "#3498db",
    "Attn Core\n(KV+Softmax)": "#e74c3c",
    "O Projection":     "#9b59b6",
    "MLP":              "#e67e22",
    "LM Head":          "#1abc9c",
    "Sampling":         "#f39c12",
    "Framework OH":     "#bdc3c7",
}


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


# ── Timing hook ────────────────────────────────────────────────────────────────
class TimingHook:
    """
    Attaches to a nn.Module and records wall-clock time for each forward call.

    CONCEPT: PyTorch Forward Hooks
    PyTorch lets you register callbacks that fire before (pre_hook) and after
    (post_hook) a module's forward() method. This is non-invasive — we don't
    need to modify the model source code at all. The hook receives the module,
    its inputs, and (for post hooks) its output.

    We call sync() in both hooks to ensure the GPU has finished all prior
    work before we start the clock, and has finished this module's work
    before we stop it.
    """
    def __init__(self, name: str, device: torch.device):
        self.name    = name
        self.device  = device
        self.samples = []   # list of elapsed times in ms, one per forward call
        self._t0     = None
        self._pre_handle  = None
        self._post_handle = None

    def pre_hook(self, _module, _args):
        sync(self.device)
        self._t0 = time.perf_counter()

    def post_hook(self, _module, _args, _output):
        sync(self.device)
        if self._t0 is not None:
            self.samples.append((time.perf_counter() - self._t0) * 1000)

    def attach(self, module):
        self._pre_handle  = module.register_forward_pre_hook(self.pre_hook)
        self._post_handle = module.register_forward_hook(self.post_hook)
        return self

    def detach(self):
        if self._pre_handle:  self._pre_handle.remove()
        if self._post_handle: self._post_handle.remove()

    def mean_ms(self):
        return statistics.mean(self.samples) if self.samples else 0.0

    def __repr__(self):
        return f"TimingHook({self.name}, n={len(self.samples)}, mean={self.mean_ms():.3f}ms)"


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_id, device):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B params | "
          f"{n_layers} decoder layers | device: {device}\n")
    return tokenizer, model


# ── Register hooks on all relevant modules ─────────────────────────────────────
def register_hooks(model, device):
    """
    Attach TimingHooks to every component we want to measure.

    TinyLlama structure per decoder layer:
      input_layernorm   → RMSNorm (pre-attention)
      self_attn         → full attention module
        ├─ q_proj       → Linear (Q)
        ├─ k_proj       → Linear (K)
        ├─ v_proj       → Linear (V)
        └─ o_proj       → Linear (output)
      post_attention_layernorm → RMSNorm (pre-MLP)
      mlp               → feed-forward network

    We hook self_attn at the top level AND its sub-projections so we can
    compute Attn Core = self_attn − q_proj − k_proj − v_proj − o_proj.
    This residual captures: rotary embeddings, KV-cache reads/writes,
    scaled dot-product attention, and softmax.
    """
    hooks = defaultdict(list)  # component_name → [TimingHook, ...]

    # Embedding
    hooks["embed"].append(
        TimingHook("embed_tokens", device).attach(model.model.embed_tokens)
    )

    # Per-layer hooks
    for i, layer in enumerate(model.model.layers):
        hooks["pre_norm"].append(
            TimingHook(f"layer{i}.input_layernorm", device).attach(layer.input_layernorm)
        )
        hooks["self_attn"].append(
            TimingHook(f"layer{i}.self_attn", device).attach(layer.self_attn)
        )
        hooks["q_proj"].append(
            TimingHook(f"layer{i}.q_proj", device).attach(layer.self_attn.q_proj)
        )
        hooks["k_proj"].append(
            TimingHook(f"layer{i}.k_proj", device).attach(layer.self_attn.k_proj)
        )
        hooks["v_proj"].append(
            TimingHook(f"layer{i}.v_proj", device).attach(layer.self_attn.v_proj)
        )
        hooks["o_proj"].append(
            TimingHook(f"layer{i}.o_proj", device).attach(layer.self_attn.o_proj)
        )
        hooks["post_norm"].append(
            TimingHook(f"layer{i}.post_attention_layernorm", device).attach(layer.post_attention_layernorm)
        )
        hooks["mlp"].append(
            TimingHook(f"layer{i}.mlp", device).attach(layer.mlp)
        )

    # Final norm + LM head
    hooks["final_norm"].append(
        TimingHook("model.norm", device).attach(model.model.norm)
    )
    hooks["lm_head"].append(
        TimingHook("lm_head", device).attach(model.lm_head)
    )

    return hooks


def detach_all(hooks):
    for hook_list in hooks.values():
        for h in hook_list:
            h.detach()


# ── Aggregate hook samples by token step ──────────────────────────────────────
def aggregate_by_token(hooks, n_tokens, _n_layers):
    """
    Each hook fires once per forward call.
    - embed/final_norm/lm_head fire once per token (n_tokens samples each)
    - per-layer hooks fire n_layers times per token

    We reshape per-layer hooks into (n_tokens, n_layers) and sum across layers
    to get the total per-layer-stack cost per token.
    """

    # Helpers to get per-token sums for a list of hooks (one hook per layer)
    def per_layer_per_token(hook_list):
        # Each hook has n_tokens samples (one per decode step)
        # hook_list has n_layers hooks
        # Stack as (n_layers, n_tokens) → sum over layers → (n_tokens,)
        mat = np.array([h.samples[:n_tokens] for h in hook_list])
        return mat.sum(axis=0)   # shape: (n_tokens,)

    embed_t     = np.array(hooks["embed"][0].samples[:n_tokens])
    pre_norm_t  = per_layer_per_token(hooks["pre_norm"])
    self_attn_t = per_layer_per_token(hooks["self_attn"])
    q_t         = per_layer_per_token(hooks["q_proj"])
    k_t         = per_layer_per_token(hooks["k_proj"])
    v_t         = per_layer_per_token(hooks["v_proj"])
    o_t         = per_layer_per_token(hooks["o_proj"])
    post_norm_t = per_layer_per_token(hooks["post_norm"])
    mlp_t       = per_layer_per_token(hooks["mlp"])
    final_t     = np.array(hooks["final_norm"][0].samples[:n_tokens])
    lm_head_t   = np.array(hooks["lm_head"][0].samples[:n_tokens])

    # Derived components
    qkv_t       = q_t + k_t + v_t
    # Attention core = total self_attn time minus the 4 linear projections
    # This residual = rotary embedding + KV-cache read/write + softmax + scaling
    attn_core_t = np.maximum(self_attn_t - qkv_t - o_t, 0)
    layernorm_t = pre_norm_t + post_norm_t

    return {
        "embed_t": embed_t,
        "layernorm_t": layernorm_t,
        "qkv_t": qkv_t,
        "attn_core_t": attn_core_t,
        "o_proj_t": o_t,
        "mlp_t": mlp_t,
        "lm_head_t": lm_head_t,
        "final_t": final_t,
    }


# ── Run profiled inference ─────────────────────────────────────────────────────
def run_profiled_inference(tokenizer, model, device, prompt, n_tokens):
    """
    Run a single generation pass with all hooks active, collecting timing.
    Returns per-token timing dict and per-token total times.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    token_total_times = []   # wall-clock for each full decode step

    # Prefill
    sync(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    sync(device)

    past_kv    = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Decode loop — measure each step end-to-end too
    for _ in range(n_tokens):
        sync(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        sync(device)

        # Sampling: pick next token (separate from model forward)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync(device)

        token_total_times.append((time.perf_counter() - t0) * 1000)
        past_kv = outputs.past_key_values

        if next_token.item() == tokenizer.eos_token_id:
            break

    return token_total_times


# ── Save CSVs ─────────────────────────────────────────────────────────────────
def save_decomposition_csvs(component_data, n_tokens, total_times):
    os.makedirs(DATA_DIR, exist_ok=True)

    labels = {
        "embed_t":     "Embedding",
        "layernorm_t": "LayerNorm",
        "qkv_t":       "QKV Projection",
        "attn_core_t": "Attn Core (KV+Softmax)",
        "o_proj_t":    "O Projection",
        "mlp_t":       "MLP",
        "lm_head_t":   "LM Head",
    }

    # Per-token CSV
    per_tok_path = os.path.join(DATA_DIR, "decomposition_per_token.csv")
    with open(per_tok_path, "w", newline="") as f:
        fieldnames = ["token"] + list(labels.values()) + ["total_ms", "accounted_ms", "framework_oh_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_tokens):
            row = {"token": i + 1}
            accounted = 0.0
            for key, label in labels.items():
                val = float(component_data[key][i])
                row[label] = f"{val:.4f}"
                accounted += val
            total = float(total_times[i])
            row["total_ms"] = f"{total:.4f}"
            row["accounted_ms"] = f"{accounted:.4f}"
            row["framework_oh_ms"] = f"{max(total - accounted, 0):.4f}"
            writer.writerow(row)
    print(f"  Per-token CSV → {per_tok_path}")

    # Summary CSV
    summary_path = os.path.join(DATA_DIR, "decomposition_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "mean_ms", "median_ms",
                                               "stdev_ms", "pct_of_total"])
        writer.writeheader()
        total_mean = float(np.mean(total_times))
        for key, label in labels.items():
            arr = component_data[key][:n_tokens]
            m   = float(np.mean(arr))
            writer.writerow({
                "component": label,
                "mean_ms":   f"{m:.4f}",
                "median_ms": f"{np.median(arr):.4f}",
                "stdev_ms":  f"{np.std(arr):.4f}",
                "pct_of_total": f"{100*m/total_mean:.1f}",
            })
        # Framework overhead
        accounted_per_tok = sum(component_data[k][:n_tokens] for k in labels)
        oh = np.maximum(np.array(total_times) - accounted_per_tok, 0)
        oh_mean = float(np.mean(oh))
        writer.writerow({
            "component": "Framework OH",
            "mean_ms":   f"{oh_mean:.4f}",
            "median_ms": f"{np.median(oh):.4f}",
            "stdev_ms":  f"{np.std(oh):.4f}",
            "pct_of_total": f"{100*oh_mean/total_mean:.1f}",
        })
    print(f"  Summary CSV  → {summary_path}")

    return labels


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_bar(component_data, labels, n_tokens, total_times, out_path):
    """
    Stacked horizontal bar chart: each token is a row, segments show component times.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))

    token_indices = np.arange(1, n_tokens + 1)
    lefts = np.zeros(n_tokens)
    color_list = list(COMPONENT_COLORS.values())

    component_keys = list(labels.keys())
    component_names = list(labels.values())

    for idx, (key, name) in enumerate(zip(component_keys, component_names)):
        vals = component_data[key][:n_tokens].astype(float)
        color = color_list[idx % len(color_list)]
        ax.barh(token_indices, vals, left=lefts, height=0.7,
                color=color, label=name, alpha=0.88)
        lefts += vals

    # Framework overhead
    accounted = sum(component_data[k][:n_tokens] for k in component_keys)
    oh = np.maximum(np.array(total_times) - accounted.astype(float), 0)
    ax.barh(token_indices, oh, left=lefts, height=0.7,
            color=COMPONENT_COLORS["Framework OH"], label="Framework OH", alpha=0.88)

    # Mean total line
    mean_total = float(np.mean(total_times))
    ax.axvline(mean_total, color="black", linewidth=1.5, linestyle="--",
               label=f"Mean total: {mean_total:.1f} ms")

    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Decode step (token index)", fontsize=11)
    ax.set_title("Per-Token Latency Decomposition — TinyLlama-1.1B on MPS\n"
                 "(stacked segments show time in each architectural component)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_ylim(0.5, n_tokens + 0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart    → {out_path}")


def plot_pie(component_data, labels, n_tokens, total_times, out_path):
    """
    Pie chart showing average time allocation across components.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    component_keys  = list(labels.keys())
    component_names = list(labels.values())
    color_list      = list(COMPONENT_COLORS.values())

    means = [float(np.mean(component_data[k][:n_tokens])) for k in component_keys]

    accounted = sum(means)
    total_mean = float(np.mean(total_times))
    oh_mean = max(total_mean - accounted, 0)

    all_labels = component_names + ["Framework OH"]
    all_means  = means + [oh_mean]
    all_colors = color_list[:len(component_names)] + [COMPONENT_COLORS["Framework OH"]]

    # Filter out near-zero slices (< 0.5%) for cleanliness
    total_sum = sum(all_means)
    filtered = [(l, v, c) for l, v, c in zip(all_labels, all_means, all_colors)
                if v / total_sum > 0.005]
    f_labels, f_vals, f_colors = zip(*filtered)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        f_vals,
        labels=None,
        colors=f_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    # Legend with absolute times
    legend_labels = [f"{l}  ({v:.2f} ms)" for l, v in zip(f_labels, f_vals)]
    ax.legend(wedges, legend_labels, loc="lower left", bbox_to_anchor=(0, -0.12),
              fontsize=9, ncol=2)

    ax.set_title(
        "Average Per-Token Latency Breakdown — TinyLlama-1.1B on MPS\n"
        f"(mean total: {total_mean:.2f} ms | {DECODE_TOKENS} decode tokens)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pie chart    → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}\n")

    tokenizer, model = load_model(MODEL_ID, device)
    n_layers = model.config.num_hidden_layers

    # Warm-up (ensures Metal shaders are compiled before hooks fire)
    print("Warm-up run (no hooks)...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(inputs["input_ids"], max_new_tokens=10, do_sample=False)
    sync(device)
    print("  Done.\n")

    # Register hooks
    print(f"Registering timing hooks on {n_layers} decoder layers + embedding + LM head...")
    hooks = register_hooks(model, device)
    total_hooks = sum(len(v) for v in hooks.values())
    print(f"  {total_hooks} hooks attached.\n")

    # Run profiled inference
    print(f"Running profiled inference for {DECODE_TOKENS} decode tokens...")
    total_times = run_profiled_inference(
        tokenizer, model, device, PROMPT, DECODE_TOKENS
    )
    n_actual = len(total_times)
    print(f"  Captured {n_actual} decode tokens.\n")

    # Detach hooks (clean up)
    detach_all(hooks)

    # Aggregate
    component_data = aggregate_by_token(hooks, n_actual, n_layers)

    # Print summary table
    labels_map = {
        "embed_t":     "Embedding",
        "layernorm_t": "LayerNorm",
        "qkv_t":       "QKV Projection",
        "attn_core_t": "Attn Core (KV+Softmax)",
        "o_proj_t":    "O Projection",
        "mlp_t":       "MLP",
        "lm_head_t":   "LM Head",
    }

    total_mean   = float(np.mean(total_times))
    accounted_arr = sum(component_data[k][:n_actual] for k in labels_map)
    oh_arr        = np.maximum(np.array(total_times) - accounted_arr.astype(float), 0)

    print(f"\n{'='*58}")
    print(f"  LATENCY DECOMPOSITION  (mean over {n_actual} decode tokens)")
    print(f"{'='*58}")
    print(f"  {'Component':<28} {'Mean (ms)':>9}  {'% of Total':>10}")
    print(f"  {'-'*50}")
    for key, label in labels_map.items():
        arr  = component_data[key][:n_actual]
        m    = float(np.mean(arr))
        pct  = 100 * m / total_mean
        print(f"  {label:<28} {m:>9.3f}  {pct:>9.1f}%")
    oh_mean = float(np.mean(oh_arr))
    print(f"  {'Framework OH':<28} {oh_mean:>9.3f}  {100*oh_mean/total_mean:>9.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<28} {total_mean:>9.3f}  {'100.0%':>10}")
    print(f"{'='*58}\n")

    # Save outputs
    print("Saving outputs...")
    save_decomposition_csvs(component_data, n_actual, total_times)
    plot_bar(component_data, labels_map, n_actual, total_times,
             os.path.join(FIGURES_DIR, "decomposition_bar.png"))
    plot_pie(component_data, labels_map, n_actual, total_times,
             os.path.join(FIGURES_DIR, "decomposition_pie.png"))

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
