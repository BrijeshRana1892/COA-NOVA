"""
Generate the additional figures referenced by the expanded paper:

  figures/int8_vs_fp16.png   — bar chart: per-token latency + throughput, INT8 vs fp16
  figures/cpu_vs_mps.png     — bar chart: latency + throughput, CPU vs MPS
  figures/roofline.png       — roofline plot with LLM decode operating point marked

Reads:
  data/benchmark_summary.csv  (fp16-MPS baseline)
  data/int8_benchmark.csv     (INT8 result)
  data/cpu_benchmark.csv      (CPU fp16 result)
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, "figures")
DATA = os.path.join(BASE, "data")
os.makedirs(FIG, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def read_kv(path):
    """Read a 2-column metric/value CSV into a dict (string values)."""
    out = {}
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)  # header
        for row in r:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def read_baseline_fp16_mps():
    """Pull the AGGREGATE row out of benchmark_summary.csv."""
    path = os.path.join(DATA, "benchmark_summary.csv")
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    agg = next(r for r in rows if r.get("trial") == "AGGREGATE")
    return {
        "per_token_median_ms": float(agg["median_ms"]),
        "per_token_p95_ms":    float(agg["p95_ms"]),
        "ttft_median_ms":      27.0,  # from paper Table III
    }


# ── INT8 vs fp16 (three-way bar: MPS-fp16, CPU-fp16, CPU-INT8) ─────────────────
def fig_int8_vs_fp16(fp16_mps, cpu_fp16, int8):
    """Plot all three configurations side-by-side. The MPS bar is the absolute
    baseline; the two CPU bars isolate the INT8-vs-fp16 effect on identical
    hardware (since bitsandbytes does not yet accelerate INT8 GEMM on MPS)."""
    mps_lat = fp16_mps["per_token_median_ms"]
    cf16    = float(cpu_fp16["per_token_median_ms"])
    cint8   = float(int8["per_token_median_ms"])

    lats = [mps_lat, cf16, cint8]
    thrs = [1000.0 / x for x in lats]

    labels = ["fp16 (MPS)\nbaseline",
              "fp16 (CPU)",
              "INT8 (CPU,\nqnnpack dyn.)"]
    colors = ["#27ae60", "#3498db", "#e67e22"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars1 = ax1.bar(labels, lats, color=colors, edgecolor="black")
    ax1.set_ylabel("Per-token latency (ms)")
    ax1.set_title("Per-token decode latency (lower is better)")
    for b, v in zip(bars1, lats):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(lats) * 0.01,
                 f"{v:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(labels, thrs, color=colors, edgecolor="black")
    ax2.set_ylabel("Throughput (tokens / sec)")
    ax2.set_title("Decode throughput (higher is better)")
    for b, v in zip(bars2, thrs):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(thrs) * 0.01,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    cpu_ratio = cint8 / cf16
    fig.suptitle(
        f"INT8 vs float16 — TinyLlama-1.1B  (CPU INT8/fp16 ratio = {cpu_ratio:.2f}×)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(FIG, "int8_vs_fp16.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ── CPU vs MPS ─────────────────────────────────────────────────────────────────
def fig_cpu_vs_mps(fp16_mps, cpu):
    mps_lat = fp16_mps["per_token_median_ms"]
    cpu_lat = float(cpu["per_token_median_ms"])
    mps_thr = 1000.0 / mps_lat
    cpu_thr = 1000.0 / cpu_lat

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = ["MPS (Apple GPU)", "CPU"]
    colors = ["#27ae60", "#c0392b"]

    bars1 = ax1.bar(labels, [mps_lat, cpu_lat], color=colors, edgecolor="black")
    ax1.set_ylabel("Per-token latency (ms)")
    ax1.set_title("Per-token decode latency")
    for b, v in zip(bars1, [mps_lat, cpu_lat]):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(mps_lat, cpu_lat) * 0.01,
                 f"{v:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(labels, [mps_thr, cpu_thr], color=colors, edgecolor="black")
    ax2.set_ylabel("Throughput (tokens / sec)")
    ax2.set_title("Decode throughput")
    for b, v in zip(bars2, [mps_thr, cpu_thr]):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(mps_thr, cpu_thr) * 0.01,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    speedup = cpu_lat / mps_lat
    fig.suptitle(f"MPS vs CPU — TinyLlama-1.1B float16 (MPS speedup = {speedup:.2f}×)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG, "cpu_vs_mps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ── Roofline ───────────────────────────────────────────────────────────────────
def fig_roofline():
    """
    Roofline plot for an Apple Silicon-class GPU. Anchors:
      - Peak compute (single-precision FP) ≈ 2.6 TFLOP/s (M2-class)
      - Peak memory bandwidth ≈ 100 GB/s (unified LPDDR5)
    Operating points marked:
      - LLM decode (TinyLlama-1.1B fp16): AI ≈ 1 FLOP/byte
      - LLM decode INT8 (estimated):       AI ≈ 2 FLOP/byte
      - GEMM training-class (high AI):     AI ≈ 100 FLOP/byte
    """
    peak_flops = 2.6e12          # FLOP/s
    peak_bw    = 100e9           # bytes/s
    ridge      = peak_flops / peak_bw   # FLOP/byte at intersection (~26)

    ai = np.logspace(-1, 3, 400)
    perf = np.minimum(peak_flops, peak_bw * ai)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ai, perf / 1e9, color="#2c3e50", linewidth=2.2, label="Roofline")
    ax.axhline(peak_flops / 1e9, color="#7f8c8d", linestyle=":", linewidth=1.2)
    ax.axvline(ridge, color="#7f8c8d", linestyle=":", linewidth=1.2,
               label=f"Ridge AI = {ridge:.1f} FLOP/byte")

    pts = [
        ("LLM decode\n(fp16)", 1.0,   "#e74c3c"),
        ("LLM decode\n(INT8 est.)", 2.0, "#e67e22"),
        ("GEMM (training class)", 100.0, "#27ae60"),
    ]
    for label, x, color in pts:
        y = min(peak_flops, peak_bw * x) / 1e9
        ax.plot(x, y, "o", color=color, markersize=10, markeredgecolor="black",
                markeredgewidth=1.0)
        ax.annotate(label, xy=(x, y), xytext=(x * 1.4, y / 2.2),
                    fontsize=9, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Attainable performance (GFLOP / s)")
    ax.set_title("Roofline Model — Apple Silicon-class GPU\n"
                 "LLM decode is firmly memory-bandwidth bound",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1, peak_flops / 1e9 * 2)

    plt.tight_layout()
    out = os.path.join(FIG, "roofline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    fp16 = read_baseline_fp16_mps()
    int8 = read_kv(os.path.join(DATA, "int8_benchmark.csv"))
    cpu  = read_kv(os.path.join(DATA, "cpu_benchmark.csv"))

    print("Generating new figures...")
    fig_int8_vs_fp16(fp16, cpu, int8)
    fig_cpu_vs_mps(fp16, cpu)
    fig_roofline()
    print("Done.")


if __name__ == "__main__":
    main()
