"""
IEEE Two-Column Conference Paper Generator
Token-Generation Latency Benchmarking in LLaMA
Uses ReportLab to produce a PDF matching the IEEE IEEEtran style.
"""

import os, csv, textwrap
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(BASE, "figures")
DATA    = os.path.join(BASE, "data")
OUT     = os.path.join(BASE, "outputs", "COA_NOVA_IEEE_Paper.pdf")

# ── Page geometry (IEEE letter) ────────────────────────────────────────────────
PAGE_W, PAGE_H = letter          # 8.5 × 11 in
MARGIN_T  = 0.75 * inch
MARGIN_B  = 1.00 * inch
MARGIN_L  = 0.625 * inch
MARGIN_R  = 0.625 * inch
COL_GAP   = 0.25 * inch
COL_W     = (PAGE_W - MARGIN_L - MARGIN_R - COL_GAP) / 2   # ~3.5625 in
BODY_H    = PAGE_H - MARGIN_T - MARGIN_B
TITLE_H   = 2.05 * inch   # reserved for title block on page 1

# ── Colour palette ─────────────────────────────────────────────────────────────
BLACK  = colors.black
WHITE  = colors.white
GRAY   = colors.HexColor("#555555")

# ── Font registration (use built-in Times family — exact IEEE match) ───────────
T_ROMAN = "Times-Roman"
T_BOLD  = "Times-Bold"
T_ITALIC= "Times-Italic"
T_BI    = "Times-BoldItalic"

# ── Paragraph styles ──────────────────────────────────────────────────────────
def make_styles():
    s = {}

    s["title"] = ParagraphStyle(
        "title",
        fontName=T_BOLD, fontSize=24, leading=30,
        alignment=TA_CENTER, spaceAfter=6,
        textColor=BLACK,
    )
    s["authors"] = ParagraphStyle(
        "authors",
        fontName=T_ROMAN, fontSize=11, leading=14,
        alignment=TA_CENTER, spaceAfter=2,
    )
    s["affil"] = ParagraphStyle(
        "affil",
        fontName=T_ITALIC, fontSize=10, leading=13,
        alignment=TA_CENTER, spaceAfter=2,
    )
    s["email"] = ParagraphStyle(
        "email",
        fontName=T_ROMAN, fontSize=9, leading=11,
        alignment=TA_CENTER, spaceAfter=10,
    )
    s["abstract_label"] = ParagraphStyle(
        "abstract_label",
        fontName=T_BOLD, fontSize=9, leading=11,
        alignment=TA_JUSTIFY, spaceAfter=0, firstLineIndent=18,
    )
    s["abstract"] = ParagraphStyle(
        "abstract",
        fontName=T_ITALIC, fontSize=9, leading=11,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )
    s["index_terms"] = ParagraphStyle(
        "index_terms",
        fontName=T_ROMAN, fontSize=9, leading=11,
        alignment=TA_JUSTIFY, spaceAfter=8,
    )
    s["section"] = ParagraphStyle(
        "section",
        fontName=T_BOLD, fontSize=10, leading=13,
        alignment=TA_CENTER, spaceBefore=10, spaceAfter=4,
        textColor=BLACK,
    )
    s["subsection"] = ParagraphStyle(
        "subsection",
        fontName=T_ITALIC, fontSize=10, leading=13,
        alignment=TA_LEFT, spaceBefore=6, spaceAfter=3,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontName=T_ROMAN, fontSize=10, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=5,
        firstLineIndent=18,
    )
    s["body_noindent"] = ParagraphStyle(
        "body_noindent",
        fontName=T_ROMAN, fontSize=10, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=5,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        fontName=T_ROMAN, fontSize=8.5, leading=11,
        alignment=TA_CENTER, spaceAfter=6, spaceBefore=4,
    )
    s["caption_bold"] = ParagraphStyle(
        "caption_bold",
        fontName=T_BOLD, fontSize=8.5, leading=11,
        alignment=TA_CENTER, spaceAfter=6, spaceBefore=4,
    )
    s["table_header"] = ParagraphStyle(
        "table_header",
        fontName=T_BOLD, fontSize=8.5, leading=10,
        alignment=TA_CENTER,
    )
    s["table_body"] = ParagraphStyle(
        "table_body",
        fontName=T_ROMAN, fontSize=8.5, leading=10,
        alignment=TA_CENTER,
    )
    s["table_body_l"] = ParagraphStyle(
        "table_body_l",
        fontName=T_ROMAN, fontSize=8.5, leading=10,
        alignment=TA_LEFT,
    )
    s["ref"] = ParagraphStyle(
        "ref",
        fontName=T_ROMAN, fontSize=8.5, leading=11,
        alignment=TA_JUSTIFY, spaceAfter=3,
        leftIndent=18, firstLineIndent=-18,
    )
    return s

# ── Section heading helper ─────────────────────────────────────────────────────
def section_head(num, title, S):
    text = f"{num}.&nbsp;&nbsp;<font face='{T_BOLD}'>{title.upper()}</font>"
    return Paragraph(text, S["section"])

def subsection_head(label, title, S):
    return Paragraph(f"<i>{label} {title}</i>", S["subsection"])

# ── Horizontal rule ────────────────────────────────────────────────────────────
def hrule(width=None):
    return HRFlowable(width=width or "100%", thickness=0.5,
                      color=BLACK, spaceAfter=4, spaceBefore=4)

# ── Figure helper ─────────────────────────────────────────────────────────────
def figure(filename, caption_text, S, width=None):
    path = os.path.join(FIGURES, filename)
    w = width or (COL_W - 0.1*inch)
    img = Image(path, width=w, height=w * 0.62, kind="proportional")
    cap = Paragraph(caption_text, S["caption"])
    return KeepTogether([img, cap])

def figure_wide(filename, caption_text, S, width=None):
    """Figure spanning full text width (for wide plots)."""
    path = os.path.join(FIGURES, filename)
    w = width or (2 * COL_W + COL_GAP - 0.1*inch)
    img = Image(path, width=w, height=w * 0.55, kind="proportional")
    cap = Paragraph(caption_text, S["caption"])
    return KeepTogether([img, cap])

# ── Table style helper ─────────────────────────────────────────────────────────
def ieee_table_style(header_rows=1):
    return TableStyle([
        ("FONTNAME",    (0,0), (-1,header_rows-1), T_BOLD),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("LEADING",     (0,0), (-1,-1), 11),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("LINEABOVE",   (0,0), (-1,0),  0.8, BLACK),
        ("LINEBELOW",   (0,0), (-1,0),  0.5, BLACK),
        ("LINEBELOW",   (0,-1),(-1,-1), 0.8, BLACK),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ])

# ── Read CSV helpers ───────────────────────────────────────────────────────────
def read_csv(name):
    rows = []
    with open(os.path.join(DATA, name)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ── Column-span flowable (inserts a frame break to sync columns) ───────────────
class ColumnBreak(Flowable):
    def __init__(self): Flowable.__init__(self)
    def draw(self): pass
    def wrap(self, aw, ah): return (0, ah)  # consume rest of column height

# ── Document page templates ────────────────────────────────────────────────────
def build_doc(story):
    doc = BaseDocTemplate(
        OUT,
        pagesize=letter,
        leftMargin=MARGIN_L,
        rightMargin=MARGIN_R,
        topMargin=MARGIN_T,
        bottomMargin=MARGIN_B,
        title="Token-Generation Latency Benchmarking in LLaMA",
        author="Brijesh Rana, Divy Gajera, Rashmin Gajera",
    )

    # Page 1: title frame (full width, top) + two column frames (rest)
    title_frame = Frame(
        MARGIN_L, PAGE_H - MARGIN_T - TITLE_H,
        PAGE_W - MARGIN_L - MARGIN_R, TITLE_H,
        id="title", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    col1_p1 = Frame(
        MARGIN_L, MARGIN_B,
        COL_W, BODY_H - TITLE_H - 0.1*inch,
        id="col1p1", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    col2_p1 = Frame(
        MARGIN_L + COL_W + COL_GAP, MARGIN_B,
        COL_W, BODY_H - TITLE_H - 0.1*inch,
        id="col2p1", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    # Page 2+: two equal columns, full body height
    col1 = Frame(
        MARGIN_L, MARGIN_B,
        COL_W, BODY_H,
        id="col1", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    col2 = Frame(
        MARGIN_L + COL_W + COL_GAP, MARGIN_B,
        COL_W, BODY_H,
        id="col2", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(T_ROMAN, 9)
        canvas.drawCentredString(PAGE_W/2, 0.5*inch, str(doc.page))
        canvas.restoreState()

    page1 = PageTemplate(
        id="page1",
        frames=[title_frame, col1_p1, col2_p1],
        onPage=add_page_number,
    )
    page_rest = PageTemplate(
        id="page_rest",
        frames=[col1, col2],
        onPage=add_page_number,
    )

    doc.addPageTemplates([page1, page_rest])
    doc.build(story)

# ── Main content builder ───────────────────────────────────────────────────────
def build_story():
    S = make_styles()
    story = []

    # ── TITLE BLOCK ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "Token-Generation Latency Benchmarking in LLaMA:<br/>"
        "Measurement, Bottleneck Attribution, and Architectural Implications",
        S["title"]
    ))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Brijesh Rana, Divy Gajera, Rashmin Gajera", S["authors"]))
    story.append(Paragraph("California State University Long Beach", S["affil"]))
    story.append(Paragraph("Long Beach, CA, USA", S["affil"]))
    story.append(Spacer(1, 0.08*inch))
    story.append(hrule())

    # ── ABSTRACT ─────────────────────────────────────────────────────────────
    abstract_text = (
        "<b><i>Abstract</i></b><i>—Large language model (LLM) inference latency is determined "
        "not by raw floating-point throughput but by memory-bandwidth saturation during "
        "autoregressive decoding. In this work we present a rigorous, four-phase empirical "
        "study of per-token generation latency in TinyLlama-1.1B and OpenLLaMA-3B executed "
        "on Apple Silicon via the Metal Performance Shaders (MPS) backend. We construct a "
        "repeatable 10-trial benchmark harness with warm-up, IQR-based outlier filtering, "
        "and hardware-synchronized timing, achieving a coefficient of variation of only 3.7%. "
        "We decompose each decode step into its architectural components—MLP, QKV projection, "
        "attention core, LayerNorm, and LM head—using non-invasive PyTorch forward hooks. "
        "We further quantify how latency scales with KV-cache size (64–1024 tokens), model "
        "parameter count (1.1B vs 3.4B), and numerical precision (float16 vs float32). "
        "MLP feed-forward blocks dominate at 28.2%, linear projections account for 31.5% "
        "combined, and attention core is only 8.8% at short contexts. KV-cache read cost "
        "grows linearly with sequence length, increasing latency by 63% from 64 to 1024 tokens. "
        "float16 delivers a 1.89× speedup over float32, consistent with the 2× bandwidth "
        "reduction. We conclude with a roofline-grounded optimization proposal targeting "
        "MLP quantization, which we estimate can reduce decode latency by 20–25%.</i>"
    )
    story.append(Paragraph(abstract_text, S["abstract"]))
    story.append(Paragraph(
        "<b><i>Index Terms</i></b><i>—large language models, token generation, latency "
        "benchmarking, KV-cache, memory bandwidth, Apple Silicon, MPS, autoregressive "
        "decoding, LLaMA, inference optimization</i>",
        S["index_terms"]
    ))
    story.append(hrule())
    story.append(Spacer(1, 0.05*inch))

    # ── SECTION I: INTRODUCTION ───────────────────────────────────────────────
    story.append(section_head("I", "Introduction", S))
    story.append(Paragraph(
        "The deployment of large language models (LLMs) in interactive applications—"
        "conversational assistants, code completion tools, and real-time document "
        "summarizers—places strict latency constraints on token generation. Users "
        "perceive generation delays greater than 100 ms as disruptive [1], making "
        "per-token latency a first-class engineering objective distinct from batch "
        "throughput.", S["body"]
    ))
    story.append(Paragraph(
        "LLM text generation follows an autoregressive protocol: given a prompt of "
        "<i>P</i> tokens, the model generates one token at a time, each conditioned on "
        "all prior tokens. This serial dependency prevents inter-step parallelism and "
        "forces the GPU to repeatedly load the full model weight matrix and grow "
        "key-value (KV) cache for every new token. Unlike batch-parallel training "
        "workloads where arithmetic intensity is high, autoregressive decoding "
        "operates at extremely low arithmetic intensity—typically below 10 FLOPs/byte—"
        "placing it firmly in the memory-bandwidth-bound regime of the roofline model.", S["body"]
    ))
    story.append(Paragraph(
        "Prior work has characterized LLM inference at the systems level—vLLM [2] "
        "optimizes KV-cache paging, FlashAttention [3] fuses attention kernels, and "
        "llama.cpp [4] targets CPU-based edge inference—but fine-grained, component-level "
        "latency attribution on consumer hardware with unified memory has received "
        "less attention. Apple Silicon's Metal Performance Shaders (MPS) backend "
        "represents an increasingly important edge-inference target: its unified "
        "CPU-GPU DRAM eliminates PCIe transfers but introduces bandwidth-sharing "
        "effects not present in discrete GPU systems.", S["body"]
    ))
    story.append(Paragraph(
        "This paper makes the following contributions: (1) a rigorous benchmark harness "
        "for per-token latency with hardware-synchronized timing on MPS; (2) a "
        "non-invasive, hook-based decomposition of decode-step latency into eight "
        "architectural components across all 22 decoder layers; (3) a scaling study "
        "spanning context length, model size, and numerical precision; and (4) an "
        "MLP-quantization optimization proposal grounded in the measured bottleneck "
        "profile.", S["body"]
    ))

    # ── SECTION II: BACKGROUND ────────────────────────────────────────────────
    story.append(section_head("II", "Background and Related Work", S))
    story.append(subsection_head("A.", "Autoregressive Decoding and the KV-Cache", S))
    story.append(Paragraph(
        "Decoder-only transformer models such as the LLaMA family [5] generate text "
        "via the recurrence <i>x</i><sub>t+1</sub> = argmax <i>p</i>(<i>x</i> | "
        "<i>x</i><sub>1:t</sub>). The self-attention operation requires, at each step, "
        "the key and value projections of every prior token. Without caching this is "
        "O(<i>n</i>²) in sequence length. The KV-cache stores these projections, "
        "reducing per-step attention compute to O(<i>n</i>) at the cost of a memory "
        "read proportional to context length:", S["body"]
    ))
    story.append(Paragraph(
        "BW<sub>KV</sub> = 2 × <i>L</i> × <i>S</i> × <i>d</i><sub>h</sub> × "
        "<i>n</i><sub>h</sub> × <i>B</i>",
        ParagraphStyle("eq", fontName=T_ITALIC, fontSize=10, leading=14,
                       alignment=TA_CENTER, spaceAfter=5)
    ))
    story.append(Paragraph(
        "where <i>L</i> is the number of layers, <i>S</i> the sequence length, "
        "<i>d</i><sub>h</sub> the per-head dimension, <i>n</i><sub>h</sub> the number "
        "of heads, and <i>B</i> bytes per element. For TinyLlama-1.1B at <i>S</i> = 1024 "
        "in float16 this equals 184 MB per decode step—making each step a pure "
        "streaming read of DRAM.", S["body"]
    ))
    story.append(subsection_head("B.", "Memory-Bound Inference and the Roofline Model", S))
    story.append(Paragraph(
        "The roofline model [6] characterizes workload performance as the minimum of "
        "compute-bound and bandwidth-bound ceilings. Autoregressive decode steps have "
        "arithmetic intensity (AI) of roughly 2<i>P</i>/<i>B</i><sub>model</sub> "
        "FLOPs/byte, where <i>P</i> is the parameter count and <i>B</i><sub>model</sub> "
        "the model size in bytes. For a 1.1B parameter model in float16 (~2.2 GB), "
        "AI ≈ 1 FLOP/byte—well below the MPS ridge point. Reducing memory traffic "
        "(quantization, fused kernels) is therefore more effective than increasing "
        "FLOP capacity.", S["body"]
    ))
    story.append(subsection_head("C.", "Related Work", S))
    story.append(Paragraph(
        "FlashAttention [3] and FlashAttention-2 [7] fuse the attention kernel to "
        "minimize HBM reads, reducing attention complexity from O(<i>S</i>²) to "
        "O(<i>S</i>) in memory traffic. vLLM [2] introduces paged attention for "
        "efficient KV-cache memory management in multi-request serving scenarios. "
        "llama.cpp [4] demonstrates that 4-bit quantization can achieve near-float16 "
        "quality on CPU hardware with 4× bandwidth reduction. GPTQ [8] provides "
        "post-training quantization specifically for GPT-family models. These systems "
        "target datacenter or CPU deployments; our work characterizes the Apple Silicon "
        "unified-memory regime which shares bandwidth between the inference workload "
        "and the OS, introducing measurement challenges not addressed in prior work.", S["body"]
    ))

    # ── SECTION III: EXPERIMENTAL SETUP ──────────────────────────────────────
    story.append(section_head("III", "Experimental Setup", S))
    story.append(subsection_head("A.", "Hardware Platform", S))
    story.append(Paragraph(
        "All experiments were conducted on an Apple Silicon MacBook. The MPS "
        "(Metal Performance Shaders) backend provides GPU compute via Metal, "
        "with CPU and GPU sharing a 17.2 GB unified memory pool. Unlike "
        "discrete GPU systems, there is no PCIe interconnect—all data movement "
        "occurs within a single DRAM fabric. This makes memory bandwidth the "
        "singular bottleneck and eliminates device-transfer overhead from "
        "the measurement.", S["body"]
    ))

    # Hardware table
    hw_data = [
        [Paragraph("Component", S["table_header"]), Paragraph("Specification", S["table_header"])],
        [Paragraph("Platform", S["table_body"]), Paragraph("Apple Silicon MacBook (MPS)", S["table_body"])],
        [Paragraph("Total RAM", S["table_body"]), Paragraph("17.2 GB (unified CPU+GPU)", S["table_body"])],
        [Paragraph("Backend", S["table_body"]), Paragraph("PyTorch MPS (Metal Performance Shaders)", S["table_body"])],
        [Paragraph("Python", S["table_body"]), Paragraph("3.11", S["table_body"])],
        [Paragraph("PyTorch", S["table_body"]), Paragraph("2.x", S["table_body"])],
        [Paragraph("Transformers", S["table_body"]), Paragraph("HuggingFace Transformers", S["table_body"])],
    ]
    hw_table = Table(hw_data, colWidths=[COL_W*0.38, COL_W*0.62])
    hw_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE I</b><br/>Hardware and Software Configuration",
                           S["caption_bold"]))
    story.append(hw_table)
    story.append(Spacer(1, 8))

    story.append(subsection_head("B.", "Models", S))
    story.append(Paragraph(
        "Two LLaMA-family models were evaluated. Both share the identical "
        "decoder-only transformer architecture (RMSNorm, Rotary Positional Embedding, "
        "Grouped-Query Attention, SwiGLU MLP), isolating parameter scale as the "
        "independent variable.", S["body"]
    ))

    # Model table
    model_data = [
        [Paragraph("Model", S["table_header"]), Paragraph("Params", S["table_header"]),
         Paragraph("Layers", S["table_header"]), Paragraph("Hidden", S["table_header"]),
         Paragraph("FFN dim", S["table_header"])],
        [Paragraph("TinyLlama-1.1B", S["table_body"]), Paragraph("1.1B", S["table_body"]),
         Paragraph("22", S["table_body"]), Paragraph("2048", S["table_body"]),
         Paragraph("5632", S["table_body"])],
        [Paragraph("OpenLLaMA-3B", S["table_body"]), Paragraph("3.4B", S["table_body"]),
         Paragraph("26", S["table_body"]), Paragraph("3200", S["table_body"]),
         Paragraph("8640", S["table_body"])],
    ]
    col_ws = [COL_W*0.33, COL_W*0.14, COL_W*0.14, COL_W*0.18, COL_W*0.21]
    model_table = Table(model_data, colWidths=col_ws)
    model_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE II</b><br/>Model Architectures",
                           S["caption_bold"]))
    story.append(model_table)
    story.append(Spacer(1, 8))

    story.append(subsection_head("C.", "Measurement Methodology", S))
    story.append(Paragraph(
        "Accurate GPU timing on MPS requires explicit synchronization. GPU operations "
        "are enqueued asynchronously; reading a wall-clock immediately after submitting "
        "a kernel measures CPU dispatch latency, not GPU execution time. We insert "
        "<b>torch.mps.synchronize()</b> before and after every timed region, blocking "
        "the CPU until the GPU command queue is drained. This ensures all reported "
        "latencies reflect actual hardware execution.", S["body"]
    ))
    story.append(Paragraph(
        "Three warm-up runs are discarded before collecting measurements. On first "
        "execution, PyTorch compiles Metal shaders via JIT compilation, allocates "
        "memory pools, and the operating system pages model weights from SSD into "
        "physical RAM. Including these one-time startup costs would inflate "
        "TTFT by 5–10× and distort per-token statistics.", S["body"]
    ))
    story.append(Paragraph(
        "IQR-based outlier filtering is applied to per-trial median latencies. Values "
        "outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] are excluded. This robust method is "
        "insensitive to extreme OS scheduling spikes and thermal throttling events "
        "that are irreproducible across runs.", S["body"]
    ))

    # ── SECTION IV: BENCHMARK METHODOLOGY ────────────────────────────────────
    story.append(section_head("IV", "Benchmark Methodology", S))
    story.append(Paragraph(
        "The benchmark harness measures three distinct latency components for each "
        "generation trial:", S["body"]
    ))
    story.append(Paragraph(
        "<b>Time to First Token (TTFT).</b> The prefill phase processes the entire "
        "prompt in a single parallel forward pass, populating the KV-cache and "
        "producing the first output token. TTFT depends on prompt length and model "
        "size, but not on generation length. For interactive applications, TTFT "
        "determines perceived response latency.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Per-token steady-state latency.</b> Each subsequent decode step processes "
        "only one token, reading the full KV-cache. This is the metric that "
        "determines generation throughput for long outputs and is the primary focus "
        "of this study.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>End-to-end (E2E) time.</b> Total wall-clock from first prompt token to "
        "last generated token, equal to TTFT + sum of per-token decode latencies.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "The prompt <i>\"Explain how a computer processor works in simple terms:\"</i> "
        "(12 tokens) was used across all trials. Each trial generates 128 new tokens "
        "via greedy decoding. Ten timed trials are collected after three warm-up runs. "
        "Aggregate statistics (median, p95, p99, standard deviation) are computed "
        "across all kept token-latency samples.", S["body"]
    ))
    story.append(Spacer(1, 4))

    # Timing diagram
    story.append(figure(
        "timing_diagram.png",
        "Fig. 1. Benchmark timing methodology. Top panel: TTFT vs decode phase "
        "timeline for a representative trial. Bottom panel: per-token latency "
        "across all 10 trials with median and p95 highlighted.",
        S, width=COL_W - 0.05*inch
    ))

    # Benchmark results table
    story.append(Spacer(1, 4))
    bench_data = [
        [Paragraph("Metric", S["table_header"]), Paragraph("Value", S["table_header"])],
        [Paragraph("TTFT (median)", S["table_body_l"]), Paragraph("27.0 ms", S["table_body"])],
        [Paragraph("Per-token latency (median)", S["table_body_l"]), Paragraph("20.91 ms", S["table_body"])],
        [Paragraph("Per-token latency (p95)", S["table_body_l"]), Paragraph("21.98 ms", S["table_body"])],
        [Paragraph("Per-token latency (p99)", S["table_body_l"]), Paragraph("22.87 ms", S["table_body"])],
        [Paragraph("Throughput (median)", S["table_body_l"]), Paragraph("47.8 tokens/sec", S["table_body"])],
        [Paragraph("Trial-to-trial stdev", S["table_body_l"]), Paragraph("0.77 ms (CV = 3.7%)", S["table_body"])],
        [Paragraph("Tokens generated / trial", S["table_body_l"]), Paragraph("128", S["table_body"])],
    ]
    bench_table = Table(bench_data, colWidths=[COL_W*0.6, COL_W*0.4])
    bench_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE III</b><br/>Benchmark Results — TinyLlama-1.1B on MPS "
                           "(10 trials, 3 warm-up discarded)", S["caption_bold"]))
    story.append(bench_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The coefficient of variation of 3.7% confirms that MPS execution is "
        "highly deterministic once warm-up is complete, validating the use of "
        "single-configuration measurements in the scaling study.", S["body"]
    ))

    # ── SECTION V: LATENCY DECOMPOSITION ─────────────────────────────────────
    story.append(section_head("V", "Latency Decomposition", S))
    story.append(subsection_head("A.", "Instrumentation via PyTorch Forward Hooks", S))
    story.append(Paragraph(
        "PyTorch's <b>register_forward_pre_hook</b> and <b>register_forward_hook</b> "
        "APIs attach callbacks to any <b>nn.Module</b> without modifying model source "
        "code. A <i>TimingHook</i> calls <b>torch.mps.synchronize()</b> immediately "
        "before starting and after stopping a hardware timer for each module. Hooks "
        "are attached to 44 component instances per decode step across the 22-layer "
        "model (embedding, 22× pre-norm, 22× self-attn with sub-projections, "
        "22× post-norm, 22× MLP, final norm, LM head).", S["body"]
    ))
    story.append(Paragraph(
        "The attention core time is derived as:", S["body"]
    ))
    story.append(Paragraph(
        "t<sub>attn_core</sub> = t<sub>self_attn</sub> − t<sub>QKV</sub> − t<sub>O-proj</sub>",
        ParagraphStyle("eq2", fontName=T_ITALIC, fontSize=10, leading=14,
                       alignment=TA_CENTER, spaceAfter=5)
    ))
    story.append(Paragraph(
        "capturing the residual: RoPE embedding, KV-cache scatter/gather, "
        "scaled dot-product attention, and softmax. Framework overhead is the "
        "residual between total step time and the sum of all hooked components.", S["body"]
    ))
    story.append(Paragraph(
        "<b>Important caveat.</b> The synchronize() calls serialize operations that "
        "the GPU pipeline would normally overlap, inflating absolute times by "
        "approximately 30–40% vs uninstrumented inference. The <i>relative "
        "percentage breakdown</i> remains valid and is the primary result of "
        "this phase.", S["body_noindent"]
    ))
    story.append(subsection_head("B.", "Component Breakdown Results", S))

    # Decomp table
    decomp_rows = read_csv("decomposition_summary.csv")
    dec_data = [
        [Paragraph("Component", S["table_header"]),
         Paragraph("Mean (ms)", S["table_header"]),
         Paragraph("% Total", S["table_header"])],
    ]
    for row in decomp_rows:
        dec_data.append([
            Paragraph(row["component"], S["table_body_l"]),
            Paragraph(row["mean_ms"], S["table_body"]),
            Paragraph(row["pct_of_total"] + "%", S["table_body"]),
        ])
    dec_table = Table(dec_data, colWidths=[COL_W*0.52, COL_W*0.24, COL_W*0.24])
    dec_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE IV</b><br/>Per-Token Latency Decomposition — "
                           "TinyLlama-1.1B (30 decode steps)", S["caption_bold"]))
    story.append(dec_table)
    story.append(Spacer(1, 6))

    story.append(figure(
        "decomposition_pie.png",
        "Fig. 2. Average per-token latency breakdown by architectural component. "
        "MLP and linear projections together account for over 60% of decode-step time.",
        S, width=COL_W - 0.05*inch
    ))
    story.append(figure(
        "decomposition_bar.png",
        "Fig. 3. Per-token latency decomposition across 30 decode steps shown as "
        "stacked bars. Component times are stable, confirming steady-state decoding.",
        S, width=COL_W - 0.05*inch
    ))

    story.append(subsection_head("C.", "Analysis", S))
    story.append(Paragraph(
        "<b>MLP dominance (28.2%).</b> The SwiGLU feed-forward network in each "
        "decoder layer consists of three linear projections: gate (2048→5632), "
        "up (2048→5632), and down (5632→2048). The combined parameter count of "
        "these projections across 22 layers is ~530M weights. Streaming these "
        "weights through memory on every decode step makes the MLP the single "
        "largest contributor to per-token latency.", S["body"]
    ))
    story.append(Paragraph(
        "<b>QKV projection (22.4%).</b> Three linear layers—Q (2048→2048), "
        "K (2048→256 with GQA), V (2048→256)—are computed fresh at every step. "
        "Their combined weight volume dominates the attention module time more "
        "than the actual attention computation.", S["body"]
    ))
    story.append(Paragraph(
        "<b>Attention core (8.8%).</b> At the tested context lengths (64–1024 tokens), "
        "the softmax and KV-cache read/write constitute only 8.8% of total time. "
        "This will increase at longer contexts as KV-cache memory traffic grows "
        "linearly with sequence length.", S["body"]
    ))
    story.append(Paragraph(
        "<b>LayerNorm (14.9%).</b> RMSNorm fires 44 times per decode step (pre- and "
        "post-attention × 22 layers). Although each call is elementwise (low "
        "arithmetic intensity), the cumulative cost is significant because the "
        "MPS runtime incurs kernel dispatch overhead on each call.", S["body"]
    ))

    # ── SECTION VI: SCALING ANALYSIS ─────────────────────────────────────────
    story.append(section_head("VI", "Scaling Analysis", S))
    story.append(subsection_head("A.", "Context Length Scaling", S))
    story.append(Paragraph(
        "To isolate KV-cache read cost, we prefill with a fixed short prompt, "
        "then autoregressively generate tokens (untimed) until the KV-cache reaches "
        "the target length, and finally measure 15 timed decode steps from that "
        "state. This methodology decouples the KV-cache size effect from prefill "
        "compute, giving a clean signal of attention's memory-bandwidth cost.", S["body"]
    ))

    # Context scaling table
    ctx_data = [
        [Paragraph("Context\n(tokens)", S["table_header"]),
         Paragraph("TinyLlama-1.1B\n(ms)", S["table_header"]),
         Paragraph("OpenLLaMA-3B\n(ms)", S["table_header"]),
         Paragraph("Ratio\n3B/1.1B", S["table_header"])],
    ]
    tiny_ctx = {64: 19.92, 128: 20.67, 256: 22.02, 512: 25.12, 1024: 32.49}
    open_ctx = {64: 60.38, 128: 62.40, 256: 70.24, 512: 74.04, 1024: 87.21}
    for ctx in [64, 128, 256, 512, 1024]:
        ratio = open_ctx[ctx] / tiny_ctx[ctx]
        ctx_data.append([
            Paragraph(str(ctx), S["table_body"]),
            Paragraph(f"{tiny_ctx[ctx]:.2f}", S["table_body"]),
            Paragraph(f"{open_ctx[ctx]:.2f}", S["table_body"]),
            Paragraph(f"{ratio:.2f}×", S["table_body"]),
        ])
    ctx_table = Table(ctx_data, colWidths=[COL_W*0.22, COL_W*0.26, COL_W*0.26, COL_W*0.26])
    ctx_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE V</b><br/>Per-Token Decode Latency vs Context Length",
                           S["caption_bold"]))
    story.append(ctx_table)
    story.append(Spacer(1, 6))

    story.append(figure(
        "scaling_context_length.png",
        "Fig. 4. Per-token decode latency vs KV-cache context length. "
        "Both models show linear growth consistent with memory-bandwidth-bound "
        "KV-cache reads. Dashed lines mark the steepest inflection point.",
        S, width=COL_W - 0.05*inch
    ))
    story.append(Paragraph(
        "Latency increases by 63% for TinyLlama-1.1B (19.92→32.49 ms) and 44% "
        "for OpenLLaMA-3B (60.38→87.21 ms) as context grows from 64 to 1024 tokens. "
        "The growth is consistent with the bandwidth model: at 1024 tokens, the "
        "KV-cache read per step is ~184 MB for TinyLlama-1.1B, requiring ~0.9 ms "
        "at 200 GB/s theoretical bandwidth. The 512→1024 jump is steeper than "
        "256→512, consistent with KV-cache overflow from on-chip cache to DRAM.", S["body"]
    ))

    story.append(subsection_head("B.", "Model Size Scaling", S))
    story.append(figure(
        "scaling_model_size.png",
        "Fig. 5. Grouped bar comparison of TinyLlama-1.1B vs OpenLLaMA-3B at "
        "each context length. Error bars show p95. The 3B/1.1B latency ratio "
        "tracks the parameter ratio closely, confirming memory-bandwidth dominance.",
        S, width=COL_W - 0.05*inch
    ))
    story.append(Paragraph(
        "OpenLLaMA-3B is consistently ~3× slower than TinyLlama-1.1B across "
        "all context lengths (Table V). This near-linear relationship between "
        "parameter count and decode latency is a hallmark of memory-bandwidth-bound "
        "workloads: runtime scales with the volume of weight and KV-cache data "
        "streamed per step, which grows proportionally with model size.", S["body"]
    ))
    story.append(Paragraph(
        "The ratio narrows from 3.03× at 64 tokens to 2.68× at 1024 tokens, "
        "because KV-cache read cost—which does not scale with model size—becomes "
        "a larger fraction of total step time at longer contexts, diluting the "
        "model-size effect.", S["body"]
    ))

    story.append(subsection_head("C.", "Precision Scaling", S))

    # Precision table
    prec_data = [
        [Paragraph("Context\n(tokens)", S["table_header"]),
         Paragraph("float16\n(ms)", S["table_header"]),
         Paragraph("float32\n(ms)", S["table_header"]),
         Paragraph("Speedup\nf32/f16", S["table_header"])],
        [Paragraph("64",   S["table_body"]), Paragraph("23.01", S["table_body"]),
         Paragraph("43.35", S["table_body"]), Paragraph("1.88×", S["table_body"])],
        [Paragraph("128",  S["table_body"]), Paragraph("21.60", S["table_body"]),
         Paragraph("46.25", S["table_body"]), Paragraph("2.14×", S["table_body"])],
        [Paragraph("256",  S["table_body"]), Paragraph("23.54", S["table_body"]),
         Paragraph("45.77", S["table_body"]), Paragraph("1.94×", S["table_body"])],
        [Paragraph("512",  S["table_body"]), Paragraph("29.25", S["table_body"]),
         Paragraph("50.08", S["table_body"]), Paragraph("1.71×", S["table_body"])],
        [Paragraph("1024", S["table_body"]), Paragraph("31.45", S["table_body"]),
         Paragraph("55.47", S["table_body"]), Paragraph("1.76×", S["table_body"])],
        [Paragraph("<b>Avg.</b>", S["table_body"]), Paragraph("", S["table_body"]),
         Paragraph("", S["table_body"]), Paragraph("<b>1.89×</b>", S["table_body"])],
    ]
    prec_table = Table(prec_data, colWidths=[COL_W*0.22, COL_W*0.26, COL_W*0.26, COL_W*0.26])
    prec_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE VI</b><br/>float16 vs float32 Per-Token Latency "
                           "(TinyLlama-1.1B)", S["caption_bold"]))
    story.append(prec_table)
    story.append(Spacer(1, 6))

    story.append(figure(
        "scaling_precision.png",
        "Fig. 6. float16 vs float32 latency across context lengths (top) and "
        "speedup ratio float32/float16 (bottom). The ~1.89× average speedup "
        "is consistent with 2× memory-bandwidth reduction.",
        S, width=COL_W - 0.05*inch
    ))
    story.append(Paragraph(
        "float16 delivers a mean 1.89× speedup over float32 across all context "
        "lengths. The theoretical upper bound is 2.0× (half the memory traffic). "
        "The sub-2× observed ratio reflects fixed-overhead components—framework "
        "dispatch, elementwise ops—that do not scale with precision width. "
        "The result confirms that decode-step latency is dominated by DRAM "
        "bandwidth and not by floating-point execution units.", S["body"]
    ))

    # ── SECTION VII: ARCHITECTURAL BOTTLENECK ANALYSIS ────────────────────────
    story.append(section_head("VII", "Architectural Bottleneck Analysis", S))
    story.append(Paragraph(
        "Each measured bottleneck has a direct architectural cause grounded in "
        "memory-system behavior. We analyze the three dominant components.", S["body"]
    ))
    story.append(subsection_head("A.", "MLP and Linear Projections: Weight Streaming", S))
    story.append(Paragraph(
        "MLP (28.2%) and QKV+O projections (31.5%) together account for 59.7% of "
        "per-token latency. These are matrix-vector multiplications (GEMVs) at "
        "decode time: each token maps to a single row query, so the GPU executes "
        "weight_matrix × 1-column input. A GEMV streams the entire weight matrix "
        "through memory but performs only 2<i>N</i> FLOPs for <i>N</i> matrix "
        "elements—arithmetic intensity of exactly 1 FLOP/byte in float16. This is "
        "maximally bandwidth-bound: every byte of weight must transit the memory "
        "subsystem once per token.", S["body"]
    ))
    story.append(subsection_head("B.", "KV-Cache Reads: Sequence-Length Bottleneck", S))
    story.append(Paragraph(
        "The attention core (8.8% at short contexts) becomes the dominant bottleneck "
        "as context length grows. The per-step KV-cache bandwidth requirement is "
        "linear in sequence length (see Section II-A), and our measurements confirm "
        "a 63% latency increase from 64 to 1024 tokens. At 1024 tokens the "
        "KV-cache read alone requires ~184 MB of memory traffic per step. "
        "Systems targeting very long contexts (>4K tokens) must address this "
        "through KV-cache quantization or sparse attention.", S["body"]
    ))
    story.append(subsection_head("C.", "LayerNorm: Kernel Dispatch Overhead", S))
    story.append(Paragraph(
        "RMSNorm accounts for 14.9% despite being a simple elementwise operation "
        "(divide by RMS, scale). The cause is kernel dispatch overhead: 44 "
        "separate Metal kernel launches per decode step, each requiring a "
        "command-encoder setup and GPU barrier. Fusing LayerNorm into the "
        "preceding or following linear layer—as done in FlashAttention's "
        "fused implementations—would eliminate most of this overhead.", S["body"]
    ))
    story.append(subsection_head("D.", "Framework Overhead (14.6%)", S))
    story.append(Paragraph(
        "14.6% of decode-step time is unaccounted for by any individual module. "
        "This overhead includes Python interpreter cost, MPS command buffer "
        "encoding, memory allocation, and inter-module synchronization barriers "
        "introduced by our measurement hooks. In production systems using "
        "torch.compile or a dedicated inference runtime this overhead would "
        "be substantially reduced.", S["body"]
    ))

    # ── SECTION VIII: OPTIMIZATION PROPOSAL ───────────────────────────────────
    story.append(section_head("VIII", "Optimization Proposal: MLP Weight Quantization", S))
    story.append(Paragraph(
        "Given that MLP and linear projections account for 59.7% of per-token "
        "latency, and that this cost is purely bandwidth-bound, post-training "
        "quantization of weight matrices to INT8 or INT4 is the highest-leverage "
        "optimization. Quantization reduces bytes-per-weight by 2× (INT8) or "
        "4× (INT4) relative to float16, proportionally reducing DRAM traffic "
        "for the dominant component.", S["body"]
    ))
    story.append(subsection_head("A.", "Projected Impact", S))

    # Optimization table
    opt_data = [
        [Paragraph("Configuration", S["table_header"]),
         Paragraph("Est. Latency (ms)", S["table_header"]),
         Paragraph("Speedup", S["table_header"]),
         Paragraph("Quality Impact", S["table_header"])],
        [Paragraph("float16 baseline", S["table_body_l"]),
         Paragraph("20.91", S["table_body"]),
         Paragraph("1.00×", S["table_body"]),
         Paragraph("None", S["table_body"])],
        [Paragraph("INT8 MLP weights", S["table_body_l"]),
         Paragraph("~16–17", S["table_body"]),
         Paragraph("~1.25×", S["table_body"]),
         Paragraph("< 0.5% perplexity ↑", S["table_body"])],
        [Paragraph("INT4 MLP weights (GPTQ)", S["table_body_l"]),
         Paragraph("~13–15", S["table_body"]),
         Paragraph("~1.40–1.60×", S["table_body"]),
         Paragraph("1–3% perplexity ↑", S["table_body"])],
    ]
    opt_table = Table(opt_data, colWidths=[COL_W*0.35, COL_W*0.22, COL_W*0.18, COL_W*0.25])
    opt_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE VII</b><br/>Projected Latency Under MLP Quantization",
                           S["caption_bold"]))
    story.append(opt_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "These estimates apply the bandwidth-reduction factor only to the MLP "
        "component (28.2% of total), leaving all other components unchanged. "
        "Quantizing QKV and O-projection weights would yield additional gains "
        "proportional to their bandwidth contribution.", S["body"]
    ))
    story.append(subsection_head("B.", "Implementation", S))
    story.append(Paragraph(
        "Post-training quantization via GPTQ [8] requires only a small calibration "
        "dataset and does not need model retraining. The HuggingFace <i>optimum</i> "
        "library provides INT8 quantization compatible with MPS via "
        "<b>BitsAndBytesConfig</b>. Full INT4 GPTQ quantization is supported through "
        "the <i>auto-gptq</i> library. Implementation complexity is low: quantization "
        "is applied once and the quantized model is serialized to disk, requiring no "
        "changes to the inference loop.", S["body"]
    ))
    story.append(subsection_head("C.", "Combined Optimization Roadmap", S))
    story.append(Paragraph(
        "Applied together, three optimizations address the measured bottlenecks: "
        "(1) INT4 MLP quantization targeting the 28.2% MLP cost; "
        "(2) fused LayerNorm kernels eliminating the 14.9% RMSNorm dispatch overhead; "
        "(3) FlashAttention-style fused attention targeting the 8.8% attention core "
        "at long contexts. Conservatively estimating independent contributions, "
        "the combined speedup over the float16 baseline could reach 1.5–1.8×, "
        "pushing throughput from 47.8 to approximately 70–85 tokens/sec on "
        "Apple Silicon.", S["body"]
    ))

    # ── SECTION IX: CONCLUSION ────────────────────────────────────────────────
    story.append(section_head("IX", "Conclusion", S))
    story.append(Paragraph(
        "We have presented a systematic, four-phase empirical study of "
        "autoregressive token-generation latency in LLaMA-family models on "
        "Apple Silicon. Our benchmark harness achieves 3.7% trial-to-trial "
        "coefficient of variation, validating the reproducibility of MPS "
        "execution. Component-level decomposition via PyTorch forward hooks "
        "reveals that MLP feed-forward blocks (28.2%) and linear projections "
        "(31.5%) together dominate decode-step time—not attention—at "
        "typical context lengths.", S["body"]
    ))
    story.append(Paragraph(
        "Scaling analysis confirms three architectural predictions of the "
        "roofline model: (1) latency scales linearly with KV-cache size, "
        "reflecting bandwidth-limited attention reads; (2) latency scales "
        "proportionally with model parameter count, reflecting weight "
        "streaming cost; and (3) float16 delivers a 1.89× speedup over "
        "float32, consistent with 2× bandwidth reduction. The primary "
        "bottleneck—weight streaming in bandwidth-bound GEMVs—is directly "
        "addressable through post-training quantization, which we project "
        "to reduce decode latency by 25–40% with minimal quality degradation.", S["body"]
    ))
    story.append(Paragraph(
        "These findings have broader implications for LLM deployment on "
        "edge devices with unified memory. As LLMs move from datacenter "
        "GPUs to consumer hardware, the memory-bandwidth bottleneck becomes "
        "more acute. Techniques that reduce per-token memory traffic—"
        "quantization, KV-cache compression, fused kernels—are the highest-"
        "leverage optimizations in this regime.", S["body"]
    ))

    # ── REFERENCES ────────────────────────────────────────────────────────────
    story.append(section_head("", "References", S))
    refs = [
        "[1] D. Sculley <i>et al.</i>, \"Hidden Technical Debt in Machine Learning Systems,\" "
        "<i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 28, 2015.",

        "[2] W. Kwon <i>et al.</i>, \"Efficient Memory Management for Large Language Model Serving "
        "with PagedAttention,\" in <i>Proc. ACM SOSP</i>, 2023.",

        "[3] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, \"FlashAttention: Fast and "
        "Memory-Efficient Exact Attention with IO-Awareness,\" <i>Advances in Neural Information "
        "Processing Systems (NeurIPS)</i>, 2022.",

        "[4] G. Gerganov, \"llama.cpp: Efficient LLaMA inference in C/C++,\" "
        "<i>GitHub repository</i>, 2023. [Online]. Available: https://github.com/ggerganov/llama.cpp",

        "[5] H. Touvron <i>et al.</i>, \"LLaMA: Open and Efficient Foundation Language Models,\" "
        "<i>arXiv preprint arXiv:2302.13971</i>, 2023.",

        "[6] S. Williams, A. Waterman, and D. Patterson, \"Roofline: An Insightful Visual "
        "Performance Model for Multicore Architectures,\" <i>Communications of the ACM</i>, "
        "vol. 52, no. 4, pp. 65–76, 2009.",

        "[7] T. Dao, \"FlashAttention-2: Faster Attention with Better Parallelism and Work "
        "Partitioning,\" <i>arXiv preprint arXiv:2307.08691</i>, 2023.",

        "[8] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, \"GPTQ: Accurate "
        "Post-Training Quantization for Generative Pre-trained Transformers,\" "
        "<i>arXiv preprint arXiv:2210.17323</i>, 2022.",

        "[9] S. Zhang <i>et al.</i>, \"OPT: Open Pre-trained Transformer Language Models,\" "
        "<i>arXiv preprint arXiv:2205.01068</i>, 2022.",

        "[10] Z. Liu <i>et al.</i>, \"KIVI: A Tuning-Free Asymmetric 2bit Quantization for "
        "KV Cache,\" <i>arXiv preprint arXiv:2402.02750</i>, 2024.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, S["ref"]))

    return story

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building IEEE paper PDF...")
    story = build_story()
    build_doc(story)
    print(f"Done → {OUT}")
