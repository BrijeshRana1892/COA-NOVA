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
        alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s["index_terms"] = ParagraphStyle(
        "index_terms",
        fontName=T_ROMAN, fontSize=9, leading=11,
        alignment=TA_JUSTIFY, spaceAfter=5,
    )
    s["section"] = ParagraphStyle(
        "section",
        fontName=T_BOLD, fontSize=10, leading=13,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=2,
        textColor=BLACK,
    )
    s["subsection"] = ParagraphStyle(
        "subsection",
        fontName=T_ITALIC, fontSize=10, leading=13,
        alignment=TA_LEFT, spaceBefore=3, spaceAfter=2,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontName=T_ROMAN, fontSize=10, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=4,
        firstLineIndent=18,
    )
    s["body_noindent"] = ParagraphStyle(
        "body_noindent",
        fontName=T_ROMAN, fontSize=10, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        fontName=T_ROMAN, fontSize=8.5, leading=10,
        alignment=TA_CENTER, spaceAfter=4, spaceBefore=2,
    )
    s["caption_bold"] = ParagraphStyle(
        "caption_bold",
        fontName=T_BOLD, fontSize=8.5, leading=10,
        alignment=TA_CENTER, spaceAfter=4, spaceBefore=2,
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
    if num:
        text = f"{num}.&nbsp;&nbsp;<font face='{T_BOLD}'>{title.upper()}</font>"
    else:
        text = f"<font face='{T_BOLD}'>{title.upper()}</font>"
    return Paragraph(text, S["section"])

def subsection_head(label, title, S):
    return Paragraph(f"<i>{label} {title}</i>", S["subsection"])

# ── Horizontal rule ────────────────────────────────────────────────────────────
def hrule(width=None):
    return HRFlowable(width=width or "100%", thickness=0.5,
                      color=BLACK, spaceAfter=4, spaceBefore=4)

# ── Figure helper ─────────────────────────────────────────────────────────────
def figure(filename, caption_text, S, width=None, height_ratio=0.48):
    """Single-column figure. Does NOT use KeepTogether so it flows freely."""
    path = os.path.join(FIGURES, filename)
    w = width or (COL_W - 0.05*inch)
    img = Image(path, width=w, height=w * height_ratio, kind="proportional")
    cap = Paragraph(caption_text, S["caption"])
    # Use KeepTogether only for caption+image to avoid orphan captions,
    # but keep the block small enough to always fit.
    return KeepTogether([img, cap])

def figure_wide(filename, caption_text, S, width=None):
    """Figure spanning full text width."""
    path = os.path.join(FIGURES, filename)
    w = width or (2 * COL_W + COL_GAP - 0.1*inch)
    img = Image(path, width=w, height=w * 0.45, kind="proportional")
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
    story.append(Spacer(1, 0.06*inch))
    story.append(Paragraph(
        "Token-Generation Latency Benchmarking in LLaMA:<br/>"
        "Measurement, Bottleneck Attribution, and Architectural Implications",
        S["title"]
    ))
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph("Brijesh Rana, Divy Gajera, Rashmin Gajera", S["authors"]))
    story.append(Paragraph("California State University Long Beach", S["affil"]))
    story.append(Paragraph("Long Beach, CA, USA", S["affil"]))
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
        "MLP quantization, which we estimate can reduce decode latency by 25–40%.</i>"
    )
    story.append(Paragraph(abstract_text, S["abstract"]))
    story.append(Paragraph(
        "<b><i>Index Terms</i></b><i>—large language models, token generation, latency "
        "benchmarking, KV-cache, memory bandwidth, Apple Silicon, MPS, autoregressive "
        "decoding, LLaMA, inference optimization</i>",
        S["index_terms"]
    ))
    story.append(hrule())
    story.append(Spacer(1, 0.02*inch))

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
        "quality on CPU hardware with 4× bandwidth reduction. These systems target "
        "datacenter or CPU deployments; our work characterizes the Apple Silicon "
        "unified-memory regime which shares bandwidth between the inference workload "
        "and the OS, introducing measurement challenges not addressed in prior work.", S["body"]
    ))

    story.append(subsection_head("D.", "Quantization Methodologies", S))
    story.append(Paragraph(
        "Quantization reduces the number of bits used to represent weights or "
        "activations, proportionally cutting the DRAM traffic that dominates "
        "decode-step latency. Several representative approaches frame our work.", S["body"]
    ))
    story.append(Paragraph(
        "<i>LLM.int8()</i> [10] introduces a mixed-precision INT8 matrix-multiplication "
        "scheme for transformer inference. Outlier activation channels (whose magnitudes "
        "are several standard deviations above the rest) are kept in fp16; the bulk of "
        "channels are quantized to INT8 with vector-wise scaling. The resulting quantized "
        "model preserves zero-shot accuracy on benchmarks up to OPT-175B while halving "
        "weight memory traffic. The HuggingFace integration via the "
        "<b>BitsAndBytesConfig(load_in_8bit=True)</b> API exposes this algorithm to "
        "Transformers users.", S["body"]
    ))
    story.append(Paragraph(
        "<i>GPTQ</i> [8] performs post-training weight-only quantization to 4 or 3 bits "
        "using a calibration dataset of a few hundred examples. It applies an approximate "
        "second-order solver to the column-by-column reconstruction error, achieving "
        "4-bit weights with sub-1% perplexity loss on LLaMA-class models. Unlike "
        "LLM.int8(), GPTQ does not separate outliers and operates entirely offline; the "
        "resulting checkpoint is loaded with no runtime overhead.", S["body"]
    ))
    story.append(Paragraph(
        "<i>SmoothQuant</i> [11] addresses the activation-outlier problem differently: "
        "it migrates the difficulty of activation quantization into the weights via a "
        "per-channel scaling transform that preserves mathematical equivalence. This "
        "permits W8A8 (weight <i>and</i> activation INT8) inference at near-fp16 "
        "quality, doubling throughput on hardware that supports INT8 tensor cores.", S["body"]
    ))
    story.append(Paragraph(
        "<i>Quantization-aware training</i> [12] integrates the quantize–dequantize "
        "operations into the training graph so the network learns to be robust to "
        "quantization noise. While most LLM deployments favor post-training methods to "
        "avoid retraining cost, the framework established by Jacob et al. underpins "
        "all subsequent integer-inference literature.", S["body"]
    ))

    story.append(subsection_head("E.", "Apple Silicon and the MPS Backend", S))
    story.append(Paragraph(
        "PyTorch's MPS backend [15] dispatches tensor operations to Apple's Metal "
        "Performance Shaders framework, providing GPU acceleration on M-series silicon. "
        "Compared with the mature CUDA stack, MPS exposes several limitations relevant "
        "to LLM inference. INT8 GEMM kernels analogous to CUDA's <b>cublasLtMatmul</b> "
        "are absent; <b>torch.int8</b> matmul currently falls back to fp16 dequantization "
        "on MPS, which is why <i>bitsandbytes</i> integrates only the storage side of "
        "LLM.int8() on this backend. FlashAttention's CUDA-specific intrinsics "
        "(warp-level shuffles, tensor-core MMA) are not directly portable to Metal; "
        "native fused-attention kernels for MPS are an open engineering problem. Finally, "
        "MPS shares its DRAM with the rest of the system, so background processes can "
        "perturb measured bandwidth—an effect we mitigate via our outlier filter.", S["body"]
    ))
    story.append(Paragraph(
        "System-level inference engines such as FlexGen [13] and the analytic framework "
        "of Pope et al. [14] target multi-GPU datacenter deployments. Pope et al. in "
        "particular derive scaling laws that match closely with our single-device "
        "measurements: at the operating point we measure, decode time is bounded by "
        "parameter-count × bytes-per-parameter divided by bandwidth, which is precisely "
        "what our 1.89× float16/float32 ratio confirms.", S["body"]
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
    story.append(Spacer(1, 2))

    story.append(subsection_head("B.", "Models", S))
    story.append(Paragraph(
        "Two LLaMA-family models were evaluated. Both share the identical "
        "decoder-only transformer architecture (RMSNorm, Rotary Positional Embedding, "
        "Grouped-Query Attention, SwiGLU MLP), isolating parameter scale as the "
        "independent variable.", S["body"]
    ))

    # Model table
    model_data = [
        [Paragraph("Model", S["table_header"]), Paragraph("Parameters", S["table_header"]),
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
    story.append(Spacer(1, 2))

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

    # ── SECTION IV: IMPLEMENTATION DETAILS ───────────────────────────────────
    story.append(section_head("IV", "Implementation Details", S))
    story.append(Paragraph(
        "This section documents the engineering choices, software stack, and "
        "platform-specific challenges encountered while building the measurement "
        "infrastructure. The artifacts described here are reproducible from the "
        "public repository.", S["body"]
    ))
    story.append(subsection_head("A.", "Software Stack", S))
    story.append(Paragraph(
        "The full pipeline is implemented in Python 3.11. Model loading and inference "
        "use HuggingFace Transformers; tensor execution is dispatched through "
        "PyTorch 2.x onto the MPS backend. INT8 quantization uses the bitsandbytes "
        "library (LLM.int8() kernel) wrapped via the BitsAndBytesConfig integration "
        "in Transformers; the CPU INT8 fallback uses "
        "<b>torch.quantization.quantize_dynamic</b> with per-channel symmetric weight "
        "quantization on every <b>nn.Linear</b>. Statistical aggregation is performed "
        "in pure Python (<b>statistics</b>) and NumPy; tabular outputs are written "
        "via the <b>csv</b> module. All plots are rendered with matplotlib using its "
        "non-interactive Agg backend so the harness can run on a headless machine.", S["body"]
    ))
    story.append(subsection_head("B.", "Platform Challenges", S))
    story.append(Paragraph(
        "Three properties of Apple Silicon required substantive workarounds in the "
        "harness.", S["body"]
    ))
    story.append(Paragraph(
        "<b>Asynchronous MPS dispatch.</b> Like every modern GPU runtime, MPS returns "
        "immediately from kernel-launch calls. Reading <b>time.perf_counter()</b> "
        "after a forward pass measures CPU submission latency, not GPU completion. "
        "We therefore wrap every timed region with <b>torch.mps.synchronize()</b>, a "
        "global barrier that drains the Metal command queue. This barrier is the "
        "only reliable way to obtain hardware-accurate timings on MPS today; "
        "PyTorch's <b>torch.mps.Event</b> API is incomplete relative to the CUDA "
        "event abstraction.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Metal shader JIT.</b> On the very first invocation of a given operator "
        "shape, PyTorch emits Metal shader source code, compiles it, and caches the "
        "resulting pipeline state object. The first decode step of an unwarmed model "
        "is therefore inflated by tens to hundreds of milliseconds of compilation "
        "overhead. We discovered this empirically when our first prototype reported "
        "a 700 ms \"per-token\" latency on trial 1 that dropped to 21 ms on trial 2. "
        "The current harness accordingly performs three full warm-up trials before "
        "any timed trial; this places every shader shape into the JIT cache.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Unified memory bandwidth contention.</b> Because the GPU shares DRAM "
        "with the OS, kernel paging, browser tabs, and indexing daemons can all "
        "transiently steal bandwidth from the inference workload. We do not suspend "
        "system processes (the harness is intended to reflect realistic deployment), "
        "but we apply IQR outlier filtering on a per-trial basis to discard the "
        "small number of trials in which background activity caused visible "
        "perturbation.", S["body_noindent"]
    ))
    story.append(subsection_head("C.", "Hook Instrumentation", S))
    story.append(Paragraph(
        "Per-component latencies are obtained without modifying the Transformers "
        "source by registering forward-pre and forward hooks on the target "
        "<b>nn.Module</b> instances. The hook design is summarized below.", S["body"]
    ))
    code_style = ParagraphStyle(
        "code", fontName="Courier", fontSize=7.5, leading=9.5,
        alignment=TA_LEFT, leftIndent=4, rightIndent=4,
        backColor=colors.HexColor("#f3f3f3"), borderColor=colors.HexColor("#dddddd"),
        borderWidth=0.5, borderPadding=4, spaceAfter=4, spaceBefore=2,
    )
    code_listing = (
        "class TimingHook:\n"
        "    def __init__(self, name, device, sink):\n"
        "        self.name, self.device, self.sink = name, device, sink\n"
        "        self._t0 = None\n"
        "    def pre(self, module, inputs):\n"
        "        sync(self.device)        # drain GPU queue\n"
        "        self._t0 = perf_counter()\n"
        "    def post(self, module, inputs, output):\n"
        "        sync(self.device)        # wait for module\n"
        "        self.sink.append((self.name,\n"
        "             (perf_counter() - self._t0) * 1000))\n"
        "\n"
        "for layer_idx, layer in enumerate(model.model.layers):\n"
        "    for sub in (\"self_attn\", \"mlp\",\n"
        "                \"input_layernorm\",\n"
        "                \"post_attention_layernorm\"):\n"
        "        h = TimingHook(f\"L{layer_idx}.{sub}\", device, sink)\n"
        "        m = getattr(layer, sub)\n"
        "        m.register_forward_pre_hook(h.pre)\n"
        "        m.register_forward_hook(h.post)"
    )
    # ReportLab Paragraph treats <br/> for line breaks; need to escape and join
    code_html = "<br/>".join(line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
                              for line in code_listing.split("\n"))
    story.append(Paragraph(code_html, code_style))
    story.append(Paragraph(
        "<b>Listing 1.</b> Hook-based instrumentation. Every module gains two "
        "synchronization barriers, which inflates absolute latency by 30–40% but "
        "preserves the relative breakdown.", S["caption"]
    ))
    story.append(Paragraph(
        "In total 44 hooks are registered (one pre + one post per module × 22 "
        "layers × 1 module instance per row of the breakdown table). Sub-projection "
        "times (Q, K, V, O) are computed by attaching additional hooks to the "
        "projection submodules of each LlamaAttention instance and subtracting "
        "from the parent self-attention time.", S["body"]
    ))
    story.append(subsection_head("D.", "Data Pipeline", S))
    story.append(Paragraph(
        "Each trial emits one row per token to a raw CSV "
        "(<b>data/benchmark_raw.csv</b>) and a per-trial summary row to "
        "<b>data/benchmark_summary.csv</b>; the cross-trial aggregate is appended at "
        "the bottom of the same summary file. This two-tier structure allows "
        "downstream analysis to either work with summary statistics or re-derive "
        "distributions from the raw token-level data. IQR outlier filtering with a "
        "1.5× threshold is applied to per-trial median latencies before computing "
        "aggregates; the number of removed trials is logged to standard output for "
        "transparency.", S["body"]
    ))
    story.append(subsection_head("E.", "Lessons Learned", S))
    story.append(Paragraph(
        "<i>Three warm-ups suffice.</i> We initially used a single warm-up, which "
        "left several shader shapes uncompiled. Trial 1 of decode would compile a "
        "fresh shape (e.g., a smaller matmul for the LM head) and the resulting "
        "outlier dragged the per-trial median upward. Increasing to three warm-ups "
        "stabilized the trial-to-trial coefficient of variation from 9.4% to 3.7%.", S["body"]
    ))
    story.append(Paragraph(
        "<i>Hook overhead is unavoidable but acceptable.</i> Each pair of "
        "synchronization barriers introduced by a hook adds approximately 50 µs of "
        "GPU–CPU round-trip cost. Across 44 hooks this is ~2.2 ms per decode step, "
        "or roughly 10% of the 21 ms median. We accept this overhead for the "
        "decomposition study and report the un-instrumented baseline (Section V) "
        "as the headline number.", S["body"]
    ))
    story.append(Paragraph(
        "<i>Greedy decoding eliminates a sampling tax.</i> An early version of the "
        "harness used <b>model.generate()</b> with default settings, which includes "
        "top-k filtering, top-p truncation, and temperature scaling on the CPU. "
        "Replacing this with an explicit greedy <b>argmax()</b> on the logits "
        "removed ~3 ms of per-token overhead unrelated to model execution.", S["body"]
    ))

    # ── SECTION V: BENCHMARK METHODOLOGY ────────────────────────────────────
    story.append(section_head("V", "Benchmark Methodology", S))
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
    story.append(Spacer(1, 2))

    # Timing diagram
    # Timing diagram — full column width for legibility
    _td_w = COL_W
    story.append(Image(os.path.join(FIGURES, "timing_diagram.png"),
                       width=_td_w, height=_td_w * 0.58, kind="proportional"))
    story.append(Paragraph(
        "Fig. 1. Benchmark timing methodology. Top panel: TTFT vs decode phase "
        "timeline for a representative trial. Bottom panel: per-token latency "
        "across all 10 trials with median and p95 highlighted.",
        S["caption"]
    ))

    # Benchmark results table
    story.append(Spacer(1, 2))
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
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "The coefficient of variation of 3.7% confirms that MPS execution is "
        "highly deterministic once warm-up is complete, validating the use of "
        "single-configuration measurements in the scaling study.", S["body"]
    ))

    # ── SECTION VI: LATENCY DECOMPOSITION ────────────────────────────────────
    story.append(section_head("VI", "Latency Decomposition", S))
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
    story.append(Spacer(1, 3))

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
        S, width=COL_W - 0.05*inch, height_ratio=0.55
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

    # ── SECTION VII: SCALING ANALYSIS ────────────────────────────────────────
    story.append(section_head("VII", "Scaling Analysis", S))
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
    story.append(Spacer(1, 3))

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
    story.append(Spacer(1, 3))

    story.append(figure(
        "scaling_precision.png",
        "Fig. 6. float16 vs float32 latency across context lengths (top) and "
        "speedup ratio float32/float16 (bottom). The ~1.89× average speedup "
        "is consistent with 2× memory-bandwidth reduction.",
        S, width=COL_W - 0.05*inch, height_ratio=0.60
    ))
    story.append(Paragraph(
        "float16 delivers a mean 1.89× speedup over float32 across all context "
        "lengths. The theoretical upper bound is 2.0× (half the memory traffic). "
        "The sub-2× observed ratio reflects fixed-overhead components—framework "
        "dispatch, elementwise ops—that do not scale with precision width. "
        "The result confirms that decode-step latency is dominated by DRAM "
        "bandwidth and not by floating-point execution units.", S["body"]
    ))

    # ── SECTION VIII: ARCHITECTURAL BOTTLENECK ANALYSIS ───────────────────────
    story.append(section_head("VIII", "Architectural Bottleneck Analysis", S))
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

    story.append(subsection_head("E.", "Roofline Interpretation", S))
    story.append(Paragraph(
        "The roofline model unifies the per-component results above into a single "
        "geometric picture (Fig. 7). On a representative Apple Silicon-class GPU—"
        "approximately 2.6 TFLOP/s peak fp16 and 100 GB/s peak unified-memory "
        "bandwidth—the ridge point lies near 26 FLOP/byte. Autoregressive decode of "
        "a 1.1B-parameter model in float16 operates at ~1 FLOP/byte: every weight "
        "matrix is multiplied by exactly one column-vector input, so each loaded "
        "byte performs only two FLOPs of useful work. The operating point therefore "
        "sits firmly on the bandwidth-bound slope, two orders of magnitude below the "
        "compute ceiling. This is the geometric reason that increasing FLOP capacity "
        "(more cores, higher clocks) buys little for decode latency while cutting "
        "bytes-per-weight (quantization) buys almost the full expected speedup, as "
        "our 1.89× float16/float32 result confirms.", S["body"]
    ))
    story.append(figure(
        "roofline.png",
        "Fig. 7. Roofline plot for an Apple Silicon-class GPU. The LLM decode "
        "operating point at 1 FLOP/byte (fp16) sits well below the compute "
        "ceiling; INT8 doubles arithmetic intensity but still leaves headroom on "
        "the bandwidth roof.",
        S, width=COL_W - 0.05*inch, height_ratio=0.62
    ))

    story.append(subsection_head("F.", "Memory Hierarchy Effects", S))
    story.append(Paragraph(
        "The 100 GB/s figure is the off-chip DRAM ceiling, but Apple Silicon GPUs "
        "also expose a multi-megabyte system-level cache (SLC) shared with the CPU. "
        "Tensors smaller than the SLC capacity can be re-read at several-hundred "
        "GB/s effective bandwidth, while tensors that overflow must be refetched "
        "from LPDDR5 at the full 100 GB/s ceiling. Two observations from our data "
        "align with this hierarchy. First, the 512→1024 context-length jump in "
        "Section VII-A is steeper than the 256→512 jump: at TinyLlama-1.1B's GQA "
        "configuration the KV-cache crosses ~90 MB at 1024 tokens, plausibly "
        "exceeding SLC residency and forcing DRAM traffic on every step. Second, "
        "the 3.4B/1.1B latency ratio narrows at long contexts (3.03× at 64 tokens, "
        "2.68× at 1024 tokens) because KV-cache reads—which do not scale with "
        "parameter count—become a larger share of the step.", S["body"]
    ))

    story.append(subsection_head("G.", "Generalization to Other Models", S))
    story.append(Paragraph(
        "The bottleneck profile we report for TinyLlama-1.1B is structurally "
        "representative of other modern decoder-only transformers in the same "
        "parameter range. LLaMA-2-7B, Mistral-7B, and Phi-3 all share the same "
        "SwiGLU-MLP and grouped-query attention motif; their layer counts (~32) "
        "and FFN-to-hidden ratios (~2.7×) place them in the same arithmetic-"
        "intensity regime. We expect the relative breakdown—MLP &gt; projections "
        "&gt; LayerNorm &gt; attention core at short context—to hold for these "
        "models on MPS, with absolute latencies scaling linearly in parameter "
        "count as Section VII-B demonstrates. The methodology and software "
        "artifacts presented here extend directly to those models by changing "
        "only the MODEL_ID constant in the harness.", S["body"]
    ))

    # ── SECTION IX: OPTIMIZATION PROPOSAL ─────────────────────────────────────
    story.append(section_head("IX", "Optimization Proposal: MLP Weight Quantization", S))
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
    story.append(Spacer(1, 3))

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

    story.append(subsection_head("D.", "INT8 Quantization: Empirical Evaluation", S))
    story.append(Paragraph(
        "To validate the bandwidth-reduction prediction with measured numbers rather "
        "than projections, we ran the full benchmark harness against a TinyLlama-1.1B "
        "model loaded under INT8 weight quantization. We attempted two paths.", S["body"]
    ))
    story.append(Paragraph(
        "<b>Attempted path: bitsandbytes on MPS.</b> Our first attempt loaded the "
        "model with <b>BitsAndBytesConfig(load_in_8bit=True)</b> and "
        "<b>device_map={\"\": \"mps\"}</b>. The bitsandbytes 0.49 release accepts this "
        "configuration but the underlying INT8 GEMM kernel is CUDA-only; on MPS the "
        "library falls back to a dequantize-to-fp16 matmul on each call. We measured "
        "this path running well under 0.2 tokens/sec—more than two orders of magnitude "
        "slower than the fp16 baseline—so we abandoned it.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Reported path: torch dynamic INT8 on CPU.</b> We instead use PyTorch's "
        "native <b>quantize_dynamic</b> pass with the qnnpack backend (the only torch "
        "quantized backend supported on arm64 macOS). This applies per-channel "
        "symmetric INT8 weight quantization to every <b>nn.Linear</b>; activations are "
        "dynamically quantized at runtime. The result is a real INT8 weight-only "
        "inference path on the same machine, although it executes on the CPU rather "
        "than the GPU.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "The same harness (3 warm-ups, 10 timed trials, 128 tokens, IQR filter) was "
        "run against this configuration. Table VIII reports both the on-CPU INT8 "
        "measurement and the on-CPU float16 control for an apples-to-apples ratio, "
        "alongside the MPS float16 absolute baseline.", S["body"]
    ))

    int8_data = [
        [Paragraph("Metric", S["table_header"]),
         Paragraph("fp16 (MPS)", S["table_header"]),
         Paragraph("fp16 (CPU)", S["table_header"]),
         Paragraph("INT8 (CPU)", S["table_header"])],
        [Paragraph("TTFT (median, ms)", S["table_body_l"]),
         Paragraph("27.0",   S["table_body"]),
         Paragraph("191.22", S["table_body"]),
         Paragraph("155.81", S["table_body"])],
        [Paragraph("Per-token latency (ms)", S["table_body_l"]),
         Paragraph("20.91", S["table_body"]),
         Paragraph("49.69", S["table_body"]),
         Paragraph("80.28", S["table_body"])],
        [Paragraph("Per-token p95 (ms)", S["table_body_l"]),
         Paragraph("21.98", S["table_body"]),
         Paragraph("57.11", S["table_body"]),
         Paragraph("81.95", S["table_body"])],
        [Paragraph("Throughput (tokens/sec)", S["table_body_l"]),
         Paragraph("47.8",  S["table_body"]),
         Paragraph("20.13", S["table_body"]),
         Paragraph("12.46", S["table_body"])],
    ]
    int8_table = Table(int8_data,
                       colWidths=[COL_W*0.40, COL_W*0.20, COL_W*0.20, COL_W*0.20])
    int8_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE VIII</b><br/>Measured INT8 vs float16 — "
                           "TinyLlama-1.1B (10 trials each)", S["caption_bold"]))
    story.append(int8_table)
    story.append(Spacer(1, 3))

    story.append(figure(
        "int8_vs_fp16.png",
        "Fig. 8. Measured per-token latency (left) and throughput (right) for "
        "TinyLlama-1.1B under three configurations. The two CPU bars isolate the "
        "INT8/fp16 effect on identical hardware; the MPS bar gives the absolute "
        "speed reference. Lower latency and higher throughput are better.",
        S, width=COL_W - 0.05*inch, height_ratio=0.50
    ))

    story.append(Paragraph(
        "<b>Discussion.</b> The honest finding is that CPU dynamic INT8 "
        "quantization <i>slows</i> TinyLlama-1.1B by 1.62× relative to the fp16 CPU "
        "control on this hardware (80.28 ms/tok vs 49.69 ms/tok). The result "
        "contradicts the naive \"2× less memory traffic, 2× faster\" projection "
        "from Section IX for two reasons specific to this configuration.", S["body"]
    ))
    story.append(Paragraph(
        "First, qnnpack's dynamic quantization re-computes the activation scale "
        "on every <b>nn.Linear</b> call by passing over the activation tensor. "
        "For TinyLlama-1.1B's 22 layers the per-call overhead (~1 ms each at the "
        "hidden size of 2048) dominates the &lt;1 ms saved on each weight load.", S["body"]
    ))
    story.append(Paragraph(
        "Second, Apple Silicon CPU cores execute fp16 matmul at the same effective "
        "throughput as fp32 (the NEON pipeline widens fp16 to fp32 internally). The "
        "fp16 \"baseline\" is therefore not bandwidth-bound in the same way an MPS "
        "fp16 decode step is, so the bandwidth reduction promised by INT8 "
        "quantization has nothing to compress against.", S["body"]
    ))
    story.append(Paragraph(
        "The result demonstrates that quantization speedups are "
        "<i>hardware-conditional</i>: they require a backend with an INT8 GEMM kernel "
        "that achieves higher throughput per byte than the fp16/fp32 path. This "
        "holds on NVIDIA tensor cores from Ampere onwards, on the Apple Neural "
        "Engine, and (when implemented) on a hypothetical MPS INT8 GEMM. It does "
        "not hold on the qnnpack CPU fallback used here.", S["body"]
    ))
    story.append(Paragraph(
        "<b>Quality implications.</b> The LLM.int8() algorithm preserves zero-shot "
        "accuracy on standard benchmarks within 0.5 pp for models up to 175B "
        "parameters [10], by routing the small fraction of outlier activation "
        "channels through fp16 mixed precision. The torch dynamic-quantization path "
        "used here applies symmetric per-channel quantization without the "
        "outlier-aware split; on the LLaMA family this typically incurs a 1–2% "
        "perplexity increase. We do not re-evaluate perplexity in this work—the "
        "focus is latency—but mark this as future work in Section XII-A. INT4 "
        "quantization via GPTQ [8] would push the bandwidth saving to 4× over "
        "float16 at the cost of a 1–3% perplexity increase (Table VII), provided "
        "it is paired with a hardware backend that exposes the bandwidth "
        "advantage.", S["body"]
    ))

    # ── SECTION X: CROSS-PLATFORM CPU vs MPS ──────────────────────────────────
    story.append(section_head("X", "Cross-Platform Comparison: CPU vs MPS", S))
    story.append(Paragraph(
        "To put the MPS results in context we ran the same harness on the same "
        "machine with <b>device='cpu'</b>. The model is loaded in float16 to match "
        "the baseline; on Apple Silicon the CPU matrix kernels widen float16 inputs "
        "to float32 internally, so the comparison primarily isolates GPU-vs-CPU "
        "execution rather than precision width.", S["body"]
    ))
    cpu_data = [
        [Paragraph("Metric", S["table_header"]),
         Paragraph("MPS (GPU)", S["table_header"]),
         Paragraph("CPU", S["table_header"]),
         Paragraph("MPS speedup", S["table_header"])],
        [Paragraph("TTFT (median, ms)", S["table_body_l"]),
         Paragraph("27.0",   S["table_body"]),
         Paragraph("191.22", S["table_body"]),
         Paragraph("7.08×",  S["table_body"])],
        [Paragraph("Per-token latency (ms)", S["table_body_l"]),
         Paragraph("20.91", S["table_body"]),
         Paragraph("49.69", S["table_body"]),
         Paragraph("2.38×", S["table_body"])],
        [Paragraph("Per-token p95 (ms)", S["table_body_l"]),
         Paragraph("21.98", S["table_body"]),
         Paragraph("57.11", S["table_body"]),
         Paragraph("—", S["table_body"])],
        [Paragraph("Throughput (tokens/sec)", S["table_body_l"]),
         Paragraph("47.8",  S["table_body"]),
         Paragraph("20.13", S["table_body"]),
         Paragraph("2.37×", S["table_body"])],
    ]
    cpu_table = Table(cpu_data,
                      colWidths=[COL_W*0.40, COL_W*0.20, COL_W*0.20, COL_W*0.20])
    cpu_table.setStyle(ieee_table_style())
    story.append(Paragraph("<b>TABLE IX</b><br/>Per-Token Latency: MPS vs CPU "
                           "(TinyLlama-1.1B, fp16)", S["caption_bold"]))
    story.append(cpu_table)
    story.append(Spacer(1, 3))

    story.append(figure(
        "cpu_vs_mps.png",
        "Fig. 9. MPS GPU versus CPU for the identical TinyLlama-1.1B float16 "
        "workload. The GPU achieves a 2.37× throughput advantage and a 7× lower "
        "TTFT.",
        S, width=COL_W - 0.05*inch, height_ratio=0.50
    ))

    story.append(Paragraph(
        "The CPU baseline executes at 20.13 tokens/sec compared with MPS at "
        "47.8 tokens/sec—a 2.37× throughput gap. TTFT shows a much larger 7.08× "
        "gap (191 ms CPU vs 27 ms MPS) because the prefill phase processes all 12 "
        "prompt tokens in parallel: the GPU exploits this concurrency through its "
        "many SIMD units while the CPU must execute the same matrix-multiplications "
        "using a comparatively small number of vector lanes. The decode phase "
        "narrows the gap because each step is a serial matrix-vector "
        "multiplication that exposes less parallelism for the GPU to absorb.", S["body"]
    ))
    story.append(Paragraph(
        "Two factors explain the GPU's advantage even on memory-bound work. First, "
        "the M-series GPU's wide vector pipeline and tens of execution units allow "
        "many partial dot-products to be issued per clock, amortising the loop "
        "overhead that a CPU thread must pay sequentially. Second, the unified "
        "memory architecture means there is no host-device copy: every byte of "
        "weight read by the GPU comes from the same DRAM the CPU would have used, "
        "but at the GPU's higher achievable bandwidth through the wide memory "
        "controller. The CPU result therefore is not limited by data movement to "
        "the device but by the narrower per-thread issue width and lower aggregate "
        "bandwidth realised through the CPU memory subsystem.", S["body"]
    ))
    story.append(Paragraph(
        "The CPU result is useful as a fallback ceiling: a deployment that loses "
        "access to the GPU—for example because Metal is being used by a foreground "
        "graphics application—can still produce 20 tokens/sec on TinyLlama-1.1B, "
        "which remains above the human reading rate of ~5 tokens/sec.", S["body"]
    ))

    # ── SECTION XI: LIMITATIONS ───────────────────────────────────────────────
    story.append(section_head("XI", "Limitations", S))
    story.append(Paragraph(
        "We discuss several limitations of the study to clarify the scope of the "
        "conclusions.", S["body"]
    ))
    story.append(Paragraph(
        "<b>Single hardware platform.</b> All measurements are taken on a single "
        "Apple Silicon machine. The relative breakdown—MLP and projections "
        "dominate, attention is small at short context, RMSNorm is dispatch-bound—"
        "should generalize to other architectures whose GPUs are similarly "
        "memory-bandwidth-limited at decode-time arithmetic intensities (NVIDIA "
        "consumer GPUs, AMD RDNA mobile parts, Qualcomm Adreno). Absolute numbers "
        "will differ. We have not verified the quantitative scaling on other "
        "hardware in this paper; doing so is explicit future work.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Hook-induced overhead.</b> The synchronization barriers introduced by "
        "our PyTorch forward hooks inflate absolute step latency by approximately "
        "30–40% versus an un-instrumented model. The <i>relative</i> percentages "
        "reported in Table IV remain valid because the overhead is uniform across "
        "modules, but readers should not interpret the 21 ms median as the ceiling "
        "of MPS performance for TinyLlama-1.1B; an un-hooked, "
        "<b>torch.compile</b>-d deployment would run several milliseconds faster "
        "per token.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Limited model coverage.</b> We measure two model sizes (1.1B and 3.4B). "
        "Larger models (LLaMA-2-7B, Mistral-7B, Phi-3) would not fit in the 17.2 GB "
        "unified-memory budget at float16 alongside the OS working set, and would "
        "therefore require quantization simply to load. Extending the study to such "
        "models is contingent on a larger-memory machine or a quantized-only "
        "configuration; we discuss this in the future work.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>No quantization quality evaluation.</b> The INT8 measurements "
        "(Section IX-D) report latency only. We rely on published perplexity "
        "results [10, 8] to characterize quality and have not reproduced these on "
        "TinyLlama-1.1B. A faithful evaluation would run a held-out perplexity "
        "benchmark (WikiText-2, C4) or downstream task accuracy (HellaSwag, ARC) "
        "on the quantized model and report the delta.", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>Batch size fixed at 1.</b> All experiments use batch size 1, matching "
        "the single-user interactive setting that motivated the work. At larger "
        "batch sizes the arithmetic intensity of decode steps rises linearly: "
        "per-token latency would still grow, but per-token throughput across the "
        "batch can multiply by an order of magnitude. The bandwidth-bound "
        "conclusions of this paper apply specifically to the batch-1 regime; "
        "multi-tenant serving scenarios behave differently and are well-"
        "characterized by prior work [2, 13].", S["body_noindent"]
    ))
    story.append(Paragraph(
        "<b>No measurement of quantization accuracy on MPS.</b> Our INT8 result "
        "uses the bitsandbytes mixed-precision LLM.int8() algorithm. Because the "
        "underlying INT8 GEMM kernel on MPS is not yet hardware-accelerated in the "
        "same way as on CUDA, the reported speedup reflects the bandwidth saving "
        "of INT8 weight storage but <i>not</i> any INT8-tensor-core compute "
        "speedup. On hardware with INT8 matmul acceleration (NVIDIA Ampere or "
        "newer, modern Apple Neural Engine), the end-to-end gain would likely be "
        "larger.", S["body_noindent"]
    ))

    # ── SECTION XII: CONCLUSION + FUTURE WORK ─────────────────────────────────
    story.append(section_head("XII", "Conclusion and Future Work", S))
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

    story.append(subsection_head("A.", "Future Work", S))
    story.append(Paragraph(
        "Several directions extend this study and would strengthen the optimization "
        "claims.", S["body"]
    ))
    story.append(Paragraph(
        "<i>INT4 GPTQ quantization with quality evaluation.</i> The strongest "
        "remaining latency lever predicted by our roofline analysis is INT4 "
        "weight-only quantization via GPTQ [8]. Implementing this on MPS requires "
        "either porting the auto-gptq dequantization kernel or composing fp16 "
        "dequantize with the existing matmul kernel. A complete evaluation would "
        "report end-to-end decode latency <i>and</i> perplexity on WikiText-2 or "
        "C4 to verify that the projected ~1.4–1.6× speedup is achievable without "
        "unacceptable quality loss.", S["body"]
    ))
    story.append(Paragraph(
        "<i>FlashAttention on MPS.</i> The 8.8% attention-core time grows linearly "
        "with context length and becomes the dominant component beyond ~4K tokens. "
        "A Metal-native FlashAttention port would fuse the attention QKV-matmul-"
        "softmax-output sequence into a single kernel, eliminating intermediate HBM "
        "round-trips. The PyTorch team has prototyped Metal kernels for scaled "
        "dot-product attention; once these stabilize we would re-run our scaling "
        "study with FlashAttention enabled and quantify the impact on long-context "
        "decode.", S["body"]
    ))
    story.append(Paragraph(
        "<i>Cross-platform validation on NVIDIA GPUs.</i> Reproducing the "
        "hook-based decomposition on an NVIDIA Ampere or Hopper GPU would test "
        "whether the relative breakdown is hardware-invariant or whether the "
        "specific dispatch-overhead profile of MPS exaggerates the LayerNorm and "
        "framework-overhead components. We anticipate the relative ordering will "
        "hold but the absolute LayerNorm fraction will shrink because CUDA "
        "kernel-launch overhead is lower than the Metal command-encoder cost.", S["body"]
    ))
    story.append(Paragraph(
        "<i>Larger models.</i> Extending the parameter-scaling study to "
        "LLaMA-2-7B and 13B (loaded in INT8 to fit the 17.2 GB memory budget) "
        "would test the linear-scaling prediction beyond the 3.4B endpoint "
        "measured here. We expect TTFT and per-token latency to continue scaling "
        "with parameter count, but absolute throughput on a 7B model in INT8 "
        "should remain interactive (≳10 tokens/sec).", S["body"]
    ))
    story.append(Paragraph(
        "<i>Speculative decoding.</i> Speculative decoding pairs a small draft "
        "model with the target model to amortize the per-step cost of the large "
        "model across several candidate tokens. Combining speculative decoding "
        "with our INT8 baseline could double effective throughput at no quality "
        "cost. Measuring the interaction between draft-model size, acceptance "
        "rate, and total throughput on Apple Silicon is an attractive next study "
        "because the unified-memory architecture permits both models to share "
        "weights with no host-device transfers.", S["body"]
    ))

    # ── REFERENCES ────────────────────────────────────────────────────────────
    # ColumnBreak balances last page columns before References
    story.append(ColumnBreak())
    story.append(section_head("", "References", S))
    refs = [
        "[1] J. Nielsen, <i>Usability Engineering</i>. San Francisco, CA: Morgan Kaufmann, 1993.",

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

        "[9] Z. Liu <i>et al.</i>, \"KIVI: A Tuning-Free Asymmetric 2bit Quantization for "
        "KV Cache,\" <i>arXiv preprint arXiv:2402.02750</i>, 2024.",

        "[10] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "
        "\"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale,\" "
        "<i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2022.",

        "[11] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, "
        "\"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large "
        "Language Models,\" in <i>Proc. Int. Conf. on Machine Learning (ICML)</i>, "
        "2023.",

        "[12] B. Jacob <i>et al.</i>, \"Quantization and Training of Neural Networks "
        "for Efficient Integer-Arithmetic-Only Inference,\" in <i>Proc. IEEE/CVF "
        "Conf. on Computer Vision and Pattern Recognition (CVPR)</i>, 2018.",

        "[13] Y. Sheng <i>et al.</i>, \"FlexGen: High-Throughput Generative "
        "Inference of Large Language Models with a Single GPU,\" in "
        "<i>Proc. Int. Conf. on Machine Learning (ICML)</i>, 2023.",

        "[14] R. Pope <i>et al.</i>, \"Efficiently Scaling Transformer Inference,\" "
        "in <i>Proc. Conf. on Machine Learning and Systems (MLSys)</i>, 2023.",

        "[15] PyTorch Team, \"Introducing Accelerated PyTorch Training on Mac,\" "
        "<i>PyTorch Blog</i>, 2022. [Online]. Available: "
        "https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/",
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
