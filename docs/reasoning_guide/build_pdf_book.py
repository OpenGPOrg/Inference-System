#!/usr/bin/env python3
"""
Comprehensive PDF Book Generator for Phase 2: Reasoning Engine Architecture
Uses ReportLab to compile a professional, publication-quality technical guide
with embedded vector/raster illustrations, mathematical definitions, and C++ listings.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ASSETS_DIR = "/media/vithurshan/vithu/opengp/ggml/opengp/docs/reasoning_guide/assets"
OUTPUT_PDF = "/media/vithurshan/vithu/opengp/ggml/opengp/docs/Phase_2_Reasoning_Engine_Architecture_Guide.pdf"

class NumberedCanvas(canvas.Canvas):
    """Adds page numbers and running header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Skip cover page

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header
        self.drawString(54, 11 * inch - 36, "OpenGP Engine Architecture: Phase 2 — The Reasoning Backbone")
        self.setStrokeColor(colors.HexColor("#334155"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — OPENGP CORE ARCHITECTURE")
        self.line(54, 46, 8.5 * inch - 54, 46)

        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor('#0284c7'),
        alignment=0,
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#475569'),
        alignment=0,
        spaceAfter=25
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=18,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0369a1'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )
    formula_style = ParagraphStyle(
        'Formula_Custom',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        backColor=colors.HexColor('#f1f5f9'),
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=8
    )
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0f172a'),
        backColor=colors.HexColor('#f8fafc'),
        borderColor=colors.HexColor('#e2e8f0'),
        borderWidth=0.8,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8
    )
    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=1,
        spaceAfter=12
    )

    story = []

    # =========================================================================
    # COVER / HEADER
    # =========================================================================
    story.append(Paragraph("LLM INFERENCE ENGINE ARCHITECTURE", ParagraphStyle('SuperTitle', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=8)))
    story.append(Paragraph("Phase 2: The Reasoning Engine", title_style))
    story.append(Paragraph("A Deep Mechanical & Mathematical Blueprint: From Latent Activations H<sub>0</sub> to the Deep Transformer Layer Backbone", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=18))

    # Executive Overview
    story.append(Paragraph("<b>Document Purpose:</b> This technical guide breaks down the complete internal mechanics of Phase 2 (The Reasoning Phase) inside a production LLM inference engine. It documents every architectural operation executed across each transformer layer—from Root Mean Square Normalization (RMSNorm) and Query-Key-Value projection to Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), persistent Paged KV Cache management, SwiGLU non-linear feed-forward gating, residual stream preservation, and dynamic weight streaming.", body_style))

    # Overview Table
    summary_data = [
        [Paragraph("<b>Pipeline Stage</b>", body_style), Paragraph("<b>Mathematical Operation</b>", body_style), Paragraph("<b>Key Innovation</b>", body_style)],
        [Paragraph("<b>1. Pre-RMSNorm</b>", body_style), Paragraph("x<sub>norm</sub> = x / RMS(x) ⊙ γ", body_style), Paragraph("Eliminates mean-shift; 30% faster than LayerNorm", body_style)],
        [Paragraph("<b>2. Q, K, V Projections</b>", body_style), Paragraph("Q = x W<sub>q</sub>, K = x W<sub>k</sub>, V = x W<sub>v</sub>", body_style), Paragraph("Decouples latent channels into head subspaces", body_style)],
        [Paragraph("<b>3. RoPE Rotations</b>", body_style), Paragraph("R<sub>θ,m</sub> · [q<sub>2i</sub>, q<sub>2i+1</sub>]<sup>T</sup>", body_style), Paragraph("Deferred complex 2D rotations on Q and K", body_style)],
        [Paragraph("<b>4. Attention & GQA</b>", body_style), Paragraph("Softmax(Q K<sup>T</sup> / √d<sub>k</sub>) · V", body_style), Paragraph("Shares KV heads; slashes VRAM bandwidth by 4x-8x", body_style)],
        [Paragraph("<b>5. KV Cache Engine</b>", body_style), Paragraph("Append K, V per layer into ring buffer", body_style), Paragraph("Turns O(N²) decode compute into O(1) step time", body_style)],
        [Paragraph("<b>6. Output Projection</b>", body_style), Paragraph("H<sub>mid</sub> = H<sub>in</sub> + W<sub>o</sub> · Attn", body_style), Paragraph("Residual stream identity highway bypass", body_style)],
        [Paragraph("<b>7. Post-RMSNorm</b>", body_style), Paragraph("x<sub>ffn</sub> = H<sub>mid</sub> / RMS(H<sub>mid</sub>) ⊙ γ<sub>ffn</sub>", body_style), Paragraph("Rescales variance before non-linear expansions", body_style)],
        [Paragraph("<b>8. SwiGLU FFN</b>", body_style), Paragraph("W<sub>down</sub> · (SiLU(W<sub>gate</sub> x) ⊙ W<sub>up</sub> x)", body_style), Paragraph("Multiplicative gating outperforms classic ReLU/GeLU", body_style)],
        [Paragraph("<b>9. Final Residual</b>", body_style), Paragraph("H<sub>l+1</sub> = H<sub>mid</sub> + FFN(x<sub>ffn</sub>)", body_style), Paragraph("Feeds layer l+1 or final de-quantization head", body_style)],
    ]
    t = Table(summary_data, colWidths=[1.8*inch, 2.5*inch, 2.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # =========================================================================
    # CHAPTER 1: COMPLETE LAYER ARCHITECTURE FLOW
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 1: The Transformer Layer Lifecycle", h1_style))
    story.append(Paragraph("Every transformer reasoning block adheres to a strictly sequenced execution pipeline. Activation vectors arrive from the Perception phase (for Layer 0) or from the preceding Layer (for Layer <i>l</i>) as a continuous matrix of shape <code>[n_tokens × d_model]</code>.", body_style))

    img_path1 = f"{ASSETS_DIR}/fig1_layer_flow.png"
    if os.path.exists(img_path1):
        story.append(Image(img_path1, width=4.5*inch, height=6.75*inch))
        story.append(Paragraph("Figure 1.1: Complete structural flow of an OpenGP Transformer Layer Block showing pre-normalization, dual residual streams, RoPE rotation, and SwiGLU gating.", caption_style))

    # =========================================================================
    # CHAPTER 2: PRE-NORMALIZATION (RMSNorm vs LayerNorm)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 2: Pre-Normalization Mechanics (RMSNorm)", h1_style))
    story.append(Paragraph("In deep networks (e.g. 32 layers in LLaMA-3, 65 layers in Qwen-35B), repeated unconstrained matrix multiplications cause token vectors to experience exponential variance drift. Without normalization, numerical underflow or overflow occurs rapidly.", body_style))

    story.append(Paragraph("<b>2.1 The Mathematical Shift: LayerNorm vs RMSNorm</b>", h2_style))
    story.append(Paragraph("In classical architectures (GPT-2, BERT), LayerNorm required computing the mean (μ), subtracting it to center activations around zero, computing standard deviation (σ), and learning two parameter vectors (gain γ and bias β):", body_style))
    story.append(Paragraph("μ = (1/d) Σ x<sub>i</sub>,   σ² = (1/d) Σ (x<sub>i</sub> - μ)²<br/>LayerNorm(x) = [(x - μ) / √(σ² + ε)] ⊙ γ + β", formula_style))

    story.append(Paragraph("In 2019, Zhang & Sennrich demonstrated that the mean-centering step (x - μ) is completely dispensable. Normalizing strictly by the <b>Root Mean Square (RMS)</b> delivers equivalent or superior regularization while slashing compute overhead by ~30%:", body_style))
    story.append(Paragraph("RMS(x) = √[ (1/d<sub>model</sub>) Σ<sub>i=1..d</sub> (x<sub>i</sub>²) + ε ]<br/><b>x<sub>norm</sub> = [ x / RMS(x) ] ⊙ γ</b>", formula_style))

    story.append(Paragraph("Where:<br/>• <b>x</b>: Input token vector ∈ ℝ<sup>d_model</sup>.<br/>• <b>ε</b>: Stability constant (typically 1e-5 or 1e-6) loaded from GGUF metadata.<br/>• <b>γ</b>: Learned gain parameter loaded from <code>blk.l.attn_norm.weight</code> (dimension d_model).<br/>• <b>Zero Bias</b>: Modern LLMs omit β entirely, reducing parameter transfers over memory buses.", body_style))

    story.append(Paragraph("<b>2.2 OpenGP High-Performance AVX2 / CUDA Kernel Pattern</b>", h2_style))
    story.append(Paragraph("In our engine, RMSNorm executes as a fused single-pass kernel. In CPU mode, AVX2 FMA instructions compute the sum of squares and element-wise scaling without auxiliary allocations:", body_style))
    story.append(Paragraph("""static void rms_norm(const float * x, const float * gamma, float * out, int d, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < d; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt((sum_sq / d) + eps);
    for (int i = 0; i < d; ++i) out[i] = (x[i] * inv_rms) * gamma[i];
}""", code_style))

    # =========================================================================
    # CHAPTER 3: PROJECTIONS & ROTARY POSITION EMBEDDING (RoPE)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 3: Attention Projections & Rotary Embeddings (RoPE)", h1_style))
    story.append(Paragraph("After normalization, the token activation vector is projected from the shared latent space ℝ<sup>d_model</sup> into specialized subspaces: Query (Q), Key (K), and Value (V).", body_style))

    story.append(Paragraph("<b>3.1 Projection Slicing</b>", h2_style))
    story.append(Paragraph("Linear projections decompose semantic features into specialized attention roles:<br/>• <b>Q (Query)</b> = x<sub>norm</sub> · W<sub>q</sub> (What this token is searching for)<br/>• <b>K (Key)</b> = x<sub>norm</sub> · W<sub>k</sub> (What information this token holds)<br/>• <b>V (Value)</b> = x<sub>norm</sub> · W<sub>v</sub> (The actual payload emitted if attended to)", body_style))

    story.append(Paragraph("<b>3.2 The RoPE Revolution: Deferred Geometric Rotation</b>", h2_style))
    story.append(Paragraph("As established in Phase 1 (Perception), modern LLMs do not add static positional vectors to the input. Instead, positional coordinates are embedded dynamically by rotating pairs of dimensions in Q and K through 2D complex planes.", body_style))

    img_path2 = f"{ASSETS_DIR}/fig2_rope_rotation.png"
    if os.path.exists(img_path2):
        story.append(Image(img_path2, width=5.5*inch, height=3.09*inch))
        story.append(Paragraph("Figure 3.1: 2D complex plane rotation of feature pair (q<sub>2i</sub>, q<sub>2i+1</sub>) by angle m · θ<sub>i</sub>. Preserves vector norm while encoding relative distances.", caption_style))

    story.append(Paragraph("For a token at sequence position <i>m</i>, each adjacent pair of coordinates [q<sub>2i</sub>, q<sub>2i+1</sub>] is multiplied by a 2D rotation matrix:", body_style))
    story.append(Paragraph("""[ q'_2i   ] = [ cos(m·θ_i)  -sin(m·θ_i) ] [ q_2i   ]
[ q'_2i+1 ]   [ sin(m·θ_i)   cos(m·θ_i) ] [ q_2i+1 ]
Where base frequency: θ_i = 10000^(-2i / d_head) (or 500000 in LLaMA-3)""", formula_style))

    story.append(Paragraph("<b>Why RoPE is Superior:</b> The dot product between rotated Query and Key vectors naturally contains relative distance (m - n):", body_style))
    story.append(Paragraph("<b>⟨ R<sub>θ,m</sub> · q,  R<sub>θ,n</sub> · k ⟩ = g(q, k, m - n)</b><br/>Attention weights depend strictly on how far apart two tokens are, allowing context window extrapolation (e.g. RoPE frequency scaling up to 128k tokens).", body_style))

    # =========================================================================
    # CHAPTER 4: ATTENTION ARCHITECTURES (MHA vs GQA vs MQA)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 4: Attention Topologies (MHA vs GQA vs MQA)", h1_style))
    story.append(Paragraph("In autoregressive inference, generating tokens one-by-one is severely memory-bandwidth bound. Multi-Head Attention (MHA) creates an unsustainable memory bottleneck because every single Query head requires its own dedicated Key and Value heads in the KV Cache.", body_style))

    img_path3 = f"{ASSETS_DIR}/fig3_attention_variants.png"
    if os.path.exists(img_path3):
        story.append(Image(img_path3, width=6.0*inch, height=2.52*inch))
        story.append(Paragraph("Figure 4.1: Comparison between Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).", caption_style))

    story.append(Paragraph("<b>4.1 Head Slicing & Memory Calculations</b>", h2_style))
    story.append(Paragraph("Let:<br/>• <b>n_heads</b>: Number of Query attention heads (e.g. 32 in LLaMA-3 8B)<br/>• <b>n_kv_heads</b>: Number of Key/Value heads (e.g. 8 in LLaMA-3 8B, 32 in classical MHA)<br/>• <b>d_head</b>: Head dimension = d_model / n_heads = 4096 / 32 = 128", body_style))

    attn_table_data = [
        [Paragraph("<b>Topology</b>", body_style), Paragraph("<b>Q:KV Ratio</b>", body_style), Paragraph("<b>KV Cache Size (ctx=8192, 32 layers)</b>", body_style), Paragraph("<b>Bandwidth Demand</b>", body_style)],
        [Paragraph("<b>MHA (Classical)</b>", body_style), Paragraph("32 : 32 (1:1)", body_style), Paragraph("8.00 Gigabytes (VRAM Overflow!)", body_style), Paragraph("Extreme (Read 8 GB per generated token)", body_style)],
        [Paragraph("<b>GQA (Modern Standard)</b>", body_style), Paragraph("32 : 8 (4:1)", body_style), Paragraph("<b>2.00 Gigabytes (Fits RTX 3060)</b>", body_style), Paragraph("<b>4x Reduction; Zero degradation in quality</b>", body_style)],
        [Paragraph("<b>MQA (Extreme Fast)</b>", body_style), Paragraph("32 : 1 (32:1)", body_style), Paragraph("0.25 Gigabytes", body_style), Paragraph("8x-32x Reduction; slight capacity drop", body_style)],
    ]
    t2 = Table(attn_table_data, colWidths=[1.8*inch, 1.3*inch, 2.3*inch, 1.8*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t2)

    # =========================================================================
    # CHAPTER 5: THE KV CACHE ENGINE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 5: The KV Cache Engine", h1_style))
    story.append(Paragraph("Autoregressive text generation predicts one token at a time. Without caching, generating token 1000 would require re-computing Attention across tokens 1..999 from scratch, resulting in quadratic O(N²) execution latency.", body_style))

    story.append(Paragraph("<b>5.1 Prefill Phase vs Decode Phase</b>", h2_style))
    story.append(Paragraph("• <b>Prompt Prefill (Compute-Bound):</b> All prompt tokens (e.g. n=14) enter the layers simultaneously. The engine calculates Q, K, V for all 14 tokens in parallel matrix multiplications, stores the 14 Keys and Values in the persistent cache, and emits the first generated token.<br/>• <b>Token Decode (Bandwidth-Bound):</b> For every subsequent step, only <b>1 single token</b> enters the layer (n=1). The engine computes Q, K, V for only this single token, appends its new K and V to the persistent cache, and performs scaled dot-product attention against all previously stored historical tokens.", body_style))

    story.append(Paragraph("<b>5.2 Paged / Ring Buffer Cache Architecture in OpenGP</b>", h2_style))
    story.append(Paragraph("In OpenGP, the KV Cache is pre-allocated as a persistent contiguous slab in backend memory (Host RAM or VRAM):", body_style))
    story.append(Paragraph("""// OpenGP KV Cache Offset Calculation for Layer l, Token Slot t:
size_t kv_token_stride = n_kv_heads * d_head * sizeof(float); // e.g. 8 * 128 * 4 = 4096 bytes
size_t layer_stride    = max_context * kv_token_stride;
float * k_dest = k_cache_base + (l * layer_stride) + (t * kv_token_stride);
float * v_dest = v_cache_base + (l * layer_stride) + (t * kv_token_stride);""", code_style))

    # =========================================================================
    # CHAPTER 6: SCALED DOT-PRODUCT ATTENTION & CAUSAL MASKING
    # =========================================================================
    story.append(Paragraph("Chapter 6: Scaled Dot-Product Attention & Causal Masking", h1_style))
    story.append(Paragraph("Once Q, K, and V are prepared and cached, the core attention equation calculates mutual relevance between tokens:", body_style))
    story.append(Paragraph("<b>Attention(Q, K, V) = Softmax[ (Q · K<sup>T</sup>) / √d<sub>k</sub> + M ] · V</b>", formula_style))

    story.append(Paragraph("<b>6.1 The Scaling Factor (1 / √d<sub>k</sub>)</b>", h2_style))
    story.append(Paragraph("For head dimension d<sub>k</sub> = 128, the dot product of two random vectors has variance 128 and standard deviation √128 ≈ 11.31. Without dividing by √d<sub>k</sub>, dot products become large numbers (e.g. +40), pushing the Softmax function into saturation regions where gradients vanish and probabilities collapse into one-hot spikes.", body_style))

    story.append(Paragraph("<b>6.2 Causal Masking (Preventing Looking Ahead)</b>", h2_style))
    story.append(Paragraph("During prompt prefill, token <i>i</i> must only attend to tokens <i>j ≤ i</i>. A triangular causal mask <i>M</i> sets attention scores for future tokens to negative infinity (-∞), guaranteeing that Softmax assigns them zero probability:", body_style))
    story.append(Paragraph("""M[i, j] = 0        if j <= i   (Past & Present allowed)
M[i, j] = -1e9     if j > i    (Future strictly blocked)""", code_style))

    # =========================================================================
    # CHAPTER 7: OUTPUT PROJECTION & RESIDUAL STREAM
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 7: Attention Output Projection & Residual Bypass", h1_style))
    story.append(Paragraph("The output of multi-head attention is a concatenated matrix of all individual head outputs: <code>[n_tokens × (n_heads · d_head)] = [n_tokens × d_model]</code>.", body_style))

    story.append(Paragraph("<b>7.1 The W<sub>o</sub> Linear Projection</b>", h2_style))
    story.append(Paragraph("The attention outputs are multiplied by the output weight matrix W<sub>o</sub> ∈ ℝ<sup>d_model × d_model</sup> (loaded from <code>blk.l.attn_output.weight</code>). This projection synthesizes the independent attention insights gathered by the 32 heads back into a unified semantic representation.", body_style))

    story.append(Paragraph("<b>7.2 The Residual Highway (x = x + Attn(x))</b>", h2_style))
    story.append(Paragraph("Directly after W<sub>o</sub>, the engine applies the first <b>Residual Connection</b> (skip connection):", body_style))
    story.append(Paragraph("<b>H<sub>mid</sub> = H<sub>in</sub> + (Attn_Output · W<sub>o</sub>)</b>", formula_style))
    story.append(Paragraph("The residual stream acts as an uninterrupted information highway flowing down the entire depth of the transformer. Instead of completely rewriting the token's identity, the Attention layer only computes a small additive delta (ΔH). This guarantees that foundational semantic meaning is preserved and gradients propagate unimpeded across 60+ layers.", body_style))

    # =========================================================================
    # CHAPTER 8: POST-ATTENTION RMSNorm & SwiGLU FFN
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 8: Post-Attention Normalization & SwiGLU FFN", h1_style))
    story.append(Paragraph("While the Attention block allows tokens to communicate with each other, the <b>Feed-Forward Network (FFN)</b> is where the model stores factual knowledge, associations, and reasoning patterns. The FFN processes each token completely independently.", body_style))

    img_path4 = f"{ASSETS_DIR}/fig4_swiglu_architecture.png"
    if os.path.exists(img_path4):
        story.append(Image(img_path4, width=5.5*inch, height=3.09*inch))
        story.append(Paragraph("Figure 8.1: Structural topology of SwiGLU Gated Feed-Forward Network. The Gate branch acts as an adaptive filter controlling information passing through the Up branch.", caption_style))

    story.append(Paragraph("<b>8.1 Post-Attention RMSNorm (blk.l.ffn_norm)</b>", h2_style))
    story.append(Paragraph("Before passing into the non-linear FFN, the intermediate activation H<sub>mid</sub> is normalized via a second dedicated RMSNorm layer:", body_style))
    story.append(Paragraph("x<sub>ffn</sub> = [ H<sub>mid</sub> / RMS(H<sub>mid</sub>) ] ⊙ γ<sub>ffn</sub>", formula_style))

    story.append(Paragraph("<b>8.2 Evolution: ReLU vs GeLU vs SwiGLU</b>", h2_style))
    story.append(Paragraph("• <b>Classical Transformers (ReLU / GeLU):</b> Used a simple two-layer expansion: W<sub>2</sub> · GeLU(W<sub>1</sub> · x).<br/>• <b>Modern LLMs (SwiGLU):</b> Introduced by Noam Shazeer (2020), Gated Linear Units split the forward pass into two parallel branches that multiply element-wise:", body_style))
    story.append(Paragraph("""Gate Branch : G = SiLU(x_ffn · W_gate)
Up Branch   : U = (x_ffn · W_up)
Activated   : A = G ⊙ U  (Element-wise multiplication)
Down Branch : FFN_Out = A · W_down""", formula_style))
    story.append(Paragraph("Where the SiLU activation is defined as: <b>SiLU(z) = z · σ(z) = z / (1 + e<sup>-z</sup>)</b>.<br/>• <b>Intermediate Dimension:</b> The hidden dimension is typically expanded by 3.5x to 4x (e.g., d_ffn = 14336 for d_model = 4096 in LLaMA-3).", body_style))

    # =========================================================================
    # CHAPTER 9: MULTI-LAYER CHAINING & RESIDUAL COMPLETION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 9: Residual Completion & Multi-Layer Chaining", h1_style))
    story.append(Paragraph("After the Down projection emits the FFN output, the second residual connection completes the layer block:", body_style))
    story.append(Paragraph("<b>H<sub>l+1</sub> = H<sub>mid</sub> + FFN_Out = H<sub>in</sub> + Attn(H<sub>in</sub>) + FFN(H<sub>mid</sub>)</b>", formula_style))

    story.append(Paragraph("<b>9.1 The Chaining Handover</b>", h2_style))
    story.append(Paragraph("The tensor H<sub>l+1</sub> is identical in shape to the original input: <code>[n_tokens × d_model]</code>.<br/>• For layers <i>l < L - 1</i>: H<sub>l+1</sub> immediately becomes H<sub>in</sub> for Layer <i>l+1</i>.<br/>• For the final layer <i>l = L - 1</i>: H<sub>L</sub> is handed over to <b>Phase 3 (Expression)</b> for the final RMSNorm (<code>output_norm</code>) and LM Head projection.", body_style))

    # =========================================================================
    # CHAPTER 10: MEMORY SYSTEMS (STREAMING, DOUBLE BUFFERING & VRAM)
    # =========================================================================
    story.append(Paragraph("Chapter 10: Engine Memory Systems & Layer Streaming", h1_style))
    story.append(Paragraph("How does an inference engine run an 8B, 14B, or 35B parameter model on a consumer GPU with only 6 GB of VRAM (such as an NVIDIA RTX 3060 Laptop GPU)?", body_style))

    story.append(Paragraph("<b>10.1 The Naive Full-Model VRAM Fallacy</b>", h2_style))
    story.append(Paragraph("An 8B model in Q8_0 quantization requires ~8.5 GB of storage. Attempting to upload the entire model to a 6 GB GPU causes an immediate Out-Of-Memory (OOM) crash.", body_style))

    story.append(Paragraph("<b>10.2 The OpenGP Layer-Streaming Solution</b>", h2_style))
    story.append(Paragraph("In OpenGP, layers are treated as <b>stateless mathematical operators</b>:<br/>1. <b>Double-Buffered Slabs:</b> The engine allocates only two alternating weight slabs in VRAM (Bank A and Bank B), each sized to hold a chunk of <i>K</i> layers (e.g. 6 layers = ~1.2 GB).<br/>2. <b>Asynchronous DMA Overlap:</b> While the GPU executes calculations on Bank A (Layers 0..5), the CPU asynchronously streams the weights for Bank B (Layers 6..11) over PCIe in the background.<br/>3. <b>Ping-Pong Execution:</b> When Bank A finishes, execution swaps instantaneously to Bank B, while Bank A is refilled with Layers 12..17.<br/>4. <b>Result:</b> Models of arbitrary size (8B, 27B, 70B) run within a fixed 3 GB - 5 GB VRAM budget at full compute saturation!", body_style))

    story.append(Spacer(1, 25))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=15))
    story.append(Paragraph("<b>Next Phase:</b> With the Reasoning pipeline fully mastered, activations are handed over to <b>Phase 3: Expression</b> (Output RMSNorm, LM Head projection into Vocabulary logits, Repetition Penalties, Softmax, and Defuzzification sampling).", body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated book: {OUTPUT_PDF}")

if __name__ == '__main__':
    build_pdf()
