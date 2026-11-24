#!/usr/bin/env python3
"""
Comprehensive PDF Book: The Deep Mathematical & Intuitive Guide to Q, K, V Projections and RoPE
Compiles an in-depth, self-contained architectural reference with diagrams, proofs, concrete examples, and C++ kernels.
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

ASSETS_DIR = "/media/vithurshan/vithu/opengp/ggml/opengp/docs/qkv_rope_guide/assets"
OUTPUT_PDF = "/media/vithurshan/vithu/opengp/ggml/opengp/docs/QKV_Projections_and_RoPE_Deep_Dive_Guide.pdf"

class NumberedCanvas(canvas.Canvas):
    """Adds running headers, footers, and dynamic page counts."""
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
            return  # Skip cover

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header
        self.drawString(54, 11 * inch - 36, "OpenGP Deep-Dive: Mathematical Intuition Behind Q, K, V Projections & RoPE")
        self.setStrokeColor(colors.HexColor("#334155"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "OPEN GENERAL PARALLEL (OPENGP) — CORE ARCHITECTURAL SERIES")
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

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=32,
        textColor=colors.HexColor('#0284c7'),
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12.5,
        leading=17,
        textColor=colors.HexColor('#475569'),
        spaceAfter=20
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0369a1'),
        spaceBefore=12,
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
        fontSize=9.5,
        leading=13.5,
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
    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0369a1'),
        backColor=colors.HexColor('#e0f2fe'),
        borderColor=colors.HexColor('#38bdf8'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # COVER / HEADER
    # =========================================================================
    story.append(Paragraph("OPENGP REASONING FOUNDATIONS: MONOGRAPH 01", ParagraphStyle('SuperTitle', fontName='Helvetica-Bold', fontSize=9.5, textColor=colors.HexColor('#64748b'), spaceAfter=6)))
    story.append(Paragraph("The Intuition & Mathematical Machinery Behind Q, K, V Projections & RoPE", title_style))
    story.append(Paragraph("Why We Project, How Dot Products Measure Compatibility, and How Rotary Complex Geometry Encodes Relative Token Distance", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=14))

    # Executive Abstract
    story.append(Paragraph("<b>Executive Summary:</b> Two foundational mechanisms power the transformer's attention backbone: linear subspace projection (Query, Key, Value) and Rotary Position Embedding (RoPE). This guide reveals the core mathematical dilemmas that forced the invention of both systems. We explain why direct vector comparison fails catastrophically due to symmetry and self-absorption, how Q, K, and V decouple roles into a content-addressable database, and how RoPE achieves pure relative distance invariance (m - n) by rotating pairs of coordinates through 2D complex planes.", body_style))

    # =========================================================================
    # PART 1: THE REASON & INTUITION BEHIND Q, K, V PROJECTIONS
    # =========================================================================
    story.append(Paragraph("Part 1: The Intuition Behind Query, Key, and Value", h1_style))
    story.append(Paragraph("When an activation vector <i>x</i> ∈ ℝ<sup>d_model</sup> arrives at a transformer layer, it represents a single dense point in semantic latent space (e.g., 4096 dimensions). A natural question arises: <i>Why can't we simply compare tokens directly using their raw vectors?</i>", body_style))

    story.append(Paragraph("<b>1.1 The Failure of Direct Dot Products</b>", h2_style))
    story.append(Paragraph("Suppose we took two token vectors <b>x<sub>A</sub></b> and <b>x<sub>B</sub></b> and directly computed their dot product: <code>Score = x<sub>A</sub> · x<sub>B</sub></code>. Three fatal mathematical breakdowns occur immediately:", body_style))

    story.append(Paragraph("<b>1. The Self-Absorption Trap:</b> For any vector <b>x</b>, the dot product with itself (<b>x · x = ||x||²</b>) is almost always strictly greater than its dot product with any other word. If tokens compared themselves directly, every word in a sentence would allocate 90%+ of its attention to itself, effectively blinding the model to context.", body_style))

    story.append(Paragraph("<b>2. The Symmetry Trap:</b> Vector dot products are commutative: <code>x · y = y · x</code>. But human language is fundamentally <b>asymmetric</b>! Consider the sentence:<br/><i>'The dog bit the man.'</i><br/>The verb <i>'bit'</i> must look for who bit (the dog) and who was bitten (the man). However, the word <i>'man'</i> is not looking for <i>'bit'</i> in the same grammatical relationship. Symmetric comparison destroys directional syntactic dependencies.", body_style))

    story.append(Paragraph("<b>3. The Dual-Role Dilemma:</b> In language, every word must simultaneously execute two conflicting functions:<br/>• <b>Function A (The Inquirer):</b> Looking outward for context (e.g. <i>'it'</i> asking <i>'which noun am I?'</i>).<br/>• <b>Function B (The Informant):</b> Advertising its own attributes to other words (e.g. <i>'the robot'</i> declaring <i>'I am a neuter noun'</i>).<br/>A single vector cannot simultaneously ask a question and provide an answer without destructive interference across its channels.", body_style))

    # Diagram 1: QKV Projection Decoupling
    img_path1 = f"{ASSETS_DIR}/fig1_qkv_intuition.png"
    if os.path.exists(img_path1):
        story.append(Spacer(1, 6))
        story.append(Image(img_path1, width=6.2*inch, height=3.3*inch))
        story.append(Paragraph("Figure 1.1: Linear projection of input vector x into Query, Key, and Value subspaces. Each projection isolates a dedicated functional role.", caption_style))

    story.append(PageBreak())
    story.append(Paragraph("<b>1.2 The Database Analogy: Queries, Keys, and Values</b>", h2_style))
    story.append(Paragraph("The transformer resolves the dual-role dilemma by borrowing the fundamental abstraction of modern search engines and databases:", body_style))

    qkv_roles_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Formula</b>", body_style), Paragraph("<b>Database Analogy</b>", body_style), Paragraph("<b>Operational Function in Language</b>", body_style)],
        [Paragraph("<b>Query (Q)</b>", body_style), Paragraph("Q = x<sub>norm</sub> · W<sub>q</sub>", body_style), Paragraph("Search Query String", body_style), Paragraph("What this token is currently <b>looking for</b>. (e.g., <i>'barking'</i> searches for a canine subject).", body_style)],
        [Paragraph("<b>Key (K)</b>", body_style), Paragraph("K = x<sub>norm</sub> · W<sub>k</sub>", body_style), Paragraph("Database Index / Tags", body_style), Paragraph("What this token <b>advertises to others</b>. (e.g., <i>'dog'</i> advertises <i>'I am a canine noun'</i>).", body_style)],
        [Paragraph("<b>Value (V)</b>", body_style), Paragraph("V = x<sub>norm</sub> · W<sub>v</sub>", body_style), Paragraph("Record Content / Payload", body_style), Paragraph("The <b>actual semantic payload</b> transferred if attention is paid (e.g., canine visual/audio features).", body_style)],
    ]
    t_qkv = Table(qkv_roles_data, colWidths=[1.3*inch, 1.6*inch, 1.7*inch, 2.6*inch])
    t_qkv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_qkv)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>1.3 Concrete Walkthrough: Resolving Pronoun Ambiguity</b>", h2_style))
    story.append(Paragraph("Consider the sentence: <code>'The dog chased the cat because it was fast.'</code><br/>When the layer processes the ambiguous pronoun <i>'it'</i>:", body_style))
    story.append(Paragraph("""1. Q_it: The Query projection turns 'it' into a query vector searching for:
   [Semantic Tag: Singular Animate Subject, Context: Prior Sentence Actor]
2. K_dog vs K_cat:
   • K_dog advertises: [Actor, Subject of 'chased', Singular Animate] -> High dot product with Q_it (+14.8)
   • K_cat advertises: [Target, Object of 'chased', Singular Animate]  -> Lower dot product with Q_it (+6.2)
3. Softmax Normalization:
   Softmax converts [14.8, 6.2] into attention weights: 99.9% on 'dog', 0.1% on 'cat'.
4. Value Extraction:
   The layer takes 0.999 * V_dog and adds it to the token 'it'. 
   The pronoun 'it' now absorbs the concrete features of the dog!""", code_style))

    story.append(Paragraph("<b>1.4 Multi-Head Subspace Slicing</b>", h2_style))
    story.append(Paragraph("A single Query-Key-Value projection is still limited to asking one question at a time. In reality, a token needs to ask multiple questions simultaneously:<br/>• Head 0: <i>'Who is my grammatical subject?'</i><br/>• Head 1: <i>'What is my tense/time frame?'</i><br/>• Head 2: <i>'What is my emotional sentiment?'</i><br/>• Head 3: <i>'Is this code syntax properly closed?'</i>", body_style))
    story.append(Paragraph("By slicing d<sub>model</sub> (4096) into <b>n_heads</b> (e.g. 32 heads of d<sub>head</sub> = 128), each attention head learns an independent, orthogonal set of projection weights (W<sub>q</sub><sup>i</sup>, W<sub>k</sub><sup>i</sup>, W<sub>v</sub><sup>i</sup>), allowing 32 parallel semantic inquiries per token across every layer.", body_style))

    # =========================================================================
    # PART 2: ROTARY POSITION EMBEDDING (RoPE) — COMPLETE MECHANICS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Part 2: Rotary Position Embedding (RoPE) Deep-Dive", h1_style))
    story.append(Paragraph("Once Query (Q) and Key (K) vectors are created, the model must know <i>where</i> words are located relative to each other. In 2021, Jianlin Su introduced <b>Rotary Position Embedding (RoPE)</b>, which has replaced all additive positional embeddings in state-of-the-art models (LLaMA, Qwen, Mistral, DeepSeek).", body_style))

    story.append(Paragraph("<b>2.1 What Exactly Does RoPE Measure the Difference Between?</b>", h2_style))
    story.append(Paragraph("RoPE is engineered to satisfy one fundamental mathematical requirement: <b>The inner product of a Query at position <i>m</i> and a Key at position <i>n</i> must depend strictly on their relative distance (m - n), not their absolute positions.</b>", body_style))

    story.append(Paragraph("<b>Δpos = m<sub>Query</sub> - n<sub>Key</sub></b>", formula_style))
    story.append(Paragraph("• If token 5 asks a question (m = 5) and token 2 answers (n = 2), the distance is <b>5 - 2 = 3</b>.<br/>• If the identical sentence appears much later in a 100,000-word document at positions m = 1005 and n = 1002, the relative distance is still <b>1005 - 1002 = 3</b>.<br/>• Under RoPE, their mathematical attention score is <b>100% IDENTICAL</b>! Absolute offsets are completely washed away.", body_style))

    # Diagram 2: RoPE Geometry & Invariance
    img_path2 = f"{ASSETS_DIR}/fig2_rope_geometry.png"
    if os.path.exists(img_path2):
        story.append(Spacer(1, 4))
        story.append(Image(img_path2, width=6.2*inch, height=2.85*inch))
        story.append(Paragraph("Figure 2.1: Left: Independent 2D rotation of Query and Key vectors on the unit circle. Right: Algebraic proof that inner product contracts into relative distance (m - n).", caption_style))

    story.append(Paragraph("<b>2.2 How Position is Encoded: The 2D Coordinate Rotation Trick</b>", h2_style))
    story.append(Paragraph("How do you encode positional distance into high-dimensional vectors without distorting their magnitude (semantic intensity)? <b>You rotate them in 2D planes!</b>", body_style))
    story.append(Paragraph("For head dimension d<sub>head</sub> = 128, RoPE groups the 128 continuous channels into <b>64 independent 2D coordinate pairs</b>:", body_style))
    story.append(Paragraph("(q<sub>0</sub>, q<sub>1</sub>),   (q<sub>2</sub>, q<sub>3</sub>),   (q<sub>4</sub>, q<sub>5</sub>),   ...,   (q<sub>126</sub>, q<sub>127</sub>)", formula_style))

    story.append(Paragraph("Each 2D coordinate pair <i>i</i> is treated as a point on a 2D Cartesian plane (or a complex number z = q<sub>2i</sub> + j·q<sub>2i+1</sub>). For a token at position <i>m</i>, the pair is rotated by an angle proportional to position: <b>α<sub>m,i</sub> = m · θ<sub>i</sub></b>.", body_style))

    story.append(Paragraph("""[ q'_2i   ] = [ cos(m·θ_i)  -sin(m·θ_i) ] [ q_2i   ]
[ q'_2i+1 ]   [ sin(m·θ_i)   cos(m·θ_i) ] [ q_2i+1 ]""", formula_style))

    story.append(PageBreak())
    story.append(Paragraph("<b>2.3 The Trigonometric Proof of Relative Distance Invariance</b>", h2_style))
    story.append(Paragraph("When attention computes the dot product between rotated Query vector <b>q'<sub>m</sub></b> and rotated Key vector <b>k'<sub>n</sub></b> for coordinate pair <i>i</i>:", body_style))

    story.append(Paragraph("""q'_m = [ q_0 cos(mθ) - q_1 sin(mθ),   q_0 sin(mθ) + q_1 cos(mθ) ]
k'_n = [ k_0 cos(nθ) - k_1 sin(nθ),   k_0 sin(nθ) + k_1 cos(nθ) ]

Dot Product = q'_m · k'_n:
= (q_0 cos(mθ) - q_1 sin(mθ))(k_0 cos(nθ) - k_1 sin(nθ))
+ (q_0 sin(mθ) + q_1 cos(mθ))(k_0 sin(nθ) + k_1 cos(nθ))

Expanding terms:
= q_0 k_0 [ cos(mθ)cos(nθ) + sin(mθ)sin(nθ) ]
+ q_1 k_1 [ sin(mθ)sin(nθ) + cos(mθ)cos(nθ) ]
+ q_0 k_1 [ -cos(mθ)sin(nθ) + sin(mθ)cos(nθ) ]
+ q_1 k_0 [ -sin(mθ)cos(nθ) + cos(mθ)sin(nθ) ]

Applying Fundamental Trigonometric Identities:
cos(A - B) = cos(A)cos(B) + sin(A)sin(B)
sin(A - B) = sin(A)cos(B) - cos(A)sin(B)

The equation contracts precisely into:
q'_m · k'_n = (q_0 k_0 + q_1 k_1) cos((m - n)θ)  +  (q_0 k_1 - q_1 k_0) sin((m - n)θ)""", formula_style))

    story.append(Paragraph("<b>The Crucial Insight:</b> Notice that <i>m</i> and <i>n</i> appear nowhere in isolation! They exist solely as the subtracted difference: <b>(m - n)</b>. Vector magnitudes are 100% preserved (since rotation matrices are orthogonal: det(R) = 1), while relative distance is automatically injected into the dot product score.", callout_style))

    # Diagram 3: Frequency Spectrum
    img_path3 = f"{ASSETS_DIR}/fig3_rope_frequencies.png"
    if os.path.exists(img_path3):
        story.append(Spacer(1, 6))
        story.append(Image(img_path3, width=6.2*inch, height=2.75*inch))
        story.append(Paragraph("Figure 2.2: Geometric progression of rotational frequency θ_i across the 64 coordinate pairs. High frequencies capture syntax; low frequencies capture document-level context.", caption_style))

    story.append(Paragraph("<b>2.4 The Multi-Frequency Spectrum: Fast vs Slow Hands</b>", h2_style))
    story.append(Paragraph("Why don't all dimensions rotate at the same speed? RoPE assigns a decaying frequency spectrum across the 64 pairs:", body_style))
    story.append(Paragraph("<b>θ<sub>i</sub> = (base)<sup>-2i / d_head</sup>,   where base = 10,000 (Classical) or 500,000 (LLaMA-3)</b>", formula_style))

    story.append(Paragraph("This creates an acoustic-like Fourier decomposition:<br/>• <b>Pair 0 (i = 0, θ₀ = 1.0):</b> The 'second hand'. Rotates rapidly with every single token. It captures immediate grammatical syntax, adjective-noun binding, and adjacent punctuation.<br/>• <b>Pair 63 (i = 63, θ₆₃ ≈ 0.0001):</b> The 'hour hand'. Rotates imperceptibly slowly. It remains almost constant over thousands of tokens, preserving macro-level topical coherence across distant paragraphs.", body_style))

    story.append(PageBreak())
    story.append(Paragraph("<b>2.5 OpenGP High-Performance RoPE Implementation in C++</b>", h2_style))
    story.append(Paragraph("In our engine, RoPE executes as an in-place SIMD loop across Query and Key tensors. Notice how sin and cos values are pre-computed in tables for maximum throughput:", body_style))

    story.append(Paragraph("""void apply_rope_inplace(float * vec, int pos, int d_head, float freq_base) {
    // Process coordinate pairs (2i, 2i+1)
    for (int i = 0; i < d_head / 2; ++i) {
        float theta = std::pow(freq_base, -2.0f * i / d_head);
        float angle = pos * theta;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        float x0 = vec[2 * i];
        float x1 = vec[2 * i + 1];

        // 2D Rotation:
        vec[2 * i]     = x0 * cos_a - x1 * sin_a;
        vec[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}""", code_style))

    story.append(Paragraph("<b>2.6 RoPE Long-Context Extrapolation: Why LLaMA-3 uses Base = 500,000</b>", h2_style))
    story.append(Paragraph("If a model is trained with <code>freq_base = 10,000</code> on context length 4,096 tokens, feeding it 32,000 tokens causes high-frequency pairs to rotate through full 360° revolutions multiple times, causing catastrophic out-of-distribution confusion.", body_style))
    story.append(Paragraph("By increasing <code>freq_base</code> to <b>500,000</b> (as done in LLaMA-3.1 and Qwen-2.5), rotation speeds are diluted across all dimensions. Angles advance much more slowly per token, enabling seamless attention span scaling up to <b>128,000 tokens</b> without trigonometric saturation!", callout_style))

    # =========================================================================
    # SUMMARY & CHEAT SHEET
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("Architectural Summary & Key Takeaways", h1_style))

    summary_final = [
        [Paragraph("<b>Concept</b>", body_style), Paragraph("<b>Core Question Answered</b>", body_style), Paragraph("<b>Key Mechanism / Formula</b>", body_style), Paragraph("<b>Failure if Omitted</b>", body_style)],
        [Paragraph("<b>Query Projection</b>", body_style), Paragraph("<i>'What am I looking for?'</i>", body_style), Paragraph("Q = x<sub>norm</sub> · W<sub>q</sub>", body_style), Paragraph("Tokens cannot search without confusing their identity.", body_style)],
        [Paragraph("<b>Key Projection</b>", body_style), Paragraph("<i>'What properties do I offer?'</i>", body_style), Paragraph("K = x<sub>norm</sub> · W<sub>k</sub>", body_style), Paragraph("Symmetric dot products destroy grammatical direction.", body_style)],
        [Paragraph("<b>Value Projection</b>", body_style), Paragraph("<i>'What information do I pass?'</i>", body_style), Paragraph("V = x<sub>norm</sub> · W<sub>v</sub>", body_style), Paragraph("Matching criteria becomes entangled with payload.", body_style)],
        [Paragraph("<b>RoPE Embedding</b>", body_style), Paragraph("<i>'How far apart are we?'</i>", body_style), Paragraph("q'_m · k'_n = f(m - n)", body_style), Paragraph("Loss of word order; quadratic position lookup crashes.", body_style)],
    ]
    t_fin = Table(summary_final, colWidths=[1.3*inch, 1.8*inch, 2.1*inch, 2.0*inch])
    t_fin.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_fin)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated deep-dive guide: {OUTPUT_PDF}")

if __name__ == '__main__':
    build_pdf()
