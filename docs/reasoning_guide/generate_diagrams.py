import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set dark/modern aesthetic
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#f8fafc'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'

ASSETS_DIR = "/media/vithurshan/vithu/opengp/ggml/opengp/docs/reasoning_guide/assets"

# -----------------------------------------------------------------------------
# Diagram 1: Transformer Layer Architecture Flow
# -----------------------------------------------------------------------------
def make_layer_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8, 12), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    def draw_box(y, text, color, subtext="", height=0.9):
        rect = patches.FancyBboxPatch((1.5, y), 7.0, height, boxstyle="round,pad=0.2",
                                      edgecolor=color, facecolor='#1e293b', linewidth=2)
        ax.add_patch(rect)
        ax.text(5.0, y + height/2 + (0.1 if subtext else 0), text, color='#f8fafc',
                fontsize=11, fontweight='bold', ha='center', va='center')
        if subtext:
            ax.text(5.0, y + height/2 - 0.22, subtext, color='#94a3b8',
                    fontsize=8.5, ha='center', va='center')

    # Residual arrows
    ax.annotate('', xy=(1.2, 7.3), xytext=(1.2, 12.0),
                arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2, linestyle='dashed'))
    ax.text(0.7, 9.6, "Residual 1\n(x + Attn(x))", color='#38bdf8', fontsize=8.5, rotation=90, va='center')

    ax.annotate('', xy=(8.8, 1.8), xytext=(8.8, 6.5),
                arrowprops=dict(arrowstyle="->", color='#ec4899', lw=2, linestyle='dashed'))
    ax.text(9.3, 4.1, "Residual 2\n(x + FFN(x))", color='#ec4899', fontsize=8.5, rotation=-90, va='center')

    # Main Blocks
    draw_box(12.5, "Input Activation H_{l} ∈ R^(n × d_model)", '#38bdf8', "From Layer l-1 or Perception Phase")
    draw_box(11.0, "Pre-Attention RMSNorm (blk.l.attn_norm)", '#22c55e', "RMS(x) normalization (Zero Mean Shift)")
    draw_box(9.5, "Q, K, V Projections (W_q, W_k, W_v)", '#eab308', "Latent space to multi-head subspace")
    draw_box(8.2, "RoPE (Rotary Position Embedding)", '#a855f7', "Complex 2D rotation of Q & K vectors")
    draw_box(6.9, "Grouped-Query Attention & KV Cache", '#06b6d4', "Softmax(Q K^T / √d_k) · V + Persistent Cache")
    draw_box(5.6, "Attention Output Projection (W_o) + Residual", '#38bdf8', "H_mid = H_in + W_o · Attn")
    draw_box(4.3, "Post-Attention RMSNorm (blk.l.ffn_norm)", '#22c55e', "Normalize before non-linear reasoning")
    draw_box(3.0, "SwiGLU Feed-Forward Network", '#f97316', "W_down · (SiLU(W_gate · x) ⊙ (W_up · x))")
    draw_box(1.7, "Final Residual Connection", '#ec4899', "H_{l+1} = H_mid + FFN(H_mid)")
    draw_box(0.4, "Output Activation H_{l+1} ∈ R^(n × d_model)", '#38bdf8', "Handover to Layer l+1 or LM Head")

    # Flow arrows
    for y in [12.5, 11.0, 9.5, 8.2, 6.9, 5.6, 4.3, 3.0, 1.7]:
        ax.annotate('', xy=(5.0, y - 0.4), xytext=(5.0, y),
                    arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=1.5))

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig1_layer_flow.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

# -----------------------------------------------------------------------------
# Diagram 2: RoPE Rotary Position Embedding Visual
# -----------------------------------------------------------------------------
def make_rope_diagram():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.5, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.3, color='#334155', fill=False, linestyle='dashed', lw=1.5)
    ax.add_artist(circle)

    # Origin & axes
    ax.axhline(0, color='#1e293b', lw=1)
    ax.axvline(0, color='#1e293b', lw=1)

    # Position 0 vector
    ax.annotate('', xy=(1.1, 0.4), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=3))
    ax.text(1.2, 0.45, "q (pos=0)", color='#38bdf8', fontsize=11, fontweight='bold')

    # Position 1 vector (rotated by theta)
    ax.annotate('', xy=(0.5, 1.15), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='#a855f7', lw=3))
    ax.text(0.55, 1.25, "R_θ · q (pos=1)", color='#a855f7', fontsize=11, fontweight='bold')

    # Position 2 vector (rotated by 2*theta)
    ax.annotate('', xy=(-0.4, 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='#ec4899', lw=3))
    ax.text(-1.1, 1.35, "R_2θ · q (pos=2)", color='#ec4899', fontsize=11, fontweight='bold')

    # Arc for rotation
    arc = patches.Arc((0, 0), 0.8, 0.8, angle=0, theta1=20, theta2=65, color='#eab308', lw=2)
    ax.add_patch(arc)
    ax.text(0.45, 0.5, "θ", color='#eab308', fontsize=12, fontweight='bold')

    ax.text(0, -1.3, "2D Slice (q_{2i}, q_{2i+1}) Rotated by m · θ_i\nPreserves vector magnitude while embedding position",
            color='#94a3b8', fontsize=9.5, ha='center')

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig2_rope_rotation.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

# -----------------------------------------------------------------------------
# Diagram 3: MHA vs GQA vs MQA Head Architecture
# -----------------------------------------------------------------------------
def make_attention_variants_diagram():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.2), facecolor='#0f172a')
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#0f172a')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    def draw_heads(ax, title, q_heads, kv_heads, color_kv, ratio_text):
        ax.text(5, 9.2, title, color='#f8fafc', fontsize=11, fontweight='bold', ha='center')
        ax.text(5, 8.4, ratio_text, color='#38bdf8', fontsize=9, ha='center')

        # Draw Q heads
        ax.text(2.2, 7.3, "Q Heads", color='#94a3b8', fontsize=8.5, ha='center')
        for i in range(q_heads):
            y = 6.4 - i * 0.7
            rect = patches.Rectangle((1.0, y), 2.4, 0.5, facecolor='#1e293b', edgecolor='#38bdf8', lw=1.5)
            ax.add_patch(rect)
            ax.text(2.2, y + 0.25, f"Q_{i}", color='#f8fafc', fontsize=7.5, ha='center', va='center')

        # Draw KV heads
        ax.text(7.5, 7.3, "KV Heads", color='#94a3b8', fontsize=8.5, ha='center')
        step = (q_heads / kv_heads) * 0.7
        for k in range(kv_heads):
            y = 6.4 - k * step
            rect = patches.Rectangle((6.3, y - (step-0.7)/2), 2.4, 0.5 + (step - 0.7),
                                     facecolor='#1e293b', edgecolor=color_kv, lw=1.5)
            ax.add_patch(rect)
            ax.text(7.5, y + 0.25, f"KV_{k}", color='#f8fafc', fontsize=7.5, ha='center', va='center')

            # Draw mapping lines
            for q_idx in range(int(q_heads / kv_heads)):
                qy = 6.4 - (k * int(q_heads / kv_heads) + q_idx) * 0.7 + 0.25
                kvy = y + 0.25
                ax.plot([3.5, 6.2], [qy, kvy], color='#475569', lw=1, linestyle=':')

    draw_heads(ax1, "MHA (Multi-Head)", 8, 8, '#22c55e', "8 Q Heads : 8 KV Heads")
    draw_heads(ax2, "GQA (Grouped-Query)", 8, 2, '#eab308', "8 Q Heads : 2 KV Heads (4x savings)")
    draw_heads(ax3, "MQA (Multi-Query)", 8, 1, '#ec4899', "8 Q Heads : 1 KV Head (8x savings)")

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig3_attention_variants.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

# -----------------------------------------------------------------------------
# Diagram 4: SwiGLU Gated Feed-Forward Architecture
# -----------------------------------------------------------------------------
def make_swiglu_diagram():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input x
    rect_in = patches.FancyBboxPatch((4.0, 7.0), 2.0, 0.6, boxstyle="round,pad=0.1",
                                     edgecolor='#38bdf8', facecolor='#1e293b', lw=1.5)
    ax.add_patch(rect_in)
    ax.text(5.0, 7.3, "x ∈ R^d_model", color='#f8fafc', fontsize=9.5, fontweight='bold', ha='center', va='center')

    # Gate & Up Branches
    rect_gate = patches.FancyBboxPatch((1.5, 5.0), 3.0, 0.7, boxstyle="round,pad=0.1",
                                       edgecolor='#eab308', facecolor='#1e293b', lw=1.5)
    ax.add_patch(rect_gate)
    ax.text(3.0, 5.35, "Gate: SiLU(W_gate · x)", color='#f8fafc', fontsize=8.5, ha='center', va='center')

    rect_up = patches.FancyBboxPatch((5.5, 5.0), 3.0, 0.7, boxstyle="round,pad=0.1",
                                     edgecolor='#22c55e', facecolor='#1e293b', lw=1.5)
    ax.add_patch(rect_up)
    ax.text(7.0, 5.35, "Up: (W_up · x)", color='#f8fafc', fontsize=8.5, ha='center', va='center')

    # Element-wise multiply circle
    circle_mul = plt.Circle((5.0, 3.5), 0.4, color='#ec4899', fill=True)
    ax.add_artist(circle_mul)
    ax.text(5.0, 3.5, "⊙", color='#0f172a', fontsize=16, fontweight='bold', ha='center', va='center')

    # Down Projection
    rect_down = patches.FancyBboxPatch((3.0, 1.8), 4.0, 0.7, boxstyle="round,pad=0.1",
                                       edgecolor='#a855f7', facecolor='#1e293b', lw=1.5)
    ax.add_patch(rect_down)
    ax.text(5.0, 2.15, "Down: W_down · (Gate ⊙ Up)", color='#f8fafc', fontsize=9, fontweight='bold', ha='center', va='center')

    # Output
    rect_out = patches.FancyBboxPatch((4.0, 0.4), 2.0, 0.6, boxstyle="round,pad=0.1",
                                      edgecolor='#38bdf8', facecolor='#1e293b', lw=1.5)
    ax.add_patch(rect_out)
    ax.text(5.0, 0.7, "Output ∈ R^d_model", color='#f8fafc', fontsize=9.5, fontweight='bold', ha='center', va='center')

    # Lines
    ax.plot([5.0, 3.0], [7.0, 5.7], color='#94a3b8', lw=1.5)
    ax.plot([5.0, 7.0], [7.0, 5.7], color='#94a3b8', lw=1.5)
    ax.plot([3.0, 4.6], [5.0, 3.5], color='#94a3b8', lw=1.5)
    ax.plot([7.0, 5.4], [5.0, 3.5], color='#94a3b8', lw=1.5)
    ax.plot([5.0, 5.0], [3.1, 2.5], color='#94a3b8', lw=1.5)
    ax.plot([5.0, 5.0], [1.8, 1.0], color='#94a3b8', lw=1.5)

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig4_swiglu_architecture.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

if __name__ == '__main__':
    print("Generating diagrams...")
    make_layer_architecture_diagram()
    make_rope_diagram()
    make_attention_variants_diagram()
    make_swiglu_diagram()
    print("All diagrams generated successfully in assets/!")
