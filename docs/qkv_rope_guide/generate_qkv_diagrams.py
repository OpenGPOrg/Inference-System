import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dark/modern architectural styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#f8fafc'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'

ASSETS_DIR = "/media/vithurshan/vithu/opengp/ggml/opengp/docs/qkv_rope_guide/assets"

# -----------------------------------------------------------------------------
# Figure 1: Q, K, V Projection & Decoupling Intuition
# -----------------------------------------------------------------------------
def make_qkv_intuition_diagram():
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Input token box
    rect_x = patches.FancyBboxPatch((0.5, 2.5), 2.2, 1.8, boxstyle="round,pad=0.2",
                                    edgecolor='#38bdf8', facecolor='#1e293b', lw=2)
    ax.add_patch(rect_x)
    ax.text(1.6, 3.6, "Token Vector x", color='#f8fafc', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.6, 3.1, "e.g., 'barking'", color='#38bdf8', fontsize=10, ha='center')
    ax.text(1.6, 2.7, "x ∈ R^d_model", color='#94a3b8', fontsize=8.5, ha='center')

    # Three projection boxes
    def draw_proj(y, name, matrix, role, color, desc):
        # Weight matrix box
        rect_w = patches.Rectangle((4.0, y+0.1), 1.6, 1.0, facecolor='#1e293b', edgecolor='#64748b', lw=1.5, linestyle='--')
        ax.add_patch(rect_w)
        ax.text(4.8, y+0.6, matrix, color='#cbd5e1', fontsize=9.5, fontweight='bold', ha='center')

        # Output vector box
        rect_out = patches.FancyBboxPatch((7.0, y), 4.5, 1.2, boxstyle="round,pad=0.15",
                                          edgecolor=color, facecolor='#1e293b', lw=2)
        ax.add_patch(rect_out)
        ax.text(7.3, y+0.8, name, color=color, fontsize=11, fontweight='bold')
        ax.text(7.3, y+0.45, f"Role: {role}", color='#f8fafc', fontsize=9)
        ax.text(7.3, y+0.2, f"Intuition: {desc}", color='#94a3b8', fontsize=8)

        # Arrows
        ax.annotate('', xy=(4.0, y+0.6), xytext=(2.7, 3.4),
                    arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=1.5))
        ax.annotate('', xy=(7.0, y+0.6), xytext=(5.6, y+0.6),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

    draw_proj(4.8, "Query (Q = x · W_q)", "W_q", "The Question", '#eab308', "'Looking for canine subject'")
    draw_proj(2.6, "Key (K = x · W_k)", "W_k", "The Advertisement", '#06b6d4', "'I am an action / present verb'")
    draw_proj(0.4, "Value (V = x · W_v)", "W_v", "The Payload", '#22c55e', "'Apply vocalization attributes'")

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig1_qkv_intuition.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

# -----------------------------------------------------------------------------
# Figure 2: The Geometry of RoPE & Angle Subtraction
# -----------------------------------------------------------------------------
def make_rope_geometry_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), facecolor='#0f172a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0f172a')
        ax.axis('off')

    # Subplot 1: Rotating Query & Key in 2D Plane
    ax1.set_xlim(-1.6, 1.6)
    ax1.set_ylim(-1.6, 1.6)
    ax1.set_aspect('equal')

    c1 = plt.Circle((0, 0), 1.2, color='#334155', fill=False, linestyle='--', lw=1.2)
    ax1.add_artist(c1)
    ax1.axhline(0, color='#1e293b', lw=1)
    ax1.axvline(0, color='#1e293b', lw=1)

    # Key vector at pos n (angle 30 deg)
    theta_k = np.radians(30)
    xk, yk = 1.2 * np.cos(theta_k), 1.2 * np.sin(theta_k)
    ax1.annotate('', xy=(xk, yk), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='#06b6d4', lw=2.5))
    ax1.text(xk+0.05, yk+0.05, "k' (pos n = 2)\nangle = 2θ", color='#06b6d4', fontsize=9, fontweight='bold')

    # Query vector at pos m (angle 75 deg)
    theta_q = np.radians(75)
    xq, yq = 1.2 * np.cos(theta_q), 1.2 * np.sin(theta_q)
    ax1.annotate('', xy=(xq, yq), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='#eab308', lw=2.5))
    ax1.text(xq-0.3, yq+0.12, "q' (pos m = 5)\nangle = 5θ", color='#eab308', fontsize=9, fontweight='bold')

    # Delta angle arc
    arc = patches.Arc((0, 0), 0.9, 0.9, angle=0, theta1=30, theta2=75, color='#ec4899', lw=2.5)
    ax1.add_patch(arc)
    ax1.text(0.35, 0.45, "Δθ = (5 - 2)θ\n    = 3θ", color='#ec4899', fontsize=9, fontweight='bold')

    ax1.text(0, -1.5, "Independent Rotations on Unit Circle\nq rotates by m·θ, k rotates by n·θ",
             color='#94a3b8', fontsize=8.5, ha='center')

    # Subplot 2: Relative Dot Product Invariance
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)

    ax2.text(5, 7.2, "Mathematical Inner Product Invariance", color='#38bdf8', fontsize=11, fontweight='bold', ha='center')

    text_block = (
        "⟨ q'_m,  k'_n ⟩ = (R_{mθ} · q)ᵀ (R_{nθ} · k)\n"
        "               = qᵀ (R_{mθ}ᵀ R_{nθ}) k\n"
        "               = qᵀ R_{(n - m)θ} k\n\n"
        "Key Proof Discovery:\n"
        "• The absolute indices m and n disappear completely!\n"
        "• The attention score depends strictly on (m - n):\n"
        "  Δpos = 5 - 2 = 3\n"
        "• If positions shift to m=105, n=102:\n"
        "  Δpos = 105 - 102 = 3\n"
        "  ==> Attention score is 100% IDENTICAL!"
    )
    rect_box = patches.FancyBboxPatch((0.5, 0.8), 9.0, 5.8, boxstyle="round,pad=0.2",
                                      edgecolor='#a855f7', facecolor='#1e293b', lw=1.5)
    ax2.add_patch(rect_box)
    ax2.text(0.8, 6.0, text_block, color='#f8fafc', fontsize=8.5, fontfamily='monospace', va='top')

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig2_rope_geometry.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

# -----------------------------------------------------------------------------
# Figure 3: Multiscale Frequency Spectrum across Head Dimensions
# -----------------------------------------------------------------------------
def make_rope_spectrum_diagram():
    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    # Dimension pairs 0..63 for d_head = 128
    pairs = np.arange(0, 64)
    # Standard theta_i = 10000^(-2i / 128)
    thetas_base10k = 10000 ** (-2 * pairs / 128.0)
    thetas_llama3 = 500000 ** (-2 * pairs / 128.0)

    ax.semilogy(pairs, thetas_base10k, color='#38bdf8', lw=2.2, label='Base = 10,000 (Standard RoPE)')
    ax.semilogy(pairs, thetas_llama3, color='#eab308', lw=2.2, linestyle='--', label='Base = 500,000 (LLaMA-3 128k Extension)')

    ax.set_title("RoPE Frequency Spectrum Across 2D Coordinate Pairs (d_head = 128)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax.set_xlabel("Coordinate Pair Index i (Channels 2i, 2i+1)", color='#94a3b8', fontsize=9.5)
    ax.set_ylabel("Rotational Frequency θ_i (log scale)", color='#94a3b8', fontsize=9.5)
    ax.grid(True, which='both', color='#334155', linestyle=':', alpha=0.6)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)

    # Annotations
    ax.annotate("High Frequency: Fast Rotations\nCaptures immediate syntax & neighboring words",
                xy=(3, thetas_base10k[3]), xytext=(8, 0.2),
                arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=1.2),
                color='#38bdf8', fontsize=8)

    ax.annotate("Low Frequency: Slow Rotations\nCaptures long-range semantics across paragraphs",
                xy=(55, thetas_base10k[55]), xytext=(22, 1e-4),
                arrowprops=dict(arrowstyle="->", color='#ec4899', lw=1.2),
                color='#ec4899', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/fig3_rope_frequencies.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

if __name__ == '__main__':
    print("Generating diagrams for QKV & RoPE guide...")
    make_qkv_intuition_diagram()
    make_rope_geometry_diagram()
    make_rope_spectrum_diagram()
    print("Diagrams successfully generated in assets/!")
