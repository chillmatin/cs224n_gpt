#!/usr/bin/env python3
"""Generate the presentation figures from our real reported numbers.
Run: python slides/make_figs.py  (writes slides/figs/*.png)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 16,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "figure.dpi": 160,
})

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# Color palette
BLUE, ORANGE, GREEN, GREY, RED = "#2563eb", "#f59e0b", "#16a34a", "#cbd5e1", "#dc2626"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# 1. Sentiment: last-linear vs full-model, with handout baselines
# --------------------------------------------------------------------------- #
def sentiment():
    import numpy as np
    labels = ["SST-5", "CFIMDB"]
    last = [0.461, 0.857]
    full = [0.513, 0.967]
    base_last = [0.462, 0.861]
    base_full = [0.513, 0.976]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w/2, last, w, label="Last-linear (frozen body)", color=GREY)
    b2 = ax.bar(x + w/2, full, w, label="Full fine-tune", color=BLUE)
    # baseline ticks
    for xi, bl, bf in zip(x, base_last, base_full):
        ax.plot([xi - w, xi], [bl, bl], color=RED, lw=2, ls="--")
        ax.plot([xi, xi + w], [bf, bf], color=RED, lw=2, ls="--")
    ax.plot([], [], color=RED, lw=2, ls="--", label="Handout baseline")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Dev accuracy")
    ax.set_title("Sentiment classification")
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    save(fig, "sentiment.png")


# --------------------------------------------------------------------------- #
# 2. Paraphrase: dev accuracy over epochs
# --------------------------------------------------------------------------- #
def paraphrase():
    ep = [0, 1, 2]
    acc = [0.863, 0.879, 0.889]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ep, acc, "-o", color=BLUE, lw=3, ms=12)
    for e, a in zip(ep, acc):
        ax.text(e, a + 0.0035, f"{a:.3f}", ha="center", fontsize=15, fontweight="bold")
    ax.scatter([2], [0.889], s=320, facecolors="none", edgecolors=GREEN, lw=3, zorder=5)
    ax.set_xticks(ep); ax.set_xlabel("Training epoch")
    ax.set_ylim(0.85, 0.90); ax.set_ylabel("Quora dev accuracy")
    ax.set_title("Cloze-style paraphrase detection")
    save(fig, "paraphrase.png")


# --------------------------------------------------------------------------- #
# 3. Sonnet: chrF vs decoding temperature
# --------------------------------------------------------------------------- #
def sonnet():
    t = [0.7, 0.9, 1.0, 1.2]
    chrf = [39.01, 40.63, 41.49, 41.12]
    colors = [GREY, GREY, GREEN, GREY]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar([str(x) for x in t], chrf, color=colors, width=0.6)
    for b, c in zip(bars, chrf):
        ax.text(b.get_x() + b.get_width()/2, c + 0.12, f"{c:.2f}",
                ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylim(38, 42.5); ax.set_ylabel("Dev chrF")
    ax.set_xlabel("Decoding temperature  (top-p = 0.9)")
    ax.set_title("Sonnet generation: temperature sweep")
    ax.annotate("selected", xy=(2, 41.49), xytext=(2.0, 38.7),
                ha="center", color=GREEN, fontweight="bold", fontsize=14,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    save(fig, "sonnet_temp.png")


# --------------------------------------------------------------------------- #
# 4. Extension: zero-shot vs fine-tuned chrF
# --------------------------------------------------------------------------- #
def extension():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = ["Zero-shot\nGPT-2", "Fine-tuned\n(source-side mask)"]
    vals = [13.85, 35.83]
    bars = ax.bar(names, vals, color=[GREY, BLUE], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.2f}",
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.annotate("", xy=(1, 35.0), xytext=(0, 14.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5))
    ax.text(0.5, 26, "+21.97 chrF", ha="center", color=GREEN, fontsize=17, fontweight="bold")
    ax.set_ylim(0, 41); ax.set_ylabel("Test chrF")
    ax.set_title("Extension: Modern → Shakespearean")
    save(fig, "extension.png")


# --------------------------------------------------------------------------- #
# 5. Prompt whitespace ablation
# --------------------------------------------------------------------------- #
def prompt_ablation():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = ['tag ends with space\n"Shakespearean: "', 'space moved to target\n"Shakespearean:" + " {t}"']
    vals = [29.52, 35.83]
    bars = ax.bar(names, vals, color=[RED, GREEN], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.2f}",
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.text(0.5, 32.5, "+6.3 chrF\nfrom one space", ha="center", color=GREEN,
            fontsize=15, fontweight="bold")
    ax.set_ylim(0, 40); ax.set_ylabel("Test chrF")
    ax.set_title("The BPE whitespace footgun")
    save(fig, "prompt_ablation.png")


# --------------------------------------------------------------------------- #
# 6. Equations (rendered offline so the deck needs no MathJax/CDN)
# --------------------------------------------------------------------------- #
def equations():
    fig = plt.figure(figsize=(7.2, 3.2))
    fig.text(0.5, 0.86, "Masked self-attention", ha="center", fontsize=18, fontweight="bold", color=BLUE)
    fig.text(0.5, 0.60,
             r"$\mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d_k}} + M\right)V$",
             ha="center", fontsize=26)
    fig.text(0.5, 0.32, "AdamW  (decoupled weight decay)", ha="center", fontsize=18, fontweight="bold", color=BLUE)
    fig.text(0.5, 0.07,
             r"$\theta_t = \theta_{t-1} - \eta\left(\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\right)$",
             ha="center", fontsize=22)
    save(fig, "equations.png")


# --------------------------------------------------------------------------- #
# 7. Architecture: the pre-LayerNorm transformer block (built from scratch)
# --------------------------------------------------------------------------- #
def architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    def box(x, y, w, h, text, fc, ec=BLUE, fs=13, tc="#0f172a"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, color="#334155"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.8, color=color))

    cx, w = 5.0, 4.2
    # bottom-up stack
    box(cx - w/2, 0.2, w, 0.8, "Token + positional embeddings", "#e0e7ff", fs=12)
    arrow(cx, 1.0, cx, 1.5)

    # the dashed transformer block (x12)
    ax.add_patch(FancyBboxPatch((cx - w/2 - 0.45, 1.55), w + 0.9, 7.55,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc="none", ec=GREY, lw=2.0, linestyle=(0, (6, 4))))
    ax.text(cx + w/2 + 0.05, 8.75, r"$\times\,12$", ha="left", va="center",
            fontsize=18, fontweight="bold", color="#475569")

    box(cx - w/2, 1.8, w, 0.75, "LayerNorm", "#f1f5f9", ec=GREY, fs=12)
    arrow(cx, 2.55, cx, 2.95)
    box(cx - w/2, 3.0, w, 0.95, "Masked multi-head\nself-attention", "#dbeafe", fs=12)
    # residual 1
    ax.add_patch(FancyArrowPatch((cx - w/2 - 0.3, 1.6), (cx - w/2 - 0.3, 4.2),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.6, color=ORANGE))
    ax.add_patch(FancyArrowPatch((cx - w/2 - 0.3, 4.2), (cx - w/2, 4.2),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.6, color=ORANGE))
    box(cx - w/2, 4.05, w, 0.55, "+  residual", "#fff7ed", ec=ORANGE, fs=11)
    arrow(cx, 4.6, cx, 5.0)
    box(cx - w/2, 5.05, w, 0.75, "LayerNorm", "#f1f5f9", ec=GREY, fs=12)
    arrow(cx, 5.8, cx, 6.2)
    box(cx - w/2, 6.25, w, 0.95, "GELU MLP\n(feed-forward)", "#dbeafe", fs=12)
    # residual 2
    ax.add_patch(FancyArrowPatch((cx - w/2 - 0.3, 4.85), (cx - w/2 - 0.3, 7.45),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.6, color=ORANGE))
    ax.add_patch(FancyArrowPatch((cx - w/2 - 0.3, 7.45), (cx - w/2, 7.45),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.6, color=ORANGE))
    box(cx - w/2, 7.3, w, 0.55, "+  residual", "#fff7ed", ec=ORANGE, fs=11)
    arrow(cx, 7.85, cx, 9.3)

    box(cx - w/2, 9.35, w, 0.75, "Final LayerNorm", "#f1f5f9", ec=GREY, fs=12)
    arrow(cx, 10.1, cx, 10.5)
    box(cx - w/2, 10.55, w, 0.85, "Output projection\n(weights tied to embedding)", "#dcfce7",
        ec=GREEN, fs=12)

    ax.text(0.15, 4.3, "pre-LayerNorm\ntransformer block", rotation=90, va="center",
            ha="center", fontsize=12, color="#475569", style="italic")
    ax.set_title("GPT-2 decoder block (implemented from scratch)", fontsize=16, fontweight="bold")
    save(fig, "architecture.png")


if __name__ == "__main__":
    architecture()
    sentiment()
    paraphrase()
    sonnet()
    extension()
    prompt_ablation()
    equations()
    print("all figures ->", OUT)
