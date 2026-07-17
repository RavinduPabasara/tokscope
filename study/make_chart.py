"""Generate the study chart from the benchmark results."""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"  # our tokenizer
NEUTRAL = "#b9b8b1"  # baseline bars (de-emphasis wash of muted)

# (label, fertility, is_ours, note) — OSCAR Sinhala sample, 26,027 words
FERTILITY = [
    ("Sinhala BPE v2 — ours (32k)", 1.58, True, ""),
    ("Sinhala BPE v1 — ours (30k)", 1.74, True, ""),
    ("mBERT (120k)", 1.31, False, "62% UNK — untrustworthy"),
    ("XLM-R (250k)", 2.04, False, ""),
    ("mT5 (250k)", 2.56, False, ""),
    ("GPT-4o · o200k (200k)", 3.26, False, ""),
    ("GPT-4 · cl100k (100k)", 9.80, False, ""),
]

# (label, premium) — 34,469 parallel si/en sentence pairs
PREMIUM = [
    ("Whitespace words", 0.84),
    ("XLM-R (250k)", 1.16),
    ("mT5 (250k)", 1.43),
    ("GPT-4o · o200k (200k)", 2.26),
    ("GPT-4 · cl100k (100k)", 7.37),
]


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.6, 7.4), facecolor=SURFACE, height_ratios=[7, 5]
)

# --- Panel A: fertility ---
rows = list(reversed(FERTILITY))
labels = [r[0] for r in rows]
values = [r[1] for r in rows]
colors = [ACCENT if r[2] else NEUTRAL for r in rows]
bars = ax1.barh(labels, values, color=colors, height=0.55)
style_axis(ax1)
ax1.axvline(1.0, color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)))
ax1.set_ylim(-1.05, len(rows) - 0.45)
ax1.text(1.08, -0.75, "1.0 = one token per word", color=MUTED, fontsize=8, va="center")
for bar, row in zip(bars, rows):
    v, note = row[1], row[3]
    label = f"{v:.2f}" + (f"   ({note})" if note else "")
    ax1.text(v + 0.12, bar.get_y() + bar.get_height() / 2, label, va="center", color=INK, fontsize=9)
ax1.set_xlim(0, 11.4)
ax1.set_title(
    "Tokens per Sinhala word (fertility) — lower is better",
    loc="left", color=INK, fontsize=11, fontweight="bold", pad=10,
)

# --- Panel B: premium ---
rows2 = list(reversed(PREMIUM))
labels2 = [r[0] for r in rows2]
values2 = [r[1] for r in rows2]
bars2 = ax2.barh(labels2, values2, color=NEUTRAL, height=0.55)
style_axis(ax2)
ax2.axvline(1.0, color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)))
ax2.set_ylim(-1.05, len(rows2) - 0.45)
ax2.text(1.08, -0.75, "1.0 = parity with English", color=MUTED, fontsize=8, va="center")
for bar, (label, v) in zip(bars2, rows2):
    ax2.text(v + 0.09, bar.get_y() + bar.get_height() / 2, f"{v:.2f}×", va="center", color=INK, fontsize=9)
ax2.set_xlim(0, 8.6)
ax2.set_title(
    "Token cost of Sinhala vs English for the same content (premium)",
    loc="left", color=INK, fontsize=11, fontweight="bold", pad=10,
)

fig.suptitle(
    "How LLM tokenizers price Sinhala", x=0.02, y=0.985,
    ha="left", color=INK, fontsize=13, fontweight="bold",
)
fig.text(
    0.02, 0.945,
    "OSCAR Sinhala corpus (26k words) · 34,469 parallel si/en sentence pairs · measured with tokscope",
    color=SECONDARY, fontsize=9,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("study/tokenizer_study.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
print("saved study/tokenizer_study.png")
