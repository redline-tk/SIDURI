import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import journal_style as js

js.apply()
P = js.PALETTE
PRIMARY = P.get("primary", "#1B4F72")
ACCENT  = P.get("accent",  "#C0392B")
GREEN   = P.get("green",   "#1E8449")
MUTED   = P.get("muted",   "#7F8C8D")

fig = plt.figure(figsize=(11.5, 5.0))

def box(ax, x, y, w, h, text, fc="white", ec="#333333", fontsize=10.5,
        weight="normal", zorder=3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.09,rounding_size=0.14",
                        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, weight=weight, zorder=zorder + 1)
    return (x, y, w, h)

def varrow(ax, x, y0, y1, color, lw=2.0, ls="solid", zorder=2):
    a = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=15,
                         linewidth=lw, edgecolor=color, facecolor=color,
                         linestyle=ls, zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

ax1 = fig.add_axes([0.02, 0.06, 0.36, 0.78])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 7.6)
ax1.axis("off")

score = box(ax1, 1.0, 6.1, 8.0, 1.1, r"anomaly score $S_t$", fc="#F5F5F5")
thresh = box(ax1, 1.0, 4.0, 8.0, 1.1, r"adaptive threshold $\tau_t$",
             fc="white", ec=PRIMARY)
buf = box(ax1, 1.0, 1.9, 8.0, 1.1, r"calibration buffer",
          fc="white", ec="#0F6E56")

varrow(ax1, 5.0, 6.1, 5.1, color="#333333")

arrow_x_left, arrow_x_right = 3.4, 6.6
varrow(ax1, arrow_x_left, 4.0, 3.0, color=ACCENT, ls=(0, (4, 2)))
varrow(ax1, arrow_x_right, 4.0, 3.0, color=GREEN)

ax1.text(arrow_x_left - 0.35, 3.5, "no\nconfirmation", fontsize=7.5, color=ACCENT,
         ha="right", va="center", style="italic", linespacing=1.3)
ax1.text(arrow_x_right + 0.35, 3.5, "confirmed\nnormal", fontsize=7.5, color=GREEN,
         ha="left", va="center", style="italic", linespacing=1.3)

legend_y0, legend_y1 = 1.15, 0.5
ax1.plot([1.0, 1.8], [legend_y0, legend_y0], color=ACCENT, lw=2.0,
         linestyle=(0, (4, 2)))
ax1.text(2.0, legend_y0, "discarded, never re-enters the buffer",
         fontsize=9, color=ACCENT, va="center")
ax1.plot([1.0, 1.8], [legend_y1, legend_y1], color=GREEN, lw=2.2)
ax1.text(2.0, legend_y1, "prevents the runaway to FPR = 1",
         fontsize=9, color=GREEN, va="center")

ax2 = fig.add_axes([0.46, 0.14, 0.40, 0.66])

eta = 0.05
q_star = 0.4737  # q* at 2x tolerance, per Table 1
q = np.linspace(0.001, 1.0, 500)
fpr_star = eta / (eta + (1 - eta) * q)
fpr_at_qstar = eta / (eta + (1 - eta) * q_star)

ax2.plot(q, fpr_star, color=PRIMARY, linewidth=2.3, zorder=4)
ax2.axhline(eta, color=MUTED, linestyle=":", linewidth=1.2, zorder=1,
            label=r"nominal $\eta=0.05$")
ax2.axhline(1.0, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.85,
            zorder=1, label=r"runaway limit ($q{=}0$)")
ax2.axvline(q_star, color=GREEN, linestyle="--", linewidth=1.2, alpha=0.85,
            zorder=1, label=r"$q^*\approx 0.47$ (2$\times$ tolerance)")

ax2.scatter([0.0], [1.0], color=ACCENT, edgecolor="white", linewidth=1.0,
            s=42, zorder=5, clip_on=False)
ax2.scatter([1.0], [eta], color=GREEN, edgecolor="white", linewidth=1.0,
            s=42, zorder=5, clip_on=False)
ax2.scatter([q_star], [fpr_at_qstar], color=GREEN, edgecolor="white",
            linewidth=1.0, s=36, zorder=5)

ax2.text(0.02, 0.90, "runaway\n($q=0$)", fontsize=9, color=ACCENT, weight="bold",
         ha="left", va="top", linespacing=1.3)

ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel(r"feedback fraction $q$")
ax2.set_ylabel(r"stationary $\mathrm{FPR}^*(q)$")
ax2.legend(loc="upper right", frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.96, 0.86))

ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)

fig.text(0.02, 0.965, "SIDURI", fontsize=20, weight="bold", color=PRIMARY,
          ha="left", va="top")
fig.text(0.16, 0.975,
          "Calibration that ignores its own flagged points silently drifts to a 100% false-positive rate, while",
          fontsize=11, ha="left", va="top", color="#2C3E50")
fig.text(0.16, 0.925,
          "confirming about half of them is enough to stop it.",
          fontsize=11, ha="left", va="top", color="#2C3E50")

fig.savefig("results/figures/fig0_graphical_abstract.pdf",
            bbox_inches="tight", pad_inches=0.15)
fig.savefig("results/figures/fig0_graphical_abstract.png",
            bbox_inches="tight", pad_inches=0.15, dpi=300)
print("Saved fig0_graphical_abstract")
