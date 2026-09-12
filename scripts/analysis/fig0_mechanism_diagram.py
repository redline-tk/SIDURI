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

fig, ax = plt.subplots(figsize=(11.0, 6.2))
ax.set_xlim(0, 17.5)
ax.set_ylim(0.3, 8.6)
ax.axis("off")

def box(x, y, w, h, text, fc="white", ec="#333333", fontsize=10, weight="normal", zorder=3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.13",
                        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
                 weight=weight, zorder=zorder+1)
    return (x, y, w, h)

def curved_arrow(p0, p1, color="#333333", rad=0.0, lw=1.6, zorder=2):
    a = FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=%.2f" % rad,
                         arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                         edgecolor=color, facecolor=color, zorder=zorder,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def elbow_line(points, color="#333333", lw=2.0, zorder=2, ls="solid"):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder, solid_capstyle="round",
            solid_joinstyle="round", linestyle=ls)

def elbow_arrowhead(p_from, p_to, color="#333333", lw=2.0, zorder=2, scale=16, linestyle="solid"):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=scale,
                         linewidth=lw, edgecolor=color, facecolor=color,
                         linestyle=linestyle, zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def inset_curve(box_xywh, color, q_dot, pad_frac=0.15):
    x, y, w, h = box_xywh
    px, py = w * pad_frac, h * pad_frac
    axins = ax.inset_axes([x + px, y + py, w - 2*px, h - 2*py], transform=ax.transData)
    q_arr = np.linspace(0.001, 1, 200)
    eta = 0.05
    axins.plot(q_arr, eta / (eta + (1 - eta) * q_arr), color=color, linewidth=1.7)
    axins.axvline(q_dot, color=color, linestyle=":", linewidth=1.0)
    axins.scatter([q_dot], [eta / (eta + (1 - eta) * max(q_dot, 1e-6))], color=color, s=20, zorder=5)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_xlabel("$q$", fontsize=7.5, labelpad=1)
    for spine in axins.spines.values():
        spine.set_linewidth(0.7)
    return axins

score = box(0.3, 5.5, 2.1, 1.0, r"score $S_t$", fc="#F5F5F5")
thresh = box(2.9, 5.5, 2.3, 1.0, r"threshold $\tau_t$", fc="white", ec=P["primary"], weight="normal")
buf = box(2.9, 3.0, 2.3, 1.0, r"calibration" "\n" r"buffer", fc="white", ec=P["primary"], fontsize=9.5)

curved_arrow((2.4, 6.0), (2.9, 6.0), color="#333333")
curved_arrow((3.5, 4.0), (3.5, 5.5), color=P["primary"], rad=-0.4, lw=1.6)
ax.text(2.85, 4.75, "recompute", fontsize=7.5, color=P["primary"], rotation=90, va="center", ha="center", style="italic")
# was P["teal"] -- now matches the "recompute" arrow's blue so the two loop
# arrows read as the same shade instead of two different blues
curved_arrow((4.4, 5.5), (4.4, 4.0), color=P["primary"], rad=-0.4, lw=1.6)
ax.text(5.0, 4.75, r"$S_t \leq \tau_t$", fontsize=7.5, color=P["primary"], rotation=-90, va="center", ha="center", style="italic")

ax.text(6.3, 6.25, r"$S_t > \tau_t$" "\n" "(flagged)", fontsize=8.5, style="italic", ha="center", va="bottom")

corner = (7.0, 6.0)
elbow_line([(5.2, 6.0), corner], color="#333333", lw=2.0)
elbow_line([corner, (7.0, 2.6)], color="#333333", lw=2.0)

unsafe = box(7.6, 5.55, 2.9, 0.9, r"no confirmation" "\n" r"(prob. $1-q$)",
             fc="#FDEDEC", ec=P["accent"], fontsize=8.5)
elbow_arrowhead(corner, (7.6, 6.0), color="#333333", lw=2.0)

safe = box(7.6, 2.15, 2.9, 0.9, r"external confirmation" "\n" r"(prob. $q$)",
           fc="#EAF3EC", ec=P["green"], fontsize=8.5)
elbow_arrowhead((7.0, 2.6), (7.6, 2.6), color="#333333", lw=2.0)

discard_box = box(8.3, 4.0, 1.5, 0.6, r"discarded", fc="#FDEDEC", ec=P["accent"], fontsize=8.0, weight="normal")
elbow_arrowhead((9.05, 5.55), (9.05, 4.6), color=P["accent"], lw=1.7)

elbow_arrowhead((10.5, 6.0), (13.8, 6.0), color=P["accent"], lw=1.2, scale=14, linestyle="--")
ax.text(12.15, 6.2, r"as $t \to \infty$", fontsize=7.5, color="#777777", ha="center", va="bottom", style="italic")

runaway = box(13.8, 5.15, 3.3, 1.7, "", fc="white", ec=P["accent"])
inset_curve(runaway, P["accent"], q_dot=0.0)
ax.text(15.45, 4.85, r"$\mathrm{FPR}(q{=}0)\to 1$", fontsize=10, color=P["accent"],
        weight="bold", ha="center", va="top")

elbow_arrowhead((10.5, 2.6), (13.8, 2.6), color=P["green"], lw=1.2, scale=14, linestyle="--")
ax.text(12.15, 2.8, r"as $t \to \infty$", fontsize=7.5, color="#777777", ha="center", va="bottom", style="italic")

safe_curve = box(13.8, 1.75, 3.3, 1.7, "", fc="white", ec=P["green"])
inset_curve(safe_curve, P["green"], q_dot=0.5)
ax.text(15.45, 1.45, r"$\mathrm{FPR}^*(q)$", fontsize=10, color=P["green"],
        weight="bold", ha="center", va="top")

elbow_line([(9.05, 2.15), (9.05, 1.5), (4.05, 1.5)], color=P["green"], lw=1.8)
elbow_arrowhead((4.05, 1.5), (4.05, 3.0), color=P["green"], lw=1.8, scale=16)
ax.text(6.55, 1.65, r"confirmed normal $\to$ buffer", fontsize=8.0, color=P["green"], ha="center", va="bottom", style="italic")

fig.savefig("results/figures/fig0_mechanism_diagram.pdf", bbox_inches="tight")
fig.savefig("results/figures/fig0_mechanism_diagram.png", bbox_inches="tight")
print("Saved tight fig0_mechanism_diagram")
