import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import journal_style as js

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.conformal_fdr import conformal_pvalues, benjamini_hochberg

js.apply()
P = js.PALETTE

CACHE_DIR = Path("results/score_cache")
M_FIXED = 175
B = 100
Q_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
HIGHLIGHT = [0.05, 0.20]

def fixed_threshold(cal, eta):
    M = len(cal)
    level = min(np.ceil((M + 1) * (1 - eta)) / M, 1.0)
    idx = min(max(int(np.ceil(level * M)) - 1, 0), M - 1)
    return float(np.sort(cal)[idx])

def recall_at_fpr(cal, ev, y, eta):
    tau = fixed_threshold(cal, eta)
    flags = ev > tau
    tp = np.sum((flags == 1) & (y == 1))
    fn = np.sum((flags == 0) & (y == 1))
    return tp / max(tp + fn, 1)

def recall_at_fdr(cal, ev, y, q):
    pvals = conformal_pvalues(cal, ev)
    reject = benjamini_hochberg(pvals, q)
    tp = np.sum(reject & (y == 1))
    fn = np.sum((~reject) & (y == 1))
    return tp / max(tp + fn, 1)

fpr_recalls = []
fdr_recalls = {q: [] for q in Q_GRID}
for p in sorted(CACHE_DIR.rglob("*__*.npz")):
    d = np.load(p)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < 10 or len(ev) < B or y.sum() == 0:
        continue
    cal_used = cal[-M_FIXED:] if len(cal) >= M_FIXED else cal
    fpr_recalls.append(recall_at_fpr(cal_used, ev, y, 0.05))
    for q in Q_GRID:
        fdr_recalls[q].append(recall_at_fdr(cal_used, ev, y, q))

fpr_mean = np.mean(fpr_recalls)
fdr_means = [np.mean(fdr_recalls[q]) for q in Q_GRID]

fig, ax = plt.subplots(figsize=(4.6, 3.7))
ax.fill_between(Q_GRID, 0, fdr_means, color=P["teal"], alpha=0.15)
ax.plot(Q_GRID, fdr_means, marker="o", markersize=4, color=P["teal"], linewidth=1.6,
        label="FDR framing (BH)")
for hq in HIGHLIGHT:
    idx = Q_GRID.index(hq)
    ax.scatter([hq], [fdr_means[idx]], color=P["orange"], s=55, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.annotate("%.2f" % fdr_means[idx], (hq, fdr_means[idx]),
                textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=P["orange"])
ax.axhline(fpr_mean, color=P["accent"], linestyle="--", linewidth=1.1,
           label=r"FPR framing ($\eta{=}0.05$)")
ax.set_xlabel(r"FDR target $q$")
ax.set_ylabel("mean recall")
ax.set_ylim(0, max(fpr_mean, max(fdr_means)) * 1.15)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/figures/fig6_fdr_operating_curve.pdf")
fig.savefig("results/figures/fig6_fdr_operating_curve.png")
print("Saved fig6")
