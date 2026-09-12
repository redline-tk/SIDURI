import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import journal_style as js

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import m_min

js.apply()

eta_grid = [0.01, 0.05, 0.10, 0.20]
colors = ["#1F4E78", "#0072B2", "#D95F02", "#009E73"]
DELTA_MIN, DELTA_MAX = 0.005, 0.49

def true_breakpoints(eta, delta_lo, delta_hi, fine_n=20000):
    fine_delta = np.linspace(delta_lo, delta_hi, fine_n)
    fine_m = np.array([m_min(eta, d) for d in fine_delta])
    change_idx = np.where(np.diff(fine_m) != 0)[0]
    bp_delta = [delta_lo] + list(fine_delta[change_idx + 1]) + [delta_hi]
    bp_m = [fine_m[0]] + list(fine_m[change_idx + 1]) + [fine_m[-1]]
    return np.array(bp_delta), np.array(bp_m)

fig, ax = plt.subplots(figsize=(4.6, 3.7))
for eta, c in zip(eta_grid, colors):
    bp_delta, bp_m = true_breakpoints(eta, DELTA_MIN, DELTA_MAX)
    ax.step(bp_delta, bp_m, where="post", linewidth=1.3, color=c)
    ax.text(DELTA_MAX + 0.015, bp_m[-1], r"$\eta=%.2f$" % eta, va="center", fontsize=8.5, color=c)

ax.set_xlabel(r"$\delta$ (failure probability)")
ax.set_ylabel(r"$M_{\min}$")
ax.set_yscale("log")
ax.set_xlim(0.0, 0.62)
fig.tight_layout()
fig.savefig("results/figures/fig5_mmin_curve.pdf")
fig.savefig("results/figures/fig5_mmin_curve.png")
print("Saved fig5")
