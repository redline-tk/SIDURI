import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import journal_style as js

js.apply()
P = js.PALETTE

eta = 0.05
q = np.linspace(0.001, 1.0, 500)
fpr_star = eta / (eta + (1 - eta) * q)

fig, ax = plt.subplots(figsize=(4.4, 3.5))
ax.fill_between(q, eta, fpr_star, color=P["primary"], alpha=0.15)
ax.plot(q, fpr_star, color=P["primary"], linewidth=2.2, label=r"$\mathrm{FPR}^*(q)$")
ax.axhline(eta, color=P["muted"], linestyle=":", linewidth=1.2, label=r"nominal $\eta=0.05$")
ax.axhline(1.0, color=P["accent"], linestyle="--", linewidth=1.2, alpha=0.85,
           label=r"runaway limit ($q{=}0$)")
ax.scatter([1.0], [eta], color=P["primary"], zorder=5, s=32, edgecolor="white", linewidth=0.8)
ax.set_xlabel(r"feedback fraction $q$")
ax.set_ylabel(r"stationary $\mathrm{FPR}^*(q)$")
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.88))
fig.tight_layout()
fig.savefig("results/figures/fig1_fpr_star_curve.pdf")
fig.savefig("results/figures/fig1_fpr_star_curve.png")
print("Saved fig1")
