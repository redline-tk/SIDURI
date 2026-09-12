import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import betabinom, kstest
import journal_style as js

js.apply()
P = js.PALETTE

CACHE_DIR = Path("results/score_cache/smd")
ETA = 0.05
M_FIXED = 175
rng = np.random.default_rng(2)

def split_conformal_threshold_and_k(cal, eta):
    M = len(cal)
    level = min(np.ceil((M + 1) * (1 - eta)) / M, 1.0)
    idx = min(max(int(np.ceil(level * M)) - 1, 0), M - 1)
    return float(np.sort(cal)[idx]), M - idx

us = []
for p in sorted(CACHE_DIR.glob("*__*.npz")):
    d = np.load(p)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    normals = ev[y == 0]
    if len(normals) == 0 or len(cal) < 2:
        continue
    cal_used = cal[-M_FIXED:] if len(cal) >= M_FIXED else cal
    M_eff = len(cal_used)
    if M_eff < 2:
        continue
    tau, k = split_conformal_threshold_and_k(cal_used, ETA)
    fp = int(np.sum(normals > tau))
    a, b = k, M_eff + 1 - k
    v = rng.uniform(0, 1)
    f_x = betabinom.cdf(fp, len(normals), a, b)
    f_x_minus_1 = betabinom.cdf(fp - 1, len(normals), a, b)
    u = f_x_minus_1 + v * (f_x - f_x_minus_1)
    us.append(u)

us = np.array(us)
stat, pval = kstest(us, "uniform")
frac_saturated = float(np.mean(us > 0.9999))

fig, ax = plt.subplots(figsize=(4.3, 4.3))
sorted_u = np.sort(us)
theoretical = np.linspace(0, 1, len(sorted_u))
ax.fill_between(theoretical, theoretical, sorted_u, color=P["primary"], alpha=0.15)
ax.plot([0, 1], [0, 1], color=P["accent"], linestyle="--", linewidth=1.1, label="ideal (Uniform)")
ax.plot(theoretical, sorted_u, color=P["primary"], linewidth=1.7)
ax.axhspan(0.9999, 1.0, color=P["accent"], alpha=0.10)
ax.set_xlabel("theoretical quantile")
ax.set_ylabel("empirical PIT quantile")
ax.legend(loc="upper left")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("results/figures/fig7_pit_qq_plot.pdf")
fig.savefig("results/figures/fig7_pit_qq_plot.png")
print("n=%d KS_stat=%.4f KS_p=%.6f frac_saturated=%.3f" % (len(us), stat, pval, frac_saturated))
print("Saved fig7")
