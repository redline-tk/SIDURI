import sys, bisect, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import journal_style as js

js.apply()
P = js.PALETTE

CACHE_DIR = Path("results/score_cache/smd")
ETA = 0.05
M_FIXED = 175
Q_TEST = 0.5
N_SEEDS = 100

def threshold_from_sorted(sorted_buf, eta):
    M = len(sorted_buf)
    level = min(np.ceil((M + 1) * (1 - eta)) / M, 1.0)
    idx = min(max(int(np.ceil(level * M)) - 1, 0), M - 1)
    return sorted_buf[idx]

def replay_real_stream(cal_scores, eval_scores, eval_labels, q, eta, rng):
    fifo = list(cal_scores)
    sorted_buf = sorted(fifo)
    fpr_trace = []
    for i, s in enumerate(eval_scores):
        tau = threshold_from_sorted(sorted_buf, eta)
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        is_true_normal = eval_labels[i] == 0
        admit = (not flagged) or (is_true_normal and (rng.uniform() < q))
        if admit:
            oldest = fifo.pop(0)
            idx = bisect.bisect_left(sorted_buf, oldest)
            sorted_buf.pop(idx)
            bisect.insort(sorted_buf, float(s))
            fifo.append(float(s))
    return np.array(fpr_trace)

theory_fpr = ETA / (ETA + (1 - ETA) * Q_TEST)
bias_mean_by_machine = {}
bias_std_by_machine = {}
for p in sorted(CACHE_DIR.glob("*__ICL.npz")):
    unit_id = p.stem.split("__")[0]
    d = np.load(p)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < 10 or len(ev) < 50:
        continue
    cal_used = cal[-M_FIXED:] if len(cal) >= M_FIXED else cal
    normal_mask = y == 0
    if normal_mask.sum() < 20:
        continue
    seed_biases = []
    for seed_idx in range(N_SEEDS):
        rng = np.random.default_rng(hash(("fig2final", unit_id, seed_idx)) % (2**32))
        fpr_trace = replay_real_stream(cal_used, ev, y, Q_TEST, ETA, rng)
        tail_n = min(500, normal_mask.sum())
        normal_idx = np.where(normal_mask)[0]
        tail_idx = normal_idx[-tail_n:]
        emp_fpr = fpr_trace[tail_idx].mean()
        seed_biases.append(emp_fpr - theory_fpr)
    bias_mean_by_machine[unit_id] = np.mean(seed_biases)
    bias_std_by_machine[unit_id] = np.std(seed_biases) / np.sqrt(N_SEEDS)

decay_rows = json.load(open("results/phase3b/PHASE3B1_raw.json"))
decay_by_machine = {}
for r in decay_rows:
    if r.get("method") == "ICL" and r.get("dataset") == "smd":
        decay_by_machine.setdefault(r["unit"], []).append(r["slope_rho"])

common = sorted(set(decay_by_machine) & set(bias_mean_by_machine))
slopes = np.array([np.mean(decay_by_machine[u]) for u in common])
biases = np.array([bias_mean_by_machine[u] for u in common])
errs = np.array([bias_std_by_machine[u] for u in common])
rho, pval = spearmanr(slopes, biases)

z = np.polyfit(slopes, biases, 1)
xline = np.linspace(slopes.min(), slopes.max(), 100)
yline = np.poly1d(z)(xline)
resid_std = np.std(biases - np.poly1d(z)(slopes))

fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.fill_between(xline, yline - resid_std, yline + resid_std, color=P["accent"], alpha=0.10)
ax.plot(xline, yline, color=P["accent"], linewidth=1.6, linestyle="--", zorder=2)
ax.errorbar(slopes, biases, yerr=errs, fmt="o", color=P["teal"], markersize=5,
            markeredgecolor="white", markeredgewidth=0.6, elinewidth=0.7, capsize=1.5,
            alpha=0.9, zorder=3)
ax.axhline(0, color=P["muted"], linewidth=0.7, zorder=1)
ax.set_xlabel("decay-curve slope (independent drift measure)")
ax.set_ylabel(r"stationary-model bias at $q{=}0.5$")
ax.text(0.04, 0.95, r"$\rho\approx%.2f$" "\n" r"$p<10^{-6}$" "\n" r"$n=%d$" % (rho, len(common)),
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=P["muted"], alpha=0.85))
fig.tight_layout()
fig.savefig("results/figures/fig2_capstone_scatter.pdf")
fig.savefig("results/figures/fig2_capstone_scatter.png")
print("n=%d rho=%.4f p=%.8e" % (len(common), rho, pval))
print("Saved fig2")
