import sys, bisect
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

def threshold_from_sorted(sorted_buf, eta):
    M = len(sorted_buf)
    level = min(np.ceil((M+1)*(1-eta))/M, 1.0)
    idx = min(max(int(np.ceil(level*M))-1, 0), M-1)
    return sorted_buf[idx]

def run_trial(M, n_steps, rng):
    fifo = list(rng.normal(0, 1, M))
    sorted_buf = sorted(fifo)
    fpr_window = []
    for t in range(n_steps):
        s = float(rng.normal(0, 1))
        tau = threshold_from_sorted(sorted_buf, eta)
        flagged = s > tau
        fpr_window.append(1 if flagged else 0)
        if not flagged:
            oldest = fifo.pop(0)
            idx = bisect.bisect_left(sorted_buf, oldest)
            sorted_buf.pop(idx)
            bisect.insort(sorted_buf, s)
            fifo.append(s)
    return np.array(fpr_window)

def run_delayed_trial(M, q, d, n_steps, rng):
    fifo = list(rng.normal(0, 1, M))
    sorted_buf = sorted(fifo)
    pending = {}
    fpr_trace = []
    for t in range(n_steps):
        tau = threshold_from_sorted(sorted_buf, eta)
        s = float(rng.normal(0, 1))
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        if not flagged:
            oldest = fifo.pop(0)
            idx = bisect.bisect_left(sorted_buf, oldest)
            sorted_buf.pop(idx)
            bisect.insort(sorted_buf, s)
            fifo.append(s)
        else:
            if rng.uniform() < q:
                pending.setdefault(t + d, []).append(s)
        if t in pending:
            for val in pending.pop(t):
                oldest = fifo.pop(0)
                idx = bisect.bisect_left(sorted_buf, oldest)
                sorted_buf.pop(idx)
                bisect.insort(sorted_buf, val)
                fifo.append(val)
    return np.array(fpr_trace)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))

# ---- Panel (a): finite-buffer convergence ----
ax = axes[0]
M = 300
N_TRIALS = 30
turnovers_list = [30, 50, 80, 120, 150, 250, 400, 600, 800]
means, sds = [], []
for turnovers in turnovers_list:
    n_steps = turnovers * M
    eval_window = min(2000, n_steps // 4)
    trial_fprs = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(hash((turnovers, trial, "combined_a")) % (2**32))
        trace = run_trial(M, n_steps, rng)
        trial_fprs.append(trace[-eval_window:].mean())
    means.append(np.mean(trial_fprs))
    sds.append(np.std(trial_fprs))
means = np.array(means); sds = np.array(sds)
ax.plot(turnovers_list, means, color=P["teal"], marker="o", markersize=4,
        markerfacecolor="white", markeredgewidth=1.2, linewidth=1.6)
ax.fill_between(turnovers_list, means - sds, means + sds, color=P["teal"], alpha=0.18)
ax.axhline(1.0, color=P["accent"], linestyle="--", linewidth=1.0, alpha=0.85)
ax.text(turnovers_list[-1], 1.005, "theoretical limit", ha="right", va="bottom",
        fontsize=7.5, color=P["accent"])
ax.set_xscale("log")
ax.set_xlabel("buffer turnovers (log scale)")
ax.set_ylabel(r"empirical FPR ($q{=}0$, $M{=}300$)")
ax.set_ylim(0.70, 1.04)
ax.set_title("(a) finite-buffer convergence", fontsize=10, loc="left")

# ---- Panel (b): delayed feedback ----
ax = axes[1]
M2, q2, d2 = 175, 0.5, 100
TURNOVERS = 150
n_steps = TURNOVERS * M2
n_checkpoints = 20
checkpoint_size = n_steps // n_checkpoints
theory_fpr = eta / (eta + (1 - eta) * q2)

all_traces = []
for trial in range(N_TRIALS):
    rng = np.random.default_rng(hash((M2, q2, d2, "combined_b", trial)) % (2**32))
    all_traces.append(run_delayed_trial(M2, q2, d2, n_steps, rng))
all_traces = np.array(all_traces)

means_b, stds_b, xs_b = [], [], []
for c in range(n_checkpoints):
    chunk = all_traces[:, c*checkpoint_size:(c+1)*checkpoint_size]
    means_b.append(chunk.mean())
    stds_b.append(chunk.mean(axis=1).std())
    xs_b.append((c + 1) * checkpoint_size)
means_b = np.array(means_b); stds_b = np.array(stds_b); xs_b = np.array(xs_b)

ax.fill_between(xs_b, means_b - stds_b, means_b + stds_b, color=P["teal"], alpha=0.18)
ax.plot(xs_b, means_b, color=P["teal"], linewidth=1.5)
ax.axhline(theory_fpr, color=P["accent"], linestyle="--", linewidth=1.0,
           label=r"theory $\mathrm{FPR}^*(0.5){=}%.3f$" % theory_fpr)
ax.set_xlabel("evaluation step")
ax.set_ylabel("checkpoint-averaged FPR")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("(b) delayed feedback ($d{=}100$)", fontsize=10, loc="left")

fig.tight_layout()
fig.savefig("results/figures/fig3_combined_robustness.pdf")
fig.savefig("results/figures/fig3_combined_robustness.png")
print("Saved fig3_combined_robustness (2-panel)")
