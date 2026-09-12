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
DC = js.DATASET_COLORS

CACHE_ROOT = Path("results/score_cache")
ETA = 0.05
M_FIXED = 175
Q_TEST = 0.5
SUBSPLITS = {"cicids2017": 5, "unsw_nb15": 10}

def threshold_from_sorted(sorted_buf, eta):
    M = len(sorted_buf)
    level = min(np.ceil((M + 1) * (1 - eta)) / M, 1.0)
    idx = min(max(int(np.ceil(level * M)) - 1, 0), M - 1)
    return sorted_buf[idx]

def replay(cal_scores, eval_scores, eval_labels, q, eta, rng):
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
datasets = ["smd", "smap", "cicids2017", "unsw_nb15"]
labels = ["SMD", "SMAP", "CICIDS2017", "UNSW-NB15"]
data_by_ds = {}

for ds in datasets:
    cache_dir = CACHE_ROOT / ds
    biases = []
    for p in sorted(cache_dir.glob("*__ICL.npz")):
        d = np.load(p)
        cal = np.asarray(d["cal_scores"], dtype=np.float64)
        ev = np.asarray(d["eval_scores"], dtype=np.float64)
        y = np.asarray(d["eval_labels"], dtype=int)
        if len(cal) < 10 or len(ev) < 50:
            continue
        cal_used = cal[-M_FIXED:] if len(cal) >= M_FIXED else cal
        k_splits = SUBSPLITS.get(ds, 1)
        chunk_size = len(ev) // k_splits
        for k in range(k_splits):
            start = k * chunk_size
            end = len(ev) if k == k_splits - 1 else (k + 1) * chunk_size
            ev_c, y_c = ev[start:end], y[start:end]
            normal_mask = y_c == 0
            if normal_mask.sum() < 20:
                continue
            rng = np.random.default_rng(hash(("fig8v2", ds, p.stem, k)) % (2**32))
            trace = replay(cal_used, ev_c, y_c, Q_TEST, ETA, rng)
            tail_n = min(200, normal_mask.sum())
            normal_idx = np.where(normal_mask)[0]
            tail_idx = normal_idx[-tail_n:]
            biases.append(trace[tail_idx].mean() - theory_fpr)
    data_by_ds[ds] = np.array(biases)

fig, ax = plt.subplots(figsize=(5.6, 3.8))
positions = list(range(len(datasets)))

vp = ax.violinplot([data_by_ds[ds] for ds in datasets], positions=positions,
                    widths=0.7, showmedians=False, showextrema=False)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor(DC[datasets[i]])
    body.set_alpha(0.25)
    body.set_edgecolor(DC[datasets[i]])
    body.set_linewidth(1.0)

rng = np.random.default_rng(0)
for ds, pos in zip(datasets, positions):
    y = data_by_ds[ds]
    x = rng.uniform(pos - 0.09, pos + 0.09, size=len(y))
    ax.scatter(x, y, s=9, color=DC[ds], alpha=0.55, edgecolor="none", zorder=3)
    med = np.median(y)
    ax.plot([pos - 0.22, pos + 0.22], [med, med], color="black", linewidth=1.8, zorder=4)

ax.axhline(0, color=P["muted"], linewidth=0.7, linestyle="--", zorder=1)
ax.set_xticks(positions)
ax.set_xticklabels(["%s\n($n{=}%d$)" % (l, len(data_by_ds[ds])) for l, ds in zip(labels, datasets)])
ax.set_ylabel(r"stationary-model bias at $q{=}0.5$")
fig.tight_layout()
fig.savefig("results/figures/fig8_cross_dataset_bias.pdf")
fig.savefig("results/figures/fig8_cross_dataset_bias.png")
for ds in datasets:
    v = data_by_ds[ds]
    print("%-12s n=%-4d mean=%+.4f median=%+.4f" % (ds, len(v), v.mean(), np.median(v)))
print("Saved fig8")
