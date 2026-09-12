from pathlib import Path
import numpy as np
from scipy.stats import betabinom

CACHE_DIR = Path("results/score_cache/smd")
ETA = 0.05
M_FIXED = 175
rng = np.random.default_rng(2)

def split_conformal_threshold_and_k(cal, eta):
    M = len(cal)
    level = min(np.ceil((M + 1) * (1 - eta)) / M, 1.0)
    idx = min(max(int(np.ceil(level * M)) - 1, 0), M - 1)
    return float(np.sort(cal)[idx]), M - idx

n_saturated = 0
n_total = 0
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
    n_total += 1
    if fp == len(normals):
        n_saturated += 1

print("n_total=%d  n_units_with_ALL_normals_flagged=%d  (%.1f%%)" % (
    n_total, n_saturated, 100.0 * n_saturated / n_total))
