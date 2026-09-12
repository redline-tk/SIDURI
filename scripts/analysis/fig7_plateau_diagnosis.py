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

rows = []
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
    n_normal = len(normals)
    a, b = k, M_eff + 1 - k
    cdf_at_fp = betabinom.cdf(fp, n_normal, a, b)
    v = rng.uniform(0, 1)
    f_x = betabinom.cdf(fp, n_normal, a, b)
    f_x_minus_1 = betabinom.cdf(fp - 1, n_normal, a, b)
    u = f_x_minus_1 + v * (f_x - f_x_minus_1)
    fpr = fp / n_normal
    rows.append((p.stem, fp, n_normal, fpr, cdf_at_fp, u))

rows.sort(key=lambda r: -r[5])
n_total = len(rows)
n_u_above_099 = sum(1 for r in rows if r[5] > 0.99)
n_u_above_09999 = sum(1 for r in rows if r[5] > 0.9999)

print("n_total=%d" % n_total)
print("units with u>0.99: %d (%.1f%%)" % (n_u_above_099, 100.0*n_u_above_099/n_total))
print("units with u>0.9999: %d (%.1f%%)" % (n_u_above_09999, 100.0*n_u_above_09999/n_total))
print()
print("Top 10 units by u:")
for stem, fp, n_normal, fpr, cdf_at_fp, u in rows[:10]:
    print("  %-24s fp=%-4d n_normal=%-4d fpr=%.4f cdf=%.6f u=%.6f" % (
        stem, fp, n_normal, fpr, cdf_at_fp, u))
