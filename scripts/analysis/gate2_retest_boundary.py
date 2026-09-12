import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import pac_threshold

N_REPLICATES = 100000   # 12.5x more than the original 8000
N_EVAL = 5000
M, eps, delta = 300, 0.10, 0.01

rng = np.random.default_rng(42)  # different seed from the main gate, deliberately

cover_count = 0
trials = 0
for r in range(N_REPLICATES):
    cal = rng.normal(0, 1, M)
    ev = rng.normal(0, 1, N_EVAL)
    q_hat, k_star, feasible = pac_threshold(cal, eps, delta)
    if not feasible:
        continue
    trials += 1
    fpr = np.mean(ev > q_hat)
    if fpr <= eps:
        cover_count += 1

rate = cover_count / trials
mc_err = 1.96 * np.sqrt(rate * (1 - rate) / trials)
target = 1 - delta
ok = rate >= target - mc_err

print("M=%d eps=%.2f delta=%.2f" % (M, eps, delta))
print("trials=%d  realised=%.5f  target>=%.5f  mc_err=%.5f  threshold=%.5f" % (
    trials, rate, target, mc_err, target - mc_err))
print("PASS" if ok else "FAIL")
