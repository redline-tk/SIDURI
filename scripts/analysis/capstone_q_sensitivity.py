import bisect
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

CACHE_DIR = Path("results/score_cache/smd")
ETA = 0.05
M_FIXED = 175
N_SEEDS = 30

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

decay_rows = json.load(open("results/phase3b/PHASE3B1_raw.json"))
decay_by_machine = {}
for r in decay_rows:
    if r.get("method") == "ICL" and r.get("dataset") == "smd":
        decay_by_machine.setdefault(r["unit"], []).append(r["slope_rho"])

Q_GRID = [0.1, 0.25, 0.5, 0.75, 0.9]
print("Sensitivity of the capstone correlation to the choice of q")
print("(rho, p) at each q, 30-seed average per machine, n=28 SMD machines")
print()
for q in Q_GRID:
    theory_fpr = ETA / (ETA + (1 - ETA) * q)
    bias_by_machine = {}
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
            rng = np.random.default_rng(hash(("qsens", q, unit_id, seed_idx)) % (2**32))
            fpr_trace = replay(cal_used, ev, y, q, ETA, rng)
            tail_n = min(500, normal_mask.sum())
            normal_idx = np.where(normal_mask)[0]
            tail_idx = normal_idx[-tail_n:]
            emp_fpr = fpr_trace[tail_idx].mean()
            seed_biases.append(emp_fpr - theory_fpr)
        bias_by_machine[unit_id] = np.mean(seed_biases)

    common = sorted(set(decay_by_machine) & set(bias_by_machine))
    slopes = np.array([np.mean(decay_by_machine[u]) for u in common])
    biases = np.array([bias_by_machine[u] for u in common])
    rho, pval = spearmanr(slopes, biases)
    print("  q=%.2f  rho=%.4f  p=%.2e  n=%d" % (q, rho, pval, len(common)))
