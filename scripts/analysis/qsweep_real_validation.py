import bisect
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from seeding import stable_seed

ETA = 0.05
M_FIXED = 175
N_SEEDS = 100
METHOD = "ICL"
Q_GRID = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]

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

CACHE_DIR = Path("results/score_cache/smd")

print(f"Real-data validation of FPR*(q), SMD, detector={METHOD}, N_SEEDS={N_SEEDS}")
print(f"{'q':>6}  {'theory FPR*(q)':>15}  {'observed mean FPR':>18}  {'observed median':>16}  {'n_machines':>10}")

results = {}
for q in Q_GRID:
    theory_fpr = ETA / (ETA + (1 - ETA) * q) if (ETA + (1 - ETA) * q) > 0 else 1.0
    per_machine_fpr = []
    for p in sorted(CACHE_DIR.glob(f"*__{METHOD}.npz")):
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
        seed_fprs = []
        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(stable_seed("qsweep", METHOD, unit_id, q, seed_idx))
            fpr_trace = replay_real_stream(cal_used, ev, y, q, ETA, rng)
            tail_n = min(500, normal_mask.sum())
            normal_idx = np.where(normal_mask)[0]
            tail_idx = normal_idx[-tail_n:]
            seed_fprs.append(fpr_trace[tail_idx].mean())
        per_machine_fpr.append(np.mean(seed_fprs))
    per_machine_fpr = np.array(per_machine_fpr)
    results[q] = dict(
        theory=float(theory_fpr),
        observed_mean=float(per_machine_fpr.mean()),
        observed_median=float(np.median(per_machine_fpr)),
        observed_sd=float(per_machine_fpr.std()),
        n=len(per_machine_fpr),
    )
    print(f"{q:>6.2f}  {theory_fpr:>15.4f}  {per_machine_fpr.mean():>18.4f}  "
          f"{np.median(per_machine_fpr):>16.4f}  {len(per_machine_fpr):>10d}")

with open("results/phase3b/QSWEEP_REAL_VALIDATION.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/phase3b/QSWEEP_REAL_VALIDATION.json")
