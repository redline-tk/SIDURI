import bisect
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr
from seeding import stable_seed

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

# ---------- PART 1: capstone correlation, per detector, SMD ----------
print("=" * 70)
print("PART 1: SMD capstone correlation, all cached detectors")
print("=" * 70)

CACHE_DIR = Path("results/score_cache/smd")
decay_rows = json.load(open("results/phase3b/PHASE3B1_raw.json"))

methods = sorted(set(p.stem.split("__", 1)[1] for p in CACHE_DIR.glob("*__*.npz")))
print("methods found:", methods)
print()

capstone_results = {}
for method in methods:
    t0 = time.time()
    avg_bias_by_machine = {}
    for p in sorted(CACHE_DIR.glob(f"*__{method}.npz")):
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
            rng = np.random.default_rng(stable_seed("capstone_all", method, unit_id, seed_idx))
            fpr_trace = replay_real_stream(cal_used, ev, y, Q_TEST, ETA, rng)
            tail_n = min(500, normal_mask.sum())
            normal_idx = np.where(normal_mask)[0]
            tail_idx = normal_idx[-tail_n:]
            emp_fpr = fpr_trace[tail_idx].mean()
            seed_biases.append(emp_fpr - theory_fpr)
        avg_bias_by_machine[unit_id] = np.mean(seed_biases)

    decay_by_machine = {}
    for r in decay_rows:
        if r.get("method") == method and r.get("dataset") == "smd":
            decay_by_machine.setdefault(r["unit"], []).append(r["slope_rho"])

    common = sorted(set(decay_by_machine) & set(avg_bias_by_machine))
    if len(common) < 3:
        print(f"  {method}: insufficient overlap (n={len(common)}), skipping")
        continue
    slopes = np.array([np.mean(decay_by_machine[u]) for u in common])
    biases = np.array([avg_bias_by_machine[u] for u in common])
    rho, pval = spearmanr(slopes, biases)
    capstone_results[method] = dict(n=len(common), rho=rho, pval=pval)
    print(f"  {method:<20} n={len(common):<4} rho={rho:.4f}  p={pval:.2e}  ({time.time()-t0:.1f}s)")

print()
print("SUMMARY (Part 1):")
rhos = [v["rho"] for v in capstone_results.values()]
print(f"  rho range across {len(capstone_results)} detectors: [{min(rhos):.4f}, {max(rhos):.4f}]")
print(f"  all significant (p<1e-4): {all(v['pval']<1e-4 for v in capstone_results.values())}")
with open("results/phase3b/CAPSTONE_ALL_DETECTORS.json", "w") as f:
    json.dump(capstone_results, f, indent=2)

# ---------- PART 2: cross-dataset bias, per detector, all datasets ----------
print()
print("=" * 70)
print("PART 2: Cross-dataset bias, all cached detectors")
print("=" * 70)

CACHE_ROOT = Path("results/score_cache")
SUBSPLITS = {"cicids2017": 5, "unsw_nb15": 10}
datasets = ["smd", "smap", "cicids2017", "unsw_nb15"]

cross_results = {}
for ds in datasets:
    cache_dir = CACHE_ROOT / ds
    ds_methods = sorted(set(p.stem.split("__", 1)[1] for p in cache_dir.glob("*__*.npz")))
    print(f"\n--- {ds} (methods: {ds_methods}) ---")
    cross_results[ds] = {}
    for method in ds_methods:
        biases = []
        for p in sorted(cache_dir.glob(f"*__{method}.npz")):
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
                tail_n = min(500, normal_mask.sum())
                normal_idx = np.where(normal_mask)[0]
                tail_idx = normal_idx[-tail_n:]
                seed_biases = []
                for seed_idx in range(N_SEEDS):
                    rng = np.random.default_rng(
                        stable_seed("cross_all", ds, method, p.stem, k, seed_idx))
                    trace = replay_real_stream(cal_used, ev_c, y_c, Q_TEST, ETA, rng)
                    seed_biases.append(trace[tail_idx].mean() - theory_fpr)
                biases.append(np.mean(seed_biases))
        if len(biases) == 0:
            continue
        biases = np.array(biases)
        cross_results[ds][method] = dict(n=len(biases), mean=float(biases.mean()), median=float(np.median(biases)))
        print(f"  {method:<20} n={len(biases):<4} mean={biases.mean():+.4f}  median={np.median(biases):+.4f}")

print()
print("SUMMARY (Part 2): mean-bias range across detectors, per dataset")
for ds, methods_d in cross_results.items():
    means = [v["mean"] for v in methods_d.values()]
    if means:
        print(f"  {ds:<12} range=[{min(means):+.4f}, {max(means):+.4f}]  n_detectors={len(means)}")

with open("results/phase3b/CROSS_DATASET_ALL_DETECTORS.json", "w") as f:
    json.dump(cross_results, f, indent=2)

print()
print("Done. Saved CAPSTONE_ALL_DETECTORS.json and CROSS_DATASET_ALL_DETECTORS.json")
