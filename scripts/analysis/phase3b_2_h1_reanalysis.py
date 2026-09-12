import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, betabinom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
M_FIXED = 175
N_PERM = 1000
SEED = 50


def threshold_and_k(cal_scores, eta):
    M = len(cal_scores)
    level = np.ceil((M + 1) * (1.0 - eta)) / M
    level = min(level, 1.0)
    idx = int(np.ceil(level * M)) - 1
    idx = min(max(idx, 0), M - 1)
    k = M - idx
    return float(np.sort(cal_scores)[idx]), k


def pit_u(fp, n_eval_normal, k, M, rng):
    a, b = k, M + 1 - k
    v = rng.uniform(0, 1)
    f_x = betabinom.cdf(fp, n_eval_normal, a, b)
    f_x_minus_1 = betabinom.cdf(fp - 1, n_eval_normal, a, b)
    return float(f_x_minus_1 + v * (f_x - f_x_minus_1))


def two_sample_tv(a, b, n_bins=20):
    pooled = np.concatenate([a, b])
    edges = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(0.5 * np.sum(np.abs(ha - hb)))


def perm_z(cal_scores, rng):
    M = len(cal_scores)
    half = M // 2
    obs = two_sample_tv(cal_scores[:half], cal_scores[half:])
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(cal_scores)
        null[i] = two_sample_tv(p[:half], p[half:])
    sd = null.std()
    if sd < 1e-12:
        return None
    return float((obs - null.mean()) / sd)


def process_unit(npz_path, rng_pit, rng_perm):
    d = np.load(npz_path)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < M_FIXED:
        return None
    cal_used = cal[-M_FIXED:]
    normals = ev[y == 0]
    if len(normals) == 0:
        return None
    tau, k = threshold_and_k(cal_used, EPS)
    fp = int(np.sum(normals > tau))
    u = pit_u(fp, len(normals), k, M_FIXED, rng_pit)
    z = perm_z(cal_used, rng_perm)
    if z is None:
        return "DEGENERATE_PERM"
    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    return dict(dataset=npz_path.parent.name, unit=unit_id, method=method, u=u, delta_z=z)


def holm(pvals_dict, alpha=0.05):
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    n = len(items)
    results = {}
    for rank, (name, p) in enumerate(items):
        adj_alpha = alpha / (n - rank)
        results[name] = dict(p=p, adj_alpha=adj_alpha, reject=p <= adj_alpha)
    return results


def main():
    rng_pit = np.random.default_rng(SEED)
    rng_perm = np.random.default_rng(SEED + 1000)
    rows = []
    n_degenerate = 0
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        r = process_unit(npz_path, rng_pit, rng_perm)
        if r is None:
            continue
        if r == "DEGENERATE_PERM":
            n_degenerate += 1
            continue
        rows.append(r)

    print("3B.2 -- H1 RE-ANALYSIS (de-confounded: PIT-u outcome, permutation-z predictor)")
    print("n=%d  DEGENERATE_PERM=%d" % (len(rows), n_degenerate))

    u = np.array([r["u"] for r in rows])
    z = np.array([r["delta_z"] for r in rows])
    rho, p = spearmanr(z, u)
    print("\nPOOLED: rho=%.4f p=%.6f  threshold: rho>0.30, p<0.01" % (rho, p))

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    n_positive = 0
    print("\nSTRATIFIED:")
    for m, sub in sorted(by_method.items()):
        if len(sub) < 10:
            print("  %-12s n=%-4d too few" % (m, len(sub)))
            continue
        z_m = np.array([r["delta_z"] for r in sub])
        u_m = np.array([r["u"] for r in sub])
        rho_m, p_m = spearmanr(z_m, u_m)
        if rho_m > 0:
            n_positive += 1
        print("  %-12s n=%-4d rho=%.4f p=%.6f" % (m, len(sub), rho_m, p_m))
    print("methods with rho>0: %d/6 (need >=3)" % n_positive)

    verdict = bool(rho > 0.30 and p < 0.01 and n_positive >= 3)
    print("\nVERDICT: %s" % ("H1 SUPPORTED on re-analysis" if verdict else "H1 REJECTED on re-analysis"))

    with open("results/phase3b/PHASE3B2_H1_reanalysis.json", "w") as f:
        json.dump(dict(n=len(rows), rho=float(rho), p=float(p), n_positive_methods=n_positive,
                        verdict=verdict, n_degenerate=n_degenerate), f, indent=2)


if __name__ == "__main__":
    main()
