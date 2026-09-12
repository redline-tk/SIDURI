import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, betabinom
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import pac_k_star
from src.chad.pac.feature_drift import delta_hat_disp

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
M_FIXED = 175
SEED = 40
MIN_ANCAL = 45


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


def process_unit(npz_path, rng):
    d = np.load(npz_path)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < MIN_ANCAL:
        return None
    normals = ev[y == 0]
    if len(normals) < 200:
        return None
    if len(np.unique(y)) > 1:
        try:
            auroc = roc_auc_score(y, ev)
        except Exception:
            return None
    else:
        return None
    if auroc < 0.70:
        return None
    if y.mean() > 0.25:
        return None

    cal_used = cal[-M_FIXED:] if len(cal) >= M_FIXED else cal
    tau, k = threshold_and_k(cal_used, EPS)
    fp = int(np.sum(normals > tau))
    u = pit_u(fp, len(normals), k, len(cal_used), rng)

    dh = delta_hat_disp(cal_used, ev)
    if dh is None:
        return None

    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    return dict(unit=unit_id, method=method, u=u, delta_hat_disp=dh, auroc=float(auroc))


def holm_correct(pvals, alpha=0.05):
    order = np.argsort(pvals)
    n = len(pvals)
    for rank, idx in enumerate(order):
        adj_alpha = alpha / (n - rank)
        if pvals[idx] > adj_alpha:
            return idx, adj_alpha, rank
    return None, alpha / n, n


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for npz_path in sorted((CACHE_DIR / "smd").glob("*__*.npz")):
        r = process_unit(npz_path, rng)
        if r is not None:
            rows.append(r)

    print("H3 TEST (SMD only, power-corrected pre-registration)")
    print("n=%d" % len(rows))
    if len(rows) < 10:
        print("Too few units. Aborting.")
        return

    dh = np.array([r["delta_hat_disp"] for r in rows])
    u = np.array([r["u"] for r in rows])
    rho, p = spearmanr(dh, u)
    print("\nPOOLED: rho=%.4f p=%.6f  threshold: rho>0.45, p<0.01" % (rho, p))
    print("SUPPORT" if (rho > 0.45 and p < 0.01) else "NO SUPPORT")

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    n_positive_methods = 0
    print("\nSTRATIFIED:")
    for m, sub in sorted(by_method.items()):
        if len(sub) < 10:
            print("  %-12s n=%-4d too few" % (m, len(sub)))
            continue
        dh_m = np.array([r["delta_hat_disp"] for r in sub])
        u_m = np.array([r["u"] for r in sub])
        rho_m, p_m = spearmanr(dh_m, u_m)
        if rho_m > 0:
            n_positive_methods += 1
        print("  %-12s n=%-4d rho=%.4f p=%.6f" % (m, len(sub), rho_m, p_m))
    print("methods with rho>0: %d/6 (need >=4)" % n_positive_methods)

    named = ["machine-3-1", "machine-3-9", "machine-3-10"]
    icl_rows = [r for r in rows if r["method"] == "ICL"]
    icl_sorted = sorted(icl_rows, key=lambda r: r["delta_hat_disp"], reverse=True)
    decile_cut = max(1, len(icl_sorted) // 10)
    top_decile_units = set(r["unit"] for r in icl_sorted[:decile_cut])
    print("\nNamed prediction check (ICL top decile, n_in_decile=%d):" % decile_cut)
    for u_name in named:
        print("  %s: %s" % (u_name, "IN top decile" if u_name in top_decile_units else "not in top decile"))

    final_verdict = bool(rho > 0.45 and p < 0.01 and n_positive_methods >= 4)
    print("\nFINAL VERDICT: %s" % ("H3 SUPPORTED" if final_verdict else "H3 REJECTED"))

    with open("results/phase3b/PHASE3B3_H3_result.json", "w") as f:
        json.dump(dict(n=len(rows), rho=float(rho), p=float(p),
                        n_positive_methods=n_positive_methods, verdict=final_verdict), f, indent=2)


if __name__ == "__main__":
    main()
