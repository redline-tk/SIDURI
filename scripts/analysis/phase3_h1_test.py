import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import pac_threshold
from src.chad.pac.weighted_conformal import delta_hat

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
DELTA = 0.10
M_FIXED = 175


def process_unit(npz_path):
    d = np.load(npz_path)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    normals = ev[y == 0]
    if len(normals) == 0:
        return None
    cal_used = cal[:M_FIXED] if len(cal) > M_FIXED else cal
    if len(cal_used) < 10:
        return None
    q_hat, k_star, feasible = pac_threshold(cal_used, EPS, DELTA)
    if not feasible:
        return None
    fpr = float(np.mean(normals > q_hat))
    excess = fpr - EPS
    dh = delta_hat(cal_used)
    stem = npz_path.stem
    if "__" in stem:
        unit_id, method = stem.split("__", 1)
    else:
        unit_id, method = stem, "unknown"
    return dict(dataset=npz_path.parent.name, unit=unit_id, method=method,
                delta_hat_tv=dh["tv"], delta_hat_ks=dh["ks_stat"],
                fpr=fpr, excess=excess)


def main():
    rows = []
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        r = process_unit(npz_path)
        if r is not None:
            rows.append(r)

    print("H1 TEST: units processed = %d" % len(rows))
    tv = np.array([r["delta_hat_tv"] for r in rows])
    ks = np.array([r["delta_hat_ks"] for r in rows])
    excess = np.array([r["excess"] for r in rows])

    rho_tv, p_tv = spearmanr(tv, excess)
    rho_ks, p_ks = spearmanr(ks, excess)
    print("\nPOOLED (all datasets, all methods):")
    print("  Delta_hat(TV) vs excess FPR: rho=%.4f p=%.6f  %s" % (
        rho_tv, p_tv, "SUPPORT" if (rho_tv > 0.3 and p_tv < 0.01) else "NO SUPPORT"))
    print("  Delta_hat(KS) vs excess FPR: rho=%.4f p=%.6f  %s" % (
        rho_ks, p_ks, "SUPPORT" if (rho_ks > 0.3 and p_ks < 0.01) else "NO SUPPORT"))

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    print("\nSTRATIFIED BY METHOD:")
    for method in sorted(by_method.keys()):
        sub = by_method[method]
        if len(sub) < 10:
            print("  %-22s n=%-4d too few for reliable correlation" % (method, len(sub)))
            continue
        tv_m = np.array([r["delta_hat_tv"] for r in sub])
        ex_m = np.array([r["excess"] for r in sub])
        rho_m, p_m = spearmanr(tv_m, ex_m)
        print("  %-22s n=%-4d rho=%.4f p=%.6f  %s" % (
            method, len(sub), rho_m, p_m, "SUPPORT" if (rho_m > 0.3 and p_m < 0.01) else "NO SUPPORT"))

    print("\nPRE-REGISTERED CHECK UNITS (from PHASE3_H1_PREREGISTRATION.md):")
    named = ["machine-3-1__ICL", "machine-3-9__ICL", "machine-3-10__ICL"]
    for r in rows:
        stem = "%s__%s" % (r["unit"], r["method"])
        if stem in named:
            print("  %-24s delta_hat_tv=%.4f delta_hat_ks=%.4f fpr=%.4f excess=%.4f" % (
                stem, r["delta_hat_tv"], r["delta_hat_ks"], r["fpr"], r["excess"]))

    all_tv = sorted(rows, key=lambda r: r["delta_hat_tv"], reverse=True)[:5]
    print("\nTop 5 by Delta_hat(TV), for cross-check against top FPR violators:")
    for r in all_tv:
        print("  %-24s delta_hat_tv=%.4f fpr=%.4f excess=%.4f" % (
            "%s__%s" % (r["unit"], r["method"]), r["delta_hat_tv"], r["fpr"], r["excess"]))


if __name__ == "__main__":
    main()
