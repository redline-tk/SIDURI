import sys
import json
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
    if len(normals) == 0 or len(cal) < 10:
        return None
    cal_used = cal[:M_FIXED] if len(cal) > M_FIXED else cal
    n_cal = len(cal_used)
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
                n_cal=n_cal, delta_hat_tv=dh["tv"], fpr=fpr, excess=excess)


def rank_residualize(x, control):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    rc = np.argsort(np.argsort(control)).astype(np.float64)
    A = np.vstack([rc, np.ones_like(rc)]).T
    coef, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    resid = rx - A @ coef
    return resid


def main():
    rows = []
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        r = process_unit(npz_path)
        if r is not None:
            rows.append(r)

    n_cal = np.array([r["n_cal"] for r in rows])
    dh = np.array([r["delta_hat_tv"] for r in rows])
    excess = np.array([r["excess"] for r in rows])

    out = {}
    out["n_units"] = len(rows)

    rho1, p1 = spearmanr(dh, n_cal)
    out["spearman_delta_vs_ncal"] = dict(rho=float(rho1), p=float(p1))

    rho2, p2 = spearmanr(excess, n_cal)
    out["spearman_excess_vs_ncal"] = dict(rho=float(rho2), p=float(p2))

    resid_dh = rank_residualize(dh, n_cal)
    resid_excess = rank_residualize(excess, n_cal)
    rho3, p3 = spearmanr(resid_dh, resid_excess)
    out["partial_spearman_delta_excess_given_ncal"] = dict(rho=float(rho3), p=float(p3))

    by_method_ncal = defaultdict(list)
    for r in rows:
        by_method_ncal[r["method"]].append(r["n_cal"])
    out["ncal_spread_by_method"] = {}
    for method, vals in sorted(by_method_ncal.items()):
        v = np.array(vals)
        out["ncal_spread_by_method"][method] = dict(
            n=len(v), min=int(v.min()), max=int(v.max()),
            iqr=float(np.percentile(v, 75) - np.percentile(v, 25)),
            range=int(v.max() - v.min()))

    strata = defaultdict(list)
    for r in rows:
        b = int(round(r["n_cal"] / 25.0)) * 25
        strata[b].append(r)
    out["within_stratum"] = {}
    for b, sub in sorted(strata.items()):
        if len(sub) < 40:
            continue
        dh_s = np.array([r["delta_hat_tv"] for r in sub])
        ex_s = np.array([r["excess"] for r in sub])
        rho_s, p_s = spearmanr(dh_s, ex_s)
        out["within_stratum"][str(b)] = dict(n=len(sub), rho=float(rho_s), p=float(p_s))

    print("3B.0 CONFOUND VERIFICATION")
    print("n_units=%d" % out["n_units"])
    print("\nSpearman(Delta_hat, n_cal): rho=%.4f p=%.6f  %s" % (
        rho1, p1, "STRONGLY NEGATIVE (expected)" if rho1 < -0.40 else "NOT STRONGLY NEGATIVE - diagnosis may be wrong"))
    print("Spearman(excess, n_cal): rho=%.4f p=%.6f  %s" % (
        rho2, p2, "POSITIVE (expected)" if rho2 > 0.15 else "NOT POSITIVE - diagnosis may be wrong"))
    print("Partial Spearman(Delta_hat, excess | n_cal): rho=%.4f p=%.6f" % (rho3, p3))
    print("\nn_cal spread by method:")
    for method, s in sorted(out["ncal_spread_by_method"].items()):
        print("  %-10s n=%-4d min=%-4d max=%-4d IQR=%.1f range=%d" % (
            method, s["n"], s["min"], s["max"], s["iqr"], s["range"]))
    widest = max(out["ncal_spread_by_method"].items(), key=lambda kv: kv[1]["range"])
    print("Widest n_cal range: %s  %s" % (widest[0], "MATCHES RDP prediction" if widest[0] == "RDP" else "DOES NOT MATCH RDP prediction"))
    print("\nWithin-stratum Spearman(Delta_hat, excess):")
    for b, s in sorted(out["within_stratum"].items(), key=lambda kv: int(kv[0])):
        print("  n_cal~%s  n=%-4d rho=%.4f p=%.6f" % (b, s["n"], s["rho"], s["p"]))

    all_four = (rho1 < -0.40) and (rho2 > 0.15) and (abs(rho3) < abs(rho1) * 0.5 or p3 > 0.01)
    print("\nVERDICT: %s" % ("CONFOUND CONFIRMED, H1 void, proceed to 3B.1" if all_four else "DIAGNOSIS UNCERTAIN - STOP, do not proceed to 3B.2/3/4 automatically"))

    with open("results/phase3b/PHASE3B0_FINDINGS.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
