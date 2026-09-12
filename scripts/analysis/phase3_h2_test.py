import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/redline/files/TSB-AD")
from src.chad.pac.pac_threshold import pac_threshold
from src.chad.pac.feature_drift import window_feature_means, delta_hat_feat
import run_extended_methods as rem

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
DELTA = 0.10
M_FIXED = 175

_raw_cache = {}

def load_raw_for_unit(dataset, unit_id):
    key = (dataset, unit_id)
    if key in _raw_cache:
        return _raw_cache[key]
    try:
        if dataset == "smd":
            units = {u[0]: u for u in rem.load_smd_units()}
        elif dataset == "smap":
            units = {u[0]: u for u in rem.load_smap_msl_units("smap")}
        elif dataset == "msl":
            units = {u[0]: u for u in rem.load_smap_msl_units("msl")}
        elif dataset == "batadal":
            units = {u[0]: u for u in rem.load_batadal_units()}
        elif dataset == "cicids2017":
            units = {u[0]: u for u in rem.load_cicids_units()}
        elif dataset == "unsw_nb15":
            units = {u[0]: u for u in rem.load_unsw_units()}
        else:
            _raw_cache[key] = None
            return None
    except Exception as e:
        print("failed to load dataset %s: %s" % (dataset, e))
        _raw_cache[key] = None
        return None
    for uid, data in units.items():
        _raw_cache[(dataset, uid)] = data
    return _raw_cache.get(key)


def process_unit(npz_path):
    stem = npz_path.stem
    if "__" not in stem:
        return None
    unit_id, method = stem.split("__", 1)
    dataset = npz_path.parent.name

    d = np.load(npz_path)
    if "cal_window_start" not in d:
        return None
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    starts = np.asarray(d["cal_window_start"], dtype=np.int64)
    normals = ev[y == 0]
    if len(normals) == 0 or len(cal) < 10:
        return None

    raw = load_raw_for_unit(dataset, unit_id)
    if raw is None:
        return None
    _, train_raw, _, test_raw, _ = raw

    n_cal_pts = int(len(test_raw) * rem.CAL_FRAC)
    cal_raw = test_raw[:n_cal_pts]
    keep, _ = rem.feature_prescreen(train_raw.astype(np.float32))
    if keep.sum() < 2:
        return None
    cal_raw_sel = cal_raw[:, keep]

    valid_starts = starts[starts + 60 <= len(cal_raw_sel)]
    if len(valid_starts) < 10:
        return None
    fmeans = window_feature_means(cal_raw_sel, valid_starts, 60)
    pooled_std = cal_raw_sel.std(axis=0)
    dh_feat = delta_hat_feat(fmeans, pooled_std)

    cal_used = cal[:M_FIXED] if len(cal) > M_FIXED else cal
    q_hat, k_star, feasible = pac_threshold(cal_used, EPS, DELTA)
    if not feasible:
        return None
    fpr = float(np.mean(normals > q_hat))
    excess = fpr - EPS

    return dict(dataset=dataset, unit=unit_id, method=method,
                delta_hat_feat=dh_feat, fpr=fpr, excess=excess)


def main():
    rows = []
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        try:
            r = process_unit(npz_path)
        except Exception as e:
            r = None
        if r is not None:
            rows.append(r)

    print("H2 TEST: units processed = %d" % len(rows))
    if len(rows) < 10:
        print("Too few units with recoverable raw features. Aborting test.")
        return

    dh = np.array([r["delta_hat_feat"] for r in rows])
    excess = np.array([r["excess"] for r in rows])
    rho, p = spearmanr(dh, excess)
    print("\nPOOLED:")
    print("  Delta_hat_feat vs excess FPR: rho=%.4f p=%.6f  %s" % (
        rho, p, "SUPPORT" if (rho > 0.3 and p < 0.01) else "NO SUPPORT"))

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    print("\nSTRATIFIED BY METHOD:")
    for method in sorted(by_method.keys()):
        sub = by_method[method]
        if len(sub) < 10:
            print("  %-22s n=%-4d too few" % (method, len(sub)))
            continue
        dh_m = np.array([r["delta_hat_feat"] for r in sub])
        ex_m = np.array([r["excess"] for r in sub])
        rho_m, p_m = spearmanr(dh_m, ex_m)
        print("  %-22s n=%-4d rho=%.4f p=%.6f  %s" % (
            method, len(sub), rho_m, p_m, "SUPPORT" if (rho_m > 0.3 and p_m < 0.01) else "NO SUPPORT"))


if __name__ == "__main__":
    main()
