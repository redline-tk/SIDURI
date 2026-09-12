import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import betabinom, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
M_FIXED = 175
SEED = 30


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


def icc_1_1(values_by_unit):
    units = [u for u in values_by_unit if len(values_by_unit[u]) >= 2]
    if len(units) < 3:
        return None, 0
    k_vals = [len(values_by_unit[u]) for u in units]
    k_bar = np.mean(k_vals)
    n = len(units)
    grand_mean = np.mean([v for u in units for v in values_by_unit[u]])
    ms_between = sum(len(values_by_unit[u]) * (np.mean(values_by_unit[u]) - grand_mean) ** 2 for u in units) / (n - 1)
    ms_within = sum(sum((v - np.mean(values_by_unit[u])) ** 2 for v in values_by_unit[u]) for u in units)
    df_within = sum(len(values_by_unit[u]) for u in units) - n
    if df_within <= 0:
        return None, n
    ms_within = ms_within / df_within
    icc = (ms_between - ms_within) / (ms_between + (k_bar - 1) * ms_within)
    return float(icc), n


def main():
    rng = np.random.default_rng(SEED)
    u_by_unit = defaultdict(dict)
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        d = np.load(npz_path)
        cal = np.asarray(d["cal_scores"], dtype=np.float64)
        ev = np.asarray(d["eval_scores"], dtype=np.float64)
        y = np.asarray(d["eval_labels"], dtype=int)
        if len(cal) < 10:
            continue
        cal_used = cal[-M_FIXED:] if len(cal) >= M_FIXED else cal
        normals = ev[y == 0]
        if len(normals) == 0:
            continue
        m_eff = len(cal_used)
        tau, k = threshold_and_k(cal_used, EPS)
        fp = int(np.sum(normals > tau))
        u = pit_u(fp, len(normals), k, m_eff, rng)
        stem = npz_path.stem
        unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
        dataset = npz_path.parent.name
        key = (dataset, unit_id)
        u_by_unit[key][method] = u

    by_dataset = defaultdict(dict)
    for (dataset, unit_id), method_vals in u_by_unit.items():
        if len(method_vals) >= 3:
            by_dataset[dataset][unit_id] = list(method_vals.values())

    print("CROSS-DETECTOR CONCORDANCE (ICC(1,1) on PIT u-values)")
    for dataset in sorted(by_dataset):
        units = by_dataset[dataset]
        if len(units) < 10:
            print("  %-14s n_units=%-4d too few for reliable ICC" % (dataset, len(units)))
            continue
        icc, n_used = icc_1_1(units)
        print("  %-14s n_units=%-4d ICC=%s" % (dataset, n_used, "%.4f" % icc if icc is not None else "N/A"))


    all_units = defaultdict(dict)
    for (dataset, unit_id), method_vals in u_by_unit.items():
        if dataset != "smd":
            continue
        if len(method_vals) >= 3:
            all_units[unit_id] = method_vals

    print("\nSMD DETAIL: n_machines=%d" % len(all_units))
    flat = defaultdict(list)
    for unit_id, method_vals in all_units.items():
        for method, u in method_vals.items():
            flat[unit_id].append(u)
    icc, n_used = icc_1_1(flat)
    print("SMD ICC(1,1) = %s  (n=%d machines)" % ("%.4f" % icc if icc is not None else "N/A", n_used))

    with open("results/phase3b/PHASE3B1c_icc.json", "w") as f:
        json.dump(dict(icc_by_dataset={ds: icc_1_1(by_dataset[ds])[0] for ds in by_dataset if len(by_dataset[ds]) >= 10},
                        smd_icc=icc, smd_n=n_used), f, indent=2)


if __name__ == "__main__":
    main()
