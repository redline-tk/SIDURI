import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, betabinom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import pac_k_star

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
DELTA = 0.10
M_FIXED = 175
N_PERM = 1000
SEED_A = 21
SEED_B = 22
CONTAM_CLEAN_MAX = 0.01


def split_conformal_threshold_and_k(cal_scores, eta):
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
    return float((obs - null.mean()) / sd), obs


def process_unit(npz_path, rng_pit, rng_perm):
    d = np.load(npz_path)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < M_FIXED:
        return None
    cal_used = cal[:M_FIXED]
    normals = ev[y == 0]
    if len(normals) == 0:
        return None
    tau, k = split_conformal_threshold_and_k(cal_used, EPS)
    fp = int(np.sum(normals > tau))
    u = pit_u(fp, len(normals), k, M_FIXED, rng_pit)
    z_result = perm_z(cal_used, rng_perm)
    if z_result is None:
        return "DEGENERATE_PERM"
    z, raw_tv = z_result

    contam = None
    if "cal_labels" in d:
        cl = np.asarray(d["cal_labels"], dtype=int)
        if len(cl) > 0:
            contam = float(np.mean(cl[:M_FIXED] != 0))

    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    return dict(dataset=npz_path.parent.name, unit=unit_id, method=method,
                m_eff=M_FIXED, u=u, delta_z=z, delta_raw=raw_tv, contam=contam)


def run_seed(seed):
    rng_pit = np.random.default_rng(seed)
    rng_perm = np.random.default_rng(seed + 1000)
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
    return rows, n_degenerate


def report(rows, label):
    print("\n=== %s ===" % label)
    print("n=%d" % len(rows))
    u = np.array([r["u"] for r in rows])
    z = np.array([r["delta_z"] for r in rows])
    rho, p = spearmanr(z, u)
    print("POOLED: rho=%.4f p=%.6f" % (rho, p))

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    print("PER DATASET:")
    for ds, sub in sorted(by_ds.items()):
        if len(sub) < 20:
            print("  %-14s n=%-4d too few, descriptive only" % (ds, len(sub)))
            continue
        u_s = np.array([r["u"] for r in sub])
        z_s = np.array([r["delta_z"] for r in sub])
        rho_s, p_s = spearmanr(z_s, u_s)
        print("  %-14s n=%-4d rho=%.4f p=%.6f  mean_z=%.3f mean_u=%.3f" % (
            ds, len(sub), rho_s, p_s, z_s.mean(), u_s.mean()))

    contam_rows = [r for r in rows if r["contam"] is not None]
    clean = [r for r in contam_rows if r["contam"] <= CONTAM_CLEAN_MAX]
    contaminated = [r for r in contam_rows if r["contam"] > CONTAM_CLEAN_MAX]
    print("CONTAMINATION STRATIFICATION (clean = cal contam <= %.2f):" % CONTAM_CLEAN_MAX)
    for name, sub in [("clean", clean), ("contaminated", contaminated)]:
        if len(sub) < 20:
            print("  %-14s n=%-4d too few" % (name, len(sub)))
            continue
        u_s = np.array([r["u"] for r in sub])
        z_s = np.array([r["delta_z"] for r in sub])
        rho_s, p_s = spearmanr(z_s, u_s)
        print("  %-14s n=%-4d rho=%.4f p=%.6f" % (name, len(sub), rho_s, p_s))

    largest_ds = max(by_ds.items(), key=lambda kv: len(kv[1]))[0]
    print("PER METHOD within largest dataset (%s):" % largest_ds)
    by_method = defaultdict(list)
    for r in by_ds[largest_ds]:
        by_method[r["method"]].append(r)
    for m, sub in sorted(by_method.items()):
        if len(sub) < 10:
            print("  %-14s n=%-4d too few" % (m, len(sub)))
            continue
        u_s = np.array([r["u"] for r in sub])
        z_s = np.array([r["delta_z"] for r in sub])
        rho_s, p_s = spearmanr(z_s, u_s)
        print("  %-14s n=%-4d rho=%.4f p=%.6f" % (m, len(sub), rho_s, p_s))

    return rho, p


def main():
    rows_a, n_deg_a = run_seed(SEED_A)
    rows_b, n_deg_b = run_seed(SEED_B)
    print("DEGENERATE_PERM: seed_a=%d seed_b=%d (of stratum candidates)" % (n_deg_a, n_deg_b))

    rho_a, p_a = report(rows_a, "SEED A (%d)" % SEED_A)
    rho_b, p_b = report(rows_b, "SEED B (%d)" % SEED_B)

    print("\n=== STABILITY ACROSS SEEDS ===")
    print("seed_a: rho=%.4f p=%.6f" % (rho_a, p_a))
    print("seed_b: rho=%.4f p=%.6f" % (rho_b, p_b))
    stable = (np.sign(rho_a) == np.sign(rho_b)) and (abs(rho_a - rho_b) < 0.1)
    print("STABLE" if stable else "SEED_UNSTABLE - default to R1 per guard")

    with open("results/phase3b/PHASE3B0b_seedA.json", "w") as f:
        json.dump(rows_a, f, indent=2)
    with open("results/phase3b/PHASE3B0b_seedB.json", "w") as f:
        json.dump(rows_b, f, indent=2)


if __name__ == "__main__":
    main()
