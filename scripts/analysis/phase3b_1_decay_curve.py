import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import pac_threshold

CACHE_DIR = Path("results/score_cache")
EPS = 0.05
DELTA = 0.10
M_FIXED = 175
MIN_PER_DECILE = 30
D_FULL = 10
D_FALLBACK = 5


def fixed_threshold(cal_scores):
    M = len(cal_scores)
    level = np.ceil((M + 1) * (1.0 - EPS)) / M
    level = min(level, 1.0)
    idx = int(np.ceil(level * M)) - 1
    idx = min(max(idx, 0), M - 1)
    return float(np.sort(cal_scores)[idx])


def decile_fpr(scores, starts, tau, n_bins):
    order = np.argsort(starts)
    s = scores[order]
    st = starts[order]
    n = len(s)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    fprs = []
    counts = []
    mean_starts = []
    for i in range(n_bins):
        chunk = s[edges[i]:edges[i + 1]]
        chunk_starts = st[edges[i]:edges[i + 1]]
        if len(chunk) == 0:
            fprs.append(np.nan)
            counts.append(0)
            mean_starts.append(np.nan)
            continue
        fprs.append(float(np.mean(chunk > tau)))
        counts.append(len(chunk))
        mean_starts.append(float(chunk_starts.mean()))
    return np.array(fprs), np.array(counts), np.array(mean_starts)


def process_unit(npz_path):
    d = np.load(npz_path)
    if "eval_window_start" not in d:
        return None
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    starts = np.asarray(d["eval_window_start"], dtype=np.int64)
    if len(cal) < 10:
        return None
    cal_used = cal[:M_FIXED] if len(cal) > M_FIXED else cal
    tau = fixed_threshold(cal_used)

    normal_mask = y == 0
    ev_n = ev[normal_mask]
    st_n = starts[normal_mask]
    if len(ev_n) < MIN_PER_DECILE * D_FALLBACK:
        return None

    n_bins = D_FULL
    fprs, counts, mean_starts = decile_fpr(ev_n, st_n, tau, n_bins)
    if np.any(counts[~np.isnan(fprs)] < MIN_PER_DECILE):
        n_bins = D_FALLBACK
        fprs, counts, mean_starts = decile_fpr(ev_n, st_n, tau, n_bins)
        if np.any(counts < MIN_PER_DECILE):
            return None

    valid = ~np.isnan(fprs)
    if valid.sum() < 3:
        return None
    fprs_v = fprs[valid]
    if np.all(fprs_v == 0) or np.all(fprs_v == 1):
        return None

    if counts[valid][-1] < 0.5 * counts[valid][0]:
        clustering_flag = True
    else:
        clustering_flag = False

    d_idx = np.arange(1, n_bins + 1)[valid]
    slope_rho, slope_p = spearmanr(d_idx, fprs_v)

    warmup_flag = None
    if "train_scores" in d:
        tr = np.asarray(d["train_scores"], dtype=np.float64)
        if len(tr) >= MIN_PER_DECILE * D_FALLBACK:
            tr_starts = np.arange(len(tr))
            tr_fprs, tr_counts, _ = decile_fpr(tr, tr_starts, tau, n_bins)
            tr_valid = ~np.isnan(tr_fprs) & (tr_counts >= MIN_PER_DECILE)
            if tr_valid.sum() >= 3:
                tr_d = np.arange(1, n_bins + 1)[tr_valid]
                warmup_rho, warmup_p = spearmanr(tr_d, tr_fprs[tr_valid])
                warmup_flag = dict(rho=float(warmup_rho), p=float(warmup_p))

    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    return dict(dataset=npz_path.parent.name, unit=unit_id, method=method,
                n_bins=n_bins, slope_rho=float(slope_rho), slope_p=float(slope_p),
                fprs=fprs_v.tolist(), clustering_flag=clustering_flag,
                warmup_control=warmup_flag, overall_fpr=float(np.mean(fprs_v)))


def main():
    rows = []
    n_excluded_thin = 0
    n_excluded_degenerate = 0
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        d = np.load(npz_path)
        if "eval_window_start" not in d:
            continue
        r = process_unit(npz_path)
        if r is None:
            n_excluded_thin += 1
            continue
        rows.append(r)

    print("3B.1 DECAY CURVE")
    print("n_units=%d  excluded(thin/degenerate)=%d" % (len(rows), n_excluded_thin))

    clustered = [r for r in rows if r["clustering_flag"]]
    print("units flagged for anomaly-clustering bias: %d/%d" % (len(clustered), len(rows)))

    slopes = np.array([r["slope_rho"] for r in rows])
    stat, p = wilcoxon(slopes)
    med = float(np.median(slopes))
    print("\nPOOLED: Wilcoxon signed-rank of slopes vs 0: stat=%.4f p=%.6f  median_rho=%.4f  n_pos=%d n_neg=%d" % (
        stat, p, med, int(np.sum(slopes > 0)), int(np.sum(slopes < 0))))

    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r["slope_rho"])
    print("\nPER DATASET:")
    for ds, vals in sorted(by_dataset.items()):
        v = np.array(vals)
        if len(v) < 5:
            print("  %-12s n=%-4d too few" % (ds, len(v)))
            continue
        st, pp = wilcoxon(v)
        print("  %-12s n=%-4d median_rho=%.4f  wilcoxon_p=%.6f" % (ds, len(v), np.median(v), pp))

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r["slope_rho"])
    print("\nPER METHOD:")
    for m, vals in sorted(by_method.items()):
        v = np.array(vals)
        if len(v) < 5:
            print("  %-12s n=%-4d too few" % (m, len(v)))
            continue
        st, pp = wilcoxon(v)
        print("  %-12s n=%-4d median_rho=%.4f  wilcoxon_p=%.6f" % (m, len(v), np.median(v), pp))

    warmup_available = [r for r in rows if r["warmup_control"] is not None]
    print("\nDETECTOR WARM-UP CONTROL: n_with_train_scores=%d" % len(warmup_available))
    if warmup_available:
        wslopes = np.array([r["warmup_control"]["rho"] for r in warmup_available])
        wst, wp = wilcoxon(wslopes)
        print("  train-split slope Wilcoxon: median_rho=%.4f p=%.6f" % (np.median(wslopes), wp))
    else:
        print("  NO train_scores available in cache - warm-up control NOT performed, flag as limitation")

    with open("results/phase3b/PHASE3B1_raw.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\n%s" % ("BRANCH: positive slope + significant" if (med > 0 and p < 0.01) else
                     "BRANCH: flat/no significant slope" if p > 0.05 else
                     "BRANCH: negative slope" if med < 0 and p < 0.01 else
                     "BRANCH: ambiguous, inspect manually"))


if __name__ == "__main__":
    main()
