import sys
import json
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import wilcoxon, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CACHE_DIR = Path("results/score_cache")
OUT_DIR = Path("results/phase3b")
EPS = 0.05
DELTA = 0.10
M_FIXED = 175
MIN_PER_DECILE = 30
D_FULL = 10
D_FALLBACK = 5
N_PERM = 1000
SEED = 20260721
M_MIN_EPS_DELTA = int(np.ceil(np.log(DELTA) / np.log(1.0 - EPS)))
CLUSTER_SPAN_RATIO = 2.0
COVERAGE_MIN = 0.80
TAIL_GAP_MAX = 0.15

_GATE_FN = None
try:
    from src.chad.pac.pac_threshold import split_conformal_threshold as _GATE_FN
except Exception:
    _GATE_FN = None


def local_threshold(cal_scores):
    M = len(cal_scores)
    r = int(np.ceil((1.0 - EPS) * (M + 1)))
    r = min(max(r, 1), M)
    return float(np.sort(cal_scores)[r - 1])


def verify_threshold_backend():
    rng = np.random.default_rng(0)
    if _GATE_FN is None:
        return "LOCAL", "gate-verified split_conformal_threshold NOT importable, using local order-statistic construction"
    for M in [45, 47, 50, 75, 100, 175, 300]:
        c = rng.normal(0, 1, M)
        try:
            g = float(_GATE_FN(c, EPS))
        except TypeError:
            try:
                g = float(_GATE_FN(c, eta=EPS))
            except Exception as e:
                return "LOCAL", "gate fn signature probe failed (%s), using local" % type(e).__name__
        except Exception as e:
            return "LOCAL", "gate fn raised %s, using local" % type(e).__name__
        if not np.isclose(g, local_threshold(c), rtol=0, atol=0):
            return "LOCAL", "MISMATCH at M=%d: gate=%.10g local=%.10g -- USING LOCAL, INVESTIGATE" % (
                M, g, local_threshold(c))
    return "GATE", "gate-verified split_conformal_threshold imported and matches local at all probed M"


def threshold(cal_scores, backend):
    if backend == "GATE":
        try:
            return float(_GATE_FN(cal_scores, EPS))
        except TypeError:
            return float(_GATE_FN(cal_scores, eta=EPS))
    return local_threshold(cal_scores)


def bin_edges(n, n_bins):
    return np.linspace(0, n, n_bins + 1).astype(int)


def fast_spearman_vs_index(fprs):
    n = len(fprs)
    a = np.arange(1.0, n + 1.0)
    a = a - a.mean()
    r = rankdata(fprs)
    r = r - r.mean()
    dr = np.sqrt(np.dot(r, r))
    if dr < 1e-12:
        return 0.0
    return float(np.dot(a, r) / (np.sqrt(np.dot(a, a)) * dr))


def binned_fpr(exceed, edges, counts):
    sums = np.add.reduceat(exceed.astype(np.float64), edges[:-1])
    return sums / counts


def permutation_null(exceed, edges, counts, rng, n_perm):
    n = len(exceed)
    out = np.empty(n_perm, dtype=np.float64)
    e = exceed.astype(np.float64)
    for i in range(n_perm):
        p = rng.permutation(n)
        out[i] = fast_spearman_vs_index(binned_fpr(e[p], edges, counts))
    return out


def process_unit(npz_path, cal_mode, backend, do_perm, rng):
    reasons = []
    try:
        d = np.load(npz_path)
    except Exception:
        return None, "unreadable_npz"
    if "eval_window_start" not in d:
        return None, "no_eval_window_start"
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    starts = np.asarray(d["eval_window_start"], dtype=np.int64)

    if len(cal) < 10:
        return None, "cal_lt_10"
    if not np.isfinite(cal).all() or not np.isfinite(ev).all():
        return None, "nonfinite_scores"

    n_cal_raw = len(cal)
    if len(cal) > M_FIXED:
        cal_used = cal[-M_FIXED:] if cal_mode == "last" else cal[:M_FIXED]
    else:
        cal_used = cal
    m_eff = len(cal_used)
    tau = threshold(cal_used, backend)
    n_distinct_cal = int(len(np.unique(cal_used)))

    normal_mask = y == 0
    ev_n = ev[normal_mask]
    st_n = starts[normal_mask]
    if len(ev_n) < MIN_PER_DECILE * D_FALLBACK:
        return None, "too_few_normals"

    order = np.argsort(st_n, kind="stable")
    ev_n = ev_n[order]
    st_n = st_n[order]
    n = len(ev_n)

    n_bins = D_FULL
    edges = bin_edges(n, n_bins)
    counts = np.diff(edges)
    if counts.min() < MIN_PER_DECILE:
        n_bins = D_FALLBACK
        edges = bin_edges(n, n_bins)
        counts = np.diff(edges)
        if counts.min() < MIN_PER_DECILE:
            return None, "bins_too_thin"

    exceed = ev_n > tau
    fprs = binned_fpr(exceed, edges, counts)

    if np.all(fprs == 0):
        return None, "all_zero_fpr"
    if np.all(fprs == 1):
        return None, "all_one_fpr"

    spans = np.array([st_n[edges[i + 1] - 1] - st_n[edges[i]] for i in range(n_bins)], dtype=np.float64)
    first_span = spans[0] if spans[0] > 0 else np.nan
    span_ratio = float(spans[-1] / first_span) if np.isfinite(first_span) else np.nan

    ev_lo, ev_hi = float(starts.min()), float(starts.max())
    ev_range = ev_hi - ev_lo
    norm_range = float(st_n[-1] - st_n[0])
    coverage_frac = float(norm_range / ev_range) if ev_range > 0 else np.nan
    tail_gap_frac = float((ev_hi - st_n[-1]) / ev_range) if ev_range > 0 else np.nan

    cluster_flag = bool(
        (np.isfinite(span_ratio) and span_ratio > CLUSTER_SPAN_RATIO)
        or (np.isfinite(coverage_frac) and coverage_frac < COVERAGE_MIN)
        or (np.isfinite(tail_gap_frac) and tail_gap_frac > TAIL_GAP_MAX)
    )

    rho = fast_spearman_vs_index(fprs)

    perm = None
    perm_p = None
    if do_perm:
        null = permutation_null(exceed, edges, counts, rng, N_PERM)
        perm = null
        perm_p = float((1 + np.sum(np.abs(null) >= abs(rho))) / (N_PERM + 1))

    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    rec = dict(
        dataset=npz_path.parent.name,
        unit=unit_id,
        method=method,
        n_bins=int(n_bins),
        n_cal_raw=int(n_cal_raw),
        m_eff=int(m_eff),
        certifiable=bool(m_eff >= M_MIN_EPS_DELTA),
        n_distinct_cal=n_distinct_cal,
        tied_flag=bool(n_distinct_cal / m_eff < 0.9),
        n_normals=int(n),
        slope_rho=float(rho),
        fprs=[float(x) for x in fprs],
        overall_fpr=float(exceed.mean()),
        span_ratio=None if not np.isfinite(span_ratio) else float(span_ratio),
        coverage_frac=None if not np.isfinite(coverage_frac) else float(coverage_frac),
        tail_gap_frac=None if not np.isfinite(tail_gap_frac) else float(tail_gap_frac),
        cluster_flag=cluster_flag,
        perm_p=perm_p,
    )
    return (rec, perm), None


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(str(path.parent), os.O_DIRECTORY)
    os.fsync(dfd)
    os.close(dfd)


def summarise(rows, label):
    if len(rows) < 5:
        return "  %-14s n=%-4d too few" % (label, len(rows))
    v = np.array([r["slope_rho"] for r in rows])
    nz = v[v != 0]
    if len(nz) < 5:
        return "  %-14s n=%-4d all-zero slopes" % (label, len(v))
    try:
        _, p = wilcoxon(v)
    except Exception:
        p = np.nan
    return "  %-14s n=%-4d median_rho=%+.4f  wilcoxon_p=%.6f  n_pos=%-4d n_neg=%-4d" % (
        label, len(v), float(np.median(v)), p, int(np.sum(v > 0)), int(np.sum(v < 0)))


def run_pass(cal_mode, backend, do_perm):
    rng = np.random.default_rng(SEED)
    rows = []
    perms = []
    reasons = Counter()
    reasons_by_ds = defaultdict(Counter)
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        res, reason = process_unit(npz_path, cal_mode, backend, do_perm, rng)
        if res is None:
            reasons[reason] += 1
            reasons_by_ds[npz_path.parent.name][reason] += 1
            continue
        rec, null = res
        rows.append(rec)
        perms.append(null)
    return rows, perms, reasons, reasons_by_ds


def check4_exclusions(reasons, reasons_by_ds, n_included):
    print("\n" + "=" * 78)
    print("CHECK 4 -- EXCLUSION AUDIT")
    print("=" * 78)
    total_ex = sum(reasons.values())
    print("included=%d  excluded=%d  total=%d" % (n_included, total_ex, n_included + total_ex))
    print("\n  reason breakdown (pooled):")
    for k, v in reasons.most_common():
        print("    %-24s %d" % (k, v))
    print("\n  all_zero_fpr by dataset (selection-on-outcome risk;")
    print("  cross-reference against CHECK 5 -- a contaminated cal set inflates tau,")
    print("  which suppresses ALL firing and shows up here as mass all_zero exclusions):")
    for ds in sorted(reasons_by_ds):
        z = reasons_by_ds[ds].get("all_zero_fpr", 0)
        t = sum(reasons_by_ds[ds].values())
        print("    %-14s all_zero=%-4d of %-4d excluded" % (ds, z, t))


def check1_recency(rows_first, rows_last):
    print("\n" + "=" * 78)
    print("CHECK 1 -- RECENCY ARTIFACT: cal[:175] (first) vs cal[-175:] (last)")
    print("=" * 78)
    key = lambda r: (r["dataset"], r["unit"], r["method"])
    a = {key(r): r["slope_rho"] for r in rows_first}
    b = {key(r): r["slope_rho"] for r in rows_last}
    shared = sorted(set(a) & set(b))
    print("units in both passes: %d  (first-only=%d, last-only=%d)" % (
        len(shared), len(set(a) - set(b)), len(set(b) - set(a))))
    by_ds = defaultdict(list)
    for k in shared:
        by_ds[k[0]].append((a[k], b[k]))
    print("\n  %-14s %-6s %-12s %-12s %-10s" % ("dataset", "n", "median_first", "median_last", "median_delta"))
    for ds in sorted(by_ds):
        arr = np.array(by_ds[ds])
        if len(arr) < 5:
            print("  %-14s %-6d too few" % (ds, len(arr)))
            continue
        print("  %-14s %-6d %-+12.4f %-+12.4f %-+10.4f" % (
            ds, len(arr), np.median(arr[:, 0]), np.median(arr[:, 1]),
            np.median(arr[:, 1] - arr[:, 0])))
    print("\n  VERDICT RULE: if median_last collapses toward 0 on SMD while median_first is +0.33,")
    print("  the decay slope was an artifact of using a STALE calibration slice, not drift.")


def check2_permutation(rows, perms):
    print("\n" + "=" * 78)
    print("CHECK 2 -- TEMPORAL PERMUTATION NULL (%d perms, seed=%d)" % (N_PERM, SEED))
    print("=" * 78)
    idx = [i for i, p in enumerate(perms) if p is not None]
    if not idx:
        print("  no permutation nulls computed")
        return
    by_ds = defaultdict(list)
    for i in idx:
        by_ds[rows[i]["dataset"]].append(i)
    print("  %-14s %-6s %-11s %-24s %-8s" % ("dataset", "n", "obs_median", "null_median_95pct_CI", "p_emp"))
    for ds in sorted(by_ds):
        ii = by_ds[ds]
        if len(ii) < 5:
            print("  %-14s %-6d too few" % (ds, len(ii)))
            continue
        obs = np.median([rows[i]["slope_rho"] for i in ii])
        mat = np.vstack([perms[i] for i in ii])
        null_medians = np.median(mat, axis=0)
        lo, hi = np.percentile(null_medians, [2.5, 97.5])
        p = (1 + np.sum(np.abs(null_medians) >= abs(obs))) / (len(null_medians) + 1)
        print("  %-14s %-6d %-+11.4f [%+.4f, %+.4f]%s %-8.5f" % (
            ds, len(ii), obs, lo, hi, " " * 6, p))
    print("\n  per-unit permutation p < 0.05 (expected 5% under the null):")
    for ds in sorted(by_ds):
        ii = by_ds[ds]
        allp = np.array([rows[i]["perm_p"] for i in ii])
        print("    %-14s %d/%d (%.1f%%)" % (ds, int(np.sum(allp < 0.05)), len(allp),
                                            100.0 * np.mean(allp < 0.05)))


def check3_clustering(rows):
    print("\n" + "=" * 78)
    print("CHECK 3 -- ANOMALY-CLUSTERING GUARD (span ratio + normal-window coverage of eval)")
    print("=" * 78)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    print("  %-14s %-5s %-9s %-10s %-10s %-9s %-24s" % (
        "dataset", "n", "n_flag", "med_span_r", "med_cover", "med_tail", "median_rho flag/clean"))
    for ds in sorted(by_ds):
        rr = by_ds[ds]
        gv = lambda key: np.array([r[key] for r in rr if r[key] is not None], dtype=float)
        sr, cv, tg = gv("span_ratio"), gv("coverage_frac"), gv("tail_gap_frac")
        fl = [r for r in rr if r["cluster_flag"]]
        cl = [r for r in rr if not r["cluster_flag"]]
        fmt = lambda g: ("%+.4f" % np.median([r["slope_rho"] for r in g])) if len(g) >= 3 else "n/a"
        print("  %-14s %-5d %-9d %-10s %-10s %-9s %s / %s" % (
            ds, len(rr), len(fl),
            "%.3f" % np.median(sr) if len(sr) else "n/a",
            "%.3f" % np.median(cv) if len(cv) else "n/a",
            "%.3f" % np.median(tg) if len(tg) else "n/a",
            fmt(fl), fmt(cl)))
    print()
    print("  med_cover = fraction of the eval time range spanned by NORMAL windows.")
    print("  med_tail  = fraction of the eval range after the LAST normal window.")
    print("  Low coverage or a large tail gap means the decay curve is TRUNCATED: the late")
    print("  deciles never reach the end of eval, so the drift axis is short. If flagged and")
    print("  clean medians differ materially, report the clean subset as primary.")


def check5_cal_contamination():
    print("\n" + "=" * 78)
    print("CHECK 5 -- CALIBRATION-PERIOD ANOMALY CONTAMINATION (Branch C mandate)")
    print("=" * 78)
    by_ds = defaultdict(list)
    for npz_path in sorted(CACHE_DIR.rglob("*__*.npz")):
        try:
            d = np.load(npz_path)
        except Exception:
            continue
        if "cal_labels" not in d:
            continue
        cl = np.asarray(d["cal_labels"], dtype=int)
        if len(cl) == 0:
            continue
        by_ds[npz_path.parent.name].append(float(np.mean(cl != 0)))
    if not by_ds:
        print("  cal_labels not present in cache -- CANNOT AUDIT. Flag as blocking for Branch C.")
        return
    print("  %-14s %-6s %-12s %-12s %-14s" % ("dataset", "n", "mean_cal_rate", "max_cal_rate", "frac_units>1%"))
    for ds in sorted(by_ds):
        v = np.array(by_ds[ds])
        print("  %-14s %-6d %-12.4f %-12.4f %-14.3f" % (
            ds, len(v), v.mean(), v.max(), float(np.mean(v > 0.01))))
    print("\n  Any dataset with a nonzero mean cal rate has a CONTAMINATED calibration set:")
    print("  tau is inflated, every downstream FPR is optimistic. CICIDS/UNSW are the suspects.")


def main():
    backend, msg = verify_threshold_backend()
    print("=" * 78)
    print("PHASE 3B.1b -- DECAY CURVE WITH FOUR CHECKS")
    print("=" * 78)
    print("threshold backend: %s" % backend)
    print("  %s" % msg)
    print("M_min(eps=%.2f, delta=%.2f) = %d  -- units below this are flagged non-certifiable" % (
        EPS, DELTA, M_MIN_EPS_DELTA))

    rows_last, perms_last, reasons, reasons_by_ds = run_pass("last", backend, True)
    rows_first, _, _, _ = run_pass("first", backend, False)

    check4_exclusions(reasons, reasons_by_ds, len(rows_last))

    print("\n" + "=" * 78)
    print("PRIMARY RESULT -- cal[-175:] (RECENCY-CORRECTED), per dataset")
    print("=" * 78)
    by_ds = defaultdict(list)
    for r in rows_last:
        by_ds[r["dataset"]].append(r)
    for ds in sorted(by_ds):
        print(summarise(by_ds[ds], ds))

    print("\n  certifiable units only (m_eff >= %d):" % M_MIN_EPS_DELTA)
    for ds in sorted(by_ds):
        print(summarise([r for r in by_ds[ds] if r["certifiable"]], ds))

    print("\n  untied units only (n_distinct/m_eff >= 0.9):")
    for ds in sorted(by_ds):
        print(summarise([r for r in by_ds[ds] if not r["tied_flag"]], ds))

    print("\n" + "=" * 78)
    print("DATASET x METHOD (per-method pooling across datasets is uninterpretable)")
    print("=" * 78)
    by_dm = defaultdict(list)
    for r in rows_last:
        by_dm[(r["dataset"], r["method"])].append(r)
    for ds in sorted(set(k[0] for k in by_dm)):
        print("  %s:" % ds)
        for k in sorted(by_dm):
            if k[0] != ds:
                continue
            print("  " + summarise(by_dm[k], k[1]))

    check1_recency(rows_first, rows_last)
    check2_permutation(rows_last, perms_last)
    check3_clustering(rows_last)
    check5_cal_contamination()

    atomic_write_json(OUT_DIR / "PHASE3B1b_rows_calLast.json", rows_last)
    atomic_write_json(OUT_DIR / "PHASE3B1b_rows_calFirst.json", rows_first)
    atomic_write_json(OUT_DIR / "PHASE3B1b_meta.json", dict(
        seed=SEED, n_perm=N_PERM, eps=EPS, delta=DELTA, m_fixed=M_FIXED,
        m_min=M_MIN_EPS_DELTA, min_per_decile=MIN_PER_DECILE,
        backend=backend, backend_msg=msg,
        cluster_span_ratio=CLUSTER_SPAN_RATIO,
        exclusions=dict(reasons)))
    print("\nwrote results/phase3b/PHASE3B1b_{rows_calLast,rows_calFirst,meta}.json")


if __name__ == "__main__":
    main()
