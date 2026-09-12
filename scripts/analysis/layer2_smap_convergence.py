import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_extended_methods as rem
from deepod.models import NeuTraL, GOAD, SLAD, RDP, ICL

CONVERGED_EPOCHS = 250
CACHE_LABEL = "smap_converged"
N_WEAK_SAMPLE = 20
N_STRONG_CONTROL = 5

MODELS = {
    "GOAD": lambda **kw: GOAD(hidden_dim=8, **kw),
    "NeuTraL": NeuTraL,
    "SLAD": SLAD,
    "RDP": RDP,
    "ICL": ICL,
}


def old_cache_auroc(unit_id, method):
    p = Path("results/score_cache/smap") / ("%s__%s.npz" % (unit_id, method))
    if not p.exists():
        return None
    d = np.load(p)
    ev = d["eval_scores"]
    y = d["eval_labels"]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, ev))


def main():
    units = {u[0]: u for u in rem.load_smap_msl_units("smap")}

    candidates = []
    for p in sorted(Path("results/score_cache/smap").glob("*__*.npz")):
        stem = p.stem
        if "__" not in stem:
            continue
        unit_id, method = stem.split("__", 1)
        if method not in MODELS or unit_id not in units:
            continue
        auroc = old_cache_auroc(unit_id, method)
        if auroc is None:
            continue
        candidates.append((unit_id, method, auroc))

    weak = sorted([c for c in candidates if c[2] < 0.6], key=lambda c: c[2])
    strong = sorted([c for c in candidates if c[2] >= 0.7], key=lambda c: -c[2])

    rng = np.random.default_rng(0)
    weak_sample = list(rng.choice(len(weak), size=min(N_WEAK_SAMPLE, len(weak)), replace=False))
    weak_sample = [weak[i] for i in weak_sample]
    strong_sample = strong[:N_STRONG_CONTROL]

    targets = weak_sample + strong_sample
    print("Sampled %d weak units (orig AUROC<0.6) + %d strong controls (orig AUROC>=0.7)" % (
        len(weak_sample), len(strong_sample)))

    results = []
    for unit_id, method, old_auroc in targets:
        _, train_raw, train_labels, test_raw, test_labels = units[unit_id]
        n_cal = int(len(test_raw) * rem.CAL_FRAC)
        cal_raw, eval_raw = test_raw[:n_cal], test_raw[n_cal:]
        cal_lbl, eval_lbl = test_labels[:n_cal], test_labels[n_cal:]

        keep, dropped_idx = rem.feature_prescreen(train_raw.astype(np.float32))
        if keep.sum() < 2:
            print("  %s/%s: insufficient features, skip" % (method, unit_id))
            continue

        train_sel = train_raw.astype(np.float32)[:, keep]
        cal_sel = cal_raw.astype(np.float32)[:, keep]
        eval_sel = eval_raw.astype(np.float32)[:, keep]

        try:
            cls = MODELS[method]
            clf = cls(epochs=CONVERGED_EPOCHS, device=rem.DEVICE, verbose=0)
            clf.fit(train_sel)
            s_cal_pt = clf.decision_function(cal_sel)
            s_eval_pt = clf.decision_function(eval_sel)
            del clf
            import torch
            torch.cuda.empty_cache()

            if np.isnan(s_cal_pt).any() or np.isnan(s_eval_pt).any():
                print("  %s/%s: nan scores, skip" % (method, unit_id))
                continue

            new_auroc = roc_auc_score(eval_lbl, s_eval_pt) if len(np.unique(eval_lbl)) > 1 else float("nan")

            s_cal_w, y_cal_w, cal_starts = rem.aggregate_to_windows_with_starts(s_cal_pt, cal_lbl)
            s_eval_w, y_eval_w, eval_starts = rem.aggregate_to_windows_with_starts(s_eval_pt, eval_lbl)
            rem.cache_scores(
                CACHE_LABEL, method, unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w,
                cal_window_start=cal_starts, eval_window_start=eval_starts,
                n_features_raw=train_raw.shape[1], n_features_dropped=int(len(dropped_idx)),
                dropped_feature_idx=dropped_idx,
            )

            delta = new_auroc - old_auroc
            group = "WEAK" if old_auroc < 0.6 else "STRONG"
            print("  [%s] %s/%s: old=%.4f new=%.4f delta=%+.4f" % (group, method, unit_id, old_auroc, new_auroc, delta))
            results.append((group, unit_id, method, old_auroc, new_auroc))
        except Exception as e:
            print("  %s/%s: FAILED %s" % (method, unit_id, e))

    weak_deltas = [n - o for g, u, m, o, n in results if g == "WEAK"]
    strong_deltas = [n - o for g, u, m, o, n in results if g == "STRONG"]
    n_weak_rose = sum(1 for d in weak_deltas if d > 0.2)

    print("\n=== SUMMARY ===")
    print("WEAK units (n=%d): mean_delta=%.4f  n_rose_materially(>0.2)=%d/%d" % (
        len(weak_deltas), np.mean(weak_deltas) if weak_deltas else float("nan"), n_weak_rose, len(weak_deltas)))
    print("STRONG controls (n=%d): mean_delta=%.4f (should be near 0)" % (
        len(strong_deltas), np.mean(strong_deltas) if strong_deltas else float("nan")))

    if n_weak_rose >= len(weak_deltas) * 0.5:
        print("\nVERDICT: WEAK UNITS ROSE MATERIALLY UNDER CONVERGENCE. SMAP retraction reasoning NEEDS REWRITING.")
    else:
        print("\nVERDICT: WEAK UNITS STAYED WEAK. Benchmark-quality retraction HOLDS.")


if __name__ == "__main__":
    main()
