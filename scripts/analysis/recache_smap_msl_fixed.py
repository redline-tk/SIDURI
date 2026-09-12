import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import run_extended_methods as rem
from deepod.models import NeuTraL, GOAD, SLAD, RDP, ICL

MODELS = {
    "GOAD": lambda **kw: GOAD(hidden_dim=8, **kw),
    "NeuTraL": NeuTraL,
    "SLAD": SLAD,
    "RDP": RDP,
    "ICL": ICL,
}

def run_dataset(dataset_name, units):
    print("=== %s: %d real units (post-fix) ===" % (dataset_name, len(units)))
    for unit_id, train_raw, train_labels, test_raw, test_labels in units:
        n_cal = int(len(test_raw) * rem.CAL_FRAC)
        cal_raw, eval_raw = test_raw[:n_cal], test_raw[n_cal:]
        cal_lbl, eval_lbl = test_labels[:n_cal], test_labels[n_cal:]

        keep, dropped_idx = rem.feature_prescreen(train_raw.astype(np.float32))
        if keep.sum() < 2:
            print("  %s: insufficient features, skip" % unit_id)
            continue
        train_sel = train_raw.astype(np.float32)[:, keep]
        cal_sel = cal_raw.astype(np.float32)[:, keep]
        eval_sel = eval_raw.astype(np.float32)[:, keep]

        for method_name, cls in MODELS.items():
            try:
                clf = cls(epochs=50, device=rem.DEVICE, verbose=0)
                clf.fit(train_sel)
                s_cal_pt = clf.decision_function(cal_sel)
                s_eval_pt = clf.decision_function(eval_sel)
                del clf
                import torch
                torch.cuda.empty_cache()

                if np.isnan(s_cal_pt).any() or np.isnan(s_eval_pt).any():
                    print("  %s/%s: nan scores, skip" % (method_name, unit_id))
                    continue

                s_cal_w, y_cal_w, cal_starts = rem.aggregate_to_windows_with_starts(s_cal_pt, cal_lbl)
                s_eval_w, y_eval_w, eval_starts = rem.aggregate_to_windows_with_starts(s_eval_pt, eval_lbl)
                rem.cache_scores(
                    dataset_name, method_name, unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w,
                    cal_window_start=cal_starts, eval_window_start=eval_starts,
                    n_features_raw=train_raw.shape[1], n_features_dropped=int(len(dropped_idx)),
                    dropped_feature_idx=dropped_idx,
                )
            except Exception as e:
                print("  %s/%s: FAILED %s" % (method_name, unit_id, e))
    print("Done: %s\n" % dataset_name)

smap_units = rem.load_smap_msl_units("smap")
msl_units = rem.load_smap_msl_units("msl")
print("Confirmed pre-run counts: smap=%d msl=%d" % (len(smap_units), len(msl_units)))
run_dataset("smap", smap_units)
run_dataset("msl", msl_units)
