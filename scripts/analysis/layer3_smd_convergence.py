import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_extended_methods as rem
from deepod.models import NeuTraL, GOAD, SLAD, RDP, ICL

CONVERGED_EPOCHS = 250
CACHE_LABEL = "smd_converged"

MODELS = {
    "GOAD": lambda **kw: GOAD(hidden_dim=8, **kw),
    "NeuTraL": NeuTraL,
    "SLAD": SLAD,
    "RDP": RDP,
    "ICL": ICL,
}


def run_unit(unit_id, train_raw, train_labels, test_raw, test_labels):
    n_cal = int(len(test_raw) * rem.CAL_FRAC)
    cal_raw, eval_raw = test_raw[:n_cal], test_raw[n_cal:]
    cal_lbl, eval_lbl = test_labels[:n_cal], test_labels[n_cal:]

    keep, dropped_idx = rem.feature_prescreen(train_raw.astype(np.float32))
    if keep.sum() < 2:
        print("  %s: insufficient features, skip" % unit_id)
        return

    train_sel = train_raw.astype(np.float32)[:, keep]
    cal_sel = cal_raw.astype(np.float32)[:, keep]
    eval_sel = eval_raw.astype(np.float32)[:, keep]

    for method_name, cls in MODELS.items():
        try:
            clf = cls(epochs=CONVERGED_EPOCHS, device=rem.DEVICE, verbose=0)
            clf.fit(train_sel)
            s_cal_pt = clf.decision_function(cal_sel)
            s_eval_pt = clf.decision_function(eval_sel)
            del clf
            import torch
            torch.cuda.empty_cache()

            n_nan_cal = int(np.isnan(s_cal_pt).sum())
            n_nan_eval = int(np.isnan(s_eval_pt).sum())
            if n_nan_cal or n_nan_eval:
                print("  %s/%s: nan scores, skip" % (method_name, unit_id))
                continue

            s_cal_w, y_cal_w, cal_starts = rem.aggregate_to_windows_with_starts(s_cal_pt, cal_lbl)
            s_eval_w, y_eval_w, eval_starts = rem.aggregate_to_windows_with_starts(s_eval_pt, eval_lbl)

            cached = rem.cache_scores(
                CACHE_LABEL, method_name, unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w,
                cal_window_start=cal_starts, eval_window_start=eval_starts,
                n_features_raw=train_raw.shape[1], n_features_dropped=int(len(dropped_idx)),
                dropped_feature_idx=dropped_idx,
            )
            print("  %s/%s: cached=%s" % (method_name, unit_id, cached))
        except Exception as e:
            print("  %s/%s: FAILED %s" % (method_name, unit_id, e))


def main():
    units = rem.load_smd_units()
    print("Loaded %d SMD units. Training at epochs=%d (converged, vs original 50)." % (len(units), CONVERGED_EPOCHS))
    for unit_id, train_raw, train_labels, test_raw, test_labels in units:
        print("Unit: %s" % unit_id)
        run_unit(unit_id, train_raw, train_labels, test_raw, test_labels)


if __name__ == "__main__":
    main()
