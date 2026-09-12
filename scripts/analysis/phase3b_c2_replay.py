import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.calibrator import MachineAdaptiveCalibrator

EPS = 0.05
M_FIXED = 175


def replay_unit(npz_path, buffer_size):
    d = np.load(npz_path)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    cal_lbl = np.asarray(d["cal_labels"], dtype=int) if "cal_labels" in d else None
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < 5:
        return None

    mac = MachineAdaptiveCalibrator(eta=EPS, buffer_size=buffer_size)
    key = "unit"
    cal_used = cal[-buffer_size:] if len(cal) > buffer_size else cal
    cal_lbl_used = cal_lbl[-buffer_size:] if cal_lbl is not None and len(cal_lbl) > buffer_size else cal_lbl

    for s in cal_used:
        mac.update(key, float(s))

    contam_trace = []
    fpr_trace = []
    flags_window = []
    labels_window = []
    checkpoint_every = max(1, len(ev) // 40)

    buf_labels = list(cal_lbl_used) if cal_lbl_used is not None else [0] * len(cal_used)

    for i, s in enumerate(ev):
        flag = mac.predict(key, float(s))
        flags_window.append(flag)
        labels_window.append(y[i])
        if y[i] == 0:
            mac.update(key, float(s))
            buf_labels.append(0)
            if len(buf_labels) > buffer_size:
                buf_labels = buf_labels[-buffer_size:]

        if (i + 1) % checkpoint_every == 0 or i == len(ev) - 1:
            contam = float(np.mean(np.array(buf_labels) != 0)) if buf_labels else np.nan
            window_flags = np.array(flags_window[-checkpoint_every:])
            window_labels = np.array(labels_window[-checkpoint_every:])
            normal_mask = window_labels == 0
            fpr = float(np.mean(window_flags[normal_mask])) if normal_mask.sum() > 0 else np.nan
            contam_trace.append(contam)
            fpr_trace.append(fpr)

    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    return dict(unit=unit_id, method=method, contam_trace=contam_trace, fpr_trace=fpr_trace)


def main():
    for dataset in ["unsw_nb15", "cicids2017", "smd", "smap", "batadal", "msl"]:
        print("\n=== %s ===" % dataset.upper())
        cache_dir = Path("results/score_cache") / dataset
        results = []
        for npz_path in sorted(cache_dir.glob("*__*.npz")):
            r = replay_unit(npz_path, M_FIXED)
            if r is not None:
                results.append(r)

        for r in results[:6]:
            initial_contam = r["contam_trace"][0] if r["contam_trace"] else float("nan")
            final_contam = r["contam_trace"][-1] if r["contam_trace"] else float("nan")
            final_fpr = np.nanmean(r["fpr_trace"][-5:]) if len(r["fpr_trace"]) >= 5 else float("nan")
            print("  %-24s initial_buf_contam=%.4f final_buf_contam=%.4f final_windowed_fpr=%.4f" % (
                "%s__%s" % (r["unit"], r["method"]), initial_contam, final_contam, final_fpr))

        with open("results/phase3b/PHASE3B_C2_replay_%s.json" % dataset, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
