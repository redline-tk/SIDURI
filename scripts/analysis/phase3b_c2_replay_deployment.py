import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.calibrator import MachineAdaptiveCalibrator

EPS = 0.05
M_FIXED = 175


def replay_unit(npz_path, buffer_size):
    d = np.load(npz_path)
    cal = np.asarray(d["cal_scores"], dtype=np.float64)
    ev = np.asarray(d["eval_scores"], dtype=np.float64)
    y = np.asarray(d["eval_labels"], dtype=int)
    if len(cal) < 5:
        return None

    mac = MachineAdaptiveCalibrator(eta=EPS, buffer_size=buffer_size)
    key = "unit"
    cal_used = cal[-buffer_size:] if len(cal) > buffer_size else cal
    for s in cal_used:
        mac.update(key, float(s))

    fpr_trace = []
    flags_window = []
    labels_window = []
    checkpoint_every = max(1, len(ev) // 40)

    for i, s in enumerate(ev):
        flag = mac.predict(key, float(s))
        flags_window.append(flag)
        labels_window.append(y[i])
        if not flag:
            mac.update(key, float(s))

        if (i + 1) % checkpoint_every == 0 or i == len(ev) - 1:
            window_flags = np.array(flags_window[-checkpoint_every:])
            window_labels = np.array(labels_window[-checkpoint_every:])
            normal_mask = window_labels == 0
            fpr = float(np.mean(window_flags[normal_mask])) if normal_mask.sum() > 0 else np.nan
            fpr_trace.append(fpr)

    stem = npz_path.stem
    unit_id, method = stem.split("__", 1) if "__" in stem else (stem, "unknown")
    return dict(unit=unit_id, method=method, fpr_trace=fpr_trace)


def main():
    for dataset in ["unsw_nb15", "cicids2017"]:
        print("\n=== %s (DEPLOYMENT ARM: skip only when NOT flagged) ===" % dataset.upper())
        cache_dir = Path("results/score_cache") / dataset
        results = []
        for npz_path in sorted(cache_dir.glob("*__*.npz")):
            r = replay_unit(npz_path, M_FIXED)
            if r is not None:
                results.append(r)
        for r in results:
            final_fpr = float(np.nanmean(r["fpr_trace"][-5:])) if len(r["fpr_trace"]) >= 5 else float("nan")
            bound = EPS + 1.0 / (M_FIXED + 1)
            ok = final_fpr <= bound
            print("  %-24s final_windowed_fpr=%.4f  bound=%.4f  %s" % (
                "%s__%s" % (r["unit"], r["method"]), final_fpr, bound, "OK" if ok else "EXCEEDS"))
        with open("results/phase3b/PHASE3B_C2_deployment_%s.json" % dataset, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
