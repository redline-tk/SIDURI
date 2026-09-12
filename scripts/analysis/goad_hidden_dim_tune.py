import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_extended_methods as rem
from deepod.models import GOAD

CANDIDATES = [8, 16, 32, 50, 100]
SEEDS = [0, 1]

LOADERS = {
    "smd": rem.load_smd_units,
    "smap": lambda: rem.load_smap_msl_units("smap"),
    "msl": lambda: rem.load_smap_msl_units("msl"),
    "batadal": rem.load_batadal_units,
    "cicids2017": rem.load_cicids_units,
    "unsw_nb15": rem.load_unsw_units,
}


def main():
    results = {}
    for dataset, loader in LOADERS.items():
        units = loader()
        chosen = None
        for unit_id, train_raw, train_labels, test_raw, test_labels in units:
            n_cal = int(len(test_raw) * rem.CAL_FRAC)
            cal_raw = test_raw[:n_cal]
            cal_lbl = test_labels[:n_cal]
            if len(np.unique(cal_lbl)) >= 2 and int(cal_lbl.sum()) >= 5:
                chosen = (unit_id, train_raw, train_labels, test_raw, test_labels, cal_raw, cal_lbl)
                break
        if chosen is None:
            print("%-14s SKIP: no unit among %d has >=5 calibration anomalies" % (dataset, len(units)))
            continue
        unit_id, train_raw, train_labels, test_raw, test_labels, cal_raw, cal_lbl = chosen

        keep, _ = rem.feature_prescreen(train_raw.astype(np.float32))
        if keep.sum() < 2:
            print("%-14s SKIP: insufficient features" % dataset)
            continue
        train_sel = train_raw.astype(np.float32)[:, keep]
        cal_sel = cal_raw.astype(np.float32)[:, keep]

        print("%s (unit=%s, n_cal=%d, n_cal_anom=%d):" % (dataset, unit_id, len(cal_lbl), int(cal_lbl.sum())))
        best_hd, best_auc = None, -1
        dataset_results = {}
        for hd in CANDIDATES:
            aucs = []
            for seed in SEEDS:
                clf = GOAD(hidden_dim=hd, epochs=50, device="cuda:0", random_state=seed, verbose=0)
                clf.fit(train_sel)
                scores = clf.decision_function(cal_sel)
                try:
                    auc = roc_auc_score(cal_lbl, scores)
                    aucs.append(auc)
                except Exception:
                    pass
            if not aucs:
                continue
            mean_auc = float(np.mean(aucs))
            dataset_results[hd] = mean_auc
            print("  hidden_dim=%-4d cal_AUROC=%.4f (n_seeds=%d)" % (hd, mean_auc, len(aucs)))
            if mean_auc > best_auc:
                best_auc, best_hd = mean_auc, hd
        results[dataset] = dict(best_hidden_dim=best_hd, best_cal_auroc=best_auc, all=dataset_results)
        print("  -> SELECTED hidden_dim=%d for %s (cal_AUROC=%.4f)\n" % (best_hd, dataset, best_auc))

    print("\nFINAL PER-DATASET hidden_dim SELECTION:")
    for ds, r in results.items():
        print("  %-14s hidden_dim=%d" % (ds, r["best_hidden_dim"]))

    import json
    with open("results/goad_hidden_dim_tuning.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
