import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import run_extended_methods as rem

def report(dataset_name, units, is_per_unit=False):
    print("=== %s ===" % dataset_name)
    if is_per_unit:
        raw_counts = []
        post_counts = []
        for unit_id, train_raw, train_labels, test_raw, test_labels in units:
            raw_counts.append(train_raw.shape[1])
            keep, _ = rem.feature_prescreen(train_raw.astype(np.float32))
            post_counts.append(int(keep.sum()))
        print("  n_units=%d" % len(units))
        print("  raw feature count: min=%d max=%d (consistent=%s)" % (
            min(raw_counts), max(raw_counts), len(set(raw_counts)) == 1))
        print("  post-prescreen feature count: min=%d max=%d mean=%.1f" % (
            min(post_counts), max(post_counts), np.mean(post_counts)))
    else:
        unit_id, train_raw, train_labels, test_raw, test_labels = units[0]
        raw = train_raw.shape[1]
        keep, _ = rem.feature_prescreen(train_raw.astype(np.float32))
        print("  n_units=%d  raw_features=%d  post_prescreen_features=%d" % (
            len(units), raw, int(keep.sum())))
    print()

report("SMD", rem.load_smd_units(), is_per_unit=True)
report("SMAP", rem.load_smap_msl_units("smap"), is_per_unit=True)
report("MSL", rem.load_smap_msl_units("msl"), is_per_unit=True)
report("BATADAL", rem.load_batadal_units(), is_per_unit=False)
report("CICIDS2017", rem.load_cicids_units(), is_per_unit=True)
report("UNSW_NB15", rem.load_unsw_units(), is_per_unit=False)
