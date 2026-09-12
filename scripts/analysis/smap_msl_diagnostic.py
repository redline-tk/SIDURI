import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import run_extended_methods as rem

# 1. Ground truth from the NASA labeled_anomalies.csv, independent of any loader
labels_path = None
for candidate in Path("data/raw/smap_msl").rglob("labeled_anomalies.csv"):
    labels_path = candidate
    break
if labels_path:
    df = pd.read_csv(labels_path)
    print("labeled_anomalies.csv found at:", labels_path)
    print("spacecraft column value counts:")
    print(df["spacecraft"].value_counts())
    print()
    print("SMAP chan_id count:", (df["spacecraft"] == "SMAP").sum())
    print("MSL chan_id count:", (df["spacecraft"] == "MSL").sum())
else:
    print("labeled_anomalies.csv NOT FOUND under data/raw/smap_msl")

print()
print("=== What rem.load_smap_msl_units('smap') actually returns ===")
smap_units = rem.load_smap_msl_units("smap")
print("n=%d" % len(smap_units))
print("first 5 unit_ids:", [u[0] for u in smap_units[:5]])
print("last 5 unit_ids:", [u[0] for u in smap_units[-5:]])
feat_dims = sorted(set(u[1].shape[1] for u in smap_units))
print("distinct raw feature dims seen:", feat_dims)

print()
print("=== What rem.load_smap_msl_units('msl') actually returns ===")
msl_units = rem.load_smap_msl_units("msl")
print("n=%d" % len(msl_units))
print("first 5 unit_ids:", [u[0] for u in msl_units[:5]])
print("last 5 unit_ids:", [u[0] for u in msl_units[-5:]])
feat_dims2 = sorted(set(u[1].shape[1] for u in msl_units))
print("distinct raw feature dims seen:", feat_dims2)
