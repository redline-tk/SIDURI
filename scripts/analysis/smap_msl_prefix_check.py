import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import Counter
import run_extended_methods as rem
import pandas as pd

labels_path = None
for candidate in Path("data/raw/smap_msl").rglob("labeled_anomalies.csv"):
    labels_path = candidate
    break
df = pd.read_csv(labels_path)
truth = dict(zip(df["chan_id"], df["spacecraft"]))

smap_call = rem.load_smap_msl_units("smap")
msl_call = rem.load_smap_msl_units("msl")

def prefix(unit_id):
    m = re.match(r"([A-Za-z]+)-", unit_id)
    return m.group(1) if m else unit_id

print("=== units returned by 'smap' call, by TRUE spacecraft label ===")
smap_true_labels = Counter(truth.get(u[0], "UNKNOWN") for u in smap_call)
print(smap_true_labels)
print("prefixes in 'smap' call:", Counter(prefix(u[0]) for u in smap_call))

print()
print("=== units returned by 'msl' call, by TRUE spacecraft label ===")
msl_true_labels = Counter(truth.get(u[0], "UNKNOWN") for u in msl_call)
print(msl_true_labels)
print("prefixes in 'msl' call:", Counter(prefix(u[0]) for u in msl_call))

print()
print("MSL channels missing from BOTH calls entirely:")
smap_ids = set(u[0] for u in smap_call)
msl_ids = set(u[0] for u in msl_call)
all_returned = smap_ids | msl_ids
true_msl_ids = set(df[df["spacecraft"]=="MSL"]["chan_id"])
print(sorted(true_msl_ids - all_returned))
