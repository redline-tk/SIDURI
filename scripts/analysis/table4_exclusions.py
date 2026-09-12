from collections import Counter
from pathlib import Path

log_path = Path("results/run_phase1.log")
reasons = Counter()
if log_path.exists():
    for line in open(log_path):
        if "EXCLUDING" in line:
            if "usable features" in line:
                reasons["zero/insufficient usable features (constant/near-constant train columns)"] += 1
            else:
                reasons["other exclusion"] += 1

print("TABLE 4: Exclusion summary (from results/run_phase1.log)")
for reason, count in reasons.most_common():
    print("  %-70s %d" % (reason, count))

nan_units = []
if log_path.exists():
    for line in open(log_path):
        if "nan cal" in line or "nan scores" in line.lower():
            nan_units.append(line.strip())

print()
print("NaN-related exclusions/warnings found in log: %d" % len(nan_units))
for line in nan_units[:10]:
    print("  " + line)
