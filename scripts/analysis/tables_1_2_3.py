import json

dataset_chars = [
    ("SMD", "Server telemetry", 28, "38 metrics/machine", 0.0536, 0.679, "n_ge_175=54/168"),
    ("SMAP", "Spacecraft telemetry", 55, "25 features", 0.0063, 0.028, "0/429 units reach M=175; D-12/D-13/G-7 excluded (constant train features)"),
    ("MSL", "Spacecraft telemetry", 27, "55 features", 0.0, 0.0, "structurally too short (12-14 raw cal length); excluded from decay-curve work"),
    ("BATADAL", "Water treatment SCADA", 1, "43 sensors/actuators", 0.0, 0.0, "n_eval~110, too short for per-unit robustness checks"),
    ("CICIDS2017", "Network flow", 7, "~78 flow features", 0.2821, 0.714, "day-file granularity; severe cal contamination, worsens under recency correction"),
    ("UNSW_NB15", "Network flow", 1, "~40 numeric features", 0.9872, 1.0, "single unit; near-total cal contamination"),
]

print("TABLE 1: Dataset characteristics")
print("%-12s %-22s %-8s %-16s %-10s %-10s" % ("dataset", "domain", "n_units", "features", "mean_contam", "frac>1%_contam"))
for name, domain, n, feats, mean_c, frac_c, note in dataset_chars:
    print("%-12s %-22s %-8d %-16s %-10.4f %-10.3f" % (name, domain, n, feats, mean_c, frac_c))

print()
print("TABLE 1b: notes column (for caption)")
for name, domain, n, feats, mean_c, frac_c, note in dataset_chars:
    print("  %s: %s" % (name, note))

detector_panel = [
    ("DeepIsolationForest", "reconstruction/isolation", "handles high-dim tabular"),
    ("NeuTraL", "learned transformations", "ICML 2021, strong on BATADAL/UNSW"),
    ("GOAD", "geometric transformation", "hidden_dim bug found+fixed on UNSW this session"),
    ("SLAD", "contrastive/supervisory signals", "best on ADBench generally per source paper"),
    ("RDP", "random distance prediction", "deepod library"),
    ("ICL", "internal contrastive learning", "strongest UNSW reproduction vs published (gap 0.018)"),
]
print()
print("TABLE 2: Detector panel (6 of 12 total methods used in the extended/decay-curve analysis; DIF alias for DeepIsolationForest)")
print("%-22s %-28s %s" % ("method", "paradigm", "note"))
for name, paradigm, note in detector_panel:
    print("%-22s %-28s %s" % (name, paradigm, note))
