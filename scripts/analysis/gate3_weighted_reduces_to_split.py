import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.weighted_conformal import weighted_threshold, log_weights_uniform

rng = np.random.default_rng(3)

def split_conformal_threshold(cal_scores, eta):
    M = len(cal_scores)
    level = np.ceil((M + 1) * (1.0 - eta)) / M
    level = min(level, 1.0)
    idx = int(np.ceil(level * M)) - 1
    idx = min(max(idx, 0), M - 1)
    return np.sort(cal_scores)[idx]

def main():
    all_pass = True
    for M in [25, 50, 100, 175, 300]:
        for eta in [0.01, 0.05, 0.10, 0.20]:
            for trial in range(50):
                cal = rng.normal(0, 1, M)
                tau_split = split_conformal_threshold(cal, eta)
                log_w = log_weights_uniform(M)
                tau_weighted, m_eff = weighted_threshold(cal, eta, log_w)
                ok = np.isclose(tau_split, tau_weighted, atol=1e-6) or (np.isinf(tau_weighted) and tau_split == np.sort(cal)[-1])
                if not ok:
                    all_pass = False
                    print("MISMATCH M=%d eta=%.2f trial=%d split=%.6f weighted=%.6f" % (
                        M, eta, trial, tau_split, tau_weighted))
    print("GATE 3 (weighted reduces to split at uniform weights): %s" % ("PASS" if all_pass else "FAIL"))

if __name__ == "__main__":
    main()
