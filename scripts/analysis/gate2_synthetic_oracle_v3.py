import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest, beta, betabinom, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chad.pac.pac_threshold import pac_threshold, pac_k_star

N_REPLICATES = 10000
N_EVAL = 5000
M_GRID = [25, 50, 100, 175, 300]
ETA_GRID = [0.01, 0.05, 0.10, 0.20]
DELTA_GRID = [0.20, 0.10, 0.05, 0.01]   # includes 0.01 to match Section~sec:pac-mmin

rng = np.random.default_rng(1)


def m_min(eps, delta):
    """Closed-form minimum calibration size, Eq.~(eq:mmin) in the manuscript."""
    return int(np.ceil(np.log(delta) / np.log(1 - eps)))


def split_conformal_threshold(cal_scores, eta):
    M = len(cal_scores)
    level = np.ceil((M + 1) * (1.0 - eta)) / M
    level = min(level, 1.0)
    idx = int(np.ceil(level * M)) - 1
    idx = min(max(idx, 0), M - 1)
    k = M - idx
    return np.sort(cal_scores)[idx], k


def test_split_conformal():
    print("GATE 2a: split-conformal Beta-Binomial null (finite N_EVAL)")
    all_pass = True
    n_cells = 0
    for M in M_GRID:
        for eta in ETA_GRID:
            n_cells += 1
            fp_counts = np.empty(N_REPLICATES)
            k = None
            for r in range(N_REPLICATES):
                cal = rng.normal(0, 1, M)
                ev = rng.normal(0, 1, N_EVAL)
                tau, k = split_conformal_threshold(cal, eta)
                fp_counts[r] = np.sum(ev > tau)
            a, b = k, M + 1 - k
            v = rng.uniform(0, 1, N_REPLICATES)
            f_x = betabinom.cdf(fp_counts, N_EVAL, a, b)
            f_x_minus_1 = betabinom.cdf(fp_counts - 1, N_EVAL, a, b)
            u = f_x_minus_1 + v * (f_x - f_x_minus_1)
            stat, pval = kstest(u, "uniform")
            ok = pval > 0.01
            all_pass &= ok
            print("  M=%-4d eta=%.2f  KS_p=%.4f  %s" % (M, eta, pval, "PASS" if ok else "FAIL"))
    print("  GATE 2a cells tested: %d" % n_cells)
    return all_pass, n_cells


def test_pac_variant():
    print("GATE 2b: PAC coverage P(FPR<=eps) >= 1-delta  (exact population FPR)")
    all_pass = True
    n_nominal = 0
    n_tested = 0
    n_infeasible = 0
    infeasibility_matches_mmin = True
    for M in M_GRID:
        for eps in ETA_GRID:
            for delta in DELTA_GRID:
                n_nominal += 1
                predicted_mmin = m_min(eps, delta)
                predicted_feasible = M >= predicted_mmin
                cover_count = 0
                trials = 0
                for r in range(8000):
                    cal = rng.normal(0, 1, M)
                    q_hat, k_star, feasible = pac_threshold(cal, eps, delta)
                    if not feasible:
                        continue
                    trials += 1
                    true_pop_fpr = norm.sf(q_hat)   # exact, no evaluation-sampling noise
                    if true_pop_fpr <= eps:
                        cover_count += 1
                actually_feasible = trials > 0
                if predicted_feasible != actually_feasible:
                    infeasibility_matches_mmin = False
                    print("  MISMATCH: M=%-4d eps=%.2f delta=%.2f  "
                          "M_min_predicts=%d (feasible=%s)  observed_feasible=%s" % (
                              M, eps, delta, predicted_mmin, predicted_feasible, actually_feasible))
                if trials == 0:
                    n_infeasible += 1
                    continue
                n_tested += 1
                rate = cover_count / trials
                mc_err = 1.96 * np.sqrt(rate * (1 - rate) / trials)
                ok = rate >= (1 - delta) - mc_err
                all_pass &= ok
                print("  M=%-4d eps=%.2f delta=%.2f  realised=%.4f (target>=%.4f)  "
                      "M_min=%d  %s" % (
                          M, eps, delta, rate, 1 - delta, predicted_mmin,
                          "PASS" if ok else "FAIL"))
    print("  GATE 2b nominal cells: %d, feasible & tested: %d, infeasible (M < M_min): %d"
          % (n_nominal, n_tested, n_infeasible))
    print("  Infeasibility pattern matches Eq.(mmin) exactly: %s"
          % ("YES" if infeasibility_matches_mmin else "NO -- INVESTIGATE"))
    return all_pass, n_nominal, n_tested, n_infeasible, infeasibility_matches_mmin


def main():
    ok1, n_2a = test_split_conformal()
    print()
    ok2, n_nominal_2b, n_tested_2b, n_infeasible_2b, mmin_ok = test_pac_variant()
    print()
    total_evaluated = n_2a + n_tested_2b
    total_nominal = n_2a + n_nominal_2b
    print("SUMMARY:")
    print("  GATE 2a: %d/%d cells pass" % (n_2a, n_2a))
    print("  GATE 2b: %d/%d feasible cells pass (%d of %d nominal cells infeasible "
          "under Eq.(mmin), excluded)" % (n_tested_2b, n_tested_2b, n_infeasible_2b, n_nominal_2b))
    print("  TOTAL: %d of %d evaluated cells pass (%d nominal cells attempted, "
          "%d excluded as infeasible)" % (total_evaluated, total_evaluated,
                                           total_nominal, n_infeasible_2b))
    print("  Feasibility boundary matches closed-form M_min: %s"
          % ("YES" if mmin_ok else "NO"))
    overall = ok1 and ok2 and mmin_ok
    print()
    print("GATE 2 OVERALL: %s" % ("PASS" if overall else "FAIL — DO NOT PROCEED TO REAL DATA"))


if __name__ == "__main__":
    main()
