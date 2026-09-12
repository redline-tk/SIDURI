import bisect
import numpy as np

ETA = 0.05
M = 175
N_TRIALS = 30
TURNOVERS = 800
N_STEPS = TURNOVERS * M

def threshold_from_sorted(sorted_buf, eta):
    Mb = len(sorted_buf)
    level = min(np.ceil((Mb + 1) * (1 - eta)) / Mb, 1.0)
    idx = min(max(int(np.ceil(level * Mb)) - 1, 0), Mb - 1)
    return sorted_buf[idx]

def buffer_replace(fifo, sorted_buf, old_val, new_val):
    idx = bisect.bisect_left(sorted_buf, old_val)
    fifo.pop(0)
    sorted_buf.pop(idx)
    bisect.insort(sorted_buf, new_val)
    fifo.append(new_val)

# ---------- (A) Decision-gated finite buffer: the paper's mechanism, no contamination ----------
def run_decision_gated(n_steps, rng):
    fifo = list(rng.normal(0, 1, M))
    sorted_buf = sorted(fifo)
    fpr_trace = []
    for _ in range(n_steps):
        s = float(rng.normal(0, 1))
        tau = threshold_from_sorted(sorted_buf, ETA)
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        if not flagged:
            buffer_replace(fifo, sorted_buf, fifo[0], s)
    return np.mean(fpr_trace[-2000:])

# ---------- (B) Label-gated vs decision-gated, SAME contaminated stream ----------
def run_contaminated_pair(n_steps, rng, pi=0.05, anomaly_shift=3.0):
    fifo_dg = list(rng.normal(0, 1, M))
    sorted_dg = sorted(fifo_dg)
    fifo_lg = list(fifo_dg)
    sorted_lg = sorted(fifo_lg)
    fpr_dg, fpr_lg = [], []
    for _ in range(n_steps):
        is_anomaly = rng.uniform() < pi
        s = float(rng.normal(anomaly_shift, 1)) if is_anomaly else float(rng.normal(0, 1))

        tau_dg = threshold_from_sorted(sorted_dg, ETA)
        flagged_dg = s > tau_dg
        if not is_anomaly:
            fpr_dg.append(1 if flagged_dg else 0)
        if not flagged_dg:
            buffer_replace(fifo_dg, sorted_dg, fifo_dg[0], s)

        tau_lg = threshold_from_sorted(sorted_lg, ETA)
        flagged_lg = s > tau_lg
        if not is_anomaly:
            fpr_lg.append(1 if flagged_lg else 0)
        if not is_anomaly:
            buffer_replace(fifo_lg, sorted_lg, fifo_lg[0], s)
    return np.mean(fpr_dg[-2000:]), np.mean(fpr_lg[-2000:])

# ---------- (C) Random, decision-independent admission ----------
def run_random_admission(n_steps, rng, p_admit=0.5):
    fifo = list(rng.normal(0, 1, M))
    sorted_buf = sorted(fifo)
    fpr_trace = []
    for _ in range(n_steps):
        s = float(rng.normal(0, 1))
        tau = threshold_from_sorted(sorted_buf, ETA)
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        if rng.uniform() < p_admit:
            buffer_replace(fifo, sorted_buf, fifo[0], s)
    return np.mean(fpr_trace[-2000:])

# ---------- (D) Constant-step stochastic-approximation quantile tracker ----------
# Robbins-Monro update for P(S > tau) = ETA. No persistent memory: the flag
# outcome is used directly as a negative-feedback error signal. NOT expected
# to reproduce the buffer's drift, since there is no selection-biased state
# for the feedback to accumulate in.
def run_quantile_sa(n_steps, rng, lr=0.01, tau0=1.645):
    tau = tau0
    fpr_trace = []
    for _ in range(n_steps):
        s = float(rng.normal(0, 1))
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        tau = tau + lr * ((1 if flagged else 0) - ETA)
    return np.mean(fpr_trace[-2000:])

# ---------- (E) Exponentially weighted mean/variance tracker, decision-gated
# admission: a persistent, leaky memory (two EWM statistics), updated ONLY
# with points the current threshold does not flag. This shares the buffer's
# causal structure (a persistent state that selection can bias) with
# exponential forgetting instead of a finite FIFO window. Unlike the finite
# buffer, this state need not collapse toward the tail: the accepted stream
# is drawn from S | S <= tau, and the EWM state tracks the mean/variance of
# that truncated distribution, which can settle at a stable point with an
# elevated but non-unit FPR. Prediction is therefore FPR > ETA if
# selection-biased persistent memory is sufficient to induce calibration
# degradation; the magnitude need not approach 1.0 and is not predicted in
# advance.
def run_ewm_decision_gated(n_steps, rng, lam=0.01):
    mean_est = 0.0
    var_est = 1.0
    fpr_trace = []
    for _ in range(n_steps):
        s = float(rng.normal(0, 1))
        std_est = np.sqrt(max(var_est, 1e-8))
        tau = mean_est + std_est * 1.645  # normal-quantile approx at each step
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        if not flagged:
            old_mean = mean_est
            mean_est = (1 - lam) * mean_est + lam * s
            var_est = (1 - lam) * var_est + lam * (s - old_mean) ** 2
    return np.mean(fpr_trace[-2000:])

def main():
    print("=== (A) Decision-gated finite buffer (paper's mechanism), no contamination ===")
    print("Prediction: FPR -> 1")
    results_a = [run_decision_gated(N_STEPS, np.random.default_rng(1000 + t)) for t in range(N_TRIALS)]
    print("  mean_fpr=%.4f (theory=1.0)" % np.mean(results_a))

    print()
    print("=== (B) Label-gated vs decision-gated, SAME 5%% contaminated stream ===")
    print("Prediction: label-gated stays near eta=0.05, decision-gated drifts unsafe")
    dg_results, lg_results = [], []
    for t in range(N_TRIALS):
        dg, lg = run_contaminated_pair(N_STEPS, np.random.default_rng(2000 + t))
        dg_results.append(dg); lg_results.append(lg)
    print("  decision-gated mean_fpr=%.4f" % np.mean(dg_results))
    print("  label-gated    mean_fpr=%.4f (theory~0.05)" % np.mean(lg_results))

    print()
    print("=== (C) Random, decision-independent admission (p_admit=0.5), no contamination ===")
    print("Prediction: no systematic drift, FPR stays near eta=0.05")
    results_c = [run_random_admission(N_STEPS, np.random.default_rng(3000 + t)) for t in range(N_TRIALS)]
    print("  mean_fpr=%.4f (theory~0.05)" % np.mean(results_c))

    print()
    print("=== (D) Constant-step stochastic-approximation quantile tracker ===")
    print("Prediction: stabilizes near ETA=0.05; this is NOT expected to reproduce the buffer drift")
    results_d = [run_quantile_sa(N_STEPS, np.random.default_rng(4000 + t)) for t in range(N_TRIALS)]
    print("  mean_fpr=%.4f" % np.mean(results_d))

    print()
    print("=== (E) Exponentially weighted mean/variance tracker, decision-gated admission ===")
    print("(persistent, leaky, selection-biased memory -- not a finite FIFO buffer)")
    print("Prediction: FPR should exceed ETA if selection-biased persistent memory")
    print("            is sufficient to induce calibration degradation; magnitude")
    print("            need not approach 1.0 and is not predicted in advance.")
    results_e = [run_ewm_decision_gated(N_STEPS, np.random.default_rng(5000 + t)) for t in range(N_TRIALS)]
    print("  mean_fpr=%.4f" % np.mean(results_e))

if __name__ == "__main__":
    main()
