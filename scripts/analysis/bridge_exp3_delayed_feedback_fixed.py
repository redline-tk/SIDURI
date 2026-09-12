import bisect
import numpy as np

eta = 0.05

def threshold_from_sorted(sorted_buf, eta):
    M = len(sorted_buf)
    level = min(np.ceil((M+1)*(1-eta))/M, 1.0)
    idx = min(max(int(np.ceil(level*M))-1, 0), M-1)
    return sorted_buf[idx]

def run_delayed_trial(M, q, d, n_steps, rng):
    fifo = list(rng.normal(0, 1, M))
    sorted_buf = sorted(fifo)
    pending = {}
    fpr_trace = []
    eval_window = min(2000, n_steps // 4)
    for t in range(n_steps):
        tau = threshold_from_sorted(sorted_buf, eta)
        s = float(rng.normal(0, 1))
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        if not flagged:
            oldest = fifo.pop(0)
            idx = bisect.bisect_left(sorted_buf, oldest)
            sorted_buf.pop(idx)
            bisect.insort(sorted_buf, s)
            fifo.append(s)
        else:
            if rng.uniform() < q:
                arrival = t + d
                pending.setdefault(arrival, []).append(s)
        if t in pending:
            for val in pending.pop(t):
                oldest = fifo.pop(0)
                idx = bisect.bisect_left(sorted_buf, oldest)
                sorted_buf.pop(idx)
                bisect.insort(sorted_buf, val)
                fifo.append(val)
    return np.array(fpr_trace)

def main():
    M = 175
    N_TRIALS = 30
    TURNOVERS = 150
    n_steps = TURNOVERS * M

    print("EXPERIMENT 3 (CORRECTED): delayed feedback, M=%d, n_steps=%d, N_TRIALS=%d" % (M, n_steps, N_TRIALS))
    print()

    print("=== PART A (fixed): does steady-state FPR depend on delay d? (q=0.5) ===")
    q = 0.5
    theory_fpr = eta / (eta + (1-eta)*q)
    for d in [0, 1, 5, 20, 50, 100, 300]:
        fprs = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(hash((M, q, d, "fixed", trial)) % (2**32))
            trace = run_delayed_trial(M, q, d, n_steps, rng)
            fprs.append(trace[-2000:].mean())
        emp = np.mean(fprs)
        print("  d=%-4d  theory=%.4f  empirical=%.4f  bias=%+.4f" % (d, theory_fpr, emp, emp - theory_fpr))

if __name__ == "__main__":
    main()
