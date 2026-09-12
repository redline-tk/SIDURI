import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import journal_style as js

js.apply()
P = js.PALETTE
ETA = 0.05

ORIGINAL_STEPS = 800 * 175          # 140,000 -- horizon reported in Table (tab:control)
EXTENDED_STEPS = 10 * ORIGINAL_STEPS  # 1,400,000
WINDOW = 5000
N_TRIALS = 30

# ---------- shared trackers ----------

def run_sa_windowed(n_steps, rng, lr=0.01, tau0=1.645, window=WINDOW):
    tau = tau0
    n_windows = n_steps // window
    window_fpr = np.zeros(n_windows)
    count_in_window = 0
    flagged_in_window = 0
    w = 0
    for t in range(n_steps):
        s = float(rng.normal(0, 1))
        flagged = s > tau
        flagged_in_window += 1 if flagged else 0
        count_in_window += 1
        tau = tau + lr * ((1 if flagged else 0) - ETA)
        if count_in_window == window:
            window_fpr[w] = flagged_in_window / window
            w += 1
            count_in_window = 0
            flagged_in_window = 0
            if w >= n_windows:
                break
    return window_fpr

def run_sa_tail(n_steps, rng, lr=0.01, tau0=1.645):
    tau = tau0
    fpr_trace = []
    for _ in range(n_steps):
        s = float(rng.normal(0, 1))
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        tau = tau + lr * ((1 if flagged else 0) - ETA)
    return np.mean(fpr_trace[-2000:])

def run_ewm_windowed(n_steps, rng, lam=0.01, window=WINDOW):
    mean_est, var_est = 0.0, 1.0
    n_windows = n_steps // window
    window_fpr = np.zeros(n_windows)
    count_in_window = 0
    flagged_in_window = 0
    w = 0
    for t in range(n_steps):
        s = float(rng.normal(0, 1))
        std_est = np.sqrt(max(var_est, 1e-8))
        tau = mean_est + std_est * 1.645
        flagged = s > tau
        flagged_in_window += 1 if flagged else 0
        count_in_window += 1
        if not flagged:
            old_mean = mean_est
            mean_est = (1 - lam) * mean_est + lam * s
            var_est = (1 - lam) * var_est + lam * (s - old_mean) ** 2
        if count_in_window == window:
            window_fpr[w] = flagged_in_window / window
            w += 1
            count_in_window = 0
            flagged_in_window = 0
            if w >= n_windows:
                break
    return window_fpr

def run_ewm_tail(n_steps, rng, lam=0.01):
    mean_est, var_est = 0.0, 1.0
    fpr_trace = []
    for _ in range(n_steps):
        s = float(rng.normal(0, 1))
        std_est = np.sqrt(max(var_est, 1e-8))
        tau = mean_est + std_est * 1.645
        flagged = s > tau
        fpr_trace.append(1 if flagged else 0)
        if not flagged:
            old_mean = mean_est
            mean_est = (1 - lam) * mean_est + lam * s
            var_est = (1 - lam) * var_est + lam * (s - old_mean) ** 2
    return np.mean(fpr_trace[-2000:])

def trend_test(mean_trace):
    half = len(mean_trace) // 2
    y = mean_trace[half:]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat
    n = len(x)
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    sxx = np.sum((x - x.mean())**2)
    se_slope = s_err / np.sqrt(sxx)
    t_stat = slope / se_slope
    return slope, t_stat

# ---------- (D) extended-horizon check + figure ----------

print("=== Condition (D): extended-horizon stability check ===")
traces_d = [run_sa_windowed(EXTENDED_STEPS, np.random.default_rng(4000 + t))
            for t in range(N_TRIALS)]
traces_d = np.array(traces_d)
mean_d, sd_d = traces_d.mean(axis=0), traces_d.std(axis=0)
slope_d, t_d = trend_test(mean_d)
print(f"  final windowed FPR: {mean_d[-1]:.4f} +/- {sd_d[-1]:.4f}")
print(f"  second-half trend: slope={slope_d:.3e}, t-stat={t_d:.2f}")

xs_d = (np.arange(len(mean_d)) + 1) * WINDOW
fig, ax = plt.subplots(figsize=(6.0, 3.7))
ax.fill_between(xs_d, mean_d - sd_d, mean_d + sd_d, color=P["teal"], alpha=0.18)
ax.plot(xs_d, mean_d, color=P["teal"], linewidth=1.5)
ax.axvline(ORIGINAL_STEPS, color=P["accent"], linestyle=":", linewidth=1.0, alpha=0.8)
ax.set_xlabel("evaluation step")
ax.set_ylabel("windowed FPR")
ax.set_title("Condition (D): FPR over 10x the reported horizon", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("results/figures/figA_sa_stability.pdf")
fig.savefig("results/figures/figA_sa_stability.png")
print("  saved figA_sa_stability")

# ---------- (E) extended-horizon check + figure ----------

print("\n=== Condition (E): extended-horizon stability check ===")
traces_e = [run_ewm_windowed(EXTENDED_STEPS, np.random.default_rng(9000 + t))
            for t in range(N_TRIALS)]
traces_e = np.array(traces_e)
mean_e, sd_e = traces_e.mean(axis=0), traces_e.std(axis=0)
slope_e, t_e = trend_test(mean_e)
print(f"  final windowed FPR: {mean_e[-1]:.4f} +/- {sd_e[-1]:.4f}")
print(f"  second-half trend: slope={slope_e:.3e}, t-stat={t_e:.2f}")

xs_e = (np.arange(len(mean_e)) + 1) * WINDOW
fig, ax = plt.subplots(figsize=(6.0, 3.7))
ax.fill_between(xs_e, mean_e - sd_e, mean_e + sd_e, color=P["teal"], alpha=0.18)
ax.plot(xs_e, mean_e, color=P["teal"], linewidth=1.5)
ax.axvline(ORIGINAL_STEPS, color=P["accent"], linestyle=":", linewidth=1.0, alpha=0.8)
ax.set_xlabel("evaluation step")
ax.set_ylabel("windowed FPR")
ax.set_title("Condition (E): FPR over 10x the reported horizon", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("results/figures/figA_ewm_stability.pdf")
fig.savefig("results/figures/figA_ewm_stability.png")
print("  saved figA_ewm_stability")

# ---------- sensitivity sweeps ----------

print("\n=== Condition (D) sensitivity to learning rate ===")
for lr in [0.005, 0.01, 0.02, 0.05, 0.1]:
    vals = [run_sa_tail(ORIGINAL_STEPS, np.random.default_rng(7000 + int(lr*1000)*100 + t), lr)
            for t in range(10)]
    print(f"  lr={lr:<6} mean_fpr={np.mean(vals):.4f}  sd={np.std(vals):.4f}")

print("\n=== Condition (E) sensitivity to decay rate lambda ===")
for lam in [0.005, 0.01, 0.02, 0.05, 0.1]:
    vals = [run_ewm_tail(ORIGINAL_STEPS, np.random.default_rng(6000 + int(lam*1000)*100 + t), lam)
            for t in range(10)]
    print(f"  lambda={lam:<6} mean_fpr={np.mean(vals):.4f}  sd={np.std(vals):.4f}")
