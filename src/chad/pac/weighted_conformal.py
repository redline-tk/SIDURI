import numpy as np
from scipy.stats import ks_2samp

def log_weights_uniform(M):
    return np.zeros(M)

def log_weights_exponential(M, rho):
    ages = np.arange(M - 1, -1, -1, dtype=np.float64)
    return ages * np.log(rho)

def weighted_threshold(cal_scores, eta, log_w):
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    M = len(cal_scores)
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w_sum = np.sum(w)
    w_tilde = w / (w_sum + 1.0)
    mass_test = 1.0 / (w_sum + 1.0)
    order = np.argsort(cal_scores)
    sorted_scores = cal_scores[order]
    sorted_w = w_tilde[order]
    cum = np.cumsum(sorted_w)
    level = 1.0 - eta
    idx = np.searchsorted(cum, level)
    if idx >= M:
        return np.inf, M_eff_from_weights(w_tilde)
    return float(sorted_scores[idx]), M_eff_from_weights(w_tilde)

def M_eff_from_weights(w_tilde):
    s1 = np.sum(w_tilde)
    s2 = np.sum(w_tilde ** 2)
    if s2 <= 0:
        return 0.0
    return float((s1 ** 2) / s2)

def delta_hat(cal_scores, n_bins=20, n_splits=1):
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    M = len(cal_scores)
    if M < 10:
        return 0.0
    half = M // 2
    older = cal_scores[:half]
    newer = cal_scores[half:]
    stat, pval = ks_2samp(older, newer)
    pooled = cal_scores
    edges = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    h_old, _ = np.histogram(older, bins=edges)
    h_new, _ = np.histogram(newer, bins=edges)
    h_old = h_old / max(h_old.sum(), 1)
    h_new = h_new / max(h_new.sum(), 1)
    tv = 0.5 * np.sum(np.abs(h_old - h_new))
    return dict(ks_stat=float(stat), ks_pval=float(pval), tv=float(tv))
