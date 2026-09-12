import numpy as np
from scipy.stats import beta

def pac_k_star(M, eps, delta):
    for k in range(1, M + 1):
        if beta.ppf(1 - delta, k, M + 1 - k) <= eps:
            continue
        return k - 1 if k > 1 else None
    return M

def pac_threshold(cal_scores, eps, delta):
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    M = len(cal_scores)
    k_star = pac_k_star(M, eps, delta)
    if k_star is None or k_star < 1:
        return None, None, False
    sorted_scores = np.sort(cal_scores)
    q_hat = sorted_scores[M - k_star]
    return float(q_hat), int(k_star), True

def m_min(eps, delta):
    return int(np.ceil(np.log(delta) / np.log(1 - eps)))
