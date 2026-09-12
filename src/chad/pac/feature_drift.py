import numpy as np

def window_feature_means(raw_features, window_starts, window_size):
    means = []
    for s in window_starts:
        w = raw_features[s:s + window_size]
        means.append(w.mean(axis=0))
    return np.array(means)

def delta_hat_feat(feature_means, pooled_std):
    M = len(feature_means)
    if M < 10:
        return 0.0
    half = M // 2
    older = feature_means[:half]
    newer = feature_means[half:]
    shift = np.abs(newer.mean(axis=0) - older.mean(axis=0))
    std_safe = np.where(pooled_std > 1e-8, pooled_std, 1.0)
    standardized = shift / std_safe
    return float(np.mean(standardized))

def conformal_pvalues_r(cal_scores, eval_scores):
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    eval_scores = np.asarray(eval_scores, dtype=np.float64)
    M = len(cal_scores)
    pvals = np.empty(len(eval_scores))
    for i, s in enumerate(eval_scores):
        count_ge = np.sum(cal_scores >= s)
        pvals[i] = (1.0 + count_ge) / (M + 1.0)
    return pvals

def delta_hat_disp(cal_scores, eval_scores):
    pvals = conformal_pvalues_r(cal_scores, eval_scores)
    denom = np.sum(pvals > 0.25)
    if denom == 0:
        return None
    numer = np.sum(pvals > 0.50)
    R = numer / denom
    return float(2.0 / 3.0 - R)
