import numpy as np

def conformal_pvalues(cal_scores, eval_scores):
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    eval_scores = np.asarray(eval_scores, dtype=np.float64)
    M = len(cal_scores)
    pvals = np.empty(len(eval_scores))
    for i, s in enumerate(eval_scores):
        count_ge = np.sum(cal_scores >= s)
        pvals[i] = (1.0 + count_ge) / (M + 1.0)
    return pvals

def benjamini_hochberg(pvals, q):
    pvals = np.asarray(pvals, dtype=np.float64)
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below)[0])
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[:max_idx + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject

def batch_fdp(pvals, labels, q, batch_size):
    pvals = np.asarray(pvals, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n = len(pvals)
    fdps = []
    n_alarms_per_batch = []
    n_skipped_no_anomaly = 0
    for start in range(0, n - batch_size + 1, batch_size):
        end = start + batch_size
        p_batch = pvals[start:end]
        y_batch = labels[start:end]
        if y_batch.sum() == 0:
            n_skipped_no_anomaly += 1
            continue
        reject = benjamini_hochberg(p_batch, q)
        n_alarms = int(np.sum(reject))
        if n_alarms == 0:
            fdps.append(0.0)
        else:
            n_false = int(np.sum(reject & (y_batch == 0)))
            fdps.append(n_false / n_alarms)
        n_alarms_per_batch.append(n_alarms)
    return np.array(fdps), np.array(n_alarms_per_batch), n_skipped_no_anomaly
