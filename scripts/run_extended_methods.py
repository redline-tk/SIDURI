import argparse, logging, sys, json, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/redline/files/TSB-AD")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("extended_methods")

CACHE_DIR = Path("results/score_cache")

def config_hash(cfg):
    import hashlib
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

def score_stats(s_cal, s_eval, y_eval):
    n_nan = int(np.isnan(s_cal).sum() + np.isnan(s_eval).sum())
    n_inf = int(np.isinf(s_cal).sum() + np.isinf(s_eval).sum())
    n_total = len(s_cal) + len(s_eval)
    n_distinct_cal = int(len(np.unique(s_cal[np.isfinite(s_cal)])))
    orientation = "unknown"
    if len(np.unique(y_eval)) > 1 and np.isfinite(s_eval).all():
        from sklearn.metrics import roc_auc_score
        try:
            a = roc_auc_score(y_eval, s_eval)
            orientation = "flipped" if a < 0.5 else "normal"
        except Exception:
            orientation = "unknown"
    return n_nan, n_inf, n_total, n_distinct_cal, orientation

def cache_scores(dataset_name, method_name, unit_id, s_cal, y_cal, s_eval, y_eval,
                  cal_window_start=None, eval_window_start=None, train_scores=None,
                  window=None, stride=None, cal_frac=None,
                  n_features_raw=None, n_features_dropped=0, dropped_feature_idx=None,
                  detector_seed=0):
    if window is None:
        window = WINDOW
    if stride is None:
        stride = STRIDE
    if cal_frac is None:
        cal_frac = CAL_FRAC
    n_nan, n_inf, n_total, n_distinct_cal, orientation = score_stats(s_cal, s_eval, y_eval)
    if n_total > 0 and n_nan / n_total > 0.001:
        logger.warning("    %s/%s: %d/%d nan scores (%.3f%%), EXCLUDING unit from cache",
                        method_name, unit_id, n_nan, n_total, 100.0 * n_nan / n_total)
        return False
    try:
        out = CACHE_DIR / dataset_name
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{unit_id}__{method_name}"
        arrays = dict(cal_scores=s_cal, cal_labels=y_cal.astype(np.int8),
                      eval_scores=s_eval, eval_labels=y_eval.astype(np.int8))
        if cal_window_start is not None:
            arrays["cal_window_start"] = cal_window_start
        if eval_window_start is not None:
            arrays["eval_window_start"] = eval_window_start
        if train_scores is not None:
            arrays["train_scores"] = train_scores
        np.savez_compressed(out / f"{stem}.npz", **arrays)
        cfg = dict(dataset=dataset_name, unit_id=unit_id, method=method_name,
                    window=window, stride=stride, cal_frac=cal_frac,
                    n_features_raw=n_features_raw, n_features_dropped=n_features_dropped,
                    dropped_feature_idx=[int(i) for i in dropped_feature_idx] if dropped_feature_idx is not None else [],
                    n_nan_in_scores=n_nan, n_inf_in_scores=n_inf,
                    n_distinct_cal_scores=n_distinct_cal,
                    score_min=float(np.nanmin(s_cal)) if len(s_cal) else None,
                    score_max=float(np.nanmax(s_cal)) if len(s_cal) else None,
                    score_orientation=orientation,
                    detector_seed=detector_seed, detector_version="deepod",
                    source_run="extended_methods_cached_phase1")
        cfg["config_hash"] = config_hash(cfg)
        import time
        cfg["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(out / f"{stem}.json", "w") as f:
            json.dump(cfg, f, indent=2)
        if orientation == "flipped":
            logger.warning("    %s/%s: AUROC < 0.5, flagged score_orientation=flipped, NOT negated",
                            method_name, unit_id)
        return True
    except Exception as e:
        logger.warning("    cache_scores failed for %s/%s: %s", dataset_name, unit_id, e)
        return False

def feature_prescreen(X_train):
    global_std = np.std(X_train) + 1e-12
    col_std = np.std(X_train, axis=0)
    keep = col_std > 1e-8 * global_std
    dropped_idx = np.where(~keep)[0]
    return keep, dropped_idx

def aggregate_to_windows_with_starts(point_scores, point_labels, window=None, stride=None):
    if window is None:
        window = WINDOW
    if stride is None:
        stride = STRIDE
    s_win, y_win, starts = [], [], []
    n = len(point_scores)
    for start in range(0, n - window + 1, stride):
        end = start + window
        s_win.append(float(point_scores[start:end].max()))
        y_win.append(int(point_labels[start:end].max()))
        starts.append(start)
    return np.array(s_win), np.array(y_win), np.array(starts, dtype=np.int64)

ETA_GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
M_GRID   = [25, 50, 75, 100, 125, 150, 175]
CAL_FRAC = 0.20
WINDOW   = 60
STRIDE   = 30
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"


def aggregate_to_windows(point_scores, point_labels):
    s_win, y_win = [], []
    n = len(point_scores)
    for start in range(0, n - WINDOW + 1, STRIDE):
        end = start + WINDOW
        s_win.append(float(point_scores[start:end].max()))
        y_win.append(int(point_labels[start:end].max()))
    return np.array(s_win), np.array(y_win)


def confusion(flags, y):
    flags, y = np.array(flags, dtype=int), np.array(y, dtype=int)
    tp = int(np.sum((flags==1)&(y==1)))
    fp = int(np.sum((flags==1)&(y==0)))
    fn = int(np.sum((flags==0)&(y==1)))
    tn = int(np.sum((flags==0)&(y==0)))
    return tp, fp, fn, tn


def eta_sweep(method_name, s_cal, y_cal, s_eval, y_eval, M=50):
    from src.chad.calibrator import MachineAdaptiveCalibrator
    s_cal  = np.array(s_cal,  dtype=np.float64)
    s_eval = np.array(s_eval, dtype=np.float64)
    y_cal  = np.array(y_cal,  dtype=int)
    y_eval = np.array(y_eval, dtype=int)
    cal_normals = s_cal[y_cal == 0]
    results = {}

    try:
        auroc = float(roc_auc_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")

    best_f1, best_t = 0.0, float(np.median(s_cal))
    for t in np.linspace(s_cal.min(), s_cal.max(), 300):
        f1 = f1_score(y_cal, (s_cal >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    flags_raw = (s_eval >= best_t).astype(int)
    tp, fp, fn, tn = confusion(flags_raw, y_eval)
    results["auroc"] = auroc
    results["raw"] = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp/max(tp+fp, 1),
        "recall":    tp/max(tp+fn, 1),
        "f1":        float(f1_score(y_eval, flags_raw, zero_division=0)),
        "fpr":       fp/max(fp+tn, 1),
        "coverage_ok": None,
    }

    if len(cal_normals) < 5:
        return results

    best_M_f1, best_M = 0.0, M
    m_results = {}
    for m_val in M_GRID:
        if m_val > len(cal_normals):
            continue
        mac = MachineAdaptiveCalibrator(eta=0.05, buffer_size=m_val)
        for s in cal_normals[:m_val]:
            mac.update(method_name, float(s))
        flags = []
        for i, s in enumerate(s_eval):
            flags.append(mac.predict(method_name, float(s)))
            if y_eval[i] == 0:
                mac.update(method_name, float(s))
        flags = np.array(flags, dtype=int)
        tp, fp, fn, tn = confusion(flags, y_eval)
        fpr = fp/max(fp+tn, 1)
        bound = 0.05 + 1.0/(m_val+1)
        f1_val = float(f1_score(y_eval, flags, zero_division=0))
        cov_ok = fpr <= bound + 0.005
        m_results[str(m_val)] = {
            "f1": f1_val, "fpr": fpr, "bound": float(bound),
            "recall": tp/max(tp+fn, 1), "coverage_ok": cov_ok,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        if cov_ok and f1_val > best_M_f1:
            best_M_f1, best_M = f1_val, m_val

    results["M_grid"] = m_results
    results["best_M"] = best_M
    results["best_f1"] = best_M_f1

    for eta in ETA_GRID:
        mac = MachineAdaptiveCalibrator(eta=eta, buffer_size=best_M)
        for s in cal_normals[:best_M]:
            mac.update(method_name, float(s))
        flags = []
        for i, s in enumerate(s_eval):
            flags.append(mac.predict(method_name, float(s)))
            if y_eval[i] == 0:
                mac.update(method_name, float(s))
        flags = np.array(flags, dtype=int)
        tp, fp, fn, tn = confusion(flags, y_eval)
        fpr   = fp/max(fp+tn, 1)
        bound = eta + 1.0/(best_M+1)
        results["eta_%.2f" % eta] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/max(tp+fp, 1),
            "recall":    tp/max(tp+fn, 1),
            "f1":        float(f1_score(y_eval, flags, zero_division=0)),
            "fpr":       fpr,
            "bound":     bound,
            "coverage_ok": bool(fpr <= bound + 0.005),
            "M": best_M,
        }
    return results


class DQNNet(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, x):
        return self.net(x)


def train_dqn(train_raw, train_labels, input_dim, epochs=30, batch_size=512, lr=1e-3):
    model = DQNNet(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    X = torch.from_numpy(train_raw.astype(np.float32)).to(DEVICE)
    y = torch.from_numpy(train_labels.astype(np.int64)).to(DEVICE)
    n = len(X)
    for epoch in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            q_vals = model(xb)
            actions = q_vals.argmax(dim=1)
            rewards = (actions == yb).float() * 2 - 1
            q_taken = q_vals.gather(1, yb.unsqueeze(1)).squeeze()
            loss = (-rewards * q_taken).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def score_dqn(model, data):
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(data.astype(np.float32)).to(DEVICE)
        q_vals = model(X)
        scores = torch.softmax(q_vals, dim=1)[:, 1].cpu().numpy()
    return scores


def run_deep_isolation_forest(train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name="unknown"):
    from deepod.models import DeepIsolationForest
    try:
        train_clean = np.nan_to_num(train_raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        cal_clean   = np.nan_to_num(cal_raw.astype(np.float32),   nan=0.0, posinf=0.0, neginf=0.0)
        eval_clean  = np.nan_to_num(eval_raw.astype(np.float32),  nan=0.0, posinf=0.0, neginf=0.0)
        clf = DeepIsolationForest(epochs=50, device=DEVICE)
        clf.fit(train_clean)
        s_cal_pt  = clf.decision_function(cal_clean)
        s_eval_pt = clf.decision_function(eval_clean)
        del clf
        import torch
        torch.cuda.empty_cache()
        s_cal_w,  y_cal_w  = aggregate_to_windows(s_cal_pt,  cal_lbl)
        s_eval_w, y_eval_w = aggregate_to_windows(s_eval_pt, eval_lbl)
        cache_scores(dataset_name, "DIF", unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w)
        result = eta_sweep(f"DIF_{unit_id}", s_cal_w, y_cal_w, s_eval_w, y_eval_w, M=50)
        logger.info("    DIF AUROC=%.4f raw_FPR=%.4f", result.get("auroc", float("nan")), result["raw"]["fpr"])
        return result
    except Exception as e:
        logger.warning("    DIF failed: %s", e)
        return {}


def run_supervised(method_name, clf, train_raw, train_labels,
                   cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name="unknown"):
    try:
        clf.fit(train_raw, train_labels)
        s_cal_pt  = clf.predict_proba(cal_raw)[:, 1]
        s_eval_pt = clf.predict_proba(eval_raw)[:, 1]
        s_cal_w,  y_cal_w  = aggregate_to_windows(s_cal_pt,  cal_lbl)
        s_eval_w, y_eval_w = aggregate_to_windows(s_eval_pt, eval_lbl)
        cache_scores(dataset_name, method_name, unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w)
        result = eta_sweep(f"{method_name}_{unit_id}", s_cal_w, y_cal_w, s_eval_w, y_eval_w, M=50)
        logger.info("    %s AUROC=%.4f raw_FPR=%.4f", method_name,
                    result.get("auroc", float("nan")), result["raw"]["fpr"])
        return result
    except Exception as e:
        logger.warning("    %s failed: %s", method_name, e)
        return {}


def run_dqn(train_raw, train_labels, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name="unknown"):
    try:
        input_dim = train_raw.shape[1]
        model = train_dqn(train_raw.astype(np.float32), train_labels, input_dim)
        s_cal_pt  = score_dqn(model, cal_raw)
        s_eval_pt = score_dqn(model, eval_raw)
        s_cal_w,  y_cal_w  = aggregate_to_windows(s_cal_pt,  cal_lbl)
        s_eval_w, y_eval_w = aggregate_to_windows(s_eval_pt, eval_lbl)
        cache_scores(dataset_name, "DQN", unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w)
        result = eta_sweep(f"DQN_{unit_id}", s_cal_w, y_cal_w, s_eval_w, y_eval_w, M=50)
        logger.info("    DQN AUROC=%.4f raw_FPR=%.4f",
                    result.get("auroc", float("nan")), result["raw"]["fpr"])
        return result
    except Exception as e:
        logger.warning("    DQN failed: %s", e)
        return {}


def run_deepod_method(method_name, cls, train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name="unknown"):
    try:
        train_raw32 = train_raw.astype(np.float32)
        keep, dropped_idx = feature_prescreen(train_raw32)
        if keep.sum() < 2:
            logger.warning("    %s/%s: only %d usable features, EXCLUDING unit", method_name, unit_id, int(keep.sum()))
            return {}
        n_features_raw = train_raw32.shape[1]
        train_sel = train_raw32[:, keep]
        cal_sel   = cal_raw.astype(np.float32)[:, keep]
        eval_sel  = eval_raw.astype(np.float32)[:, keep]
        assert np.isfinite(train_sel).all(), "non-finite values survived feature prescreen in train"
        clf = cls(epochs=50, device=DEVICE)
        clf.fit(train_sel)
        s_cal_pt  = clf.decision_function(cal_sel)
        s_eval_pt = clf.decision_function(eval_sel)
        del clf
        import torch
        torch.cuda.empty_cache()
        n_nan_cal  = int(np.isnan(s_cal_pt).sum())
        n_nan_eval = int(np.isnan(s_eval_pt).sum())
        if n_nan_cal or n_nan_eval:
            ratio = (n_nan_cal + n_nan_eval) / max(len(s_cal_pt) + len(s_eval_pt), 1)
            logger.warning("    %s/%s: %d nan cal, %d nan eval scores (%.3f%%)",
                            method_name, unit_id, n_nan_cal, n_nan_eval, 100.0 * ratio)
        s_cal_w,  y_cal_w,  cal_starts  = aggregate_to_windows_with_starts(s_cal_pt,  cal_lbl)
        s_eval_w, y_eval_w, eval_starts = aggregate_to_windows_with_starts(s_eval_pt, eval_lbl)
        cached = cache_scores(dataset_name, method_name, unit_id, s_cal_w, y_cal_w, s_eval_w, y_eval_w,
                               cal_window_start=cal_starts, eval_window_start=eval_starts,
                               n_features_raw=n_features_raw, n_features_dropped=int(len(dropped_idx)),
                               dropped_feature_idx=dropped_idx)
        if not cached:
            return {}
        if not np.isfinite(s_cal_w).all() or not np.isfinite(s_eval_w).all():
            logger.warning("    %s/%s: non-finite scores after windowing despite cache pass, skipping eta_sweep", method_name, unit_id)
            return {}
        result = eta_sweep(f"{method_name}_{unit_id}", s_cal_w, y_cal_w, s_eval_w, y_eval_w, M=50)
        logger.info("    %s AUROC=%.4f raw_FPR=%.4f", method_name,
                    result.get("auroc", float("nan")), result["raw"]["fpr"])
        return result
    except Exception as e:
        logger.warning("    %s failed: %s", method_name, e)
        return {}


def run_all_extended(train_raw, train_labels, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name="unknown"):
    from deepod.models import NeuTraL, GOAD, SLAD, RDP, ICL

    results = {}

    results["DeepIsolationForest"] = run_deep_isolation_forest(
        train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)

    results["NeuTraL"] = run_deepod_method(
        "NeuTraL", NeuTraL, train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)

    goad_hidden_dim = 100 if dataset_name == "unsw_nb15" else 8
    results["GOAD"] = run_deepod_method(
        "GOAD", lambda **kw: GOAD(hidden_dim=goad_hidden_dim, **kw), train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)

    results["SLAD"] = run_deepod_method(
        "SLAD", SLAD, train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)

    results["RDP"] = run_deepod_method(
        "RDP", RDP, train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)

    results["ICL"] = run_deepod_method(
        "ICL", ICL, train_raw, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)

    return results


def load_smd_units():
    base = Path("data/raw/smd/train")
    units = []
    for f in sorted((base / "train").glob("*.txt")):
        mid = f.stem
        train = np.loadtxt(str(f), delimiter=",")
        test  = np.loadtxt(str(base / "test" / f"{mid}.txt"), delimiter=",")
        label = np.loadtxt(str(base / "test_label" / f"{mid}.txt"), delimiter=",").astype(int)
        train_labels = np.zeros(len(train), dtype=int)
        units.append((mid, train, train_labels, test, label))
    return units


def load_smap_msl_units(dataset):
    import ast
    base      = Path("data/raw/smap_msl")
    label_df  = pd.read_csv(base / "labeled_anomalies.csv")
    label_df.columns = [c.strip().lower() for c in label_df.columns]
    target = "MSL" if dataset == "msl" else "SMAP"
    channels = label_df[label_df["spacecraft"] == target]["chan_id"].tolist()
    units = []
    for ch in sorted(channels):
        tf = base / "train" / f"{ch}.npy"
        xf = base / "test"  / f"{ch}.npy"
        if not tf.exists() or not xf.exists():
            continue
        train = np.load(tf)
        test  = np.load(xf)
        row   = label_df[label_df["chan_id"] == ch]
        label = np.zeros(len(test), dtype=int)
        if not row.empty:
            for s, e in ast.literal_eval(row["anomaly_sequences"].values[0]):
                label[int(s):int(e)] = 1
        train_labels = np.zeros(len(train), dtype=int)
        units.append((ch, train, train_labels, test, label))
    return units


def load_batadal_units():
    base = Path("data/raw/batadal")
    drop = {"datetime", "att_flag"}
    train_df = pd.read_csv(base / "dataset03.csv")
    test_df  = pd.read_csv(base / "dataset04.csv")
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns  = [c.strip().lower() for c in test_df.columns]
    att_col   = next(c for c in test_df.columns if "att_flag" in c)
    test_label = pd.to_numeric(test_df[att_col], errors="coerce").fillna(0).astype(int).values
    feat_cols = [c for c in train_df.columns if c not in drop
                 and pd.api.types.is_numeric_dtype(train_df[c])]
    for c in feat_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c]  = pd.to_numeric(test_df[c],  errors="coerce")
    train_df[feat_cols] = train_df[feat_cols].ffill().fillna(0.0)
    test_df[feat_cols]  = test_df[feat_cols].ffill().fillna(0.0)
    train_arr = train_df[feat_cols].values.astype(np.float32)
    test_arr  = test_df[feat_cols].values.astype(np.float32)
    train_labels = np.zeros(len(train_arr), dtype=int)
    return [("batadal", train_arr, train_labels, test_arr, test_label)]


def load_cicids_units():
    base = Path("data/raw/cicids2017/MachineLearningCVE")
    drop_frags = {"flow id","source ip","destination ip","src ip","dst ip",
                  "source port","destination port","timestamp","label"}
    def load_csv(path):
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    mon = load_csv(base / "Monday-WorkingHours.pcap_ISCX.csv")
    feat_cols = [c for c in mon.columns
                 if not any(f in c for f in drop_frags)
                 and pd.api.types.is_numeric_dtype(mon[c])]
    mon[feat_cols] = mon[feat_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    mu  = mon[feat_cols].mean()
    std = mon[feat_cols].std().replace(0, 1)
    train_raw    = ((mon[feat_cols] - mu) / std).values.astype(np.float32)
    train_raw    = np.clip(train_raw, -10, 10)
    train_labels = np.zeros(len(train_raw), dtype=int)
    units = []
    for f in sorted(base.glob("*.csv")):
        if "monday" in f.name.lower():
            continue
        df = load_csv(f)
        label_col = next((c for c in df.columns if c == "label"), None)
        if label_col is None:
            continue
        labels = (df[label_col].str.strip().str.lower() != "benign").astype(int).values
        arr = np.zeros((len(df), len(feat_cols)), dtype=np.float32)
        for i, c in enumerate(feat_cols):
            if c in df.columns:
                arr[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
        arr = np.clip(((arr - mu.values) / std.values).astype(np.float32), -10, 10)
        unit_id = f.stem.replace(".pcap_ISCX", "")
        units.append((unit_id, train_raw, train_labels, arr, labels))
    return units


def load_unsw_units():
    base = Path("data/raw/unsw_nb15/Training and Testing Sets")
    drop = {"id","proto","service","state","attack_cat","label"}
    train_df = pd.read_csv(base / "UNSW_NB15_training-set.csv")
    test_df  = pd.read_csv(base / "UNSW_NB15_testing-set.csv")
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns  = [c.strip().lower() for c in test_df.columns]
    feat_cols = [c for c in train_df.columns if c not in drop
                 and pd.api.types.is_numeric_dtype(train_df[c])]
    train_normal = train_df[train_df["label"] == 0][feat_cols].copy()
    train_normal = train_normal.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    test_df[feat_cols] = test_df[feat_cols].replace(
        [float("inf"), float("-inf")], float("nan")).fillna(0.0)
    mu  = train_normal.mean()
    std = train_normal.std().replace(0, 1)
    train_arr    = np.clip(((train_normal - mu) / std).values.astype(np.float32), -10, 10)
    test_arr     = np.clip(((test_df[feat_cols] - mu) / std).values.astype(np.float32), -10, 10)
    test_label   = test_df["label"].astype(int).values
    train_labels = np.zeros(len(train_arr), dtype=int)
    return [("unsw_nb15", train_arr, train_labels, test_arr, test_label)]


LOADERS = {
    "smd":        load_smd_units,
    "smap":       lambda: load_smap_msl_units("smap"),
    "msl":        lambda: load_smap_msl_units("msl"),
    "batadal":    load_batadal_units,
    "cicids2017": load_cicids_units,
    "unsw_nb15":  load_unsw_units,
}


def run_dataset(dataset_name, units, out_dir):
    logger.info("=== %s ===", dataset_name.upper())
    all_results = {}
    existing_json = out_dir / "results.json"
    if existing_json.exists():
        import json as _json
        with open(existing_json) as f:
            all_results = _json.load(f)
        logger.info("  Resuming from %d existing units", len(all_results))
    for unit_id, train_raw, train_labels, test_raw, test_labels in units:
        logger.info("  Unit: %s | train=%d test=%d anomalies=%d (%.1f%%)",
                    unit_id, len(train_raw), len(test_raw),
                    test_labels.sum(), 100*test_labels.mean())
        n_cal     = int(len(test_raw) * CAL_FRAC)
        cal_raw,  eval_raw  = test_raw[:n_cal],    test_raw[n_cal:]
        cal_lbl,  eval_lbl  = test_labels[:n_cal], test_labels[n_cal:]
        _, y_eval_w = aggregate_to_windows(np.zeros(len(eval_raw)), eval_lbl)
        if y_eval_w.sum() == 0:
            logger.warning("  %s: no anomalies in eval, skipping", unit_id)
            continue
        if unit_id in all_results:
            logger.info("  %s: already done, skipping", unit_id)
            continue
        unit_results = run_all_extended(
            train_raw, train_labels, cal_raw, cal_lbl, eval_raw, eval_lbl, unit_id, dataset_name)
        all_results[unit_id] = unit_results
        logger.info("  %s done: %d methods", unit_id, len(unit_results))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    logger.info("\n%s SUMMARY:", dataset_name.upper())
    method_aurocs = {}
    for unit, methods in all_results.items():
        for method, res in methods.items():
            auroc = res.get("auroc", float("nan"))
            if not np.isnan(auroc):
                method_aurocs.setdefault(method, []).append(auroc)
    for method, aurocs in sorted(method_aurocs.items()):
        logger.info("  %-25s AUROC=%.4f +/-%.4f n=%d",
                    method, np.mean(aurocs), np.std(aurocs), len(aurocs))
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=list(LOADERS.keys()) + ["all"])
    parser.add_argument("--output-dir", default="results/extended_methods")
    args = parser.parse_args()
    datasets = list(LOADERS.keys()) if "all" in args.datasets else args.datasets
    base_out = Path(args.output_dir)
    logger.info("Device: %s", DEVICE)
    logger.info("Datasets: %s", datasets)
    for ds in datasets:
        units = LOADERS[ds]()
        run_dataset(ds, units, base_out / ds)
    logger.info("All done. Results in %s/", args.output_dir)


if __name__ == "__main__":
    main()
