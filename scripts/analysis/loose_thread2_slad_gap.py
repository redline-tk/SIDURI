import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deepod.models import SLAD

base = Path("data/raw/unsw_nb15/Training and Testing Sets")
drop = {"id", "proto", "service", "state", "attack_cat", "label"}
train_df = pd.read_csv(base / "UNSW_NB15_training-set.csv")
test_df = pd.read_csv(base / "UNSW_NB15_testing-set.csv")
train_df.columns = [c.strip().lower() for c in train_df.columns]
test_df.columns = [c.strip().lower() for c in test_df.columns]
full = pd.concat([train_df, test_df], ignore_index=True)
feat_cols = [c for c in full.columns if c not in drop and pd.api.types.is_numeric_dtype(full[c])]
full[feat_cols] = full[feat_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
X = full[feat_cols].values.astype(np.float32)
y = full["label"].astype(int).values

PUBLISHED_SLAD = 0.941
SIDURI_TEMPORAL_SPLIT_SLAD = 0.8287

def main():
    X_normal = X[y == 0]
    X_train, X_holdout_normal = train_test_split(X_normal, test_size=0.3, random_state=0)
    X_anom = X[y == 1]
    X_test = np.concatenate([X_holdout_normal, X_anom])
    y_test = np.concatenate([np.zeros(len(X_holdout_normal)), np.ones(len(X_anom))])

    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_n = np.clip((X_train - mu) / std, -10, 10)
    X_test_n = np.clip((X_test - mu) / std, -10, 10)

    print("Train(normal-only)=%d  Test=%d (anomaly_rate=%.4f)" % (len(X_train_n), len(X_test_n), y_test.mean()))

    clf = SLAD(epochs=100, device="cuda:0", random_state=0, verbose=0)
    clf.fit(X_train_n)
    scores = clf.decision_function(X_test_n)
    auc = roc_auc_score(y_test, scores)
    print("SLAD standard-split AUC=%.4f" % auc)
    print("Published (SLAD paper, UNSW-NB15): %.4f" % PUBLISHED_SLAD)
    print("SIDURI temporal cal/eval split: %.4f" % SIDURI_TEMPORAL_SPLIT_SLAD)
    print("Gap: standard-split vs published = %.4f" % abs(auc - PUBLISHED_SLAD))
    print("Gap: standard-split vs SIDURI temporal split = %.4f" % abs(auc - SIDURI_TEMPORAL_SPLIT_SLAD))

if __name__ == "__main__":
    main()
