import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deepod.models import GOAD

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

X_normal = X[y == 0]
X_train, X_holdout_normal = train_test_split(X_normal, test_size=0.3, random_state=0)
X_anom = X[y == 1]
X_test = np.concatenate([X_holdout_normal, X_anom])
y_test = np.concatenate([np.zeros(len(X_holdout_normal)), np.ones(len(X_anom))])

mu = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0
X_train = np.clip((X_train - mu) / std, -10, 10)
X_test = np.clip((X_test - mu) / std, -10, 10)

def chunked_score(clf, X, chunk_size=5000):
    scores = []
    for i in range(0, len(X), chunk_size):
        chunk = X[i:i + chunk_size]
        s = clf.decision_function(chunk)
        scores.append(np.asarray(s))
        torch.cuda.empty_cache()
    return np.concatenate(scores)

print("Train=%d Test=%d (anomaly_rate=%.4f)" % (len(X_train), len(X_test), y_test.mean()))

for hd in [32, 64, 128]:
    torch.cuda.empty_cache()
    clf = GOAD(epochs=100, device="cuda:0", random_state=0, hidden_dim=hd)
    clf.fit(X_train)
    scores = chunked_score(clf, X_test, chunk_size=2000)
    auc = roc_auc_score(y_test, scores)
    print("GOAD hidden_dim=%d: AUC=%.4f  published=0.903  gap=%.4f" % (hd, auc, abs(auc - 0.903)))
