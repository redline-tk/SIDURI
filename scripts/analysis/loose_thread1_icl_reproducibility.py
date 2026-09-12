import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deepod.models import ICL, NeuTraL

base = Path("data/raw/unsw_nb15/Training and Testing Sets")
drop = {"id", "proto", "service", "state", "attack_cat", "label"}
train_df = pd.read_csv(base / "UNSW_NB15_training-set.csv")
test_df = pd.read_csv(base / "UNSW_NB15_testing-set.csv")
train_df.columns = [c.strip().lower() for c in train_df.columns]
test_df.columns = [c.strip().lower() for c in test_df.columns]
feat_cols = [c for c in train_df.columns if c not in drop and pd.api.types.is_numeric_dtype(train_df[c])]
train_normal = train_df[train_df["label"] == 0][feat_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
test_df[feat_cols] = test_df[feat_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
mu = train_normal.mean()
std = train_normal.std().replace(0, 1)
X_train = np.clip(((train_normal - mu) / std).values.astype(np.float32), -10, 10)
X_test = np.clip(((test_df[feat_cols] - mu) / std).values.astype(np.float32), -10, 10)
y_test = test_df["label"].astype(int).values

def main():
    for name, cls in [("ICL", ICL), ("NeuTraL", NeuTraL)]:
        print("=== %s ===" % name)
        results = []
        for run in range(3):
            clf = cls(epochs=50, device="cuda:0", verbose=0)
            clf.fit(X_train)
            scores = clf.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
            results.append(auc)
            print("  run %d: AUC=%.8f" % (run, auc))
        all_identical = len(set(results)) == 1
        print("  all runs bit-identical: %s" % all_identical)
        print("  max pairwise diff: %.8f" % (max(results) - min(results)))
        print()

if __name__ == "__main__":
    main()
