"""One-shot evaluation of LDA_60.pkl using the same method as main_ni.py."""
import os, sys, pickle
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from non_instruction.runner_ni import load_personality_ndarrays, build_train_test_arrays

cfg = yaml.safe_load(open("config.yaml"))
for k, v in cfg["paths"].items():
    if not os.path.isabs(str(v)):
        cfg["paths"][k] = os.path.normpath(os.path.join(os.getcwd(), v))

personalities = cfg["training"]["personalities"]
num_tournaments = cfg["training"]["num_tournaments"]

print("Loading data...")
arrays = load_personality_ndarrays(cfg, num_tournaments)
X_train, X_test, y_train, y_test = build_train_test_arrays(arrays, cfg)

model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "LDA_models", "LDA_60.pkl"
)
print(f"Loading model: {model_path}")
with open(model_path, "rb") as f:
    payload = pickle.load(f)

pca    = payload["pca"]
lda    = payload["lda"]
params = payload["params"]

print(f"Model params: solver={params['solver']}, shrinkage={params['shrinkage']}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)

# 5-fold CV on training set (same as main_ni.py full-run mode)
kfold_splits = cfg["lda"]["kfold_splits"]
skf = StratifiedKFold(n_splits=kfold_splits)
cv_scores = cross_val_score(lda, X_train_pca, y_train, cv=skf)
print(f"\nCV scores ({kfold_splits}-fold): {cv_scores}")
print(f"CV mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

# Test set evaluation
y_pred = lda.predict(X_test_pca)
n_wrong = int((y_pred != y_test).sum())
mislabel_pct = 100.0 * n_wrong / len(y_test)
print(f"Mislabeled: {n_wrong}/{len(y_test)} ({mislabel_pct:.1f}%)")

print("\nConfusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred)
header = "         " + "  ".join(f"{p[:4]:>6}" for p in personalities)
print(header)
for i, p in enumerate(personalities):
    row = "  ".join(f"{cm[i, j]:6d}" for j in range(len(personalities)))
    print(f"  {p[:4]:>6}  {row}")
