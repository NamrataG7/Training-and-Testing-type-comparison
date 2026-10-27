"""Latency measurement for the 8 already-identified best (task, classifier,
preprocessing, feature_set) combos from preprocessing_feature_grid.py.
Only re-fits those 8 (not the full 24-combo grid) to stay fast.
"""
import json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from enhanced_experiment import CombinedFeatures

warnings.filterwarnings("ignore")

TUNED = {
    "binary": {
        "RF": RandomForestClassifier(n_estimators=10, max_depth=20, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=42),
        "LR": LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=42),
        "NB": MultinomialNB(alpha=0.1),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=1),
    },
    "six_class": {
        "RF": RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=42),
        "LR": LogisticRegression(C=10.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=42),
        "NB": MultinomialNB(alpha=0.1),
        "KNN": KNeighborsClassifier(n_neighbors=3, n_jobs=1),
    },
}
FEATURE_CONFIGS = {
    "word_only": {"use_word": True, "use_char": False},
    "char_only": {"use_word": False, "use_char": True},
    "hybrid": {"use_word": True, "use_char": True},
}

df = pd.read_csv("Categorised_dataset.csv")
df = df.dropna(subset=["text", "risk_label", "risk_category"]).copy()
df["text"] = df["text"].astype(str)
split = df["split"].astype(str).str.lower()
train_mask, test_mask = split.eq("train"), split.eq("test")
X_train_all = df.loc[train_mask, "text"].reset_index(drop=True)
X_test_all = df.loc[test_mask, "text"].reset_index(drop=True)

best = pd.read_csv("results/preprocessing_feature_grid/best_combo_per_classifier.csv")

N_LATENCY = 200
rows = []
for _, r in best.iterrows():
    task, clf_name, prep, feat = r["task"], r["classifier"], r["preprocessing"], r["feature_set"]
    if task == "binary":
        y_train = pd.to_numeric(df.loc[train_mask, "risk_label"]).astype(int).reset_index(drop=True)
    else:
        y_train = df.loc[train_mask, "risk_category"].astype(str).reset_index(drop=True)

    from sklearn.base import clone
    clf = clone(TUNED[task][clf_name])
    pipe = Pipeline([
        ("features", CombinedFeatures(preprocessing=prep, **FEATURE_CONFIGS[feat])),
        ("classifier", clf),
    ])
    pipe.fit(X_train_all, y_train)

    sample_texts = X_test_all.iloc[:N_LATENCY].tolist()
    for t in sample_texts[:5]:
        pipe.predict([t])
    lat_ms = []
    for t in sample_texts:
        t0 = time.perf_counter()
        pipe.predict([t])
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    lat_ms = np.asarray(lat_ms)

    rows.append({
        "task": task, "classifier": clf_name, "preprocessing": prep, "feature_set": feat,
        "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
        "end_to_end_mean_latency_ms": float(np.mean(lat_ms)),
        "end_to_end_p95_latency_ms": float(np.percentile(lat_ms, 95)),
    })
    print(f"[{task}] {clf_name} prep={prep} feat={feat} latency_mean={np.mean(lat_ms):.3f}ms p95={np.percentile(lat_ms,95):.3f}ms")

out = pd.DataFrame(rows)
out.to_csv("results/preprocessing_feature_grid/best_combo_per_classifier_with_latency.csv", index=False)
print("\nSaved: results/preprocessing_feature_grid/best_combo_per_classifier_with_latency.csv")
