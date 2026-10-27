"""Fast preprocessing x feature-set grid, held-out evaluation only.

Scope (deliberately narrow for speed):
  - 2 preprocessing methods: none, lemmatization
  - 3 feature sets: word-only TF-IDF, char-only TF-IDF, hybrid word+char
  - 4 fast classifiers only (RF, LR, NB, KNN) using their already-tuned
    best hyperparameters from hyperparameter_search.py. SVM and GB are
    excluded from this grid -- each combo costs minutes for them and their
    hyperparameters are already tuned at the fixed E2+lemmatization config.
  - Held-out split only (CV skipped -- known leakage-inflated, not the
    deciding protocol).

Run: python preprocessing_feature_grid.py --task both
"""
import argparse, json, os, time, threading, warnings
import numpy as np, pandas as pd, psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline

from enhanced_experiment import CombinedFeatures

warnings.filterwarnings("ignore")

CATEGORY_ORDER = ["non_risk", "harassment", "threats", "hate_speech", "sexual_content", "self_harm_risk"]

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
PREPROCESSING = ["none", "lemmatization"]


def run_combo(X_train, y_train, X_test, y_test, prep, feat_name, clf_name, clf_template, class_labels):
    clf = clone_estimator(clf_template)
    pipe = Pipeline([
        ("features", CombinedFeatures(preprocessing=prep, **FEATURE_CONFIGS[feat_name])),
        ("classifier", clf),
    ])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=class_labels, average="macro", zero_division=0)
    return {"preprocessing": prep, "feature_set": feat_name, "classifier": clf_name,
            "accuracy": acc, "macro_f1": f1, "fit_time_seconds": fit_s}


def clone_estimator(est):
    from sklearn.base import clone
    return clone(est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Categorised_dataset.csv")
    ap.add_argument("--task", choices=["binary", "six_class", "both"], default="both")
    ap.add_argument("--output-dir", default="results/preprocessing_feature_grid")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df = df.dropna(subset=["text", "risk_label", "risk_category"]).copy()
    df["text"] = df["text"].astype(str)
    split = df["split"].astype(str).str.lower()
    train_mask, test_mask = split.eq("train"), split.eq("test")
    X_train = df.loc[train_mask, "text"].reset_index(drop=True)
    X_test = df.loc[test_mask, "text"].reset_index(drop=True)

    tasks = ["binary", "six_class"] if args.task == "both" else [args.task]
    all_rows = []
    for task in tasks:
        if task == "binary":
            y_train = pd.to_numeric(df.loc[train_mask, "risk_label"]).astype(int).reset_index(drop=True)
            y_test = pd.to_numeric(df.loc[test_mask, "risk_label"]).astype(int).reset_index(drop=True)
            class_labels = [0, 1]
        else:
            y_train = df.loc[train_mask, "risk_category"].astype(str).reset_index(drop=True)
            y_test = df.loc[test_mask, "risk_category"].astype(str).reset_index(drop=True)
            class_labels = CATEGORY_ORDER

        for clf_name, clf_template in TUNED[task].items():
            for prep in PREPROCESSING:
                for feat_name in FEATURE_CONFIGS:
                    t0 = time.perf_counter()
                    row = run_combo(X_train, y_train, X_test, y_test, prep, feat_name, clf_name, clf_template, class_labels)
                    row["task"] = task
                    all_rows.append(row)
                    print(f"[{task}] {clf_name} prep={prep} feat={feat_name} "
                          f"acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} ({time.perf_counter()-t0:.1f}s)")

    result = pd.DataFrame(all_rows)
    result.to_csv(os.path.join(args.output_dir, "grid_results.csv"), index=False)

    # Best combo per (task, classifier)
    best = result.loc[result.groupby(["task", "classifier"])["macro_f1"].idxmax()]
    best.to_csv(os.path.join(args.output_dir, "best_combo_per_classifier.csv"), index=False)
    print("\n=== BEST COMBO PER (TASK, CLASSIFIER) ===")
    print(best[["task", "classifier", "preprocessing", "feature_set", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
