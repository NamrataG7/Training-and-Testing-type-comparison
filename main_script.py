"""Final consolidated benchmark suite.

Runs the full cross-product of:
  - preprocessing: none, stopword, stemming, lemmatization, stopword_stemming, stopword_lemmatization
  - feature sets: word (word-only TF-IDF), char (char-only TF-IDF), hybrid (word+char)
  - classifiers: RF, LR, NB, KNN, SVM, GB (tuned hyperparameters, fixed -- not re-searched here)
  - protocol: held-out (single preassigned split) and/or stratified 5-fold CV
  - task: binary (risk_label) and/or six-class (risk_category)

Every combo writes ONE row, appended incrementally to the output CSV, so a
partial/interrupted run still leaves usable results.

Documented cost-control exclusions (not silent skips):
  - GB six-class is the single most expensive cell in the entire grid
    (~400-430s per single fit; 5-fold CV would be ~35-60+ min for ONE combo,
    let alone 18). By default, GB six-class is excluded from BOTH the
    preprocessing grid AND cross-validation. Use --include-gb-six-class to
    force it (will be slow).
  - The full 18-combo grid is run under the held-out protocol only, for
    picking each classifier's best-fit preprocessing/feature combo. The CV
    protocol is then run ONLY for the per-(classifier,task) winning combo
    from the held-out grid (not all 18 combos) -- running the full grid
    under 5-fold CV would be ~5x the already large held-out cost for very
    little additional information (CV rankings were already shown to track
    held-out rankings closely in this project's earlier work).

Run examples:
  python full_benchmark_suite.py --phase grid --protocol heldout
  python full_benchmark_suite.py --phase winners --protocol cv --winners-file results/full_suite/grid_heldout.csv
  python full_benchmark_suite.py --phase baseline --protocol both --preprocessing lemmatization --features hybrid
"""
import argparse, json, os, time, threading, warnings, tempfile
import numpy as np, pandas as pd, psutil, joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)
from sklearn.pipeline import Pipeline

from enhanced_experiment import CombinedFeatures

warnings.filterwarnings("ignore")

CATEGORY_ORDER = ["non_risk", "harassment", "threats", "hate_speech", "sexual_content", "self_harm_risk"]
ALL_PREPROCESSING = ["none", "stopword", "stemming", "lemmatization", "stopword_stemming", "stopword_lemmatization"]
ALL_FEATURES = ["word", "char", "hybrid"]
FEATURE_FLAGS = {"word": dict(use_word=True, use_char=False),
                 "char": dict(use_word=False, use_char=True),
                 "hybrid": dict(use_word=True, use_char=True)}

TUNED = {
    "binary": {
        "RF": lambda: RandomForestClassifier(n_estimators=10, max_depth=20, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=42),
        "LR": lambda: LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=42),
        "NB": lambda: MultinomialNB(alpha=0.1),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=1),
        "SVM": lambda: SVC(kernel="linear", C=10.0, class_weight="balanced"),
        "GB": lambda: Pipeline([("svd", TruncatedSVD(n_components=200, random_state=42)),
                                ("gb", GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42))]),
    },
    "six_class": {
        "RF": lambda: RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=42),
        "LR": lambda: LogisticRegression(C=10.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=42),
        "NB": lambda: MultinomialNB(alpha=0.1),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=3, n_jobs=1),
        "SVM": lambda: SVC(kernel="linear", C=10.0, class_weight="balanced"),
        "GB": lambda: Pipeline([("svd", TruncatedSVD(n_components=200, random_state=42)),
                                ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))]),
    },
}


class RSSPeakSampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._t = None
    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self._t = threading.Thread(target=self._sample, daemon=True)
        self._t.start()
        return self
    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)
    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def mb(n): return n / (1024 ** 2)


def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["text", "risk_label", "risk_category"]).copy()
    df["text"] = df["text"].astype(str)
    return df


def class_labels_for(task):
    return [0, 1] if task == "binary" else CATEGORY_ORDER


def y_for(df, task):
    if task == "binary":
        return pd.to_numeric(df["risk_label"]).astype(int)
    return df["risk_category"].astype(str)


def evaluate_once(X_train, y_train, X_test, y_test, clf_factory, prep, feat, class_labels, latency_samples=150):
    clf = clf_factory()
    pipe = Pipeline([
        ("features", CombinedFeatures(preprocessing=prep, **FEATURE_FLAGS[feat])),
        ("classifier", clf),
    ])
    t0 = time.perf_counter()
    with RSSPeakSampler() as fit_sampler:
        pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_test, y_pred, labels=class_labels, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_test, y_pred, labels=class_labels, average="weighted", zero_division=0)

    # Specificity: per-class one-vs-rest TN/(TN+FP), macro-averaged across
    # class_labels. For binary this reduces to the single negative-class
    # specificity; for six-class it is the mean one-vs-rest specificity.
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    total = cm.sum()
    specificities = []
    for i in range(len(class_labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) else 0.0)
    specificity_macro = float(np.mean(specificities))

    # Classifier-only latency: time the already-extracted-feature predict
    # step separately from the full pipeline (feature-extraction + predict).
    X_test_features = pipe.named_steps["features"].transform(X_test)
    classifier_step = pipe.named_steps["classifier"]
    n_clf = min(latency_samples, X_test_features.shape[0])
    for i in range(min(5, n_clf)):
        classifier_step.predict(X_test_features[i])
    clf_lat = []
    for i in range(n_clf):
        s = time.perf_counter()
        classifier_step.predict(X_test_features[i])
        clf_lat.append((time.perf_counter() - s) * 1000.0)
    clf_lat = np.asarray(clf_lat)

    # End-to-end latency: full pipeline (preprocessing + feature extraction
    # + predict) on raw text, one message at a time.
    n_lat = min(latency_samples, len(X_test))
    sample = list(X_test)[:n_lat] if not hasattr(X_test, "iloc") else X_test.iloc[:n_lat].tolist()
    for t in sample[:5]:
        pipe.predict([t])
    lat = []
    for t in sample:
        s = time.perf_counter()
        pipe.predict([t])
        lat.append((time.perf_counter() - s) * 1000.0)
    lat = np.asarray(lat)

    # Model size on disk (joblib-serialized, compressed, matching the
    # project's established convention from earlier benchmark scripts).
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
        model_path = tf.name
    joblib.dump(pipe, model_path, compress=3)
    model_mb = mb(os.path.getsize(model_path))
    os.remove(model_path)

    return {
        "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
        "weighted_precision": wp, "weighted_recall": wr, "weighted_f1": wf1,
        "specificity_macro": specificity_macro,
        "fit_time_seconds": fit_s, "fit_peak_rss_mb": mb(fit_sampler.peak),
        "e2e_mean_latency_ms": float(np.mean(lat)), "e2e_p95_latency_ms": float(np.percentile(lat, 95)),
        "classifier_mean_latency_ms": float(np.mean(clf_lat)), "classifier_p95_latency_ms": float(np.percentile(clf_lat, 95)),
        "model_file_mb": model_mb,
    }


def run_heldout_grid(df, tasks, classifiers, preprocessing_list, feature_list, out_path, include_gb_six_class):
    split = df["split"].astype(str).str.lower()
    train_mask, test_mask = split.eq("train"), split.eq("test")
    X_train_all = df.loc[train_mask, "text"].reset_index(drop=True)
    X_test_all = df.loc[test_mask, "text"].reset_index(drop=True)

    header_written = os.path.exists(out_path)
    for task in tasks:
        y_all = y_for(df, task)
        y_train = y_all.loc[train_mask].reset_index(drop=True)
        y_test = y_all.loc[test_mask].reset_index(drop=True)
        class_labels = class_labels_for(task)
        for clf_name in classifiers:
            if clf_name == "GB" and task == "six_class" and not include_gb_six_class:
                print(f"[SKIP] GB x six_class excluded by default (cost). Use --include-gb-six-class to force.")
                continue
            for prep in preprocessing_list:
                for feat in feature_list:
                    t0 = time.perf_counter()
                    try:
                        row = evaluate_once(X_train_all, y_train, X_test_all, y_test,
                                             TUNED[task][clf_name], prep, feat, class_labels)
                        row.update({"protocol": "heldout", "task": task, "classifier": clf_name,
                                    "preprocessing": prep, "feature_set": feat, "status": "ok", "error": ""})
                    except Exception as e:
                        row = {"protocol": "heldout", "task": task, "classifier": clf_name,
                               "preprocessing": prep, "feature_set": feat, "status": "failed", "error": str(e)}
                    pd.DataFrame([row]).to_csv(out_path, mode="a", header=not header_written, index=False)
                    header_written = True
                    print(f"[heldout][{task}] {clf_name} prep={prep} feat={feat} "
                          f"acc={row.get('accuracy', float('nan')):.4f} f1={row.get('macro_f1', float('nan')):.4f} "
                          f"({time.perf_counter()-t0:.1f}s)")


def run_cv_for_combo(df, task, clf_name, prep, feat, n_splits=5, latency_samples=150):
    X = df["text"].reset_index(drop=True)
    y = y_for(df, task).reset_index(drop=True)
    class_labels = class_labels_for(task)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rows = []
    for i, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        y_tr, y_te = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        r = evaluate_once(X_tr, y_tr, X_te, y_te, TUNED[task][clf_name], prep, feat, class_labels, latency_samples)
        fold_rows.append(r)
        print(f"  [cv fold {i}/{n_splits}] acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} fit={r['fit_time_seconds']:.1f}s")
    agg = pd.DataFrame(fold_rows)
    out = {f"mean_{c}": agg[c].mean() for c in agg.columns}
    out.update({f"std_{c}": agg[c].std() for c in agg.columns})
    out.update({"protocol": "cv", "task": task, "classifier": clf_name, "preprocessing": prep,
                "feature_set": feat, "n_splits": n_splits, "status": "ok", "error": ""})
    return out


def run_winners_cv(df, winners_df, out_path, include_gb_six_class):
    header_written = os.path.exists(out_path)
    for _, w in winners_df.iterrows():
        task, clf_name, prep, feat = w["task"], w["classifier"], w["preprocessing"], w["feature_set"]
        if clf_name == "GB" and task == "six_class" and not include_gb_six_class:
            print("[SKIP] GB x six_class CV excluded by default (cost).")
            continue
        try:
            out = run_cv_for_combo(df, task, clf_name, prep, feat)
        except Exception as e:
            out = {"protocol": "cv", "task": task, "classifier": clf_name, "preprocessing": prep,
                   "feature_set": feat, "status": "failed", "error": str(e)}
        pd.DataFrame([out]).to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        print(f"[cv-winner][{task}] {clf_name} prep={prep} feat={feat} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Categorised_dataset.csv")
    ap.add_argument("--phase", choices=["grid", "winners", "baseline"], required=True,
                     help="grid: full preprocessing x feature sweep, held-out only. "
                          "winners: run CV for each row in --winners-file. "
                          "baseline: run a single fixed preprocessing+feature combo for all classifiers/tasks/protocols.")
    ap.add_argument("--task", choices=["binary", "six_class", "both"], default="both")
    ap.add_argument("--classifiers", default="RF,LR,NB,KNN,SVM,GB")
    ap.add_argument("--preprocessing", default=",".join(ALL_PREPROCESSING))
    ap.add_argument("--features", default=",".join(ALL_FEATURES))
    ap.add_argument("--protocol", choices=["heldout", "cv", "both"], default="heldout")
    ap.add_argument("--winners-file", default=None)
    ap.add_argument("--include-gb-six-class", action="store_true")
    ap.add_argument("--output-dir", default="results/full_suite")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.dataset)
    tasks = ["binary", "six_class"] if args.task == "both" else [args.task]
    classifiers = args.classifiers.split(",")
    preps = args.preprocessing.split(",")
    feats = args.features.split(",")

    if args.phase == "grid":
        out_path = os.path.join(args.output_dir, "grid_heldout.csv")
        run_heldout_grid(df, tasks, classifiers, preps, feats, out_path, args.include_gb_six_class)

    elif args.phase == "winners":
        winners = pd.read_csv(args.winners_file)
        # pick best macro_f1 per (task, classifier) if raw grid passed
        if "macro_f1" in winners.columns and len(winners) > winners[["task", "classifier"]].drop_duplicates().shape[0]:
            winners = winners.loc[winners.groupby(["task", "classifier"])["macro_f1"].idxmax()]
        out_path = os.path.join(args.output_dir, "cv_winners.csv")
        run_winners_cv(df, winners, out_path, args.include_gb_six_class)

    elif args.phase == "baseline":
        assert len(preps) == 1 and len(feats) == 1, "baseline phase expects exactly one preprocessing and one feature set"
        prep, feat = preps[0], feats[0]
        rows = []
        for task in tasks:
            for clf_name in classifiers:
                if clf_name == "GB" and task == "six_class" and not args.include_gb_six_class and args.protocol in ("cv", "both"):
                    print("[SKIP] GB x six_class CV baseline excluded by default (cost).")
                    if args.protocol == "cv":
                        continue
                if args.protocol in ("heldout", "both"):
                    split = df["split"].astype(str).str.lower()
                    tm, sm = split.eq("train"), split.eq("test")
                    y_all = y_for(df, task)
                    r = evaluate_once(df.loc[tm, "text"].reset_index(drop=True), y_all.loc[tm].reset_index(drop=True),
                                       df.loc[sm, "text"].reset_index(drop=True), y_all.loc[sm].reset_index(drop=True),
                                       TUNED[task][clf_name], prep, feat, class_labels_for(task))
                    r.update({"protocol": "heldout", "task": task, "classifier": clf_name, "preprocessing": prep, "feature_set": feat})
                    rows.append(r)
                    print(f"[baseline-heldout][{task}] {clf_name} acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f}")
                if args.protocol in ("cv", "both") and not (clf_name == "GB" and task == "six_class" and not args.include_gb_six_class):
                    r = run_cv_for_combo(df, task, clf_name, prep, feat)
                    rows.append(r)
                    print(f"[baseline-cv][{task}] {clf_name} done")
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "common_baseline.csv"), index=False)


if __name__ == "__main__":
    main()
