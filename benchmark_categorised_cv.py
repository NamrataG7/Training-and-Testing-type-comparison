"""Stratified 5-fold cross-validation benchmark for SendWise binary and
fine-grained risk classification.

Unlike benchmark_categorised.py (single preassigned held-out split), this
script runs StratifiedKFold(n_splits=5) over the FULL dataset for each
(task, classifier) combination, fitting and evaluating a fresh pipeline per
fold, then aggregates metrics (mean/std) across folds plus a summed
confusion matrix.

Benchmarks run across all classifiers supported by enhanced_experiment.py's
create_model() (RF, SVM, LR, KNN, NB, GB by default) with the shared
CombinedFeatures(preprocessing="lemmatization", use_word=True, use_char=True)
representation — only the classifier varies.

Run:
  python benchmark_categorised_cv.py --task both
  python benchmark_categorised_cv.py --task binary
  python benchmark_categorised_cv.py --task multiclass
  python benchmark_categorised_cv.py --task both --classifiers RF,SVM
"""
import argparse
import os
import time
import traceback
import threading
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from enhanced_experiment import CombinedFeatures, create_model, RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD

# Ad-hoc RF latency-lever variants (on top of enhanced_experiment.create_model,
# which is left untouched). n_jobs=1 avoids per-single-message thread-pool
# spawn overhead; n_estimators is reduced from the original 200.
RF_VARIANTS = {
    "RF10J1": (10, None, 1),
    "RF20J1": (20, None, 1),
}

# Exact winning hyperparameters from hyperparameter_search.py's validation-split
# search (results/hyperparam_search + results/hyperparam_search_gb), reproduced
# here to run the SAME configs under 5-fold CV. Keyed as "{CLF}_BINTUNED" /
# "{CLF}_SIXTUNED" since the winning config differs per task.
TUNED_CONFIGS = {
    "RF_BINTUNED": lambda: RandomForestClassifier(
        n_estimators=10, max_depth=20, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE),
    "RF_SIXTUNED": lambda: RandomForestClassifier(
        n_estimators=10, max_depth=None, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE),
    "LR_BINTUNED": lambda: LogisticRegression(
        C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    "LR_SIXTUNED": lambda: LogisticRegression(
        C=10.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    "KNN_BINTUNED": lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=1),
    "KNN_SIXTUNED": lambda: KNeighborsClassifier(n_neighbors=3, n_jobs=1),
    "NB_BINTUNED": lambda: MultinomialNB(alpha=0.1),
    "NB_SIXTUNED": lambda: MultinomialNB(alpha=0.1),
    "SVM_BINTUNED": lambda: SVC(kernel="linear", C=10.0, class_weight="balanced"),
    "SVM_SIXTUNED": lambda: SVC(kernel="linear", C=10.0, class_weight="balanced"),
    "GB_BINTUNED": lambda: Pipeline([
        ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
        ("gb", GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)),
    ]),
    "GB_SIXTUNED": lambda: Pipeline([
        ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)),
    ]),
}


def resolve_classifier(clf_name):
    if clf_name in TUNED_CONFIGS:
        return TUNED_CONFIGS[clf_name]()
    if clf_name in RF_VARIANTS:
        n_estimators, max_depth, n_jobs = RF_VARIANTS[clf_name]
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        )
    return create_model(clf_name)

warnings.filterwarnings("ignore")

CATEGORY_ORDER = [
    "non_risk", "harassment", "threats", "hate_speech",
    "sexual_content", "self_harm_risk"
]
CATEGORY_DISPLAY = [
    "Non-risk", "Harassment", "Threats", "Hate speech",
    "Sexual content", "Self-harm risk"
]


class RSSPeakSampler:
    def __init__(self, interval=0.005):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = None
    def _sample(self):
        while not self.stop_event.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self.stop_event.wait(self.interval)
    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join()
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass


def mb(n):
    return n / (1024 ** 2)


def run_fold(fold_idx, X_train, X_test, y_train, y_test, task_slug, clf_name,
             class_labels, target_names, latency_samples, output_dir, save_model):
    process = psutil.Process(os.getpid())

    pipeline = Pipeline([
        ("features", CombinedFeatures(preprocessing="lemmatization", use_word=True, use_char=True)),
        ("classifier", resolve_classifier(clf_name)),
    ])

    t0 = time.perf_counter()
    with RSSPeakSampler() as fit_sampler:
        pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test_features = pipeline.named_steps["features"].transform(X_test)
    feature_transform_batch_seconds = time.perf_counter() - t0
    classifier = pipeline.named_steps["classifier"]
    t0 = time.perf_counter()
    with RSSPeakSampler() as clf_sampler:
        y_pred = classifier.predict(X_test_features)
    classifier_batch_seconds = time.perf_counter() - t0

    n_latency = min(max(1, latency_samples), len(X_test))
    sample_texts = X_test.iloc[:n_latency].tolist()
    for txt in sample_texts[:min(5, len(sample_texts))]:
        pipeline.predict([txt])
    e2e_ms = []
    with RSSPeakSampler() as e2e_sampler:
        for txt in sample_texts:
            start = time.perf_counter()
            pipeline.predict([txt])
            e2e_ms.append((time.perf_counter() - start) * 1000.0)
    e2e_ms = np.asarray(e2e_ms, dtype=float)

    classifier_ms = []
    for i in range(min(n_latency, X_test_features.shape[0])):
        row = X_test_features[i]
        start = time.perf_counter()
        classifier.predict(row)
        classifier_ms.append((time.perf_counter() - start) * 1000.0)
    classifier_ms = np.asarray(classifier_ms, dtype=float)

    # Only persist a model file for the last fold to avoid disk bloat; model
    # size is still recorded per fold (re-measured each time a model is saved,
    # and reused for folds where we skip saving so the aggregate has a value
    # for every fold).
    model_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_cv_last_fold_model.joblib")
    if save_model:
        joblib.dump(pipeline, model_path, compress=3)
    model_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else np.nan

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="weighted", zero_division=0
    )
    risk_precision = risk_recall = risk_f1 = np.nan
    if task_slug == "binary":
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[1], average=None, zero_division=0
        )
        risk_precision, risk_recall, risk_f1 = float(p[0]), float(r[0]), float(f[0])

    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average=None, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    fold_result = {
        "fold": fold_idx,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fit_time_seconds": fit_seconds,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "risk_precision_positive_class": risk_precision,
        "risk_recall_positive_class": risk_recall,
        "risk_f1_positive_class": risk_f1,
        "test_feature_transform_batch_seconds": feature_transform_batch_seconds,
        "classifier_batch_prediction_seconds": classifier_batch_seconds,
        "classifier_mean_latency_ms_per_message": float(np.mean(classifier_ms)),
        "classifier_median_latency_ms_per_message": float(np.median(classifier_ms)),
        "end_to_end_mean_latency_ms_per_message": float(np.mean(e2e_ms)),
        "end_to_end_median_latency_ms_per_message": float(np.median(e2e_ms)),
        "end_to_end_p95_latency_ms": float(np.percentile(e2e_ms, 95)),
        "fit_peak_rss_mb": mb(fit_sampler.peak),
        "classifier_peak_rss_mb": mb(clf_sampler.peak),
        "end_to_end_peak_rss_mb": mb(e2e_sampler.peak),
        "model_file_mb": (model_bytes / (1024 ** 2)) if not np.isnan(model_bytes) else np.nan,
    }
    for name, p, r, f1v in zip(target_names, per_class_precision, per_class_recall, per_class_f1):
        key = name.lower().replace(" ", "_").replace("-", "_")
        fold_result[f"{key}_precision"] = float(p)
        fold_result[f"{key}_recall"] = float(r)
        fold_result[f"{key}_f1"] = float(f1v)

    report = classification_report(
        y_test, y_pred, labels=class_labels, target_names=target_names,
        zero_division=0, digits=4
    )
    return fold_result, cm, report


def run_task(df, task, clf_name, output_dir, latency_samples, n_splits, args):
    if task == "binary":
        target_col = "risk_label"
        target_names = ["Non-risk", "Risk detected"]
        task_slug = "binary"
        y_all = pd.to_numeric(df[target_col], errors="raise").astype(int)
        if not set(y_all.unique()).issubset({0, 1}):
            raise ValueError("risk_label must contain only 0 and 1.")
        class_labels = [0, 1]
    else:
        target_col = "risk_category"
        task_slug = "six_class"
        y_all = df[target_col].astype(str)
        class_labels = CATEGORY_ORDER
        target_names = CATEGORY_DISPLAY
        unexpected = sorted(set(y_all.unique()) - set(CATEGORY_ORDER))
        if unexpected:
            raise ValueError(f"Unexpected risk_category values: {unexpected}")

    X_all = df["text"].astype(str).reset_index(drop=True)
    y_all = y_all.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    reports = []
    cm_sum = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)

    folds = list(skf.split(X_all, y_all))
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        X_test = X_all.iloc[test_idx].reset_index(drop=True)
        y_train = y_all.iloc[train_idx].reset_index(drop=True)
        y_test = y_all.iloc[test_idx].reset_index(drop=True)

        save_model = (i == len(folds))  # only persist last fold's model
        fold_result, cm, report = run_fold(
            i, X_train, X_test, y_train, y_test, task_slug, clf_name, class_labels,
            target_names, latency_samples, output_dir, save_model
        )
        fold_results.append(fold_result)
        cm_sum += cm
        reports.append(f"=== Fold {i} ===\n{report}\n")

        print(f"[{task_slug}/{clf_name}] fold {i}/{len(folds)} "
              f"accuracy={fold_result['accuracy']:.4f} "
              f"macro_f1={fold_result['macro_f1']:.4f} "
              f"fit_time={fold_result['fit_time_seconds']:.2f}s")

    per_fold_df = pd.DataFrame(fold_results)
    per_fold_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_cv_per_fold_summary.csv")
    per_fold_df.to_csv(per_fold_path, index=False)

    # Aggregate mean/std across folds for every numeric column (except "fold").
    numeric_cols = [c for c in per_fold_df.columns if c != "fold"]
    agg = {}
    for col in numeric_cols:
        agg[f"mean_{col}"] = float(np.nanmean(per_fold_df[col].values))
        agg[f"std_{col}"] = float(np.nanstd(per_fold_df[col].values))
    agg_meta = {
        "task": task_slug,
        "classifier": clf_name,
        "dataset_file": os.path.abspath(args.dataset),
        "dataset_rows": int(len(df)),
        "n_splits": n_splits,
        "features": "lemmatization + hybrid word/character TF-IDF",
        "split_protocol": "stratified 5-fold cross-validation",
        "status": "ok",
        "error": "",
    }
    agg_row = {**agg_meta, **agg}
    agg_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_cv_aggregate_summary.csv")
    pd.DataFrame([agg_row]).to_csv(agg_path, index=False)

    cm_df = pd.DataFrame(cm_sum, index=target_names, columns=target_names)
    cm_df.index.name = "true_label"
    cm_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_cv_confusion_matrix_aggregated.csv")
    cm_df.to_csv(cm_path)

    report_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_cv_classification_report_per_fold.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_slug}\nClassifier: {clf_name}\nDataset rows: {len(df)}\nFolds: {n_splits}\n\n")
        f.write("\n".join(reports))

    print(f"\n=== {task_slug.upper()} / {clf_name} CV BENCHMARK COMPLETE ===")
    print(f"Mean accuracy: {agg_row['mean_accuracy']:.4f} +/- {agg_row['std_accuracy']:.4f}")
    print(f"Mean macro F1: {agg_row['mean_macro_f1']:.4f} +/- {agg_row['std_macro_f1']:.4f}")
    print(f"Per-fold summary: {per_fold_path}")
    print(f"Aggregate summary: {agg_path}")
    print(f"Confusion matrix (aggregated): {cm_path}")
    print(f"Classification reports: {report_path}")

    return agg_row


def main():
    global args
    parser = argparse.ArgumentParser(
        description="Stratified 5-fold CV benchmark for binary and six-class SendWise risk detection"
    )
    parser.add_argument("--dataset", default="Categorised_dataset.csv", help="Path to categorised dataset CSV")
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--classifiers", default="RF,SVM,LR,KNN,NB,GB",
                         help="Comma-separated classifier names understood by enhanced_experiment.create_model")
    parser.add_argument("--latency-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="results/categorised_benchmark_cv")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.dataset)
    required = {"text", "risk_label", "risk_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "risk_label", "risk_category"]).copy()
    df["text"] = df["text"].astype(str)
    print(f"Dataset loaded: {len(df):,} rows from {args.dataset}")
    print("Risk category distribution:")
    print(df["risk_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0).to_string())

    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    print(f"Classifiers to benchmark: {classifiers}")

    summaries = []
    for task in tasks:
        for clf_name in classifiers:
            print(f"\n----- Running CV task={task} classifier={clf_name} -----")
            task_slug = "binary" if task == "binary" else "six_class"
            try:
                agg_row = run_task(df, task, clf_name, args.output_dir, args.latency_samples, args.n_splits, args)
                summaries.append(agg_row)
            except Exception as exc:
                print(f"ERROR: CV task={task} classifier={clf_name} failed: {exc}")
                traceback.print_exc()
                summaries.append({
                    "task": task_slug,
                    "classifier": clf_name,
                    "status": "failed",
                    "error": str(exc),
                })
                continue

    combined_path = os.path.join(args.output_dir, "all_algorithms_cv_benchmark_summary.csv")
    pd.DataFrame(summaries).to_csv(combined_path, index=False)
    print(f"\nCombined algorithm CV comparison summary: {combined_path}")

    print("\nAll requested CV tasks completed. Outputs are in:", args.output_dir)


if __name__ == "__main__":
    main()
