"""Benchmark SendWise binary and fine-grained risk classification.

Uses the authoritative Categorised_dataset.csv and its preassigned train/test
split. Binary task uses risk_label; six-class task uses risk_category.

Benchmarks run across all classifiers supported by enhanced_experiment.py's
create_model() (RF, SVM, LR, KNN, NB, GB by default) with the shared
CombinedFeatures(preprocessing="lemmatization", use_word=True, use_char=True)
representation — only the classifier varies.

Run:
  python benchmark_categorised.py --task both
  python benchmark_categorised.py --task binary
  python benchmark_categorised.py --task multiclass
  python benchmark_categorised.py --task both --classifiers RF,SVM
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
from sklearn.pipeline import Pipeline

from enhanced_experiment import CombinedFeatures, create_model, RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier


RF_VARIANTS = {
    # name: (n_estimators, max_depth, n_jobs)
    "RF50": (50, None, -1),
    "RF50J1": (50, None, 1),        # isolate n_jobs thread-pool overhead
    "RF20J1": (20, None, 1),
    "RF10J1": (10, None, 1),
    "RF5J1": (5, None, 1),
    "RF3J1": (3, None, 1),
    "RF1J1": (1, None, 1),
    "RF50J1D10": (50, 10, 1),       # also cap depth
    "RF20J1D10": (20, 10, 1),
    "RF200J1": (200, None, 1),      # same trees as original RF, just n_jobs=1
}


def resolve_classifier(clf_name):
    """Resolve a classifier name to an estimator. Adds ad-hoc RF latency-lever
    variants on top of enhanced_experiment.create_model, without modifying
    that canonical function."""
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

def run_task(df, task, clf_name, output_dir, latency_samples, dataset_path):
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

    # Prefer the dataset's documented split. Fall back to a stratified split
    # only if a split column is absent or does not contain train and test.
    if "split" in df.columns and {"train", "test"}.issubset(set(df["split"].astype(str).str.lower())):
        split = df["split"].astype(str).str.lower()
        train_mask, test_mask = split.eq("train"), split.eq("test")
        split_protocol = "preassigned dataset split column (train/test)"
        X_train = df.loc[train_mask, "text"].astype(str).reset_index(drop=True)
        X_test = df.loc[test_mask, "text"].astype(str).reset_index(drop=True)
        y_train = y_all.loc[train_mask].reset_index(drop=True)
        y_test = y_all.loc[test_mask].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].astype(str), y_all, test_size=0.25,
            random_state=42, stratify=y_all
        )
        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
        split_protocol = "stratified 75/25 hold-out (fallback; no valid split column)"

    process = psutil.Process(os.getpid())
    rss_before_mb = mb(process.memory_info().rss)
    # Reuse the project's selected hybrid TF-IDF + lemmatization representation.
    # Only the classifier step varies across runs (RF, SVM, LR, KNN, NB, GB).
    # Note: create_model("GB") returns a Pipeline (TruncatedSVD->GB) itself,
    # which is still a drop-in estimator supporting fit/predict.
    pipeline = Pipeline([
        ("features", CombinedFeatures(preprocessing="lemmatization", use_word=True, use_char=True)),
        ("classifier", resolve_classifier(clf_name)),
    ])

    t0 = time.perf_counter()
    with RSSPeakSampler() as fit_sampler:
        pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    rss_after_fit_mb = mb(process.memory_info().rss)

    # Evaluate full held-out set for classification metrics.
    t0 = time.perf_counter()
    X_test_features = pipeline.named_steps["features"].transform(X_test)
    feature_transform_batch_seconds = time.perf_counter() - t0
    classifier = pipeline.named_steps["classifier"]
    t0 = time.perf_counter()
    with RSSPeakSampler() as clf_sampler:
        y_pred = classifier.predict(X_test_features)
    classifier_batch_seconds = time.perf_counter() - t0

    # Latency timing on individual raw messages (end-to-end).
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

    # Classifier-only per-row timings on already extracted features.
    classifier_ms = []
    for i in range(min(n_latency, X_test_features.shape[0])):
        row = X_test_features[i]
        start = time.perf_counter()
        classifier.predict(row)
        classifier_ms.append((time.perf_counter() - start) * 1000.0)
    classifier_ms = np.asarray(classifier_ms, dtype=float)

    model_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_runtime_model.joblib")
    joblib.dump(pipeline, model_path, compress=3)
    model_bytes = os.path.getsize(model_path)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="weighted", zero_division=0
    )
    # For binary, also report risk-positive metrics; for multiclass, these
    # class-specific scores appear in the per-class report.
    risk_precision = risk_recall = risk_f1 = np.nan
    if task == "binary":
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[1], average=None, zero_division=0
        )
        risk_precision, risk_recall, risk_f1 = float(p[0]), float(r[0]), float(f[0])

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    summary = {
        "task": task_slug,
        "classifier": clf_name,
        "dataset_file": os.path.abspath(dataset_path),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": target_col,
        "split_protocol": split_protocol,
        "classes": "|".join(map(str, class_labels)),
        "features": "lemmatization + hybrid word/character TF-IDF",
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
        "rss_before_fit_mb": rss_before_mb,
        "rss_after_fit_mb": rss_after_fit_mb,
        "fit_peak_rss_mb": mb(fit_sampler.peak),
        "classifier_peak_rss_mb": mb(clf_sampler.peak),
        "end_to_end_peak_rss_mb": mb(e2e_sampler.peak),
        "model_file_bytes": model_bytes,
        "model_file_mb": model_bytes / (1024 ** 2),
        "timing_note": "Python-process measurement; not Android-device latency. RSS includes Python/libraries, not model-only RAM.",
        "status": "ok",
        "error": "",
    }

    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, f"{task_slug}_{clf_name}_benchmark_summary.csv"), index=False)
    pd.DataFrame({
        "test_row_position": np.arange(1, len(e2e_ms) + 1),
        "end_to_end_latency_ms": e2e_ms,
        "classifier_only_latency_ms": classifier_ms,
    }).to_csv(os.path.join(output_dir, f"{task_slug}_{clf_name}_benchmark_per_message.csv"), index=False)

    report = classification_report(
        y_test, y_pred, labels=class_labels, target_names=target_names,
        zero_division=0, digits=4
    )
    with open(os.path.join(output_dir, f"{task_slug}_{clf_name}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Task: {task_slug}\nClassifier: {clf_name}\nDataset rows: {len(df)}\nTrain rows: {len(X_train)}\nTest rows: {len(X_test)}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, columns=predicted; order follows class list):\n")
        f.write(np.array2string(cm))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.index.name = "true_label"
    cm_df.to_csv(os.path.join(output_dir, f"{task_slug}_{clf_name}_confusion_matrix.csv"))

    print(f"\n=== {task_slug.upper()} / {clf_name} BENCHMARK COMPLETE ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Summary: {os.path.join(output_dir, task_slug + '_' + clf_name + '_benchmark_summary.csv')}")
    print(f"Report: {os.path.join(output_dir, task_slug + '_' + clf_name + '_classification_report.txt')}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Benchmark binary and six-class SendWise risk detection")
    parser.add_argument("--dataset", default="Categorised_dataset.csv", help="Path to categorised dataset CSV")
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--classifiers", default="RF,SVM,LR,KNN,NB,GB",
                         help="Comma-separated classifier names understood by enhanced_experiment.create_model")
    parser.add_argument("--latency-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="results/categorised_benchmark")
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
            print(f"\n----- Running task={task} classifier={clf_name} -----")
            try:
                summary = run_task(df, task, clf_name, args.output_dir, args.latency_samples, args.dataset)
                summaries.append(summary)
            except Exception as exc:
                print(f"ERROR: task={task} classifier={clf_name} failed: {exc}")
                traceback.print_exc()
                task_slug = "binary" if task == "binary" else "six_class"
                summaries.append({
                    "task": task_slug,
                    "classifier": clf_name,
                    "status": "failed",
                    "error": str(exc),
                })
                continue

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.output_dir, "all_algorithms_benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCombined algorithm comparison summary: {summary_path}")
    print("\nAll requested tasks completed. Outputs are in:", args.output_dir)

if __name__ == "__main__":
    main()
