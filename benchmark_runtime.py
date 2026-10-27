"""Runtime/resource benchmark for SendWise's classical ML research pipeline.

This is an ADDITIONAL hold-out benchmark. It does not overwrite or change the
repository's existing stratified 5-fold CV results.

Run from this directory:
    python benchmark_runtime.py --dataset Updated_dataset.csv

Outputs:
    results/runtime_benchmark_summary.csv
    results/runtime_benchmark_per_message.csv
    results/runtime_benchmark_model.joblib
"""
import argparse
import json
import os
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from enhanced_experiment import CombinedFeatures, create_model, RANDOM_STATE

warnings.filterwarnings("ignore")


class RSSPeakSampler:
    """Sample process RSS while a timed operation is running."""
    def __init__(self, interval=0.005):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark model runtime and resource use")
    parser.add_argument("--dataset", default="Updated_dataset.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--latency-samples", type=int, default=500,
                        help="Number of test messages to time individually; capped by test-set size")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.dataset)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    df = df.dropna(subset=["label"]).copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("This benchmark expects binary labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    process = psutil.Process(os.getpid())
    rss_start_mb = process.memory_info().rss / (1024 ** 2)

    # E2 hybrid word + character TF-IDF with lemmatization, matching the
    # repository's selected representation; RF parameters come from its
    # create_model("RF") implementation.
    pipeline = Pipeline([
        ("features", CombinedFeatures(
            preprocessing="lemmatization", use_word=True, use_char=True
        )),
        ("classifier", create_model("RF")),
    ])

    fit_start = time.perf_counter()
    with RSSPeakSampler() as fit_memory:
        pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    rss_after_fit_mb = process.memory_info().rss / (1024 ** 2)
    fit_peak_mb = fit_memory.peak / (1024 ** 2)

    # Batch prediction supports the classification metrics.
    pred_start = time.perf_counter()
    X_test_features = pipeline.named_steps["features"].transform(X_test)
    feature_transform_seconds = time.perf_counter() - pred_start

    classifier = pipeline.named_steps["classifier"]
    clf_start = time.perf_counter()
    with RSSPeakSampler() as clf_memory:
        y_pred = classifier.predict(X_test_features)
    classifier_batch_seconds = time.perf_counter() - clf_start

    # End-to-end single-message latency: raw text through preprocessing,
    # word+character TF-IDF and RF prediction. Warm up before timing.
    n_latency = min(max(1, args.latency_samples), len(X_test))
    latency_texts = X_test.iloc[:n_latency].tolist()
    for text in latency_texts[:min(5, len(latency_texts))]:
        pipeline.predict([text])

    e2e_ms = []
    with RSSPeakSampler() as e2e_memory:
        for text in latency_texts:
            t0 = time.perf_counter()
            pipeline.predict([text])
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
    e2e_ms = np.asarray(e2e_ms, dtype=float)

    # Classifier-only latency on already-extracted features. This is measured
    # separately from end-to-end latency.
    n_clf = min(n_latency, X_test_features.shape[0])
    classifier_ms = []
    for i in range(n_clf):
        row = X_test_features[i]
        t0 = time.perf_counter()
        classifier.predict(row)
        classifier_ms.append((time.perf_counter() - t0) * 1000.0)
    classifier_ms = np.asarray(classifier_ms, dtype=float)

    model_path = os.path.join(args.output_dir, "runtime_benchmark_model.joblib")
    joblib.dump(pipeline, model_path, compress=3)
    model_size_bytes = os.path.getsize(model_path)

    # Explicit positive label 1 = abusive, matching dataset documentation.
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "abusive_precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "abusive_recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "abusive_f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    summary = {
        "dataset_path": os.path.abspath(args.dataset),
        "dataset_rows_used": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
        "split_protocol": f"stratified hold-out ({1-args.test_size:.0%} train/{args.test_size:.0%} test)",
        "preprocessing": "lemmatization",
        "features": "hybrid word + character TF-IDF",
        "classifier": "RandomForest (repository create_model RF parameters)",
        "fit_time_seconds": fit_seconds,
        "test_feature_transform_batch_seconds": feature_transform_seconds,
        "classifier_batch_prediction_seconds": classifier_batch_seconds,
        "classifier_mean_latency_ms_per_message": float(np.mean(classifier_ms)),
        "classifier_median_latency_ms_per_message": float(np.median(classifier_ms)),
        "end_to_end_mean_latency_ms_per_message": float(np.mean(e2e_ms)),
        "end_to_end_median_latency_ms_per_message": float(np.median(e2e_ms)),
        "end_to_end_p95_latency_ms": float(np.percentile(e2e_ms, 95)),
        "rss_at_start_mb": rss_start_mb,
        "rss_after_fit_mb": rss_after_fit_mb,
        "fit_peak_rss_mb": fit_peak_mb,
        "classifier_peak_rss_mb": clf_memory.peak / (1024 ** 2),
        "end_to_end_peak_rss_mb": e2e_memory.peak / (1024 ** 2),
        "model_file_bytes": model_size_bytes,
        "model_file_mb": model_size_bytes / (1024 ** 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        **metrics,
        "timing_note": "Python process; end-to-end includes repository preprocessing, hybrid TF-IDF and classifier; not Android-device latency.",
    }

    summary_path = os.path.join(args.output_dir, "runtime_benchmark_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    per_message_path = os.path.join(args.output_dir, "runtime_benchmark_per_message.csv")
    pd.DataFrame({
        "test_row_position": np.arange(1, len(e2e_ms) + 1),
        "end_to_end_latency_ms": e2e_ms,
        "classifier_only_latency_ms": classifier_ms,
    }).to_csv(per_message_path, index=False)

    report_path = os.path.join(args.output_dir, "runtime_benchmark_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred,
                                     target_names=["Non-abusive", "Abusive"],
                                     zero_division=0))
        f.write("\nConfusion matrix (labels [0, 1]):\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred, labels=[0, 1])))

    print("\nRuntime benchmark complete. This is a new hold-out evaluation, not the existing 5-fold CV result.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSummary CSV: {summary_path}")
    print(f"Per-message CSV: {per_message_path}")
    print(f"Model artifact: {model_path}")
    print(f"Classification report: {report_path}")
    print("\nNote: run on the target Android phone separately for actual mobile runtime figures.")


if __name__ == "__main__":
    main()
