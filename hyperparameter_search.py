"""Legitimate, validation-split-based hyperparameter search for all 6
classifiers (RF, SVM, LR, KNN, NB, GB) on both the binary (risk_label) and
six-class (risk_category) tasks.

Protocol:
  - Load Categorised_dataset.csv and use its authoritative `split` column.
    Rows with split=="train" are available for tuning; rows with
    split=="test" are held out and used EXACTLY ONCE, at the very end, to
    evaluate the winning config per (classifier, task).
  - Within the train rows, carve out an inner stratified 80/20
    train/validation split (random_state=42) used only for hyperparameter
    selection (separately per task, since class balance differs).
  - For each (task, classifier) pair, try every hyperparameter combo in
    GRIDS, fit on inner-train, score macro-F1 on inner-val. Pick the best
    combo (ties broken by faster fit time).
  - Refit the winning config on the FULL train split and evaluate ONCE on
    the untouched test split, reusing benchmark_categorised.py's exact
    metrics/latency/RSS measurement approach (RSSPeakSampler, end-to-end
    per-message latency with 5-message warmup, classifier-only latency,
    confusion matrix, model size).

The CombinedFeatures(preprocessing="lemmatization", use_word=True,
use_char=True) representation is fixed and not searched over -- only
classifier hyperparameters vary.

Run:
  python hyperparameter_search.py --task both --classifiers RF,SVM,LR,KNN,NB,GB
  python hyperparameter_search.py --task binary --classifiers NB
"""
import argparse
import json
import os
import time
import traceback
import threading
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from enhanced_experiment import CombinedFeatures

warnings.filterwarnings("ignore")

CATEGORY_ORDER = [
    "non_risk", "harassment", "threats", "hate_speech",
    "sexual_content", "self_harm_risk"
]
CATEGORY_DISPLAY = [
    "Non-risk", "Harassment", "Threats", "Hate speech",
    "Sexual content", "Self-harm risk"
]

# ---------------------------------------------------------------------------
# Hyperparameter grids -- kept intentionally small. GB and SVM are expensive;
# do not expand these without re-budgeting the run time.
# ---------------------------------------------------------------------------
GRIDS = {
    "RF": [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": ml, "max_features": "sqrt",
         "class_weight": "balanced", "n_jobs": 1, "random_state": 42}
        for n in [10, 20, 50]
        for d in [None, 20]
        for ml in [1, 2]
    ],  # 12 combos
    "LR": [
        {"C": c, "penalty": "l2", "class_weight": "balanced", "max_iter": 2000, "random_state": 42}
        for c in [0.1, 1.0, 10.0]
    ],  # 3 combos
    "SVM": [
        {"kernel": "linear", "C": c, "class_weight": "balanced"}
        for c in [0.1, 1.0, 10.0]
    ],  # 3 combos -- SLOW (libsvm SVC)
    "KNN": [
        {"n_neighbors": k, "n_jobs": 1}
        for k in [3, 5, 10, 15]
    ],  # 4 combos
    "NB": [
        {"alpha": a}
        for a in [0.1, 0.5, 1.0, 2.0]
    ],  # 4 combos
    "GB": [
        {"n_estimators": n, "learning_rate": 0.1, "max_depth": d, "random_state": 42}
        for n in [50, 100]
        for d in [2, 3]
    ],  # 4 combos -- VERY SLOW, especially on six-class
}


def build_estimator(clf_name, params):
    """Construct a fresh classifier (or classifier pipeline for GB) from a
    hyperparameter dict. Mirrors enhanced_experiment.create_model's
    construction choices but is NOT imported from there, per spec."""
    if clf_name == "RF":
        return RandomForestClassifier(**params)
    if clf_name == "LR":
        return LogisticRegression(**params)
    if clf_name == "SVM":
        return SVC(**params)
    if clf_name == "KNN":
        return KNeighborsClassifier(**params)
    if clf_name == "NB":
        return MultinomialNB(**params)
    if clf_name == "GB":
        return Pipeline([
            ("svd", TruncatedSVD(n_components=200, random_state=42)),
            ("gb", GradientBoostingClassifier(**params)),
        ])
    raise ValueError(f"Unknown classifier: {clf_name}")


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


def task_config(task, df):
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
    return target_col, task_slug, class_labels, target_names, y_all


def search_classifier(task_slug, clf_name, X_inner_train, y_inner_train,
                       X_inner_val, y_inner_val, class_labels, output_dir):
    grid = GRIDS[clf_name]
    rows = []
    best = None  # (val_macro_f1, -fit_time, params, combo_idx)

    for idx, params in enumerate(grid, start=1):
        row = dict(params)
        try:
            pipeline = Pipeline([
                ("features", CombinedFeatures(preprocessing="lemmatization", use_word=True, use_char=True)),
                ("classifier", build_estimator(clf_name, params)),
            ])
            t0 = time.perf_counter()
            pipeline.fit(X_inner_train, y_inner_train)
            fit_time = time.perf_counter() - t0
            y_pred = pipeline.predict(X_inner_val)
            _, _, f1_macro, _ = precision_recall_fscore_support(
                y_inner_val, y_pred, labels=class_labels, average="macro", zero_division=0
            )
            acc = accuracy_score(y_inner_val, y_pred)
            row["val_macro_f1"] = float(f1_macro)
            row["val_accuracy"] = float(acc)
            row["fit_time_seconds"] = fit_time
            row["error"] = ""

            candidate_key = (float(f1_macro), -fit_time)
            if best is None or candidate_key > (best[0], -best[1]):
                best = (float(f1_macro), fit_time, params, idx)

            print(f"[task={task_slug} clf={clf_name}] combo {idx}/{len(grid)} "
                  f"params={params} val_macro_f1={f1_macro:.4f} fit_time={fit_time:.2f}s")
        except Exception as exc:
            row["val_macro_f1"] = np.nan
            row["val_accuracy"] = np.nan
            row["fit_time_seconds"] = np.nan
            row["error"] = str(exc)
            print(f"[task={task_slug} clf={clf_name}] combo {idx}/{len(grid)} FAILED: {exc}")
            traceback.print_exc()
        rows.append(row)

    log_df = pd.DataFrame(rows)
    log_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_search_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"Search log written: {log_path}")

    if best is None:
        raise RuntimeError(f"All hyperparameter combos failed for task={task_slug} clf={clf_name}")

    best_val_f1, best_fit_time, best_params, best_idx = best
    print(f"[task={task_slug} clf={clf_name}] BEST combo #{best_idx}: {best_params} "
          f"val_macro_f1={best_val_f1:.4f}")
    return best_params, best_val_f1


def final_evaluate(task, task_slug, clf_name, best_params, best_val_f1,
                    X_train_full, y_train_full, X_test, y_test, class_labels,
                    target_names, output_dir, latency_samples, dataset_path, df):
    process = psutil.Process(os.getpid())
    rss_before_mb = mb(process.memory_info().rss)

    pipeline = Pipeline([
        ("features", CombinedFeatures(preprocessing="lemmatization", use_word=True, use_char=True)),
        ("classifier", build_estimator(clf_name, best_params)),
    ])

    t0 = time.perf_counter()
    with RSSPeakSampler() as fit_sampler:
        pipeline.fit(X_train_full, y_train_full)
    fit_seconds = time.perf_counter() - t0
    rss_after_fit_mb = mb(process.memory_info().rss)

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

    model_path = os.path.join(output_dir, f"{task_slug}_{clf_name}_final_runtime_model.joblib")
    joblib.dump(pipeline, model_path, compress=3)
    model_bytes = os.path.getsize(model_path)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="weighted", zero_division=0
    )
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
        "train_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "target_column": "risk_label" if task == "binary" else "risk_category",
        "split_protocol": "preassigned dataset split column (train/test); winning config chosen via inner 80/20 train/val split on train rows only",
        "classes": "|".join(map(str, class_labels)),
        "features": "lemmatization + hybrid word/character TF-IDF",
        "best_hyperparams": json.dumps(best_params),
        "validation_macro_f1": best_val_f1,
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

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{task_slug}_{clf_name}_final_benchmark_summary.csv"), index=False
    )
    pd.DataFrame({
        "test_row_position": np.arange(1, len(e2e_ms) + 1),
        "end_to_end_latency_ms": e2e_ms,
        "classifier_only_latency_ms": classifier_ms,
    }).to_csv(os.path.join(output_dir, f"{task_slug}_{clf_name}_final_per_message.csv"), index=False)

    report = classification_report(
        y_test, y_pred, labels=class_labels, target_names=target_names,
        zero_division=0, digits=4
    )
    with open(os.path.join(output_dir, f"{task_slug}_{clf_name}_final_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Task: {task_slug}\nClassifier: {clf_name}\nBest hyperparams: {best_params}\n"
                f"Validation macro-F1: {best_val_f1:.4f}\n"
                f"Train rows: {len(X_train_full)}\nTest rows: {len(X_test)}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, columns=predicted; order follows class list):\n")
        f.write(np.array2string(cm))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.index.name = "true_label"
    cm_df.to_csv(os.path.join(output_dir, f"{task_slug}_{clf_name}_final_confusion_matrix.csv"))

    with open(os.path.join(output_dir, f"{task_slug}_{clf_name}_best_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": task_slug,
            "classifier": clf_name,
            "best_hyperparams": best_params,
            "validation_macro_f1": best_val_f1,
        }, f, indent=2)

    print(f"\n=== {task_slug.upper()} / {clf_name} FINAL TEST EVALUATION COMPLETE ===")
    print(f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f} "
          f"fit_time={summary['fit_time_seconds']:.2f}s")
    return summary


def run_task_classifier(df, task, clf_name, output_dir, latency_samples, dataset_path):
    target_col, task_slug, class_labels, target_names, y_all = task_config(task, df)

    split = df["split"].astype(str).str.lower()
    if not {"train", "test"}.issubset(set(split.unique())):
        raise ValueError("Dataset's split column must contain both 'train' and 'test' values.")
    train_mask, test_mask = split.eq("train"), split.eq("test")

    X_train_full = df.loc[train_mask, "text"].astype(str).reset_index(drop=True)
    y_train_full = y_all.loc[train_mask].reset_index(drop=True)
    X_test = df.loc[test_mask, "text"].astype(str).reset_index(drop=True)
    y_test = y_all.loc[test_mask].reset_index(drop=True)

    # Inner stratified 80/20 split of the TRAIN rows only, for model
    # selection. The test split above is never touched until final eval.
    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    X_inner_train = X_inner_train.reset_index(drop=True)
    X_inner_val = X_inner_val.reset_index(drop=True)
    y_inner_train = y_inner_train.reset_index(drop=True)
    y_inner_val = y_inner_val.reset_index(drop=True)

    best_params, best_val_f1 = search_classifier(
        task_slug, clf_name, X_inner_train, y_inner_train, X_inner_val, y_inner_val,
        class_labels, output_dir
    )

    summary = final_evaluate(
        task, task_slug, clf_name, best_params, best_val_f1,
        X_train_full, y_train_full, X_test, y_test, class_labels, target_names,
        output_dir, latency_samples, dataset_path, df
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validation-split-based hyperparameter search for SendWise risk classifiers"
    )
    parser.add_argument("--dataset", default="Categorised_dataset.csv", help="Path to categorised dataset CSV")
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--classifiers", default="RF,SVM,LR,KNN,NB,GB",
                         help="Comma-separated classifier names (subset of RF,SVM,LR,KNN,NB,GB)")
    parser.add_argument("--latency-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="results/hyperparam_search")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.dataset)
    required = {"text", "risk_label", "risk_category", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "risk_label", "risk_category", "split"]).copy()
    df["text"] = df["text"].astype(str)
    print(f"Dataset loaded: {len(df):,} rows from {args.dataset}")
    print(df["split"].str.lower().value_counts().to_string())
    print("Risk category distribution:")
    print(df["risk_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0).to_string())

    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    unknown = sorted(set(classifiers) - set(GRIDS.keys()))
    if unknown:
        raise ValueError(f"Unknown classifier names: {unknown}. Must be subset of {sorted(GRIDS.keys())}")
    print(f"Classifiers to search: {classifiers}")

    summaries = []
    for task in tasks:
        task_slug = "binary" if task == "binary" else "six_class"
        for clf_name in classifiers:
            print(f"\n===== Hyperparameter search: task={task} classifier={clf_name} =====")
            try:
                summary = run_task_classifier(df, task, clf_name, args.output_dir, args.latency_samples, args.dataset)
                summaries.append(summary)
            except Exception as exc:
                print(f"ERROR: task={task} classifier={clf_name} failed entirely: {exc}")
                traceback.print_exc()
                summaries.append({
                    "task": task_slug,
                    "classifier": clf_name,
                    "status": "failed",
                    "error": str(exc),
                })
                continue

    combined_path = os.path.join(args.output_dir, "all_best_configs_final_summary.csv")
    pd.DataFrame(summaries).to_csv(combined_path, index=False)
    print(f"\nCombined best-config final summary: {combined_path}")
    print("\nAll requested hyperparameter searches completed. Outputs are in:", args.output_dir)


if __name__ == "__main__":
    main()
