import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATASET = "Updated_dataset.csv"
RANDOM_STATE = 42
N_SPLITS = 5

TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    max_features=10000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)
)

GB_SVD_COMPONENTS = 200

RF_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Specificity": specificity,
        "Recall": recall,
        "Macro-F1": macro_f1
    }


def make_classifier(name):
    if name == "SVM":
        return SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced"
        )

    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )

    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if name == "LR":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    if name == "NB":
        return MultinomialNB(alpha=1.0)

    if name == "GB":
        return Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**TFIDF_PARAMS)
            ),
            (
                "svd",
                TruncatedSVD(
                    n_components=GB_SVD_COMPONENTS,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=RANDOM_STATE
                )
            )
        ])

    raise ValueError(f"Unknown classifier: {name}")


def make_text_pipeline(name):
    # GB uses its own TF-IDF -> TruncatedSVD -> GB pipeline directly
    # (dense features are needed for GradientBoostingClassifier).
    # The other classifiers share a common TF-IDF -> classifier pipeline.
    if name == "GB":
        return make_classifier(name)
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("classifier", make_classifier(name))
    ])


df = pd.read_csv(DATASET)
df["text"] = df["text"].fillna("").astype(str)
df["label"] = pd.to_numeric(df["label"]).astype(int)

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nDataset validation passed.")

print("\nAbusive:", int((df["label"] == 1).sum()))
print("Non-abusive:", int((df["label"] == 0).sum()))
print(
    "Abusive %:",
    round((df["label"] == 1).mean() * 100, 2)
)

X = df["text"]
y = df["label"].values

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = {}

# ------------------------------------------------------------
# SVM, KNN, LR, NB and GB
# ------------------------------------------------------------

for model_name in ["SVM", "KNN", "LR", "NB", "GB"]:

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y), start=1
    ):

        print(f"\nStarting Fold {fold}...")

        model = make_text_pipeline(model_name)

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        prediction = model.predict(
            X.iloc[test_idx]
        )

        fold_metrics = calculate_metrics(
            y[test_idx],
            prediction
        )

        fold_results.append(fold_metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={fold_metrics['Accuracy']:.4f}, "
            f"Precision={fold_metrics['Precision']:.4f}, "
            f"Specificity={fold_metrics['Specificity']:.4f}, "
            f"Recall={fold_metrics['Recall']:.4f}, "
            f"Macro-F1={fold_metrics['Macro-F1']:.4f}"
        )

    mean_result = pd.DataFrame(
        fold_results
    ).mean()

    results[model_name] = mean_result

    print("\n" + "-" * 70)
    print(f"{model_name} MEAN RESULTS")
    print("-" * 70)

    for metric in [
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Macro-F1"
    ]:
        print(
            f"{metric:<12}: "
            f"{mean_result[metric]:.4f}"
        )


# ------------------------------------------------------------
# Random Forest
#
# RF configuration is selected only from the training data.
# The threshold is selected on an inner validation split and
# then applied unchanged to the outer test fold.
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RF")
print("=" * 70)

rf_configs = [
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },
    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },
    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": {0: 1, 1: 2}
    },
    {
        "n_estimators": 300,
        "max_depth": 40,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "class_weight": {0: 1, 1: 2}
    }
]

# Configuration selection using inner validation.
config_scores = []

for config in rf_configs:

    inner_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y), start=1
    ):

        inner_train_idx, validation_idx = train_test_split(
            train_idx,
            test_size=0.20,
            stratify=y[train_idx],
            random_state=RANDOM_STATE + fold
        )

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**TFIDF_PARAMS)
            ),
            (
                "rf",
                RandomForestClassifier(
                    **config,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ])

        model.fit(
            X.iloc[inner_train_idx],
            y[inner_train_idx]
        )

        probabilities = model.predict_proba(
            X.iloc[validation_idx]
        )[:, 1]

        scores = []

        for threshold in RF_THRESHOLDS:

            prediction = (
                probabilities >= threshold
            ).astype(int)

            scores.append(
                f1_score(
                    y[validation_idx],
                    prediction,
                    average="macro",
                    zero_division=0
                )
            )

        inner_scores.append(max(scores))

    config_scores.append(
        np.mean(inner_scores)
    )

best_config = rf_configs[
    int(np.argmax(config_scores))
]

print("\nSelected Random Forest configuration:")
print(best_config)

rf_fold_results = []
selected_thresholds = []

for fold, (train_idx, test_idx) in enumerate(
    skf.split(X, y), start=1
):

    print(f"\nStarting Fold {fold}...")

    inner_train_idx, validation_idx = train_test_split(
        train_idx,
        test_size=0.20,
        stratify=y[train_idx],
        random_state=RANDOM_STATE + fold
    )

    validation_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(**TFIDF_PARAMS)
        ),
        (
            "rf",
            RandomForestClassifier(
                **best_config,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    validation_model.fit(
        X.iloc[inner_train_idx],
        y[inner_train_idx]
    )

    validation_probability = validation_model.predict_proba(
        X.iloc[validation_idx]
    )[:, 1]

    best_threshold = 0.50
    best_score = -1

    for threshold in RF_THRESHOLDS:

        validation_prediction = (
            validation_probability >= threshold
        ).astype(int)

        score = f1_score(
            y[validation_idx],
            validation_prediction,
            average="macro",
            zero_division=0
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    selected_thresholds.append(best_threshold)

    # Refit on the entire outer training fold.
    final_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(**TFIDF_PARAMS)
        ),
        (
            "rf",
            RandomForestClassifier(
                **best_config,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    final_model.fit(
        X.iloc[train_idx],
        y[train_idx]
    )

    test_probability = final_model.predict_proba(
        X.iloc[test_idx]
    )[:, 1]

    prediction = (
        test_probability >= best_threshold
    ).astype(int)

    fold_metrics = calculate_metrics(
        y[test_idx],
        prediction
    )

    rf_fold_results.append(
        fold_metrics
    )

    print(
        f"Fold {fold}: "
        f"Accuracy={fold_metrics['Accuracy']:.4f}, "
        f"Precision={fold_metrics['Precision']:.4f}, "
        f"Specificity={fold_metrics['Specificity']:.4f}, "
        f"Recall={fold_metrics['Recall']:.4f}, "
        f"Macro-F1={fold_metrics['Macro-F1']:.4f}"
    )

rf_mean = pd.DataFrame(
    rf_fold_results
).mean()

results["RF"] = rf_mean

print("\n" + "-" * 70)
print("RF MEAN RESULTS")
print("-" * 70)

for metric in [
    "Accuracy",
    "Precision",
    "Specificity",
    "Recall",
    "Macro-F1"
]:
    print(
        f"{metric:<12}: "
        f"{rf_mean[metric]:.4f}"
    )

print(
    "\nSelected thresholds:",
    selected_thresholds
)

print(
    "Mean threshold:",
    round(np.mean(selected_thresholds), 2)
)


# ============================================================
# FINAL COMPARISON
# ============================================================

summary = pd.DataFrame(results).T

summary = summary[
    [
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Macro-F1"
    ]
]

print("\n\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

best_model = summary["Macro-F1"].idxmax()

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print("Model:", best_model)

for metric in [
    "Macro-F1",
    "Accuracy",
    "Precision",
    "Specificity",
    "Recall"
]:
    print(
        f"{metric}: "
        f"{summary.loc[best_model, metric]:.4f}"
    )

summary.to_csv(
    "algorithm_results.csv"
)

print("\nResults saved to:")
print("algorithm_results.csv")

print("\nExperiment completed.")
