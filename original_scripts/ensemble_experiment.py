import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

def metrics(y_true, y_pred):
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

def make_model(name):
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)

    if name == "SVM":
        clf = SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced"
        )

    elif name == "KNN":
        clf = KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )

    elif name == "RF":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif name == "LR":
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    elif name == "NB":
        clf = MultinomialNB(alpha=1.0)

    else:
        raise ValueError(name)

    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])

df = pd.read_csv(DATASET)
df["text"] = df["text"].fillna("").astype(str)
df["label"] = pd.to_numeric(df["label"]).astype(int)

print("=" * 70)
print("LOADING DEDUPLICATED DATASET")
print("=" * 70)
print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

X = df["text"]
y = df["label"].values

# The combinations correspond to the ensemble families described
# in the manuscript. All base learners are fitted independently
# inside each training fold and combined by hard majority voting.
ensemble_sets = {
    "A": ["SVM", "KNN", "RF"],
    "B": ["SVM", "RF", "LR"],
    "C": ["SVM", "KNN", "LR"],
    "D": ["KNN", "RF", "LR"],
    "E": ["SVM", "KNN", "RF", "LR"]
}

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = {}

for ensemble_name, members in ensemble_sets.items():

    print("\n" + "=" * 70)
    print(f"ENSEMBLE {ensemble_name}: " + " + ".join(members))
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y), start=1
    ):

        estimators = [
            (
                f"{name.lower()}_{fold}",
                make_model(name)
            )
            for name in members
        ]

        voting = VotingClassifier(
            estimators=estimators,
            voting="hard",
            n_jobs=-1
        )

        voting.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        prediction = voting.predict(
            X.iloc[test_idx]
        )

        fold_metrics = metrics(
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

    mean_result = pd.DataFrame(fold_results).mean()
    results[ensemble_name] = mean_result

    print("\nMean:")
    print(
        f"Accuracy={mean_result['Accuracy']:.4f}, "
        f"Precision={mean_result['Precision']:.4f}, "
        f"Specificity={mean_result['Specificity']:.4f}, "
        f"Recall={mean_result['Recall']:.4f}, "
        f"Macro-F1={mean_result['Macro-F1']:.4f}"
    )

summary = pd.DataFrame(results).T

print("\n" + "=" * 70)
print("FINAL ENSEMBLE COMPARISON")
print("=" * 70)
print(
    summary.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

summary.to_csv(
    "ensemble_results.csv"
)

print("\nResults saved to: ensemble_results.csv")
print("Experiment completed.")
