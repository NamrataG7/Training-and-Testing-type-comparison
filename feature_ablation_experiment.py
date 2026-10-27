import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATASET = "Updated_dataset.csv"
OUTPUT_DIR = "results/feature_ablation"
RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

WORD_TFIDF = {
    "lowercase": True,
    "strip_accents": "unicode",
    "sublinear_tf": True,
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.95,
    "ngram_range": (1, 2)
}

CHAR_TFIDF = {
    "lowercase": True,
    "strip_accents": "unicode",
    "sublinear_tf": True,
    "max_features": 15000,
    "min_df": 2,
    "max_df": 0.95,
    "analyzer": "char",
    "ngram_range": (3, 5)
}

# Experimental placeholder lexicon.
# Replace with a properly documented/cited lexicon before publication.
ABUSE_TERMS = {
    "abuse", "abusive", "idiot", "stupid", "fool", "moron",
    "loser", "trash", "bastard", "jerk", "hate", "kill",
    "threat", "threaten", "shut", "disgusting"
}

NLTK_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk import word_tokenize, pos_tag
    NLTK_AVAILABLE = True
except Exception:
    print("WARNING: NLTK is not available. Sentiment/POS experiments require NLTK.")


class SocialMediaFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        use_lexical=False,
        use_punct_emoji=False,
        use_lexicon=False,
        use_sentiment=False,
        use_pos=False
    ):
        self.use_lexical = use_lexical
        self.use_punct_emoji = use_punct_emoji
        self.use_lexicon = use_lexicon
        self.use_sentiment = use_sentiment
        self.use_pos = use_pos
        self.sia_ = None

    def fit(self, X, y=None):
        if self.use_sentiment:
            if not NLTK_AVAILABLE:
                raise RuntimeError("NLTK is required for sentiment analysis.")
            try:
                self.sia_ = SentimentIntensityAnalyzer()
            except LookupError:
                raise RuntimeError(
                    "VADER lexicon is missing. Run: "
                    "python -m nltk.downloader vader_lexicon"
                )
        return self

    @staticmethod
    def basic_features(text):
        text = str(text)
        characters = max(len(text), 1)
        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        word_count = len(words)

        uppercase_count = sum(c.isupper() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        alphabetic_count = sum(c.isalpha() for c in text)

        average_word_length = (
            np.mean([len(w) for w in words]) if words else 0.0
        )

        elongated_words = sum(
            bool(re.search(r"(.)\1{2,}", word.lower()))
            for word in words
        )

        repeated_punctuation = len(re.findall(r"([!?.,])\1+", text))

        emoji_count = sum(
            (0x1F300 <= ord(c) <= 0x1FAFF)
            or (0x2600 <= ord(c) <= 0x27BF)
            for c in text
        )

        abuse_hits = sum(
            1 for word in words if word.lower() in ABUSE_TERMS
        )

        punctuation_count = sum(
            1 for c in text if c in "!?.,;:'\"-()[]{}"
        )

        return {
            "word_count": word_count,
            "character_count": characters,
            "average_word_length": average_word_length,
            "uppercase_ratio": uppercase_count / characters,
            "digit_ratio": digit_count / characters,
            "alphabetic_ratio": alphabetic_count / characters,
            "elongated_words": elongated_words,
            "repeated_punctuation": repeated_punctuation,
            "emoji_count": emoji_count,
            "abuse_hits": abuse_hits,
            "punctuation_count": punctuation_count
        }

    def transform(self, X):
        rows = []

        for text in X:
            text = str(text)
            base = self.basic_features(text)
            features = []

            if self.use_lexical:
                features.extend([
                    base["word_count"],
                    base["character_count"],
                    base["average_word_length"],
                    base["uppercase_ratio"],
                    base["digit_ratio"],
                    base["alphabetic_ratio"]
                ])

            if self.use_punct_emoji:
                features.extend([
                    base["elongated_words"],
                    base["repeated_punctuation"],
                    base["emoji_count"],
                    base["punctuation_count"]
                ])

            if self.use_lexicon:
                features.append(base["abuse_hits"])

            if self.use_sentiment:
                sentiment = self.sia_.polarity_scores(text)
                features.extend([
                    sentiment["neg"],
                    sentiment["neu"],
                    sentiment["pos"],
                    sentiment["compound"]
                ])

            if self.use_pos:
                try:
                    tokens = word_tokenize(text)
                    tagged = pos_tag(tokens)
                except LookupError:
                    raise RuntimeError(
                        "Required NLTK tokenizer/POS data is missing. Run:\n"
                        "python -m nltk.downloader punkt punkt_tab "
                        "averaged_perceptron_tagger averaged_perceptron_tagger_eng"
                    )

                total = max(len(tagged), 1)
                counts = {
                    "noun": 0, "verb": 0, "adjective": 0,
                    "adverb": 0, "pronoun": 0,
                    "preposition": 0, "determiner": 0
                }

                for _, tag in tagged:
                    if tag.startswith("NN"):
                        counts["noun"] += 1
                    elif tag.startswith("VB"):
                        counts["verb"] += 1
                    elif tag.startswith("JJ"):
                        counts["adjective"] += 1
                    elif tag.startswith("RB"):
                        counts["adverb"] += 1
                    elif tag.startswith("PRP"):
                        counts["pronoun"] += 1
                    elif tag.startswith("IN"):
                        counts["preposition"] += 1
                    elif tag.startswith("DT"):
                        counts["determiner"] += 1

                features.extend([
                    counts["noun"] / total,
                    counts["verb"] / total,
                    counts["adjective"] / total,
                    counts["adverb"] / total,
                    counts["pronoun"] / total,
                    counts["preposition"] / total,
                    counts["determiner"] / total
                ])

            rows.append(features)

        if not rows or not rows[0]:
            return csr_matrix((len(X), 0))

        return csr_matrix(np.asarray(rows, dtype=float))


class CombinedFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        use_word=True,
        use_char=False,
        use_lexical=False,
        use_punct_emoji=False,
        use_lexicon=False,
        use_sentiment=False,
        use_pos=False
    ):
        self.use_word = use_word
        self.use_char = use_char
        self.use_lexical = use_lexical
        self.use_punct_emoji = use_punct_emoji
        self.use_lexicon = use_lexicon
        self.use_sentiment = use_sentiment
        self.use_pos = use_pos

    def fit(self, X, y=None):
        if self.use_word:
            self.word_vectorizer_ = TfidfVectorizer(**WORD_TFIDF)
            self.word_vectorizer_.fit(X)
        else:
            self.word_vectorizer_ = None

        if self.use_char:
            self.char_vectorizer_ = TfidfVectorizer(**CHAR_TFIDF)
            self.char_vectorizer_.fit(X)
        else:
            self.char_vectorizer_ = None

        self.numeric_features_ = SocialMediaFeatures(
            use_lexical=self.use_lexical,
            use_punct_emoji=self.use_punct_emoji,
            use_lexicon=self.use_lexicon,
            use_sentiment=self.use_sentiment,
            use_pos=self.use_pos
        )
        self.numeric_features_.fit(X, y)

        return self

    def transform(self, X):
        blocks = []

        if self.use_word:
            blocks.append(self.word_vectorizer_.transform(X))

        if self.use_char:
            blocks.append(self.char_vectorizer_.transform(X))

        numeric_requested = any([
            self.use_lexical,
            self.use_punct_emoji,
            self.use_lexicon,
            self.use_sentiment,
            self.use_pos
        ])

        if numeric_requested:
            blocks.append(self.numeric_features_.transform(X))

        return hstack(blocks, format="csr")


def create_model(name):

    if name == "SVM":
        return SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced"
        )

    if name == "LR":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
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

    if name == "NB":
        return MultinomialNB(alpha=1.0)

    if name == "GB":
        return Pipeline([
            (
                "svd",
                TruncatedSVD(
                    n_components=200,
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

    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Specificity": specificity,
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "Macro-F1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
    }


FEATURE_EXPERIMENTS = {
    "Baseline_Word_TFIDF": {
        "use_word": True,
        "use_char": False
    },

    "E1_Character_TFIDF": {
        "use_word": False,
        "use_char": True
    },

    "E2_Hybrid_Word_Character": {
        "use_word": True,
        "use_char": True
    },

    "E3_Hybrid_Lexical": {
        "use_word": True,
        "use_char": True,
        "use_lexical": True
    },

    "E4_Hybrid_Lexical_Punctuation_Emoji": {
        "use_word": True,
        "use_char": True,
        "use_lexical": True,
        "use_punct_emoji": True
    },

    "E5_Hybrid_Lexical_Punctuation_Emoji_Lexicon": {
        "use_word": True,
        "use_char": True,
        "use_lexical": True,
        "use_punct_emoji": True,
        "use_lexicon": True
    },

    "E6_Add_Sentiment": {
        "use_word": True,
        "use_char": True,
        "use_lexical": True,
        "use_punct_emoji": True,
        "use_lexicon": True,
        "use_sentiment": True
    },

    "E7_Full_Feature_Configuration": {
        "use_word": True,
        "use_char": True,
        "use_lexical": True,
        "use_punct_emoji": True,
        "use_lexicon": True,
        "use_sentiment": True,
        "use_pos": True
    }
}


def run_rf_ablation(df):

    X = df["text"].fillna("").astype(str)
    y = pd.to_numeric(df["label"]).astype(int).values

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for experiment_name, config in FEATURE_EXPERIMENTS.items():

        print("\n" + "=" * 90)
        print(f"FEATURE EXPERIMENT: {experiment_name}")
        print("=" * 90)

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y), start=1
        ):

            print(f"\nFold {fold}/{N_SPLITS}")

            features = CombinedFeatures(**config)
            model = create_model("RF")

            feature_start = time.perf_counter()

            features.fit(X.iloc[train_idx], y[train_idx])

            X_train = features.transform(X.iloc[train_idx])
            X_test = features.transform(X.iloc[test_idx])

            feature_time = time.perf_counter() - feature_start

            train_start = time.perf_counter()

            model.fit(X_train, y[train_idx])

            train_time = time.perf_counter() - train_start

            prediction_start = time.perf_counter()

            predictions = model.predict(X_test)

            prediction_time = (
                time.perf_counter() - prediction_start
            )

            metrics = calculate_metrics(
                y[test_idx],
                predictions
            )

            results.append({
                "Experiment": experiment_name,
                "Model": "Random Forest",
                "Fold": fold,
                "Features": X_train.shape[1],
                "Feature_Fit_Transform_Seconds": feature_time,
                "Training_Seconds": train_time,
                "Prediction_Seconds": prediction_time,
                **metrics
            })

            print(
                f"Macro-F1={metrics['Macro-F1']:.4f} | "
                f"Recall={metrics['Recall']:.4f} | "
                f"Prediction={prediction_time:.4f}s"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stage1_rf_feature_ablation_folds.csv"
        ),
        index=False
    )

    summary = (
        results_df
        .groupby("Experiment")[
            [
                "Accuracy",
                "Precision",
                "Specificity",
                "Recall",
                "Macro-F1",
                "Feature_Fit_Transform_Seconds",
                "Training_Seconds",
                "Prediction_Seconds"
            ]
        ]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stage1_rf_feature_ablation_summary.csv"
        )
    )

    return results_df


def run_classifier_comparison(df, best_experiment_name):

    X = df["text"].fillna("").astype(str)
    y = pd.to_numeric(df["label"]).astype(int).values

    config = FEATURE_EXPERIMENTS[best_experiment_name]

    models = [
        "SVM",
        "KNN",
        "LR",
        "NB",
        "GB",
        "RF"
    ]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for model_name in models:

        print("\n" + "=" * 90)
        print(f"MODEL: {model_name}")
        print(f"FEATURE SET: {best_experiment_name}")
        print("=" * 90)

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y), start=1
        ):

            features = CombinedFeatures(**config)
            model = create_model(model_name)

            features.fit(
                X.iloc[train_idx],
                y[train_idx]
            )

            X_train = features.transform(
                X.iloc[train_idx]
            )

            X_test = features.transform(
                X.iloc[test_idx]
            )

            train_start = time.perf_counter()

            model.fit(
                X_train,
                y[train_idx]
            )

            training_time = (
                time.perf_counter() - train_start
            )

            prediction_start = time.perf_counter()

            predictions = model.predict(
                X_test
            )

            prediction_time = (
                time.perf_counter() - prediction_start
            )

            metrics = calculate_metrics(
                y[test_idx],
                predictions
            )

            results.append({
                "Experiment": best_experiment_name,
                "Model": model_name,
                "Fold": fold,
                "Features": X_train.shape[1],
                "Training_Seconds": training_time,
                "Prediction_Seconds": prediction_time,
                **metrics
            })

            print(
                f"Fold {fold}: "
                f"Macro-F1={metrics['Macro-F1']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stage2_classifier_comparison_folds.csv"
        ),
        index=False
    )

    summary = (
        results_df
        .groupby("Model")[
            [
                "Accuracy",
                "Precision",
                "Specificity",
                "Recall",
                "Macro-F1",
                "Training_Seconds",
                "Prediction_Seconds"
            ]
        ]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "stage2_classifier_comparison_summary.csv"
        )
    )

    return results_df


def main():

    print("\n" + "=" * 90)
    print("ABUSIVE LANGUAGE DETECTION")
    print("FEATURE ABLATION + CLASSIFIER EVALUATION")
    print("=" * 90)

    df = pd.read_csv(DATASET)

    required_columns = {"text", "label"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"]).astype(int)

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    print("\n" + "#" * 90)
    print("# STAGE 1: RANDOM FOREST FEATURE ABLATION")
    print("#" * 90)

    stage1_results = run_rf_ablation(df)

    stage1_summary = (
        stage1_results
        .groupby("Experiment")["Macro-F1"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 90)
    print("STAGE 1 RESULTS")
    print("=" * 90)
    print(stage1_summary)

    best_experiment = stage1_summary.index[0]
    best_score = stage1_summary.iloc[0]

    print(
        f"\nBEST FEATURE SET: {best_experiment}"
    )
    print(
        f"RF Macro-F1: {best_score:.4f}"
    )

    print("\n" + "#" * 90)
    print("# STAGE 2: CLASSIFIER COMPARISON")
    print("#" * 90)

    stage2_results = run_classifier_comparison(
        df,
        best_experiment
    )

    final_summary = (
        stage2_results
        .groupby("Model")[
            [
                "Accuracy",
                "Precision",
                "Specificity",
                "Recall",
                "Macro-F1",
                "Training_Seconds",
                "Prediction_Seconds"
            ]
        ]
        .mean()
        .sort_values("Macro-F1", ascending=False)
    )

    print("\n" + "=" * 90)
    print("FINAL CLASSIFIER COMPARISON")
    print("=" * 90)
    print(final_summary)

    final_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "FINAL_RESULTS.csv"
        )
    )

    print("\nExperiments complete.")
    print(f"Results saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
