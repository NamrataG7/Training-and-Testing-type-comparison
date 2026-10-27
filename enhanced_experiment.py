"""
Enhanced experimental framework for abusive-language detection.

Stages
------
0. Reproduce the current E2 Hybrid Word+Character TF-IDF baseline with RF.
1. Preprocessing ablation: none, stopword removal, stemming, lemmatization,
   stopword+stemming, stopword+lemmatization.
2. Feature ablation: eight progressively enriched configurations using the
   best preprocessing selected in Stage 1.
3. Classifier comparison: SVM, KNN, LR, Multinomial NB, Gradient Boosting,
   and Random Forest using the selected preprocessing + feature set.
4. Optional annotation agreement: if the CSV contains annotator columns,
   calculate pairwise Cohen's kappa and Krippendorff's alpha.

Important methodological safeguards
------------------------------------
- All text transformations are fitted/applied within each training fold.
- No train/test leakage is introduced by TF-IDF fitting.
- Numerical handcrafted features can be scaled without scaling TF-IDF.
- No data augmentation, one-hot encoding, RFE, or pretrained embeddings are
  included in the primary experiment. They are intentionally excluded rather
  than added without a research question.
- The abusive-term list is explicitly marked as exploratory. Replace it with
  a documented/cited lexicon before publication if E5-E7 are retained.
"""

import argparse
import os
import re
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as ENGLISH_STOPWORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATASET = "Updated_dataset.csv"
OUTPUT_DIR = "results/enhanced"
RANDOM_STATE = 42
N_SPLITS = 5

# If True, the script attempts optional annotation-agreement analysis when
# suitable annotator columns are present. It does not invent labels when they
# are absent.
RUN_ANNOTATION_AGREEMENT = True

# New diagnostics are additive and do not alter the existing experiment.
RUN_NEW_DIAGNOSTICS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

WORD_TFIDF = {
    "lowercase": True,
    "strip_accents": "unicode",
    "sublinear_tf": True,
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.95,
    "ngram_range": (1, 2),
}

CHAR_TFIDF = {
    "lowercase": True,
    "strip_accents": "unicode",
    "sublinear_tf": True,
    "max_features": 15000,
    "min_df": 2,
    "max_df": 0.95,
    "analyzer": "char",
    "ngram_range": (3, 5),
}

# Exploratory term list only. Do not describe this as a validated lexicon in
# the paper unless it is replaced by a documented/cited resource.
ABUSE_TERMS = {
    "abuse", "abusive", "idiot", "stupid", "fool", "moron",
    "loser", "trash", "bastard", "jerk", "hate", "kill",
    "threat", "threaten", "shut", "disgusting",
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Deterministic text normalization used inside each CV fold."""

    def __init__(self, method="none"):
        self.method = method

    def fit(self, X, y=None):
        # sklearn-style transformer; no corpus statistics are learned here.
        if self.method not in {
            "none", "stopword", "stemming", "lemmatization",
            "stopword_stemming", "stopword_lemmatization",
        }:
            raise ValueError(f"Unknown preprocessing method: {self.method}")
        return self

    @staticmethod
    def _tokens(text):
        return re.findall(r"\b\w+(?:['’]\w+)?\b", str(text), flags=re.UNICODE)

    def transform(self, X):
        return [self._transform_one(text) for text in X]

    def _transform_one(self, text):
        text = str(text)
        if self.method == "none":
            return text

        tokens = self._tokens(text)
        use_stop = self.method in {"stopword", "stopword_stemming", "stopword_lemmatization"}
        use_stem = self.method in {"stemming", "stopword_stemming"}
        use_lemma = self.method in {"lemmatization", "stopword_lemmatization"}

        if use_stop:
            tokens = [t for t in tokens if t.lower() not in ENGLISH_STOPWORDS]

        if use_stem:
            if not NLTK_AVAILABLE:
                raise RuntimeError("NLTK is required for stemming. Install nltk.")
            tokens = [STEMMER.stem(t) for t in tokens]

        if use_lemma:
            if not NLTK_AVAILABLE:
                raise RuntimeError("NLTK is required for lemmatization. Install nltk and WordNet data.")
            try:
                tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
            except LookupError:
                raise RuntimeError(
                    "NLTK WordNet data is missing. Run: "
                    "python -m nltk.downloader wordnet omw-1.4"
                )

        return " ".join(tokens)


# Optional NLTK components are imported once. Stopwords use scikit-learn's
# built-in list so that the stopword experiment does not require a download.
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = set()

NLTK_AVAILABLE = False
STEMMER = None
LEMMATIZER = None

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    pass


class SocialMediaFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        use_lexical=False,
        use_punct_emoji=False,
        use_lexicon=False,
        use_sentiment=False,
        use_pos=False,
        scale_numeric=False,
    ):
        self.use_lexical = use_lexical
        self.use_punct_emoji = use_punct_emoji
        self.use_lexicon = use_lexicon
        self.use_sentiment = use_sentiment
        self.use_pos = use_pos
        self.scale_numeric = scale_numeric
        self.sia_ = None
        self.scaler_ = None

    def fit(self, X, y=None):
        if self.use_sentiment:
            if not NLTK_AVAILABLE:
                raise RuntimeError("NLTK is required for sentiment analysis.")
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                self.sia_ = SentimentIntensityAnalyzer()
            except LookupError:
                raise RuntimeError(
                    "VADER lexicon is missing. Run: "
                    "python -m nltk.downloader vader_lexicon"
                )

        if self.scale_numeric and self._numeric_requested():
            raw = self._raw_numeric_features(X)
            self.scaler_ = StandardScaler()
            self.scaler_.fit(raw)
        return self

    def _numeric_requested(self):
        return any([
            self.use_lexical,
            self.use_punct_emoji,
            self.use_lexicon,
            self.use_sentiment,
            self.use_pos,
        ])

    @staticmethod
    def basic_features(text):
        text = str(text)
        characters = max(len(text), 1)
        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        word_count = len(words)

        uppercase_count = sum(c.isupper() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        alphabetic_count = sum(c.isalpha() for c in text)

        average_word_length = np.mean([len(w) for w in words]) if words else 0.0

        elongated_words = sum(
            bool(re.search(r"(.)\1{2,}", word.lower())) for word in words
        )
        repeated_punctuation = len(re.findall(r"([!?.,])\1+", text))
        emoji_count = sum(
            (0x1F300 <= ord(c) <= 0x1FAFF)
            or (0x2600 <= ord(c) <= 0x27BF)
            for c in text
        )
        abuse_hits = sum(1 for word in words if word.lower() in ABUSE_TERMS)
        punctuation_count = sum(c in "!?.,;:'\"-()[]{}" for c in text)

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
            "punctuation_count": punctuation_count,
        }

    def _raw_numeric_features(self, X):
        rows = []
        for text in X:
            base = self.basic_features(text)
            features = []
            if self.use_lexical:
                features.extend([
                    base["word_count"], base["character_count"],
                    base["average_word_length"], base["uppercase_ratio"],
                    base["digit_ratio"], base["alphabetic_ratio"],
                ])
            if self.use_punct_emoji:
                features.extend([
                    base["elongated_words"], base["repeated_punctuation"],
                    base["emoji_count"], base["punctuation_count"],
                ])
            if self.use_lexicon:
                features.append(base["abuse_hits"])
            if self.use_sentiment:
                sentiment = self.sia_.polarity_scores(str(text))
                features.extend([
                    sentiment["neg"], sentiment["neu"],
                    sentiment["pos"], sentiment["compound"],
                ])
            if self.use_pos:
                try:
                    from nltk import pos_tag, word_tokenize
                    tokens = word_tokenize(str(text))
                    tagged = pos_tag(tokens)
                except LookupError:
                    raise RuntimeError(
                        "Required NLTK tokenizer/POS data is missing. Run: "
                        "python -m nltk.downloader punkt punkt_tab "
                        "averaged_perceptron_tagger averaged_perceptron_tagger_eng"
                    )
                total = max(len(tagged), 1)
                counts = {
                    "noun": 0, "verb": 0, "adjective": 0,
                    "adverb": 0, "pronoun": 0,
                    "preposition": 0, "determiner": 0,
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
                    counts["noun"] / total, counts["verb"] / total,
                    counts["adjective"] / total, counts["adverb"] / total,
                    counts["pronoun"] / total, counts["preposition"] / total,
                    counts["determiner"] / total,
                ])
            rows.append(features)
        return np.asarray(rows, dtype=float)

    def transform(self, X):
        if not self._numeric_requested():
            return csr_matrix((len(X), 0))
        raw = self._raw_numeric_features(X)
        if self.scale_numeric:
            if self.scaler_ is None:
                raise RuntimeError("Scaler was not fitted before transform.")
            raw = self.scaler_.transform(raw)
        return csr_matrix(raw)


class CombinedFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        preprocessing="none",
        use_word=True,
        use_char=False,
        use_lexical=False,
        use_punct_emoji=False,
        use_lexicon=False,
        use_sentiment=False,
        use_pos=False,
        scale_numeric=False,
    ):
        self.preprocessing = preprocessing
        self.use_word = use_word
        self.use_char = use_char
        self.use_lexical = use_lexical
        self.use_punct_emoji = use_punct_emoji
        self.use_lexicon = use_lexicon
        self.use_sentiment = use_sentiment
        self.use_pos = use_pos
        self.scale_numeric = scale_numeric

    def fit(self, X, y=None):
        self.preprocessor_ = TextPreprocessor(self.preprocessing)
        self.preprocessor_.fit(X, y)
        Xp = self.preprocessor_.transform(X)

        if self.use_word:
            self.word_vectorizer_ = TfidfVectorizer(**WORD_TFIDF)
            self.word_vectorizer_.fit(Xp)
        else:
            self.word_vectorizer_ = None

        if self.use_char:
            self.char_vectorizer_ = TfidfVectorizer(**CHAR_TFIDF)
            self.char_vectorizer_.fit(Xp)
        else:
            self.char_vectorizer_ = None

        self.numeric_features_ = SocialMediaFeatures(
            use_lexical=self.use_lexical,
            use_punct_emoji=self.use_punct_emoji,
            use_lexicon=self.use_lexicon,
            use_sentiment=self.use_sentiment,
            use_pos=self.use_pos,
            scale_numeric=self.scale_numeric,
        )
        self.numeric_features_.fit(Xp, y)
        return self

    def transform(self, X):
        Xp = self.preprocessor_.transform(X)
        blocks = []
        if self.use_word:
            blocks.append(self.word_vectorizer_.transform(Xp))
        if self.use_char:
            blocks.append(self.char_vectorizer_.transform(Xp))
        if self.numeric_features_._numeric_requested():
            blocks.append(self.numeric_features_.transform(Xp))
        if not blocks:
            return csr_matrix((len(X), 0))
        return hstack(blocks, format="csr")


FEATURE_EXPERIMENTS = {
    "Baseline_Word_TFIDF": {"use_word": True, "use_char": False},
    "E1_Character_TFIDF": {"use_word": False, "use_char": True},
    "E2_Hybrid_Word_Character": {"use_word": True, "use_char": True},
    "E3_Hybrid_Lexical": {
        "use_word": True, "use_char": True, "use_lexical": True,
    },
    "E4_Hybrid_Lexical_Punctuation_Emoji": {
        "use_word": True, "use_char": True, "use_lexical": True,
        "use_punct_emoji": True,
    },
    "E5_Hybrid_Lexical_Punctuation_Emoji_Lexicon": {
        "use_word": True, "use_char": True, "use_lexical": True,
        "use_punct_emoji": True, "use_lexicon": True,
    },
    "E6_Add_Sentiment": {
        "use_word": True, "use_char": True, "use_lexical": True,
        "use_punct_emoji": True, "use_lexicon": True,
        "use_sentiment": True,
    },
    "E7_Full_Feature_Configuration": {
        "use_word": True, "use_char": True, "use_lexical": True,
        "use_punct_emoji": True, "use_lexicon": True,
        "use_sentiment": True, "use_pos": True,
    },
}

PREPROCESSING_EXPERIMENTS = {
    "P0_None": "none",
    "P1_Stopword_Removal": "stopword",
    "P2_Stemming": "stemming",
    "P3_Lemmatization": "lemmatization",
    "P4_Stopword_Stemming": "stopword_stemming",
    "P5_Stopword_Lemmatization": "stopword_lemmatization",
}


# ------------------------- Models and metrics -------------------------

def create_model(name, balanced=True):
    if name == "SVM":
        return SVC(kernel="linear", C=1.0, class_weight=("balanced" if balanced else None))
    if name == "LR":
        return LogisticRegression(
            C=1.0, penalty="l2", class_weight=("balanced" if balanced else None),
            max_iter=2000, random_state=RANDOM_STATE,
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", class_weight=("balanced" if balanced else None),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    if name == "NB":
        return MultinomialNB(alpha=1.0)
    if name == "GB":
        return Pipeline([
            ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
            ("gb", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=3, random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    report = classification_report(
        y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "NonAbusive_Precision": report["0"]["precision"],
        "NonAbusive_Recall": report["0"]["recall"],
        "NonAbusive_F1": report["0"]["f1-score"],
        "Abusive_Precision": report["1"]["precision"],
        "Abusive_Recall": report["1"]["recall"],
        "Abusive_F1": report["1"]["f1-score"],
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def mean_std_table(df, group_col):
    metrics = ["Accuracy", "Precision", "Specificity", "Recall", "Macro-F1"]
    grouped = df.groupby(group_col)[metrics].agg(["mean", "std"])
    grouped.to_csv(os.path.join(OUTPUT_DIR, f"{group_col}_summary.csv"))
    return grouped


# ------------------------- Stage 0 -------------------------

def run_current_e2_reproduction(X, y, cv):
    rows = []
    print("\n" + "=" * 90)
    print("STAGE 0: CURRENT E2 REPRODUCTION (RF + HYBRID WORD/CHAR TF-IDF)")
    print("=" * 90)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        features = CombinedFeatures(
            preprocessing="none", use_word=True, use_char=True,
        )
        model = create_model("RF")
        features.fit(X.iloc[train_idx], y[train_idx])
        X_train = features.transform(X.iloc[train_idx])
        X_test = features.transform(X.iloc[test_idx])

        start = time.perf_counter()
        model.fit(X_train, y[train_idx])
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_test)
        pred_time = time.perf_counter() - start
        m = calculate_metrics(y[test_idx], pred)
        rows.append({"Fold": fold, **m, "Training_Seconds": train_time, "Prediction_Seconds": pred_time})
        print(f"Fold {fold}: Macro-F1={m['Macro-F1']:.4f}, Recall={m['Recall']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "stage0_current_e2_reproduction_folds.csv"), index=False)
    summary = out[["Accuracy", "Precision", "Specificity", "Recall", "Macro-F1", "Training_Seconds", "Prediction_Seconds"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUTPUT_DIR, "stage0_current_e2_reproduction_summary.csv"))
    return out


# ------------------------- Stage 1 -------------------------

def run_preprocessing_ablation(X, y, cv):
    rows = []
    print("\n" + "=" * 90)
    print("STAGE 1: PREPROCESSING ABLATION (RF + E2 HYBRID)")
    print("=" * 90)

    for exp_name, method in PREPROCESSING_EXPERIMENTS.items():
        print(f"\n{exp_name}")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            features = CombinedFeatures(
                preprocessing=method, use_word=True, use_char=True,
            )
            model = create_model("RF")
            features.fit(X.iloc[train_idx], y[train_idx])
            X_train = features.transform(X.iloc[train_idx])
            X_test = features.transform(X.iloc[test_idx])
            start = time.perf_counter()
            model.fit(X_train, y[train_idx])
            train_time = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(X_test)
            pred_time = time.perf_counter() - start
            m = calculate_metrics(y[test_idx], pred)
            rows.append({
                "Experiment": exp_name, "Preprocessing": method, "Fold": fold,
                "Features": X_train.shape[1],
                "Training_Seconds": train_time, "Prediction_Seconds": pred_time,
                **m,
            })
            print(f"  Fold {fold}: Macro-F1={m['Macro-F1']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "stage1_preprocessing_ablation_folds.csv"), index=False)
    summary = out.groupby(["Experiment", "Preprocessing"])[
        ["Accuracy", "Precision", "Specificity", "Recall", "Macro-F1",
         "Training_Seconds", "Prediction_Seconds"]
    ].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUTPUT_DIR, "stage1_preprocessing_ablation_summary.csv"))

    best_row = summary["Macro-F1"]["mean"].idxmax()
    best_experiment = best_row[0]
    best_method = best_row[1]
    best_score = summary.loc[best_row, ("Macro-F1", "mean")]
    print(f"\nSelected preprocessing: {best_experiment} ({best_method}), mean Macro-F1={best_score:.4f}")
    return out, best_method, best_experiment


# ------------------------- Stage 2 -------------------------

def run_feature_ablation(X, y, cv, preprocessing):
    rows = []
    print("\n" + "=" * 90)
    print(f"STAGE 2: FEATURE ABLATION USING PREPROCESSING={preprocessing}")
    print("=" * 90)

    for exp_name, config in FEATURE_EXPERIMENTS.items():
        print(f"\n{exp_name}")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            features = CombinedFeatures(preprocessing=preprocessing, **config)
            model = create_model("RF")
            start = time.perf_counter()
            features.fit(X.iloc[train_idx], y[train_idx])
            X_train = features.transform(X.iloc[train_idx])
            X_test = features.transform(X.iloc[test_idx])
            feature_time = time.perf_counter() - start
            start = time.perf_counter()
            model.fit(X_train, y[train_idx])
            train_time = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(X_test)
            pred_time = time.perf_counter() - start
            m = calculate_metrics(y[test_idx], pred)
            rows.append({
                "Experiment": exp_name, "Preprocessing": preprocessing,
                "Fold": fold, "Features": X_train.shape[1],
                "Feature_Fit_Transform_Seconds": feature_time,
                "Training_Seconds": train_time, "Prediction_Seconds": pred_time,
                **m,
            })
            print(f"  Fold {fold}: Macro-F1={m['Macro-F1']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "stage2_feature_ablation_folds.csv"), index=False)
    summary = out.groupby("Experiment")[[
        "Accuracy", "Precision", "Specificity", "Recall", "Macro-F1",
        "Feature_Fit_Transform_Seconds", "Training_Seconds", "Prediction_Seconds",
    ]].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUTPUT_DIR, "stage2_feature_ablation_summary.csv"))

    best_experiment = summary["Macro-F1"]["mean"].idxmax()
    best_score = summary.loc[best_experiment, ("Macro-F1", "mean")]
    print(f"\nSelected feature set: {best_experiment}, mean Macro-F1={best_score:.4f}")
    return out, best_experiment


# ------------------------- Stage 3 -------------------------

def run_scaling_ablation(X, y, cv, preprocessing):
    """Compare scaling of handcrafted numeric features for RF and KNN.

    The TF-IDF blocks remain untouched; only the small numerical feature block
    is standardized. E7 is used deliberately because it contains all of the
    handcrafted numeric features. KNN is included because distance-based
    methods are the most sensitive to feature scale.
    """
    feature_config = FEATURE_EXPERIMENTS["E7_Full_Feature_Configuration"]
    rows = []
    print("\n" + "=" * 90)
    print("STAGE 3: NUMERICAL FEATURE SCALING ABLATION (E7)")
    print("=" * 90)

    for model_name in ["RF", "KNN"]:
        for scale in [False, True]:
            label = "S0_No_Scaling" if not scale else "S1_Scaled_Numerical"
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                features = CombinedFeatures(
                    preprocessing=preprocessing,
                    scale_numeric=scale,
                    **feature_config,
                )
                model = create_model(model_name)
                features.fit(X.iloc[train_idx], y[train_idx])
                X_train = features.transform(X.iloc[train_idx])
                X_test = features.transform(X.iloc[test_idx])
                model.fit(X_train, y[train_idx])
                pred = model.predict(X_test)
                m = calculate_metrics(y[test_idx], pred)
                rows.append({
                    "Model": model_name, "Scaling": label, "Fold": fold, **m
                })
                print(
                    f"{model_name} | {label} | Fold {fold}: "
                    f"Macro-F1={m['Macro-F1']:.4f}"
                )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "stage3_scaling_ablation_folds.csv"), index=False)
    out.groupby(["Model", "Scaling"])[
        ["Accuracy", "Precision", "Specificity", "Recall", "Macro-F1"]
    ].agg(["mean", "std"]).to_csv(
        os.path.join(OUTPUT_DIR, "stage3_scaling_ablation_summary.csv")
    )
    return out


def run_classifier_comparison(X, y, cv, preprocessing, feature_config):
    rows = []
    models = ["SVM", "KNN", "LR", "NB", "GB", "RF"]
    print("\n" + "=" * 90)
    print("STAGE 4: CLASSIFIER COMPARISON")
    print(f"Preprocessing={preprocessing}")
    print(f"Feature configuration={feature_config}")
    print("=" * 90)

    for model_name in models:
        print(f"\n{model_name}")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            features = CombinedFeatures(preprocessing=preprocessing, **feature_config)
            model = create_model(model_name)
            feature_start = time.perf_counter()
            features.fit(X.iloc[train_idx], y[train_idx])
            X_train = features.transform(X.iloc[train_idx])
            X_test = features.transform(X.iloc[test_idx])
            feature_time = time.perf_counter() - feature_start
            train_start = time.perf_counter()
            model.fit(X_train, y[train_idx])
            train_time = time.perf_counter() - train_start
            pred_start = time.perf_counter()
            pred = model.predict(X_test)
            pred_time = time.perf_counter() - pred_start
            m = calculate_metrics(y[test_idx], pred)
            rows.append({
                "Model": model_name, "Preprocessing": preprocessing,
                "Fold": fold, "Features": X_train.shape[1],
                "Feature_Fit_Transform_Seconds": feature_time,
                "Training_Seconds": train_time, "Prediction_Seconds": pred_time,
                **m,
            })
            print(
                f"  Fold {fold}: Accuracy={m['Accuracy']:.4f}, "
                f"Precision={m['Precision']:.4f}, Specificity={m['Specificity']:.4f}, "
                f"Recall={m['Recall']:.4f}, Macro-F1={m['Macro-F1']:.4f}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "stage4_classifier_comparison_folds.csv"), index=False)
    summary = out.groupby("Model")[[
        "Accuracy", "Precision", "Specificity", "Recall", "Macro-F1",
        "Feature_Fit_Transform_Seconds", "Training_Seconds", "Prediction_Seconds",
    ]].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUTPUT_DIR, "stage4_classifier_comparison_summary.csv"))

    final = summary[[(m, s) for m in ["Accuracy", "Precision", "Specificity", "Recall", "Macro-F1"] for s in ["mean", "std"]]]
    final.to_csv(os.path.join(OUTPUT_DIR, "FINAL_ENHANCED_RESULTS.csv"))

    best_model = summary["Macro-F1"]["mean"].idxmax()
    print(f"\nBest individual classifier by mean Macro-F1: {best_model} ({summary.loc[best_model, ('Macro-F1', 'mean')]:.4f})")
    return out, best_model



# ------------------------- New additive diagnostics -------------------------

def run_majority_baseline(y, cv):
    """Evaluate the trivial majority-class baseline using the same CV folds."""
    rows = []
    majority = int(pd.Series(y).mode().iloc[0])
    for fold, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        pred = np.full(len(test_idx), majority, dtype=int)
        rows.append({"Fold": fold, **calculate_metrics(y[test_idx], pred)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "majority_baseline_folds.csv"), index=False)
    out.agg(["mean", "std"]).to_csv(os.path.join(OUTPUT_DIR, "majority_baseline_summary.csv"))
    return out


def run_class_weight_ablation(X, y, cv, preprocessing, feature_config):
    """Compare balanced vs unbalanced RF/LR/SVM without changing the main pipeline."""
    rows = []
    for model_name in ["RF", "LR", "SVM"]:
        for balanced in [False, True]:
            label = "balanced" if balanced else "unbalanced"
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                features = CombinedFeatures(preprocessing=preprocessing, **feature_config)
                features.fit(X.iloc[train_idx], y[train_idx])
                X_train = features.transform(X.iloc[train_idx])
                X_test = features.transform(X.iloc[test_idx])
                model = create_model(model_name, balanced=balanced)
                model.fit(X_train, y[train_idx])
                pred = model.predict(X_test)
                rows.append({
                    "Model": model_name, "Class_Weight": label, "Fold": fold,
                    **calculate_metrics(y[test_idx], pred)
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "class_weight_ablation_folds.csv"), index=False)
    out.groupby(["Model", "Class_Weight"])[
        ["Accuracy", "Precision", "Specificity", "Recall", "Macro-F1",
         "NonAbusive_F1", "Abusive_F1"]
    ].agg(["mean", "std"]).to_csv(
        os.path.join(OUTPUT_DIR, "class_weight_ablation_summary.csv")
    )
    return out


def run_final_rf_oof_analysis(X, y, cv, preprocessing, feature_config):
    """Generate aggregate OOF predictions, confusion matrix and error files for RF."""
    records = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        features = CombinedFeatures(preprocessing=preprocessing, **feature_config)
        features.fit(X.iloc[train_idx], y[train_idx])
        X_train = features.transform(X.iloc[train_idx])
        X_test = features.transform(X.iloc[test_idx])
        model = create_model("RF")
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = np.full(len(pred), np.nan)
        for pos, idx in enumerate(test_idx):
            records.append({
                "row_index": int(idx), "fold": fold,
                "text": X.iloc[idx], "true_label": int(y[idx]),
                "predicted_label": int(pred[pos]),
                "abusive_probability": float(prob[pos]),
            })
    oof = pd.DataFrame(records).sort_values("row_index")
    oof.to_csv(os.path.join(OUTPUT_DIR, "rf_oof_predictions.csv"), index=False)

    cm = confusion_matrix(oof["true_label"], oof["predicted_label"], labels=[0, 1])
    pd.DataFrame(cm, index=["Actual_0", "Actual_1"],
                 columns=["Predicted_0", "Predicted_1"]).to_csv(
        os.path.join(OUTPUT_DIR, "rf_oof_confusion_matrix.csv")
    )
    pd.DataFrame([calculate_metrics(oof["true_label"], oof["predicted_label"])]).to_csv(
        os.path.join(OUTPUT_DIR, "rf_oof_metrics.csv"), index=False
    )

    fp = oof[(oof.true_label == 0) & (oof.predicted_label == 1)].copy()
    fn = oof[(oof.true_label == 1) & (oof.predicted_label == 0)].copy()
    fp.to_csv(os.path.join(OUTPUT_DIR, "rf_false_positives.csv"), index=False)
    fn.to_csv(os.path.join(OUTPUT_DIR, "rf_false_negatives.csv"), index=False)

    # Useful for later threshold/error analysis; does not change the reported 0.5 model.
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    oof["probability_bin"] = pd.cut(oof["abusive_probability"], bins=bins,
                                      include_lowest=True, right=True)
    confidence = oof.groupby("probability_bin", observed=False).agg(
        Count=("true_label", "size"),
        Abusive_Rate=("true_label", "mean"),
        Mean_Probability=("abusive_probability", "mean")
    ).reset_index()
    confidence.to_csv(os.path.join(OUTPUT_DIR, "rf_probability_bins.csv"), index=False)
    return oof


def run_oof_model_predictions(X, y, cv, preprocessing, feature_config, model_names):
    """Generate paired out-of-fold predictions for multiple models on identical folds."""
    records = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        features = CombinedFeatures(preprocessing=preprocessing, **feature_config)
        features.fit(X.iloc[train_idx], y[train_idx])
        X_train = features.transform(X.iloc[train_idx])
        X_test = features.transform(X.iloc[test_idx])
        for model_name in model_names:
            model = create_model(model_name)
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_test)
            for pos, idx in enumerate(test_idx):
                records.append({
                    "row_index": int(idx), "fold": fold, "model": model_name,
                    "true_label": int(y[idx]), "predicted_label": int(pred[pos]),
                })
    return pd.DataFrame(records)


def run_paired_permutation_test(oof_predictions, model_a="RF", model_b="LR", n_permutations=5000):
    """Paired permutation test on OOF correctness differences.

    The test is deliberately secondary/exploratory. It uses identical held-out
    examples for both models, so each example contributes a paired correctness
    difference. The test does not alter model selection or the reported CV mean.
    """
    a = oof_predictions[oof_predictions["model"] == model_a].set_index("row_index")
    b = oof_predictions[oof_predictions["model"] == model_b].set_index("row_index")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("No paired OOF predictions available for the requested models.")

    ca = (a.loc[common, "predicted_label"].values == a.loc[common, "true_label"].values).astype(int)
    cb = (b.loc[common, "predicted_label"].values == b.loc[common, "true_label"].values).astype(int)
    observed = float(np.mean(ca - cb))

    rng = np.random.default_rng(RANDOM_STATE)
    diffs = ca - cb
    exceed = 0
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=len(diffs), dtype=np.int8)
        permuted = np.where(swap == 1, -diffs, diffs)
        if abs(float(np.mean(permuted))) >= abs(observed):
            exceed += 1
    p_value = (exceed + 1) / (n_permutations + 1)

    # Also provide a paired bootstrap CI for the primary metric (macro-F1).
    y_true = a.loc[common, "true_label"].values
    pa = a.loc[common, "predicted_label"].values
    pb = b.loc[common, "predicted_label"].values
    rng = np.random.default_rng(RANDOM_STATE)
    boot_diffs = np.empty(n_permutations, dtype=float)
    n = len(common)
    for i in range(n_permutations):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = (
            f1_score(y_true[idx], pa[idx], average="macro", zero_division=0)
            - f1_score(y_true[idx], pb[idx], average="macro", zero_division=0)
        )
    observed_macro = (
        f1_score(y_true, pa, average="macro", zero_division=0)
        - f1_score(y_true, pb, average="macro", zero_division=0)
    )
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    result = pd.DataFrame([{
        "Model_A": model_a,
        "Model_B": model_b,
        "Paired_OOF_Observations": len(common),
        "Observed_Accuracy_Difference": observed,
        "Observed_MacroF1_Difference": observed_macro,
        "MacroF1_Difference_Bootstrap_CI_Low": ci_low,
        "MacroF1_Difference_Bootstrap_CI_High": ci_high,
        "Bootstrap_Iterations": n_permutations,
        "Permutation_Iterations": n_permutations,
        "Two_Sided_Accuracy_Permutation_P": p_value,
    }])
    result.to_csv(os.path.join(OUTPUT_DIR, "rf_vs_lr_paired_permutation_test.csv"), index=False)
    return result


def run_repeated_rf_stability(X, y, preprocessing, feature_config, repeats=3):
    """Optional repeated 5-fold RF stability check; separate from the main fixed-seed results."""
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=repeats, random_state=RANDOM_STATE
    )
    rows = []
    for split_no, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        features = CombinedFeatures(preprocessing=preprocessing, **feature_config)
        features.fit(X.iloc[train_idx], y[train_idx])
        X_train = features.transform(X.iloc[train_idx])
        X_test = features.transform(X.iloc[test_idx])
        model = create_model("RF")
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        rows.append({"Split": split_no, **calculate_metrics(y[test_idx], pred)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "rf_repeated5fold_folds.csv"), index=False)
    out.agg(["mean", "std"]).to_csv(os.path.join(OUTPUT_DIR, "rf_repeated5fold_summary.csv"))
    return out


# ------------------------- Optional annotation agreement -------------------------

def krippendorff_alpha_nominal(data):
    """Minimal nominal Krippendorff alpha implementation for complete labels.

    data: list of annotator rows, each row a list of labels. Missing values can
    be represented by None/NaN. This is used only when annotator columns exist.
    """
    matrix = np.asarray(data, dtype=object)
    values = []
    for row in matrix:
        values.extend([v for v in row if not pd.isna(v)])
    if len(values) == 0:
        return np.nan

    categories = list(pd.unique(values))
    cat_to_i = {c: i for i, c in enumerate(categories)}
    n = len(categories)

    observed_disagreement = 0.0
    pair_count = 0
    coincidence = np.zeros((n, n), dtype=float)

    for item in matrix.T:
        vals = [v for v in item if not pd.isna(v)]
        m = len(vals)
        if m < 2:
            continue
        for a, b in combinations(vals, 2):
            ia, ib = cat_to_i[a], cat_to_i[b]
            coincidence[ia, ib] += 1
            coincidence[ib, ia] += 1
            pair_count += 1
            observed_disagreement += 0 if ia == ib else 1

    if pair_count == 0:
        return np.nan
    Do = observed_disagreement / pair_count

    total = coincidence.sum()
    marginals = coincidence.sum(axis=1)
    expected_agreement = ((marginals * marginals).sum() - np.trace(coincidence)) / max(total * (total - 1), 1)
    De = 1 - expected_agreement
    if De <= 0:
        return 1.0 if Do == 0 else np.nan
    return 1 - (Do / De)


def run_annotation_agreement(df):
    """Calculate pairwise Cohen's kappa and Krippendorff alpha if possible."""
    candidate_groups = [
        ["annotator_1", "annotator_2"],
        ["annotator1", "annotator2"],
        ["label_annotator_1", "label_annotator_2"],
    ]
    found = None
    for cols in candidate_groups:
        if all(c in df.columns for c in cols):
            found = cols
            break

    if found is None:
        print("\nAnnotation agreement: skipped. No independent annotator-label columns were found in the dataset.")
        return None

    from sklearn.metrics import cohen_kappa_score
    a, b = found
    kappa = cohen_kappa_score(df[a], df[b])
    alpha = krippendorff_alpha_nominal([df[a].tolist(), df[b].tolist()])
    result = pd.DataFrame([{
        "Annotator_A": a, "Annotator_B": b,
        "Raw_Agreement": np.mean(df[a].values == df[b].values),
        "Cohen_Kappa": kappa,
        "Krippendorff_Alpha_Nominal": alpha,
    }])
    result.to_csv(os.path.join(OUTPUT_DIR, "annotation_agreement.csv"), index=False)
    print("\nAnnotation agreement:")
    print(result.to_string(index=False))
    return result


# ------------------------- Dataset / metadata audit -------------------------

def run_dataset_audit(df):
    """Document duplicates and availability of metadata without changing the dataset."""
    text = df["text"].fillna("").astype(str)
    duplicate_rows = int(text.duplicated(keep=False).sum())
    exact_duplicate_excess = int(text.duplicated(keep="first").sum())
    metadata_candidates = {
        "author": [c for c in df.columns if c.lower() in {"author", "author_id", "user_id", "username"}],
        "thread": [c for c in df.columns if c.lower() in {"thread", "thread_id", "conversation_id"}],
        "time": [c for c in df.columns if c.lower() in {"timestamp", "time", "datetime", "created_at", "date"}],
        "language": [c for c in df.columns if c.lower() in {"language", "lang"}],
    }
    rows = [{
        "Rows": len(df),
        "Unique_Texts": int(text.nunique()),
        "Rows_In_Duplicate_Clusters": duplicate_rows,
        "Exact_Duplicate_Excess": exact_duplicate_excess,
        "Author_Metadata_Available": bool(metadata_candidates["author"]),
        "Thread_Metadata_Available": bool(metadata_candidates["thread"]),
        "Time_Metadata_Available": bool(metadata_candidates["time"]),
        "Language_Metadata_Available": bool(metadata_candidates["language"]),
        "Author_Columns": ";".join(metadata_candidates["author"]),
        "Thread_Columns": ";".join(metadata_candidates["thread"]),
        "Time_Columns": ";".join(metadata_candidates["time"]),
        "Language_Columns": ";".join(metadata_candidates["language"]),
    }]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "dataset_metadata_audit.csv"), index=False)
    return out


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced abusive-language ML experiment")
    parser.add_argument("--run-stage0", action="store_true", help="Re-run the existing E2 RF baseline; omitted by default because the current result is already available.")
    parser.add_argument("--skip-annotation", action="store_true")
    parser.add_argument("--skip-new-diagnostics", action="store_true", help="Skip additive diagnostics; existing stages remain unchanged.")
    parser.add_argument("--run-repeated-stability", action="store_true", help="Run optional repeated 5-fold RF stability check (3 repeats).")
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"]).astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")

    X = df["text"]
    y = df["label"].values
    run_dataset_audit(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 90)
    print("ENHANCED ABUSIVE-LANGUAGE EXPERIMENT")
    print("=" * 90)
    print("Dataset shape:", df.shape)
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
    print("Random state:", RANDOM_STATE)
    print("CV folds:", N_SPLITS)

    if RUN_ANNOTATION_AGREEMENT and not args.skip_annotation:
        run_annotation_agreement(df)

    if args.run_stage0:
        run_current_e2_reproduction(X, y, cv)

    stage1, best_preprocessing, best_preprocessing_name = run_preprocessing_ablation(X, y, cv)
    stage2, best_feature_experiment = run_feature_ablation(
        X, y, cv, best_preprocessing
    )

    best_feature_config = FEATURE_EXPERIMENTS[best_feature_experiment].copy()

    # Scaling is a separate controlled ablation only when handcrafted numeric
    # features are present in the selected configuration.
    scaling_result = run_scaling_ablation(
        X, y, cv, best_preprocessing
    )

    # Do not silently change the selected feature set based on scaling. Scaling
    # is reported as a separate sensitivity experiment. The main classifier
    # comparison uses the selected feature set without scaling unless the user
    # later chooses the scaled variant based on the results.
    classifier_result, best_model = run_classifier_comparison(
        X, y, cv, best_preprocessing, best_feature_config
    )

    # Additive diagnostics: these do not modify the existing selected model or reported stages.
    if RUN_NEW_DIAGNOSTICS and not args.skip_new_diagnostics:
        print("\nRunning additive diagnostics (does not alter the main experiment)...")
        run_majority_baseline(y, cv)
        run_class_weight_ablation(
            X, y, cv, best_preprocessing, best_feature_config
        )
        run_final_rf_oof_analysis(
            X, y, cv, best_preprocessing, best_feature_config
        )
        paired = run_oof_model_predictions(
            X, y, cv, best_preprocessing, best_feature_config, ["RF", "LR"]
        )
        paired.to_csv(os.path.join(OUTPUT_DIR, "rf_lr_paired_oof_predictions.csv"), index=False)
        run_paired_permutation_test(paired, "RF", "LR", n_permutations=5000)
        if args.run_repeated_stability:
            run_repeated_rf_stability(
                X, y, best_preprocessing, best_feature_config, repeats=3
            )

    manifest = pd.DataFrame([{
        "Dataset": args.dataset,
        "Rows": len(df),
        "Non_Abusive": int((y == 0).sum()),
        "Abusive": int((y == 1).sum()),
        "Abusive_Percent": float((y == 1).mean() * 100),
        "CV": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "Selected_Preprocessing": best_preprocessing,
        "Selected_Preprocessing_Experiment": best_preprocessing_name,
        "Selected_Feature_Experiment": best_feature_experiment,
        "Selected_Model_By_MacroF1": best_model,
        "SVM": "linear kernel, C=1.0, class_weight=balanced",
        "KNN": "n_neighbors=5, n_jobs=-1",
        "RF": "200 trees, max_depth=None, min_samples_leaf=1, max_features=sqrt, class_weight=balanced, random_state=42",
        "LR": "L2, C=1.0, class_weight=balanced, max_iter=2000, random_state=42",
        "NB": "MultinomialNB(alpha=1.0)",
        "GB": "100 estimators, learning_rate=0.1, max_depth=3, SVD=200, random_state=42",
        "Additional_Diagnostics": "majority baseline, class-weight ablation, RF OOF confusion/error analysis, probability bins; optional repeated 5-fold RF stability",
    }])
    manifest.to_csv(os.path.join(OUTPUT_DIR, "experiment_manifest.csv"), index=False)

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETED")
    print("=" * 90)
    print("Selected preprocessing:", best_preprocessing)
    print("Selected feature set:", best_feature_experiment)
    print("Best classifier by Macro-F1:", best_model)
    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
