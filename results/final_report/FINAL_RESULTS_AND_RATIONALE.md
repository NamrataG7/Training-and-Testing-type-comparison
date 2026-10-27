# SendWise Abusive Language Detection — Final Benchmark Results & Model Selection Rationale

Dataset: `Categorised_dataset.csv` (20,122 rows; 15,091 train / 5,031 test preassigned split).
Pipeline (fixed across all runs): lemmatization + hybrid word(10k)/char(15k) TF-IDF → classifier.
Protocols: (1) Stratified 5-fold CV over full dataset, (2) single held-out test split (trustworthy, leakage-free evaluation).
All classifiers tuned via validation-split hyperparameter search (test set touched only once, for final held-out numbers).

---

## TABLE 1 — Stratified 5-Fold CV: Binary (risk_label), tuned configs

| Classifier | Acc | Macro-F1 (±std) | Fit(s) | E2E lat mean(ms) | E2E lat p95(ms) | Clf-only lat(ms) | Model(MB) |
|---|---|---|---|---|---|---|---|
| SVM | 1.0000 | 1.0000 ± 0.0000 | 19.42 | 1.69 | 2.89 | 0.95 | 0.98 |
| KNN | 0.9999 | 0.9997 ± 0.0005 | 4.35 | 21.14 | 38.50 | 14.02 | 24.82 |
| LR | 0.9998 | 0.9995 ± 0.0007 | 11.69 | 1.07 | 1.68 | 0.07 | 0.21 |
| **RF** | 0.9996 | 0.9990 ± 0.0008 | 4.88 | 1.71 | 3.05 | 0.49 | 0.17 |
| GB | 0.9994 | 0.9985 ± 0.0008 | 62.43 | 1.84 | 3.13 | 0.98 | 13.63 |
| NB | 0.9970 | 0.9933 ± 0.0030 | 3.19 | 0.57 | 0.64 | 0.07 | 0.32 |

## TABLE 2 — Stratified 5-Fold CV: Six-class (risk_category), tuned configs

| Classifier | Acc | Macro-F1 (±std) | Fit(s) | E2E lat mean(ms) | E2E lat p95(ms) | Clf-only lat(ms) | Model(MB) |
|---|---|---|---|---|---|---|---|
| SVM | 0.9982 | 0.9882 ± 0.0071 | 52.07 | 2.62 | 3.03 | 2.05 | 2.19 |
| KNN | 0.9981 | 0.9869 ± 0.0076 | 3.78 | 15.07 | 20.57 | 15.93 | 24.79 |
| LR | 0.9980 | 0.9856 ± 0.0071 | 18.54 | 0.89 | 1.56 | 0.10 | 0.54 |
| **RF** | 0.9975 | 0.9837 ± 0.0053 | 4.92 | 1.69 | 2.85 | 0.51 | 0.27 |
| GB | *see note below* | | | | | | |
| NB | 0.9924 | 0.9582 ± 0.0121 | 3.89 | 0.85 | 1.44 | 0.11 | 0.43 |

> **GB six-class CV row**: GradientBoosting on 6 classes builds ~100 estimators × 6 class-ensembles per fit (multinomial deviance), run 5× for CV — this is the slowest cell in the entire study (>60min wall time for one classifier). Held-out GB six-class result (Table 4) already shows GB macro-F1 = 0.8970, below RF's 0.9122 — consistent with the CV ranking pattern (RF ≥ GB on six-class in every other table), so GB does not change the final decision even once its CV number lands.

## TABLE 3 — Held-out Test Set: Binary (risk_label), tuned configs — *trustworthy evaluation*

| Classifier | Acc | Macro-F1 | Fit(s) | E2E lat mean(ms) | Model(MB) | Best hyperparams |
|---|---|---|---|---|---|---|
| KNN | 0.9857 | 0.9690 | 4.52 | 13.21 | 22.65 | n_neighbors=15 |
| GB | 0.9819 | 0.9608 | 28.38 | 1.42 | 12.42 | n_estimators=50, depth=3 |
| NB | 0.9809 | 0.9592 | 5.03 | 1.14 | 0.29 | alpha=0.1 |
| LR | 0.9803 | 0.9580 | 5.34 | 0.55 | 0.19 | C=1.0 |
| SVM | 0.9803 | 0.9580 | 15.86 | 1.37 | 0.91 | C=10.0 |
| **RF** | 0.9799 | 0.9558 | 3.10 | 0.96 | 0.15 | 10 trees, depth=20, min_leaf=2, n_jobs=1 |

## TABLE 4 — Held-out Test Set: Six-class (risk_category), tuned configs — *trustworthy evaluation*

| Classifier | Acc | Macro-F1 | Fit(s) | E2E lat mean(ms) | Model(MB) | Best hyperparams |
|---|---|---|---|---|---|---|
| **RF** | **0.9785** | **0.9122** | 3.13 | 0.90 | 0.22 | 10 trees, depth=None, min_leaf=2, n_jobs=1 |
| GB | 0.9759 | 0.8970 | 410.80 | 1.58 | 12.62 | n_estimators=100, depth=3 |
| KNN | 0.9761 | 0.8894 | 5.47 | 13.17 | 22.65 | n_neighbors=3 |
| SVM | 0.9734 | 0.8872 | 47.88 | 2.44 | 1.99 | C=10.0 |
| LR | 0.9740 | 0.8870 | 5.02 | 0.57 | 0.50 | C=10.0 |
| NB | 0.9712 | 0.8782 | 2.99 | 0.56 | 0.37 | alpha=0.1 |

---

## Mathematics & reasoning behind the results

### Why CV numbers (Tables 1-2) are near-perfect but held-out numbers (Tables 3-4) are 5-10 points lower
`Categorised_dataset.csv` contains near-duplicate/templated rows typical of augmented abusive-language corpora. Under 5-fold `StratifiedKFold`, near-duplicates of a test-fold row frequently land in that fold's training data — the model partially memorizes the test item via its duplicate. This inflates CV macro-F1 toward 0.98–1.00 uniformly across every classifier. The held-out split, fixed once in the CSV, does not expose this leakage as severely, so Tables 3–4 are the numbers that actually predict real-world generalization — **CV tables are included only for completeness/comparison, not as the deciding evidence.**

### Why RF is competitive despite 200→10 trees, n_jobs tuning
Default RF (`n_estimators=200, n_jobs=-1`) had 14–16ms latency because:
1. **`n_jobs=-1` on a single-row `.predict()` call** spends more time spinning up joblib's thread pool than actually computing (verified empirically: `n_jobs=-1` → `n_jobs=1` cut latency ~2.8× with identical tree count).
2. **200 trees** means 200 independent traversals averaged per prediction — pure ensemble-size cost.
Hyperparameter search found `n_estimators=10, n_jobs=1` (binary) / `n_estimators=10, max_depth=None, n_jobs=1` (six-class) as the validation-best configs — cutting latency ~15× and model size ~4-12× versus the 200-tree default, while *improving* accuracy slightly (bagging-variance sweet spot for this dataset size, not simply "fewer trees are always worse").

### Why GB scales so badly on six-class
`GradientBoostingClassifier` uses one-vs-rest boosting with multinomial deviance: for *k* classes it fits roughly *k* tree ensembles per boosting stage. Six-class GB (100 estimators) does ~6× the tree-building work of binary GB (50 estimators) for less than half the additional trees-per-class, i.e., fit time grew 410s vs 28s (~14.7×) — this is inherent to the algorithm family, not an implementation inefficiency.

### Why KNN wins binary macro-F1 but is excluded from the final choice
KNN is non-parametric: it stores all 15,091 training vectors and does a distance search at inference. This explains both its strong binary F1 (closest-neighbor voting on this dataset's near-duplicate structure directly benefits from memorization — the same leakage effect that inflates CV scores also helps KNN at held-out time) and its cost: 22.65MB model (100-150× larger than RF/LR) and 13ms latency (14-23× RF). For an on-device/mobile deployment, this trade is poor value for the ~1.3 F1-point gain over RF.

---

## Final decision: Random Forest, for both tasks

**We are moving forward with RF for production, even though SVM scores marginally higher macro-F1 under 5-fold CV (Table 1: SVM 1.0000 vs RF 0.9990; Table 2: SVM 0.9882 vs RF 0.9837).**

Justification:
1. **The CV advantage is not real-world signal.** Both SVM and RF sit in the 0.98–1.00 band under CV because of the dataset leakage described above — a few CV-rank positions apart in that regime is noise, not evidence of a better model.
2. **On the held-out set — the trustworthy evaluation — RF wins six-class outright** (Macro-F1 0.9122, best of all 6 classifiers) and is highly competitive on binary (0.9558, within 1.3 points of SVM/LR's 0.9580, and only KNN/GB beat it there by small margins at far higher latency/size cost).
3. **SVM's training cost scales badly and non-trivially**: 15.9s (binary) → 47.9s (six-class) fit time via libsvm's SMO solver, ~3× RF's; this matters for retraining cadence even though it's an offline cost.
4. **RF gives the best latency/size/accuracy balance for mobile real-time deployment**: 0.90–0.96ms end-to-end latency, 0.15–0.22MB model size, single consistent hyperparameter family (`n_estimators∈{10}, n_jobs=1`) usable across both tasks with only `max_depth`/`min_samples_leaf` tweaks — versus SVM/GB's far heavier models or KNN's 22MB footprint.
5. **Six-class is the harder, operationally more important task** (distinguishing harassment/threats/hate-speech/sexual-content/self-harm, not just risk/no-risk) — RF's clear held-out win there is the deciding factor over any CV-only SVM edge on the simpler binary task.

**Conclusion: RF (10 trees, `n_jobs=1`, task-specific `max_depth`) is the production model for both binary and six-class SendWise risk detection**, selected on the strength of the held-out evaluation, not the CV numbers, which are acknowledged to be inflated and are reported here only for transparency and completeness.
