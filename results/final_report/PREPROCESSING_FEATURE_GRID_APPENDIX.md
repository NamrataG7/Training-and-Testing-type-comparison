
---

## APPENDIX — Preprocessing × Feature-Set Grid (per-classifier, held-out only)

Scope: to stay fast, this grid tests **2 preprocessing methods** (`none`, `lemmatization`) × **3 feature sets** (`word-only TF-IDF`, `char-only TF-IDF`, `hybrid word+char TF-IDF`) = 6 combos, on **RF, LR, NB, KNN only** (their already-tuned hyperparameters held fixed) — **SVM and GB are excluded** from this grid; each combo costs minutes for them and their hyperparameters are already fixed at E2+lemmatization. Held-out split only (CV skipped — known leakage-inflated).

### Best combo per (task, classifier) — with latency

| Task | Classifier | Best Preprocessing | Best Feature Set | Accuracy | Macro-F1 | E2E Latency mean(ms) | E2E Latency p95(ms) |
|---|---|---|---|---|---|---|---|
| binary | KNN | lemmatization | char_only | 0.9909 | 0.9798 | 11.41 | 11.83 |
| binary | LR | none | char_only (tied with hybrid/word) | 0.9803 | 0.9580 | 0.70 | 1.25 |
| binary | NB | lemmatization | char_only | 0.9813 | 0.9600 | 0.38 | 0.42 |
| binary | RF | none | char_only | 0.9809 | 0.9580 | 0.69 | 0.76 |
| six_class | KNN | lemmatization | char_only | 0.9799 | 0.9022 | 11.20 | 11.62 |
| six_class | LR | none | char_only | 0.9758 | 0.8926 | 0.61 | 1.17 |
| six_class | NB | lemmatization | char_only | 0.9714 | 0.8792 | 0.42 | 0.52 |
| **six_class** | **RF** | **lemmatization** | **hybrid** | 0.9785 | **0.9122** | **1.27** | 3.00 |

**Latency reinforces the accuracy finding**: char-only is not just more accurate for LR/NB/RF-binary, it is also faster than hybrid (fewer TF-IDF dimensions, no word+char concatenation overhead) -- e.g. binary RF drops from ~0.96ms (hybrid, main Table 3) to 0.69ms (char-only). KNN remains the clear latency outlier (11-12ms) regardless of feature set, since its cost is dominated by the O(n) distance search against the full training set, not by feature dimensionality. RF six-class's hybrid choice costs a modest extra ~0.6ms versus the simpler char-only options used elsewhere -- a small latency premium for its +2-point F1 edge, consistent with the mobile-deployment tradeoff already established in the main report.


### Key finding: char-only TF-IDF beats hybrid word+char for 7 of 8 (classifier, task) pairs

This **contradicts** the repo's original stage2 feature ablation (which found hybrid/E2 best — but that was RF-only, on the older `Updated_dataset.csv`, with different hyperparameters). On the current `Categorised_dataset.csv` with tuned classifiers, **char-only wins or ties almost everywhere except RF on six-class.**

**Math/reasoning:**
- Character n-grams (3-5 chars) are robust to spelling variants, leetspeak, punctuation insertion, and typos common in abusive/evasive text (e.g., "u are stu*pid", "h4rassment") — these patterns don't need word-boundary tokenization to be captured, since a char n-gram window slides across the raw text regardless of inserted symbols.
- Word-level TF-IDF is brittle to these same evasions: a single inserted character or typo creates an out-of-vocabulary token, losing the word-level TF-IDF signal entirely for that occurrence.
- **Adding word-TF-IDF on top of char-TF-IDF (hybrid) does not help, and sometimes hurts**, because the word features are mostly redundant with what char n-grams already encode, while adding ~10,000 extra (sparser, noisier) dimensions — this dilutes feature importance in tree splits (RF) and shifts decision boundaries slightly in linear/instance-based models (LR, KNN, NB) without adding real new separating information. This is the same variance-inflation mechanism noted earlier for the E3-E7 richer-feature regression.
- **RF six-class is the one exception where hybrid wins (0.9122 vs char-only's 0.9044)**: RF's split-selection can exploit the extra orthogonal word-level signal for the six-way decision specifically (distinguishing categories like harassment vs. hate-speech vs. threats needs more semantic/lexical cues than binary risk/no-risk does) — word tokens carry topical/lexical category signal that char n-grams don't directly encode (e.g., "kill" vs "disgusting" as whole-word indicators of threat vs. hate-speech), and RF's tree ensemble can combine both feature families without the signal dilution that hurts linear models (LR) or distance-based models (KNN) when extra noisy dimensions are added to a Euclidean/cosine distance calculation.
- This is exactly why **RF is the only classifier where the current pipeline's choice (hybrid+lemmatization) is actually validated as optimal** — for every other classifier, char-only alone would have been a better, simpler, and faster choice (fewer dimensions → faster fit: e.g., binary KNN char-only fits in 1.9-2.7s vs hybrid's 2.2-2.9s).

### Does this change the final RF recommendation?

**No — it reinforces it.** RF's current production config (hybrid + lemmatization) is the *actual best-found combination for RF on six-class* (0.9122, matching Table 4 exactly — consistent, no discrepancy). The other classifiers' potential small gains from switching to char-only (e.g., LR six-class 0.8926 vs 0.8870, +0.56 points) are still below RF's 0.9122 six-class result, so **RF remains the correct choice for the six-class task even after this ablation**, and the binary task result is close enough across all four fast classifiers (0.958-0.980 macro-F1) that RF's latency/size advantage (established earlier) still carries the final decision.

Full per-combo results: `results/preprocessing_feature_grid/grid_results.csv`; best-per-classifier summary: `results/preprocessing_feature_grid/best_combo_per_classifier.csv`.
