# SendWise Abusive Language Detection — Binary and Fine-Grained Benchmarking

This project supports two separate experiments using the categorised SendWise corpus:

1. **Binary risk detection**: `risk_label` (0 = non-risk, 1 = risk detected).
2. **Six-class risk categorisation**: `risk_category` (`non_risk`, `harassment`, `threats`, `hate_speech`, `sexual_content`, `self_harm_risk`).

The authoritative corpus is `Categorised_dataset.csv` (20,122 rows). `Updated_dataset.csv` is a backward-compatible copy containing the same records and an additional `label` alias for `risk_label`, so the repository's existing binary scripts can continue to run.

## Dataset distribution

| Category | Records |
|---|---:|
| Non-risk | 17,589 |
| Harassment | 950 |
| Threats | 550 |
| Hate speech | 400 |
| Sexual content | 333 |
| Self-harm risk | 300 |
| **Total** | **20,122** |

The CSV includes a preassigned `split` column (15,091 train / 5,031 test). The new benchmark uses that split instead of silently creating a different one. Keep the test split untouched when tuning the model.

## Setup

Use Python 3.10 or 3.11 in a virtual environment where possible.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

## Run the new benchmark

Run commands from the project root, where `benchmark_categorised.py` and `Categorised_dataset.csv` are located.

### Run both tasks

```bash
python benchmark_categorised.py --task both
```

### Run binary classification only

```bash
python benchmark_categorised.py --task binary
```

### Run six-class categorisation only

```bash
python benchmark_categorised.py --task multiclass
```

Optional arguments:

```bash
python benchmark_categorised.py --task both --latency-samples 500 --output-dir results/categorised_benchmark
```

The benchmark trains the repository's Random Forest using the existing lemmatisation + hybrid word/character TF-IDF representation. The same model family and feature pipeline are used for each task, but each task is trained against its appropriate target column. This is a hold-out evaluation using the CSV's train/test split, not the existing 5-fold cross-validation experiment.

## Output files

Outputs are written to `results/categorised_benchmark/`:

- `binary_benchmark_summary.csv` — binary task performance and resource metrics.
- `six_class_benchmark_summary.csv` — six-class performance and resource metrics.
- `all_tasks_benchmark_summary.csv` — one-row-per-task summary.
- `binary_classification_report.txt` and `six_class_classification_report.txt` — precision, recall and F1 by class.
- `binary_confusion_matrix.csv` and `six_class_confusion_matrix.csv` — confusion matrices.
- `binary_benchmark_per_message.csv` and `six_class_benchmark_per_message.csv` — individual latency samples.
- `binary_runtime_model.joblib` and `six_class_runtime_model.joblib` — saved trained pipelines.

Measurements include accuracy, macro/weighted precision-recall-F1, positive-class metrics for binary detection, fit time, classifier-only latency, end-to-end Python latency, process RSS memory observations, and saved pipeline size. RSS includes Python and imported libraries; it is not a model-only RAM measure. Python timings are not Android-device timings.

## Important interpretation notes

- The binary task predicts `risk_label`; the six-class task predicts `risk_category`.
- The six-class task has one non-risk class and five risk categories. It is not five-class classification.
- Existing files under `results/` may contain results from earlier experiments on a 20,114-row dataset. They are historical outputs and must not be described as results from the new 20,122-row categorised corpus unless regenerated.
- The new benchmark script reports a new held-out evaluation. Do not substitute its values for prior five-fold CV results in a paper without clearly identifying the evaluation protocol.
- The model used here is Random Forest. Existing experiment scripts can still compare classifiers for the binary target. Extending the six-class experiment to all six classifiers would be a separate experiment.
- Validate the pipeline on the actual Android device before making mobile real-time performance claims.

## Existing experiment scripts

- `single_algorithms_experiment.py` — binary classifier comparison.
- `ensemble_experiment.py` — binary hard-voting ensemble experiment.
- `feature_ablation_experiment.py` — binary feature ablation.
- `enhanced_experiment.py` — binary controlled evaluation and diagnostics.
- `benchmark_runtime.py` — earlier binary runtime benchmark; use `benchmark_categorised.py` for the new two-task benchmark.

## License

See `LICENSE` for the repository's license.
