# Online Shopping Session Conversion Analysis

A reproducible portfolio study of purchase outcomes in public e-commerce session data. The project includes a data audit, SQL exploration, feature engineering, model comparison, a saved scoring pipeline, and measured results.

This guided portfolio project was created with AI assistance. All reported results were produced by the included scripts. The original data comes from the credited UCI dataset authors.

## Start here

1. Read `PROJECT_WALKTHROUGH.md` to understand the question and modelling choices.
2. Read `artifacts/report.md` and inspect its three charts.
3. Run the project and verification commands below.
4. Use the walkthrough to understand the split, model comparison, threshold and limitations.

## Run on Windows

Python 3.12 is recommended. Clone or download the repository, then open a terminal in the project folder.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_project.py
python verify_project.py
python predict.py data/example_sessions.csv
```

In PyCharm, choose this folder as the project, select `.venv\Scripts\python.exe` as the interpreter, and run `run_project.py`. All file paths are relative to the scripts' location, so the experiment does not need changes to your folder names. The training run itself needs no network access because the credited source data is included. Installing dependencies needs internet access.

On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`; the remaining Python commands are the same.

## Experiment

- Source: 12,330 public shopping sessions; target `Revenue` indicates whether the session ended in a purchase.
- Inputs: page counts and durations, bounce and exit rates, visitor type and other supplied session attributes.
- Data handling: retain repeated records but put identical predictor rows in the same split and cross-validation fold.
- Feature engineering: total pages, total duration, seconds per page and product-page share.
- Preprocessing: median imputation and numeric scaling plus one-hot categorical encoding, fitted inside each training fold.
- Model comparison: dummy baseline, balanced logistic regression and histogram gradient boosting; three training folds, selected by average precision.
- Threshold selection: maximum validation F1 on a fixed grid. The final model is not refitted after this step.
- Evaluation: one separate test partition of 2,466 rows. Neither model selection nor threshold selection uses test performance.

`PageValues` is deliberately excluded because its transaction-related definition raises a concern about what would be known when predicting. This is a conservative choice, not proof of leakage in the original dataset. Since these are session-level aggregates, this study makes no claim of live, early-session prediction.

## Recorded results

| Measure | Result |
|---|---:|
| Selected model | Gradient boosting |
| Test ROC-AUC | 0.7851 |
| Test average precision | 0.3726 |
| No-skill test average precision | 0.1545 |
| Test precision at threshold 0.20 | 0.3259 |
| Test recall at threshold 0.20 | 0.6483 |
| Test F1 at threshold 0.20 | 0.4337 |

These are measured offline results, not a claim of business impact. Precision at the selected threshold is limited: many flagged sessions do not purchase. Choosing a threshold to maximize F1 is an educational choice; a deployed system would need explicit costs and a new evaluation protocol.

The full confusion matrix, accuracy, validation scores, split sizes and package versions are in `artifacts/metrics.json`. Cross-validation results are in `artifacts/cv_comparison.csv`. Small numerical differences can occur across platforms and package versions. If you change the experiment after studying its test results, use a new independent evaluation set before making a new generalization claim.

## Files

| File | Purpose |
|---|---|
| `run_project.py` | Run the audit, SQL analysis, modelling and report generation |
| `session_model.py` | Input validation, row-level features and preprocessing pipeline |
| `predict.py` | Score new records from a CSV using the saved model |
| `verify_project.py` | Check split isolation, excluded inputs, edge cases and metrics |
| `queries.sql` | Readable SQL used for segment exploration |
| `DATA_SOURCE.md` | Dataset source, license, attribution and download URL |
| `PROJECT_WALKTHROUGH.md` | Method explanations and questions to practise |
| `data/online_shoppers_intention.csv` | Original public source data |
| `data/example_sessions.csv` | Five example inputs drawn from the test partition |
| `artifacts/report.md` | Results and limitations, with charts |
| `artifacts/model.joblib` | Saved preprocessing and trained model |
| `artifacts/sessions.sqlite` | Public records and explicit split assignments |
| `artifacts/splits.csv` | Row and predictor-group partition assignments |
| `artifacts/test_predictions.csv` | Auditable final test predictions |

Do not load model files from unknown sources: Python serialization can execute code. This package contains the model produced by its included training script; retraining is preferable when changing Python or scikit-learn versions.

## Scope

This is structured-data classification and statistical exploration. It does not include LLM fine-tuning, a replicated research paper, a production deployment, or independently collected customer observations. The source dataset is public; generated features and analysis tables are derived from it. See `DATA_SOURCE.md` for attribution.
