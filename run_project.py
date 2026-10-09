"""Run the complete, reproducible session-conversion experiment.

Usage: python run_project.py
All model choices use training folds; only a threshold uses validation data.
The untouched test partition is scored after those choices are frozen.
"""
from pathlib import Path
import hashlib
import json
import platform
import sqlite3
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from threadpoolctl import threadpool_limits

from session_model import RAW_FEATURES, SEED, fingerprint_groups, make_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
SOURCE = ROOT / "data" / "online_shoppers_intention.csv"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_splits(X, y, groups):
    trainval, test = next(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    ).split(X, y, groups))
    train_rel, val_rel = next(StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=SEED + 1
    ).split(X.iloc[trainval], y.iloc[trainval], groups[trainval]))
    return {"train": trainval[train_rel], "validation": trainval[val_rel], "test": test}


def score(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def wilson_interval(positive, total):
    z = 1.959963984540054
    p = positive / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return centre-radius, centre+radius


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(SOURCE)
    target = raw["Revenue"].astype(str).str.lower().map({"true": 1, "false": 0})
    if target.isna().any():
        raise ValueError("Revenue must contain only True or False.")
    y = target.astype(int)
    X = raw[RAW_FEATURES].copy()
    groups = fingerprint_groups(X)
    splits = make_splits(X, y, groups)
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(groups[splits[a]]) & set(groups[splits[b]])
    assignment = pd.DataFrame({"row_id": np.arange(len(raw)), "group_id": groups.astype(str), "split": ""})
    for name, ix in splits.items():
        assignment.loc[ix, "split"] = name
    assignment.to_csv(OUT/"splits.csv", index=False)
    tr, va, te = (splits[k] for k in ["train", "validation", "test"])
    audit = {
        "rows": len(raw), "raw_columns": len(raw.columns),
        "missing_cells": int(raw.isna().sum().sum()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_predictor_rows": int(X.duplicated().sum()),
        "duplicates_policy": "Retained; identical predictor rows are grouped into the same partition and CV fold.",
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "excluded_predictor": "PageValues",
        "split_summary": {n: {"rows": len(ix), "positive": int(y.iloc[ix].sum()), "positive_rate": float(y.iloc[ix].mean())} for n, ix in splits.items()},
    }
    write_json(OUT/"data_audit.json", audit)

    # Store the original public records with split labels for inspectable SQL EDA.
    database = raw.copy()
    database["Revenue"] = y
    database["split"] = assignment["split"]
    database["row_id"] = assignment["row_id"]
    with sqlite3.connect(OUT/"sessions.sqlite") as con:
        database.to_sql("sessions", con, if_exists="replace", index=False)
        visitor = pd.read_sql_query('''SELECT VisitorType AS visitor_type,
            COUNT(*) AS sessions, SUM(Revenue) AS purchases,
            AVG(Revenue) AS conversion_rate FROM sessions
            WHERE split = 'train' GROUP BY VisitorType ORDER BY sessions DESC''', con)
        monthly = pd.read_sql_query('''SELECT Month AS month, COUNT(*) AS sessions,
            SUM(Revenue) AS purchases, AVG(Revenue) AS conversion_rate
            FROM sessions WHERE split = 'train' GROUP BY Month''', con)
    intervals = [wilson_interval(r.purchases, r.sessions) for r in visitor.itertuples()]
    visitor["ci95_low"], visitor["ci95_high"] = zip(*intervals)
    visitor.to_csv(OUT/"visitor_segments_train.csv", index=False)
    monthly.to_csv(OUT/"monthly_segments_train.csv", index=False)
    contingency = pd.crosstab(raw.iloc[tr]["VisitorType"], y.iloc[tr])
    chi2, pvalue, dof, expected = chi2_contingency(contingency, correction=False)
    statistics = {"analysis_partition": "train", "chi2": float(chi2), "p_value": float(pvalue),
        "degrees_of_freedom": int(dof), "minimum_expected_cell": float(expected.min()),
        "interpretation": "Exploratory association only; neither causality nor future business impact."}
    write_json(OUT/"statistics.json", statistics)

    candidates = {
        "Dummy baseline": make_pipeline(DummyClassifier(strategy="prior")),
        "Logistic regression": make_pipeline(LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED)),
        "Gradient boosting": make_pipeline(HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, early_stopping=False, random_state=SEED)),
    }
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED+2).split(X.iloc[tr], y.iloc[tr], groups[tr]))
    cv_rows = []
    for name, model in candidates.items():
        print(f"Cross-validating {name}...", flush=True)
        result = cross_validate(model, X.iloc[tr], y.iloc[tr], cv=cv,
            scoring={"ap": "average_precision", "roc_auc": "roc_auc"}, n_jobs=1, error_score="raise")
        cv_rows.append({"model": name, "cv_ap_mean": float(result["test_ap"].mean()),
            "cv_ap_std": float(result["test_ap"].std(ddof=1)),
            "cv_roc_auc_mean": float(result["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(result["test_roc_auc"].std(ddof=1))})
    comparison = pd.DataFrame(cv_rows).sort_values("cv_ap_mean", ascending=False)
    comparison.to_csv(OUT/"cv_comparison.csv", index=False)
    winner_name = comparison.iloc[0]["model"]
    winner = clone(candidates[winner_name]).fit(X.iloc[tr], y.iloc[tr])
    print(f"Selected {winner_name} by training CV average precision.", flush=True)
    val_proba = winner.predict_proba(X.iloc[va])[:, 1]
    grid = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        result = score(y.iloc[va], val_proba, threshold)
        grid.append({"threshold": float(threshold), **{k:result[k] for k in ["precision", "recall", "f1"]}})
    thresholds = pd.DataFrame(grid)
    thresholds.to_csv(OUT/"validation_thresholds.csv", index=False)
    picked = thresholds.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    threshold = float(picked["threshold"])

    # Selection is finished. Score the final model on the test partition once.
    test_proba = winner.predict_proba(X.iloc[te])[:, 1]
    test_metrics = score(y.iloc[te], test_proba, threshold)
    validation_metrics = score(y.iloc[va], val_proba, threshold)
    predictions = pd.DataFrame({"row_id": te, "actual": y.iloc[te].to_numpy(),
        "score": test_proba, "predicted": (test_proba >= threshold).astype(int)})
    predictions.to_csv(OUT/"test_predictions.csv", index=False)
    joblib.dump({"pipeline": winner, "threshold": threshold, "features": RAW_FEATURES}, OUT/"model.joblib")
    X.iloc[te[:5]].to_csv(ROOT/"data/example_sessions.csv", index=False)

    # Inspection is on validation data, after selection. It is descriptive only.
    importance = permutation_importance(winner, X.iloc[va], y.iloc[va],
        scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    ranks = pd.DataFrame({"feature": RAW_FEATURES, "ap_decrease": importance.importances_mean,
        "std": importance.importances_std}).sort_values("ap_decrease", ascending=False)
    ranks.to_csv(OUT/"validation_feature_importance.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    counts = y.iloc[tr].value_counts().sort_index()
    axes[0].bar(["No purchase", "Purchase"], counts.values, color=["#75849a", "#166a86"])
    for i, n in enumerate(counts.values): axes[0].text(i, n+60, f"{n:,}", ha="center")
    axes[0].set(title="Class balance in training data", ylabel="Sessions", ylim=(0, counts.max()*1.15))
    positions = np.arange(len(visitor))
    rates = visitor["conversion_rate"].to_numpy()
    errors = np.vstack([rates-visitor["ci95_low"], visitor["ci95_high"]-rates])*100
    axes[1].bar(positions, rates*100, color="#166a86", yerr=errors, capsize=4)
    axes[1].set_xticks(positions, visitor.visitor_type.str.replace("_", " "))
    axes[1].set(title="Purchase rate by visitor type", ylabel="Purchase rate (%)")
    axes[1].text(0.01, 0.98, "Training data; 95% Wilson intervals", transform=axes[1].transAxes, va="top", fontsize=9)
    fig.savefig(OUT/"eda.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    precision, recall, _ = precision_recall_curve(y.iloc[te], test_proba)
    axes[0].plot(recall, precision, color="#166a86", linewidth=2,
        label=f"{winner_name}: AP {test_metrics['average_precision']:.3f}")
    axes[0].axhline(y.iloc[te].mean(), color="#75849a", linestyle="--", label=f"No-skill baseline: {y.iloc[te].mean():.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Final test precision and recall", xlim=(0,1), ylim=(0,1.02))
    axes[0].legend(fontsize=8, loc="upper right")
    sns.heatmap(test_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"], ax=axes[1])
    axes[1].set(title=f"Test confusion matrix at threshold {threshold:.2f}", xlabel="Predicted", ylabel="Actual")
    fig.savefig(OUT/"evaluation.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.7), layout="constrained")
    top = ranks.head(8).sort_values("ap_decrease")
    ax.barh(top.feature, top.ap_decrease, xerr=top["std"], color="#166a86", capsize=3)
    ax.set(title="Most influential inputs on validation data", xlabel="Decrease in average precision after permutation")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.savefig(OUT/"feature_importance.png", dpi=170); plt.close(fig)

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "seaborn": sns.__version__, "joblib": joblib.__version__}
    metrics = {"selected_model": winner_name, "selection_metric": "mean training CV average precision",
        "validation_threshold": threshold, "threshold_selection": "maximum validation F1 on a fixed grid",
        "test": test_metrics, "validation": validation_metrics,
        "test_baseline_ap": float(y.iloc[te].mean()), "split_summary": audit["split_summary"],
        "seed": SEED, "versions": versions, "runtime_seconds": round(time.perf_counter()-start, 2)}
    write_json(OUT/"metrics.json", metrics)
    lines = ["# Session Conversion Analysis Results", "", "## Question", "",
        "Can the supplied session attributes distinguish sessions that ended in a purchase? This is an offline classification study, not a live pre-checkout system.", "",
        "## Data and preparation", "",
        f"The public UCI dataset contains {len(raw):,} sessions. The file has {audit['missing_cells']} missing cells and {audit['exact_duplicate_rows']} exact duplicate rows. Repeated predictor rows are retained and kept together across all partitions and cross-validation folds.", "",
        "PageValues is omitted because its transaction-linked definition creates a potential information-availability concern. This is a conservative design choice, not proof that the original column is leaked. Four row-level features describe total pages, duration, seconds per page and product-page share. Imputation, scaling and encoding are learned within each training fold.", "",
        "| Partition | Sessions | Purchases | Purchase rate |", "|---|---:|---:|---:|"]
    for n, entry in audit["split_summary"].items(): lines.append(f"| {n} | {entry['rows']:,} | {entry['positive']:,} | {entry['positive_rate']:.2%} |")
    lines += ["", "## Training model comparison", "", "All comparisons use three folds inside the training partition. The same folds are used for every candidate; average precision is the preselected ranking metric.", "",
        "| Model | Mean CV average precision | Fold standard deviation | Mean CV ROC-AUC |", "|---|---:|---:|---:|"]
    for r in comparison.itertuples(): lines.append(f"| {r.model} | {r.cv_ap_mean:.4f} | {r.cv_ap_std:.4f} | {r.cv_roc_auc_mean:.4f} |")
    lines += ["", "## Final test evaluation", "", f"Selected model: **{winner_name}**. The threshold **{threshold:.2f}** was chosen using validation F1 before inspecting the test scores. The model remains fitted on training rows only; it is not refitted after threshold selection.", "",
        "| Metric | Test result |", "|---|---:|"]
    for k in ["roc_auc", "average_precision", "precision", "recall", "f1", "accuracy"]: lines.append(f"| {k.replace('_', ' ')} | {test_metrics[k]:.4f} |")
    lines += [f"| No-skill average precision | {y.iloc[te].mean():.4f} |", "", "![Model evaluation](evaluation.png)", "",
        "## Exploratory statistics", "", "Visitor-type purchase rates use training data only and include Wilson confidence intervals. A chi-square test describes association; it does not establish causation. These intervals assume independent sessions and may be optimistic if repeated feature rows reflect correlated observations.", "",
        f"Chi-square = {chi2:.3f}; degrees of freedom = {dof}; p = {pvalue:.4g}; minimum expected count = {expected.min():.2f}.", "", "![Exploratory analysis](eda.png)", "", "![Validation importance](feature_importance.png)", "",
        "## Limits", "", "The source is a historical, single-site dataset. A grouped random split does not prove performance on a later period, another site or live events. Session-level aggregates are not a timestamped event stream. The output scores are not calibrated probabilities. There is no claimed revenue lift, deployment, experiment on real customers, or research-paper replication. Correlated features can share permutation importance, and importance is not causal.", "",
        "## Reproducibility", "", f"Seed: {SEED}. CSV SHA-256: `{audit['sha256']}`. Exact package versions are recorded in `metrics.json`. Run `python run_project.py` and `python verify_project.py` from the project folder.", "",
        "## Source", "", "Sakar, C. and Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q . Dataset license: CC BY 4.0. See `../DATA_SOURCE.md`."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"selected_model": winner_name, "threshold": threshold, "test": test_metrics, "runtime_seconds": metrics["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
