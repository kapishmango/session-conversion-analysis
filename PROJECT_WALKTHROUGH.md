# Project walkthrough

## What problem does it solve?

It examines whether the attributes of an online shopping session distinguish sessions that ended in a purchase. It is a historical, offline classification study. It does not predict revenue amounts or demonstrate that a company can increase sales by using the model.

## Why this is a data-science project

The work follows the complete path from data audit and descriptive analysis to a reproducible modelling dataset, model comparison, evaluation and error inspection. SQL and charts communicate patterns. The model is one part of that process.

## What happens to the data?

`Revenue` becomes a binary target. Sixteen allowed input columns remain after excluding `Revenue` and `PageValues`. The feature transformer adds total pages, total duration, seconds per page, and the share of pages related to products. Zero-page sessions receive zero for the two ratios; missing values propagate until training-only imputation handles them.

There are 125 exactly repeated rows in the supplied file. Without reliable observation IDs, they cannot automatically be declared corrupt duplicates. They are retained, but identical input patterns are grouped so the same pattern cannot appear in both training and evaluation partitions. This is stricter than an ordinary random row split.

## Why are there three partitions?

| Partition | Rows | Purpose |
|---|---:|---|
| Training | 7,398 | Compare candidates with three folds and fit the selected model |
| Validation | 2,466 | Pick the classification threshold and inspect the fixed model |
| Test | 2,466 | Measure the final chosen model once |

The three training folds refit imputation, scaling and encoding separately. This stops information from an evaluation fold from influencing the fitted preprocessing. After the threshold is chosen, the model is not refitted on training plus validation; that keeps the scores aligned with the model for which the threshold was selected.

## Why compare multiple models?

The dummy baseline ignores all input features. It answers whether the more complex models learn anything useful. Logistic regression is a straightforward linear baseline. Gradient boosting can represent nonlinear patterns and interactions. Model selection uses mean cross-validation average precision, specified before examining the final test performance.

## Why not advertise accuracy alone?

Only about 15.5% of the sessions ended with a purchase. A classifier that always predicts no purchase would have about 84.5% accuracy but would detect no purchases. The project therefore reports ranking metrics and the precision/recall tradeoff.

The selected threshold is 0.20. On the test data, the model detected 247 of 381 purchases and flagged 511 sessions that did not purchase. This gives recall of 64.8%, precision of 32.6%, and F1 of 0.434. The false-positive count matters; this is a learning project, not a deployment recommendation.

## What do ROC-AUC and average precision mean?

ROC-AUC summarizes how well the scores rank positive cases above negative cases across thresholds. A value of 0.785 is a ranking measure, not 78.5% classification accuracy.

Average precision summarizes precision across recall levels. It is useful here because purchases are uncommon. The test average precision is 0.373, compared with a constant-score baseline of 0.155. Average precision is not identical to a trapezoidal area under the precision-recall curve, so the report uses the correct name.

## What statistical work is included?

Training-only SQL queries calculate purchase rates by visitor type and month. Wilson 95% confidence intervals describe uncertainty around visitor-type rates. A chi-square test checks whether visitor type and purchase outcome are associated. The small p-value does not show that changing a visitor's type causes a purchase. Historical sampling and potentially correlated observations limit the interpretation.

## What is permutation importance?

After the model is fixed, one validation input is shuffled and the drop in average precision is measured. A larger drop indicates that this trained model relies more on that input. Correlated inputs can substitute for each other, so importance values are not a complete explanation and do not prove causality.

## What should you say about your contribution?

This implementation was prepared with AI assistance. Describe the parts you have actually reviewed, rerun, understood or extended. Do not present it as past employment, an independent publication, or an earlier university project. Keep a short change log as you make your own changes, and retain the original benchmark results separately.

## Questions to practise

1. Why can a high-accuracy model be useless when classes are imbalanced?
2. Which inputs were excluded and what prediction-time concern motivated that choice?
3. Why are repeated predictor patterns kept in the same partition?
4. Where are scaling and imputation fitted?
5. How were the model and threshold selected, and which data informed each decision?
6. What do the false positives imply for any proposed use?
7. How would you collect timestamped events and design a future-period test for a live system?
8. What did you personally change after reviewing the supplied implementation?

## References

- Dataset and variable definitions: https://doi.org/10.24432/C5F88Q
- scikit-learn on data leakage: https://scikit-learn.org/stable/common_pitfalls.html
- Grouped stratification: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
