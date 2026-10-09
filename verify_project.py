"""Check partition leakage, reported metrics and inference edge cases."""
from pathlib import Path
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
from session_model import SessionFeatures

ROOT = Path(__file__).resolve().parent


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT/"data/online_shoppers_intention.csv")
        cls.saved = joblib.load(ROOT/"artifacts/model.joblib")
        cls.metrics = json.loads((ROOT/"artifacts/metrics.json").read_text())

    def test_no_rows_or_identical_predictors_cross_partitions(self):
        split = pd.read_csv(ROOT/"artifacts/splits.csv", dtype={"group_id":str})
        self.assertEqual(len(split), len(self.raw))
        self.assertEqual(split.row_id.nunique(), len(self.raw))
        self.assertEqual(set(split.split), {"train", "validation", "test"})
        self.assertEqual(int(split.groupby("group_id")["split"].nunique().max()), 1)

    def test_revenue_and_page_value_cannot_enter_the_model(self):
        rows = self.raw.head(8).copy()
        a = self.saved["pipeline"].predict_proba(rows)
        rows["Revenue"] = ~rows["Revenue"]
        rows["PageValues"] = 9999999
        b = self.saved["pipeline"].predict_proba(rows)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        transformed = SessionFeatures().fit_transform(rows)
        self.assertNotIn("Revenue", transformed.columns)
        self.assertNotIn("PageValues", transformed.columns)

    def test_unseen_category_missing_numeric_and_zero_page_session(self):
        rows = self.raw.head(3).copy()
        rows.loc[rows.index[0], "VisitorType"] = "Previously_unseen_type"
        rows.loc[rows.index[1], "Informational_Duration"] = np.nan
        for col in ["Administrative", "Informational", "ProductRelated",
                    "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]:
            rows.loc[rows.index[2], col] = 0
        p = self.saved["pipeline"].predict_proba(rows)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.saved["pipeline"].predict_proba(self.raw.head(2).drop(columns=["Month"]))
        rows = self.raw.head(2).copy()
        rows.loc[rows.index[0], "ProductRelated"] = -1
        with self.assertRaises(ValueError):
            self.saved["pipeline"].predict_proba(rows)

    def test_published_metrics_match_saved_predictions(self):
        predictions = pd.read_csv(ROOT/"artifacts/test_predictions.csv")
        p = self.saved["pipeline"].predict_proba(self.raw.iloc[predictions.row_id])[:, 1]
        np.testing.assert_allclose(p, predictions["score"], rtol=1e-12, atol=1e-12)
        y = predictions.actual
        self.assertAlmostEqual(roc_auc_score(y, p), self.metrics["test"]["roc_auc"], places=10)
        self.assertAlmostEqual(average_precision_score(y, p), self.metrics["test"]["average_precision"], places=10)
        self.assertAlmostEqual(f1_score(y, p >= self.saved["threshold"]), self.metrics["test"]["f1"], places=10)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        unittest.main(verbosity=2)
