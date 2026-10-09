"""Feature preparation shared by training and inference.

The only learned preprocessing lives in the scikit-learn pipeline. This module
adds deterministic, row-level features and rejects unusable input values.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "SpecialDay",
]
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
RAW_FEATURES = NUMERIC + CATEGORICAL
DERIVED = ["TotalPages", "TotalDuration", "SecondsPerPage", "ProductPageShare"]


class SessionFeatures(TransformerMixin, BaseEstimator):
    """Keep an explicit input allowlist and create four session features."""

    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pass a pandas DataFrame with named session columns.")
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        # Revenue and PageValues are deliberately absent from this allowlist.
        out = X[RAW_FEATURES].copy()
        for col in NUMERIC:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
            if np.isinf(out[col]).any() or out[col].lt(0).any():
                raise ValueError(f"{col} must be nonnegative and finite (or missing).")
        for col in ["BounceRates", "ExitRates", "SpecialDay"]:
            if out[col].gt(1).any():
                raise ValueError(f"{col} must be between 0 and 1.")
        for col in CATEGORICAL:
            out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)
        out["Weekend"] = out["Weekend"].str.lower()
        page_cols = ["Administrative", "Informational", "ProductRelated"]
        time_cols = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
        out["TotalPages"] = out[page_cols].sum(axis=1, min_count=3)
        out["TotalDuration"] = out[time_cols].sum(axis=1, min_count=3)
        denominator = out["TotalPages"].replace(0, np.nan)
        out["SecondsPerPage"] = out["TotalDuration"] / denominator
        out["ProductPageShare"] = out["ProductRelated"] / denominator
        zero_pages = out["TotalPages"].eq(0)
        out.loc[zero_pages, ["SecondsPerPage", "ProductPageShare"]] = 0.0
        return out


def make_pipeline(classifier):
    """Learn numeric imputation/scaling and category encoding on training data."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("numbers", numeric, NUMERIC + DERIVED),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([
        ("features", SessionFeatures()),
        ("preprocess", prep),
        ("model", classifier),
    ])


def fingerprint_groups(X):
    """Keep identical predictor rows together, even if their labels differ."""
    canonical = SessionFeatures().fit_transform(X)[RAW_FEATURES]
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy()
