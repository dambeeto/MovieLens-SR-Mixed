"""Feature selection diagnostics: VarianceThreshold + mutual_info_classif vs cluster_id."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler


def variance_report(features: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Per-feature variance; columns below threshold are flagged as "low_variance" = True."""
    numeric = features.select_dtypes(include="number")
    var = numeric.var()
    out = (
        pd.DataFrame({"feature": var.index, "variance": var.values.astype("float32")})
        .assign(low_variance=lambda d: d["variance"] < threshold)
        .sort_values("variance")
        .reset_index(drop=True)
    )
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric.to_numpy())
    return out, selector


def mutual_info_report(
    features: pd.DataFrame, labels: np.ndarray, top_k: int = 20
) -> pd.DataFrame:
    """Top-K features ranked by mutual information against an arbitrary class label vector."""
    numeric = features.select_dtypes(include="number")
    X = StandardScaler().fit_transform(numeric.to_numpy())
    mi = mutual_info_classif(X, labels, random_state=0)
    return (
        pd.DataFrame({"feature": numeric.columns, "mutual_info": mi.astype("float32")})
        .sort_values("mutual_info", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
