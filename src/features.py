"""Spectral feature definitions and sklearn-compatible feature transforms."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = b.copy()
    denom[np.abs(denom) < eps] = np.nan
    return a / denom


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _safe_ratio(a - b, a + b)


def compute_spectral_indices(
    bands: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute only indices supported by the available bands.

    No red-edge index is fabricated if the required red-edge band is absent.
    """
    out: dict[str, np.ndarray] = {}

    if {"B08", "B04"} <= bands.keys():
        out["NDVI"] = _normalized_difference(bands["B08"], bands["B04"])
        out["RATIO_NIR_RED"] = _safe_ratio(bands["B08"], bands["B04"])
    if {"B03", "B08"} <= bands.keys():
        out["NDWI"] = _normalized_difference(bands["B03"], bands["B08"])
    if {"B11", "B08"} <= bands.keys():
        out["NDBI"] = _normalized_difference(bands["B11"], bands["B08"])
        out["NDMI"] = _normalized_difference(bands["B08"], bands["B11"])
        out["RATIO_SWIR1_NIR"] = _safe_ratio(bands["B11"], bands["B08"])
    if {"B12", "B08"} <= bands.keys():
        out["NBR"] = _normalized_difference(bands["B08"], bands["B12"])
        out["RATIO_SWIR2_NIR"] = _safe_ratio(bands["B12"], bands["B08"])
    if {"B08", "B05"} <= bands.keys():
        out["NDRE_B05"] = _normalized_difference(bands["B08"], bands["B05"])
    if {"B8A", "B05"} <= bands.keys():
        out["NDRE_B8A_B05"] = _normalized_difference(bands["B8A"], bands["B05"])
    if {"B06", "B05"} <= bands.keys():
        out["RE_RATIO_B06_B05"] = _safe_ratio(bands["B06"], bands["B05"])
    if {"B07", "B05"} <= bands.keys():
        out["RE_RATIO_B07_B05"] = _safe_ratio(bands["B07"], bands["B05"])

    return out


class FeatureSubsetTransformer(BaseEstimator, TransformerMixin):
    """Select fixed columns without learning from labels."""

    def __init__(self, indices: tuple[int, ...] | list[int]):
        self.indices = tuple(indices)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in self.indices], dtype=object)
        values = np.asarray(input_features, dtype=object)
        return values[list(self.indices)]


@dataclass(frozen=True)
class FeatureDefinition:
    all_names: tuple[str, ...]
    raw_band_names: tuple[str, ...]

    @property
    def raw_indices(self) -> tuple[int, ...]:
        mapping = {name: i for i, name in enumerate(self.all_names)}
        return tuple(mapping[name] for name in self.raw_band_names)
