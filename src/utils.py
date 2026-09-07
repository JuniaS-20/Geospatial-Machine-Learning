"""Small reusable utilities."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def check_projected_metric_crs(crs: rasterio.crs.CRS) -> None:
    if crs is None:
        raise ValueError("Raster/vector CRS is missing.")
    if crs.is_geographic:
        raise ValueError(
            "Spatial block distances are expressed in projected coordinate units. "
            "Use a projected CRS (typically a local UTM CRS) before spatial validation."
        )


def valid_mask_from_masked_bands(bands: np.ma.MaskedArray) -> np.ndarray:
    data = np.asarray(bands.filled(np.nan), dtype=np.float32)
    valid = np.all(np.isfinite(data), axis=0)
    if np.ma.isMaskedArray(bands):
        valid &= ~np.any(np.ma.getmaskarray(bands), axis=0)
    return valid


def xy_to_rowcol(transform, x: float, y: float) -> tuple[int, int]:
    """Convert x/y to NumPy row/column indices.

    Affine inverse returns (column, row), which is deliberately converted to
    the (row, column) order used by NumPy/raster arrays.
    """
    col_f, row_f = ~transform * (x, y)
    return int(np.floor(row_f)), int(np.floor(col_f))


def environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "scikit_learn": sklearn.__version__,
    }
