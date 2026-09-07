"""Training sample extraction for point and polygon reference data."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import Point

from .config import TrainingConfig
from .utils import check_projected_metric_crs, valid_mask_from_masked_bands, xy_to_rowcol


@dataclass
class TrainingDataset:
    X: np.ndarray
    y: np.ndarray
    coordinates: np.ndarray
    feature_names: list[str]
    class_names: dict[int, str]
    source_feature_ids: np.ndarray
    extraction_report: dict[str, int]


def _coerce_labels(gdf: gpd.GeoDataFrame, cfg: TrainingConfig) -> tuple[np.ndarray, dict[int, str]]:
    if cfg.label_column not in gdf.columns:
        raise ValueError(
            f"Training column '{cfg.label_column}' is missing. Available columns: {list(gdf.columns)}"
        )
    raw = gdf[cfg.label_column]
    if raw.isna().any():
        raise ValueError(f"Column '{cfg.label_column}' contains missing labels.")

    numeric = np.asarray(raw)
    try:
        as_float = numeric.astype(float)
        if np.all(np.isfinite(as_float)) and np.allclose(as_float, np.round(as_float)):
            y = np.round(as_float).astype(np.int32)
        else:
            raise ValueError
    except (ValueError, TypeError):
        unique = sorted(map(str, np.unique(numeric)))
        mapping = {value: i + 1 for i, value in enumerate(unique)}
        y = np.asarray([mapping[str(v)] for v in numeric], dtype=np.int32)

    name_col = next((c for c in cfg.class_name_columns if c in gdf.columns), None)
    class_names: dict[int, str] = {}
    for code in np.unique(y):
        idx = np.flatnonzero(y == code)[0]
        class_names[int(code)] = str(gdf.iloc[idx][name_col]) if name_col else str(code)
    return y, class_names


def _sample_point(src: rasterio.DatasetReader, point: Point):
    row, col = xy_to_rowcol(src.transform, point.x, point.y)
    if not (0 <= row < src.height and 0 <= col < src.width):
        return None
    data = src.read(window=rasterio.windows.Window(col, row, 1, 1), masked=True)[:, 0, 0]
    if np.ma.is_masked(data) or not np.all(np.isfinite(np.asarray(data.filled(np.nan), dtype=float))):
        return None
    return np.asarray(data, dtype=np.float32), (point.x, point.y)


def _sample_polygon(
    src: rasterio.DatasetReader,
    geom,
    max_pixels: int,
    rng: np.random.Generator,
):
    left, bottom, right, top = geom.bounds
    window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
    full = rasterio.windows.Window(0, 0, src.width, src.height)
    try:
        window = window.intersection(full)
    except rasterio.errors.WindowError:
        return []
    if window.width <= 0 or window.height <= 0:
        return []

    bands = src.read(window=window, masked=True)
    valid = valid_mask_from_masked_bands(bands)
    w_transform = src.window_transform(window)
    inside = geometry_mask([geom], out_shape=(int(window.height), int(window.width)), transform=w_transform, invert=True)
    candidates = np.argwhere(valid & inside)
    if candidates.size == 0:
        return []
    if len(candidates) > max_pixels:
        candidates = candidates[rng.choice(len(candidates), size=max_pixels, replace=False)]

    data = np.asarray(bands.filled(np.nan), dtype=np.float32)
    samples = []
    for r, c in candidates:
        pixel = data[:, r, c]
        if not np.all(np.isfinite(pixel)):
            continue
        x, y = rasterio.transform.xy(w_transform, int(r), int(c), offset="center")
        samples.append((pixel, (float(x), float(y))))
    return samples


def extract_training_dataset(
    feature_stack: str,
    training_path: str,
    feature_names: list[str],
    cfg: TrainingConfig,
) -> TrainingDataset:
    gdf = gpd.read_file(training_path)
    if gdf.empty:
        raise ValueError("Training vector contains no features.")
    if gdf.crs is None:
        raise ValueError("Training vector has no CRS.")

    feature_labels, class_names = _coerce_labels(gdf, cfg)
    rng = np.random.default_rng(cfg.random_state)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    coordinates: list[tuple[float, float]] = []
    source_ids: list[int] = []
    report = {
        "source_objects": int(len(gdf)), "sampled_pixels": 0, "skipped_objects": 0,
        "point_objects": 0, "polygon_objects": 0, "unsupported_objects": 0,
    }

    with rasterio.open(feature_stack) as src:
        if cfg.require_projected_crs:
            check_projected_metric_crs(src.crs)
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        for feature_id, (row, label) in enumerate(zip(gdf.itertuples(), feature_labels)):
            geom = row.geometry
            if geom is None or geom.is_empty:
                report["skipped_objects"] += 1
                continue

            samples = []
            if geom.geom_type == "Point":
                report["point_objects"] += 1
                result = _sample_point(src, geom)
                if result is not None:
                    samples = [result]
            elif geom.geom_type in {"Polygon", "MultiPolygon"}:
                report["polygon_objects"] += 1
                samples = _sample_polygon(src, geom, cfg.max_pixels_per_geometry, rng)
            elif geom.geom_type == "MultiPoint":
                report["point_objects"] += 1
                samples = [r for p in geom.geoms if (r := _sample_point(src, p)) is not None]
            else:
                report["unsupported_objects"] += 1
                report["skipped_objects"] += 1
                continue

            if not samples:
                report["skipped_objects"] += 1
                continue
            for pixel, xy in samples:
                X_rows.append(pixel)
                y_rows.append(int(label))
                coordinates.append(xy)
                source_ids.append(feature_id)

    if not X_rows:
        raise ValueError("No valid training samples could be extracted from the feature stack.")
    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    coords = np.asarray(coordinates, dtype=np.float64)
    source_ids_arr = np.asarray(source_ids, dtype=np.int64)
    report["sampled_pixels"] = int(len(X))

    if X.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature stack has {X.shape[1]} bands but {len(feature_names)} feature names were supplied."
        )
    return TrainingDataset(X, y, coords, list(feature_names), class_names, source_ids_arr, report)
