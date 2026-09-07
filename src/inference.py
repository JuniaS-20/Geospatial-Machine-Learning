"""Memory-efficient window-based raster inference and post-processing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import generic_filter

from .config import InferenceConfig
from .evaluation import calculate_normalized_entropy
from .utils import valid_mask_from_masked_bands


def _probability_capable(model) -> bool:
    return callable(getattr(model, "predict_proba", None))


def classify_feature_stack(
    model,
    feature_stack: Path,
    outputs_dir: Path,
    cfg: InferenceConfig,
) -> dict[str, Path]:
    """Predict one raster block at a time; the full feature raster is never loaded into RAM."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    class_path = outputs_dir / "landcover_raw.tif"
    confidence_path = outputs_dir / "model_confidence.tif"
    uncertainty_path = outputs_dir / "uncertainty_proxy_1_minus_confidence.tif"
    entropy_path = outputs_dir / "entropy_uncertainty_proxy.tif"
    prob_path = outputs_dir / "class_probability_scores.tif"

    with rasterio.open(feature_stack) as src:
        int_profile = src.profile.copy()
        int_profile.update(count=1, dtype="int32", nodata=cfg.integer_nodata, compress="deflate", tiled=True)
        float_profile = src.profile.copy()
        float_profile.update(count=1, dtype="float32", nodata=cfg.float_nodata, compress="deflate", tiled=True)

        has_proba = _probability_capable(model)
        classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model.named_steps["classifier"].classes_)
        if has_proba and cfg.write_class_probability_scores:
            prob_profile = src.profile.copy()
            prob_profile.update(count=len(classes), dtype="float32", nodata=cfg.float_nodata, compress="deflate", tiled=True)
            prob_dst = rasterio.open(prob_path, "w", **prob_profile)
            for i, cls in enumerate(classes, start=1):
                prob_dst.set_band_description(i, f"class_{cls}_score")
        else:
            prob_dst = None

        with rasterio.open(class_path, "w", **int_profile) as class_dst, \
             rasterio.open(confidence_path, "w", **float_profile) as conf_dst, \
             rasterio.open(uncertainty_path, "w", **float_profile) as unc_dst, \
             rasterio.open(entropy_path, "w", **float_profile) as ent_dst:
            try:
                for _, window in src.block_windows(1):
                    bands = src.read(window=window, masked=True)
                    valid = valid_mask_from_masked_bands(bands)
                    h, w = valid.shape
                    class_arr = np.full((h, w), cfg.integer_nodata, dtype=np.int32)
                    conf_arr = np.full((h, w), cfg.float_nodata, dtype=np.float32)
                    unc_arr = np.full((h, w), cfg.float_nodata, dtype=np.float32)
                    ent_arr = np.full((h, w), cfg.float_nodata, dtype=np.float32)
                    prob_arr = None

                    if valid.any():
                        data = np.asarray(bands.filled(np.nan), dtype=np.float32)
                        X = data[:, valid].T
                        pred = model.predict(X).astype(np.int32)
                        class_arr[valid] = pred

                        if has_proba:
                            probabilities = np.asarray(model.predict_proba(X), dtype=np.float32)
                            confidence = probabilities.max(axis=1)
                            uncertainty = 1.0 - confidence
                            entropy = calculate_normalized_entropy(probabilities)
                            conf_arr[valid] = confidence
                            unc_arr[valid] = uncertainty
                            ent_arr[valid] = entropy
                            if prob_dst is not None:
                                prob_arr = np.full((len(classes), h, w), cfg.float_nodata, dtype=np.float32)
                                for k in range(len(classes)):
                                    layer = prob_arr[k]
                                    layer[valid] = probabilities[:, k]

                    class_dst.write(class_arr, 1, window=window)
                    conf_dst.write(conf_arr, 1, window=window)
                    unc_dst.write(unc_arr, 1, window=window)
                    ent_dst.write(ent_arr, 1, window=window)
                    if prob_dst is not None and prob_arr is not None:
                        prob_dst.write(prob_arr, window=window)
            finally:
                if prob_dst is not None:
                    prob_dst.close()

    result = {
        "classification_raw": class_path,
        "confidence": confidence_path,
        "uncertainty_proxy": uncertainty_path,
        "entropy_proxy": entropy_path,
    }
    if prob_path.exists():
        result["class_probability_scores"] = prob_path
    return result


def _expanded_window(window: Window, width: int, height: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    col0 = max(0, int(window.col_off) - halo)
    row0 = max(0, int(window.row_off) - halo)
    col1 = min(width, int(window.col_off + window.width) + halo)
    row1 = min(height, int(window.row_off + window.height) + halo)
    expanded = Window(col0, row0, col1 - col0, row1 - row0)
    r0 = int(window.row_off) - row0
    c0 = int(window.col_off) - col0
    center = (slice(r0, r0 + int(window.height)), slice(c0, c0 + int(window.width)))
    return expanded, center


def majority_filter_classification(raw_path: Path, output_path: Path, size: int, nodata: int = -9999) -> Path:
    """Apply a categorical majority (mode) filter with haloed windows.

    A median filter is intentionally not used: land-cover class codes are nominal,
    not ordinal, so their numeric median has no categorical meaning. NoData values
    are ignored when computing the local mode.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("majority filter size must be a positive odd integer.")
    halo = size // 2

    def local_mode(values):
        valid = values[values != nodata].astype(np.int64)
        if valid.size == 0:
            return nodata
        labels, counts = np.unique(valid, return_counts=True)
        winners = labels[counts == counts.max()]
        center = int(values[len(values) // 2])
        if center in winners:
            return center
        return winners.min()  # deterministic tie-break only when the center is not a winner

    with rasterio.open(raw_path) as src:
        profile = src.profile.copy()
        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                expanded, center = _expanded_window(window, src.width, src.height, halo)
                arr = src.read(1, window=expanded)
                original_valid = arr != nodata
                filtered = generic_filter(arr, function=local_mode, size=size, mode="nearest")
                filtered[~original_valid] = nodata
                dst.write(filtered[center].astype(np.int32), 1, window=window)
    return output_path


def sample_prediction_raster(path: Path, coordinates: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        values = [next(src.sample([(float(x), float(y))], indexes=1))[0] for x, y in coordinates]
    return np.asarray(values)
