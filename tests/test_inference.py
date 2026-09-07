from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.inference import majority_filter_classification, sample_prediction_raster


def test_majority_filter_preserves_nodata_and_removes_isolated_class(tmp_path: Path):
    src_path = tmp_path / "raw.tif"
    dst_path = tmp_path / "filtered.tif"
    arr = np.array(
        [
            [-9999, -9999, -9999, -9999, -9999],
            [-9999, 1, 1, 1, -9999],
            [-9999, 1, 2, 1, -9999],
            [-9999, 1, 1, 1, -9999],
            [-9999, -9999, -9999, -9999, -9999],
        ],
        dtype=np.int32,
    )
    profile = {
        "driver": "GTiff", "height": 5, "width": 5, "count": 1, "dtype": "int32",
        "crs": "EPSG:32635", "transform": from_origin(0, 50, 10, 10), "nodata": -9999,
    }
    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(arr, 1)
    majority_filter_classification(src_path, dst_path, 3, -9999)
    with rasterio.open(dst_path) as src:
        result = src.read(1)
    assert result[0, 0] == -9999
    assert result[2, 2] == 1


def test_sample_prediction_raster(tmp_path: Path):
    path = tmp_path / "pred.tif"
    arr = np.array([[4, 5], [7, 8]], dtype=np.int32)
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1, dtype="int32",
        crs="EPSG:32635", transform=from_origin(0, 20, 10, 10), nodata=-9999,
    ) as dst:
        dst.write(arr, 1)
    values = sample_prediction_raster(path, np.array([[5, 15], [15, 5]], dtype=float))
    assert values.tolist() == [4, 8]
