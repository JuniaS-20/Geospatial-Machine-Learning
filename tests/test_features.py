import numpy as np

from src.features import compute_spectral_indices


def test_spectral_indices_are_computed_only_when_supported():
    bands = {
        "B03": np.array([[0.2]], dtype=float),
        "B04": np.array([[0.3]], dtype=float),
        "B08": np.array([[0.7]], dtype=float),
        "B11": np.array([[0.5]], dtype=float),
        "B12": np.array([[0.4]], dtype=float),
    }
    out = compute_spectral_indices(bands)
    assert "NDVI" in out and "NDWI" in out and "NDBI" in out
    assert "NDRE_B05" not in out
    assert np.isclose(out["NDVI"][0, 0], 0.4)
