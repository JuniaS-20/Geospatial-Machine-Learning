import numpy as np
from affine import Affine

from src.utils import valid_mask_from_masked_bands, xy_to_rowcol


def test_raster_coordinate_indexing():
    transform = Affine.translation(100, 200) * Affine.scale(10, -10)
    row, col = xy_to_rowcol(transform, 125, 175)
    assert (row, col) == (2, 2)


def test_nodata_mask():
    data = np.ma.array(
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        mask=np.array([[[False, True], [False, False]], [[False, False], [False, False]]]),
    )
    valid = valid_mask_from_masked_bands(data)
    assert valid.tolist() == [[True, False], [True, True]]
