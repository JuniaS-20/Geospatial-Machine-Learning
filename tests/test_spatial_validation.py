import numpy as np

from src.spatial_validation import create_label_blind_holdout, create_spatial_blocks


def test_create_spatial_blocks():
    coords = np.array([[10, 10], [20, 20], [1010, 10], [1015, 15]], dtype=float)
    groups = create_spatial_blocks(coords, block_size=1000)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


def test_label_blind_holdout_is_reproducible_and_has_no_overlap():
    groups = np.repeat(np.arange(10), 3)
    split1 = create_label_blind_holdout(groups, test_fraction=0.2, random_state=42)
    split2 = create_label_blind_holdout(groups, test_fraction=0.2, random_state=42)
    assert np.array_equal(split1.test_idx, split2.test_idx)
    assert np.intersect1d(split1.development_groups, split1.test_groups).size == 0
    assert len(split1.test_groups) == 2
