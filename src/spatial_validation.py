"""Spatial grouping, label-blind hold-out, and development CV helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class HoldoutSplit:
    development_idx: np.ndarray
    test_idx: np.ndarray
    development_groups: np.ndarray
    test_groups: np.ndarray
    excluded_gap_idx: np.ndarray


def create_spatial_blocks(
    coordinates: np.ndarray,
    block_size: float,
    origin: tuple[float, float] | None = None,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_samples, 2).")
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")
    if len(coordinates) == 0:
        raise ValueError("No coordinates were provided.")

    if origin is None:
        origin = (
            np.floor(coordinates[:, 0].min() / block_size) * block_size,
            np.floor(coordinates[:, 1].min() / block_size) * block_size,
        )
    bx = np.floor((coordinates[:, 0] - origin[0]) / block_size).astype(np.int64)
    by = np.floor((coordinates[:, 1] - origin[1]) / block_size).astype(np.int64)
    pairs = np.column_stack([bx, by])
    _, groups = np.unique(pairs, axis=0, return_inverse=True)
    return groups.astype(np.int64)


def create_label_blind_holdout(
    groups: np.ndarray,
    test_fraction: float = 0.20,
    random_state: int = 42,
    coordinates: np.ndarray | None = None,
    spatial_gap_distance: float = 0.0,
) -> HoldoutSplit:
    """Reserve test blocks without inspecting class labels.

    This function intentionally has no ``y`` parameter. The test set cannot be
    optimized for class representation, preventing the label-based test-set
    selection problem present in the previous script.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two spatial groups are required for a hold-out.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_groups)
    n_test = int(np.clip(np.round(len(unique_groups) * test_fraction), 1, len(unique_groups) - 1))
    test_groups = np.sort(shuffled[:n_test])
    test_mask = np.isin(groups, test_groups)
    dev_mask = ~test_mask
    excluded_gap = np.zeros(len(groups), dtype=bool)

    if spatial_gap_distance > 0:
        if coordinates is None:
            raise ValueError("coordinates are required when spatial_gap_distance > 0.")
        coordinates = np.asarray(coordinates, dtype=float)
        tree = cKDTree(coordinates[test_mask])
        distances, _ = tree.query(coordinates[dev_mask], k=1)
        dev_indices = np.flatnonzero(dev_mask)
        to_exclude = dev_indices[distances < spatial_gap_distance]
        excluded_gap[to_exclude] = True
        dev_mask[to_exclude] = False

    development_idx = np.flatnonzero(dev_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(development_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Hold-out/gap settings leave an empty development or test set.")

    dev_groups = np.unique(groups[development_idx])
    overlap = np.intersect1d(dev_groups, test_groups)
    if overlap.size:
        raise RuntimeError("Spatial group leakage detected between development and test sets.")

    return HoldoutSplit(
        development_idx=development_idx,
        test_idx=test_idx,
        development_groups=dev_groups,
        test_groups=test_groups,
        excluded_gap_idx=np.flatnonzero(excluded_gap),
    )


def class_block_report(y: np.ndarray, groups: np.ndarray) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for value in np.unique(y):
        mask = y == value
        report[str(value)] = {
            "samples": int(mask.sum()),
            "spatial_blocks": int(np.unique(groups[mask]).size),
        }
    return report


def ensure_development_class_support(
    all_classes: list[int] | np.ndarray,
    y_development: np.ndarray,
) -> None:
    """Ensure every declared class can be learned, without using test labels for split selection."""
    expected = set(np.asarray(all_classes).tolist())
    present = set(np.unique(y_development).tolist())
    missing = sorted(expected - present)
    if missing:
        raise ValueError(
            "The fixed label-blind split left class(es) absent from development data: "
            f"{missing}. The code will not reselect the hold-out using labels. Improve the geographic "
            "training design or pre-declare a different block design/seed before analysis."
        )


def holdout_class_coverage(all_classes, y_test: np.ndarray) -> dict[str, object]:
    expected = set(np.asarray(all_classes).tolist())
    present = set(np.unique(y_test).tolist())
    return {
        "classes_test": sorted(present),
        "classes_absent_from_test": sorted(expected - present),
    }


def determine_cv_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    unique_groups = np.unique(groups)
    counts = [np.unique(groups[y == c]).size for c in np.unique(y)]
    if min(counts) < 2:
        bad = {str(c): int(np.unique(groups[y == c]).size) for c in np.unique(y) if np.unique(groups[y == c]).size < 2}
        raise ValueError(
            "Spatial CV cannot assess class generalization because some classes occur in fewer than "
            f"two development blocks: {bad}. Add geographically separated reference data for those classes."
        )
    n_splits = min(requested, len(unique_groups), min(counts))
    if n_splits < 2:
        raise ValueError("At least two spatial CV folds are required.")
    return int(n_splits)


def make_spatial_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Return the largest feasible spatial CV with every class present in every fold."""
    max_splits = determine_cv_splits(y, groups, requested_splits)
    all_classes = set(np.unique(y).tolist())
    for n_splits in range(max_splits, 1, -1):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y, groups=groups))
        valid_design = True
        for train_idx, valid_idx in splits:
            if np.intersect1d(np.unique(groups[train_idx]), np.unique(groups[valid_idx])).size:
                raise RuntimeError("Spatial group overlap detected inside cross-validation.")
            if set(np.unique(y[train_idx]).tolist()) != all_classes:
                valid_design = False
                break
            if set(np.unique(y[valid_idx]).tolist()) != all_classes:
                valid_design = False
                break
        if valid_design:
            return splits, n_splits
    raise ValueError(
        "No spatial CV design from 2 to the requested number of folds contains every class in every "
        "training and validation fold. Increase the geographic spread of training data or reconsider block size."
    )

