"""Central configuration objects for the geospatial ML workflow.

The project keeps the user-facing variables in the root main script, while this
module provides validated dataclasses and scientifically conservative defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class PathConfig:
    project_root: Path
    imagery_path: Path
    training_path: Path
    aoi_path: Path | None
    outputs_dir: Path

    def ensure_output_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "prepared").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "rasters").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "models").mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class RasterPrepConfig:
    required_bands: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")
    optional_bands: tuple[str, ...] = ("B05", "B06", "B07", "B8A")
    target_resolution: float | None = 10.0
    reflectance_resampling: str = "bilinear"
    mask_resampling: str = "nearest"
    use_cloud_masks: bool = False
    cloud_mask_regex: str | None = None
    cloud_mask_mode: Literal["nonzero", "values"] = "nonzero"
    cloud_invalid_values: tuple[int, ...] = ()
    require_cloud_mask: bool = False
    parallel_preprocessing: bool = False
    max_workers: int | None = None
    prepared_stack_name: str = "sentinel2_feature_stack.tif"
    float_nodata: float = -9999.0


@dataclass(slots=True)
class TrainingConfig:
    label_column: str = "label"
    class_name_columns: tuple[str, ...] = ("Classe", "Name")
    max_pixels_per_geometry: int = 200
    random_state: int = 42
    require_projected_crs: bool = True


@dataclass(slots=True)
class SpatialValidationConfig:
    block_size: float = 1000.0
    holdout_fraction: float = 0.20
    random_state: int = 42
    requested_cv_splits: int = 5
    spatial_gap_distance: float = 0.0
    minimum_class_blocks_for_cv: int = 2


@dataclass(slots=True)
class ModelingConfig:
    random_state: int = 42
    primary_metric: Literal["balanced_accuracy", "macro_f1"] = "balanced_accuracy"
    randomized_search_iterations: int = 18
    n_jobs: int = -1
    include_svm: bool = True
    include_dummy: bool = True
    run_feature_experiments: bool = True
    feature_selector_threshold: str | float = "median"
    rf_search_space: dict = field(default_factory=lambda: {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [None, 10, 20, 30, 40],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__min_samples_split": [2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", 0.5, 0.8],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    })
    svm_search_space: dict = field(default_factory=lambda: {
        "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "classifier__kernel": ["rbf"],
    })


@dataclass(slots=True)
class InferenceConfig:
    integer_nodata: int = -9999
    float_nodata: float = -9999.0
    write_class_probability_scores: bool = True
    apply_majority_filter: bool = True
    majority_filter_size: int = 3


@dataclass(slots=True)
class ProjectConfig:
    paths: PathConfig
    raster: RasterPrepConfig = field(default_factory=RasterPrepConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    spatial: SpatialValidationConfig = field(default_factory=SpatialValidationConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
