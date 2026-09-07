"""Geospatial Machine Learning — Spatially Validated Sentinel-2 Land-Cover Classification.

Main executable script. All reusable logic lives in ``src/``.

Scientific design highlights
----------------------------
1. Label-blind spatial hold-out: test blocks are selected once, reproducibly,
   without inspecting class labels.
2. Development-only spatial CV for feature experiments, model comparison and
   hyperparameter tuning.
3. Dummy baseline + tuned Random Forest + tuned SVM comparison.
4. Raw bands vs bands+indices vs leakage-safe embedded feature selection.
5. Window-based raster inference; the full feature raster is not loaded into RAM.
6. Confidence and entropy-based uncertainty *proxies* only. The workflow does
   not claim Bayesian or calibrated uncertainty.
7. Optional categorical majority-filter post-processing. Median filtering is
   deliberately avoided because class codes are nominal, not ordinal.
8. Point and polygon training data support, explicit NoData handling, tests and CI.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.config import (
    InferenceConfig,
    ModelingConfig,
    PathConfig,
    ProjectConfig,
    RasterPrepConfig,
    SpatialValidationConfig,
    TrainingConfig,
)
from src.evaluation import (
    classification_metrics,
    save_confusion_matrix_figure,
    save_per_class_table,
)
from src.features import FeatureDefinition
from src.inference import (
    classify_feature_stack,
    majority_filter_classification,
    sample_prediction_raster,
)
from src.models import tune_and_select_model
from src.outputs import create_legend_json, create_qml, save_metadata, save_model
from src.raster_prep import inspect_existing_stack, prepare_feature_stack
from src.spatial_validation import (
    ensure_development_class_support,
    holdout_class_coverage,
    class_block_report,
    create_label_blind_holdout,
    create_spatial_blocks,
    make_spatial_cv_splits,
)
from src.training_data import extract_training_dataset
from src.utils import save_json


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Edit only this section for a normal run. Robust defaults remain in src/config.py.

PROJECT_ROOT = Path(__file__).resolve().parent

# Default repository layout requested for the project.
IMAG_PATH = PROJECT_ROOT / "data"
ZOI_PATH = PROJECT_ROOT / "data" / "ZOI" / "zone_of_interest.shp"
TDATA_PATH = PROJECT_ROOT / "data" / "Tdata" / "Training_data.shp"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# If you already have a scientifically prepared, co-registered multiband stack,
# set this to its path and set PREPARE_FROM_SCENES=False.
PREPARE_FROM_SCENES = True
PREPARED_STACK_PATH: Path | None = None

# Sentinel-2 raw bands required by the default feature study.
REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
# Optional red-edge bands are used only when physically present; no feature is fabricated.
OPTIONAL_BANDS = ("B05", "B06", "B07", "B8A")
TARGET_RESOLUTION = 10.0  # metres; B11/B12 are resampled to the common grid.

# Cloud masks are product-specific. Leave False unless you know the mask semantics.
USE_CLOUD_MASKS = False
# Example only for a MAJA-like filename; verify against your product documentation.
CLOUD_MASK_REGEX = r".*_CLM_R[12]\.(tif|tiff)$"
CLOUD_MASK_MODE = "nonzero"      # "nonzero" or "values"
CLOUD_INVALID_VALUES: tuple[int, ...] = ()
REQUIRE_CLOUD_MASK = False
PARALLEL_PREPROCESSING = False

# Training_data.shp structure in your document includes id / Classe / label / Name.
LABEL_COLUMN = "label"
MAX_PIXELS_PER_GEOMETRY = 200  # polygon cap; points contribute one pixel each.

# Spatial design. Test blocks are sampled *without y* and then frozen.
RANDOM_STATE = 42
SPATIAL_BLOCK_SIZE = 1000.0     # coordinate units; projected CRS required.
HOLDOUT_FRACTION = 0.20
SPATIAL_GAP_DISTANCE = 0.0      # e.g. 1000.0 to exclude nearby dev samples label-blindly.
INNER_N_SPLITS = 5

# Model study and tuning.
PRIMARY_METRIC = "balanced_accuracy"  # or "macro_f1"
RANDOMIZED_SEARCH_ITERATIONS = 18
INCLUDE_SVM = True
RUN_FEATURE_EXPERIMENTS = True

# Raster outputs.
FLOAT_NODATA = -9999.0
INTEGER_NODATA = -9999
WRITE_CLASS_PROBABILITY_SCORES = True
APPLY_MAJORITY_FILTER = True
MAJORITY_FILTER_SIZE = 3


def build_config() -> ProjectConfig:
    aoi = ZOI_PATH if ZOI_PATH.exists() else None
    return ProjectConfig(
        paths=PathConfig(
            project_root=PROJECT_ROOT,
            imagery_path=IMAG_PATH,
            training_path=TDATA_PATH,
            aoi_path=aoi,
            outputs_dir=OUTPUTS_DIR,
        ),
        raster=RasterPrepConfig(
            required_bands=REQUIRED_BANDS,
            optional_bands=OPTIONAL_BANDS,
            target_resolution=TARGET_RESOLUTION,
            use_cloud_masks=USE_CLOUD_MASKS,
            cloud_mask_regex=CLOUD_MASK_REGEX,
            cloud_mask_mode=CLOUD_MASK_MODE,
            cloud_invalid_values=CLOUD_INVALID_VALUES,
            require_cloud_mask=REQUIRE_CLOUD_MASK,
            parallel_preprocessing=PARALLEL_PREPROCESSING,
            float_nodata=FLOAT_NODATA,
        ),
        training=TrainingConfig(
            label_column=LABEL_COLUMN,
            max_pixels_per_geometry=MAX_PIXELS_PER_GEOMETRY,
            random_state=RANDOM_STATE,
        ),
        spatial=SpatialValidationConfig(
            block_size=SPATIAL_BLOCK_SIZE,
            holdout_fraction=HOLDOUT_FRACTION,
            random_state=RANDOM_STATE,
            requested_cv_splits=INNER_N_SPLITS,
            spatial_gap_distance=SPATIAL_GAP_DISTANCE,
        ),
        modeling=ModelingConfig(
            random_state=RANDOM_STATE,
            primary_metric=PRIMARY_METRIC,
            randomized_search_iterations=RANDOMIZED_SEARCH_ITERATIONS,
            include_svm=INCLUDE_SVM,
            run_feature_experiments=RUN_FEATURE_EXPERIMENTS,
        ),
        inference=InferenceConfig(
            integer_nodata=INTEGER_NODATA,
            float_nodata=FLOAT_NODATA,
            write_class_probability_scores=WRITE_CLASS_PROBABILITY_SCORES,
            apply_majority_filter=APPLY_MAJORITY_FILTER,
            majority_filter_size=MAJORITY_FILTER_SIZE,
        ),
    )


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    started = time.time()
    cfg = build_config()
    cfg.paths.ensure_output_dirs()

    if not cfg.paths.training_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {cfg.paths.training_path}\n"
            "Place Training_data.shp (+ .dbf/.shx/.prj) in data/Tdata/ or edit TDATA_PATH."
        )

    _section("1. PREPARE / LOAD FEATURE STACK")
    if PREPARE_FROM_SCENES:
        feature_stack, feature_names, raw_names = prepare_feature_stack(cfg.paths, cfg.raster)
    else:
        if PREPARED_STACK_PATH is None:
            raise ValueError("PREPARED_STACK_PATH must be set when PREPARE_FROM_SCENES=False.")
        feature_stack = PREPARED_STACK_PATH
        feature_names, raw_names = inspect_existing_stack(feature_stack)
    print(f"Feature stack: {feature_stack}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    _section("2. EXTRACT TRAINING DATA")

    data = extract_training_dataset(
        str(feature_stack),
        str(cfg.paths.training_path),
        feature_names,
        cfg.training,
    )

    class_names = data.class_names
    print(f"Source training objects: {data.extraction_report['source_objects']}")
    print(f"Valid sampled pixels: {len(data.y)}")
    print(f"Classes: {sorted(data.class_names.items())}")

    _section("3. CREATE SPATIAL BLOCKS AND LABEL-BLIND HOLD-OUT")
    groups = create_spatial_blocks(data.coordinates, cfg.spatial.block_size)
    pre_split_report = class_block_report(data.y, groups)
    split = create_label_blind_holdout(
        groups=groups,
        test_fraction=cfg.spatial.holdout_fraction,
        random_state=cfg.spatial.random_state,
        coordinates=data.coordinates,
        spatial_gap_distance=cfg.spatial.spatial_gap_distance,
    )
    print(f"Development blocks: {len(split.development_groups)}")
    print(f"Test blocks: {len(split.test_groups)}")
    print(f"Gap-excluded samples: {len(split.excluded_gap_idx)}")

    X_dev = data.X[split.development_idx]
    y_dev = data.y[split.development_idx]
    groups_dev = groups[split.development_idx]
    X_test = data.X[split.test_idx]
    y_test = data.y[split.test_idx]
    coords_test = data.coordinates[split.test_idx]
    ensure_development_class_support(sorted(data.class_names), y_dev)

    _section("4. DEVELOPMENT-ONLY SPATIAL CROSS-VALIDATION")
    cv_splits, n_splits = make_spatial_cv_splits(
        X_dev, y_dev, groups_dev, cfg.spatial.requested_cv_splits, cfg.spatial.random_state
    )
    print(f"Spatial CV folds: {n_splits}")
    print("No hold-out labels or hold-out samples are used for tuning/model selection.")

    _section("5. FEATURE STUDY, BASELINES, TUNING AND MODEL SELECTION")
    feature_def = FeatureDefinition(tuple(feature_names), tuple(raw_names))
    selection = tune_and_select_model(X_dev, y_dev, cv_splits, feature_def, cfg.modeling)
    print(f"Selected feature experiment: {selection.feature_experiment}")
    print(f"Selected model: {selection.model_name}")
    print(f"Selected/retained features: {selection.selected_feature_names}")
    print(f"Best parameters: {selection.best_params}")

    _section("6. ONE-TIME INDEPENDENT SPATIAL HOLD-OUT EVALUATION")
    raw_test_pred = selection.estimator.predict(X_test)
    class_labels = sorted(data.class_names)
    holdout_audit = holdout_class_coverage(class_labels, y_test)
    if holdout_audit["classes_absent_from_test"]:
        print(
            "WARNING: class(es) absent from the frozen spatial test set: "
            f"{holdout_audit['classes_absent_from_test']}. Their spatial test performance cannot be estimated."
        )
    raw_metrics = classification_metrics(y_test, raw_test_pred, class_labels)
    print(
        f"Raw hold-out — Balanced Accuracy: {raw_metrics['balanced_accuracy']:.4f}; "
        f"Macro F1: {raw_metrics['macro_f1']:.4f}; Accuracy: {raw_metrics['accuracy']:.4f}"
    )
    metrics_dir = cfg.paths.outputs_dir / "metrics"
    figures_dir = cfg.paths.outputs_dir / "figures"
    save_json(raw_metrics, metrics_dir / "holdout_raw_metrics.json")
    save_per_class_table(raw_metrics, data.class_names, metrics_dir / "holdout_raw_per_class.csv")
    save_confusion_matrix_figure(
        raw_metrics,
        figures_dir / "holdout_raw_confusion_matrix.png",
        normalized=False,
        class_names=class_names,
    )

    save_confusion_matrix_figure(
        raw_metrics,
        figures_dir / "holdout_raw_confusion_matrix_normalized.png",
        normalized=True,
        class_names=class_names,
    )

    _section("7. WINDOW-BASED FULL-RASTER INFERENCE")
    raster_outputs = classify_feature_stack(
        selection.estimator, feature_stack, cfg.paths.outputs_dir / "rasters", cfg.inference
    )
    for name, path in raster_outputs.items():
        print(f"{name}: {path}")

    postprocess_metrics = None
    if cfg.inference.apply_majority_filter:
        _section("8. CATEGORICAL MAJORITY-FILTER POST-PROCESSING + HOLD-OUT COMPARISON")
        filtered_path = cfg.paths.outputs_dir / "rasters" / "landcover_majority_filtered.tif"
        majority_filter_classification(
            raster_outputs["classification_raw"], filtered_path,
            cfg.inference.majority_filter_size, cfg.inference.integer_nodata,
        )
        filtered_pred = sample_prediction_raster(filtered_path, coords_test)
        valid = filtered_pred != cfg.inference.integer_nodata
        if not valid.all():
            print(f"WARNING: {np.sum(~valid)} hold-out samples map to NoData in the filtered raster.")
        postprocess_metrics = classification_metrics(y_test[valid], filtered_pred[valid], class_labels)
        save_json(postprocess_metrics, metrics_dir / "holdout_majority_filter_metrics.json")
        save_per_class_table(
            postprocess_metrics, data.class_names, metrics_dir / "holdout_majority_filter_per_class.csv"
        )
        print(
            f"Majority-filter hold-out — Balanced Accuracy: {postprocess_metrics['balanced_accuracy']:.4f}; "
            f"Macro F1: {postprocess_metrics['macro_f1']:.4f}; Accuracy: {postprocess_metrics['accuracy']:.4f}"
        )
        print("Both raw and post-processed results are retained; the test set is not used to tune the filter.")
        raster_outputs["classification_majority_filtered"] = filtered_path

    _section("9. SAVE MODEL, METADATA AND GIS SUPPORT FILES")
    model_path = save_model(selection.estimator, cfg.paths.outputs_dir / "models" / "selected_model.joblib")
    metadata = {
        "methodological_contract": {
            "holdout_selection": "label-blind random selection of spatial blocks with fixed random_state",
            "holdout_fraction": cfg.spatial.holdout_fraction,
            "spatial_block_size": cfg.spatial.block_size,
            "spatial_gap_distance": cfg.spatial.spatial_gap_distance,
            "holdout_used_for_tuning": False,
            "primary_metric": cfg.modeling.primary_metric,
            "uncertainty_statement": (
                "predict_proba-derived confidence and normalized Shannon entropy are predictive uncertainty "
                "proxies; they are not claimed to be Bayesian or calibrated uncertainty."
            ),
            "postprocessing_statement": (
                "Categorical majority filtering is evaluated separately on the frozen spatial hold-out and is "
                "not used for model selection."
            ),
        },
        "training_extraction": data.extraction_report,
        "class_names": data.class_names,
        "class_block_report_before_split": pre_split_report,
        "holdout_audit": holdout_audit,
        "development_samples": int(len(X_dev)),
        "test_samples": int(len(X_test)),
        "cv_folds": n_splits,
        "feature_names_all": feature_names,
        "raw_band_names": raw_names,
        "feature_experiment_selected": selection.feature_experiment,
        "selected_feature_names": selection.selected_feature_names,
        "feature_experiment_results": selection.feature_results,
        "model_comparison_results": selection.model_results,
        "selected_model": selection.model_name,
        "best_params": selection.best_params,
        "holdout_raw": raw_metrics,
        "holdout_postprocessed": postprocess_metrics,
        "raster_outputs": {k: str(v) for k, v in raster_outputs.items()},
        "feature_stack": str(feature_stack),
        "runtime_seconds": round(time.time() - started, 2),
    }
    metadata_path = save_metadata(metadata, cfg.paths.outputs_dir / "models" / "model_metadata.json")
    create_qml(data.class_names, cfg.paths.outputs_dir / "rasters" / "landcover.qml")
    create_legend_json(data.class_names, cfg.paths.outputs_dir / "rasters" / "legend.json")
    save_json(selection.feature_results, metrics_dir / "feature_experiments.json")
    save_json(selection.model_results, metrics_dir / "model_comparison.json")

    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Total runtime: {time.time() - started:.1f} s")
    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
