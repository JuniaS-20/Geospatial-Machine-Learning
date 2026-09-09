 # Geospatial Machine Learning — Spatially Validated Sentinel-2 Land-Cover Classification

A reproducible geospatial machine-learning pipeline for Sentinel-2 land-cover
classification, with label-blind spatial hold-out validation, development-only spatial
cross-validation and hyperparameter tuning, feature experiments, model baselines,
window-based raster inference, uncertainty-aware diagnostic layers, automated tests,
and GIS-ready outputs.

## Why the validation design matters

Random pixel splits are often too optimistic for remote-sensing data because nearby
samples are spatially autocorrelated. This project therefore groups training samples
into spatial blocks. A fixed fraction of blocks is selected **once** for the final test
set using only block IDs and a random seed; class labels are not an input to that
selection. Feature experiments, model comparison and hyperparameter tuning use only
the remaining development blocks. The final spatial hold-out is evaluated once.

The code deliberately does **not** re-select a hold-out because some classes are absent
from it. Such absences are reported as a data-coverage limitation. If a class is absent
from development data, the run stops and the training design must be improved rather
than adapting the test set after seeing its labels.

## Models and feature experiments

The development workflow compares a `DummyClassifier` baseline with tuned Random
Forest and SVM models. SVM scaling occurs inside an sklearn `Pipeline`, preventing
pre-validation scaling leakage. Hyperparameters are tuned with `RandomizedSearchCV`
using the same spatial folds.

Three feature experiments are available:

- **A — raw spectral bands**: B02, B03, B04, B08, B11, B12 (plus optional red-edge bands if present).
- **B — bands + derived features**: NDVI, NDWI, NDBI, NDMI, NBR, supported red-edge indices and spectral ratios.
- **C — embedded selection**: feature selection is fitted inside each CV fold via `SelectFromModel`; it is never fitted once on the complete dataset before validation.

Red-edge indices are computed only if their required Sentinel-2 red-edge bands are
actually present.

## Imagery preparation

The preprocessing module recursively discovers scenes under `data/`, checks required
bands, aligns resolutions/CRS, optionally applies a product-specific cloud mask to each
scene **before** mosaicking, mosaics overlapping scenes, clips/masks to the study area,
and writes one multiband feature stack.

Cloud-mask semantics differ by processing chain. The pipeline therefore refuses to
guess them: cloud masking is disabled by default and must be explicitly configured from
the metadata/documentation of the imagery product.


> *Note:* The example dataset uses two adjacent Sentinel-2 Level-2A tiles
> acquired in January and July 2023. These scenes are provided for workflow
> demonstration purposes; users can run the pipeline with their own imagery
> and training data.


## Training data

The default path is `data/Tdata/Training_data.shp`, with `label` as target. The project
notes describe the fields `id`, `Classe`, `label`, and `Name`; `Classe`/`Name` are used
as human-readable names when available. Point, MultiPoint, Polygon and MultiPolygon
training geometries are supported. Polygon pixels are deterministically capped per
feature to reduce uncontrolled pseudo-replication.

The absolute number of training objects is not sufficient to judge adequacy. The
pipeline writes the number of samples and, more importantly, the number of independent
spatial blocks represented by each class. Geographic coverage per class should guide
additional sampling.

## Inference and uncertainty language

Full-raster prediction is performed with `rasterio` windows, so the complete feature
stack is not loaded into memory. Outputs include raw classification, maximum model
confidence, `1 - confidence`, normalized Shannon entropy and optional per-class
`predict_proba` scores.

These are **confidence / predictive-uncertainty proxies**. They are not described as
Bayesian uncertainty or calibrated probabilities. If probability calibration is added
in a future version, it should itself respect the spatial validation design.

## Post-processing

The old median filter has been replaced by a categorical **majority (mode) filter**.
Land-cover IDs are nominal classes, so a numeric median imposes an unjustified ordering
on class codes. Raw and post-processed maps are both retained and evaluated on the same
frozen hold-out. The hold-out result is not used to tune the filter size.

## Repository structure

```text
Geospatial-Machine-Learning/
├── Geospatial-Machine-Learning.py  # main executable
├── src/
│   ├── config.py
│   ├── raster_prep.py
│   ├── features.py
│   ├── training_data.py
│   ├── spatial_validation.py
│   ├── models.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── outputs.py
│   └── utils.py
├── tests/
├── .github/workflows/tests.yml
├── data/
├── outputs/
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest
python spatial_rf_landcover_classification_V2.py
```

Edit the **USER CONFIGURATION** section at the top of the main script. By default,
imagery is searched recursively under `data/`, the AOI is expected under `data/ZOI/`,
and `Training_data.shp` under `data/Tdata/`.

## Outputs

The `outputs/` directory contains prepared feature stacks, raw and post-processed land
cover rasters, confidence/entropy diagnostics, class score rasters, confusion matrices,
per-class metrics, feature/model comparison JSON, the fitted model, reproducibility
metadata, QGIS styling and a class legend.

## Methodological limits to report

Spatial block size is a scientific hyperparameter and should be justified from the
spatial autocorrelation scale, sensor resolution and intended transfer distance; 1 km
is only a configurable default. A random block hold-out ensures disjoint groups but is
not identical to transfer to a different region. For stronger geographic extrapolation
claims, use a pre-declared regional hold-out and/or a positive spatial gap. Training
reference quality, temporal mismatch, class imbalance, cloud/shadow treatment and
land-cover heterogeneity remain important sources of error.
