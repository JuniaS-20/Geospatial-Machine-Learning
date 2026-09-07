

# Data

This directory contains the input geospatial data used by the
Sentinel-2 land-cover classification workflow.

## Directory structure

The expected structure is:

```text
data/
├── README.md
├── Tdata/
│   └── Training_data.*
├── ZOI/
│   └── Zone_Of_Interest.*
├── S2A_MSIL2A_20230127T081211_N0509_R078_T35LNG_20230127T120321.SAFE/
└── S2A_MSIL2A_20230706T080611_N0509_R078_T35LNH_20230706T120603.SAFE/```

#==========================================================*****************=============================
# Data layout

This repository intentionally does not version large imagery or training datasets.

Place Sentinel-2 scenes below `data/` (nested `.SAFE`/MUSCATE folders are supported),
put the study-area vector in `data/ZOI/`, and put `Training_data.shp` plus its sidecar
files in `data/Tdata/`.

The default training label field is `label`. The supplied project notes describe a
training table with `id`, `Classe`, `label`, and `Name`; the pipeline uses `label` as
the model target and uses `Classe` or `Name` as a display name when available.

Cloud-mask semantics are product-specific. Do not enable cloud masking until
`CLOUD_MASK_REGEX`, `CLOUD_MASK_MODE`, and (when needed) `CLOUD_INVALID_VALUES` are
verified against the metadata/documentation of the imagery product you are using.