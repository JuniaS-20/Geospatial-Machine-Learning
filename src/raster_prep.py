"""Sentinel-2 scene discovery, alignment, mosaicking, masking, clipping and feature stacking.

The preprocessing deliberately avoids guessing cloud-mask semantics. If cloud masks
are enabled, the user must configure how invalid mask values are interpreted for the
specific upstream product (e.g. MAJA/MUSCATE, Sen2Cor SCL, custom binary mask).
"""
from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
import xml.etree.ElementTree as ET
import os

os.environ.pop("PROJ_LIB", None)
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .config import PathConfig, RasterPrepConfig
from .features import compute_spectral_indices
from .utils import check_projected_metric_crs

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jp2"}


def _band_regex(band: str) -> re.Pattern:
    number = band.replace("B", "")
    if band == "B8A":
        token = r"B8A"
    else:
        token = rf"B0?{int(number)}"
    # Supports MAJA-style *_FRE_B2.tif and standard Sentinel names *_B02_10m.jp2.
    return re.compile(rf"(?:^|[_-]){token}(?:[_\. -]|$)", re.IGNORECASE)


def _candidate_resolution(path: Path) -> float | None:
    """Infer a Sentinel-2 spatial resolution from a filename or parent folder.

    Standard L2A SAFE products encode resolution either in the filename
    (e.g. ``*_B02_10m.jp2``) and/or in a parent directory (``R10m``).
    Returns ``None`` when no unambiguous resolution token is available.
    """
    text = " ".join([path.name, *[part for part in path.parts[-4:]]])
    match = re.search(r"(?:^|[_\\/.-])R?(10|20|60)m(?:[_\\/.-]|$)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _select_band_candidate(candidates: list[Path], band: str, target_resolution: float | None) -> Path:
    """Select one physical Sentinel-2 band file deterministically.

    SAFE L2A products contain resampled copies of several bands at 10, 20 and
    60 m. We prefer the band's native resolution to avoid an unnecessary
    resampling step before the common-grid warp. If native resolution cannot be
    inferred, the candidate nearest to ``target_resolution`` is preferred.
    """
    if not candidates:
        raise ValueError(f"No candidate supplied for {band}.")
    if len(candidates) == 1:
        return candidates[0]

    native_resolution = {
        "B02": 10.0, "B03": 10.0, "B04": 10.0, "B08": 10.0,
        "B05": 20.0, "B06": 20.0, "B07": 20.0, "B8A": 20.0,
        "B11": 20.0, "B12": 20.0,
    }.get(band)

    ranked: list[tuple[tuple[float, float, str], Path]] = []
    for path in candidates:
        resolution = _candidate_resolution(path)
        native_penalty = (
            abs(resolution - native_resolution)
            if resolution is not None and native_resolution is not None
            else float("inf")
        )
        target_penalty = (
            abs(resolution - target_resolution)
            if resolution is not None and target_resolution is not None
            else float("inf")
        )
        ranked.append(((native_penalty, target_penalty, str(path).lower()), path))

    ranked.sort(key=lambda item: item[0])
    best_key, best_path = ranked[0]

    # If resolution information is unavailable for all candidates, do not guess.
    if best_key[0] == float("inf") and best_key[1] == float("inf"):
        names = "\n  - ".join(str(p) for p in sorted(candidates))
        raise ValueError(
            f"Ambiguous spectral files for {band}; resolution cannot be inferred:\n  - {names}\n"
            "Tighten the imagery naming pattern or separate the scenes."
        )
    return best_path


def _safe_granule_dirs(safe_dir: Path) -> list[Path]:
    """Return Sentinel-2 L2A granule directories that contain IMG_DATA."""
    granule_root = safe_dir / "GRANULE"
    if not granule_root.exists():
        return []
    return sorted(
        granule for granule in granule_root.iterdir()
        if granule.is_dir() and (granule / "IMG_DATA").exists()
    )


def discover_scenes(imagery_path: Path, cfg: RasterPrepConfig) -> dict[str, dict[str, Path]]:
    """Discover spectral imagery while explicitly excluding SAFE quality layers.

    For Sentinel-2 ``.SAFE`` products, only files below ``GRANULE/*/IMG_DATA``
    are considered spectral reflectance bands. ``QI_DATA`` and ``AUX_DATA`` are
    intentionally excluded because their filenames can also contain tokens such
    as ``B02`` (for example ``MSK_DETFOO_B02.jp2``), but those files are masks or
    ancillary data rather than reflectance imagery.

    Each SAFE granule is treated as one spatial scene. This also makes future
    mask discovery naturally granule-specific.
    """
    if not imagery_path.exists():
        raise FileNotFoundError(
            f"Imagery directory does not exist: {imagery_path}. Place imagery under data/ or set IMAG_PATH."
        )

    band_regexes = {b: _band_regex(b) for b in (*cfg.required_bands, *cfg.optional_bands)}
    scenes: dict[str, dict[str, Path]] = {}

    safe_dirs = sorted(p for p in imagery_path.iterdir() if p.is_dir() and p.name.upper().endswith(".SAFE"))
    for safe_dir in safe_dirs:
        granules = _safe_granule_dirs(safe_dir)
        if not granules:
            raise FileNotFoundError(
                f"Sentinel-2 SAFE product contains no GRANULE/*/IMG_DATA directory: {safe_dir}"
            )

        for granule_dir in granules:
            img_data = granule_dir / "IMG_DATA"
            spectral_files = [
                p for p in img_data.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            scene_id = str(granule_dir.relative_to(imagery_path))
            band_map: dict[str, Path] = {}

            for band, rx in band_regexes.items():
                candidates = [p for p in spectral_files if rx.search(p.name)]
                if candidates:
                    band_map[band] = _select_band_candidate(candidates, band, cfg.target_resolution)

            if band_map:
                scenes[scene_id] = band_map

    # Generic fallback for non-SAFE inputs (e.g. MAJA/MUSCATE GeoTIFF directories).
    safe_roots = {safe.resolve() for safe in safe_dirs}
    generic_files = []
    for path in imagery_path.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        if any(safe_root in path.resolve().parents for safe_root in safe_roots):
            continue
        # Never interpret obvious quality/auxiliary directories as reflectance data.
        if any(part.upper() in {"QI_DATA", "AUX_DATA", "DATASTRIP"} for part in path.parts):
            continue
        generic_files.append(path)

    generic_candidates: dict[str, dict[str, list[Path]]] = {}
    for path in generic_files:
        matched = [band for band, rx in band_regexes.items() if rx.search(path.name)]
        if not matched:
            continue
        rel = path.relative_to(imagery_path)
        scene_id = rel.parts[0] if len(rel.parts) > 1 else "."
        per_scene = generic_candidates.setdefault(scene_id, {})
        for band in matched:
            per_scene.setdefault(band, []).append(path)

    for scene_id, per_band in generic_candidates.items():
        band_map = {
            band: _select_band_candidate(candidates, band, cfg.target_resolution)
            for band, candidates in per_band.items()
        }
        if scene_id in scenes:
            raise ValueError(f"Duplicate scene identifier during discovery: {scene_id}")
        scenes[scene_id] = band_map

    complete = {sid: d for sid, d in scenes.items() if all(b in d for b in cfg.required_bands)}
    if not complete:
        details = {sid: sorted(d) for sid, d in scenes.items()}
        raise FileNotFoundError(
            "No scene contains all required bands "
            f"{cfg.required_bands}. Discovered partial scenes: {details}"
        )
    return complete


def _find_scene_mask(scene_dir: Path, cfg: RasterPrepConfig) -> Path | None:
    if not cfg.use_cloud_masks:
        return None
    if not cfg.cloud_mask_regex:
        raise ValueError(
            "USE_CLOUD_MASKS=True requires CLOUD_MASK_REGEX. Mask semantics must be explicit; the pipeline "
            "does not guess product-specific cloud-mask conventions."
        )
    rx = re.compile(cfg.cloud_mask_regex, re.IGNORECASE)
    candidates = [p for p in scene_dir.rglob("*") if p.is_file() and rx.search(p.name)]
    if len(candidates) > 1:
        raise ValueError(f"Multiple cloud masks match scene {scene_dir}: {candidates}")
    if not candidates:
        if cfg.require_cloud_mask:
            raise FileNotFoundError(f"No cloud mask found for scene {scene_dir}")
        return None
    return candidates[0]


def _invalid_mask_values(mask_data: np.ndarray, cfg: RasterPrepConfig) -> np.ndarray:
    if cfg.cloud_mask_mode == "nonzero":
        return mask_data != 0
    if not cfg.cloud_invalid_values:
        raise ValueError("cloud_mask_mode='values' requires CLOUD_INVALID_VALUES.")
    return np.isin(mask_data, np.asarray(cfg.cloud_invalid_values))


def _resampling(name: str) -> Resampling:
    try:
        return getattr(Resampling, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported resampling method: {name}") from exc

def _crs_from_safe_metadata(raster_path: Path) -> CRS | None:
    """
    Recover the CRS of a Sentinel-2 SAFE granule from MTD_TL.xml.

    Some GDAL/Rasterio installations correctly read the JP2 affine transform
    but return ``src.crs = None``. Sentinel-2 tile metadata explicitly stores
    the CRS in ``HORIZONTAL_CS_CODE`` (for example ``EPSG:32735``).

    Parameters
    ----------
    raster_path : pathlib.Path
        Path to a Sentinel-2 spectral raster, typically located below
        GRANULE/<granule>/IMG_DATA/R10m, R20m or R60m.

    Returns
    -------
    rasterio.crs.CRS or None
        CRS recovered from MTD_TL.xml, or None if no suitable metadata file
        could be found.
    """
    raster_path = Path(raster_path)

    # Walk upwards from the raster until the granule-level MTD_TL.xml is found.
    for parent in raster_path.parents:
        metadata_path = parent / "MTD_TL.xml"

        if not metadata_path.exists():
            continue

        try:
            root = ET.parse(metadata_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(
                f"Cannot parse Sentinel-2 metadata file: {metadata_path}"
            ) from exc

        # Sentinel XML uses namespaces. Comparing only the local tag name
        # makes this work independently of the exact namespace version.
        for element in root.iter():
            tag_name = element.tag.split("}")[-1]

            if tag_name == "HORIZONTAL_CS_CODE":
                crs_text = (element.text or "").strip()

                if crs_text:
                    return CRS.from_string(crs_text)

    return None


def _get_raster_crs(raster_path: Path) -> CRS:
    """
    Return the CRS of a raster, with a Sentinel-2 SAFE metadata fallback.

    Rasterio is used first. If the JP2 exposes no CRS, the function attempts
    to recover it from the granule's MTD_TL.xml.
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        crs = src.crs

    if crs is None:
        crs = _crs_from_safe_metadata(raster_path)

    if crs is None:
        raise ValueError(
            "Unable to determine raster CRS from either Rasterio/GDAL "
            f"or Sentinel-2 MTD_TL.xml: {raster_path}"
        )

    return crs


def _reference_grid(
    scenes: dict[str, dict[str, Path]],
    cfg: RasterPrepConfig,
    aoi_path: Path | None,
):
    """Determine the common CRS, resolution and processing extent."""

    first_scene = next(iter(scenes.values()))
    first_path = first_scene[cfg.required_bands[0]]

    # Rasterio/GDAL may return src.crs=None for valid Sentinel-2 JP2 files.
    # _get_raster_crs() therefore falls back to the SAFE MTD_TL.xml metadata.
    target_crs = _get_raster_crs(first_path)

    check_projected_metric_crs(target_crs)

    with rasterio.open(first_path) as src:
        default_res = min(abs(src.res[0]), abs(src.res[1]))

    target_res = cfg.target_resolution or default_res

    if aoi_path is not None:
        if not aoi_path.exists():
            raise FileNotFoundError(f"AOI file not found: {aoi_path}")

        aoi = gpd.read_file(aoi_path)

        if aoi.empty or aoi.crs is None:
            raise ValueError("AOI is empty or has no CRS.")

        aoi = aoi.to_crs(target_crs)
        bounds = tuple(aoi.total_bounds)

    else:
        all_bounds = []

        for scene in scenes.values():
            scene_path = scene[cfg.required_bands[0]]
            scene_crs = _get_raster_crs(scene_path)

            with rasterio.open(scene_path) as src:
                all_bounds.append(
                    transform_bounds(
                        scene_crs,
                        target_crs,
                        *src.bounds,
                        densify_pts=21,
                    )
                )

        bounds = (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        )

    return target_crs, float(target_res), bounds


def _make_masked_scene_band(
    band_path: Path,
    mask_path: Path | None,
    dst_path: Path,
    target_crs,
    target_res: float,
    bounds: tuple[float, float, float, float],
    cfg: RasterPrepConfig,
) -> Path:
    """Warp a scene band to the common grid and apply that scene's mask before mosaicking."""
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(band_path))

        # GDAL may expose the JP2 transform but not its CRS.
        # Recover the CRS from SAFE metadata when necessary.
        source_crs = src.crs or _crs_from_safe_metadata(band_path)

        if source_crs is None:
            raise ValueError(
                "Unable to determine source CRS for Sentinel-2 band: "
                f"{band_path}"
            )

        src_bounds = transform_bounds(
            source_crs,
            target_crs,
            *src.bounds,
            densify_pts=21,
        )

        scene_bounds = (
            max(bounds[0], src_bounds[0]),
            max(bounds[1], src_bounds[1]),
            min(bounds[2], src_bounds[2]),
            min(bounds[3], src_bounds[3]),
        )

        if scene_bounds[0] >= scene_bounds[2] or scene_bounds[1] >= scene_bounds[3]:
            raise ValueError(
                "Scene band does not intersect the requested AOI/mosaic extent: "
                f"{band_path}"
            )

        vrt = stack.enter_context(
            WarpedVRT(
                src,
                src_crs=source_crs,
                crs=target_crs,
                resampling=_resampling(cfg.reflectance_resampling),
            )
        )
        if scene_bounds[0] >= scene_bounds[2] or scene_bounds[1] >= scene_bounds[3]:
            raise ValueError(f"Scene band does not intersect the requested AOI/mosaic extent: {band_path}")
        vrt = stack.enter_context(WarpedVRT(src, crs=target_crs, resampling=_resampling(cfg.reflectance_resampling)))
        arr, transform = merge(
            [vrt], bounds=scene_bounds, res=target_res, nodata=cfg.float_nodata,
            dtype="float32", resampling=_resampling(cfg.reflectance_resampling),
            target_aligned_pixels=True, masked=True,
        )
        data = np.asarray(arr[0].filled(cfg.float_nodata), dtype=np.float32)
        invalid = np.ma.getmaskarray(arr[0]) | ~np.isfinite(data) | (data == cfg.float_nodata)

#       if mask_path is not None:
#            msrc = stack.enter_context(rasterio.open(mask_path))
#            mvrt = stack.enter_context(WarpedVRT(msrc, crs=target_crs, resampling=_resampling(cfg.mask_resampling)))

        if mask_path is not None:
            msrc = stack.enter_context(rasterio.open(mask_path))

            mask_crs = msrc.crs or _crs_from_safe_metadata(mask_path)

            if mask_crs is None:
                raise ValueError(
                    f"Unable to determine CRS for mask raster: {mask_path}"
                )

            mvrt = stack.enter_context(
                WarpedVRT(
                    msrc,
                    src_crs=mask_crs,
                    crs=target_crs,
                    resampling=_resampling(cfg.mask_resampling),
                )
            )
            marr, _ = merge(
                [mvrt], bounds=scene_bounds, res=target_res, nodata=0,
                resampling=_resampling(cfg.mask_resampling), target_aligned_pixels=True, masked=False,
            )
            invalid |= _invalid_mask_values(np.asarray(marr[0]), cfg)

        data[invalid] = cfg.float_nodata
        profile = {
            "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
            "dtype": "float32", "crs": target_crs, "transform": transform, "nodata": cfg.float_nodata,
            "compress": "deflate", "tiled": True,
        }
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)
    return dst_path


def _mosaic_band(scene_band_paths: list[Path], dst_path: Path, cfg: RasterPrepConfig) -> Path:
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in scene_band_paths]
        merge(
            sources, nodata=cfg.float_nodata, dtype="float32", method="first",
            masked=False, dst_path=dst_path,
            dst_kwds={"compress": "deflate", "tiled": True, "nodata": cfg.float_nodata},
        )
    return dst_path


def prepare_feature_stack(paths: PathConfig, cfg: RasterPrepConfig) -> tuple[Path, list[str], list[str]]:
    """Create one aligned feature stack from one or more scenes.

    Cloud masking is applied per scene *before* mosaicking so an overlapping clear
    observation can replace a masked cloudy one. This is more defensible than
    mosaicking raw reflectance first and applying a global cloud mask afterwards.
    """
    prepared_dir = paths.outputs_dir / "prepared"
    tmp_dir = prepared_dir / "_tmp_scene_bands"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(paths.imagery_path, cfg)
    target_crs, target_res, bounds = _reference_grid(scenes, cfg, paths.aoi_path)
    # Required bands exist in every retained scene by construction. Optional bands
    # are included only if present in every scene; otherwise they would create
    # artificial NoData holes and change the spatial support of the experiment.
    optional_complete = [b for b in cfg.optional_bands if all(b in scene for scene in scenes.values())]
    all_available = [*cfg.required_bands, *optional_complete]

    tasks = []
    scene_outputs: dict[str, list[Path]] = {b: [] for b in all_available}
    workers = cfg.max_workers if cfg.parallel_preprocessing else 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for scene_id, band_map in scenes.items():
            scene_dir = paths.imagery_path / scene_id if scene_id != "." else paths.imagery_path
            mask_path = _find_scene_mask(scene_dir, cfg)
            for band in all_available:
                if band not in band_map:
                    continue
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id)
                dst = tmp_dir / f"{safe_id}_{band}.tif"
                fut = executor.submit(
                    _make_masked_scene_band, band_map[band], mask_path, dst,
                    target_crs, target_res, bounds, cfg,
                )
                tasks.append((fut, band))
        for fut, band in tasks:
            scene_outputs[band].append(fut.result())

    mosaic_paths: dict[str, Path] = {}
    for band in all_available:
        if not scene_outputs[band]:
            continue
        dst = prepared_dir / f"mosaic_{band}.tif"
        mosaic_paths[band] = _mosaic_band(scene_outputs[band], dst, cfg)

    # Build final feature stack, preserving a common spatial mask and AOI geometry.
    raw_names = [b for b in all_available if b in mosaic_paths]
    arrays: dict[str, np.ndarray] = {}
    profile = None
    for band in raw_names:
        with rasterio.open(mosaic_paths[band]) as src:
            arr = src.read(1).astype(np.float32)
            arrays[band] = arr
            if profile is None:
                profile = src.profile.copy()
                transform = src.transform
                shape = (src.height, src.width)

    if profile is None:
        raise RuntimeError("No mosaicked bands were produced.")

    common_valid = np.ones(shape, dtype=bool)
    for arr in arrays.values():
        common_valid &= np.isfinite(arr) & (arr != cfg.float_nodata)

    if paths.aoi_path is not None:
        aoi = gpd.read_file(paths.aoi_path).to_crs(target_crs)
        inside = geometry_mask(aoi.geometry, out_shape=shape, transform=transform, invert=True)
        common_valid &= inside

    indices = compute_spectral_indices({k: np.where(common_valid, v, np.nan) for k, v in arrays.items()})
    for arr in arrays.values():
        arr[~common_valid] = cfg.float_nodata
    for name, arr in indices.items():
        bad = ~np.isfinite(arr) | ~common_valid
        arr = arr.astype(np.float32)
        arr[bad] = cfg.float_nodata
        indices[name] = arr

    feature_names = raw_names + list(indices.keys())
    stack_path = prepared_dir / cfg.prepared_stack_name
    profile.update(count=len(feature_names), dtype="float32", nodata=cfg.float_nodata, compress="deflate", tiled=True)
    with rasterio.open(stack_path, "w", **profile) as dst:
        for i, name in enumerate(feature_names, start=1):
            dst.write((arrays.get(name) if name in arrays else indices[name]).astype(np.float32), i)
            dst.set_band_description(i, name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return stack_path, feature_names, raw_names


def inspect_existing_stack(path: Path) -> tuple[list[str], list[str]]:
    """Use an already prepared multiband raster while preserving feature names where available."""
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        names = [d if d else f"band_{i}" for i, d in enumerate(src.descriptions, start=1)]
        if not names:
            names = [f"band_{i}" for i in range(1, src.count + 1)]
    # Existing stacks cannot reliably identify raw-vs-derived bands without descriptions.
    raw = [n for n in names if re.fullmatch(r"B(?:0?[1-9]|1[0-2]|8A)", n, re.IGNORECASE)] or names.copy()
    return names, raw
