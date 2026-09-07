from pathlib import Path

from src.config import RasterPrepConfig
from src.raster_prep import discover_scenes


def test_safe_discovery_uses_img_data_and_native_resolution(tmp_path: Path):
    safe = tmp_path / "S2A_TEST.SAFE"
    granule = safe / "GRANULE" / "L2A_TEST"
    img = granule / "IMG_DATA"
    qi = granule / "QI_DATA"

    for resolution in ("R10m", "R20m", "R60m"):
        (img / resolution).mkdir(parents=True, exist_ok=True)
    qi.mkdir(parents=True, exist_ok=True)

    # Standard L2A spectral copies at multiple resolutions.
    for band in ("B02", "B03", "B04", "B08"):
        for res in (10, 20, 60):
            (img / f"R{res}m" / f"T_TEST_{band}_{res}m.jp2").touch()
    for band in ("B11", "B12"):
        for res in (20, 60):
            (img / f"R{res}m" / f"T_TEST_{band}_{res}m.jp2").touch()

    # Quality files deliberately contain a band token but must never be
    # interpreted as reflectance imagery.
    (qi / "MSK_DETFOO_B02.jp2").touch()
    (qi / "MSK_QUALIT_B02.jp2").touch()

    scenes = discover_scenes(tmp_path, RasterPrepConfig())
    assert len(scenes) == 1
    scene = next(iter(scenes.values()))

    assert "IMG_DATA" in str(scene["B02"])
    assert "QI_DATA" not in str(scene["B02"])
    assert "_10m.jp2" in scene["B02"].name
    assert "_20m.jp2" in scene["B11"].name
