"""Model persistence, metadata and GIS symbology outputs."""
from __future__ import annotations

import colorsys
from pathlib import Path

import joblib

from .utils import environment_metadata, save_json


def save_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_metadata(metadata: dict, path: Path) -> Path:
    payload = {**metadata, "environment": environment_metadata()}
    save_json(payload, path)
    return path


def _class_color(index: int, n: int) -> tuple[int, int, int]:
    h = (index / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.90)
    return int(r * 255), int(g * 255), int(b * 255)


def create_qml(class_names: dict[int, str], output_path: Path, nodata: int = -9999) -> Path:
    items = []
    for i, (label, name) in enumerate(sorted(class_names.items())):
        r, g, b = _class_color(i, len(class_names))
        items.append(
            f'<paletteEntry alpha="255" color="#{r:02x}{g:02x}{b:02x}" value="{label}" label="{name}"/>'
        )
    qml = f'''<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34">
  <pipe>
    <rasterrenderer band="1" type="paletted" opacity="1" alphaBand="-1">
      <rasterTransparency/>
      <colorPalette>{''.join(items)}</colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
'''
    output_path.write_text(qml, encoding="utf-8")
    return output_path


def create_legend_json(class_names: dict[int, str], output_path: Path) -> Path:
    entries = []
    for i, (label, name) in enumerate(sorted(class_names.items())):
        r, g, b = _class_color(i, len(class_names))
        entries.append({"label": int(label), "name": name, "rgb": [r, g, b]})
    save_json({"classes": entries}, output_path)
    return output_path
