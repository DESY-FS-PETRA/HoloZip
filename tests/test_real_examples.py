"""Validate the published TIFF execution examples without pretrained weights."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

FOLDER = Path(__file__).resolve().parents[1] / "examples/real_holograms"


@pytest.mark.parametrize(
    "name", ["img_y0_1_0268.tiff", "img_y0_2_0076.tiff", "img_y0_3_0055.tiff"]
)
def test_real_tiff_integrity(name):
    manifest = json.loads((FOLDER / "manifest.json").read_text())
    entry = next(item for item in manifest["files"] if item["filename"] == name)
    path = FOLDER / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert path.stat().st_size == entry["size_bytes"]
    image = tifffile.imread(path)
    assert list(image.shape) == entry["shape"] == [2048, 2048]
    assert str(image.dtype) == entry["dtype"] == "float32"
    assert np.isfinite(image).all()
    stats = json.loads((FOLDER / "global_stats.json").read_text())
    assert (
        image.min() >= stats["global_raw_min"]
        and image.max() <= stats["global_raw_max"]
    )


def test_real_example_normalization_provenance():
    manifest = json.loads((FOLDER / "manifest.json").read_text())
    stats = FOLDER / manifest["normalization"]["filename"]
    assert (
        hashlib.sha256(stats.read_bytes()).hexdigest()
        == manifest["normalization"]["sha256"]
    )
    assert json.loads(stats.read_text())["frame_count"] == 2206
    assert manifest["checkpoint"]["bundled"] is True
