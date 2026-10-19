"""data.io extracted from the supplied v5.5 codec."""

from __future__ import annotations

import glob
import os
from typing import List


def list_images(data_dir: str, image_glob: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, image_glob)))
    if not files and image_glob == "*.tiff":
        files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    if not files:
        raise FileNotFoundError(f"No images matching {image_glob!r} in {data_dir!r}")
    return files
