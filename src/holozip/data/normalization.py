"""data.normalization extracted from the supplied v5.5 codec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import tifffile


def compute_global_stats(files: Sequence[str], output_json: str) -> Dict[str, Any]:
    """Compute one shared reversible log normalization for the whole dataset.

    We intentionally use global extrema rather than per-image extrema so that the same
    normalized value has the same physical intensity meaning for every hologram.
    """
    global_raw_min = float("inf")
    global_raw_max = -float("inf")
    dtypes: set[str] = set()

    print(f"[Stats] pass 1/{len(files)}: global raw min/max")
    for i, path in enumerate(files, 1):
        raw = tifffile.imread(path).squeeze()
        if raw.ndim != 2:
            raise ValueError(f"Expected 2-D TIFF, got {raw.shape} for {path}")
        dtypes.add(np.dtype(raw.dtype).str)
        x = raw.astype(np.float32)
        global_raw_min = min(global_raw_min, float(np.nanmin(x)))
        global_raw_max = max(global_raw_max, float(np.nanmax(x)))
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    # For float data, subtract the true global minimum before log1p.
    # For integer data, non-negative values are assumed as in the previous pipeline.
    float_data = any(np.issubdtype(np.dtype(dt), np.floating) for dt in dtypes)
    if float_data:
        shifted_min = 0.0
        shifted_max = max(global_raw_max - global_raw_min, 0.0)
    else:
        shifted_min = max(global_raw_min, 0.0)
        shifted_max = max(global_raw_max, 0.0)

    global_log_min = float(np.log1p(shifted_min))
    global_log_max = float(np.log1p(shifted_max))
    if global_log_max <= global_log_min:
        raise ValueError("Degenerate global normalization range")

    stats = {
        "version": 1,
        "frame_count": int(len(files)),
        "global_raw_min": float(global_raw_min),
        "global_raw_max": float(global_raw_max),
        "global_log_min": global_log_min,
        "global_log_max": global_log_max,
        "float_data": bool(float_data),
        "dtypes": sorted(dtypes),
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Stats] saved {output_json}")
    print(json.dumps(stats, indent=2))
    return stats


def load_stats(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    required = {"global_raw_min", "global_log_min", "global_log_max", "float_data"}
    missing = required - set(stats)
    if missing:
        raise ValueError(f"Stats file missing keys: {sorted(missing)}")
    return stats


def global_log_normalize(raw: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw).squeeze()
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {raw.shape}")
    x = raw.astype(np.float32)
    if stats["float_data"]:
        shifted = np.maximum(x - float(stats["global_raw_min"]), 0.0)
    else:
        shifted = np.maximum(x, 0.0)
    log_x = np.log1p(shifted)
    lo = float(stats["global_log_min"])
    hi = float(stats["global_log_max"])
    normalized = (log_x - lo) / max(hi - lo, 1e-8)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def global_log_denormalize(
    normalized: np.ndarray, stats: Dict[str, Any], dtype: np.dtype
) -> np.ndarray:
    x = np.asarray(normalized, dtype=np.float64)
    lo = float(stats["global_log_min"])
    hi = float(stats["global_log_max"])
    log_x = np.clip(x, 0.0, 1.0) * (hi - lo) + lo
    shifted = np.expm1(log_x)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        raw = shifted + float(stats["global_raw_min"])
        return raw.astype(np.float32)
    info = np.iinfo(dtype)
    return np.clip(shifted, info.min, info.max).astype(dtype)
