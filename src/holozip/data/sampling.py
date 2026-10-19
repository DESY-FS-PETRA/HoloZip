"""data.sampling extracted from the supplied v5.5 codec."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from holozip.data.normalization import global_log_normalize


def fringe_importance_lowres(image: np.ndarray) -> np.ndarray:
    """Low-resolution fringe/texture importance map.

    This is deliberately *not* a hard Top-K detector. It produces a smooth score
    used only as a probability distribution for weighted random candidate sampling.
    Band/oscillation cues dominate and the gradient term is intentionally weak so
    strong specimen/background edges do not monopolize the training crops.
    """
    x = torch.from_numpy(np.ascontiguousarray(image)[None, None]).float()
    blur_small = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    blur_large = F.avg_pool2d(x, kernel_size=21, stride=1, padding=10)
    band = torch.abs(blur_small - blur_large)

    local_mean = F.avg_pool2d(x, 9, 1, 4)
    local_sq = F.avg_pool2d(x * x, 9, 1, 4)
    local_std = torch.sqrt(
        torch.clamp(local_sq - local_mean * local_mean, min=0.0) + 1e-8
    )

    gx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    gy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    grad = torch.sqrt(gx * gx + gy * gy + 1e-8)

    score = 0.55 * band + 0.35 * local_std + 0.10 * grad
    score = F.avg_pool2d(score, 9, 1, 4)[0, 0].numpy()
    score -= score.min()
    score /= score.max() + 1e-8
    return score.astype(np.float32)


def precompute_sampling_pool(
    files: Sequence[str],
    stats: Dict[str, Any],
    output_path: str,
    crop_size: int,
    pool_size: int = 128,
    downsample: int = 4,
    uniform_floor: float = 0.15,
    seed: int = 42,
) -> None:
    """Precompute soft fringe-aware crop candidates once for the whole dataset.

    Each image gets ``pool_size`` weighted-random crop top-left coordinates. The
    candidates are sampled from a low-resolution soft importance map with a uniform
    probability floor. They are *not* the top-scoring K locations.

    Training can then choose 60% fully random crops and 40% coordinates from this
    pool without recalculating a 2048x2048 importance map every epoch.
    """
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if not 0.0 <= uniform_floor <= 1.0:
        raise ValueError("uniform_floor must be in [0,1]")

    probe = tifffile.imread(files[0]).squeeze()
    height, width = probe.shape
    if crop_size > min(height, width):
        raise ValueError("crop_size exceeds image dimensions")

    centers = np.empty((len(files), pool_size, 2), dtype=np.int32)
    rng_master = np.random.RandomState(seed)
    margin = crop_size // 2

    print(
        f"[SamplingPool] images={len(files)} pool_size={pool_size} "
        f"downsample={downsample} crop={crop_size}",
        flush=True,
    )

    for i, path in enumerate(files):
        raw = tifffile.imread(path).squeeze()
        if raw.shape != (height, width):
            raise ValueError(
                f"Image shape changed: {path} has {raw.shape}, expected {(height, width)}"
            )

        normalized = global_log_normalize(raw, stats)
        if downsample > 1:
            tensor = torch.from_numpy(normalized[None, None])
            low = F.avg_pool2d(
                tensor,
                kernel_size=downsample,
                stride=downsample,
                ceil_mode=False,
            )[0, 0].numpy()
        else:
            low = normalized

        importance = fringe_importance_lowres(low)
        low_h, low_w = importance.shape

        # Only centers that can support a full-resolution crop are valid.
        low_margin = int(math.ceil(margin / downsample))
        valid = np.zeros_like(importance, dtype=np.float32)
        y0, y1 = low_margin, max(low_margin + 1, low_h - low_margin)
        x0, x1 = low_margin, max(low_margin + 1, low_w - low_margin)
        valid[y0:y1, x0:x1] = 1.0

        weights = valid * (uniform_floor + (1.0 - uniform_floor) * importance)
        flat = weights.ravel().astype(np.float64)
        total = flat.sum()
        if total <= 0:
            flat = valid.ravel().astype(np.float64)
            total = flat.sum()
        probs = flat / max(total, 1e-12)

        valid_count = int(np.count_nonzero(flat))
        replace = pool_size > valid_count
        # Seed per image so the pool is reproducible independent of worker count.
        rng = np.random.RandomState(int(rng_master.randint(0, 2**31 - 1)))
        chosen = rng.choice(flat.size, size=pool_size, replace=replace, p=probs)
        cy_low, cx_low = np.unravel_index(chosen, importance.shape)

        cy = np.clip(cy_low * downsample + downsample // 2, margin, height - margin)
        cx = np.clip(cx_low * downsample + downsample // 2, margin, width - margin)
        top = np.clip(cy - margin, 0, height - crop_size)
        left = np.clip(cx - margin, 0, width - crop_size)
        centers[i, :, 0] = top.astype(np.int32)
        centers[i, :, 1] = left.astype(np.int32)

        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"  {i + 1}/{len(files)}", flush=True)

    metadata = {
        "version": 1,
        "kind": "soft_weighted_candidate_pool",
        "crop_size": int(crop_size),
        "pool_size": int(pool_size),
        "downsample": int(downsample),
        "uniform_floor": float(uniform_floor),
        "height": int(height),
        "width": int(width),
        "seed": int(seed),
        "hard_topk": False,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        filenames=np.asarray([os.path.basename(p) for p in files]),
        centers=centers,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"[SamplingPool] saved {output}")
    print(json.dumps(metadata, indent=2))


def load_sampling_pool(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        filenames = [str(x) for x in data["filenames"].tolist()]
        centers = np.asarray(data["centers"], dtype=np.int32)
        metadata = json.loads(str(data["metadata"].item()))
    if (
        centers.ndim != 3
        or centers.shape[-1] != 2
        or centers.shape[0] != len(filenames)
    ):
        raise ValueError(f"Invalid sampling-pool shape: {centers.shape}")
    mapping = {name: centers[i] for i, name in enumerate(filenames)}
    return mapping, metadata
