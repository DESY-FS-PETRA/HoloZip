"""metrics extracted from the supplied v5.5 codec."""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def normalized_metrics(
    reference: np.ndarray, reconstruction: np.ndarray
) -> Dict[str, float]:
    reference = reference.astype(np.float32)
    reconstruction = reconstruction.astype(np.float32)
    diff = reference - reconstruction
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    ssim = float(
        skimage_ssim(
            reference,
            reconstruction,
            data_range=1.0,
            gaussian_weights=True,
            win_size=11,
        )
    )
    return {
        "ssim": ssim,
        "psnr": psnr,
        "mse": mse,
        "mae": mae,
        "max_error": float(np.max(np.abs(diff))),
    }


def raw_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> Dict[str, float]:
    ref = reference.astype(np.float64)
    rec = reconstruction.astype(np.float64)
    diff = ref - rec
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    data_range = float(ref.max() - ref.min())
    psnr = (
        float("inf") if mse <= 1e-18 else float(10.0 * np.log10((data_range**2) / mse))
    )
    return {
        "raw_psnr": psnr,
        "raw_mse": mse,
        "raw_mae": mae,
        "raw_max_error": float(np.max(np.abs(diff))),
        "raw_mean": float(ref.mean()),
        "reconstruction_mean": float(rec.mean()),
        "mean_ratio": float(rec.mean() / (ref.mean() + 1e-12)),
    }
