from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import random
import struct
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    from compressai.layers import GDN
except ImportError as exc:
    raise ImportError("CompressAI is required: pip install compressai") from exc

try:
    from pytorch_msssim import ms_ssim
except ImportError as exc:
    raise ImportError("pytorch-msssim is required: pip install pytorch-msssim") from exc

from skimage.metrics import structural_similarity as skimage_ssim


# =============================================================================
# Runtime
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

MAGIC = b"HIW52\x00\x00\x00"  # batched deterministic CPU entropy path
VERSION = 55
@dataclasses.dataclass
class Config:
    # Data/training
    image_glob: str = "*.tiff"
    crop_size: int = 256
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    aux_learning_rate: float = 1e-3
    num_workers: int = 4
    seed: int = 42
    val_fraction: float = 0.05
    val_batches: int = 8
    grad_clip: float = 1.0
    amp: bool = True

    # Sampling: deliberately NOT hard Top-K.
    # A large uniform component prevents ROI bias.
    fringe_sampling_probability: float = 0.40

    # Model capacity
    analysis_channels: int = 128
    latent_channels: int = 192
    hyper_channels: int = 128

    # Full image coding
    tile_size: int = 256
    tile_overlap: int = 32
    tile_batch: int = 8

    # Rate-distortion
    lambda_rate: float = 0.015
    w_charbonnier: float = 1.00
    w_ms_ssim: float = 0.50
    w_complex_fft: float = 0.12
    w_phase: float = 0.05
    w_band_fft: float = 0.15
    w_gradient: float = 0.12
    w_wavelet: float = 0.10
    w_range: float = 0.05

    # Fringe frequency band, cycles / pixel
    fringe_low: float = 0.04
    fringe_high: float = 0.28

    def validate(self) -> None:
        if self.crop_size % 16 != 0:
            raise ValueError("crop_size must be divisible by 16")
        if self.tile_size % 16 != 0:
            raise ValueError("tile_size must be divisible by 16")
        if not 0 <= self.tile_overlap < self.tile_size:
            raise ValueError("tile_overlap must be in [0, tile_size)")
        if not 0 <= self.fringe_sampling_probability <= 1:
            raise ValueError("fringe_sampling_probability must be in [0, 1]")


# =============================================================================
# Utilities
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_decode_profile(profile: str = "native") -> None:
    """Select a CUDA backend profile for decode.

    On the target PyTorch/CUDA stack used for this project, forcing cuDNN
    deterministic mode with TF32 disabled can make some ConvTranspose2d/GDN
    synthesis paths produce NaNs for otherwise valid entropy-decoded latents.
    The empirically stable profile is the same profile used during training and
    compression (benchmark=True, deterministic=False, TF32=True).

    Additional profiles are kept only as fallbacks for individual tiles.
    """
    if not torch.cuda.is_available():
        return

    if profile == "native":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif profile == "benchmark_off":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif profile == "no_tf32":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif profile == "deterministic":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        raise ValueError(f"Unknown decode profile: {profile}")


def require_finite_tensor(x: torch.Tensor, where: str) -> None:
    if not torch.isfinite(x).all():
        n_nan = int(torch.isnan(x).sum().item())
        n_inf = int(torch.isinf(x).sum().item())
        raise FloatingPointError(
            f"Non-finite tensor at {where}: NaN={n_nan}, Inf={n_inf}, shape={tuple(x.shape)}"
        )


def require_finite_array(x: np.ndarray, where: str) -> None:
    if not np.isfinite(x).all():
        n_nan = int(np.isnan(x).sum())
        n_inf = int(np.isinf(x).sum())
        raise FloatingPointError(
            f"Non-finite array at {where}: NaN={n_nan}, Inf={n_inf}, shape={x.shape}"
        )


def list_images(data_dir: str, image_glob: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, image_glob)))
    if not files and image_glob == "*.tiff":
        files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    if not files:
        raise FileNotFoundError(f"No images matching {image_glob!r} in {data_dir!r}")
    return files


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


def global_log_denormalize(normalized: np.ndarray, stats: Dict[str, Any], dtype: np.dtype) -> np.ndarray:
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


def normalized_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> Dict[str, float]:
    reference = reference.astype(np.float32)
    reconstruction = reconstruction.astype(np.float32)
    diff = reference - reconstruction
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    ssim = float(skimage_ssim(reference, reconstruction, data_range=1.0,
                              gaussian_weights=True, win_size=11))
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
    psnr = float("inf") if mse <= 1e-18 else float(10.0 * np.log10((data_range ** 2) / mse))
    return {
        "raw_psnr": psnr,
        "raw_mse": mse,
        "raw_mae": mae,
        "raw_max_error": float(np.max(np.abs(diff))),
        "raw_mean": float(ref.mean()),
        "reconstruction_mean": float(rec.mean()),
        "mean_ratio": float(rec.mean() / (ref.mean() + 1e-12)),
    }


# =============================================================================
# Fast soft fringe-aware sampling: precomputed weighted candidate pools
# =============================================================================


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
    local_std = torch.sqrt(torch.clamp(local_sq - local_mean * local_mean, min=0.0) + 1e-8)

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
        raise ValueError('crop_size must be positive')
    if pool_size <= 0:
        raise ValueError('pool_size must be positive')
    if downsample < 1:
        raise ValueError('downsample must be >= 1')
    if not 0.0 <= uniform_floor <= 1.0:
        raise ValueError('uniform_floor must be in [0,1]')

    probe = tifffile.imread(files[0]).squeeze()
    height, width = probe.shape
    if crop_size > min(height, width):
        raise ValueError('crop_size exceeds image dimensions')

    centers = np.empty((len(files), pool_size, 2), dtype=np.int32)
    rng_master = np.random.RandomState(seed)
    margin = crop_size // 2

    print(
        f'[SamplingPool] images={len(files)} pool_size={pool_size} '
        f'downsample={downsample} crop={crop_size}',
        flush=True,
    )

    for i, path in enumerate(files):
        raw = tifffile.imread(path).squeeze()
        if raw.shape != (height, width):
            raise ValueError(f'Image shape changed: {path} has {raw.shape}, expected {(height, width)}')

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
            print(f'  {i + 1}/{len(files)}', flush=True)

    metadata = {
        'version': 1,
        'kind': 'soft_weighted_candidate_pool',
        'crop_size': int(crop_size),
        'pool_size': int(pool_size),
        'downsample': int(downsample),
        'uniform_floor': float(uniform_floor),
        'height': int(height),
        'width': int(width),
        'seed': int(seed),
        'hard_topk': False,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        filenames=np.asarray([os.path.basename(p) for p in files]),
        centers=centers,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f'[SamplingPool] saved {output}')
    print(json.dumps(metadata, indent=2))


def load_sampling_pool(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        filenames = [str(x) for x in data['filenames'].tolist()]
        centers = np.asarray(data['centers'], dtype=np.int32)
        metadata = json.loads(str(data['metadata'].item()))
    if centers.ndim != 3 or centers.shape[-1] != 2 or centers.shape[0] != len(filenames):
        raise ValueError(f'Invalid sampling-pool shape: {centers.shape}')
    mapping = {name: centers[i] for i, name in enumerate(filenames)}
    return mapping, metadata


class HologramCropDataset(Dataset):
    """Independent hologram crops with fast precomputed soft sampling.

    Sampling policy:
      - (1 - p): fully uniform random crop
      - p: choose one candidate from a precomputed weighted-random fringe pool

    The pool is not hard Top-K. It has a uniform probability floor and is generated
    by probabilistic sampling. A small random jitter at training time further avoids
    repeatedly seeing exactly the same coordinates.
    """

    def __init__(
        self,
        files: Sequence[str],
        stats: Dict[str, Any],
        cfg: Config,
        sampling_pool: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False,
        sampling_jitter: int = 32,
    ) -> None:
        self.files = list(files)
        self.stats = stats
        self.cfg = cfg
        self.sampling_pool = sampling_pool
        self.deterministic = deterministic
        self.sampling_jitter = max(0, int(sampling_jitter))
        probe = tifffile.imread(self.files[0]).squeeze()
        self.height, self.width = probe.shape
        if cfg.crop_size > min(self.height, self.width):
            raise ValueError('crop_size exceeds image dimensions')
        if cfg.fringe_sampling_probability > 0 and sampling_pool is None:
            raise ValueError(
                'fringe_sampling_probability > 0 requires a precomputed sampling pool. '
                'Run the precompute-sampling-pool command and pass --sampling-pool.'
            )

    def __len__(self) -> int:
        return len(self.files)

    def _choose_crop(self, index: int, rng: np.random.RandomState) -> Tuple[int, int]:
        crop = self.cfg.crop_size
        max_top = self.height - crop
        max_left = self.width - crop

        if (
            self.sampling_pool is None
            or rng.rand() >= self.cfg.fringe_sampling_probability
        ):
            return rng.randint(0, max_top + 1), rng.randint(0, max_left + 1)

        name = os.path.basename(self.files[index])
        candidates = self.sampling_pool.get(name)
        if candidates is None or len(candidates) == 0:
            return rng.randint(0, max_top + 1), rng.randint(0, max_left + 1)

        if self.deterministic:
            candidate_index = index % len(candidates)
        else:
            candidate_index = rng.randint(0, len(candidates))
        top, left = (int(v) for v in candidates[candidate_index])

        # Jitter only during training. This increases crop diversity without any
        # expensive image analysis inside __getitem__.
        if not self.deterministic and self.sampling_jitter > 0:
            j = self.sampling_jitter
            top += int(rng.randint(-j, j + 1))
            left += int(rng.randint(-j, j + 1))
        return int(np.clip(top, 0, max_top)), int(np.clip(left, 0, max_left))

    @staticmethod
    def _read_raw_crop(path: str, top: int, left: int, crop: int) -> np.ndarray:
        """Read only the requested crop when TIFF layout allows memory mapping."""
        try:
            mm = tifffile.memmap(path)
            arr = np.asarray(mm).squeeze()
            result = np.asarray(arr[top:top + crop, left:left + crop])
            if result.shape == (crop, crop):
                return result
        except Exception:
            pass
        raw = tifffile.imread(path).squeeze()
        return raw[top:top + crop, left:left + crop]

    def __getitem__(self, index: int) -> torch.Tensor:
        # Coordinates no longer depend on computing a full-resolution map here.
        rng = np.random.RandomState(index if self.deterministic else None)
        top, left = self._choose_crop(index, rng)
        crop = self.cfg.crop_size
        raw_crop = self._read_raw_crop(self.files[index], top, left, crop)

        # Global normalization is point-wise, so normalize only the 256x256 crop
        # rather than converting the entire 2048x2048 image every epoch.
        image_crop = global_log_normalize(raw_crop, self.stats)
        return torch.from_numpy(image_crop[None].astype(np.float32))


# =============================================================================
# Wavelet + hyperprior codec
# =============================================================================


class HaarWavelet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[1, 1], [1, 1]],
                [[-1, -1], [1, 1]],
                [[-1, 1], [-1, 1]],
                [[1, -1], [-1, 1]],
            ],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("analysis_kernel", kernels[:, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.analysis_kernel, stride=2)

    def inverse(self, bands: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(bands, self.analysis_kernel, stride=2)


def conv(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)


def deconv(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Module:
    return nn.ConvTranspose2d(
        in_ch, out_ch, kernel, stride=stride,
        padding=kernel // 2, output_padding=stride - 1,
    )


class IndependentWaveletHyperprior(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        N = cfg.analysis_channels
        M = cfg.latent_channels
        H = cfg.hyper_channels
        self.cfg = cfg
        self.wavelet = HaarWavelet()

        self.g_a = nn.Sequential(
            conv(4, N), GDN(N),
            conv(N, N), GDN(N),
            conv(N, N), GDN(N),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            deconv(M, N), GDN(N, inverse=True),
            deconv(N, N), GDN(N, inverse=True),
            deconv(N, N), GDN(N, inverse=True),
            deconv(N, 4),
        )
        self.h_a = nn.Sequential(
            nn.Conv2d(M, H, 3, padding=1), nn.LeakyReLU(inplace=True),
            conv(H, H, 5, 2), nn.LeakyReLU(inplace=True),
            conv(H, H, 5, 2),
        )
        self.h_s = nn.Sequential(
            deconv(H, H, 5, 2), nn.LeakyReLU(inplace=True),
            deconv(H, H * 2, 5, 2), nn.LeakyReLU(inplace=True),
            nn.Conv2d(H * 2, M * 2, 3, padding=1),
        )

        self.entropy_bottleneck = EntropyBottleneck(H)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        bands = self.wavelet(x)
        y = self.g_a(bands)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        bands_hat = self.g_s(y_hat)
        preclamp = self.wavelet.inverse(bands_hat)
        reconstruction = torch.clamp(preclamp, 0.0, 1.0)
        return {
            "bands": bands,
            "bands_hat": bands_hat,
            "preclamp": preclamp,
            "reconstruction": reconstruction,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        bands = self.wavelet(x)
        y = self.g_a(bands)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_shape = tuple(int(v) for v in z.size()[-2:])
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z_shape}

    @torch.no_grad()
    def decompress(self, strings: List[List[bytes]], shape: Tuple[int, int]) -> torch.Tensor:
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
        bands_hat = self.g_s(y_hat)
        return torch.clamp(self.wavelet.inverse(bands_hat), 0.0, 1.0)

    def update(self, force: bool = False) -> bool:
        updated = self.gaussian_conditional.update_scale_table(
            torch.exp(torch.linspace(math.log(0.11), math.log(256.0), 64)),
            force=force,
        )
        updated |= self.entropy_bottleneck.update(force=force)
        return bool(updated)

    def aux_loss(self) -> torch.Tensor:
        losses = [m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)]
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(losses)


# =============================================================================
# Losses
# =============================================================================


def rate_bpp(likelihoods: Dict[str, torch.Tensor], pixels: int) -> torch.Tensor:
    bits = torch.zeros((), device=next(iter(likelihoods.values())).device)
    for likelihood in likelihoods.values():
        bits = bits + torch.log(likelihood.clamp_min(1e-9)).sum() / (-math.log(2.0))
    return bits / pixels


def charbonnier(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + 1e-6).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def frequency_grids(height: int, width: int, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, device=device).reshape(height, 1)
    fx = torch.fft.rfftfreq(width, device=device).reshape(1, width // 2 + 1)
    return torch.sqrt(fy * fy + fx * fx)


def spectral_losses(pred: torch.Tensor, target: torch.Tensor, low: float, high: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_f = torch.fft.rfft2(pred[:, 0].float(), norm="ortho")
    target_f = torch.fft.rfft2(target[:, 0].float(), norm="ortho")
    radial = frequency_grids(pred.shape[-2], pred.shape[-1], pred.device)
    band = ((radial >= low) & (radial <= high)).float()[None]

    complex_loss = torch.abs(pred_f - target_f).mean()
    cross = pred_f * torch.conj(target_f)
    phase_weight = torch.abs(target_f).detach()
    phase_weight = phase_weight / (phase_weight.mean(dim=(-2, -1), keepdim=True) + 1e-8)
    phase_loss = ((1.0 - torch.cos(torch.angle(cross))) * phase_weight * band).sum() / (
        (phase_weight * band).sum() + 1e-8
    )
    band_loss = (
        torch.abs(torch.log1p(torch.abs(pred_f)) - torch.log1p(torch.abs(target_f))) * band
    ).sum() / (band.sum() * pred.shape[0] + 1e-8)
    return complex_loss, phase_loss, band_loss


def compute_loss(output: Dict[str, Any], target: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, Dict[str, float]]:
    recon = output["reconstruction"]
    pixels = target.shape[0] * target.shape[-2] * target.shape[-1]
    bpp = rate_bpp(output["likelihoods"], pixels)
    l_charb = charbonnier(recon, target)
    l_ms = 1.0 - ms_ssim(recon.float(), target.float(), data_range=1.0, size_average=True)
    l_complex, l_phase, l_band = spectral_losses(recon, target, cfg.fringe_low, cfg.fringe_high)
    l_grad = gradient_loss(recon, target)
    l_wavelet = F.l1_loss(output["bands_hat"], output["bands"])
    pre = output["preclamp"]
    l_range = (F.relu(pre - 1.0).mean() + F.relu(-pre).mean())

    total = (
        cfg.lambda_rate * bpp
        + cfg.w_charbonnier * l_charb
        + cfg.w_ms_ssim * l_ms
        + cfg.w_complex_fft * l_complex
        + cfg.w_phase * l_phase
        + cfg.w_band_fft * l_band
        + cfg.w_gradient * l_grad
        + cfg.w_wavelet * l_wavelet
        + cfg.w_range * l_range
    )
    stats = {
        "loss": float(total.detach()),
        "bpp": float(bpp.detach()),
        "charb": float(l_charb.detach()),
        "ms_ssim_loss": float(l_ms.detach()),
        "complex_fft": float(l_complex.detach()),
        "phase": float(l_phase.detach()),
        "band_fft": float(l_band.detach()),
        "gradient": float(l_grad.detach()),
        "wavelet": float(l_wavelet.detach()),
        "range": float(l_range.detach()),
    }
    return total, stats


# =============================================================================
# Training
# =============================================================================


def configure_optimizers(model: IndependentWaveletHyperprior, cfg: Config):
    aux_names = {name for name, _ in model.named_parameters() if name.endswith(".quantiles")}
    main_params = [p for n, p in model.named_parameters() if n not in aux_names and p.requires_grad]
    aux_params = [p for n, p in model.named_parameters() if n in aux_names and p.requires_grad]
    optimizer = torch.optim.AdamW(main_params, lr=cfg.learning_rate, weight_decay=1e-5)
    aux_optimizer = torch.optim.Adam(aux_params, lr=cfg.aux_learning_rate)
    return optimizer, aux_optimizer


@torch.no_grad()
def validate(model: IndependentWaveletHyperprior, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "ssim": 0.0, "psnr": 0.0, "bpp": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= cfg.val_batches:
            break
        batch = batch.to(DEVICE, non_blocking=True)
        out = model(batch)
        loss, stats = compute_loss(out, batch, cfg)
        recon = out["reconstruction"].float()
        mse = F.mse_loss(recon, batch.float()).item()
        sums["loss"] += float(loss)
        sums["bpp"] += stats["bpp"]
        sums["ssim"] += float(ms_ssim(recon, batch.float(), data_range=1.0, size_average=True))
        sums["psnr"] += 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        count += 1
    if count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return {k: v / count for k, v in sums.items()}


def save_checkpoint(path: str, model: IndependentWaveletHyperprior, cfg: Config,
                    stats_path: str, epoch: int, best_ssim: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "config": dataclasses.asdict(cfg),
            "stats_filename": os.path.basename(stats_path),
            "epoch": epoch,
            "best_ssim": best_ssim,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str) -> Tuple[IndependentWaveletHyperprior, Config, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    cfg.validate()
    model = IndependentWaveletHyperprior(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    return model, cfg, ckpt


def train_command(args: argparse.Namespace) -> None:
    cfg = Config(
        image_glob=args.image_glob,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        val_batches=args.val_batches,
        fringe_sampling_probability=args.fringe_sampling_probability,
        analysis_channels=args.analysis_channels,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_batch=args.tile_batch,
        lambda_rate=args.lambda_rate,
        fringe_low=args.fringe_low,
        fringe_high=args.fringe_high,
    )
    cfg.validate()
    seed_everything(cfg.seed)

    files = list_images(args.data_dir, cfg.image_glob)
    stats = load_stats(args.stats)
    split = max(1, int(len(files) * (1.0 - cfg.val_fraction)))
    train_files = files[:split]
    val_files = files[split:] if split < len(files) else files[-max(1, len(files)//20):]

    sampling_pool = None
    if cfg.fringe_sampling_probability > 0:
        if not args.sampling_pool:
            raise ValueError(
                "fringe sampling is enabled but --sampling-pool was not provided. "
                "Run precompute-sampling-pool first."
            )
        sampling_pool, pool_meta = load_sampling_pool(args.sampling_pool)
        if int(pool_meta.get("crop_size", cfg.crop_size)) != cfg.crop_size:
            raise ValueError(
                f"Sampling pool crop_size={pool_meta.get('crop_size')} does not match "
                f"training crop_size={cfg.crop_size}"
            )
        print(
            f"[SamplingPool] loaded {args.sampling_pool} "
            f"pool_size={pool_meta.get('pool_size')} downsample={pool_meta.get('downsample')}"
        )

    train_ds = HologramCropDataset(
        train_files, stats, cfg, sampling_pool=sampling_pool,
        deterministic=False, sampling_jitter=args.sampling_jitter,
    )
    val_ds = HologramCropDataset(
        val_files, stats, cfg, sampling_pool=sampling_pool,
        deterministic=True, sampling_jitter=0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=min(2, cfg.num_workers), pin_memory=True,
    )

    model = IndependentWaveletHyperprior(cfg).to(DEVICE)
    optimizer, aux_optimizer = configure_optimizers(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and DEVICE.type == "cuda")

    best_ssim = -float("inf")
    patience = 20
    stale = 0
    print(f"[Device] {DEVICE}")
    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}")
    print(f"[Sampling] uniform={1-cfg.fringe_sampling_probability:.2f}, fringe-soft={cfg.fringe_sampling_probability:.2f}")
    print(f"[Model] parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.time()
        sums: Dict[str, float] = {}
        steps = 0
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
                out = model(batch)
                loss, stats_batch = compute_loss(out, batch, cfg)

            if not torch.isfinite(loss):
                print("[Warning] non-finite main loss; batch skipped")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            aux_loss = model.aux_loss()
            if torch.isfinite(aux_loss):
                aux_loss.backward()
                aux_optimizer.step()

            for k, v in stats_batch.items():
                sums[k] = sums.get(k, 0.0) + v
            sums["aux"] = sums.get("aux", 0.0) + float(aux_loss.detach())
            steps += 1

        scheduler.step()
        if steps == 0:
            raise RuntimeError("No valid optimization steps completed")
        train_stats = {k: v / steps for k, v in sums.items()}
        val = validate(model, val_loader, cfg)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"loss={train_stats.get('loss', float('nan')):.5f} "
            f"bpp={train_stats.get('bpp', float('nan')):.3f} "
            f"valSSIM={val['ssim']:.5f} valPSNR={val['psnr']:.2f} "
            f"valBPP={val['bpp']:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
            f"time={elapsed:.0f}s"
        )

        if val["ssim"] > best_ssim + 1e-4:
            best_ssim = val["ssim"]
            stale = 0
            save_checkpoint(args.checkpoint, model, cfg, args.stats, epoch, best_ssim)
            print(f"  [best] saved {args.checkpoint}")
        else:
            stale += 1
            if stale >= patience:
                print(f"[Early stop] no validation improvement for {patience} epochs")
                break


# =============================================================================
# Full-image tiling and archive
# =============================================================================


def tile_positions(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def tile_coordinates(height: int, width: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    return [(t, l) for t in tile_positions(height, tile, overlap)
                   for l in tile_positions(width, tile, overlap)]


def hann_window_2d(size: int, device: torch.device) -> torch.Tensor:
    w = torch.hann_window(size, periodic=False, device=device)
    return torch.outer(w, w).clamp_min(1e-3)[None, None]


def pad_tile(tile: torch.Tensor, size: int) -> torch.Tensor:
    ph = size - tile.shape[-2]
    pw = size - tile.shape[-1]
    if ph or pw:
        tile = F.pad(tile, (0, pw, 0, ph), mode="reflect")
    return tile


def write_u32(f: BinaryIO, value: int) -> None:
    f.write(struct.pack(">I", int(value)))


def read_u32(f: BinaryIO) -> int:
    data = f.read(4)
    if len(data) != 4:
        raise EOFError("Unexpected end of archive")
    return struct.unpack(">I", data)[0]


def write_bytes(f: BinaryIO, payload: bytes) -> None:
    write_u32(f, len(payload))
    f.write(payload)


def read_bytes(f: BinaryIO) -> bytes:
    n = read_u32(f)
    payload = f.read(n)
    if len(payload) != n:
        raise EOFError("Truncated archive payload")
    return payload


def write_json(f: BinaryIO, obj: Dict[str, Any]) -> None:
    write_bytes(f, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def read_json(f: BinaryIO) -> Dict[str, Any]:
    return json.loads(read_bytes(f).decode("utf-8"))


@torch.no_grad()
def make_cpu_entropy_model(model: IndependentWaveletHyperprior) -> IndependentWaveletHyperprior:
    """Build a CPU copy for deterministic entropy coding/decoding.

    CompressAI entropy models contain runtime CDF buffers whose shapes are empty
    immediately after construction.  A direct ``load_state_dict`` from an already
    updated model therefore fails with size-mismatch errors for buffers such as
    ``_quantized_cdf`` and ``_offset``.

    We copy only learned/model parameters first (including the learned
    EntropyBottleneck quantiles), deliberately skip the generated entropy-table
    buffers, and then rebuild those buffers on CPU with ``update(force=True)``.
    This also guarantees that compression and later reconstruction derive the
    entropy tables using the same CPU implementation.
    """
    cpu_model = IndependentWaveletHyperprior(model.cfg).cpu()

    source_state = model.state_dict()

    # These are generated/runtime buffers, not learned weights.  Their shapes are
    # empty in a freshly constructed CompressAI model, so they must be rebuilt
    # rather than loaded.
    generated_entropy_suffixes = (
        "._offset",
        "._quantized_cdf",
        "._cdf_length",
        ".scale_table",
    )

    learned_state = {
        key: value.detach().cpu()
        for key, value in source_state.items()
        if not key.endswith(generated_entropy_suffixes)
    }

    incompatible = cpu_model.load_state_dict(learned_state, strict=False)

    # Only generated entropy buffers are expected to be missing.
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            "Unexpected keys while creating CPU entropy model: "
            + ", ".join(unexpected)
        )

    allowed_missing_suffixes = generated_entropy_suffixes
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.endswith(allowed_missing_suffixes)
    ]
    if bad_missing:
        raise RuntimeError(
            "Missing learned parameters while creating CPU entropy model: "
            + ", ".join(bad_missing)
        )

    cpu_model.eval()

    # Rebuild EntropyBottleneck CDFs and GaussianConditional scale/CDF tables
    # from the copied learned parameters, entirely on CPU.
    cpu_model.update(force=True)

    # Defensive checks: the runtime tables must now exist.
    if cpu_model.entropy_bottleneck._quantized_cdf.numel() == 0:
        raise RuntimeError("CPU EntropyBottleneck CDF table was not initialized")
    if cpu_model.gaussian_conditional._quantized_cdf.numel() == 0:
        raise RuntimeError("CPU GaussianConditional CDF table was not initialized")
    if cpu_model.gaussian_conditional.scale_table.numel() == 0:
        raise RuntimeError("CPU GaussianConditional scale table was not initialized")

    return cpu_model



class StageTimer:
    """Accumulate wall-clock stage timings, synchronizing CUDA only at stage boundaries."""
    def __init__(self):
        self.totals = {}
        self.calls = {}

    def add(self, name, dt):
        self.totals[name] = self.totals.get(name, 0.0) + float(dt)
        self.calls[name] = self.calls.get(name, 0) + 1

    def report(self):
        total = sum(self.totals.values())
        return {
            k: {
                "seconds": round(v, 3),
                "percent": round(100.0 * v / max(total, 1e-12), 1),
                "calls": self.calls.get(k, 0),
            }
            for k, v in sorted(self.totals.items(), key=lambda kv: -kv[1])
        }


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()



def configure_cpu_entropy_threads(requested: Optional[int] = None) -> int:
    """Tune PyTorch CPU threading for the small hyperprior convolutions.

    Very large OpenMP thread counts are counterproductive for the tiny h_a/h_s
    tensors used here.  A modest thread count usually gives much better latency.
    """
    if requested is None or requested <= 0:
        cpu_count = os.cpu_count() or 8
        requested = min(8, max(1, cpu_count // 2))
    torch.set_num_threads(int(requested))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before parallel work starts.
        pass
    return int(requested)


@torch.no_grad()
def compress_image_tiles(
    model: IndependentWaveletHyperprior,
    entropy_model: IndependentWaveletHyperprior,
    image: torch.Tensor,
    cfg: Config,
    profiler: Optional[StageTimer] = None,
    entropy_batch_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Compress one full image using GPU analysis + batched deterministic CPU entropy coding.

    V5.1 moved the entropy path to CPU for reproducibility, but processed every
    tile separately.  V5.2 keeps exactly the same deterministic CPU arithmetic
    while batching all tiles in each ``cfg.tile_batch`` group:

        GPU: wavelet + g_a
        one GPU->CPU transfer for y_batch
        CPU: h_a -> EntropyBottleneck -> h_s -> indexes -> GaussianConditional
        archive: one independent y/z byte stream per tile

    The archive remains independently decodable tile-by-tile; batching is only
    an execution optimization and does not introduce inter-tile dependencies.
    """
    _, _, height, width = image.shape
    coords_all = tile_coordinates(height, width, cfg.tile_size, cfg.tile_overlap)
    records: List[Dict[str, Any]] = []
    payload_bytes = 0

    ebs = int(entropy_batch_size or cfg.tile_batch)
    if ebs < 1:
        raise ValueError("entropy_batch_size must be >= 1")

    for start in range(0, len(coords_all), ebs):
        coords = coords_all[start:start + ebs]
        tiles: List[torch.Tensor] = []
        sizes: List[Tuple[int, int]] = []

        for top, left in coords:
            h = min(cfg.tile_size, height - top)
            w = min(cfg.tile_size, width - left)
            sizes.append((h, w))
            tiles.append(
                pad_tile(
                    image[..., top:top + h, left:left + w],
                    cfg.tile_size,
                )
            )

        # Expensive analysis stays on the GPU.
        _cuda_sync()
        _t = time.perf_counter()
        batch = torch.cat(tiles, dim=0)
        bands = model.wavelet(batch)
        y_gpu = model.g_a(bands)
        require_finite_tensor(y_gpu, f"encode batch {start} GPU y")
        _cuda_sync()
        if profiler: profiler.add("gpu_analysis_wavelet_g_a", time.perf_counter() - _t)

        # One transfer per tile batch rather than one transfer per tile.
        _t = time.perf_counter()
        y_cpu = y_gpu.float().cpu().contiguous()
        require_finite_tensor(y_cpu, f"encode batch {start} CPU y")
        if profiler: profiler.add("gpu_to_cpu_y", time.perf_counter() - _t)

        # Entire deterministic entropy/hyperprior path runs batched on CPU.
        _t = time.perf_counter()
        z_cpu = entropy_model.h_a(torch.abs(y_cpu))
        require_finite_tensor(z_cpu, f"encode batch {start} CPU z")
        if profiler: profiler.add("cpu_h_a", time.perf_counter() - _t)

        _t = time.perf_counter()
        z_strings = entropy_model.entropy_bottleneck.compress(z_cpu)
        if profiler: profiler.add("entropy_bottleneck_compress", time.perf_counter() - _t)
        z_shape = tuple(int(v) for v in z_cpu.size()[-2:])

        if len(z_strings) != len(coords):
            raise RuntimeError(
                f"EntropyBottleneck returned {len(z_strings)} strings for "
                f"{len(coords)} tiles"
            )

        # IMPORTANT: derive the y-model parameters from the entropy-decoded z,
        # exactly as the decoder will.
        _t = time.perf_counter()
        z_hat_cpu = entropy_model.entropy_bottleneck.decompress(
            z_strings,
            z_shape,
        )
        require_finite_tensor(z_hat_cpu, f"encode batch {start} decoded CPU z_hat")
        if profiler: profiler.add("entropy_bottleneck_roundtrip_decode", time.perf_counter() - _t)

        _t = time.perf_counter()
        gaussian_params = entropy_model.h_s(z_hat_cpu)
        require_finite_tensor(
            gaussian_params,
            f"encode batch {start} CPU hyper parameters",
        )
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        require_finite_tensor(scales_hat, f"encode batch {start} CPU scales")
        require_finite_tensor(means_hat, f"encode batch {start} CPU means")
        if profiler: profiler.add("cpu_h_s", time.perf_counter() - _t)

        _t = time.perf_counter()
        indexes = entropy_model.gaussian_conditional.build_indexes(scales_hat)
        if profiler: profiler.add("gaussian_build_indexes", time.perf_counter() - _t)

        _t = time.perf_counter()
        y_strings = entropy_model.gaussian_conditional.compress(
            y_cpu,
            indexes,
            means=means_hat,
        )
        if profiler: profiler.add("gaussian_compress", time.perf_counter() - _t)

        if len(y_strings) != len(coords):
            raise RuntimeError(
                f"GaussianConditional returned {len(y_strings)} strings for "
                f"{len(coords)} tiles"
            )

        for local, ((top, left), (h, w)) in enumerate(zip(coords, sizes)):
            ys = y_strings[local]
            zs = z_strings[local]
            payload_bytes += len(ys) + len(zs)
            records.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "height": int(h),
                    "width": int(w),
                    "shape": [int(v) for v in z_shape],
                    "y": ys,
                    "z": zs,
                }
            )

    return records, payload_bytes


@torch.no_grad()
def decompress_tile_batch_deterministic(
    model: IndependentWaveletHyperprior,
    entropy_model: IndependentWaveletHyperprior,
    batch_records: Sequence[Dict[str, Any]],
    batch_start_index: int,
) -> torch.Tensor:
    """Decode a batch of independent tile bitstreams safely and deterministically.

    CPU performs the full entropy/hyperprior path for the whole batch.  The
    resulting y_hat batch is transferred to GPU once, then g_s and inverse Haar
    run in one GPU batch.  No probability state is shared between tiles; each
    tile still owns its own y/z byte strings.
    """
    if not batch_records:
        raise ValueError("Empty tile batch")

    shapes = {tuple(int(v) for v in r["shape"]) for r in batch_records}
    if len(shapes) != 1:
        raise ValueError(f"Incompatible latent shapes in tile batch: {shapes}")
    shape = next(iter(shapes))

    z_strings = [r["z"] for r in batch_records]
    y_strings = [r["y"] for r in batch_records]

    z_hat_cpu = entropy_model.entropy_bottleneck.decompress(
        z_strings,
        shape,
    )
    require_finite_tensor(
        z_hat_cpu,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} CPU z_hat",
    )

    gaussian_params = entropy_model.h_s(z_hat_cpu)
    require_finite_tensor(
        gaussian_params,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} CPU hyper parameters",
    )
    scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
    scales_hat = F.softplus(scales_hat) + 1e-6
    require_finite_tensor(
        scales_hat,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} CPU scales",
    )
    require_finite_tensor(
        means_hat,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} CPU means",
    )

    indexes = entropy_model.gaussian_conditional.build_indexes(scales_hat)
    y_hat_cpu = entropy_model.gaussian_conditional.decompress(
        y_strings,
        indexes,
        means=means_hat,
    ).float().contiguous()
    require_finite_tensor(
        y_hat_cpu,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} CPU y_hat",
    )

    # Catch entropy/index corruption before inverse-GDN can overflow.
    max_abs_per_tile = y_hat_cpu.abs().flatten(1).amax(dim=1)
    bad = torch.nonzero(max_abs_per_tile > 1.0e4, as_tuple=False).flatten()
    if bad.numel():
        details = ", ".join(
            f"tile {batch_start_index + int(i)} max_abs={float(max_abs_per_tile[i]):.3g}"
            for i in bad
        )
        raise RuntimeError(
            "Implausible decoded latent(s): "
            + details
            + ". Archive/codec mismatch or entropy corruption suspected."
        )

    # One CPU->GPU transfer and one synthesis call for the whole batch.
    y_hat_gpu = y_hat_cpu.to(DEVICE, non_blocking=False)
    bands_hat = model.g_s(y_hat_gpu)
    require_finite_tensor(
        bands_hat,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} GPU synthesis bands",
    )

    preclamp = model.wavelet.inverse(bands_hat)
    require_finite_tensor(
        preclamp,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} inverse wavelet",
    )

    recon_batch = torch.clamp(preclamp, 0.0, 1.0)
    require_finite_tensor(
        recon_batch,
        f"tiles {batch_start_index}:{batch_start_index + len(batch_records)} reconstruction",
    )
    return recon_batch


@torch.no_grad()
def decompress_image_tiles(
    model: IndependentWaveletHyperprior,
    entropy_model: IndependentWaveletHyperprior,
    records: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    cfg: Config,
    decode_retries: int = 1,
    entropy_decode_batch_size: Optional[int] = None,
    synthesis_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Archive-compatible deterministic decoder.

    V5.4 compression used entropy_batch_size=32 while cfg.tile_batch stayed 8.
    The CPU hyperprior/index path must use the same batch grouping at decode.
    GPU synthesis batching is independent and may use a different size.
    """
    entropy_bs = int(entropy_decode_batch_size or 32)
    synth_bs = int(synthesis_batch_size or cfg.tile_batch)
    if entropy_bs < 1 or synth_bs < 1:
        raise ValueError("decode batch sizes must be >= 1")

    window = hann_window_2d(cfg.tile_size, DEVICE)
    require_finite_tensor(window, "Hann blending window")
    acc = torch.zeros((1, 1, height, width), device=DEVICE, dtype=torch.float32)
    weights = torch.zeros_like(acc)

    for entropy_start in range(0, len(records), entropy_bs):
        entropy_records = list(records[entropy_start:entropy_start + entropy_bs])
        shapes = {tuple(int(v) for v in r["shape"]) for r in entropy_records}
        if len(shapes) != 1:
            raise ValueError(f"Incompatible latent shapes in entropy batch: {shapes}")
        shape = next(iter(shapes))
        z_strings = [r["z"] for r in entropy_records]
        y_strings = [r["y"] for r in entropy_records]

        z_hat_cpu = entropy_model.entropy_bottleneck.decompress(z_strings, shape)
        require_finite_tensor(z_hat_cpu, f"tiles {entropy_start}:{entropy_start+len(entropy_records)} CPU z_hat")
        gaussian_params = entropy_model.h_s(z_hat_cpu)
        require_finite_tensor(gaussian_params, f"tiles {entropy_start}:{entropy_start+len(entropy_records)} CPU hyper parameters")
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        require_finite_tensor(scales_hat, f"tiles {entropy_start}:{entropy_start+len(entropy_records)} CPU scales")
        require_finite_tensor(means_hat, f"tiles {entropy_start}:{entropy_start+len(entropy_records)} CPU means")
        indexes = entropy_model.gaussian_conditional.build_indexes(scales_hat)
        y_hat_cpu = entropy_model.gaussian_conditional.decompress(
            y_strings, indexes, means=means_hat
        ).float().contiguous()
        require_finite_tensor(y_hat_cpu, f"tiles {entropy_start}:{entropy_start+len(entropy_records)} CPU y_hat")

        max_abs_per_tile = y_hat_cpu.abs().flatten(1).amax(dim=1)
        bad = torch.nonzero(max_abs_per_tile > 1.0e4, as_tuple=False).flatten()
        if bad.numel():
            details = ", ".join(
                f"tile {entropy_start + int(i)} max_abs={float(max_abs_per_tile[i]):.3g}" for i in bad
            )
            raise RuntimeError(
                "Implausible decoded latent(s): " + details +
                ". Check --entropy-decode-batch-size against compression --entropy-batch-size."
            )

        for local_start in range(0, len(entropy_records), synth_bs):
            local_end = min(local_start + synth_bs, len(entropy_records))
            synth_records = entropy_records[local_start:local_end]
            y_hat_gpu = y_hat_cpu[local_start:local_end].to(DEVICE, non_blocking=False)
            bands_hat = model.g_s(y_hat_gpu)
            require_finite_tensor(bands_hat, f"tiles {entropy_start+local_start}:{entropy_start+local_end} GPU synthesis bands")
            preclamp = model.wavelet.inverse(bands_hat)
            require_finite_tensor(preclamp, f"tiles {entropy_start+local_start}:{entropy_start+local_end} inverse wavelet")
            recon_batch = torch.clamp(preclamp, 0.0, 1.0)
            require_finite_tensor(recon_batch, f"tiles {entropy_start+local_start}:{entropy_start+local_end} reconstruction")

            for j, r in enumerate(synth_records):
                top, left = int(r["top"]), int(r["left"])
                h, w = int(r["height"]), int(r["width"])
                tile = recon_batch[j:j+1, :, :h, :w]
                tile_window = window[..., :h, :w]
                acc[..., top:top+h, left:left+w] += tile * tile_window
                weights[..., top:top+h, left:left+w] += tile_window

    require_finite_tensor(acc, "full-image accumulator")
    require_finite_tensor(weights, "full-image blending weights")
    if torch.any(weights <= 0):
        raise RuntimeError(f"Uncovered pixels during tile blending: {int((weights <= 0).sum())}")
    reconstruction = torch.clamp(acc / weights, 0.0, 1.0)
    require_finite_tensor(reconstruction, "final normalized reconstruction")
    return reconstruction


def write_frame_record(f: BinaryIO, meta: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> None:
    write_json(f, meta)
    write_u32(f, len(records))
    for r in records:
        tile_meta = {k: v for k, v in r.items() if k not in {"y", "z"}}
        write_json(f, tile_meta)
        write_bytes(f, r["y"])
        write_bytes(f, r["z"])


def read_frame_record(f: BinaryIO) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta = read_json(f)
    n = read_u32(f)
    records = []
    for _ in range(n):
        r = read_json(f)
        r["y"] = read_bytes(f)
        r["z"] = read_bytes(f)
        records.append(r)
    return meta, records


def compress_dataset_command(args: argparse.Namespace) -> None:
    model, cfg, _ = load_checkpoint(args.checkpoint)
    model.eval()
    model.update(force=True)
    entropy_model = make_cpu_entropy_model(model)
    print("[Entropy] batched deterministic CPU hyperprior/index path enabled")
    stats = load_stats(args.stats)
    files_all = list_images(args.data_dir, cfg.image_glob)

    start_frame = max(0, int(args.start_frame))
    files = files_all[start_frame:]
    if args.max_frames is not None:
        files = files[: int(args.max_frames)]
    if not files:
        raise ValueError("No frames selected")
    if args.tile_batch is not None:
        cfg.tile_batch = int(args.tile_batch)

    out = Path(args.archive)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Archive exists: {out}; pass --overwrite")

    header = {
        "version": VERSION,
        "frame_count": len(files),
        "source_start_frame": start_frame,
        "source_dataset_frame_count": len(files_all),
        "filenames": [os.path.basename(p) for p in files],
        "config": dataclasses.asdict(cfg),
        "stats": stats,
        "checkpoint": os.path.basename(args.checkpoint),
        "entropy_batch_size": int(getattr(args, "entropy_batch_size", 0) or cfg.tile_batch),
    }

    total_raw = 0
    total_payload = 0
    profiler = StageTimer()
    cpu_threads = configure_cpu_entropy_threads(getattr(args, "cpu_entropy_threads", 0))
    entropy_batch_size = int(getattr(args, "entropy_batch_size", 0) or cfg.tile_batch)
    print(
        f"[Fast] entropy_batch_size={entropy_batch_size} "
        f"cpu_entropy_threads={cpu_threads}"
    )
    t_start = time.time()
    with open(out, "wb") as archive:
        archive.write(MAGIC)
        write_json(archive, header)
        for index, path in enumerate(files):
            frame_start = time.time()
            _t = time.perf_counter()
            raw = tifffile.imread(path).squeeze()
            norm = global_log_normalize(raw, stats)
            x = torch.from_numpy(norm)[None, None].to(DEVICE, non_blocking=True)
            _cuda_sync()
            profiler.add("read_normalize_h2d", time.perf_counter() - _t)

            records, payload = compress_image_tiles(
                model,
                entropy_model,
                x,
                cfg,
                profiler=profiler,
                entropy_batch_size=entropy_batch_size,
            )
            meta = {
                "filename": os.path.basename(path),
                "index": index,
                "source_index": start_frame + index,
                "height": int(raw.shape[0]),
                "width": int(raw.shape[1]),
                "dtype": np.dtype(raw.dtype).str,
            }
            _t = time.perf_counter()
            write_frame_record(archive, meta, records)
            profiler.add("archive_write", time.perf_counter() - _t)
            total_raw += os.path.getsize(path)
            total_payload += payload

            if (index + 1) % args.log_every == 0 or index == len(files) - 1:
                archive.flush()
                size = archive.tell()
                elapsed = max(time.time() - t_start, 1e-8)
                print(
                    f"[{index+1:5d}/{len(files)}] payload={payload/1024:.1f} KiB "
                    f"archive_ratio={total_raw/max(size,1):.2f}x "
                    f"speed={(index+1)/elapsed:.3f} frame/s last={time.time()-frame_start:.2f}s",
                    flush=True,
                )

    final_size = out.stat().st_size
    elapsed = max(time.time() - t_start, 1e-8)
    print(
        f"[Done] raw={total_raw/1e9:.3f} GB archive={final_size/1e6:.2f} MB "
        f"ratio={total_raw/final_size:.2f}x speed={len(files)/elapsed:.3f} frame/s"
    )
    print("[Profile] compression stage timing:")
    profile = profiler.report()
    for name, item in profile.items():
        print(
            f"  {name:36s} {item['seconds']:8.3f}s "
            f"{item['percent']:5.1f}% calls={item['calls']}"
        )


def reconstruct_command(args: argparse.Namespace) -> None:
    configure_decode_profile("native")
    model, cfg, _ = load_checkpoint(args.checkpoint)
    model.eval()
    model.update(force=True)
    entropy_model = make_cpu_entropy_model(model)
    print("[Entropy] batched deterministic CPU hyperprior/index path enabled")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.archive, "rb") as archive:
        if archive.read(len(MAGIC)) != MAGIC:
            raise ValueError("Not an HIWH archive")
        header = read_json(archive)
        stats = header["stats"]
        frame_count = int(header["frame_count"])
        entropy_decode_bs = int(
            getattr(args, "entropy_decode_batch_size", 0)
            or header.get("entropy_batch_size", 32)
        )
        synthesis_bs = int(
            getattr(args, "synthesis_batch_size", 0)
            or cfg.tile_batch
        )
        if args.target_index is not None and not (0 <= args.target_index < frame_count):
            raise ValueError(f"target_index must be in [0,{frame_count-1}]")

        for index in range(frame_count):
            meta, records = read_frame_record(archive)
            if args.target_index is not None and index != args.target_index:
                continue
            recon = decompress_image_tiles(
                model, entropy_model, records, int(meta["height"]), int(meta["width"]), cfg,
                decode_retries=args.decode_retries,
                entropy_decode_batch_size=entropy_decode_bs,
                synthesis_batch_size=synthesis_bs,
            )
            recon_np = recon[0, 0].cpu().numpy()
            require_finite_array(recon_np, f"normalized reconstruction for {meta['filename']}")
            raw_recon = global_log_denormalize(
                recon_np, stats, np.dtype(meta["dtype"])
            )
            require_finite_array(raw_recon, f"raw reconstruction for {meta['filename']}")
            out_path = out_dir / f"reconstructed_{meta['filename']}"
            tifffile.imwrite(out_path, raw_recon)
            print(f"[Saved] {out_path}")
            if args.target_index is not None:
                break


def evaluate_target_command(args: argparse.Namespace) -> None:
    configure_decode_profile("native")
    model, cfg, _ = load_checkpoint(args.checkpoint)
    model.eval()
    model.update(force=True)
    entropy_model = make_cpu_entropy_model(model)
    files = list_images(args.data_dir, cfg.image_glob)

    with open(args.archive, "rb") as archive:
        if archive.read(len(MAGIC)) != MAGIC:
            raise ValueError("Not an HIWH archive")
        header = read_json(archive)
        stats = header["stats"]
        frame_count = int(header["frame_count"])
        entropy_decode_bs = int(
            getattr(args, "entropy_decode_batch_size", 0)
            or header.get("entropy_batch_size", 32)
        )
        synthesis_bs = int(
            getattr(args, "synthesis_batch_size", 0)
            or cfg.tile_batch
        )
        if not (0 <= args.target_index < frame_count):
            raise ValueError(f"target_index must be in [0,{frame_count-1}]")
        meta = None
        records = None
        for index in range(frame_count):
            m, r = read_frame_record(archive)
            if index == args.target_index:
                meta, records = m, r
                break
        assert meta is not None and records is not None

    recon_norm_t = decompress_image_tiles(
        model, entropy_model, records, int(meta["height"]), int(meta["width"]), cfg,
        decode_retries=args.decode_retries,
        entropy_decode_batch_size=entropy_decode_bs,
        synthesis_batch_size=synthesis_bs,
    )
    recon_norm = recon_norm_t[0, 0].cpu().numpy()
    require_finite_array(recon_norm, f"normalized reconstruction for {meta['filename']}")
    source_index = int(meta["source_index"])
    raw_path = files[source_index]
    raw = tifffile.imread(raw_path).squeeze()
    raw_norm = global_log_normalize(raw, stats)
    recon_raw = global_log_denormalize(recon_norm, stats, np.dtype(meta["dtype"]))
    require_finite_array(recon_raw, f"raw reconstruction for {meta['filename']}")

    metrics = normalized_metrics(raw_norm, recon_norm)
    metrics.update(raw_metrics(raw, recon_raw))
    metrics["archive_size_bytes"] = os.path.getsize(args.archive)
    metrics["source_index"] = source_index
    metrics["filename"] = meta["filename"]
    print(json.dumps(metrics, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"reconstructed_{meta['filename']}"
        tifffile.imwrite(out_path, recon_raw)
        print(f"[Saved] {out_path}")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Wavelet Hyperprior codec for X-ray holograms"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("compute-stats")
    stats.add_argument("--data-dir", required=True)
    stats.add_argument("--output", required=True)
    stats.add_argument("--image-glob", default="*.tiff")
    stats.set_defaults(func=lambda a: compute_global_stats(list_images(a.data_dir, a.image_glob), a.output))

    pool = sub.add_parser("precompute-sampling-pool")
    pool.add_argument("--data-dir", required=True)
    pool.add_argument("--stats", required=True)
    pool.add_argument("--output", required=True)
    pool.add_argument("--image-glob", default="*.tiff")
    pool.add_argument("--crop-size", type=int, default=256)
    pool.add_argument("--pool-size", type=int, default=128)
    pool.add_argument("--downsample", type=int, default=4)
    pool.add_argument("--uniform-floor", type=float, default=0.15)
    pool.add_argument("--seed", type=int, default=42)
    pool.set_defaults(
        func=lambda a: precompute_sampling_pool(
            list_images(a.data_dir, a.image_glob),
            load_stats(a.stats),
            a.output,
            crop_size=a.crop_size,
            pool_size=a.pool_size,
            downsample=a.downsample,
            uniform_floor=a.uniform_floor,
            seed=a.seed,
        )
    )

    train = sub.add_parser("train")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--stats", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--image-glob", default="*.tiff")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--crop-size", type=int, default=256)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--val-batches", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--lambda-rate", type=float, default=0.015)
    train.add_argument("--fringe-sampling-probability", type=float, default=0.40)
    train.add_argument(
        "--sampling-pool",
        help="NPZ produced by precompute-sampling-pool. Required when fringe sampling > 0.",
    )
    train.add_argument(
        "--sampling-jitter", type=int, default=32,
        help="Random ±pixel jitter around precomputed candidate coordinates during training.",
    )
    train.add_argument("--analysis-channels", type=int, default=128)
    train.add_argument("--latent-channels", type=int, default=192)
    train.add_argument("--hyper-channels", type=int, default=128)
    train.add_argument("--fringe-low", type=float, default=0.04)
    train.add_argument("--fringe-high", type=float, default=0.28)
    train.add_argument("--tile-size", type=int, default=256)
    train.add_argument("--tile-overlap", type=int, default=32)
    train.add_argument("--tile-batch", type=int, default=8)
    train.set_defaults(func=train_command)

    comp = sub.add_parser("compress-dataset")
    comp.add_argument("--data-dir", required=True)
    comp.add_argument("--stats", required=True)
    comp.add_argument("--checkpoint", required=True)
    comp.add_argument("--archive", required=True)
    comp.add_argument("--start-frame", type=int, default=0)
    comp.add_argument("--max-frames", type=int)
    comp.add_argument("--tile-batch", type=int)
    comp.add_argument("--log-every", type=int, default=10)
    comp.add_argument("--overwrite", action="store_true")

    comp.add_argument(
        "--entropy-batch-size",
        type=int,
        default=32,
        help="Number of tiles processed together by deterministic CPU entropy path (default: 32).",
    )
    comp.add_argument(
        "--cpu-entropy-threads",
        type=int,
        default=8,
        help="PyTorch CPU threads used by h_a/h_s during entropy coding (default: 8).",
    )

    comp.set_defaults(func=compress_dataset_command)

    recon = sub.add_parser("reconstruct")
    recon.add_argument("--checkpoint", required=True)
    recon.add_argument("--archive", required=True)
    recon.add_argument("--output-dir", required=True)
    recon.add_argument("--target-index", type=int)
    recon.add_argument(
        "--entropy-decode-batch-size", type=int, default=0,
        help="CPU entropy decode batch size; use 32 for existing V5.4 archives encoded with --entropy-batch-size 32.",
    )
    recon.add_argument(
        "--synthesis-batch-size", type=int, default=0,
        help="GPU synthesis batch size; 0 uses checkpoint tile_batch.",
    )
    recon.add_argument(
        "--decode-retries", type=int, default=3,
        help="Retry a tile decode if any intermediate tensor becomes NaN/Inf (default: 3).",
    )
    recon.set_defaults(func=reconstruct_command)

    ev = sub.add_parser("evaluate-target")
    ev.add_argument("--data-dir", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--archive", required=True)
    ev.add_argument("--target-index", type=int, required=True)
    ev.add_argument("--output-dir")
    ev.add_argument(
        "--entropy-decode-batch-size", type=int, default=0,
        help="CPU entropy decode batch size; use 32 for existing V5.4 archives encoded with --entropy-batch-size 32.",
    )
    ev.add_argument(
        "--synthesis-batch-size", type=int, default=0,
        help="GPU synthesis batch size; 0 uses checkpoint tile_batch.",
    )
    ev.add_argument(
        "--decode-retries", type=int, default=3,
        help="Retry a tile decode if any intermediate tensor becomes NaN/Inf (default: 3).",
    )
    ev.set_defaults(func=evaluate_target_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
