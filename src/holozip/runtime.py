"""runtime extracted from the supplied v5.5 codec."""

from __future__ import annotations

import random

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

MAGIC = b"HIW52\x00\x00\x00"
VERSION = 55


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
