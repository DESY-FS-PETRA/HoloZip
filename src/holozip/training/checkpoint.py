"""training.checkpoint extracted from the supplied v5.5 codec."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from holozip.config import Config
from holozip.models.hyperprior import IndependentWaveletHyperprior
from holozip.runtime import DEVICE, VERSION


def save_checkpoint(
    path: str,
    model: IndependentWaveletHyperprior,
    cfg: Config,
    stats_path: str,
    epoch: int,
    best_ssim: float,
) -> None:
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


def load_checkpoint(
    path: str,
) -> Tuple[IndependentWaveletHyperprior, Config, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    cfg.validate()
    model = IndependentWaveletHyperprior(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    return model, cfg, ckpt
