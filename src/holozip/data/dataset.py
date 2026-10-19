"""data.dataset extracted from the supplied v5.5 codec."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from holozip.config import Config
from holozip.data.normalization import global_log_normalize


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
            raise ValueError("crop_size exceeds image dimensions")
        if cfg.fringe_sampling_probability > 0 and sampling_pool is None:
            raise ValueError(
                "fringe_sampling_probability > 0 requires a precomputed sampling pool. "
                "Run the precompute-sampling-pool command and pass --sampling-pool."
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
            result = np.asarray(arr[top : top + crop, left : left + crop])
            if result.shape == (crop, crop):
                return result
        except Exception:
            pass
        raw = tifffile.imread(path).squeeze()
        return raw[top : top + crop, left : left + crop]

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
