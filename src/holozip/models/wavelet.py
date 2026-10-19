"""models.wavelet extracted from the supplied v5.5 codec."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernels = (
            torch.tensor(
                [
                    [[1, 1], [1, 1]],
                    [[-1, -1], [1, 1]],
                    [[-1, 1], [-1, 1]],
                    [[1, -1], [-1, 1]],
                ],
                dtype=torch.float32,
            )
            / 2.0
        )
        self.register_buffer("analysis_kernel", kernels[:, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.analysis_kernel, stride=2)

    def inverse(self, bands: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(bands, self.analysis_kernel, stride=2)
