"""models.hyperprior extracted from the supplied v5.5 codec."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

from holozip.config import Config
from holozip.models.wavelet import HaarWavelet


def conv(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)


def deconv(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Module:
    return nn.ConvTranspose2d(
        in_ch,
        out_ch,
        kernel,
        stride=stride,
        padding=kernel // 2,
        output_padding=stride - 1,
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
            conv(4, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 4),
        )
        self.h_a = nn.Sequential(
            nn.Conv2d(M, H, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            conv(H, H, 5, 2),
            nn.LeakyReLU(inplace=True),
            conv(H, H, 5, 2),
        )
        self.h_s = nn.Sequential(
            deconv(H, H, 5, 2),
            nn.LeakyReLU(inplace=True),
            deconv(H, H * 2, 5, 2),
            nn.LeakyReLU(inplace=True),
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
    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int]
    ) -> torch.Tensor:
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, indexes, means=means_hat
        )
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
