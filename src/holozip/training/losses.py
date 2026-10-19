"""training.losses extracted from the supplied v5.5 codec."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from holozip.config import Config


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


def spectral_losses(
    pred: torch.Tensor, target: torch.Tensor, low: float, high: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        torch.abs(torch.log1p(torch.abs(pred_f)) - torch.log1p(torch.abs(target_f)))
        * band
    ).sum() / (band.sum() * pred.shape[0] + 1e-8)
    return complex_loss, phase_loss, band_loss


def compute_loss(
    output: Dict[str, Any], target: torch.Tensor, cfg: Config
) -> Tuple[torch.Tensor, Dict[str, float]]:
    recon = output["reconstruction"]
    pixels = target.shape[0] * target.shape[-2] * target.shape[-1]
    bpp = rate_bpp(output["likelihoods"], pixels)
    l_charb = charbonnier(recon, target)
    l_ms = 1.0 - ms_ssim(
        recon.float(), target.float(), data_range=1.0, size_average=True
    )
    l_complex, l_phase, l_band = spectral_losses(
        recon, target, cfg.fringe_low, cfg.fringe_high
    )
    l_grad = gradient_loss(recon, target)
    l_wavelet = F.l1_loss(output["bands_hat"], output["bands"])
    pre = output["preclamp"]
    l_range = F.relu(pre - 1.0).mean() + F.relu(-pre).mean()

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
