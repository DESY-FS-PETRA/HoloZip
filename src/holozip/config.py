"""config extracted from the supplied v5.5 codec."""

from __future__ import annotations

import dataclasses


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
