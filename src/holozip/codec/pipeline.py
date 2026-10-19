"""codec.pipeline extracted from the supplied v5.5 codec."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from holozip.codec.tiling import hann_window_2d, pad_tile, tile_coordinates
from holozip.config import Config
from holozip.models.hyperprior import IndependentWaveletHyperprior
from holozip.profiling import StageTimer, _cuda_sync
from holozip.runtime import DEVICE, require_finite_tensor


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
        coords = coords_all[start : start + ebs]
        tiles: List[torch.Tensor] = []
        sizes: List[Tuple[int, int]] = []

        for top, left in coords:
            h = min(cfg.tile_size, height - top)
            w = min(cfg.tile_size, width - left)
            sizes.append((h, w))
            tiles.append(
                pad_tile(
                    image[..., top : top + h, left : left + w],
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
        if profiler:
            profiler.add("gpu_analysis_wavelet_g_a", time.perf_counter() - _t)

        # One transfer per tile batch rather than one transfer per tile.
        _t = time.perf_counter()
        y_cpu = y_gpu.float().cpu().contiguous()
        require_finite_tensor(y_cpu, f"encode batch {start} CPU y")
        if profiler:
            profiler.add("gpu_to_cpu_y", time.perf_counter() - _t)

        # Entire deterministic entropy/hyperprior path runs batched on CPU.
        _t = time.perf_counter()
        z_cpu = entropy_model.h_a(torch.abs(y_cpu))
        require_finite_tensor(z_cpu, f"encode batch {start} CPU z")
        if profiler:
            profiler.add("cpu_h_a", time.perf_counter() - _t)

        _t = time.perf_counter()
        z_strings = entropy_model.entropy_bottleneck.compress(z_cpu)
        if profiler:
            profiler.add("entropy_bottleneck_compress", time.perf_counter() - _t)
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
        if profiler:
            profiler.add(
                "entropy_bottleneck_roundtrip_decode", time.perf_counter() - _t
            )

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
        if profiler:
            profiler.add("cpu_h_s", time.perf_counter() - _t)

        _t = time.perf_counter()
        indexes = entropy_model.gaussian_conditional.build_indexes(scales_hat)
        if profiler:
            profiler.add("gaussian_build_indexes", time.perf_counter() - _t)

        _t = time.perf_counter()
        y_strings = entropy_model.gaussian_conditional.compress(
            y_cpu,
            indexes,
            means=means_hat,
        )
        if profiler:
            profiler.add("gaussian_compress", time.perf_counter() - _t)

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
    y_hat_cpu = (
        entropy_model.gaussian_conditional.decompress(
            y_strings,
            indexes,
            means=means_hat,
        )
        .float()
        .contiguous()
    )
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
        entropy_records = list(records[entropy_start : entropy_start + entropy_bs])
        shapes = {tuple(int(v) for v in r["shape"]) for r in entropy_records}
        if len(shapes) != 1:
            raise ValueError(f"Incompatible latent shapes in entropy batch: {shapes}")
        shape = next(iter(shapes))
        z_strings = [r["z"] for r in entropy_records]
        y_strings = [r["y"] for r in entropy_records]

        z_hat_cpu = entropy_model.entropy_bottleneck.decompress(z_strings, shape)
        require_finite_tensor(
            z_hat_cpu,
            f"tiles {entropy_start}:{entropy_start + len(entropy_records)} CPU z_hat",
        )
        gaussian_params = entropy_model.h_s(z_hat_cpu)
        require_finite_tensor(
            gaussian_params,
            f"tiles {entropy_start}:{entropy_start + len(entropy_records)} CPU hyper parameters",
        )
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 1e-6
        require_finite_tensor(
            scales_hat,
            f"tiles {entropy_start}:{entropy_start + len(entropy_records)} CPU scales",
        )
        require_finite_tensor(
            means_hat,
            f"tiles {entropy_start}:{entropy_start + len(entropy_records)} CPU means",
        )
        indexes = entropy_model.gaussian_conditional.build_indexes(scales_hat)
        y_hat_cpu = (
            entropy_model.gaussian_conditional.decompress(
                y_strings, indexes, means=means_hat
            )
            .float()
            .contiguous()
        )
        require_finite_tensor(
            y_hat_cpu,
            f"tiles {entropy_start}:{entropy_start + len(entropy_records)} CPU y_hat",
        )

        max_abs_per_tile = y_hat_cpu.abs().flatten(1).amax(dim=1)
        bad = torch.nonzero(max_abs_per_tile > 1.0e4, as_tuple=False).flatten()
        if bad.numel():
            details = ", ".join(
                f"tile {entropy_start + int(i)} max_abs={float(max_abs_per_tile[i]):.3g}"
                for i in bad
            )
            raise RuntimeError(
                "Implausible decoded latent(s): "
                + details
                + ". Check --entropy-decode-batch-size against compression --entropy-batch-size."
            )

        for local_start in range(0, len(entropy_records), synth_bs):
            local_end = min(local_start + synth_bs, len(entropy_records))
            synth_records = entropy_records[local_start:local_end]
            y_hat_gpu = y_hat_cpu[local_start:local_end].to(DEVICE, non_blocking=False)
            bands_hat = model.g_s(y_hat_gpu)
            require_finite_tensor(
                bands_hat,
                f"tiles {entropy_start + local_start}:{entropy_start + local_end} GPU synthesis bands",
            )
            preclamp = model.wavelet.inverse(bands_hat)
            require_finite_tensor(
                preclamp,
                f"tiles {entropy_start + local_start}:{entropy_start + local_end} inverse wavelet",
            )
            recon_batch = torch.clamp(preclamp, 0.0, 1.0)
            require_finite_tensor(
                recon_batch,
                f"tiles {entropy_start + local_start}:{entropy_start + local_end} reconstruction",
            )

            for j, r in enumerate(synth_records):
                top, left = int(r["top"]), int(r["left"])
                h, w = int(r["height"]), int(r["width"])
                tile = recon_batch[j : j + 1, :, :h, :w]
                tile_window = window[..., :h, :w]
                acc[..., top : top + h, left : left + w] += tile * tile_window
                weights[..., top : top + h, left : left + w] += tile_window

    require_finite_tensor(acc, "full-image accumulator")
    require_finite_tensor(weights, "full-image blending weights")
    if torch.any(weights <= 0):
        raise RuntimeError(
            f"Uncovered pixels during tile blending: {int((weights <= 0).sum())}"
        )
    reconstruction = torch.clamp(acc / weights, 0.0, 1.0)
    require_finite_tensor(reconstruction, "final normalized reconstruction")
    return reconstruction
