"""cli extracted from the supplied v5.5 codec."""

from __future__ import annotations

import argparse

from holozip.commands import (
    compress_dataset_command,
    evaluate_target_command,
    reconstruct_command,
)
from holozip.data.io import list_images
from holozip.data.normalization import compute_global_stats, load_stats
from holozip.data.sampling import precompute_sampling_pool
from holozip.training.engine import train_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Wavelet Hyperprior codec for X-ray holograms"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("compute-stats")
    stats.add_argument("--data-dir", required=True)
    stats.add_argument("--output", required=True)
    stats.add_argument("--image-glob", default="*.tiff")
    stats.set_defaults(
        func=lambda a: compute_global_stats(
            list_images(a.data_dir, a.image_glob), a.output
        )
    )

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
        "--sampling-jitter",
        type=int,
        default=32,
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
        "--entropy-decode-batch-size",
        type=int,
        default=0,
        help="CPU entropy decode batch size; use 32 for existing V5.4 archives encoded with --entropy-batch-size 32.",
    )
    recon.add_argument(
        "--synthesis-batch-size",
        type=int,
        default=0,
        help="GPU synthesis batch size; 0 uses checkpoint tile_batch.",
    )
    recon.add_argument(
        "--decode-retries",
        type=int,
        default=3,
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
        "--entropy-decode-batch-size",
        type=int,
        default=0,
        help="CPU entropy decode batch size; use 32 for existing V5.4 archives encoded with --entropy-batch-size 32.",
    )
    ev.add_argument(
        "--synthesis-batch-size",
        type=int,
        default=0,
        help="GPU synthesis batch size; 0 uses checkpoint tile_batch.",
    )
    ev.add_argument(
        "--decode-retries",
        type=int,
        default=3,
        help="Retry a tile decode if any intermediate tensor becomes NaN/Inf (default: 3).",
    )
    ev.set_defaults(func=evaluate_target_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
