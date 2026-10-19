"""commands extracted from the supplied v5.5 codec."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from holozip.codec.archive import (
    read_frame_record,
    read_json,
    write_frame_record,
    write_json,
)
from holozip.codec.entropy import make_cpu_entropy_model
from holozip.codec.pipeline import compress_image_tiles, decompress_image_tiles
from holozip.data.io import list_images
from holozip.data.normalization import (
    global_log_denormalize,
    global_log_normalize,
    load_stats,
)
from holozip.metrics import normalized_metrics, raw_metrics
from holozip.profiling import StageTimer, _cuda_sync, configure_cpu_entropy_threads
from holozip.runtime import (
    DEVICE,
    MAGIC,
    VERSION,
    configure_decode_profile,
    require_finite_array,
)
from holozip.training.checkpoint import load_checkpoint


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
        "entropy_batch_size": int(
            getattr(args, "entropy_batch_size", 0) or cfg.tile_batch
        ),
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
                    f"[{index + 1:5d}/{len(files)}] payload={payload / 1024:.1f} KiB "
                    f"archive_ratio={total_raw / max(size, 1):.2f}x "
                    f"speed={(index + 1) / elapsed:.3f} frame/s last={time.time() - frame_start:.2f}s",
                    flush=True,
                )

    final_size = out.stat().st_size
    elapsed = max(time.time() - t_start, 1e-8)
    print(
        f"[Done] raw={total_raw / 1e9:.3f} GB archive={final_size / 1e6:.2f} MB "
        f"ratio={total_raw / final_size:.2f}x speed={len(files) / elapsed:.3f} frame/s"
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
        synthesis_bs = int(getattr(args, "synthesis_batch_size", 0) or cfg.tile_batch)
        if args.target_index is not None and not (0 <= args.target_index < frame_count):
            raise ValueError(f"target_index must be in [0,{frame_count - 1}]")

        for index in range(frame_count):
            meta, records = read_frame_record(archive)
            if args.target_index is not None and index != args.target_index:
                continue
            recon = decompress_image_tiles(
                model,
                entropy_model,
                records,
                int(meta["height"]),
                int(meta["width"]),
                cfg,
                decode_retries=args.decode_retries,
                entropy_decode_batch_size=entropy_decode_bs,
                synthesis_batch_size=synthesis_bs,
            )
            recon_np = recon[0, 0].cpu().numpy()
            require_finite_array(
                recon_np, f"normalized reconstruction for {meta['filename']}"
            )
            raw_recon = global_log_denormalize(recon_np, stats, np.dtype(meta["dtype"]))
            require_finite_array(
                raw_recon, f"raw reconstruction for {meta['filename']}"
            )
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
        synthesis_bs = int(getattr(args, "synthesis_batch_size", 0) or cfg.tile_batch)
        if not (0 <= args.target_index < frame_count):
            raise ValueError(f"target_index must be in [0,{frame_count - 1}]")
        meta = None
        records = None
        for index in range(frame_count):
            m, r = read_frame_record(archive)
            if index == args.target_index:
                meta, records = m, r
                break
        assert meta is not None and records is not None

    recon_norm_t = decompress_image_tiles(
        model,
        entropy_model,
        records,
        int(meta["height"]),
        int(meta["width"]),
        cfg,
        decode_retries=args.decode_retries,
        entropy_decode_batch_size=entropy_decode_bs,
        synthesis_batch_size=synthesis_bs,
    )
    recon_norm = recon_norm_t[0, 0].cpu().numpy()
    require_finite_array(
        recon_norm, f"normalized reconstruction for {meta['filename']}"
    )
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
