"""Evaluate real archive records; model bytes and timing scopes stay explicit."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
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
from holozip.evaluation.plots import plot_comparison
from holozip.evaluation.selection import select_indices
from holozip.metrics import normalized_metrics, raw_metrics
from holozip.profiling import _cuda_sync, configure_cpu_entropy_threads
from holozip.runtime import (
    DEVICE,
    MAGIC,
    VERSION,
    configure_decode_profile,
    require_finite_array,
)
from holozip.training.checkpoint import load_checkpoint


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Use JSON null for non-finite metrics (e.g. perfect-image PSNR)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def prepare_output(path):
    folder = Path(path)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {folder}. Choose a new --output-dir."
        )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def setup_model(checkpoint, threads):
    start = time.perf_counter()
    configure_decode_profile("native")
    # Older PyTorch/OpenMP builds can abort on repeated pool reconfiguration
    # after native libraries have initialized. Do not call a native setter for
    # an already-correct value. A value of zero preserves the current setting.
    if threads is not None and threads > 0 and torch.get_num_threads() != threads:
        configure_cpu_entropy_threads(threads)
    model, cfg, _ = load_checkpoint(checkpoint)
    model.eval()
    model.update(force=True)
    entropy = make_cpu_entropy_model(model)
    _cuda_sync()
    return model, cfg, entropy, time.perf_counter() - start


def benchmark_command(args):
    if args.entropy_batch_size < 1 or args.synthesis_batch_size < 0:
        raise ValueError(
            "Entropy batch size must be positive; synthesis batch size must be nonnegative"
        )
    folder = prepare_output(args.output_dir)
    model, cfg, entropy, setup_seconds = setup_model(
        args.checkpoint, args.cpu_entropy_threads
    )
    stats = load_stats(args.stats)
    files = list_images(args.data_dir, cfg.image_glob)
    selected = select_indices(len(files), args.count, args.indices)
    if args.profile_row is not None and args.profile_row < 0:
        raise ValueError("profile-row must be nonnegative")
    archive_path = folder / "selected.hiwh"
    checkpoint_hash = sha256_file(args.checkpoint)
    header = {
        "version": VERSION,
        "frame_count": len(selected),
        "source_start_frame": selected[0],
        "source_dataset_frame_count": len(files),
        "filenames": [Path(files[i]).name for i in selected],
        "selected_source_indices": selected,
        "config": dataclasses.asdict(cfg),
        "stats": stats,
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256": checkpoint_hash,
        "entropy_batch_size": args.entropy_batch_size,
    }
    encoding = {}
    start = time.perf_counter()
    with open(archive_path, "wb") as archive:
        archive.write(MAGIC)
        write_json(archive, header)
        for number, index in enumerate(selected):
            _cuda_sync()
            frame_start = time.perf_counter()
            raw = tifffile.imread(files[index]).squeeze()
            if raw.ndim != 2 or min(raw.shape) < 11:
                raise ValueError(
                    f"Expected 2-D TIFF at least 11 pixels per side: {files[index]}"
                )
            require_finite_array(raw, f"input {files[index]}")
            normalized = global_log_normalize(raw, stats)
            x = torch.from_numpy(normalized)[None, None].to(DEVICE)
            records, payload = compress_image_tiles(
                model, entropy, x, cfg, entropy_batch_size=args.entropy_batch_size
            )
            meta = {
                "filename": Path(files[index]).name,
                "index": number,
                "source_index": index,
                "height": int(raw.shape[0]),
                "width": int(raw.shape[1]),
                "dtype": np.dtype(raw.dtype).str,
            }
            position = archive.tell()
            write_frame_record(archive, meta, records)
            _cuda_sync()
            encoding[index] = {
                "encode_frame_seconds": time.perf_counter() - frame_start,
                "entropy_payload_bytes": payload,
                "record_bytes": archive.tell() - position,
            }
            print(
                f"[Encode {number + 1}/{len(selected)}] source_index={index} {meta['filename']}",
                flush=True,
            )
        archive.flush()
    encode_seconds = time.perf_counter() - start
    # Decode from the persisted archive, not from in-memory encoder tensors.
    rows, decode_seconds = evaluate_archive(
        args, folder, archive_path, model, cfg, entropy, encoding
    )
    summary = make_summary(
        args,
        archive_path,
        rows,
        setup_seconds,
        encode_seconds,
        decode_seconds,
        checkpoint_hash,
    )
    save_reports(folder, rows, summary)


def compare_command(args):
    if args.synthesis_batch_size < 0:
        raise ValueError("synthesis-batch-size must be nonnegative")
    folder = prepare_output(args.output_dir)
    model, cfg, entropy, setup_seconds = setup_model(
        args.checkpoint, args.cpu_entropy_threads
    )
    archive = Path(args.archive)
    checkpoint_hash = sha256_file(args.checkpoint)
    # New benchmark archives carry a fingerprint. Legacy archives have no hash.
    with open(archive, "rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("Not an HIWH archive")
        header = read_json(stream)
    if header.get("checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError("Checkpoint SHA-256 does not match this archive")
    rows, decode_seconds = evaluate_archive(args, folder, archive, model, cfg, entropy)
    summary = make_summary(
        args, archive, rows, setup_seconds, None, decode_seconds, checkpoint_hash
    )
    summary["archive_model_hash_verified"] = "checkpoint_sha256" in header
    save_reports(folder, rows, summary)


def evaluate_archive(args, folder, archive_path, model, cfg, entropy, encoding=None):
    rows = []
    decode_total = 0.0
    with open(archive_path, "rb") as archive:
        if archive.read(len(MAGIC)) != MAGIC:
            raise ValueError("Not an HIWH archive")
        header = read_json(archive)
        stats = header["stats"]
        count = int(header["frame_count"])
        batch = int(header.get("entropy_batch_size", 32))
        if batch < 1:
            raise ValueError("Archive has invalid entropy batch size")
        for archive_index in range(count):
            _cuda_sync()
            start = time.perf_counter()
            position = archive.tell()
            meta, records = read_frame_record(archive)
            record_bytes = archive.tell() - position
            name = meta["filename"]
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Archive filename must be a basename")
            decoded = decompress_image_tiles(
                model,
                entropy,
                records,
                int(meta["height"]),
                int(meta["width"]),
                cfg,
                decode_retries=3,
                entropy_decode_batch_size=batch,
                synthesis_batch_size=args.synthesis_batch_size or cfg.tile_batch,
            )
            normalized = decoded[0, 0].cpu().numpy()
            require_finite_array(normalized, f"decode {name}")
            raw_decoded = global_log_denormalize(
                normalized, stats, np.dtype(meta["dtype"])
            )
            require_finite_array(raw_decoded, f"denormalized decode {name}")
            _cuda_sync()
            seconds = time.perf_counter() - start
            decode_total += seconds
            # Source read, metrics, plotting and TIFF export are outside decode timing.
            source = Path(args.data_dir) / name
            raw = tifffile.imread(source).squeeze()
            require_finite_array(raw, f"reference {name}")
            if (
                raw.shape != raw_decoded.shape
                or np.dtype(raw.dtype).str != meta["dtype"]
            ):
                raise ValueError(f"Source shape/dtype mismatch for {name}")
            reference_normalized = global_log_normalize(raw, stats)
            row = {
                "archive_index": archive_index,
                "source_index": int(meta["source_index"]),
                "filename": name,
                "height": raw.shape[0],
                "width": raw.shape[1],
                "dtype": str(raw.dtype),
                "source_tiff_bytes": source.stat().st_size,
                "source_pixel_bytes": raw.nbytes,
                "pixels": raw.size,
                "record_bytes": record_bytes,
                "record_bpp": 8 * record_bytes / raw.size,
                "entropy_payload_bytes": sum(
                    len(r["y"]) + len(r["z"]) for r in records
                ),
                "decode_frame_seconds": seconds,
            }
            row.update(normalized_metrics(reference_normalized, normalized))
            row.update(raw_metrics(raw, raw_decoded))
            row["rmse"] = math.sqrt(row["mse"])
            row["raw_rmse"] = math.sqrt(row["raw_mse"])
            row["mean_change_percent"] = (row["mean_ratio"] - 1) * 100
            if encoding:
                row.update(encoding[row["source_index"]])
            stem = f"{archive_index:04d}_source_{row['source_index']:06d}"
            if not args.no_plots:
                plot_comparison(
                    raw,
                    raw_decoded,
                    folder / "comparisons" / (stem + ".png"),
                    f"{name} | normalized SSIM={row['ssim']:.5f}, PSNR={row['psnr']:.2f} dB",
                    args.profile_row,
                )
                line = (
                    raw.shape[0] // 2 if args.profile_row is None else args.profile_row
                )
                if not 0 <= line < raw.shape[0]:
                    raise ValueError("profile-row is outside the image")
                profile_path = folder / "profiles" / (stem + ".csv")
                profile_path.parent.mkdir(exist_ok=True)
                profile = np.column_stack(
                    (
                        np.arange(raw.shape[1]),
                        raw[line],
                        raw_decoded[line],
                        raw_decoded[line].astype(np.float64)
                        - raw[line].astype(np.float64),
                    )
                )
                np.savetxt(
                    profile_path,
                    profile,
                    delimiter=",",
                    header="column,original,decoded,signed_error",
                    comments="",
                )
            if args.save_decoded:
                output = folder / "decoded"
                output.mkdir(exist_ok=True)
                tifffile.imwrite(output / ("reconstructed_" + name), raw_decoded)
            rows.append(row)
            print(
                f"[Evaluate {archive_index + 1}/{count}] {name} SSIM={row['ssim']:.5f} PSNR={row['psnr']:.2f} dB",
                flush=True,
            )
    return rows, decode_total


def make_summary(
    args,
    archive_path,
    rows,
    setup_seconds,
    encode_seconds,
    decode_seconds,
    checkpoint_hash,
):
    if not rows:
        raise ValueError("Archive contains no frames")
    archive_bytes = Path(archive_path).stat().st_size
    model_bytes = Path(args.checkpoint).stat().st_size
    tiff_bytes = sum(r["source_tiff_bytes"] for r in rows)
    pixel_bytes = sum(r["source_pixel_bytes"] for r in rows)
    pixels = sum(r["pixels"] for r in rows)
    metric_keys = [
        "ssim",
        "psnr",
        "mse",
        "mae",
        "rmse",
        "max_error",
        "raw_psnr",
        "raw_mae",
        "raw_rmse",
        "raw_max_error",
        "mean_change_percent",
    ]
    aggregates = {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "min": float(min(r[k] for r in rows)),
            "max": float(max(r[k] for r in rows)),
        }
        for k in metric_keys
    }
    versions = {}
    for name in [
        "holozip",
        "torch",
        "compressai",
        "numpy",
        "tifffile",
        "scikit-image",
        "matplotlib",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return {
        "schema_version": 1,
        "mode": "benchmark" if encode_seconds is not None else "compare",
        "frame_count": len(rows),
        "source_indices": [r["source_index"] for r in rows],
        "checkpoint_sha256": checkpoint_hash,
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_bytes,
        "checkpoint_bytes": model_bytes,
        "source_tiff_bytes": tiff_bytes,
        "source_pixel_bytes": pixel_bytes,
        "archive_bpp": 8 * archive_bytes / pixels,
        "tiff_to_archive_ratio": tiff_bytes / archive_bytes,
        "pixel_to_archive_ratio": pixel_bytes / archive_bytes,
        "tiff_to_archive_plus_checkpoint_ratio": tiff_bytes
        / (archive_bytes + model_bytes),
        "archive_plus_checkpoint_bpp": 8 * (archive_bytes + model_bytes) / pixels,
        "model_setup_seconds": setup_seconds,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "encode_frames_per_second": None
        if encode_seconds is None
        else len(rows) / encode_seconds,
        "decode_frames_per_second": len(rows) / decode_seconds
        if decode_seconds > 0
        else None,
        "timing_scope": {
            "encode": "archive header + source TIFF read, normalization, H2D, coding, record writes and close; excludes setup and plotting; no fsync",
            "decode": "sum of frame-record reads, decode, D2H and denormalization; excludes model setup, source reads, metrics, TIFF export and plots",
            "warmup": "no warmup; cold-start effects included; short runs are not steady-state throughput",
        },
        "quality_scope": "per-image means in normalized and raw domains; normalized SSIM is single-scale Gaussian-window SSIM, not MS-SSIM; no downstream phase-retrieval validation",
        "metrics": aggregates,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "versions": versions,
        "source_integrity": "names, shape and dtype checked; original source content not fingerprinted in legacy archives",
        "nonfinite_json_policy": "non-finite metrics are null in JSON (e.g. perfect-match PSNR); CSV retains inf",
    }


def save_reports(folder, rows, summary):
    with open(folder / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(folder / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)
    (folder / "README.txt").write_text(
        "metrics.csv: per-frame normalized/raw metrics and record sizes.\nsummary.json: total archive size includes the header; checkpoint cost is also reported separately.\ncomparisons/: shared-scale images, absolute errors, center detail and raw line profiles.\nprofiles/: numeric profile CSVs. Ratios are for this selected subset only.\nTiming excludes diagnostic plotting and quality metrics. See summary timing_scope.\n",
        encoding="utf-8",
    )
    print(f"[Saved] {folder / 'metrics.csv'}\n[Saved] {folder / 'summary.json'}")
    print(
        f"[Summary] n={len(rows)} ratio={summary['tiff_to_archive_ratio']:.2f}x meanSSIM={summary['metrics']['ssim']['mean']:.5f} meanPSNR={summary['metrics']['psnr']['mean']:.2f} dB"
    )
