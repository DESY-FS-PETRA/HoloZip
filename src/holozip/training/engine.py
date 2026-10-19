"""training.engine extracted from the supplied v5.5 codec."""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader

from holozip.config import Config
from holozip.data.dataset import HologramCropDataset
from holozip.data.io import list_images
from holozip.data.normalization import load_stats
from holozip.data.sampling import load_sampling_pool
from holozip.models.hyperprior import IndependentWaveletHyperprior
from holozip.runtime import DEVICE, seed_everything
from holozip.training.checkpoint import save_checkpoint
from holozip.training.losses import compute_loss


def configure_optimizers(model: IndependentWaveletHyperprior, cfg: Config):
    aux_names = {
        name for name, _ in model.named_parameters() if name.endswith(".quantiles")
    }
    main_params = [
        p for n, p in model.named_parameters() if n not in aux_names and p.requires_grad
    ]
    aux_params = [
        p for n, p in model.named_parameters() if n in aux_names and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(main_params, lr=cfg.learning_rate, weight_decay=1e-5)
    aux_optimizer = torch.optim.Adam(aux_params, lr=cfg.aux_learning_rate)
    return optimizer, aux_optimizer


@torch.no_grad()
def validate(
    model: IndependentWaveletHyperprior, loader: DataLoader, cfg: Config
) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "ssim": 0.0, "psnr": 0.0, "bpp": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= cfg.val_batches:
            break
        batch = batch.to(DEVICE, non_blocking=True)
        out = model(batch)
        loss, stats = compute_loss(out, batch, cfg)
        recon = out["reconstruction"].float()
        mse = F.mse_loss(recon, batch.float()).item()
        sums["loss"] += float(loss)
        sums["bpp"] += stats["bpp"]
        sums["ssim"] += float(
            ms_ssim(recon, batch.float(), data_range=1.0, size_average=True)
        )
        sums["psnr"] += 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        count += 1
    if count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return {k: v / count for k, v in sums.items()}


def train_command(args: argparse.Namespace) -> None:
    cfg = Config(
        image_glob=args.image_glob,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        val_batches=args.val_batches,
        fringe_sampling_probability=args.fringe_sampling_probability,
        analysis_channels=args.analysis_channels,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_batch=args.tile_batch,
        lambda_rate=args.lambda_rate,
        fringe_low=args.fringe_low,
        fringe_high=args.fringe_high,
    )
    cfg.validate()
    seed_everything(cfg.seed)

    files = list_images(args.data_dir, cfg.image_glob)
    stats = load_stats(args.stats)
    split = max(1, int(len(files) * (1.0 - cfg.val_fraction)))
    train_files = files[:split]
    val_files = (
        files[split:] if split < len(files) else files[-max(1, len(files) // 20) :]
    )

    sampling_pool = None
    if cfg.fringe_sampling_probability > 0:
        if not args.sampling_pool:
            raise ValueError(
                "fringe sampling is enabled but --sampling-pool was not provided. "
                "Run precompute-sampling-pool first."
            )
        sampling_pool, pool_meta = load_sampling_pool(args.sampling_pool)
        if int(pool_meta.get("crop_size", cfg.crop_size)) != cfg.crop_size:
            raise ValueError(
                f"Sampling pool crop_size={pool_meta.get('crop_size')} does not match "
                f"training crop_size={cfg.crop_size}"
            )
        print(
            f"[SamplingPool] loaded {args.sampling_pool} "
            f"pool_size={pool_meta.get('pool_size')} downsample={pool_meta.get('downsample')}"
        )

    train_ds = HologramCropDataset(
        train_files,
        stats,
        cfg,
        sampling_pool=sampling_pool,
        deterministic=False,
        sampling_jitter=args.sampling_jitter,
    )
    val_ds = HologramCropDataset(
        val_files,
        stats,
        cfg,
        sampling_pool=sampling_pool,
        deterministic=True,
        sampling_jitter=0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, cfg.num_workers),
        pin_memory=True,
    )

    model = IndependentWaveletHyperprior(cfg).to(DEVICE)
    optimizer, aux_optimizer = configure_optimizers(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and DEVICE.type == "cuda")

    best_ssim = -float("inf")
    patience = 20
    stale = 0
    print(f"[Device] {DEVICE}")
    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}")
    print(
        f"[Sampling] uniform={1 - cfg.fringe_sampling_probability:.2f}, fringe-soft={cfg.fringe_sampling_probability:.2f}"
    )
    print(f"[Model] parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.time()
        sums: Dict[str, float] = {}
        steps = 0
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
                out = model(batch)
                loss, stats_batch = compute_loss(out, batch, cfg)

            if not torch.isfinite(loss):
                print("[Warning] non-finite main loss; batch skipped")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            aux_loss = model.aux_loss()
            if torch.isfinite(aux_loss):
                aux_loss.backward()
                aux_optimizer.step()

            for k, v in stats_batch.items():
                sums[k] = sums.get(k, 0.0) + v
            sums["aux"] = sums.get("aux", 0.0) + float(aux_loss.detach())
            steps += 1

        scheduler.step()
        if steps == 0:
            raise RuntimeError("No valid optimization steps completed")
        train_stats = {k: v / steps for k, v in sums.items()}
        val = validate(model, val_loader, cfg)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"loss={train_stats.get('loss', float('nan')):.5f} "
            f"bpp={train_stats.get('bpp', float('nan')):.3f} "
            f"valSSIM={val['ssim']:.5f} valPSNR={val['psnr']:.2f} "
            f"valBPP={val['bpp']:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
            f"time={elapsed:.0f}s"
        )

        if val["ssim"] > best_ssim + 1e-4:
            best_ssim = val["ssim"]
            stale = 0
            save_checkpoint(args.checkpoint, model, cfg, args.stats, epoch, best_ssim)
            print(f"  [best] saved {args.checkpoint}")
        else:
            stale += 1
            if stale >= patience:
                print(f"[Early stop] no validation improvement for {patience} epochs")
                break
