"""Real CPU entropy coding with small random models: mechanics, not quality."""

import dataclasses
import io

import numpy as np
import pytest
import tifffile
import torch

from holozip.cli import build_parser
from holozip.codec.archive import read_frame_record, write_frame_record
from holozip.codec.entropy import make_cpu_entropy_model
from holozip.codec.pipeline import compress_image_tiles, decompress_image_tiles
from holozip.config import Config
from holozip.models.hyperprior import IndependentWaveletHyperprior
from holozip.runtime import DEVICE
from holozip.training.checkpoint import load_checkpoint, save_checkpoint
from holozip.training.engine import configure_optimizers
from holozip.training.losses import compute_loss


@pytest.fixture
def model():
    torch.manual_seed(42)
    cfg = Config(analysis_channels=8, latent_channels=8, hyper_channels=8, tile_batch=2)
    model = IndependentWaveletHyperprior(cfg).to(DEVICE).eval()
    model.update(force=True)
    return model


@pytest.mark.integration
@pytest.mark.parametrize("shape", [(256, 256), (257, 300)])
def test_actual_codec_roundtrip(model, original, tmp_path, shape):
    entropy = make_cpu_entropy_model(model)
    x = torch.rand(1, 1, *shape, device=DEVICE)
    records, payload = compress_image_tiles(
        model, entropy, x, model.cfg, entropy_batch_size=2
    )
    assert payload > 0
    stream = io.BytesIO()
    write_frame_record(stream, {"height": shape[0], "width": shape[1]}, records)
    stream.seek(0)
    meta, stored = read_frame_record(stream)
    result = decompress_image_tiles(
        model,
        entropy,
        stored,
        *shape,
        model.cfg,
        entropy_decode_batch_size=2,
        synthesis_batch_size=2,
    )
    assert result.shape == x.shape and torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    # Compare complete byte streams against the unmodified implementation.
    ref_cfg = original.Config(**dataclasses.asdict(model.cfg))
    ref = original.IndependentWaveletHyperprior(ref_cfg).to(DEVICE).eval()
    # Tables are regenerated from the same learned parameters on CPU.
    learned = {
        k: v
        for k, v in model.state_dict().items()
        if not k.endswith(
            ("._offset", "._quantized_cdf", "._cdf_length", ".scale_table")
        )
    }
    ref.load_state_dict(learned, strict=False)
    ref.update(force=True)
    ref_entropy = original.make_cpu_entropy_model(ref)
    ref_records, ref_payload = original.compress_image_tiles(
        ref, ref_entropy, x, ref_cfg, entropy_batch_size=2
    )
    assert payload == ref_payload and stored == ref_records
    expected = original.decompress_image_tiles(
        ref,
        ref_entropy,
        ref_records,
        *shape,
        ref_cfg,
        entropy_decode_batch_size=2,
        synthesis_batch_size=2,
    )
    # CPU remains bit-exact. CUDA synthesis can vary by a few float32 ULPs
    # between identical executions because backend kernels need not be bitwise
    # deterministic. This does not relax the exact entropy byte checks above.
    if result.device.type == "cuda":
        torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-6)
    else:
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    # Exercise an actual filesystem TIFF output rather than tensors only.
    p = tmp_path / "decoded.tiff"
    tifffile.imwrite(p, result[0, 0].cpu().numpy())
    np.testing.assert_array_equal(tifffile.imread(p), result[0, 0].cpu().numpy())


def test_checkpoint_roundtrip(tmp_path):
    cfg = Config(analysis_channels=8, latent_channels=8, hyper_channels=8)
    m = IndependentWaveletHyperprior(cfg).to(DEVICE)
    p = str(tmp_path / "model.pth")
    save_checkpoint(p, m, cfg, "stats.json", 1, 0.5)
    restored, rcfg, meta = load_checkpoint(p)
    assert meta["epoch"] == 1 and dataclasses.asdict(rcfg) == dataclasses.asdict(cfg)
    for k, v in m.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[k], v, atol=0, rtol=0)


def test_loss_and_training_step():
    cfg = Config(analysis_channels=8, latent_channels=8, hyper_channels=8)
    m = IndependentWaveletHyperprior(cfg).to(DEVICE).train()
    opt, aux = configure_optimizers(m, cfg)
    x = torch.rand(1, 1, 256, 256, device=DEVICE)
    output = m(x)
    loss, stats = compute_loss(output, x, cfg)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in stats.values())
    before = m.g_a[0].weight.detach().clone()
    opt.zero_grad()
    aux.zero_grad()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    opt.step()
    aux_loss = m.aux_loss()
    assert torch.isfinite(aux_loss)
    aux_loss.backward()
    aux.step()
    assert not torch.equal(before, m.g_a[0].weight)


@pytest.mark.integration
def test_full_dataset_commands(tmp_path):
    data = tmp_path / "input"
    data.mkdir()
    image = (np.random.default_rng(42).random((256, 256)) * 1000).astype(np.uint16)

    for index in range(4):
        tifffile.imwrite(data / f"frame_{index:03d}.tiff", image + index)
    cfg = Config(analysis_channels=8, latent_channels=8, hyper_channels=8)
    model = IndependentWaveletHyperprior(cfg).to(DEVICE)
    ckpt = str(tmp_path / "model.pth")
    stats = str(tmp_path / "stats.json")
    archive = str(tmp_path / "data.hiwh")
    out = str(tmp_path / "decoded")
    save_checkpoint(ckpt, model, cfg, stats, 0, 0)
    commands = [
        ["compute-stats", "--data-dir", str(data), "--output", stats],
        [
            "precompute-sampling-pool",
            "--data-dir",
            str(data),
            "--stats",
            stats,
            "--output",
            str(tmp_path / "pool.npz"),
            "--pool-size",
            "4",
        ],
        [
            "train",
            "--data-dir",
            str(data),
            "--stats",
            stats,
            "--sampling-pool",
            str(tmp_path / "pool.npz"),
            "--checkpoint",
            ckpt,
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--val-batches",
            "1",
            "--analysis-channels",
            "8",
            "--latent-channels",
            "8",
            "--hyper-channels",
            "8",
        ],
        [
            "compress-dataset",
            "--data-dir",
            str(data),
            "--stats",
            stats,
            "--checkpoint",
            ckpt,
            "--archive",
            archive,
            "--entropy-batch-size",
            "2",
            "--cpu-entropy-threads",
            "2",
        ],
        [
            "reconstruct",
            "--checkpoint",
            ckpt,
            "--archive",
            archive,
            "--output-dir",
            out,
        ],
        [
            "evaluate-target",
            "--data-dir",
            str(data),
            "--checkpoint",
            ckpt,
            "--archive",
            archive,
            "--target-index",
            "0",
        ],
    ]
    for command in commands:
        args = build_parser().parse_args(command)
        args.func(args)
    decoded = tifffile.imread(tmp_path / "decoded/reconstructed_frame_000.tiff")
    assert decoded.shape == image.shape and decoded.dtype == image.dtype


@pytest.mark.integration
def test_default_architecture_codec():
    cfg = Config()
    m = IndependentWaveletHyperprior(cfg).to(DEVICE).eval()
    m.update(force=True)
    entropy = make_cpu_entropy_model(m)
    x = torch.rand(1, 1, 256, 256, device=DEVICE)
    records, payload = compress_image_tiles(m, entropy, x, cfg, entropy_batch_size=1)
    decoded = decompress_image_tiles(
        m,
        entropy,
        records,
        256,
        256,
        cfg,
        entropy_decode_batch_size=1,
        synthesis_batch_size=1,
    )
    assert payload > 0 and decoded.shape == x.shape and torch.isfinite(decoded).all()
