import io

import numpy as np
import pytest
import tifffile
import torch

from holozip.codec.archive import (
    read_bytes,
    read_frame_record,
    read_u32,
    write_frame_record,
)
from holozip.codec.tiling import hann_window_2d, pad_tile, tile_coordinates
from holozip.config import Config
from holozip.data.dataset import HologramCropDataset
from holozip.data.io import list_images
from holozip.data.normalization import (
    compute_global_stats,
    global_log_denormalize,
    global_log_normalize,
    load_stats,
)
from holozip.data.sampling import load_sampling_pool, precompute_sampling_pool
from holozip.metrics import normalized_metrics, raw_metrics
from holozip.models.wavelet import HaarWavelet


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_size": 255},
        {"tile_size": 255},
        {"tile_overlap": 256},
        {"tile_overlap": -1},
        {"fringe_sampling_probability": 1.1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()


def test_valid_config():
    Config().validate()


@pytest.mark.parametrize("shape", [(1, 1, 32, 48), (2, 1, 256, 256)])
def test_haar_roundtrip(shape):
    x = torch.randn(shape)
    w = HaarWavelet()
    torch.testing.assert_close(w.inverse(w(x)), x, atol=5e-7, rtol=1e-6)


def test_archive_roundtrip():
    meta = {"filename": "a.tiff", "height": 256, "dtype": "<u2"}
    records = [
        {"top": 0, "left": 0, "shape": [2, 2], "y": b"\x00abc", "z": b"\xff\x00"}
    ]
    stream = io.BytesIO()
    write_frame_record(stream, meta, records)
    stream.seek(0)
    assert read_frame_record(stream) == (meta, records)


@pytest.mark.parametrize("payload", [b"", b"\x00\x00", b"\x00\x00\x00\x03ab"])
def test_truncated_payload(payload):
    with pytest.raises(EOFError):
        read_bytes(io.BytesIO(payload))


def test_truncated_u32():
    with pytest.raises(EOFError):
        read_u32(io.BytesIO(b"\x00"))


@pytest.mark.parametrize("shape", [(256, 256), (257, 513), (129, 199), (512, 800)])
def test_tiling_coverage(shape):
    coverage = np.zeros(shape, dtype=bool)
    for y, x in tile_coordinates(*shape, 256, 32):
        coverage[y : y + 256, x : x + 256] = True
    assert coverage.all()
    tile = torch.rand(1, 1, min(shape[0], 256), min(shape[1], 256))
    padded = pad_tile(tile, 256)
    assert padded.shape[-2:] == (256, 256)
    torch.testing.assert_close(padded[..., : tile.shape[-2], : tile.shape[-1]], tile)
    assert torch.all(hann_window_2d(256, torch.device("cpu")) > 0)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_normalization(tmp_path, dtype):
    raw = np.linspace(10, 2000, 256 * 256).reshape(256, 256).astype(dtype)
    tifffile.imwrite(tmp_path / "a.tiff", raw)
    stats = compute_global_stats(
        list_images(str(tmp_path), "*.tiff"), str(tmp_path / "s.json")
    )
    x = global_log_normalize(raw, stats)
    assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    np.testing.assert_allclose(
        global_log_denormalize(x, stats, raw.dtype), raw, atol=1.01, rtol=1e-6
    )
    assert load_stats(str(tmp_path / "s.json")) == stats


def test_constant_dataset_rejected(tmp_path):
    p = tmp_path / "a.tiff"
    tifffile.imwrite(p, np.ones((256, 256), np.uint16))
    with pytest.raises(ValueError, match="Degenerate"):
        compute_global_stats([str(p)], str(tmp_path / "s.json"))


def test_discovery(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((16, 16), np.uint16))
    assert len(list_images(str(tmp_path), "*.tiff")) == 1
    with pytest.raises(FileNotFoundError):
        list_images(str(tmp_path), "*.absent")


def test_metrics():
    x = np.random.default_rng(42).random((32, 32)).astype(np.float32)
    same = normalized_metrics(x, x)
    changed = normalized_metrics(x, x * 0.9)
    assert (
        same["ssim"] == pytest.approx(1) and same["mae"] == 0 and np.isinf(same["psnr"])
    )
    assert (
        changed["mae"] > 0 and changed["ssim"] < 1 and raw_metrics(x, x)["raw_mae"] == 0
    )


def test_sampling_dataset(tmp_path):
    raw = (np.random.default_rng(42).random((256, 256)) * 1000).astype(np.uint16)
    p = tmp_path / "a.tiff"
    tifffile.imwrite(p, raw)
    s = compute_global_stats([str(p)], str(tmp_path / "s.json"))
    precompute_sampling_pool(
        [str(p)], s, str(tmp_path / "pool.npz"), crop_size=256, pool_size=4, seed=42
    )
    pool, meta = load_sampling_pool(str(tmp_path / "pool.npz"))
    assert meta["crop_size"] == 256
    ds = HologramCropDataset(
        [str(p)], s, Config(), sampling_pool=pool, deterministic=True
    )
    a = ds[0]
    assert a.shape == (1, 256, 256) and torch.isfinite(a).all()
    torch.testing.assert_close(a, ds[0], atol=0, rtol=0)
