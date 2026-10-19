"""Evaluation reports must describe actual decoded records and file sizes."""

import csv
import json

import numpy as np
import pytest
import tifffile

from holozip.app import build_parser
from holozip.config import Config
from holozip.data.normalization import compute_global_stats
from holozip.evaluation.plots import plot_comparison
from holozip.evaluation.runner import json_safe, sha256_file
from holozip.evaluation.selection import select_indices
from holozip.models.hyperprior import IndependentWaveletHyperprior
from holozip.runtime import DEVICE
from holozip.training.checkpoint import save_checkpoint


def run(*argv):
    args = build_parser().parse_args(list(map(str, argv)))
    args.func(args)


def test_even_spacing_and_explicit_order():
    assert select_indices(2206, 3) == [0, 1102, 2205]
    assert select_indices(3, 12) == [0, 1, 2]
    assert select_indices(10, indices=[9, 0, 4]) == [0, 4, 9]


@pytest.mark.parametrize(
    "total,count,indices",
    [(0, 3, None), (3, 0, None), (3, 2, []), (3, 2, [0, 0]), (3, 2, [-1]), (3, 2, [3])],
)
def test_invalid_selection(total, count, indices):
    with pytest.raises(ValueError):
        select_indices(total, count, indices)


def test_json_nonfinite_policy():
    assert json.loads(
        json.dumps(
            json_safe(
                {
                    "psnr": float("inf"),
                    "raw": float("-inf"),
                    "nan": float("nan"),
                    "ok": 1.2,
                }
            ),
            allow_nan=False,
        )
    ) == {"psnr": None, "raw": None, "nan": None, "ok": 1.2}


def test_plot_and_profile_bounds(tmp_path):
    x = np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48)
    p = tmp_path / "comparison.png"
    plot_comparison(x, x * 0.99, p, "Synthetic test", row=0)
    assert p.read_bytes().startswith(b"\x89PNG") and p.stat().st_size > 1000
    with pytest.raises(ValueError, match="profile-row"):
        plot_comparison(x, x, p, "invalid", row=32)


@pytest.fixture
def experiment(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(42)
    for i in range(3):
        tifffile.imwrite(
            data / f"frame_{i:03d}.tiff",
            (0.1 + rng.random((256, 256))).astype(np.float32),
        )
    stats = tmp_path / "stats.json"
    compute_global_stats(sorted(map(str, data.glob("*.tiff"))), str(stats))
    cfg = Config(analysis_channels=8, latent_channels=8, hyper_channels=8)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(
        str(checkpoint),
        IndependentWaveletHyperprior(cfg).to(DEVICE),
        cfg,
        str(stats),
        1,
        0.1,
    )
    return data, checkpoint, stats


@pytest.mark.integration
def test_benchmark_and_existing_archive_comparison(experiment, tmp_path):
    data, checkpoint, stats = experiment
    out = tmp_path / "benchmark"
    run(
        "benchmark",
        "--data-dir",
        data,
        "--checkpoint",
        checkpoint,
        "--stats",
        stats,
        "--output-dir",
        out,
        "--count",
        2,
        "--entropy-batch-size",
        2,
        "--cpu-entropy-threads",
        2,
        "--save-decoded",
    )
    summary = json.loads((out / "summary.json").read_text())
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert summary["frame_count"] == 2 and summary["source_indices"] == [0, 2]
    assert summary["archive_bytes"] == (out / "selected.hiwh").stat().st_size
    assert summary["checkpoint_bytes"] == checkpoint.stat().st_size
    assert summary["archive_sha256"] == sha256_file(out / "selected.hiwh")
    source_bytes = sum((data / rows[i]["filename"]).stat().st_size for i in range(2))
    assert summary["tiff_to_archive_ratio"] == pytest.approx(
        source_bytes / summary["archive_bytes"]
    )
    assert (
        summary["tiff_to_archive_plus_checkpoint_ratio"]
        < summary["tiff_to_archive_ratio"]
    )
    assert summary["encode_seconds"] > 0 and summary["decode_seconds"] > 0
    assert len(list((out / "comparisons").glob("*.png"))) == 2
    assert len(list((out / "decoded").glob("*.tiff"))) == 2
    profile = np.loadtxt(
        next((out / "profiles").glob("*.csv")), delimiter=",", skiprows=1
    )
    np.testing.assert_allclose(profile[:, 3], profile[:, 2] - profile[:, 1], atol=1e-12)
    assert summary["metrics"]["ssim"]["mean"] == pytest.approx(
        np.mean([float(r["ssim"]) for r in rows])
    )
    compare = tmp_path / "comparison"
    run(
        "compare",
        "--data-dir",
        data,
        "--checkpoint",
        checkpoint,
        "--archive",
        out / "selected.hiwh",
        "--output-dir",
        compare,
        "--cpu-entropy-threads",
        2,
        "--no-plots",
    )
    other = json.loads((compare / "summary.json").read_text())
    assert other["archive_model_hash_verified"] is True
    assert other["encode_seconds"] is None
    assert other["metrics"]["psnr"]["mean"] == pytest.approx(
        summary["metrics"]["psnr"]["mean"], abs=1e-4
    )
    # The user's existing v5.5 archives have no checkpoint fingerprint.
    from holozip.codec.archive import (
        read_frame_record,
        read_json,
        write_frame_record,
        write_json,
    )
    from holozip.runtime import MAGIC

    legacy = tmp_path / "legacy.hiwh"
    with (out / "selected.hiwh").open("rb") as src, legacy.open("wb") as dst:
        assert src.read(len(MAGIC)) == MAGIC
        header = read_json(src)
        header.pop("checkpoint_sha256")
        dst.write(MAGIC)
        write_json(dst, header)
        for _ in range(header["frame_count"]):
            meta, records = read_frame_record(src)
            write_frame_record(dst, meta, records)
    legacy_out = tmp_path / "legacy_comparison"
    run(
        "compare",
        "--data-dir",
        data,
        "--checkpoint",
        checkpoint,
        "--archive",
        legacy,
        "--output-dir",
        legacy_out,
        "--cpu-entropy-threads",
        2,
        "--no-plots",
    )
    legacy_summary = json.loads((legacy_out / "summary.json").read_text())
    assert legacy_summary["archive_model_hash_verified"] is False
    assert legacy_summary["metrics"]["ssim"]["mean"] == pytest.approx(
        summary["metrics"]["ssim"]["mean"], abs=1e-5
    )
    # Existing results cannot be silently overwritten.
    with pytest.raises(FileExistsError):
        run(
            "compare",
            "--data-dir",
            data,
            "--checkpoint",
            checkpoint,
            "--archive",
            out / "selected.hiwh",
            "--output-dir",
            compare,
        )


@pytest.mark.integration
def test_wrong_checkpoint_hash_rejected(experiment, tmp_path):
    from holozip.codec.archive import write_json
    from holozip.runtime import MAGIC

    data, checkpoint, stats = experiment
    archive = tmp_path / "bad.hiwh"
    with archive.open("wb") as f:
        f.write(MAGIC)
        write_json(f, {"checkpoint_sha256": "wrong"})
    with pytest.raises(ValueError, match="SHA-256"):
        run(
            "compare",
            "--data-dir",
            data,
            "--checkpoint",
            checkpoint,
            "--archive",
            archive,
            "--output-dir",
            tmp_path / "bad_out",
            "--no-plots",
            "--cpu-entropy-threads",
            2,
        )


def test_repeated_setup_does_not_reconfigure_native_pools(experiment, monkeypatch):
    # Regression for the user's native abort during a redundant thread setter.
    import torch
    from holozip.evaluation import runner

    _, checkpoint, _ = experiment
    current = torch.get_num_threads()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Native thread setter must not be called for unchanged settings"
        )

    monkeypatch.setattr(runner, "configure_cpu_entropy_threads", forbidden)
    for threads in [current, current, 0]:
        model, cfg, entropy, seconds = runner.setup_model(str(checkpoint), threads)
        assert seconds >= 0 and next(model.parameters()).numel() > 0
        assert entropy.entropy_bottleneck._quantized_cdf.numel() > 0
