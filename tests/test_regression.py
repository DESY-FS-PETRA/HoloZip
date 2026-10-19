"""Compare preserved definitions and numerical outputs with the supplied script."""

import ast
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from holozip.cli import build_parser
from holozip.codec.tiling import tile_positions
from holozip.config import Config
from holozip.data.normalization import global_log_denormalize, global_log_normalize
from holozip.models.hyperprior import IndependentWaveletHyperprior

ROOT = Path(__file__).resolve().parents[1]


def test_all_extracted_definitions_are_unchanged():
    manifest = json.loads((ROOT / "provenance/extraction_manifest.json").read_text())
    assert (
        hashlib.sha256((ROOT / "provenance/original_v5_5.py").read_bytes()).hexdigest()
        == manifest["source_sha256"]
    )
    # AST fields differ across Python versions (e.g. type_params in 3.12).
    # Parse both files with the running interpreter instead of comparing a
    # serialized AST hash generated on a different Python version.
    source_tree = ast.parse((ROOT / "provenance/original_v5_5.py").read_text())
    source_definitions = {
        node.name: node
        for node in source_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert set(source_definitions) == set(manifest["symbols"])
    for name, item in manifest["symbols"].items():
        tree = ast.parse(
            (
                ROOT / "src/holozip" / Path(item["module"].replace(".", "/") + ".py")
            ).read_text()
        )
        node = next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
        assert ast.dump(node, include_attributes=False) == ast.dump(
            source_definitions[name], include_attributes=False
        ), name


def test_every_module_imports():
    for file in (ROOT / "src/holozip").rglob("*.py"):
        if file.name in ("__init__.py", "__main__.py"):
            continue
        importlib.import_module(
            "holozip."
            + ".".join(file.relative_to(ROOT / "src/holozip").with_suffix("").parts)
        )


def test_config_and_all_six_commands_match(original):
    assert dataclasses.asdict(Config()) == dataclasses.asdict(original.Config())
    commands = [
        ["compute-stats", "--data-dir", "d", "--output", "s.json"],
        [
            "precompute-sampling-pool",
            "--data-dir",
            "d",
            "--stats",
            "s",
            "--output",
            "p.npz",
        ],
        ["train", "--data-dir", "d", "--stats", "s", "--checkpoint", "m"],
        [
            "compress-dataset",
            "--data-dir",
            "d",
            "--stats",
            "s",
            "--checkpoint",
            "m",
            "--archive",
            "a",
        ],
        ["reconstruct", "--checkpoint", "m", "--archive", "a", "--output-dir", "o"],
        [
            "evaluate-target",
            "--data-dir",
            "d",
            "--checkpoint",
            "m",
            "--archive",
            "a",
            "--target-index",
            "0",
        ],
    ]
    for argv in commands:
        a = vars(build_parser().parse_args(argv))
        b = vars(original.build_parser().parse_args(argv))
        a.pop("func")
        b.pop("func")
        assert a == b


def test_model_forward_matches_original(original):
    cfg = Config(analysis_channels=8, latent_channels=8, hyper_channels=8)
    torch.manual_seed(42)
    model = IndependentWaveletHyperprior(cfg).cpu().eval()
    reference = (
        original.IndependentWaveletHyperprior(
            original.Config(**dataclasses.asdict(cfg))
        )
        .cpu()
        .eval()
    )
    reference.load_state_dict(model.state_dict())
    x = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        a = model(x)
        b = reference(x)
    for key in ["bands", "bands_hat", "preclamp", "reconstruction"]:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for key in ["y", "z"]:
        torch.testing.assert_close(
            a["likelihoods"][key], b["likelihoods"][key], rtol=0, atol=0
        )


def test_normalization_and_tiling_match(original):
    raw = np.arange(256, dtype=np.uint16).reshape(16, 16)
    s = {
        "float_data": False,
        "global_raw_min": 0,
        "global_log_min": 0,
        "global_log_max": float(np.log1p(255)),
    }
    a = global_log_normalize(raw, s)
    b = original.global_log_normalize(raw, s)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        global_log_denormalize(a, s, raw.dtype),
        original.global_log_denormalize(b, s, raw.dtype),
    )
    for length in [128, 256, 257, 511, 2048]:
        assert tile_positions(length, 256, 32) == original.tile_positions(
            length, 256, 32
        )
