import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    import torch

    old = torch.get_num_threads()
    if old != 2:
        torch.set_num_threads(2)
    yield
    if torch.get_num_threads() != old:
        torch.set_num_threads(old)


@pytest.fixture(scope="session")
def original():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "provenance/original_v5_5.py"
    spec = importlib.util.spec_from_file_location("original_v5_5", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
