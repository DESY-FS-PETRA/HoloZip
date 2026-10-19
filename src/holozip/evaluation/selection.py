"""Deterministic source-frame selection, in sorted TIFF order."""

import numpy as np


def select_indices(
    total: int, count: int = 12, indices: list[int] | None = None
) -> list[int]:
    if total < 1:
        raise ValueError("No input frames")
    if indices is not None:
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Explicit indices must be nonempty and unique")
        if any(i < 0 or i >= total for i in indices):
            raise ValueError(f"Source indices must be between 0 and {total - 1}")
        return sorted(indices)
    if count < 1:
        raise ValueError("count must be positive")
    return np.linspace(0, total - 1, min(count, total), dtype=int).tolist()
