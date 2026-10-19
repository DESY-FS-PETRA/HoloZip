"""codec.tiling extracted from the supplied v5.5 codec."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def tile_positions(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def tile_coordinates(
    height: int, width: int, tile: int, overlap: int
) -> List[Tuple[int, int]]:
    return [
        (t, l)
        for t in tile_positions(height, tile, overlap)
        for l in tile_positions(width, tile, overlap)
    ]


def hann_window_2d(size: int, device: torch.device) -> torch.Tensor:
    w = torch.hann_window(size, periodic=False, device=device)
    return torch.outer(w, w).clamp_min(1e-3)[None, None]


def pad_tile(tile: torch.Tensor, size: int) -> torch.Tensor:
    ph = size - tile.shape[-2]
    pw = size - tile.shape[-1]
    if ph or pw:
        tile = F.pad(tile, (0, pw, 0, ph), mode="reflect")
    return tile
