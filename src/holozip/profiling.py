"""profiling extracted from the supplied v5.5 codec."""

from __future__ import annotations

import os
from typing import Optional

import torch


class StageTimer:
    """Accumulate wall-clock stage timings, synchronizing CUDA only at stage boundaries."""

    def __init__(self):
        self.totals = {}
        self.calls = {}

    def add(self, name, dt):
        self.totals[name] = self.totals.get(name, 0.0) + float(dt)
        self.calls[name] = self.calls.get(name, 0) + 1

    def report(self):
        total = sum(self.totals.values())
        return {
            k: {
                "seconds": round(v, 3),
                "percent": round(100.0 * v / max(total, 1e-12), 1),
                "calls": self.calls.get(k, 0),
            }
            for k, v in sorted(self.totals.items(), key=lambda kv: -kv[1])
        }


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def configure_cpu_entropy_threads(requested: Optional[int] = None) -> int:
    """Tune PyTorch CPU threading for the small hyperprior convolutions.

    Very large OpenMP thread counts are counterproductive for the tiny h_a/h_s
    tensors used here.  A modest thread count usually gives much better latency.
    """
    if requested is None or requested <= 0:
        cpu_count = os.cpu_count() or 8
        requested = min(8, max(1, cpu_count // 2))
    torch.set_num_threads(int(requested))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before parallel work starts.
        pass
    return int(requested)
