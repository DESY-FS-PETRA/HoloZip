"""Static diagnostics: common intensity scale, raw error and line profiles."""

from pathlib import Path

import numpy as np


def plot_comparison(reference, decoded, output, title, row=None):
    # Import lazily so a metrics-only evaluation does not initialize matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ref = np.asarray(reference)
    rec = np.asarray(decoded)
    if ref.ndim != 2 or rec.shape != ref.shape:
        raise ValueError("Comparison requires two 2-D images of equal shape")
    line = ref.shape[0] // 2 if row is None else row
    if line < 0 or line >= ref.shape[0]:
        raise ValueError(f"profile-row must be in [0, {ref.shape[0] - 1}]")
    difference = rec.astype(np.float64) - ref.astype(np.float64)
    lo, hi = float(min(ref.min(), rec.min())), float(max(ref.max(), rec.max()))
    if hi <= lo:
        hi = lo + 1
    maximum = float(np.abs(difference).max()) or np.finfo(float).eps
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, data, label in [
        (axes[0, 0], ref, "Original"),
        (axes[0, 1], rec, "Decoded"),
    ]:
        im = ax.imshow(data, cmap="gray", vmin=lo, vmax=hi)
        ax.axhline(line, color="cyan", lw=0.6)
        ax.set_title(label + " (shared intensity scale)")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)
    im = axes[0, 2].imshow(np.abs(difference), cmap="magma", vmin=0, vmax=maximum)
    axes[0, 2].set_title("Absolute error (raw units)")
    axes[0, 2].set_axis_off()
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)
    axes[1, 0].plot(ref[line], lw=0.8, label="Original")
    axes[1, 0].plot(rec[line], lw=0.8, alpha=0.8, label="Decoded")
    axes[1, 0].set(
        title=f"Intensity profile: row {line}",
        xlabel="Column (pixel)",
        ylabel="Raw intensity",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(difference[line], lw=0.7, color="tab:red")
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].set(
        title="Decoded minus original",
        xlabel="Column (pixel)",
        ylabel="Raw intensity error",
    )
    # A central detail allows fringe inspection without rescaling either panel.
    h, w = ref.shape
    side = min(128, h, w)
    y = (h - side) // 2
    x = (w - side) // 2
    detail = np.concatenate(
        [ref[y : y + side, x : x + side], rec[y : y + side, x : x + side]], axis=1
    )
    axes[1, 2].imshow(detail, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    axes[1, 2].axvline(side - 0.5, color="cyan", lw=0.6)
    axes[1, 2].set_title("Central detail: original | decoded")
    axes[1, 2].set_axis_off()
    fig.suptitle(title, fontsize=13)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
