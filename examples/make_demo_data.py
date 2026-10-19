"""Create deterministic software-test TIFFs; not a physical hologram simulator."""

import argparse
from pathlib import Path
import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:256, :256]
    for i in range(4):
        image = 2000 + 400 * np.sin(0.15 * x + 0.002 * y * y + i * 0.1)
        image += 100 * np.cos(0.07 * y)
        tifffile.imwrite(folder / f"frame_{i:03d}.tiff", image.astype(np.uint16))
    print(folder)


if __name__ == "__main__":
    main()
