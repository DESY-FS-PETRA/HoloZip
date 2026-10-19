# Architecture and module boundaries
The supplied v5.5 implementation processes independent 2-D holograms. A fixed
Haar transform produces four subbands; learned analysis/synthesis transforms
use GDN and an entropy hyperprior. This release does not implement the earlier
3-D windowed HoloZip model.

- `models`: Haar transform and neural analysis/synthesis/hyperprior transforms.
- `data`: TIFF discovery, shared log normalization, soft fringe sampling, crops.
- `training`: rate-distortion losses, optimizer loop, checkpoint handling.
- `codec`: tiling, CPU entropy path, binary records, encode/decode pipeline.
- `commands` / `cli`: dataset orchestration and the original six subcommands.
- `runtime` / `profiling`: device setup, finite-value checks, timing.

The existing `reconstruct` command decompresses holograms; it does not perform
phase retrieval or tomographic reconstruction. `ssim` in training validation is
MS-SSIM; the image evaluator uses Gaussian-window SSIM. These are distinct.
