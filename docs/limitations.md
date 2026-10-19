# Scope and limitations
- CPU synthetic integration tests verify mechanics, not compression quality.
  A measured 12-frame within-dataset run is included separately; it is not
  clinical or held-out validation. Earlier model variants are separate.
- CUDA backend behavior must be checked on the deployment GPU. The original
  decoder includes backend retries; CPU tests cannot validate these GPU paths.
- Use the default 256-pixel crops and tiles for the first run. The inherited
  configuration only checks divisibility by 16, which is weaker than the full
  hyperprior/MS-SSIM shape constraints. Arbitrary configured sizes are not
  guaranteed. Very small images can exceed reflection-padding limits.
- The original dataset split can overlap for very small datasets. This is a
  smoke-test convenience, not evidence of held-out generalization. Real
  experiments require independent splits and normalization fitted as appropriate.
- Original normalization assumes nonnegative integer TIFFs and returns float32
  for floating-point data. Integer denormalization truncates rather than rounds;
  it is not an exact reversible integer transform. Constant datasets are rejected.
- Archive/source association during evaluation uses stored source indices; keep
  the original sorted input ordering. Archive size ratios reported by the CLI
  use source TIFF file sizes and exclude external model weights.
- Source functions are retained for phase-one compatibility. Safety-hardening,
  archive/model identity checks, and external-codec baseline comparisons are separate work.

- v0.2.0 adds single-codec benchmark and comparison commands. The author reported the full 46-test suite and a 12-frame
  benchmark passing on the target GPU environment. Local GPU verification
  was unavailable. No real experimental TIFFs are bundled.
