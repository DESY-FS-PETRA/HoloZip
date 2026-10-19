# Validation record

## Local checks
- Python 3.12.14, CPU; 46 tests passed after the native-thread setup correction.
- Original 55 function/class bodies match the supplied source structurally.
- Actual entropy bytes and CPU decode outputs match the original implementation
  for tested single- and multi-tile fixtures.
- Small-model train/optimizer steps, default-model entropy coding, checkpoint
  loading, CLI workflows, report accounting and PNG export were executed.
- A new command was also run with the author's supplied trained checkpoint on
  synthetic software fixtures. This verifies mechanics, not scientific fidelity.

## Target-system evidence supplied by the project author
- Python 3.11.13, PyTorch 2.0.1, CompressAI 1.2.8, NumPy 1.26.4,
  tifffile 2025.6.11, scikit-image 0.25.2, matplotlib 3.10.3.
- NVIDIA RTX A6000, CUDA device; 46 tests passed, 2 warnings, 22.27 seconds.
- Native thread counts set at startup with OMP_NUM_THREADS=2,
  MKL_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2.
- The 12-frame real-data archive benchmark completed; its unmodified CSV and
  JSON reports are in `benchmarks/results/mg_12_frames/`.

CUDA synthesis comparisons allow small float32 differences (atol=1e-7,
rtol=1e-6); CPU outputs and entropy byte comparisons remain exact in the
regression tests. AST comparisons parse both sources using the same Python
interpreter, avoiding Python-version-dependent AST serialization hashes.

No hosted CI run, independent held-out dataset study, external codec baseline
or downstream phase-retrieval fidelity validation is claimed.

## Bundled real-data examples
Three original 2048 x 2048 float32 TIFFs and the author's shared global
normalization statistics were added unchanged. The uploaded duplicate suffix
`(1)` was removed from one filename only. SHA-256 checksums are in the example
manifest. No image pixels were resized, cropped or converted.

All three completed actual compression, archive decoding, TIFF export and
visual-diagnostic generation using the author's matching checkpoint on CPU.
Mean normalized SSIM was 0.96141, mean normalized PSNR 42.47 dB, and archive-only
compression ratio 68.48x. Full local run metadata is recorded under
`benchmarks/results/real_examples_cpu/`; these are execution examples with
unknown training overlap, not independent generalization measurements.

The expanded suite passed **50 tests** on local CPU (4 non-failing warnings,
11.05 seconds). Four added checks validate sample checksums, dimensions, dtype,
finite pixels and normalization provenance. The target GPU report remains
46 tests for the earlier suite; the added sample tests have not been rerun
there. The matching trained checkpoint remains external.
