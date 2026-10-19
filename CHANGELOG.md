# Changelog

## 0.2.0
- Modular packaging of the supplied independent 2-D wavelet hyperprior codec.
- Preserved original model, losses, normalization and archive record functions.
- Added benchmark/compare commands, explicit model-size accounting, CSV/JSON
  reports, visual comparisons and intensity-profile export.
- Added CPU regression/integration tests and cross-Python source comparisons.
- Accommodated CUDA floating-point synthesis variability while keeping exact
  entropy-byte regression assertions.
- Avoided redundant native thread-pool configuration in evaluation setup.
- Included the measured 12-frame GPU benchmark and a separate three-frame
  visual example. Added three unmodified real TIFF execution examples and their shared
  normalization statistics; pretrained weights remain external.
