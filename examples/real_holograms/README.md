# Real hologram examples

Three real 2048 × 2048 float32 TIFF images are included for demonstrating
HoloZip compression and decompression. Original TIFF bytes and pixel values
are preserved. File checksums and dimensions are recorded in `manifest.json`.

These images are execution examples, not an independently held-out test set.
Their overlap with the training data has not been established.

The accompanying `global_stats.json` contains the shared normalization
statistics from the original 2,206-frame dataset. These statistics match
the bundled pretrained checkpoint and were not recomputed from these examples.

## Run the example

The pretrained model is included at
`checkpoints/hiwh_v2_100e_best.pt`.
See [checkpoint details](../../checkpoints/README.md).

From the repository root, run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m holozip benchmark \
  --data-dir examples/real_holograms \
  --checkpoint checkpoints/hiwh_v2_100e_best.pt \
  --stats examples/real_holograms/global_stats.json \
  --count 3 \
  --output-dir runs/real_examples \
  --cpu-entropy-threads 0 \
  --save-decoded
```

The output directory must be new or empty. Choose a different directory
name when repeating the example.

## Outputs

- `selected.hiwh`: compressed archive.
- `decoded/`: decompressed TIFF images.
- `metrics.csv`: per-image quality metrics and timing.
- `summary.json`: aggregate metrics, file sizes and execution environment.
- `comparisons/`: comparison images, error maps and intensity profiles.
- `profiles/`: numeric intensity-profile CSV files.

## Example results

A verified run on these three images produced approximately:

| Metric | Result |
| --- | ---: |
| Compression ratio, excluding checkpoint | 68.48× |
| Mean normalized SSIM | 0.96141 |
| Mean normalized PSNR | 42.47 dB |

Small numerical differences may occur across hardware and software versions.
CPU execution is supported but can be slower than CUDA.
These measurements do not establish downstream phase-retrieval fidelity.