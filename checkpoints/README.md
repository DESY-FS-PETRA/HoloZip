# Model checkpoints

The pretrained checkpoint `hiwh_v2_100e_best.pt` is included in this directory.

- File size: 24,655,101 bytes
- SHA-256: `3463234e49aa6f646462286ad911b4ca3ea5127b16e17380b1a77be3bba119a8`

This checkpoint was used for the reported benchmarks. It can be used directly
with the three real hologram examples and the matching normalization statistics
in `examples/real_holograms/`.

From the repository root, run:

    python -m holozip benchmark \
      --data-dir examples/real_holograms \
      --checkpoint checkpoints/hiwh_v2_100e_best.pt \
      --stats examples/real_holograms/global_stats.json \
      --count 3 \
      --output-dir runs/real_examples \
      --cpu-entropy-threads 0 \
      --save-decoded

The output directory must be new or empty. On systems requiring the native
thread workaround, prefix the command with:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

Decoding requires the exact checkpoint used for compression. Normalization
statistics are stored in the archive and used during decompression.

The bundled images are execution examples, not an independently held-out
test set. The full experimental dataset is not included.