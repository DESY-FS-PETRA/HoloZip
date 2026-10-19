# Benchmark and comparison commands

Run from the project root with the existing environment and matching checkpoint.
Both commands require a **new or empty output directory** to protect old results.
No existing archive, original TIFF or checkpoint is overwritten.

## Compare an existing archive (no re-encoding)

```bash
python -m holozip compare \
  --data-dir /path/to/original_tiffs \
  --checkpoint ./hiwh_v2_100e_best.pt \
  --archive ./runs/real_check/first3.hiwh \
  --output-dir ./runs/compare_first3
```

References are located by the stored filename rather than a changed directory
sort order. Their shape and dtype must match. Legacy archives do not contain a
model fingerprint, so the exact matching checkpoint remains the user's
responsibility. New benchmark archives contain a checkpoint SHA-256 checked by
`compare`. Do not substitute different source images with identical names.

## Benchmark evenly spaced source frames

```bash
python -m holozip benchmark \
  --data-dir /path/to/original_tiffs \
  --checkpoint ./hiwh_v2_100e_best.pt \
  --stats ./global_stats.json \
  --count 12 \
  --output-dir ./runs/benchmark_12
```

For explicit zero-based indices in sorted filename order, replace `--count 12`
with `--indices 0 700 1500 2205`. Indices are validated, deduplicated by rejection,
and processed in sorted index order. Automatic selection uses integer-spaced
linspace from the first to last source index. It is not a random or independent
held-out split. For 2206 inputs and count 3, it selects 0, 1102, 2205.

Add `--save-decoded` to save reconstructed TIFFs. Omit it to avoid extra large
files. `--profile-row 1024` changes the line profile; the default is the middle
row. `--no-plots` skips both PNGs and numeric line-profile CSVs. Entropy grouping
is stored in the header; use `--entropy-batch-size 32` to specify it on encode.
`--cpu-entropy-threads` defaults to 8 and `--synthesis-batch-size 0` uses the
checkpoint's tile_batch. The original shape/padding constraints still apply.

## Output files

| File/directory | Meaning |
| --- | --- |
| `selected.hiwh` | Actual encoded archive, benchmark only |
| `metrics.csv` | Per-frame metrics, pixel/TIFF/record bytes, timing |
| `summary.json` | Aggregate metrics, archive/model cost, timing, environment and hashes |
| `comparisons/` | Original, decoded, raw absolute difference, line profiles and central detail |
| `profiles/` | Numeric column/original/decoded/signed-error CSVs |
| `decoded/` | TIFFs, only with `--save-decoded` |

Original and decoded image panels share one intensity scale (full-range, no
percentile clipping). Error maps show absolute raw-intensity error on their own
explicit scale. Profiles and signed errors use the raw intensity domain. PNGs
are inspection aids; they cannot establish phase-retrieval fidelity.

## Definitions and accounting

- `ssim`, `psnr`, `mse`, `mae`, `rmse`, `max_error`: shared log-normalized domain.
  PSNR data range is 1; SSIM is single-scale Gaussian-window SSIM, window 11.
- `raw_*`: original TIFF intensity domain. Raw PSNR uses each reference frame's
  max-minus-min range. `mean_change_percent = (decoded mean / reference mean - 1)*100`,
  using the original metric's small denominator stabilizer.
- Summary means are arithmetic **per-frame means**, not pooled pixel metrics.
- `record_bpp`: frame record bytes times 8 divided by frame pixels; includes
  tile metadata and lengths, excludes the global header.
- `archive_bpp`: complete on-disk archive times 8 divided by all selected pixels;
  includes the global header.
- TIFF ratio: sum of selected source TIFF file sizes / complete archive bytes.
  Pixel ratio uses uncompressed array nbytes instead. Neither is a model-inclusive ratio.
- `tiff_to_archive_plus_checkpoint_ratio`: source TIFF bytes divided by archive
  plus the complete external checkpoint size. The model cost is amortized over
  **this subset only**; it is not an estimate for the whole dataset.
- Timings synchronize CUDA at boundaries. Encoding includes image read,
  normalization, H2D, coding and archive writes/close; it excludes setup,
  checkpoint hashing, metrics and plots. No fsync is requested.
- Decode time is the sum of per-frame archive record reads, decoding, D2H and
  denormalization; excludes reference image reads, metrics, plots and TIFF exports.
  Setup is reported separately. No warmup is performed, so short runs include
  cold-start effects and are not steady-state throughput measurements.
- `null` in JSON denotes non-finite metrics (e.g. infinite perfect-match PSNR);
  CSV retains `inf`. Shape/dtype checks do not establish source-content identity.

This release provides single-codec evaluation. JPEG2000/SZ3/ZFP baselines,
rate-distortion sweeps, repeated performance runs, and downstream phase
retrieval are not implemented or claimed here.

## Older native thread-pool runtimes
If PyTorch aborts while changing CPU threads in a reused process, set native
thread counts before process startup. `--cpu-entropy-threads 0` leaves PyTorch's
startup setting unchanged for the evaluation commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

Apply the same prefix and add `--cpu-entropy-threads 0` when running `compare`
or `benchmark` on such a runtime. This is a compatibility workaround, not a
claim of performance optimization. Process startup settings are preferable to
late thread-pool reconfiguration on affected older builds.
