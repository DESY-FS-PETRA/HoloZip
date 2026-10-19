# Benchmarks

Measured single-codec evaluation is implemented in `src/holozip/evaluation`.
Run `python -m holozip benchmark --help`; definitions and timing scopes are in
[../docs/benchmark.md](../docs/benchmark.md).

`results/mg_12_frames/` contains the user-supplied GPU benchmark CSV and JSON
for 12 evenly spaced frames. It is a within-dataset evaluation, not a
held-out generalization test. The trained model and raw dataset are external.
