# Mg hologram benchmark: 12 selected frames

Results supplied by the project author from the target Python 3.11 / CUDA
system on 2026-10-19. Full-resolution 2048 x 2048 float32 holograms were decoded
from an actual entropy-coded archive before metrics were computed.

Source indices: 0, 200, 400, 601, 801, 1002, 1202, 1403, 1603, 1804, 2004, 2205.
The selection follows sorted filenames over 2,206 frames. No independently
held-out dataset or downstream phase-retrieval quality is established here.

The summary records the checkpoint/archive hashes, package versions, device,
thread count, timing definitions and per-image metric aggregation.
Original TIFF sizes and archive sizes are decimal bytes. Model-inclusive ratio
uses the complete external checkpoint cost amortized over these 12 frames.
The timings include cold-start effects and exclude diagnostic plotting.
