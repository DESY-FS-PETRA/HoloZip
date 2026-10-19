# Existing v5.5 archive format
All binary lengths are big-endian unsigned 32-bit integers. File prefix:
`b"HIW52\x00\x00\x00"`. The JSON header retains `version=55`, frame count,
source names, configuration, normalization statistics, checkpoint basename and
entropy batch size. Each JSON value is encoded as a byte-string record: u32
length followed by UTF-8 JSON. Each frame has a JSON metadata record, u32 tile
count, then per tile: JSON metadata, length-prefixed y stream and z stream.

The checkpoint is external and required for decoding. The checkpoint basename
in the header is not a cryptographic model identifier. Use the exact matching
weights; this release does not add an integrity checksum or model fingerprint.
Preserve the encoder's entropy batch grouping on decode. Synthesis batch size
is independently configurable. This refactor retains magic and version values;
it does not introduce a new archive format.

The phase-two benchmark adds optional JSON fields `selected_source_indices` and
`checkpoint_sha256` to the existing header without changing binary record
layout, magic or version. The original decoder ignores these extra fields.
The new compare command validates the hash when present.
