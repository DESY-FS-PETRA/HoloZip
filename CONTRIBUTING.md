# Contributing
Install with `pip install -e ".[dev]"`. Run `python -m pytest` and
`ruff check src tests` before proposing changes. Add regression tests for codec
changes. Changes to entropy grouping, normalization, architecture or the archive
layout require explicit compatibility tests and documentation. Do not include
large data, checkpoints, credentials, or institution-specific filesystem paths.
