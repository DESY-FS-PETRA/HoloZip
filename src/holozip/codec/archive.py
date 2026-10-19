"""codec.archive extracted from the supplied v5.5 codec."""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO, Dict, List, Sequence, Tuple


def write_u32(f: BinaryIO, value: int) -> None:
    f.write(struct.pack(">I", int(value)))


def read_u32(f: BinaryIO) -> int:
    data = f.read(4)
    if len(data) != 4:
        raise EOFError("Unexpected end of archive")
    return struct.unpack(">I", data)[0]


def write_bytes(f: BinaryIO, payload: bytes) -> None:
    write_u32(f, len(payload))
    f.write(payload)


def read_bytes(f: BinaryIO) -> bytes:
    n = read_u32(f)
    payload = f.read(n)
    if len(payload) != n:
        raise EOFError("Truncated archive payload")
    return payload


def write_json(f: BinaryIO, obj: Dict[str, Any]) -> None:
    write_bytes(f, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def read_json(f: BinaryIO) -> Dict[str, Any]:
    return json.loads(read_bytes(f).decode("utf-8"))


def write_frame_record(
    f: BinaryIO, meta: Dict[str, Any], records: Sequence[Dict[str, Any]]
) -> None:
    write_json(f, meta)
    write_u32(f, len(records))
    for r in records:
        tile_meta = {k: v for k, v in r.items() if k not in {"y", "z"}}
        write_json(f, tile_meta)
        write_bytes(f, r["y"])
        write_bytes(f, r["z"])


def read_frame_record(f: BinaryIO) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta = read_json(f)
    n = read_u32(f)
    records = []
    for _ in range(n):
        r = read_json(f)
        r["y"] = read_bytes(f)
        r["z"] = read_bytes(f)
        records.append(r)
    return meta, records
