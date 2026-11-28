"""MicroPython ROMFS images as OpenMV builds them (S7 bite 4, D82): read one, and write one, on the laptop.

The N6 loads models only from its model area (ROMFS0), and an image replaces the whole partition, so the image we
write must hold the board's own vendor files too (the earlier `ml/build_romfs_n6.sh`). This module reads the
records of an existing image (the board's, from the rig backup) and writes a new image with the same encoding as
OpenMV's `tools/mkromfs.py` (`encode_vint`, `encode_record`, `encode_file`; MIT licence, read 2026-11-27 at the
tree's commit 7d4dbf7a): an image is one header record holding file records; a file record holds a name record
and a data record whose payload is aligned (32 bytes for a model, 4 otherwise). Nothing here touches a board.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

HEADER = 0x14A6B1
HEADER_ALIGN, FILEREC_ALIGN = 16, 8
KIND_PADDING, KIND_DATA, KIND_FILE = 1, 2, 5
PAD = 0x80


class RomfsError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    data: bytes
    align: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def default_align(name: str) -> int:
    """mkromfs: a model entry in OpenMV's configs has alignment 32; the labels and other files get its default 4."""
    return 32 if name.endswith(".tflite") else 4


# --- writing (mkromfs's encoding) ----------------------------------------------------------------------------

def vint(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(out)


def record(kind: int, payload: bytes, align: int = 0, offset: int = 0) -> bytes:
    padding = 0
    if align:
        offset += len(vint(kind) + vint(len(payload)))
        padding = ((offset + align - 1) & ~(align - 1)) - offset
    return (vint(kind) if kind else b"") + bytes([PAD]) * padding + vint(len(payload)) + payload


def file_record(name: bytes, data: bytes, align: int, offset: int) -> bytes:
    name_rec = record(0, name)
    data_rec = record(KIND_DATA, data, align, offset + 1 + (FILEREC_ALIGN - 1) + len(name_rec))
    return record(KIND_FILE, name_rec + data_rec, FILEREC_ALIGN)


def encode(entries: list[Entry], limit: int | None = None) -> bytes:
    """An image of `entries`, in order. `limit`: the partition's size; a larger image is refused."""
    body, offset = bytearray(), HEADER_ALIGN
    for e in entries:
        rec = file_record(e.name.encode("ascii"), e.data, e.align, offset)
        offset += len(rec)
        body += rec
    image = record(HEADER, bytes(body), HEADER_ALIGN)
    if limit is not None and len(image) > limit:
        raise RomfsError(f"the image is {len(image):,} bytes, larger than the {limit:,}-byte model area")
    return image


# --- reading ---------------------------------------------------------------------------------------------------

def _vint(b: bytes, i: int) -> tuple[int, int]:
    v = 0
    while True:
        if i >= len(b):
            raise RomfsError("the image ends inside a number")
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i


def _record(b: bytes, i: int, has_kind: bool = True) -> tuple[int, int, int]:
    kind = 0
    if has_kind:
        kind, i = _vint(b, i)
    while i < len(b) and b[i] == PAD:              # alignment padding sits between the kind and the length
        i += 1
    n, i = _vint(b, i)
    if i + n > len(b):
        raise RomfsError("a record runs past the end of the image")
    return kind, i, i + n


def decode(b: bytes) -> tuple[list[Entry], int]:
    """(entries, image length). The partition read back from a board is longer than its image: the rest is ignored."""
    kind, start, end = _record(b, 0)
    if kind != HEADER:
        raise RomfsError(f"not a ROMFS image (first record kind {kind:#x}, expected {HEADER:#x})")
    entries, i = [], start
    while i < end:
        kind, ps, pe = _record(b, i)
        if kind == KIND_PADDING:
            i = pe
            continue
        if kind != KIND_FILE:
            raise RomfsError(f"record kind {kind} at byte {i}: only files are expected in a model area")
        _, ns, ne = _record(b, ps, has_kind=False)
        dk, ds, de = _record(b, ne)
        if dk != KIND_DATA:
            raise RomfsError(f"file record at byte {i} has no data record")
        name = b[ns:ne].decode("ascii")
        entries.append(Entry(name, bytes(b[ds:de]), default_align(name)))
        i = pe
    return entries, end


def with_ours(vendor: list[Entry], ours: list[Entry], limit: int) -> bytes:
    """The N6 image D82 asks for: the board's own files first, byte for byte and in their order, then ours. A name
    of ours that the board already has is refused (one would hide the other in /rom)."""
    clash = sorted({e.name for e in ours} & {e.name for e in vendor})
    if clash:
        raise RomfsError(f"{', '.join(clash)} is already in the board's model area; give ours another name")
    if len({e.name for e in ours}) != len(ours):
        raise RomfsError("two of our files have the same name")
    return encode([*vendor, *ours], limit)
