"""S7 bite 4: ROMFS images read and written on the laptop, as OpenMV's tools/mkromfs.py writes them. Synthetic files
only; the check against the board's own image (the rig backup) is in the PR's self-test, not in git."""

import importlib.util
import sys
from pathlib import Path

import pytest

from app.bench import romfs
from app.bench.romfs import Entry

OPENMV_TOOLS = Path.home() / "openmv-dev" / "openmv" / "tools"
AREA = 0x01800000


def sample():
    return [Entry("a_model.tflite", bytes(range(256)) * 37, 32), Entry("a_model.txt", b"urchin\n", 4),
            Entry("b.tflite", b"\x00" * 129, 32)]


def test_write_then_read_gives_the_same_files_with_payloads_aligned():
    image = romfs.encode(sample(), AREA)
    entries, n = romfs.decode(image + b"\xff" * 1000)                 # a read-back runs to the partition's end
    assert n == len(image) and entries == sample()
    for e in entries:
        at = image.index(e.data) if len(e.data) > 32 else None
        assert at is None or at % e.align == 0


@pytest.mark.skipif(not (OPENMV_TOOLS / "mkromfs.py").is_file(), reason="no OpenMV tree on this machine")
def test_the_encoding_is_openmvs_own():
    sys.path.insert(0, str(OPENMV_TOOLS))
    try:
        spec = importlib.util.spec_from_file_location("omv_mkromfs", OPENMV_TOOLS / "mkromfs.py")
        mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk)
    finally:
        sys.path.remove(str(OPENMV_TOOLS))
    body, offset = bytearray(), mk.ROMFS_HEADER_ALIGN
    for e in sample():
        rec = mk.encode_file(e.name.encode(), e.data, e.align, offset)
        offset += len(rec)
        body += rec
    assert romfs.encode(sample()) == mk.encode_record(mk.ROMFS_HEADER, bytes(body), mk.ROMFS_HEADER_ALIGN)


def test_our_files_follow_the_boards_and_never_replace_one():
    vendor = sample()[:2]
    image = romfs.with_ours(vendor, [Entry("yolox-nano-conv-256-c5.tflite", b"ours" * 100, 32)], AREA)
    names = [e.name for e in romfs.decode(image)[0]]
    assert names == ["a_model.tflite", "a_model.txt", "yolox-nano-conv-256-c5.tflite"]
    with pytest.raises(romfs.RomfsError, match="already in the board"):
        romfs.with_ours(vendor, [Entry("a_model.tflite", b"x", 32)], AREA)
    with pytest.raises(romfs.RomfsError, match="same name"):
        romfs.with_ours(vendor, [Entry("o.tflite", b"x", 32), Entry("o.tflite", b"y", 32)], AREA)
    with pytest.raises(romfs.RomfsError, match="larger than"):
        romfs.with_ours(vendor, [Entry("big.tflite", b"x" * 5000, 32)], 4096)


def test_what_is_not_an_image_is_refused():
    for bad, why in ((b"\x05\x02hi", "not a ROMFS image"), (romfs.encode(sample())[:100], "past the end")):
        with pytest.raises(romfs.RomfsError, match=why):
            romfs.decode(bad)
