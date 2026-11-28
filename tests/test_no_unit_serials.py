"""SPEC hard constraint: no unit names, addresses or serial numbers in git (lead 2, after PR 123: the bench N6's
DFU serial had been put in app/bench/n6_board.py). A board's chip serial is 24 hex digits (an STM32's 96-bit UID);
any run of exactly 24 hex digits in app/, tests/, docs/ or the example config fails this test. Longer runs
(sha256 values) and shorter ones (shortened hashes) are not serials and pass."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIAL = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{24}(?![0-9A-Fa-f])")


def files():
    for top in ("app", "tests", "docs"):
        for p in (ROOT / top).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".html", ".toml", ".txt", ".js", ".css", ".json", ".sh"}:
                yield p
    yield ROOT / "config.example.toml"


def test_no_24_hex_digit_serial_is_in_git():
    found = [f"{p.relative_to(ROOT)}:{n}" for p in files() for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
             if SERIAL.search(line)]
    assert not found, "a 24-hex-digit serial (a board's chip id) is in the repo: " + ", ".join(found)


def test_the_guard_sees_one():
    assert SERIAL.search('DFU_SERIAL = "' + "0A" * 12 + '"') and not SERIAL.search("a" * 64) and not SERIAL.search("b" * 16)
