"""A merge conflict that is committed with its markers breaks no test by itself (it happened to
docs/DESIGN.md, PR 84). This one fails on any line that is only a marker, in every text file
of the repo's own folders."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKS = ("<<<<<<< ", ">>>>>>> ")
SUFFIXES = {".py", ".md", ".html", ".js", ".css", ".toml", ".txt", ".json", ".sh", ""}


def test_no_file_holds_a_merge_conflict_marker():
    found = []
    tops = [p for p in REPO.iterdir() if p.is_file()] + [
        p for d in ("app", "docs", "tests", ".claude") for p in (REPO / d).rglob("*") if p.is_file()]
    for p in tops:
        if p.suffix not in SUFFIXES or "__pycache__" in p.parts or p.name == "uv.lock":
            continue
        try:
            lines = p.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        found += [f"{p.relative_to(REPO)}:{n}" for n, line in enumerate(lines, 1)
                  if line.startswith(MARKS)]
    assert not found, f"merge conflict markers left in: {', '.join(found)}"
