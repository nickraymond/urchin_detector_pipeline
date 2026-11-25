"""The project's state.sqlite: tables (DESIGN, Planned for S2) and small read / write helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Pipeline stages in order, with the label shown in the UI.
STAGES = [("ingest", "Ingest"), ("clips", "Pick clips"), ("extract", "Extract frames"),
          ("review", "Review frames"), ("labeling", "SAM labeling"),
          ("export", "Review labels + export")]
STATES = ("not started", "in progress", "done", "skipped")

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    path TEXT PRIMARY KEY, path_code TEXT NOT NULL, recording TEXT NOT NULL,
    chapter INTEGER NOT NULL DEFAULT 1, duration_s REAL, size_bytes INTEGER, width INTEGER,
    height INTEGER, fps REAL, codec TEXT, warning TEXT);
CREATE TABLE IF NOT EXISTS frames (
    file TEXT PRIMARY KEY, video TEXT, timestamp_ms INTEGER, blur_score REAL, hash TEXT,
    status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stage_status (
    stage TEXT PRIMARY KEY, state TEXT NOT NULL, counts TEXT NOT NULL DEFAULT '',
    updated TEXT NOT NULL DEFAULT (datetime('now', 'localtime')));
CREATE TABLE IF NOT EXISTS stills (
    path TEXT PRIMARY KEY, name TEXT NOT NULL, size_bytes INTEGER NOT NULL, file TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, frame TEXT NOT NULL, old_status TEXT NOT NULL,
    new_status TEXT NOT NULL, at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')));
"""


def connect(path: Path, create: bool = False) -> sqlite3.Connection:
    """Open a project database; callers close it. Only project.create passes create=True:
    a missing file on open is an error, never silently replaced by an empty database
    (that would lose review state)."""
    path = Path(path).absolute()
    if not create and not path.is_file():
        raise FileNotFoundError(
            f"Project database missing: {path}. It holds this project's progress; restore it "
            "from a backup. The app will not make a new empty one in its place.")
    conn = sqlite3.connect(path.as_uri() + ("?mode=rwc" if create else "?mode=rw"), uri=True)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executemany("INSERT OR IGNORE INTO stage_status (stage, state) VALUES (?, 'not started')",
                     [(s,) for s, _ in STAGES])
    conn.commit()
    return conn


def stage_statuses(conn: sqlite3.Connection) -> list[dict]:
    """Every stage in pipeline order with its label, state, counts and last update."""
    rows = {r["stage"]: dict(r) for r in conn.execute("SELECT * FROM stage_status")}
    return [{"label": label, **rows[stage]} for stage, label in STAGES]


def set_stage(conn: sqlite3.Connection, stage: str, state: str, counts: str = "") -> None:
    if stage not in dict(STAGES) or state not in STATES:
        raise ValueError(f"unknown stage {stage!r} or state {state!r}")
    conn.execute("UPDATE stage_status SET state = ?, counts = ?, "
                 "updated = datetime('now', 'localtime') WHERE stage = ?", (state, counts, stage))
    conn.commit()
