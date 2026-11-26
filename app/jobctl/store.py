"""Job records: one row per job in <data_dir>/jobs/jobs.sqlite (DESIGN, Planned for S4; D32).

Plain sqlite, one connection shared under a lock, autocommit. Nothing here knows what a job
does: params and summary are JSON text the caller owns.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

STATES = ("scheduled", "queued", "running", "pausing", "paused", "interrupted",
          "done", "cancelled", "error")
ACTIVE = ("running", "pausing")          # a worker process exists (or did, before a quit)
FINISHED = ("done", "cancelled", "error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    project     TEXT,
    params      TEXT NOT NULL,        -- JSON
    state       TEXT NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0,
    total       INTEGER,
    message     TEXT,
    start_at    TEXT,                 -- schedule (bite 5a)
    stop_by     TEXT,
    keep_awake  INTEGER NOT NULL DEFAULT 0,
    limits      TEXT,                 -- JSON, the limits the job ran with (bite 4)
    created     TEXT NOT NULL,
    started     TEXT,
    ended       TEXT,
    stop_reason TEXT,
    summary     TEXT,                 -- JSON (bite 5b)
    worker_pid  INTEGER               -- the worker process while it may be alive (QE PR 32)
);
"""
# Columns added after a jobs.sqlite may exist: name -> type (added on open).
ADDED = {"worker_pid": "INTEGER"}
JSON_COLUMNS = ("params", "limits", "summary")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(jobs)")}
        for col, kind in ADDED.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {kind}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, kind: str, params: dict, project: str | None = None,
            state: str = "queued", **fields) -> int:
        row = {"kind": kind, "project": project, "params": json.dumps(params),
               "state": state, "created": now(), **_encode(fields)}
        cols = ", ".join(row)
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO jobs ({cols}) VALUES "
                                     f"({', '.join('?' * len(row))})", list(row.values()))
            return int(cur.lastrowid)

    def get(self, job_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _decode(r) if r else None

    def all(self, limit: int = 200) -> list[dict]:
        """Newest first."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?",
                                      (limit,)).fetchall()
        return [_decode(r) for r in rows]

    def update(self, job_id: int, **fields) -> None:
        if not fields:
            return
        fields = _encode(fields)
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?",
                               [*fields.values(), job_id])

    def set_state(self, job_id: int, state: str, from_states: tuple[str, ...],
                  **fields) -> bool:
        """Move a job to `state` only if it is in one of `from_states` (checked and written
        in one statement). Returns whether it moved."""
        assert state in STATES
        fields = _encode({"state": state, **fields})
        sets = ", ".join(f"{k} = ?" for k in fields)
        marks = ", ".join("?" * len(from_states))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET {sets} WHERE id = ? AND state IN ({marks})",
                [*fields.values(), job_id, *from_states])
            return cur.rowcount == 1

    def with_worker(self) -> list[dict]:
        """Jobs with a worker PID recorded (it may still be finishing an item)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs WHERE worker_pid IS NOT NULL "
                                      "ORDER BY id").fetchall()
        return [_decode(r) for r in rows]

    def next_queued(self) -> dict | None:
        """The oldest queued job."""
        with self._lock:
            r = self._conn.execute("SELECT * FROM jobs WHERE state = 'queued' "
                                   "ORDER BY id LIMIT 1").fetchone()
        return _decode(r) if r else None

    def mark_interrupted(self) -> list[int]:
        """At app start: jobs left running or pausing by a quit or crash become 'interrupted'."""
        marks = ", ".join("?" * len(ACTIVE))
        with self._lock:
            ids = [r[0] for r in self._conn.execute(
                f"SELECT id FROM jobs WHERE state IN ({marks})", ACTIVE)]
            self._conn.execute(
                f"UPDATE jobs SET state = 'interrupted', stop_reason = ? "
                f"WHERE state IN ({marks})", ("app quit or crashed during the job", *ACTIVE))
        return ids


def _encode(fields: dict) -> dict:
    return {k: (json.dumps(v) if k in JSON_COLUMNS and v is not None else v)
            for k, v in fields.items()}


def _decode(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in JSON_COLUMNS:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    d["keep_awake"] = bool(d["keep_awake"])
    return d
