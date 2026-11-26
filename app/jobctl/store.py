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

STATES = ("scheduled", "queued", "running", "pausing", "paused", "interrupted", "missed",
          "done", "cancelled", "error")
ACTIVE = ("running", "pausing")          # a worker process exists (or did, before a quit)
FINISHED = ("done", "cancelled", "error", "missed")

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
    worker_pid  INTEGER,              -- the worker process while it may be alive (QE PR 32)
    eta_s       INTEGER,              -- remaining seconds while running (bite 3, D35)
    warnings    TEXT                  -- JSON list of warnings the worker sent (bite 4)
);
"""
# Columns added after a jobs.sqlite may exist: name -> type (added on open).
ADDED = {"worker_pid": "INTEGER", "eta_s": "INTEGER", "warnings": "TEXT"}
JSON_COLUMNS = ("params", "limits", "summary", "warnings")


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

    def latest(self, kind: str | None = None, project: str | None = None) -> dict | None:
        """The newest job, optionally of one kind and / or project."""
        where, args = [], []
        for col, v in (("kind", kind), ("project", project)):
            if v is not None:
                where.append(f"{col} = ?")
                args.append(v)
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            r = self._conn.execute(sql + " ORDER BY id DESC LIMIT 1", args).fetchone()
        return _decode(r) if r else None

    def keep_awake_needed(self) -> bool:
        """D38: some keep-awake job is scheduled, queued or running."""
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM jobs WHERE keep_awake = 1 AND state IN "
                "('scheduled', 'queued', 'running', 'pausing') LIMIT 1").fetchone() is not None

    def last_summary(self) -> dict | None:
        """The newest job that has stopped (finished or paused) with a summary: the dashboard's
        morning summary (D38)."""
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE summary IS NOT NULL AND state NOT IN "
                "('scheduled', 'queued', 'running', 'pausing') ORDER BY ended IS NULL, "
                "COALESCE(ended, started) DESC, id DESC LIMIT 1").fetchone()
        return _decode(r) if r else None

    def scheduled(self) -> list[dict]:
        """Jobs waiting for their start time, earliest first."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs WHERE state = 'scheduled' "
                                      "ORDER BY start_at, id").fetchall()
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
                f"UPDATE jobs SET state = 'interrupted', stop_reason = ?, message = ? "
                f"WHERE state IN ({marks})", ("app quit or crashed during the job",
                                              "interrupted: the app quit or crashed during it",
                                              *ACTIVE))
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
