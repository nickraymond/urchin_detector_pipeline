"""Schedule (D37): "Start at" and "Stop by" are clock times; each means the next time that clock
time comes (Stop by: the next time after the start). Times are local, stored as ISO strings in
the job's row, so a schedule survives an app restart. The app must be running for it to fire.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

CLOCK = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class ScheduleError(ValueError):
    """A clock time that cannot be read; the message says what is expected."""


def next_time(clock: str, after: datetime) -> datetime:
    """The next moment the clock time "HH:MM" comes strictly after `after` (today or tomorrow)."""
    m = CLOCK.match(clock.strip())
    if not m:
        raise ScheduleError(f"{clock!r} is not a clock time; use HH:MM, e.g. 02:00 or 22:30.")
    at = after.replace(hour=int(m[1]), minute=int(m[2]), second=0, microsecond=0)
    return at if at > after else at + timedelta(days=1)


def plan(start_clock: str = "", stop_clock: str = "",
         now: datetime | None = None) -> tuple[datetime | None, datetime | None]:
    """(start_at, stop_by) from the form's two optional clock times. No start: the job starts
    now and Stop by counts from now."""
    now = now or datetime.now()
    start = next_time(start_clock, now) if start_clock.strip() else None
    stop = next_time(stop_clock, start or now) if stop_clock.strip() else None
    return start, stop


def iso(t: datetime | None) -> str | None:
    return t.isoformat(timespec="seconds") if t else None


def parse(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def show(s: str | None, now: datetime | None = None) -> str:
    """"2:00 AM", with the weekday when it is not today, e.g. "Thu 2:00 AM"."""
    t = parse(s)
    if t is None:
        return ""
    now = now or datetime.now()
    clock = t.strftime("%I:%M %p").lstrip("0")
    return clock if t.date() == now.date() else t.strftime("%a ") + clock
