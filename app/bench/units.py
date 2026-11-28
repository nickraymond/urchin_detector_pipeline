"""Bench units (D81, SPEC constraint 9). They are listed only in config.local.toml, never in git:

    [bench]
    field_units = ["<a field unit's name>"]     # refused by name, always
    [bench.units.<unit name>]
    ssh_host = "<alias in ~/.ssh/config>"
    cameras = ["n6", "ae3"]                     # which camera kinds hang on that Pi's USB

With no [bench] table only the fake board exists. A unit that is not listed, or whose name or ssh host
is a field unit, is refused with a reason; nothing here ever opens a connection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FAKE = "fake"                      # the one unit that always exists: FakeBoard, for builders and QE
CAMERA_KINDS = ("n6", "ae3")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Refused(Exception):
    """A unit the app must not touch. The message says why and what to change."""


@dataclass(frozen=True)
class Unit:
    name: str
    ssh_host: str = ""
    cameras: tuple[str, ...] = ()

    @property
    def fake(self) -> bool:
        return self.name == FAKE

    @property
    def kinds(self) -> tuple[str, ...]:
        """The boards on this unit: the Pi itself, then its cameras."""
        return ("pi", *self.cameras)


@dataclass
class Bench:
    units: dict[str, Unit] = field(default_factory=dict)
    field_units: frozenset[str] = frozenset()
    problems: list[str] = field(default_factory=list)   # shown on the page; a bad entry is left out

    def unit(self, name: str) -> Unit:
        """The unit to run on, or Refused. The fake unit always exists and touches nothing."""
        if name == FAKE:
            return Unit(FAKE, cameras=CAMERA_KINDS)
        if name in self.field_units:
            raise Refused(f"{name} is a field unit: the app never touches field units (SPEC constraint 9).")
        if name not in self.units:
            listed = ", ".join(sorted(self.units)) or "none"
            raise Refused(f"{name} is not a bench unit listed under [bench] in config.local.toml "
                          f"(listed: {listed}). Only the fake board can be used until a unit is listed.")
        return self.units[name]

    def names(self) -> list[str]:
        return [FAKE, *sorted(self.units)]


def parse(raw) -> Bench:
    """The [bench] table as read from config.local.toml (None when absent)."""
    bench = Bench()
    if raw is None:
        return bench
    if not isinstance(raw, dict):
        bench.problems.append("[bench] in config.local.toml must be a table; no bench unit is used.")
        return bench
    fields = raw.get("field_units", [])
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        # Without a trustworthy refusal list no real unit is safe to use.
        bench.problems.append("[bench] field_units must be a list of names; no bench unit is used until it is fixed.")
        return bench
    bench.field_units = frozenset(f.strip() for f in fields if f.strip())
    units = raw.get("units", {})
    if not isinstance(units, dict):
        bench.problems.append("[bench.units] must be a table of units; none is used.")
        return bench
    for name, body in units.items():
        why = _problem(name, body, bench.field_units)
        if why:
            bench.problems.append(f"Bench unit {name!r} left out: {why}")
            continue
        bench.units[name] = Unit(name, body["ssh_host"].strip(), tuple(body.get("cameras", [])))
    return bench


def _problem(name, body, field_units) -> str:
    if name == FAKE:
        return f'"{FAKE}" is the fake board\'s name; pick another.'
    if not NAME.match(name):
        return "a name is letters, digits, '.', '_' or '-'."
    if not isinstance(body, dict):
        return "it must be a table with ssh_host and cameras."
    host = body.get("ssh_host")
    if not isinstance(host, str) or not host.strip():
        return "it has no ssh_host (an alias from ~/.ssh/config)."
    if name in field_units or host.strip() in field_units:
        return "it is listed in field_units; the app never touches field units."
    cams = body.get("cameras", [])
    if not isinstance(cams, list) or any(c not in CAMERA_KINDS for c in cams) or len(set(cams)) != len(cams):
        return f"cameras must be a list of {', '.join(CAMERA_KINDS)}, each at most once."
    return ""
